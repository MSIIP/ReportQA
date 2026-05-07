import argparse
import json
import os
from collections import defaultdict

import pandas as pd


QASCORE_LAMBDA = 10.0


args = argparse.ArgumentParser()
args.add_argument("--results_path", type=str, default=None)
args.add_argument("--scores_dir", type=str, default=None)
args = args.parse_args()
os.makedirs(args.scores_dir, exist_ok=True)


def preprocess(
    results_path,
    scores_dir,
):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    unit_scores = defaultdict(list)
    presence_keys = set()

    bad_cases = {}
    for outer_report_id, item in data.items():
        for qa in item.get("qas", []):
            report_id = qa["report_id"]
            clinical_entity_type = qa["clinical_entity_type"]
            clinical_entity = qa["clinical_entity"]
            clinical_entity_idx = qa["clinical_entity_idx"]
            attribute = qa["attribute"]
            question_idx = qa["question_idx"]
            question_type = qa["question_type"]
            key = (report_id, clinical_entity_type, clinical_entity, clinical_entity_idx, attribute, question_idx, question_type)

            gt = qa["answer_option"]
            pred = qa["predict"]

            correct = 1.0 if pred.strip().lower() == gt.strip().lower() else 0.0
            unit_scores[key].append(correct)

            if correct == 0.0:
                if outer_report_id not in bad_cases:
                    bad_cases[outer_report_id] = {
                        "report": item.get("report", ""),
                        "qas": []
                    }
                bad_cases[outer_report_id]["qas"].append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "report_id": report_id,
                    "clinical_entity_type": clinical_entity_type,
                    "clinical_entity": clinical_entity,
                    "clinical_entity_idx": clinical_entity_idx,
                    "attribute": attribute,
                    "question_idx": question_idx,
                    "question_type": question_type,
                    "answer_option": gt,
                    "answer_str": qa["answer_str"],
                    "predict": pred
                })

            # clinical_entity真实存在却回答为否的时候，后续关于该clinical_entity的其他属性都应该算错
            if attribute == "presence":
                if qa["answer_str"].strip().lower() == "yes" and correct == 0.0:
                    presence_keys.add((report_id, clinical_entity_type, clinical_entity, clinical_entity_idx))

    # Save bad cases
    with open(os.path.join(scores_dir, "bad_cases.json"), "w", encoding="utf-8") as f:
        json.dump(bad_cases, f, ensure_ascii=False, indent=4)

    # unit level
    unit_acc = {
        key: sum(v) / len(v)
        for key, v in unit_scores.items()
    }

    # punish presence errors
    for (report_id, clinical_entity_type, clinical_entity, clinical_entity_idx, attribute, question_idx, question_type), acc in unit_acc.items():
        if (report_id, clinical_entity_type, clinical_entity, clinical_entity_idx) in presence_keys:
            unit_acc[(report_id, clinical_entity_type, clinical_entity, clinical_entity_idx, attribute, question_idx, question_type)] = 0.0

    return unit_acc


def compute_metrics(
    unit_acc,
    scores_dir,
):
    QUESTION_TYPE_ORDER = [
        "base",
        "hierarchical",
        "negative",
    ]
    CLINICAL_ENTITY_TYPE_ORDER = [
        "finding",
        "diagnosis",
    ]
    ATTRIBUTE_ORDER = [
        "presence",
        "location",
        "distribution",
        "number",
        "dimension",
        "density",
        "shape",
        "margin",
        "enhancement",
        "internal_features",
        "secondary_effects",
        "severity",
        "chronicity",
        "clinical_score",
        "certainty",
    ]

    def build_order_map(order_list):
        return {v: i for i, v in enumerate(order_list)}

    def sort_by_custom_order(df, col, order_list):
        order_map = build_order_map(order_list)
        df = df.copy()
        df["_order"] = df[col].map(lambda x: order_map.get(x, len(order_map)))
        df["_value"] = df[col].astype(str)
        df = df.sort_values(by=["_order", "_value"])
        df = df.drop(columns=["_order", "_value"])
        return df

    # 聚合容器
    report_scores = defaultdict(list)
    clinical_entity_type_scores = defaultdict(list)
    attribute_scores = defaultdict(list)

    # key = (report_id, clinical_entity_type, clinical_entity, clinical_entity_idx, attribute, question_idx, question_type)
    for key, acc in unit_acc.items():
        (
            report_id,
            clinical_entity_type,
            clinical_entity,
            clinical_entity_idx,
            attribute,
            question_idx,
            question_type,
        ) = key

        report_scores[(question_type, report_id)].append(acc)
        clinical_entity_type_scores[(question_type, clinical_entity_type)].append(acc)
        attribute_scores[(question_type, attribute)].append(acc)

    overall_rows = []

    for question_type in QUESTION_TYPE_ORDER:
        # 1. report_id_acc
        report_rows = []
        report_id_acc_list = []

        for (qtype, report_id), scores in report_scores.items():
            if qtype != question_type:
                continue
            acc = sum(scores) / len(scores)
            report_rows.append(
                {
                    "report_id": report_id,
                    "acc": acc,
                    "count": len(scores),
                }
            )
            report_id_acc_list.append(acc)

        report_df = pd.DataFrame(report_rows)
        if len(report_df) > 0:
            report_df = report_df.sort_values(by="report_id", key=lambda s: s.astype(str))
        report_df.to_csv(
            os.path.join(scores_dir, f"{question_type}_report_id.csv"),
            index=False,
        )

        # 2. overall_acc: 所有 report_id_acc 取平均
        overall_acc = (
            sum(report_id_acc_list) / len(report_id_acc_list)
            if len(report_id_acc_list) > 0
            else None
        )
        overall_rows.append(
            {
                "question_type": question_type,
                "overall_acc": overall_acc,
                "num_report_ids": len(report_id_acc_list),
            }
        )

        # 3. clinical_entity_type_acc
        clinical_entity_type_rows = []
        for (qtype, clinical_entity_type), scores in clinical_entity_type_scores.items():
            if qtype != question_type:
                continue
            acc = sum(scores) / len(scores)
            clinical_entity_type_rows.append(
                {
                    "clinical_entity_type": clinical_entity_type,
                    "acc": acc,
                    "count": len(scores),
                }
            )

        clinical_entity_type_df = pd.DataFrame(clinical_entity_type_rows)
        if len(clinical_entity_type_df) > 0:
            clinical_entity_type_df = sort_by_custom_order(
                clinical_entity_type_df,
                "clinical_entity_type",
                CLINICAL_ENTITY_TYPE_ORDER,
            )
        clinical_entity_type_df.to_csv(
            os.path.join(scores_dir, f"{question_type}_clinical_entity_type.csv"),
            index=False,
        )

        # 4. attribute_acc
        attribute_rows = []
        for (qtype, attribute), scores in attribute_scores.items():
            if qtype != question_type:
                continue
            acc = sum(scores) / len(scores)
            attribute_rows.append(
                {
                    "attribute": attribute,
                    "acc": acc,
                    "count": len(scores),
                }
            )

        attribute_df = pd.DataFrame(attribute_rows)
        if len(attribute_df) > 0:
            attribute_df = sort_by_custom_order(
                attribute_df,
                "attribute",
                ATTRIBUTE_ORDER,
            )
        attribute_df.to_csv(
            os.path.join(scores_dir, f"{question_type}_attribute.csv"),
            index=False,
        )

    # 保存 overall_acc.csv
    overall_df = pd.DataFrame(overall_rows)
    if len(overall_df) > 0:
        overall_df = sort_by_custom_order(
            overall_df,
            "question_type",
            QUESTION_TYPE_ORDER,
        )
    overall_df.to_csv(
        os.path.join(scores_dir, "overall_acc.csv"),
        index=False,
    )


# =========================
# QAScore extra outputs
# =========================

def _qascore_mean(values):
    valid_values = [float(value) for value in values if value is not None]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)


def _qascore_exp_negative(fpr, qascore_lambda):
    if fpr is None:
        return None
    import math

    return math.exp(-qascore_lambda * fpr)


def _qascore_combine(score_pos, score_neg):
    if score_pos is None:
        return score_neg
    if score_neg is None:
        return score_pos
    if score_pos == 0.0 or score_neg == 0.0:
        return 0.0
    return 2.0 * score_pos * score_neg / (score_pos + score_neg)


def _qascore_normalize_label(value):
    if value is None:
        return ""
    return str(value).strip().lower()


def _qascore_is_false_positive_negative(qa):
    # Negative questions are absent-sample checks; predicting "yes" is a false positive.
    return _qascore_normalize_label(qa.get("predict")) == "yes"


def compute_qascore_outputs(
    results_path,
    unit_acc,
    scores_dir,
):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    report_pos_scores = defaultdict(list)
    report_pos_counts = defaultdict(lambda: defaultdict(int))
    corpus_pos_scores = []
    corpus_pos_counts = defaultdict(int)

    # key = (report_id, clinical_entity_type, clinical_entity, clinical_entity_idx, attribute, question_idx, question_type)
    for key, acc in unit_acc.items():
        (
            report_id,
            clinical_entity_type,
            clinical_entity,
            clinical_entity_idx,
            attribute,
            question_idx,
            question_type,
        ) = key

        if question_type not in {"base", "hierarchical"}:
            continue

        report_pos_scores[report_id].append(acc)
        report_pos_counts[report_id][question_type] += 1
        corpus_pos_scores.append(acc)
        corpus_pos_counts[question_type] += 1

    negative_unit_fp_scores = defaultdict(list)
    all_report_ids = set(report_pos_scores.keys())

    for outer_report_id, item in data.items():
        qas = item.get("qas", [])
        if not qas:
            all_report_ids.add(outer_report_id)
            continue

        for qa in qas:
            report_id = qa.get("report_id", outer_report_id)
            all_report_ids.add(report_id)

            if qa.get("question_type") != "negative":
                continue

            key = (
                report_id,
                qa["clinical_entity_type"],
                qa["clinical_entity"],
                qa["clinical_entity_idx"],
                qa["attribute"],
                qa["question_idx"],
                qa["question_type"],
            )
            negative_unit_fp_scores[key].append(
                1.0 if _qascore_is_false_positive_negative(qa) else 0.0
            )

    report_neg_fp_scores = defaultdict(list)
    corpus_neg_fp_scores = []
    for (
        report_id,
        clinical_entity_type,
        clinical_entity,
        clinical_entity_idx,
        attribute,
        question_idx,
        question_type,
    ), fp_scores in negative_unit_fp_scores.items():
        unit_fpr = sum(fp_scores) / len(fp_scores)
        report_neg_fp_scores[report_id].append(unit_fpr)
        corpus_neg_fp_scores.append(unit_fpr)

    rows = []
    for report_id in sorted(all_report_ids, key=lambda value: str(value)):
        pos_scores = report_pos_scores.get(report_id, [])
        score_pos = _qascore_mean(pos_scores)

        neg_fp_scores = report_neg_fp_scores.get(report_id, [])
        fpr_neg = _qascore_mean(neg_fp_scores)
        score_neg = _qascore_exp_negative(fpr_neg, QASCORE_LAMBDA)

        qascore = _qascore_combine(score_pos, score_neg)
        n_base = report_pos_counts[report_id].get("base", 0)
        n_hierarchical = report_pos_counts[report_id].get("hierarchical", 0)
        neg_fp_units = sum(1 for score in neg_fp_scores if score > 0.0)

        rows.append(
            {
                "report_id": report_id,
                "QAScore": qascore,
                "Score_pos": score_pos,
                "Score_neg": score_neg,
                "FPR_neg": fpr_neg,
                "n_positive": len(pos_scores),
                "n_base": n_base,
                "n_hierarchical": n_hierarchical,
                "n_negative": len(neg_fp_scores),
                "n_negative_false_positive": neg_fp_units,
            }
        )

    report_df = pd.DataFrame(rows)
    report_df.to_csv(
        os.path.join(scores_dir, "qascore_report_scores.csv"),
        index=False,
    )

    corpus_score_pos = _qascore_mean(corpus_pos_scores)
    corpus_fpr_neg = _qascore_mean(corpus_neg_fp_scores)
    corpus_score_neg = _qascore_exp_negative(corpus_fpr_neg, QASCORE_LAMBDA)
    corpus_qascore = _qascore_combine(corpus_score_pos, corpus_score_neg)
    report_qascores = [row["QAScore"] for row in rows if row["QAScore"] is not None]

    summary = {
        "formula": {
            "QAScore": "harmonic_mean(Score_pos, Score_neg) if both exist, else the single available score",
            "Score_pos": "accuracy(base + hierarchical), using original presence punishment from unit_acc",
            "Score_neg": "exp(-lambda * FPR_neg)",
            "FPR_neg": "mean(false_positive_rate per negative unit)",
        },
        "parameters": {
            "lambda": QASCORE_LAMBDA,
            "missing_component": "use_the_single_available_score",
        },
        "corpus_scores": {
            "Corpus_QAScore": corpus_qascore,
            "Score_pos": corpus_score_pos,
            "Score_neg": corpus_score_neg,
            "FPR_neg": corpus_fpr_neg,
        },
        "report_score_mean": _qascore_mean(report_qascores),
        "counts": {
            "reports": len(rows),
            "positive_units": len(corpus_pos_scores),
            "base_units": corpus_pos_counts.get("base", 0),
            "hierarchical_units": corpus_pos_counts.get("hierarchical", 0),
            "negative_units": len(corpus_neg_fp_scores),
            "negative_false_positive_units": sum(
                1 for score in corpus_neg_fp_scores if score > 0.0
            ),
        },
    }

    with open(os.path.join(scores_dir, "qascore_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    unit_acc = preprocess(
        results_path=args.results_path,
        scores_dir=args.scores_dir,
    )

    compute_metrics(
        unit_acc=unit_acc,
        scores_dir=args.scores_dir,
    )

    compute_qascore_outputs(
        results_path=args.results_path,
        unit_acc=unit_acc,
        scores_dir=args.scores_dir,
    )
