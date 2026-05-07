import argparse
import json
import os
from collections import defaultdict


args = argparse.ArgumentParser()
args.add_argument("--results_path", type=str, default=None)
args.add_argument("--filter_text_path", type=str, default=None)
args = args.parse_args()


def filter_text_units(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    unit_scores = defaultdict(list)
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

    # unit level
    unit_acc = {
        key: sum(v) / len(v)
        for key, v in unit_scores.items()
    }

    # filter correct cases
    unit_list = []
    for key, acc in unit_acc.items():
        if acc == 1.0:
            unit_list.append(key)

    return unit_list


if __name__ == "__main__":
    unit_list = filter_text_units(args.results_path)

    os.makedirs(os.path.dirname(args.filter_text_path), exist_ok=True)
    with open(args.filter_text_path, "w", encoding="utf-8") as f:
        json.dump(unit_list, f, ensure_ascii=False, indent=4)
