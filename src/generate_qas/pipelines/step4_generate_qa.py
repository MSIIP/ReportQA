import argparse
import json
import os
import random
import sys
from collections import defaultdict
from types import SimpleNamespace

import yaml


TASK_SPECS = {
    # 各类 QA 的语言模板、树根节点名和采样策略统一集中在这里。
    "finding": {
        "report_items_key": "findings",
        "tree_key": "FINDINGS_TREE",
        "lang_config": {
            "zh": {
                "yes": "是",
                "no": "否",
                "present": "有(present)",
                "absent": "无(absent)",
                "q_presence": "是否存在【{}】？",
                "q_presence_hier": "是否存在【{}】？",
                "q_location": "【{}】中是否存在【{}】？",
                "q_attribute": "{}， 其【{}】描述为？",
                "prefix_with_loc": "针对【{}】的【{}】",
                "prefix_no_loc": "针对【{}】",
                "other_finding": "异常征象",
                "unknown_abn": "Unknown",
                "default_attr_cat": "特征",
                "anatomy_root": ["解剖结构"],
                "attr_root": ["属性"],
            },
            "en": {
                "yes": "Yes",
                "no": "No",
                "present": "present",
                "absent": "absent",
                "q_presence": "Is [{}] present?",
                "q_presence_hier": "Is [{}] present?",
                "q_location": "Is [{}] present in the [{}]?",
                "q_attribute": "{}, what is its [{}]?",
                "prefix_with_loc": "Regarding the [{}] in the [{}]",
                "prefix_no_loc": "Regarding the [{}]",
                "other_finding": "Abnormal finding",
                "unknown_abn": "Unknown",
                "default_attr_cat": "Feature",
                "anatomy_root": ["Anatomy"],
                "attr_root": ["Attributes"],
            },
        },
        "allow_other_location_label": True,
        "hier_include_absent": False,
        "fallback_non_attribute_distractors": False,
        "negative_ignore_words": {"Other", "其他", "疾病", "病变", "异常", "影像征象", "Disease", "Imaging Findings"},
    },
    "diagnosis": {
        "report_items_key": "diagnoses",
        "tree_key": "DIAGNOSIS_TREE",
        "lang_config": {
            "zh": {
                "yes": "是",
                "no": "否",
                "present": "有(present)",
                "absent": "无(absent)",
                "q_presence": "是否存在【{}】？",
                "q_presence_hier": "是否存在【{}】？",
                "q_location": "【{}】中是否存在【{}】？",
                "q_attribute": "{}， 其【{}】描述为？",
                "prefix_with_loc": "针对【{}】的【{}】",
                "prefix_no_loc": "针对【{}】",
                "unknown_abn": "未知疾病",
                "default_attr_cat": "特征",
                "anatomy_root": ["解剖结构"],
                "attr_root": ["属性"],
            },
            "en": {
                "yes": "Yes",
                "no": "No",
                "present": "present",
                "absent": "absent",
                "q_presence": "Is [{}] present?",
                "q_presence_hier": "Is [{}] present?",
                "q_location": "Is [{}] in the [{}]?",
                "q_attribute": "{}, what is its [{}]?",
                "prefix_with_loc": "Regarding the [{}] in the [{}]",
                "prefix_no_loc": "Regarding the [{}]",
                "unknown_abn": "Unknown diagnosis",
                "default_attr_cat": "Feature",
                "anatomy_root": ["Anatomy"],
                "attr_root": ["Attributes"],
            },
        },
        "allow_other_location_label": False,
        "hier_include_absent": False,
        "fallback_non_attribute_distractors": True,
        "negative_ignore_words": {"Other", "其他", "疾病", "病变", "异常", "影像征象", "Disease"},
    },
}

DEFAULT_ATTRIBUTE_ORDER = {
    "presence": 0,
    "location": 1,
    "distribution": 2,
    "number": 3,
    "dimension": 4,
    "density": 5,
    "shape": 6,
    "margin": 7,
    "enhancement": 8,
    "internal_features": 9,
    "secondary_effects": 10,
    "severity": 11,
    "chronicity": 12,
    "clinical_score": 13,
    "certainty": 14,
}


def apply_lang_suffix(template, lang):
    if not isinstance(template, str):
        return template
    lang_suffix = "zh" if lang == "zh" else "en"
    try:
        # 允许 YAML 路径模板中使用 `{lang_suffix}`，便于中英文共用一份配置。
        return template.format(lang_suffix=lang_suffix)
    except (KeyError, IndexError, ValueError):
        return template


def resolve_run_paths(args):
    args.input = apply_lang_suffix(getattr(args, "input", None), args.lang)
    args.finding_input = apply_lang_suffix(getattr(args, "finding_input", None), args.lang)
    args.diagnosis_input = apply_lang_suffix(getattr(args, "diagnosis_input", None), args.lang)
    raw_json_files = getattr(args, "raw_json_files", None)
    if isinstance(raw_json_files, list):
        args.raw_json_files = [apply_lang_suffix(path, args.lang) for path in raw_json_files]
    args.output = apply_lang_suffix(getattr(args, "output", None), args.lang)
    args.knowledge_tree = apply_lang_suffix(getattr(args, "knowledge_tree", None), args.lang)
    return args


def apply_yaml_config(args, yaml_data):
    setting = yaml_data.get("setting", {})
    paths = yaml_data.get("paths", {})

    for key, value in setting.items():
        setattr(args, key.replace("-", "_"), value)
    for key, value in paths.items():
        setattr(args, key.replace("-", "_"), value)

    for key, value in yaml_data.items():
        if key in {"setting", "paths"}:
            continue
        setattr(args, key.replace("-", "_"), value)

    if not getattr(args, "task", None) and getattr(args, "qa_type", None):
        args.task = "both" if args.qa_type == "all" else args.qa_type
    elif getattr(args, "task", None) == "all":
        args.task = "both"

    return args


class BaseQAGenerator:
    def __init__(self, config_data, task_type="finding", lang="zh"):
        self.task_type = task_type
        self.spec = TASK_SPECS[task_type]
        self.lang = lang
        self.lc = self.spec["lang_config"][lang]
        self.attributes_tree = config_data.get("ATTRIBUTES_TREE", {})
        self.fallback_attributes = config_data.get("FALLBACK_DISTRACTORS_ATTRIBUTES", [])
        self.report_items_key = self.spec["report_items_key"]

    def _get_siblings_from_tree(self, path, current_node):
        if not path:
            return []
        if len(path) == 1:
            target = path[0]
            if isinstance(current_node, list):
                return [item for item in current_node if item != target]
            if isinstance(current_node, dict):
                return [k for k in current_node.keys() if k != target]
            return []
        next_key = path[0]
        if isinstance(current_node, dict) and next_key in current_node:
            return self._get_siblings_from_tree(path[1:], current_node[next_key])
        return []

    def get_distractors(self, mapped_path, needed, conflict_set=None):
        conflict_set = conflict_set or set()
        is_attribute_path = mapped_path and mapped_path[0] in self.lc["attr_root"]

        if not mapped_path or len(mapped_path) < 2 or mapped_path == ["Other"]:
            if self.spec["fallback_non_attribute_distractors"] and not is_attribute_path:
                siblings = self.fallback_attributes.copy()
            else:
                return []
        else:
            # 属性问题优先从同一父节点下采样干扰项，保证“像真但不对”。
            siblings = self._get_siblings_from_tree(mapped_path[1:], self.attributes_tree)

        siblings = [
            self.lc["present"] if s == "present" else self.lc["absent"] if s == "absent" else s
            for s in siblings
        ]
        valid_siblings = [s for s in siblings if s not in conflict_set]
        random.shuffle(valid_siblings)

        if is_attribute_path:
            return valid_siblings[:needed]

        if len(valid_siblings) < needed and self.spec["fallback_non_attribute_distractors"]:
            # 某些树分支过浅时，回退到全局兜底干扰项池，避免选项数不足。
            fallback = [
                f for f in self.fallback_attributes
                if f not in valid_siblings and f not in conflict_set
            ]
            random.shuffle(fallback)
            valid_siblings.extend(fallback[:needed - len(valid_siblings)])

        random.shuffle(valid_siblings)
        return valid_siblings[:needed]

    def _build_options(self, correct_answer, distractors):
        display_answer = (
            self.lc["present"] if correct_answer == "present"
            else self.lc["absent"] if correct_answer == "absent"
            else correct_answer
        )
        options_list = [display_answer] + distractors
        random.shuffle(options_list)
        return options_list, options_list.index(display_answer)

    def process_report(self, report_data, num_distractors=3, question_type="base"):
        qa_list = []
        report_id = report_data.get("report_id", "Unknown")
        items = report_data.get(self.report_items_key, [])

        report_positive_set = set()
        subject_counter = defaultdict(int)

        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("presence", {}).get("normalized_name") == "absent":
                continue

            # 收集整份报告里已经出现过的标准化值，后面构造干扰项时避免“真值混入干扰项”。
            abn_name = item.get("clinical_entity", {}).get("normalized_name")
            loc_name = item.get("location", {}).get("normalized_name")
            if abn_name:
                report_positive_set.add(abn_name)
            if loc_name:
                report_positive_set.add(loc_name)
            for value in item.values():
                if isinstance(value, dict) and "normalized_name" in value:
                    report_positive_set.add(value["normalized_name"])

            if abn_name:
                subject_counter[abn_name] += 1
                dist_obj = item.get("distribution", {})
                is_laterality = any(x in dist_obj.get("mapped_path", []) for x in ["侧性", "Laterality"])
                side_val = dist_obj.get("normalized_name", "") if is_laterality else ""
                full_loc = f"{side_val} {loc_name or ''}".strip()
                if full_loc:
                    # 同一个征象在不同部位/侧别可重复出现，后面要靠这个计数避免问题指代不清。
                    subject_counter[f"{abn_name}_{full_loc}"] += 1

        entity_counter = defaultdict(int)
        for item in items:
            if not isinstance(item, dict):
                continue

            full_loc_desc = ""
            abn_obj = item.get("clinical_entity", {})
            raw_text = abn_obj.get("raw", self.lc["unknown_abn"])
            sign_name = abn_obj.get("normalized_name", raw_text)
            sign_path = abn_obj.get("mapped_path", [])
            current_idx = entity_counter[sign_name]
            entity_counter[sign_name] += 1

            is_absent = item.get("presence", {}).get("normalized_name") == "absent"
            if not sign_name or (sign_path == ["Other"] and not is_absent):
                continue

            ans_bool = self.lc["no"] if is_absent else self.lc["yes"]
            opts_bool, idx_bool = self._build_options(
                ans_bool,
                [self.lc["yes"] if ans_bool == self.lc["no"] else self.lc["no"]],
            )
            evidence = item.get("evidence_span", "")

            qa_list.append(
                make_qa_record(
                    report_id=report_id,
                    clinical_entity_type=self.task_type,
                    clinical_entity=sign_name,
                    clinical_entity_idx=current_idx,
                    attribute="presence",
                    question_idx=0,
                    question_type=question_type,
                    question=self.lc["q_presence"].format(sign_name),
                    options=opts_bool,
                    answer=idx_bool,
                    evidence=evidence,
                )
            )

            loc_obj = item.get("location", {})
            if loc_obj.get("mapped_path", []) and loc_obj.get("mapped_path", [])[0] in self.lc["anatomy_root"]:
                dist_obj = item.get("distribution", {})
                is_laterality = any(x in dist_obj.get("mapped_path", []) for x in ["侧性", "Laterality"])
                side_val = dist_obj.get("normalized_name", "") if is_laterality else ""
                loc_raw = loc_obj.get("normalized_name", "") or loc_obj.get("raw", "")
                full_loc_desc = f"{side_val}{' ' if self.lang == 'en' else ''}{loc_raw}".strip()

                display_sign = sign_name
                if self.spec["allow_other_location_label"] and sign_name == "Other":
                    display_sign = self.lc["other_finding"]

                question = (
                    self.lc["q_location"].format(full_loc_desc, display_sign)
                    if self.lang == "zh"
                    else self.lc["q_location"].format(display_sign, full_loc_desc)
                )
                qa_list.append(
                    make_qa_record(
                        report_id=report_id,
                        clinical_entity_type=self.task_type,
                        clinical_entity=sign_name,
                        clinical_entity_idx=current_idx,
                        attribute="location",
                        question_idx=0,
                        question_type=question_type,
                        question=question,
                        options=opts_bool,
                        answer=idx_bool,
                        evidence=evidence,
                    )
                )

            if is_absent:
                continue

            if full_loc_desc and full_loc_desc.lower() != "other":
                subject_key = f"{sign_name}_{full_loc_desc}"
                if subject_counter[subject_key] > 1:
                    # 同一“征象+部位”重复时，属性问句无法唯一对应到一个实例，直接跳过。
                    continue
                prefix = (
                    self.lc["prefix_with_loc"].format(full_loc_desc, sign_name)
                    if self.lang == "zh"
                    else self.lc["prefix_with_loc"].format(sign_name, full_loc_desc)
                )
            else:
                if subject_counter.get(sign_name, 0) > 1:
                    # 没有部位信息时若同名实体出现多次，也不生成属性题，避免指代歧义。
                    continue
                prefix = self.lc["prefix_no_loc"].format(sign_name)

            for attr_name, attr_obj in item.items():
                if not isinstance(attr_obj, dict):
                    continue
                attr_path = attr_obj.get("mapped_path", [])
                if not attr_path or attr_path[0] not in self.lc["attr_root"]:
                    continue
                if attr_name == "presence" and "Presence" not in attr_path and "存在性" not in attr_path:
                    # 兜底拦截被错误映射成普通属性的 presence，避免生成 `what is its [Attributes]?`。
                    continue
                if any(x in attr_path for x in ["存在性", "Presence", "侧性", "Laterality"]):
                    continue

                attr_val = attr_obj.get("normalized_name")
                if (
                    not attr_val
                    or attr_path == ["Other"]
                    or str(attr_val).strip() == "Other"
                    or any(str(node).strip() == "Other" for node in attr_path)
                ):
                    continue

                cat = attr_path[-2] if len(attr_path) >= 2 else self.lc["default_attr_cat"]
                if cat in self.lc["attr_root"]:
                    continue
                full_path = attr_path if attr_path[-1] == attr_val else attr_path + [attr_val]
                distractors = self.get_distractors(full_path, num_distractors, report_positive_set)
                opts, idx = self._build_options(attr_val, distractors)
                if len(opts) <= 1:
                    # 没有足够干扰项时放弃该题，避免退化成只有一个选项的无效样本。
                    continue

                qa_list.append(
                    make_qa_record(
                        report_id=report_id,
                        clinical_entity_type=self.task_type,
                        clinical_entity=sign_name,
                        clinical_entity_idx=current_idx,
                        attribute=attr_name,
                        question_idx=0,
                        question_type=question_type,
                        question=self.lc["q_attribute"].format(prefix, cat),
                        options=opts,
                        answer=idx,
                        evidence=evidence,
                    )
                )

        return qa_list


class HierarchicalQAGenerator:
    def __init__(self, task_type="finding", lang="zh"):
        self.task_type = task_type
        self.spec = TASK_SPECS[task_type]
        self.lang = lang
        self.lc = self.spec["lang_config"][lang]
        self.report_items_key = self.spec["report_items_key"]

    def _build_options(self, answer_text):
        options_list = [self.lc["yes"], self.lc["no"]]
        return options_list, options_list.index(answer_text)

    def process_report(self, report_data, question_type="hierarchical"):
        qa_list = []
        report_id = report_data.get("report_id", "Unknown")
        items = report_data.get(self.report_items_key, [])

        for item in items:
            if not isinstance(item, dict):
                continue

            sign_name = item.get("clinical_entity", {}).get("normalized_name")
            sign_path = item.get("clinical_entity", {}).get("mapped_path", [])
            evidence = item.get("evidence_span", "")
            is_absent = item.get("presence", {}).get("normalized_name") == "absent"

            if not sign_path or sign_path == ["Other"] or sign_path[-1] == "Other":
                continue
            if not self.spec["hier_include_absent"] and (not sign_name or is_absent):
                continue

            answer_text = self.lc["no"] if is_absent else self.lc["yes"]
            for ancestor in sign_path[1:-1]:
                # 对路径上的中间父节点生成存在性问题，训练模型做层级泛化推理。
                opts_bool, idx_bool = self._build_options(answer_text)
                qa_list.append(
                    make_qa_record(
                        report_id=report_id,
                        clinical_entity_type=self.task_type,
                        clinical_entity=ancestor,
                        clinical_entity_idx=0,
                        attribute="presence",
                        question_idx=0,
                        question_type=question_type,
                        question=self.lc["q_presence_hier"].format(ancestor),
                        options=opts_bool,
                        answer=idx_bool,
                        evidence=evidence,
                    )
                )

        return qa_list


class NegativeQAGenerator:
    def __init__(self, config_data, task_type="finding", lang="zh", fallback_leaf_nodes=None):
        self.task_type = task_type
        self.spec = TASK_SPECS[task_type]
        self.lang = lang
        self.lc = self.spec["lang_config"][lang]
        self.report_items_key = self.spec["report_items_key"]
        self.tree = config_data.get(self.spec["tree_key"], {})
        self.fallback_leaf_nodes = set(fallback_leaf_nodes or [])
        self.all_tree_paths = []
        self.parent_to_leaf_children = defaultdict(set)
        self._extract_all_paths(self.tree, current_path=[])

        self.all_tree_nodes = set()
        for path in self.all_tree_paths:
            self.all_tree_nodes.update(path)

        self.leaf_nodes = self._get_leaf_nodes(self.tree)

    def _get_leaf_nodes(self, node):
        leaves = set()
        if isinstance(node, dict):
            if not node:
                return leaves
            for key, value in node.items():
                if not value:
                    leaves.add(key)
                else:
                    leaves.update(self._get_leaf_nodes(value))
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, str):
                    leaves.add(item)
                else:
                    leaves.update(self._get_leaf_nodes(item))
        elif isinstance(node, str):
            leaves.add(node)
        return leaves

    def _extract_all_paths(self, node, current_path):
        if isinstance(node, dict):
            for key, value in node.items():
                new_path = current_path + [key]
                self.all_tree_paths.append(new_path)
                if not value:
                    self.parent_to_leaf_children[tuple(current_path)].add(key)
                self._extract_all_paths(value, new_path)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, str):
                    self.parent_to_leaf_children[tuple(current_path)].add(item)
                    self.all_tree_paths.append(current_path + [item])
                else:
                    self._extract_all_paths(item, current_path)
        elif isinstance(node, str):
            self.parent_to_leaf_children[tuple(current_path)].add(node)
            self.all_tree_paths.append(current_path + [node])

    def _build_options(self):
        options_list = [self.lc["yes"], self.lc["no"]]
        return options_list, options_list.index(self.lc["no"])

    def _extract_masked_fallback_nodes(self, items):
        masked_nodes = set()
        if not self.fallback_leaf_nodes:
            return masked_nodes

        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("presence", {}).get("normalized_name") == "absent":
                continue

            clinical_entity = item.get("clinical_entity", {})
            normalized_name = clinical_entity.get("normalized_name")
            mapped_path = clinical_entity.get("mapped_path", [])

            if not normalized_name or normalized_name not in self.leaf_nodes:
                continue
            if normalized_name in self.fallback_leaf_nodes:
                continue

            if mapped_path and mapped_path[-1] == normalized_name:
                parent_path = tuple(mapped_path[:-1])
            else:
                parent_path = tuple(mapped_path)

            for sibling in self.parent_to_leaf_children.get(parent_path, set()):
                if sibling in self.fallback_leaf_nodes:
                    # 如果同父节点下已有阳性叶子，就不再把兜底叶子拿来做阴性题，减少语义冲突。
                    masked_nodes.add(sibling)

        return masked_nodes

    def build_full_negative_qas(self, report_id, question_type="negative"):
        qa_list = []
        opts, ans_idx = self._build_options()
        for node in sorted(self.leaf_nodes):
            if node in self.spec["negative_ignore_words"]:
                continue
            qa_list.append(
                make_qa_record(
                    report_id=report_id,
                    clinical_entity_type=self.task_type,
                    clinical_entity=node,
                    clinical_entity_idx=0,
                    attribute="presence",
                    question_idx=0,
                    question_type=question_type,
                    question=self.lc["q_presence"].format(node),
                    options=opts,
                    answer=ans_idx,
                    evidence="",
                )
            )
        return qa_list

    def process_report(self, report_data, max_samples=0, question_type="negative"):
        qa_list = []
        report_id = report_data.get("report_id", "Unknown")
        items = report_data.get(self.report_items_key, [])
        masked_fallback_nodes = self._extract_masked_fallback_nodes(items)

        mentioned_nodes = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized_name = item.get("clinical_entity", {}).get("normalized_name")
            mapped_path = item.get("clinical_entity", {}).get("mapped_path", [])

            if normalized_name and normalized_name != "Other":
                mentioned_nodes.add(normalized_name)
            if mapped_path and mapped_path != ["Other"]:
                mentioned_nodes.update(mapped_path)

        tainted_nodes = set()
        for path in self.all_tree_paths:
            if set(path).intersection(mentioned_nodes):
                # 只要某条路径与报告中已提及节点有交集，就整条路径都视为“污染”，
                # 避免把父子/兄弟近邻节点误采成阴性。
                tainted_nodes.update(path)

        safe_negative_nodes = list((self.all_tree_nodes - tainted_nodes).intersection(self.leaf_nodes))
        safe_negative_nodes = [
            node for node in safe_negative_nodes
            if node not in self.spec["negative_ignore_words"] and node not in masked_fallback_nodes
        ]
        random.shuffle(safe_negative_nodes)
        if max_samples > 0:
            safe_negative_nodes = safe_negative_nodes[:max_samples]

        for node in safe_negative_nodes:
            opts, ans_idx = self._build_options()
            qa_list.append(
                make_qa_record(
                    report_id=report_id,
                    clinical_entity_type=self.task_type,
                    clinical_entity=node,
                    clinical_entity_idx=0,
                    attribute="presence",
                    question_idx=0,
                    question_type=question_type,
                    question=self.lc["q_presence"].format(node),
                    options=opts,
                    answer=ans_idx,
                    evidence="",
                )
            )

        return qa_list


def normalize_evidence(evidence):
    # 统一把 evidence 规整成字符串列表，便于去重合并。
    if isinstance(evidence, list):
        return [str(e) for e in evidence if e]
    if isinstance(evidence, str):
        return [evidence] if evidence else []
    if evidence is None:
        return []
    return [str(evidence)]


def make_qa_record(
    report_id,
    clinical_entity,
    attribute,
    question,
    options,
    answer,
    evidence,
    clinical_entity_idx=0,
    clinical_entity_type="finding",
    question_type="base",
    question_idx=0,
):
    return {
        "report_id": report_id,
        "clinical_entity_type": clinical_entity_type,
        "clinical_entity": clinical_entity,
        "clinical_entity_idx": clinical_entity_idx,
        "attribute": attribute,
        "question_idx": question_idx,
        "question_type": question_type,
        "question": question,
        "options": options,
        "answer": answer,
        "evidence": evidence,
    }


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_intermediate_dir(output_path):
    output_dir = os.path.dirname(output_path) or "."
    output_name = os.path.splitext(os.path.basename(output_path))[0]
    intermediate_dir = os.path.join(output_dir, f"{output_name}_intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    return intermediate_dir


def import_tree_output_cleaner():
    try:
        from .step3_map_to_tree import TreeOutputCleaner
    except ImportError:
        from generate_qas.pipelines.step3_map_to_tree import TreeOutputCleaner
    return TreeOutputCleaner


def build_cleaning_paths(intermediate_dir, task_type):
    return {
        "cleaned_input": os.path.join(intermediate_dir, f"{task_type}_cleaned_input.jsonl"),
        "dropped_output": os.path.join(intermediate_dir, f"{task_type}_cleaning_dropped.jsonl"),
        "stats_output": os.path.join(intermediate_dir, f"{task_type}_cleaning_stats.json"),
    }


def clean_inputs_for_generation(args, task_types, input_paths, intermediate_dir):
    TreeOutputCleaner = import_tree_output_cleaner()
    cleaned_input_paths = {}
    cleaning_stats = {}

    for task_type in task_types:
        # QA 生成前再次复用 tree cleaning 逻辑，保证输入是干净且一致的。
        input_path = input_paths[task_type]
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"未找到输入文件: {input_path}")
        cleaning_paths = build_cleaning_paths(intermediate_dir, task_type)
        cleaner = TreeOutputCleaner(
            knowledge_config_path=args.knowledge_tree,
            task_type=task_type,
            lang=args.lang,
        )
        stats = cleaner.process_file(
            input_path,
            cleaning_paths["cleaned_input"],
            cleaning_paths["dropped_output"],
            cleaning_paths["stats_output"],
        )
        cleaned_input_paths[task_type] = cleaning_paths["cleaned_input"]
        cleaning_stats[task_type] = stats or {}

    return cleaned_input_paths, cleaning_stats


def deduplicate_qas(qa_list):
    merged = {}
    merge_logs = []
    for data in qa_list:
        key = (
            data.get("report_id"),
            data.get("clinical_entity"),
            data.get("clinical_entity_type"),
            data.get("attribute"),
            data.get("question_type"),
            data.get("question"),
        )
        evidence_list = normalize_evidence(data.get("evidence"))

        if key not in merged:
            new_data = data.copy()
            new_data["evidence"] = sorted(set(evidence_list))
            merged[key] = new_data
        else:
            # 相同题干/实体/属性的重复题目只保留一条，但把证据做并集保留下来。
            existing_data = merged[key].copy()
            merged[key]["evidence"] = sorted(set(merged[key]["evidence"] + evidence_list))
            merge_logs.append(
                {
                    "action": "deduplicate_merge",
                    "report_id": data.get("report_id"),
                    "clinical_entity_type": data.get("clinical_entity_type"),
                    "clinical_entity": data.get("clinical_entity"),
                    "clinical_entity_idx": data.get("clinical_entity_idx", 0),
                    "attribute": data.get("attribute"),
                    "question_idx": data.get("question_idx", 0),
                    "question_type": data.get("question_type"),
                    "question": data.get("question"),
                    "kept_record_before": existing_data,
                    "merged_record_incoming": data,
                    "kept_record_after": merged[key].copy(),
                }
            )

    return list(merged.values()), merge_logs


def assign_question_indices(qa_list):
    grouped = defaultdict(list)
    change_logs = []
    for qa in qa_list:
        key = (
            qa.get("report_id"),
            qa.get("clinical_entity"),
            qa.get("clinical_entity_idx", 0),
            qa.get("attribute"),
            qa.get("question_idx", 0),
        )
        grouped[key].append(qa)

    for group in grouped.values():
        if len(group) <= 1:
            continue

        distinct_questions = []
        question_to_idx = {}
        for qa in group:
            question = qa.get("question")
            if question not in question_to_idx:
                question_to_idx[question] = len(distinct_questions)
                distinct_questions.append(question)

        if len(distinct_questions) <= 1:
            continue

        for qa in group:
            old_idx = qa.get("question_idx", 0)
            new_idx = question_to_idx[qa.get("question")]
            if old_idx != new_idx:
                # 同一实体属性下如果存在多个不同问法，重排 question_idx 保证编号稳定连续。
                change_logs.append(
                    {
                        "action": "question_idx_reassigned",
                        "report_id": qa.get("report_id"),
                        "clinical_entity_idx": qa.get("clinical_entity_idx", 0),
                        "clinical_entity": qa.get("clinical_entity"),
                        "clinical_entity_type": qa.get("clinical_entity_type"),
                        "attribute": qa.get("attribute"),
                        "question_type": qa.get("question_type"),
                        "question": qa.get("question"),
                        "old_question_idx": old_idx,
                        "new_question_idx": new_idx,
                    }
                )
            qa["question_idx"] = new_idx

    return qa_list, change_logs


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, 1):
            if not line.strip():
                continue
            yield line_num, json.loads(line)


def extract_report_ids_from_raw_json_files(paths):
    report_ids = []
    for path in paths or []:
        with open(path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
        if not isinstance(data, list):
            raise ValueError(f"{path} 不是 case 列表。")

        for case in data:
            if not isinstance(case, dict):
                continue
            videos = case.get("videos", [])
            if not isinstance(videos, list):
                continue
            # for video in videos:
            #     if video:
            #         report_ids.append(str(video))
            report_ids.append(str(videos[0]))
    return list(dict.fromkeys(report_ids))


def build_parser(default_task_type=None):
    parser = argparse.ArgumentParser(description="Unified Medical QA Generator")
    parser.add_argument("-c", "--config", required=True, help="YAML 配置文件路径")
    return parser


def run_generation(args, parser=None):
    args = make_args(**vars(args))

    should_load_yaml = bool(args.config) and not any(
        [
            args.output,
            args.input,
            args.finding_input,
            args.diagnosis_input,
            args.raw_json_files,
        ]
    )
    if should_load_yaml:
        yaml_config_path = args.config
        try:
            with open(yaml_config_path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}
                args = apply_yaml_config(args, yaml_data)
                print(f"📄 已从 YAML 加载配置: {yaml_config_path}")
        except Exception as e:
            print(f"❌ YAML 加载失败: {e}")
            return 1

    args = resolve_run_paths(args)
    if args.task not in {"finding", "diagnosis", "both"}:
        print(f"❌ 不支持的 task: {args.task}，只支持 finding / diagnosis / both。")
        return 1
    if not all([args.output, args.knowledge_tree]):
        print("❌ 缺少必要参数 (output, knowledge_tree)。请通过 YAML 提供。")
        return 1
    intermediate_dir = build_intermediate_dir(args.output)

    task_types = sorted(TASK_SPECS.keys()) if args.task == "both" else [args.task]
    input_paths = {}
    for task_type in task_types:
        specific_input = getattr(args, f"{task_type}_input", None)
        input_path = specific_input or args.input
        if not input_path:
            print(
                f"❌ 缺少 {task_type} 的输入文件。"
                f" 请提供 --{task_type}-input，或使用包含全部内容的 --input。"
            )
            return 1
        input_paths[task_type] = input_path

    cleaned_input_paths, cleaning_stats = clean_inputs_for_generation(
        args,
        task_types,
        input_paths,
        intermediate_dir,
    )

    with open(args.knowledge_tree, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    config_data = config_data.get(args.lang, {})

    generator_specs = []
    negative_generators = {}

    if args.gen_base:
        for task_type in task_types:
            generator_specs.append(
                ("base", BaseQAGenerator(config_data, task_type=task_type, lang=args.lang))
            )
            print(f"🟢 启用: 基础样本生成器 (Base, {task_type})")
    if args.gen_hier:
        for task_type in task_types:
            generator_specs.append(
                ("hierarchical", HierarchicalQAGenerator(task_type=task_type, lang=args.lang))
            )
            print(f"🟢 启用: 父节点推理生成器 (Hierarchical, {task_type})")
    if args.gen_neg:
        fallback_leaf_nodes_config = getattr(args, "negative_fallback_leaf_nodes", {})
        for task_type in task_types:
            if isinstance(fallback_leaf_nodes_config, dict):
                fallback_leaf_nodes = fallback_leaf_nodes_config.get(task_type, [])
            elif isinstance(fallback_leaf_nodes_config, list):
                fallback_leaf_nodes = fallback_leaf_nodes_config
            else:
                fallback_leaf_nodes = []
            negative_generator = NegativeQAGenerator(
                config_data,
                task_type=task_type,
                lang=args.lang,
                fallback_leaf_nodes=fallback_leaf_nodes,
            )
            generator_specs.append(("negative", negative_generator))
            negative_generators[task_type] = negative_generator
            print(f"🟢 启用: 阴性样本生成器 (Negative, {task_type}, Max={args.neg_max})")

    if not generator_specs:
        print("⚠️ 警告: 未开启任何生成器。请至少指定 --gen-base, --gen-hier 或 --gen-neg 中的一个。")
        return 1

    bucketed_qas = defaultdict(list)
    base_report_ids_by_task = defaultdict(set)
    for task_type in task_types:
        task_generators = [
            (question_type, generator)
            for question_type, generator in generator_specs
            if generator.task_type == task_type
        ]
        input_path = cleaned_input_paths[task_type]
        print(f"📥 读取输入: {task_type} <- {input_paths[task_type]}")
        print(f"🧹 清洗输出: {task_type} -> {input_path}")

        for line_num, report in iter_jsonl(input_path):
            for question_type, generator in task_generators:
                if isinstance(generator, NegativeQAGenerator):
                    generated = generator.process_report(
                        report,
                        max_samples=args.neg_max,
                        question_type=question_type,
                    )
                elif isinstance(generator, BaseQAGenerator):
                    generated = generator.process_report(
                        report,
                        num_distractors=args.num_distractors,
                        question_type=question_type,
                    )
                else:
                    generated = generator.process_report(report, question_type=question_type)

                bucket_key = (task_type, question_type)
                bucketed_qas[bucket_key].extend(generated)
                if question_type == "base":
                    report_id = report.get("report_id")
                    if report_id and generated:
                        base_report_ids_by_task[task_type].add(str(report_id))
            if line_num % 1000 == 0:
                print(f"正在处理... {task_type} 已解析 {line_num} 条报告")

    missing_negative_logs = []
    if args.gen_neg and args.raw_json_files and not args.gen_base:
        print("⚠️ 警告: 提供了 raw_json_files 但未开启 --gen-base，已跳过基于 base 覆盖率的纯阴性补齐。")

    if args.gen_neg and args.raw_json_files and args.gen_base:
        all_report_ids = [str(report_id) for report_id in extract_report_ids_from_raw_json_files(args.raw_json_files)]
        for task_type in task_types:
            covered_report_ids = base_report_ids_by_task.get(task_type, set())
            missing_report_ids = [report_id for report_id in all_report_ids if report_id not in covered_report_ids]
            generator = negative_generators.get(task_type)
            if generator is not None:
                for report_id in missing_report_ids:
                    # 对完全没有 base 样本的报告补全“纯阴性”题，提升报告级覆盖率。
                    generated = generator.build_full_negative_qas(report_id, question_type="negative")
                    bucketed_qas[(task_type, "negative")].extend(generated)
                    missing_negative_logs.append(
                        {
                            "action": "missing_negative_added",
                            "task_type": task_type,
                            "report_id": report_id,
                            "generated_count": len(generated),
                            "coverage_mode": "per_task_base",
                        }
                    )

    before_count = sum(len(qa_list) for qa_list in bucketed_qas.values())
    all_qas = []
    dedup_logs = []
    for (task_type, question_type), qa_list in bucketed_qas.items():
        raw_path = os.path.join(intermediate_dir, f"{task_type}_{question_type}_raw.jsonl")
        write_jsonl(raw_path, qa_list)

        deduped_qas, bucket_dedup_logs = deduplicate_qas(qa_list)
        for log in bucket_dedup_logs:
            log["bucket"] = f"{task_type}_{question_type}"
        dedup_logs.extend(bucket_dedup_logs)
        all_qas.extend(deduped_qas)

    all_qas, question_idx_logs = assign_question_indices(all_qas)

    dedup_log_path = os.path.join(intermediate_dir, "dedup_changes.jsonl")
    write_jsonl(dedup_log_path, dedup_logs)

    question_idx_log_path = os.path.join(intermediate_dir, "question_idx_changes.jsonl")
    write_jsonl(question_idx_log_path, question_idx_logs)

    missing_negative_log_path = os.path.join(intermediate_dir, "missing_negative_added.jsonl")
    write_jsonl(missing_negative_log_path, missing_negative_logs)

    cleaning_summary_path = os.path.join(intermediate_dir, "cleaning_summary.json")
    with open(cleaning_summary_path, "w", encoding="utf-8") as outfile:
        json.dump(cleaning_stats, outfile, ensure_ascii=False, indent=2)

    after_count = len(all_qas)

    question_type_order = {"base": 0, "hierarchical": 1, "negative": 2}
    clinical_entity_type_order = {name: idx for idx, name in enumerate(sorted(TASK_SPECS.keys()))}
    all_qas.sort(
        key=lambda x: (
            # 固定排序让主输出和中间 diff 更稳定，便于追踪生成逻辑变化。
            x["report_id"],
            clinical_entity_type_order.get(x.get("clinical_entity_type"), 99),
            x["clinical_entity"],
            x.get("clinical_entity_idx", 0),
            DEFAULT_ATTRIBUTE_ORDER.get(x["attribute"], 99),
            x["question_idx"],
            question_type_order.get(x.get("question_type"), 99),
            x["question"],
        )
    )

    write_jsonl(args.output, all_qas)

    print("-" * 30)
    print(f"✅ QA生成及去重完成！(类型: {args.task}, 语言: {args.lang})")
    print(f"🗂️ 中间过程已保存: {intermediate_dir}")
    print(f"📄 主输出文件: {args.output}")
    if before_count > 0:
        print("📊 统计:")
        print(f"   - 去重前总题数: {before_count}")
        print(f"   - 去重后总题数: {after_count}")
        print(f"   - 移除重复题目: {before_count - after_count}")
        print(f"   - 去重比例: {(before_count - after_count) / before_count:.2%}")
        print(f"   - 去重变更记录数: {len(dedup_logs)}")
        print(f"   - question_idx 调整数: {len(question_idx_logs)}")
        print(f"   - 纯阴性补齐记录数: {len(missing_negative_logs)}")
    for task_type in task_types:
        stats = cleaning_stats.get(task_type, {})
        if stats:
            item_label = "finding" if task_type == "finding" else "diagnosis"
            print(
                f"   - {task_type} 清洗: absent_{item_label}s={stats.get(f'absent_{item_label}s', 0)}, "
                f"dropped_{item_label}s={stats.get(f'dropped_{item_label}s', 0)}"
            )
    print("-" * 30)
    return 0


def make_args(**kwargs):
    defaults = {
        "task": "both",
        "lang": "zh",
        "input": None,
        "finding_input": None,
        "diagnosis_input": None,
        "raw_json_files": None,
        "output": None,
        "knowledge_tree": None,
        "gen_base": False,
        "gen_hier": False,
        "gen_neg": False,
        "num_distractors": 3,
        "neg_max": 3,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def main(default_task_type=None):
    parser = build_parser(default_task_type=default_task_type)
    args = parser.parse_args()
    return run_generation(args, parser=parser)


if __name__ == "__main__":
    raise SystemExit(main())
