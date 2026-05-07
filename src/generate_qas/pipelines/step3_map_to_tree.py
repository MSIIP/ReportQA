import argparse
import ast
import concurrent.futures
import json
import logging
import os
import re
import threading
import time

import openai
import yaml
from json_repair import repair_json
from tqdm import tqdm


TASK_CONFIG = {
    # 不同任务的差异集中在这里配置，后面流程尽量共用。
    "finding": {
        "result_key": "findings",
        "item_label": "finding",
        "task_label": "Finding",
        "prompt_input_mode": "features_only",
        "empty_result": [],
        "knowledge_tree_key": "FINDINGS_TREE",
        "extra_prompt_tree_placeholders": {},
        "extra_valid_main_entities": ["影像征象", "abnormal_imaging_findings"],
        "extra_valid_attributes": ["present", "uncertain", "definite", "probable", "possible", "影像征象"],
    },
    "diagnosis": {
        "result_key": "diagnoses",
        "item_label": "diagnosis",
        "task_label": "Diagnosis",
        "prompt_input_mode": "features_only",
        "empty_result": [],
        "knowledge_tree_key": "DIAGNOSIS_TREE",
        "extra_prompt_tree_placeholders": {
            "{imaging_findings_tree}": "FINDINGS_TREE",
        },
        "extra_valid_main_entities": ["影像征象", "abnormal_imaging_findings", "Cerebral infarction"],
        "extra_valid_attributes": ["absent", "present", "uncertain", "definite", "probable", "possible", "影像征象"],
    },
}


def is_absent_presence_feature(feature):
    if not isinstance(feature, dict):
        return False

    presence = feature.get("presence")
    if isinstance(presence, dict):
        candidates = [
            presence.get("normalized_name"),
            presence.get("raw"),
            presence.get("value"),
            presence.get("name"),
        ]
    else:
        candidates = [presence]

    return any(str(candidate).strip().lower() == "absent" for candidate in candidates if candidate is not None)


def try_parse_ast_to_json(function_string: str) -> tuple[str, dict]:
    try:
        tree = ast.parse(str(function_string).strip())
        json_result = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                args = {kw.arg: kw.value for kw in node.keywords}
                for arg, value in args.items():
                    json_result[arg] = ast.literal_eval(value)
        return function_string, json_result
    except Exception:
        return function_string, {}


def try_parse_json_object(input_str: str):
    if not input_str:
        return {}

    try:
        return json.loads(input_str)
    except json.JSONDecodeError:
        pass

    input_str = input_str.strip()
    if input_str.startswith("```"):
        # 兼容模型把 JSON 包在代码块里的情况。
        input_str = re.sub(r"^```json\s*|^```\s*|```$", "", input_str, flags=re.MULTILINE)

    match = re.search(r"(\{.*\})|(\[.*\])", input_str, re.DOTALL)
    if match:
        # 去掉前后解释性自然语言，只保留最可能的 JSON 主体。
        input_str = match.group(0)

    try:
        repaired = repair_json(json_str=input_str, return_objects=True)
        if isinstance(repaired, (dict, list)):
            return repaired
    except Exception:
        pass

    _, ast_result = try_parse_ast_to_json(input_str)
    return ast_result


def load_records_from_file(file_path: str):
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        return records

    jsonl_records = []
    jsonl_success = False
    # 优先按 JSONL 解析，因为上游多为逐行增量产出。
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if isinstance(record, dict):
                jsonl_records.append(record)
                jsonl_success = True
            else:
                jsonl_success = False
                jsonl_records = []
                break
        except json.JSONDecodeError:
            jsonl_success = False
            jsonl_records = []
            break

    if jsonl_success:
        return jsonl_records

    try:
        # 其次兼容标准 JSON 数组/对象文件。
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    text_len = len(raw_text)
    # 最后兜底处理“多个 JSON 对象直接拼接”的非标准文件。
    while idx < text_len:
        while idx < text_len and raw_text[idx].isspace():
            idx += 1
        if idx >= text_len:
            break
        try:
            record, next_idx = decoder.raw_decode(raw_text, idx)
            if isinstance(record, dict):
                records.append(record)
            elif isinstance(record, list):
                records.extend(item for item in record if isinstance(item, dict))
            idx = next_idx
        except json.JSONDecodeError:
            break

    return records


class TreeMapper:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.task_type = self.cfg["setting"]["task_type"].lower()
        if self.task_type not in TASK_CONFIG:
            supported = ", ".join(sorted(TASK_CONFIG))
            raise ValueError(f"Unsupported task_type: {self.task_type}. Supported values: {supported}")

        self.lang = self.cfg["setting"]["lang"].lower()
        self.lang_suffix = "zh" if self.lang == "zh" else "en"
        self.task_cfg = TASK_CONFIG[self.task_type]
        self.include_evidence_span_in_prompt = bool(
            self.cfg["setting"].get("include_evidence_span_in_prompt", False)
        )

        path_format_kwargs = {
            "lang_suffix": self.lang_suffix,
            "task_type": self.task_type,
        }
        self.input_file = self.cfg["paths"]["input"].format(**path_format_kwargs)
        self.output_file = self.cfg["paths"]["output"].format(**path_format_kwargs)
        self.cleaned_output_file = self.cfg["paths"]["cleaned_output"].format(
            **path_format_kwargs
        )
        cleaned_root, cleaned_ext = os.path.splitext(self.cleaned_output_file)
        self.dropped_output_file = f"{cleaned_root}_dropped{cleaned_ext or '.jsonl'}"
        self.knowledge_config_path = self.cfg["cleaning"]["knowledge_config"]
        self.knowledge_tree = self._load_knowledge_tree(self.knowledge_config_path)
        raw_prompt_tmpl = self.cfg["prompts"][self.task_type][self.lang]
        self.prompt_tmpl = self._inject_tree_definitions(raw_prompt_tmpl)
        self.env_key = self.cfg["api_config"].get("env_key", "OPENAI_API_KEY")
        api_key = os.getenv(self.env_key)
        if not api_key:
            raise ValueError(f"缺少环境变量 {self.env_key}。")

        self.client = openai.OpenAI(
            base_url=self.cfg["api_config"]["base_url"],
            api_key=api_key,
        )

        self.lock = threading.Lock()
        self.buffer = []
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        # Suppress per-request success logs from the HTTP client; keep warnings/errors visible.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    def _load_knowledge_tree(self, knowledge_config_path: str):
        with open(knowledge_config_path, "r", encoding="utf-8") as f:
            knowledge_config = json.load(f)

        if self.lang in knowledge_config:
            return knowledge_config[self.lang]
        return knowledge_config

    def _render_tree(self, tree_key: str):
        tree = self.knowledge_tree.get(tree_key)
        if tree is None:
            raise KeyError(f"Missing `{tree_key}` in knowledge_config for lang `{self.lang}`")
        return json.dumps(tree, ensure_ascii=False, indent=2)

    def _inject_tree_definitions(self, prompt_tmpl: str):
        placeholder_map = {
            "{anatomy_tree}": self._render_tree("ANATOMY_TREE"),
            "{clinical_entity_tree}": self._render_tree(self.task_cfg["knowledge_tree_key"]),
            "{attributes_tree}": self._render_tree("ATTRIBUTES_TREE"),
        }
        for placeholder, tree_key in self.task_cfg.get("extra_prompt_tree_placeholders", {}).items():
            placeholder_map[placeholder] = self._render_tree(tree_key)
        missing_placeholders = [
            placeholder for placeholder in placeholder_map if placeholder not in prompt_tmpl
        ]
        if missing_placeholders:
            raise ValueError(
                "Prompt template is missing required tree placeholders: "
                + ", ".join(missing_placeholders)
            )

        for placeholder, rendered_tree in placeholder_map.items():
            # 在发给模型前把树结构直接展开，避免运行时再做额外上下文拼接。
            prompt_tmpl = prompt_tmpl.replace(placeholder, rendered_tree)
        return prompt_tmpl

    def _build_prompt_input(self, report_id, features):
        if self.task_cfg["prompt_input_mode"] == "wrapped_with_report":
            return {
                "report_id": report_id,
                "raw_extracted_features": features,
            }
        return features

    def _normalize_mapped_result(self, parsed_data):
        result_key = self.task_cfg["result_key"]

        if isinstance(parsed_data, list):
            return {result_key: parsed_data}

        if isinstance(parsed_data, dict):
            if result_key in parsed_data:
                return parsed_data
            # 模型若只返回单个 item 对象，也转成统一列表格式。
            return {result_key: [parsed_data]}

        return {result_key: list(self.task_cfg["empty_result"])}

    def _is_absent_feature(self, feature):
        # “absent” 特征不会参与树映射，避免把否定信息映射成正向实体。
        return is_absent_presence_feature(feature)

    def process_record(self, record):
        report_id = record.get("report_id")
        original_text = record.get("original_text", "")
        extracted_features = record.get("extracted_features", [])
        result_key = self.task_cfg["result_key"]

        if not extracted_features:
            return {
                "report_id": report_id,
                "original_text": original_text,
                result_key: list(self.task_cfg["empty_result"]),
            }

        filtered_features = [feature for feature in extracted_features if not self._is_absent_feature(feature)]
        if not filtered_features:
            return {
                "report_id": report_id,
                "original_text": original_text,
                result_key: list(self.task_cfg["empty_result"]),
            }

        clean_features = []
        for feature in filtered_features:
            clean_feature = {}
            for key, value in feature.items():
                if not value:
                    continue
                if key == "evidence_span" and not self.include_evidence_span_in_prompt:
                    # 证据片段可选是否喂给模型；默认关闭以减少 prompt 噪声。
                    continue
                clean_feature[key] = value
            clean_features.append(clean_feature)
        prompt_input = self._build_prompt_input(report_id, clean_features)
        prompt = self.prompt_tmpl.replace("{input_json}", json.dumps(prompt_input, ensure_ascii=False))
        
        raw_content = ""
        for attempt in range(self.cfg["setting"]["max_retries"]):
            try:
                response = self.client.chat.completions.create(
                    model=self.cfg["api_config"]["model"],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                raw_content = response.choices[0].message.content
                parsed_data = try_parse_json_object(raw_content)
                mapped_result = self._normalize_mapped_result(parsed_data)
                mapped_result.update({"report_id": report_id, "original_text": original_text})

                for idx, item in enumerate(mapped_result.get(result_key, [])):
                    # 默认按顺序把原始 evidence 挂回映射结果，便于后续 QA 回溯。
                    # 若模型返回数量和输入不一致，则显式打上 mismatch 标记。
                    if idx < len(filtered_features):
                        item["evidence_span"] = filtered_features[idx].get("evidence_span", "")
                    else:
                        item["evidence_span"] = "Mapping mismatch"

                return mapped_result

            except Exception as exc:
                if attempt < self.cfg["setting"]["max_retries"] - 1:
                    time.sleep(2)
                    continue

                logging.error("ID %s 映射失败: %s", report_id, exc)
                return {
                    "report_id": report_id,
                    "original_text": original_text,
                    result_key: [{"error": "Mapping Failed"}],
                    "raw": raw_content,
                }

    def save_disk(self):
        if not self.buffer:
            return

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, "a", encoding="utf-8") as f:
            for item in self.buffer:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        # 追加写后立即清空 buffer，避免重复落盘。
        self.buffer.clear()

    def run(self):
        processed = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        processed.add(json.loads(line)["report_id"])
                    except Exception:
                        continue

        tasks = []
        for record in load_records_from_file(self.input_file):
            try:
                if record["report_id"] not in processed:
                    # 仅处理未完成记录，支持中断后续跑。
                    tasks.append(record)
            except Exception:
                continue

        if not tasks:
            logging.info("所有记录已映射完毕，开始刷新 cleaned 输出。")
        else:
            logging.info(
                "开始映射 %s, 总数: %s, 语言: %s",
                self.task_cfg["task_label"],
                len(tasks),
                self.lang,
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.cfg["setting"]["max_workers"]
            ) as pool:
                futures = [pool.submit(self.process_record, task) for task in tasks]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(tasks),
                    desc=f"{self.task_cfg['task_label']} Mapping",
                ):
                    result = future.result()
                    with self.lock:
                        self.buffer.append(result)
                        if len(self.buffer) >= self.cfg["setting"]["save_batch_size"]:
                            self.save_disk()

            self.save_disk()
            logging.info("%s 原始映射输出已完成。", self.task_cfg["task_label"])

        cleaner = TreeOutputCleaner(
            knowledge_config_path=self.cfg["cleaning"]["knowledge_config"],
            task_type=self.task_type,
            lang=self.lang,
        )
        cleaner.process_file(
            self.output_file,
            self.cleaned_output_file,
            self.dropped_output_file,
        )
        logging.info("%s 清洗输出已完成。", self.task_cfg["task_label"])


class TreeOutputCleaner:
    def __init__(self, knowledge_config_path: str, task_type: str, lang: str):
        with open(knowledge_config_path, "r", encoding="utf-8") as f:
            knowledge_config = json.load(f)

        self.task_type = task_type
        self.lang = lang
        self.task_cfg = TASK_CONFIG[task_type]
        self.result_key = self.task_cfg["result_key"]
        self.item_label = self.task_cfg["item_label"]

        if lang in knowledge_config:
            self.knowledge_tree = knowledge_config[lang]
        else:
            self.knowledge_tree = knowledge_config

        self.valid_main_entities = self._extract_entities(
            self.knowledge_tree.get(self.task_cfg["knowledge_tree_key"], {})
        )
        self.valid_diagnosis_entities = self._extract_entities(
            self.knowledge_tree.get("DIAGNOSIS_TREE", {})
        )
        self.valid_attributes = self._extract_entities(self.knowledge_tree.get("ATTRIBUTES_TREE", {}))
        self.valid_anatomy = self._extract_entities(self.knowledge_tree.get("ANATOMY_TREE", {}))

        self.valid_main_entities.update(self.task_cfg["extra_valid_main_entities"])
        self.valid_attributes.update(self.task_cfg["extra_valid_attributes"])
        # 额外白名单用于兼容知识树未覆盖、但业务上可接受的过渡标签。

    def _extract_entities(self, tree_node):
        entities = set()

        def recurse(node):
            # 将树的所有层级键和值拍平成集合，用于后续合法性校验。
            if isinstance(node, dict):
                for key, value in node.items():
                    entities.add(key.strip())
                    recurse(value)
            elif isinstance(node, list):
                for item in node:
                    recurse(item)
            elif isinstance(node, str):
                entities.add(node.strip())

        recurse(tree_node)
        return entities

    def _is_valid_value(self, normalized_name, tree_set, allow_numeric=False):
        if not normalized_name:
            return False

        normalized_name = str(normalized_name).strip()
        if normalized_name in tree_set:
            return True

        if allow_numeric and (
            any(char.isdigit() for char in normalized_name)
            or "mm" in normalized_name.lower()
            or "cm" in normalized_name.lower()
        ):
            # 尺寸、数量级这类开放值不一定在树里穷举，因此允许数字类值直接通过。
            return True

        return False

    def _is_diagnosis_clinical_entity_wrong_tree(self, value_obj):
        if self.task_type != "diagnosis":
            return False

        mapped_path = value_obj.get("mapped_path")
        if not isinstance(mapped_path, list) or not mapped_path:
            return False

        root_name = str(mapped_path[0]).strip()
        root_name_lower = root_name.lower()
        return ("征象" in root_name) or ("findings" in root_name_lower)

    def _is_other_mapping(self, value_obj):
        if not isinstance(value_obj, dict):
            return False

        normalized_name = str(value_obj.get("normalized_name", "")).strip()
        if normalized_name == "Other":
            return True

        mapped_path = value_obj.get("mapped_path")
        if not isinstance(mapped_path, list):
            return False

        return any(str(node).strip() == "Other" for node in mapped_path)

    def _clean_item(self, item):
        keys_to_delete = []
        is_main_entity_valid = True
        drop_reason = None
        clinical_entity_normalized = None

        for key, value_obj in item.items():
            if key == "evidence_span":
                continue

            if not (isinstance(value_obj, dict) and "normalized_name" in value_obj):
                continue

            normalized_name = value_obj.get("normalized_name")
            raw_value = str(value_obj.get("raw", "")).strip().lower()

            if key == "clinical_entity":
                clinical_entity_normalized = str(normalized_name).strip() if normalized_name else None
                if clinical_entity_normalized == "Other":
                    is_main_entity_valid = False
                    drop_reason = "clinical_entity_other"
                    break
                if self.task_type == "diagnosis":
                    if not self._is_valid_value(normalized_name, self.valid_diagnosis_entities):
                        is_main_entity_valid = False
                        if self._is_diagnosis_clinical_entity_wrong_tree(value_obj):
                            drop_reason = "clinical_entity_wrong_tree"
                        else:
                            drop_reason = "invalid_clinical_entity"
                        break
                    if self._is_diagnosis_clinical_entity_wrong_tree(value_obj):
                        is_main_entity_valid = False
                        drop_reason = "clinical_entity_wrong_tree"
                        break
                elif not self._is_valid_value(normalized_name, self.valid_main_entities):
                    # 主实体非法时整条 item 失去语义锚点，需要直接丢弃。
                    is_main_entity_valid = False
                    drop_reason = "invalid_clinical_entity"
                    break
            elif key == "presence":
                if is_absent_presence_feature({"presence": value_obj}):
                    # QA 和统计默认都围绕正向存在项构建，否定项在清洗阶段剔除。
                    is_main_entity_valid = False
                    drop_reason = "presence_absent"
                    break
                if self._is_other_mapping(value_obj):
                    # `presence` 若落到 Other，说明存在性映射失败；整条实体不再可信。
                    is_main_entity_valid = False
                    drop_reason = "invalid_presence"
                    break
            elif key == "location":
                if self._is_other_mapping(value_obj) or not self._is_valid_value(normalized_name, self.valid_anatomy):
                    # 解剖部位非法时只删除该属性，不直接丢弃整条实体。
                    keys_to_delete.append(key)
            else:
                if self._is_other_mapping(value_obj) or not self._is_valid_value(normalized_name, self.valid_attributes, allow_numeric=True):
                    keys_to_delete.append(key)

        if not is_main_entity_valid or "clinical_entity" not in item:
            if "clinical_entity" not in item and drop_reason is None:
                drop_reason = "missing_clinical_entity"
            return None, 0, 1, drop_reason

        for key in keys_to_delete:
            del item[key]

        return item, len(keys_to_delete), 0, None

    def process_file(
        self,
        input_path: str,
        output_path: str,
        dropped_output_path: str | None = None,
        stats_output_path: str | None = None,
    ):
        if not os.path.exists(input_path):
            logging.warning("未找到原始输出文件，跳过清洗: %s", input_path)
            return {}

        valid_samples = []
        dropped_samples = []
        total_samples = 0
        total_item_count = 0
        stripped_attrs_count = 0
        dropped_items_count = 0
        absent_items_count = 0
        dropped_reason_counts = {}
        dropped_due_to_other = []

        with open(input_path, "r", encoding="utf-8") as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue

                total_samples += 1
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    continue

                items = sample.get(self.result_key, [])
                valid_items = []
                dropped_items = []

                for item in items:
                    total_item_count += 1
                    if is_absent_presence_feature(item):
                        absent_items_count += 1

                    cleaned_item, stripped_count, dropped_count, drop_reason = self._clean_item(item)
                    stripped_attrs_count += stripped_count
                    dropped_items_count += dropped_count
                    if cleaned_item is not None:
                        valid_items.append(cleaned_item)
                    else:
                        # dropped 输出保留原因和原始 item，方便人工复盘知识树覆盖缺口。
                        reason = drop_reason or "filtered_out"
                        dropped_reason_counts[reason] = dropped_reason_counts.get(reason, 0) + 1
                        if reason == "clinical_entity_other":
                            dropped_due_to_other.append(
                                {
                                    "report_id": sample.get("report_id"),
                                    "original_text": sample.get("original_text", ""),
                                    self.item_label: item,
                                }
                            )
                        dropped_items.append(
                            {
                                "drop_reason": reason,
                                self.item_label: item,
                            }
                        )

                sample[self.result_key] = valid_items
                if valid_items:
                    valid_samples.append(sample)
                if dropped_items:
                    dropped_samples.append(
                        {
                            "report_id": sample.get("report_id"),
                            "original_text": sample.get("original_text", ""),
                            self.result_key: dropped_items,
                        }
                    )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as outfile:
            for sample in valid_samples:
                outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

        if dropped_output_path:
            os.makedirs(os.path.dirname(dropped_output_path), exist_ok=True)
            with open(dropped_output_path, "w", encoding="utf-8") as outfile:
                for sample in dropped_samples:
                    outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

        stats = {
            "input_path": input_path,
            "output_path": output_path,
            "dropped_output_path": dropped_output_path,
            "task_type": self.task_type,
            "lang": self.lang,
            "total_reports": total_samples,
            f"total_{self.item_label}s": total_item_count,
            f"absent_{self.item_label}s": absent_items_count,
            f"dropped_{self.item_label}s": dropped_items_count,
            "stripped_attributes": stripped_attrs_count,
            "dropped_reason_counts": dropped_reason_counts,
            "dropped_due_to_other": dropped_due_to_other,
        }

        if stats_output_path:
            os.makedirs(os.path.dirname(stats_output_path), exist_ok=True)
            with open(stats_output_path, "w", encoding="utf-8") as outfile:
                json.dump(stats, outfile, ensure_ascii=False, indent=2)

        logging.info(
            "清洗完成: 输入报告=%s, 保留报告=%s, 输入%s数=%s, absent=%s, 删除属性=%s, 丢弃%s=%s, Other导致丢弃=%s, 输出=%s, 丢弃输出=%s, 统计输出=%s",
            total_samples,
            len(valid_samples),
            self.item_label,
            total_item_count,
            absent_items_count,
            stripped_attrs_count,
            self.item_label,
            dropped_items_count,
            dropped_reason_counts.get("clinical_entity_other", 0),
            output_path,
            dropped_output_path,
            stats_output_path,
        )
        return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="YAML 配置文件路径")
    args = parser.parse_args()

    mapper = TreeMapper(args.config)
    mapper.run()


if __name__ == "__main__":
    main()
