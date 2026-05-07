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

from tqdm import tqdm


def try_parse_ast_to_json(function_string: str):
    try:
        # 兜底解析：有些模型会返回近似 Python 函数调用/关键字参数格式，
        # 这里把 `foo(a=1, b="x")` 这类文本尽量还原成字典。
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


def try_parse_json(input_str: str):
    if not input_str:
        return None

    try:
        # 先走标准 JSON 解析，速度最快，且不会引入额外“修复”带来的歧义。
        return json.loads(input_str)
    except json.JSONDecodeError:
        pass

    input_str = input_str.strip()
    if input_str.startswith("```"):
        # 兼容模型常见的 Markdown fenced code block 输出。
        input_str = re.sub(r"^```json\s*|^```\s*|```$", "", input_str, flags=re.MULTILINE).strip()

    for pattern in [r"(\[.*\])", r"(\{.*\})"]:
        # 如果前后夹杂了解释性文本，只截取最外层数组/对象主体尝试解析。
        match = re.search(pattern, input_str, re.DOTALL)
        if match:
            candidate = match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                input_str = candidate
                break

    try:
        from json_repair import repair_json
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "未找到 json_repair。请先安装 `json-repair` 后再运行。"
        ) from exc

    try:
        # `json_repair` 用于修复缺引号、尾逗号等常见 LLM 输出格式问题。
        repaired = repair_json(json_str=input_str, return_objects=True)
        if isinstance(repaired, (dict, list)):
            return repaired
    except Exception:
        pass

    _, ast_result = try_parse_ast_to_json(input_str)
    if ast_result:
        return ast_result

    return None


def parse_llm_response(response_text, target_field):
    parsed = try_parse_json(response_text)
    expected_key = "findings" if target_field == "finding" else "diagnoses"

    if isinstance(parsed, list):
        # 直接返回列表，兼容模型输出裸数组。
        return parsed
    if isinstance(parsed, dict):
        if isinstance(parsed.get(expected_key), list):
            return parsed[expected_key]
        fallback_key = "diagnoses" if expected_key == "findings" else "findings"
        if isinstance(parsed.get(fallback_key), list):
            # 容忍 prompt/模型偶发返回了另一种任务字段名。
            return parsed[fallback_key]
        if parsed:
            # 单对象输出时包装成长度为 1 的列表，保持主流程统一按“多条特征”处理。
            return [parsed]
        return []
    return []


def process_reports_placeholder(input_file, lang):
    """
    输入原始报告文件，输出处理好的报告列表。

    返回格式:
    [
        {
            "report_id": 0,
            "finding": "...",
            "diagnosis": "...",
        }
    ]

    当前为占位实现，默认按 jsonl 逐行读取。
    后续你可以在这里替换成任意解析逻辑，而无需修改主流程。
    """
    results = []
    # print(input_file)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for case in data:
        try:
            # 1. report_id
            report_id = None
            if "videos" in case and len(case["videos"]) > 0:
                # 这里约定首个 video 标识唯一报告；后续步骤都依赖这个 ID 做去重与关联。
                report_id = case["videos"][0]

            # 2. 找 assistant 内容
            assistant_text = ""
            for msg in case.get("messages", []):
                if msg.get("role") == "assistant":
                    assistant_text = msg.get("content", "")
                    break

            if not assistant_text:
                continue

            # 3. 提取 所见 和 印象
            # 兼容可能的中文标点 variations
            finding = ""
            diagnosis = ""

            # 使用正则更稳健
            if lang == "zh":
                match = re.search(r"所见[:：](.*?)印象[:：](.*)", assistant_text, re.S)
            elif lang == "en":
                match = re.search(r"Findings[:：](.*?)Impression[:：](.*)", assistant_text, re.S)

            if match:
                finding = match.group(1).strip()
                diagnosis = match.group(2).strip()
            else:
                # 主正则失败时退化为独立提取，避免因为缺少某一段导致整条报告被丢掉。
                # fallback：只找到其中一个
                if lang == "zh":
                    finding_match = re.search(r"所见[:：](.*)", assistant_text, re.S)
                    diagnosis_match = re.search(r"印象[:：](.*)", assistant_text, re.S)
                elif lang == "en":
                    finding_match = re.search(r"Findings[:：](.*)", assistant_text, re.S)
                    diagnosis_match = re.search(r"Impression[:：](.*)", assistant_text, re.S)                   

                if finding_match:
                    finding = finding_match.group(1).strip()
                if diagnosis_match:
                    diagnosis = diagnosis_match.group(1).strip()

            results.append({
                "report_id": report_id,
                "finding": finding,
                "diagnosis": diagnosis
            })

            # print(results)

        except Exception as e:
            # 可选：打印或记录错误
            # print(f"Error processing case {case.get('id')}: {e}")
            continue

    return results


class UnifiedExtractor:
    def __init__(self, config_path):

        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        lang = self.cfg["setting"]["lang"].lower()
        if lang not in {"zh", "en"}:
            raise ValueError("setting.lang 只支持 zh 或 en。")

        target_field = self.cfg["setting"]["target_field"].lower()
        if target_field not in {"finding", "diagnosis"}:
            raise ValueError("setting.target_field 只支持 finding 或 diagnosis。")

        lang_suffix = "zh" if lang == "zh" else "en"

        self.lang = lang
        self.target_field = target_field
        self.input_file = self.cfg["paths"]["input"].format(lang_suffix=lang_suffix)
        # print(self.input_file)
        self.output_file = self.cfg["paths"]["output"].format(
            lang_suffix=lang_suffix,
            target_field=target_field,
        )
        self.prompt_template = self.cfg["prompts"][target_field][lang]
        self.model = self.cfg["api_config"]["model"]
        self.base_url = self.cfg["api_config"]["base_url"]
        self.env_key = self.cfg["api_config"].get("env_key", "OPENAI_API_KEY")
        self.max_workers = self.cfg["setting"]["max_workers"]
        self.save_batch_size = self.cfg["setting"]["save_batch_size"]
        self.max_retries = self.cfg["setting"]["max_retries"]

        api_key = os.getenv(self.env_key)
        if not api_key:
            raise ValueError(f"缺少环境变量 {self.env_key}。")

        self.client = openai.OpenAI(base_url=self.base_url, api_key=api_key)
        self.lock = threading.Lock()
        # 多线程 worker 先写入内存 buffer，再按批次落盘，减少频繁 IO。
        self.buffer = []
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        # Suppress per-request success logs from the HTTP client; keep warnings/errors visible.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    def load_processed_ids(self):
        processed_ids = set()
        if not os.path.exists(self.output_file):
            return processed_ids

        with open(self.output_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # 历史脏行不影响断点续跑，直接跳过。
                    continue
                if "report_id" in data:
                    processed_ids.add(data["report_id"])
        return processed_ids

    def build_report_items(self):
        # 输入解析与后续抽取逻辑解耦，便于后续替换不同来源的数据格式。
        reports = process_reports_placeholder(
            input_file=self.input_file,
            lang = self.lang
        )
        return reports

    def extract_tasks(self):
        processed_ids = self.load_processed_ids()
        logging.info("已发现 %s 条历史处理记录，将自动跳过。", len(processed_ids))

        reports = self.build_report_items()
        tasks = []
        for report in reports:
            report_id = report["report_id"]
            report_text = (report.get(self.target_field) or "").strip()
            if not report_text or report_id in processed_ids:
                continue
            tasks.append(report)
        return tasks

    def process_single_report(self, report_item):
        report_id = report_item["report_id"]
        report_text = report_item[self.target_field]
        prompt = self.prompt_template.replace("{report_text}", report_text)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                result_text = response.choices[0].message.content
                extracted_array = parse_llm_response(result_text, self.target_field)
                return {
                    "report_id": report_id,
                    "target_field": self.target_field,
                    "finding": report_item.get("finding", ""),
                    "diagnosis": report_item.get("diagnosis", ""),
                    "original_text": report_text,
                    "extracted_features": extracted_array,
                }
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    time.sleep(2)
                else:
                    logging.error("report_id %s 提取失败 (已达最大重试次数): %s", report_id, exc)
                    return {
                        "report_id": report_id,
                        "target_field": self.target_field,
                        "finding": report_item.get("finding", ""),
                        "diagnosis": report_item.get("diagnosis", ""),
                        "original_text": report_text,
                        "extracted_features": [],
                    }

    def flush_buffer_to_disk(self):
        if not self.buffer:
            return
        with open(self.output_file, "a", encoding="utf-8") as f:
            for result in self.buffer:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        self.buffer.clear()

    def run(self):
        tasks = self.extract_tasks()
        total_tasks = len(tasks)
        logging.info(
            "本次需处理的新报告数量: %s, 语言: %s, 提取目标: %s",
            total_tasks,
            self.lang,
            self.target_field,
        )

        if total_tasks == 0:
            logging.info("所有报告已处理完毕。")
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_single_report, task) for task in tasks]
            with tqdm(total=total_tasks, desc="多线程提取进度") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        with self.lock:
                            self.buffer.append(result)
                            if len(self.buffer) >= self.save_batch_size:
                                self.flush_buffer_to_disk()
                    except Exception as exc:
                        logging.error("线程执行发生异常: %s", exc)
                    finally:
                        pbar.update(1)

        with self.lock:
            self.flush_buffer_to_disk()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    extractor = UnifiedExtractor(args.config)
    extractor.run()


if __name__ == "__main__":
    main()
