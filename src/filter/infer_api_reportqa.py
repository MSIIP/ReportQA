import argparse
import json
import os
from tqdm import tqdm

from openai import OpenAI


args = argparse.ArgumentParser()
args.add_argument("--reports_path", type=str, default=None)
args.add_argument("--qas_path", type=str, default=None)
args.add_argument("--results_path", type=str, default=None)
args.add_argument("--language", type=str, default="en", choices=["en", "zh"])
args.add_argument("--model", type=str, default=None)
args.add_argument("--base_url", type=str, default=None)
args.add_argument("--api_key", type=str, default=None)
args = args.parse_args()

client = OpenAI(
    base_url=args.base_url,
    api_key=args.api_key,
)
system_prompt=""
PROMPT_TEMPLATE = {
    "en": """You are a medical report analysis assistant.

Answer the question based primarily on the given report.
You may use general medical knowledge to assist interpretation.

Decision policy:
1. If the report contains explicit evidence, use it to answer.
2. If the report does not mention the queried abnormality, but the question refers to a standard clinical finding that would normally be reported if present, assume it is absent.
3. Only choose the option indicating "Insufficient information" if the report truly contains no relevant information and no reasonable inference can be made.

Additional rules:
- Prefer to infer from the report whenever possible.
- If the report provides partial or indirect evidence, choose the best supported answer.
- Do not guess or assume facts not grounded in the report.
- "Insufficient information" should be selected only as a last resort.

Report:
{report}

Question:
{question}""",

    "zh": """你是一个医学报告分析助手。

请主要基于给定的报告回答问题，
可以使用医学常识进行辅助理解，但不能替代报告内容。

决策原则：
1. 如果报告中有明确证据，直接根据证据作答；
2. 如果报告未提及所问异常，但该异常属于临床常规检查项（通常出现时会被报告），则优先判断为“不存在”；
3. 只有在报告中确实没有相关信息，且无法进行合理推断时，才选择“信息不足”或类似选项。

补充规则：
- 尽量根据报告内容进行推断；
- 如果存在部分或间接证据，应选择最合理的答案；
- 不要猜测，也不要假设报告中不存在的信息；
- “信息不足”应作为最后选择。

报告：
{report}

问题：
{question}"""
}


if __name__ == "__main__":
    with open(args.reports_path, "r", encoding="utf-8") as f:
        reports = json.load(f)
    with open(args.qas_path, "r", encoding="utf-8") as f:
        qas = json.load(f)

    results = {item["videos"][0]: {"report": item["messages"][-1]["content"], "qas": []} for item in reports}
    for item in qas:
        filepath = item["videos"][0]
        results[filepath]["qas"].append({
            "id": item["id"],
            "question": item["messages"][-2]["content"],
            "answer": item["messages"][-1]["content"],
            "report_id": item["report_id"],
            "clinical_entity_type": item["clinical_entity_type"],
            "clinical_entity": item["clinical_entity"],
            "clinical_entity_idx": item["clinical_entity_idx"],
            "attribute": item["attribute"],
            "question_idx": item["question_idx"],
            "question_type": item["question_type"],
            "answer_option": item["answer_option"],
            "answer_str": item["answer_str"],
        })

    for filepath, result in tqdm(results.items()):
        report = result["report"]
        qas = result["qas"]

        for qa in tqdm(qas):
            question = qa["question"]
            answer = qa["answer"]
            prompt = PROMPT_TEMPLATE[args.language].format(report=report, question=question)
            # print(f"Prompt:\n{prompt}\n")

            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ],
                # temperature=0.2,
                # max_tokens=1024,
            )
            output_text = response.choices[0].message.content

            qa["predict"] = output_text
            # print(f"Predict:\n{output_text}\n")

    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    with open(args.results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
