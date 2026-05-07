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
    "en": """You are a medical reasoning assistant.

You may use general medical knowledge.

If the question cannot be answered using general knowledge alone, choose the option that indicates:
"Insufficient information" or similar meaning.

Do not guess.

Only choose a specific answer if you are confident.

Question:
{question}""",

    "zh": """你是一个医学推理助手。

你可以使用医学常识进行判断。

如果仅凭常识无法回答该问题，请选择表示“信息不足”或类似含义的选项。

不要猜测。

只有在有把握时才选择具体答案。

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
        qas = result["qas"]

        for qa in tqdm(qas):
            question = qa["question"]
            answer = qa["answer"]
            prompt = PROMPT_TEMPLATE[args.language].format(question=question)
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
