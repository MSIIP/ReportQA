import argparse
import json
import os
import random
import string


args = argparse.ArgumentParser()
args.add_argument("--reports_path_txt", type=str, default=None)
args.add_argument("--reports_path", type=str, default=None)
args.add_argument("--qas_path_jsonl", type=str, default=None)
args.add_argument("--qas_path", type=str, default=None)
args.add_argument("--language", type=str, default="en", choices=["en", "zh"])
args = args.parse_args()

PROMPT_TEMPLATE = {
    "en": """{question}

Options:
{option_str}

Please select the most appropriate answer from the above options, and only output the option letter.""",

    "zh": """{question}

选项：
{option_str}

请从上述选项中选择一个最合适的答案，只输出选项字母。"""
}

insuff_info = {
    "en": "Insufficient information",
    "zh": "信息不足",
}


def build_qa_pairs(sample, do_shuffle=False):
    """
    输入:
        sample: dict
        do_shuffle: 是否在每个样本上再随机打乱（增强多样性）

    输出:
        list[(prompt, answer_letter)]
    """

    options = sample["options"] + [insuff_info[args.language]]  # 添加“信息不足”选项
    answer_idx = sample["answer"]
    n = len(options)

    if n == 0:
        raise ValueError("options is empty")

    letters = list(string.ascii_uppercase)
    if n > len(letters):
        raise ValueError("too many options (>26)")

    # ---------- 工具函数 ----------
    def rotate(lst, k):
        k = k % len(lst)
        return lst[-k:] + lst[:-k]

    def shuffle_with_answer(opts, ans_idx):
        paired = list(enumerate(opts))
        random.shuffle(paired)

        new_opts = [opt for _, opt in paired]
        for new_idx, (old_idx, _) in enumerate(paired):
            if old_idx == ans_idx:
                return new_opts, new_idx

    def build_prompt(opts):
        option_str = "\n".join([
            f"{letters[i]}. {opt}" for i, opt in enumerate(opts)
        ])

        return PROMPT_TEMPLATE[args.language].format(question=sample['question'], option_str=option_str)

    # ---------- 主流程 ----------
    results = []

    # ctx = context if context is not None else sample.get("evidence", "")

    for shift in range(n):
        # 1. 旋转（保证均衡）
        new_options = rotate(options, shift)
        new_answer_idx = (answer_idx + shift) % n

        # 2. 可选 shuffle
        if do_shuffle:
            new_options, new_answer_idx = shuffle_with_answer(new_options, new_answer_idx)

        # 3. 构建 prompt + answer
        prompt = build_prompt(new_options)
        answer = letters[new_answer_idx]

        results.append((prompt, answer))

    return results


if __name__ == "__main__":
    if args.reports_path_txt is not None:
        reports = []
        with open(args.reports_path_txt, "r", encoding="utf-8") as f:
            idx = 0
            for line in f:
                line = line.strip()  # 去掉换行符和首尾空格
                reports.append({
                    "id": str(idx),
                    "messages": [
                        {"role": "user", "content": ""},
                        {"role": "assistant", "content": line},
                    ],
                    "videos": [str(idx)],
                })
                idx += 1
        os.makedirs(os.path.dirname(args.reports_path), exist_ok=True)
        with open(args.reports_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, ensure_ascii=False, indent=4)

    qas = []
    with open(args.qas_path_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            qa_pairs= build_qa_pairs(item)
            for question, answer in qa_pairs:
                qas.append({
                    "id": f"{item['report_id']}.{item['clinical_entity_type']}.{item['clinical_entity']}.{item['clinical_entity_idx']}.{item['attribute']}.{item['question_idx']}.{item['question_type']}.{answer}",
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    "videos": [str(item["report_id"])],
                    "report_id": item["report_id"],
                    "clinical_entity_type": item["clinical_entity_type"],
                    "clinical_entity": item["clinical_entity"],
                    "clinical_entity_idx": item["clinical_entity_idx"],
                    "attribute": item["attribute"],
                    "question_idx": item["question_idx"],
                    "question_type": item["question_type"],
                    "answer_option": answer,
                    "answer_str": item["options"][item["answer"]],
                })
    os.makedirs(os.path.dirname(args.qas_path), exist_ok=True)
    with open(args.qas_path, "w", encoding="utf-8") as f:
        json.dump(qas, f, ensure_ascii=False, indent=4)
