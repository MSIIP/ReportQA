import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 json 中的 (report_id, clinical_entity_type, clinical_entity, clinical_entity_idx, attribute, question_idx, question_type) 删除 jsonl 中对应条目"
    )
    parser.add_argument(
        "--filter_path",
        type=str,
        required=True,
        help="保存待删除七元组列表的 json 文件路径"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="输入的 jsonl 文件路径"
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="输出的过滤后 jsonl 文件路径"
    )
    return parser.parse_args()


def load_filter_set(json_path):
    """
    读取过滤条件 json 文件，返回一个 set，
    每个元素是 (report_id, clinical_entity_type, clinical_entity, clinical_entity_idx, attribute, question_idx, question_type)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filter_set = set()
    for item in data:
        if len(item) != 7:
            raise ValueError(f"过滤文件中存在非法条目（长度不是7）：{item}")

        report_id, clinical_entity_type, clinical_entity, clinical_entity_idx, attribute, question_idx, question_type = item

        key = (
            str(report_id),
            str(clinical_entity_type),
            str(clinical_entity),
            str(clinical_entity_idx),
            str(attribute),
            str(question_idx),
            str(question_type),
        )
        filter_set.add(key)

    return filter_set


def filter_jsonl(jsonl_path, output_path, filter_set):
    """
    删除 filter_set 中出现的 jsonl 条目，并写入新的 jsonl 文件
    """
    total_count = 0
    removed_count = 0
    kept_count = 0

    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total_count += 1
            item = json.loads(line)

            key = (
                str(item["report_id"]),
                str(item["clinical_entity_type"]),
                str(item["clinical_entity"]),
                str(item["clinical_entity_idx"]),
                str(item["attribute"]),
                str(item["question_idx"]),
                str(item["question_type"]),
            )

            # 如果在 filter_set 中，则删除；否则保留
            if key in filter_set:
                removed_count += 1
            else:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept_count += 1

    return total_count, removed_count, kept_count


def main():
    args = parse_args()

    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    filter_set = load_filter_set(args.filter_path)
    total_count, removed_count, kept_count = filter_jsonl(
        jsonl_path=args.input_jsonl,
        output_path=args.output_jsonl,
        filter_set=filter_set
    )

    print(f"待删除键数量: {len(filter_set)}")
    print(f"原始 jsonl 条目数: {total_count}")
    print(f"删除条目数: {removed_count}")
    print(f"保留条目数: {kept_count}")
    print(f"输出文件已保存到: {args.output_jsonl}")


if __name__ == "__main__":
    main()
