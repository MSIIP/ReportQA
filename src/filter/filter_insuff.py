import argparse
import json
import os


args = argparse.ArgumentParser()
args.add_argument("--results_path", type=str, default=None)
args.add_argument("--filter_insuff_path", type=str, default=None)
args.add_argument("--language", type=str, default="en", choices=["en", "zh"])
args = args.parse_args()

insuff_info = {
    "en": "Insufficient information",
    "zh": "信息不足",
}


def filter_insuff_units(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    unit_list = []
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

            pred = qa["predict"]
            pred_str = f"{pred.strip()}. {insuff_info[args.language]}"
            if pred_str.lower() in qa["question"].lower():
                unit_list.append(key)

    return unit_list


if __name__ == "__main__":
    unit_list = filter_insuff_units(args.results_path)

    os.makedirs(os.path.dirname(args.filter_insuff_path), exist_ok=True)
    with open(args.filter_insuff_path, "w", encoding="utf-8") as f:
        json.dump(unit_list, f, ensure_ascii=False, indent=4)
