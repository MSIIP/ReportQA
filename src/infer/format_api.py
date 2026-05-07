import argparse
import json
import os


args = argparse.ArgumentParser()
args.add_argument("--refer_path", type=str, required=True)
args.add_argument("--input_path", type=str, required=True)
args.add_argument("--output_path", type=str, required=True)
args = args.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_refer_item(item, index):
    if "messages" not in item or not isinstance(item["messages"], list) or not item["messages"]:
        raise ValueError(f"Invalid refer item at index {index}: missing non-empty 'messages'.")

    last_message = item["messages"][-1]
    if not isinstance(last_message, dict) or "content" not in last_message:
        raise ValueError(
            f"Invalid refer item at index {index}: messages[-1] must be a dict with 'content'."
        )


def build_predict_map(data):
    predict_map = {}

    for index, item in enumerate(data):
        item_id = item.get("id")
        if item_id is None:
            raise ValueError(f"Invalid input item at index {index}: missing 'id'.")
        if "predict" not in item:
            raise ValueError(f"Invalid input item '{item_id}': missing 'predict'.")
        if item_id in predict_map:
            raise ValueError(f"Duplicate id found in input data: {item_id}")

        predict_map[item_id] = item["predict"]

    return predict_map


if __name__ == "__main__":
    refer_data = load_json(args.refer_path)
    input_data = load_json(args.input_path)

    if not isinstance(refer_data, list):
        raise ValueError("refer_path must contain a JSON list.")
    if not isinstance(input_data, list):
        raise ValueError("input_path must contain a JSON list.")

    predict_map = build_predict_map(input_data)

    output_data = []
    for index, item in enumerate(refer_data):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid refer item at index {index}: each item must be a dict.")

        validate_refer_item(item, index)

        item_id = item.get("id")
        if item_id is None:
            raise ValueError(f"Invalid refer item at index {index}: missing 'id'.")
        if item_id not in predict_map:
            raise ValueError(f"Missing prediction for refer item id: {item_id}")

        new_item = dict(item)
        new_messages = list(item["messages"])
        last_message = dict(new_messages[-1])
        last_message["content"] = predict_map[item_id]
        new_messages[-1] = last_message
        new_item["messages"] = new_messages
        output_data.append(new_item)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
