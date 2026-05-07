#!/usr/bin/env python3
"""Format TRL JSON output using a reference report JSON file.

The output keeps the same structure and order as ``refer_path``. For each
sample, ``input_item["messages_inferred"][-1]["content"]`` replaces
``reference_item["messages"][-1]["content"]``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any


def load_json_list(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a top-level JSON list.")
    if not all(isinstance(item, dict) for item in data):
        raise TypeError(f"{path} must contain a list of JSON objects.")
    return data


def validate_reference_messages(
    item: dict[str, Any], index: int
) -> list[dict[str, Any]]:
    messages = item.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(
            f"Reference item {index}: field 'messages' must be a non-empty list."
        )
    if not isinstance(messages[-1], dict):
        raise ValueError(f"Reference item {index}: last message must be an object.")
    if "content" not in messages[-1]:
        raise ValueError(
            f"Reference item {index}: last message is missing field 'content'."
        )
    return messages


def get_inferred_content(item: dict[str, Any], index: int) -> str:
    messages_inferred = item.get("messages_inferred")
    if not isinstance(messages_inferred, list) or not messages_inferred:
        raise ValueError(
            f"Input item {index}: field 'messages_inferred' must be a non-empty list."
        )
    last_message = messages_inferred[-1]
    if not isinstance(last_message, dict):
        raise ValueError(f"Input item {index}: last inferred message must be an object.")
    if "content" not in last_message:
        raise ValueError(
            f"Input item {index}: last inferred message is missing field 'content'."
        )
    return last_message["content"]


def format_reports(
    refer_reports: list[dict[str, Any]],
    input_reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(refer_reports) != len(input_reports):
        raise ValueError(
            f"refer_path has {len(refer_reports)} items, but input_path has "
            f"{len(input_reports)} items."
        )

    formatted_reports = []
    for index, (refer_item, input_item) in enumerate(
        zip(refer_reports, input_reports), start=1
    ):
        output_item = copy.deepcopy(refer_item)
        messages = validate_reference_messages(output_item, index)
        messages[-1]["content"] = get_inferred_content(input_item, index)
        formatted_reports.append(output_item)

    return formatted_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refer_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    refer_reports = load_json_list(args.refer_path)
    input_reports = load_json_list(args.input_path)
    formatted_reports = format_reports(refer_reports, input_reports)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(formatted_reports, f, indent=4, ensure_ascii=False)

    print(
        json.dumps(
            {
                "refer_path": args.refer_path,
                "input_path": args.input_path,
                "output_path": args.output_path,
                "refer_items": len(refer_reports),
                "input_items": len(input_reports),
                "output_items": len(formatted_reports),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
