#!/usr/bin/env python3
"""Format SWIFT JSONL output using a reference report JSON file.

The output keeps the same structure and order as ``refer_path``. For each
sample, the corresponding line's ``response`` from ``input_path`` replaces
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


def load_swift_responses(path: str) -> list[str]:
    responses: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            if not isinstance(item, dict):
                raise TypeError(f"Line {line_no}: expected a JSON object.")
            if "response" not in item:
                raise ValueError(f"Line {line_no}: missing required field 'response'.")
            responses.append(item["response"])
    return responses


def validate_messages(item: dict[str, Any], index: int) -> list[dict[str, Any]]:
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


def format_reports(
    refer_reports: list[dict[str, Any]],
    responses: list[str],
) -> list[dict[str, Any]]:
    if len(refer_reports) != len(responses):
        raise ValueError(
            f"refer_path has {len(refer_reports)} items, but input_path has "
            f"{len(responses)} non-empty lines."
        )

    formatted_reports = []
    for index, (refer_item, response) in enumerate(
        zip(refer_reports, responses), start=1
    ):
        output_item = copy.deepcopy(refer_item)
        messages = validate_messages(output_item, index)
        messages[-1]["content"] = response
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
    responses = load_swift_responses(args.input_path)
    formatted_reports = format_reports(refer_reports, responses)

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
                "input_items": len(responses),
                "output_items": len(formatted_reports),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
