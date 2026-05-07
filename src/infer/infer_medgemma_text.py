import argparse
import copy
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


def normalize_token_id(token_id):
    if isinstance(token_id, int):
        return token_id
    if isinstance(token_id, (list, tuple)):
        if not token_id:
            return None
        return normalize_token_id(token_id[0])
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MedGemma text-only inference on a JSON file."
    )
    parser.add_argument("--input_json", default=None, help="Input dataset JSON path.")
    parser.add_argument("--output_json", default=None, help="Output predictions JSON path.")
    parser.add_argument("--model_id", default=None, help="Model directory or Hugging Face model id.")
    parser.add_argument("--max_samples", type=int, default=None, help="Only run the first N samples.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of generated tokens.")
    return parser.parse_args()


def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {json_path}, but got {type(data).__name__}.")
    return data


def normalize_message_content(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if isinstance(content, list):
        normalized_content = []
        for item in content:
            if isinstance(item, str):
                normalized_content.append({"type": "text", "text": item})
                continue

            if not isinstance(item, dict):
                raise ValueError(f"Unsupported content item type: {type(item).__name__}")

            item_type = item.get("type")
            if item_type == "text":
                normalized_content.append({"type": "text", "text": item.get("text", "")})
                continue

            raise ValueError(
                "Text-only inference only supports text content, "
                f"but found content item type: {item_type!r}"
            )

        return normalized_content

    raise ValueError(f"Unsupported message content type: {type(content).__name__}")


def build_messages(sample):
    if "messages" not in sample or not sample["messages"]:
        raise ValueError("Each sample must contain a non-empty 'messages' field.")

    messages = sample["messages"]
    prompt_messages = messages[:-1] if messages[-1]["role"] == "assistant" else messages
    if not prompt_messages:
        raise ValueError("No prompt messages found after removing the reference assistant response.")

    return [
        {
            "role": message["role"],
            "content": normalize_message_content(message["content"]),
        }
        for message in prompt_messages
    ]


def run_inference(model, processor, messages, max_new_tokens):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=normalize_token_id(model.generation_config.pad_token_id),
        )
        generation = generation[0][input_len:]

    return processor.decode(generation, skip_special_tokens=True)


def main():
    args = parse_args()
    samples = load_dataset(args.input_json)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    pad_token_id = normalize_token_id(model.generation_config.pad_token_id)
    eos_token_id = normalize_token_id(model.generation_config.eos_token_id)
    processor_pad_token_id = normalize_token_id(getattr(processor, "pad_token_id", None))
    if pad_token_id is None:
        pad_token_id = processor_pad_token_id if processor_pad_token_id is not None else eos_token_id
    model.generation_config.pad_token_id = pad_token_id

    results = []
    for sample in tqdm(samples, desc="Running text-only inference", unit="sample"):
        messages = build_messages(sample)
        prediction = run_inference(model, processor, messages, args.max_new_tokens)

        result = copy.deepcopy(sample)
        if len(result["messages"]) > 1 and result["messages"][-1]["role"] == "assistant":
            result["messages"][-1]["content"] = prediction
        else:
            result["messages"].append({"role": "assistant", "content": prediction})
        results.append(result)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
