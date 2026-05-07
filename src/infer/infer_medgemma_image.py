import argparse
import copy
import json
import os
from pathlib import Path


import torch
from PIL import Image
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
        description="Run MedGemma inference on a JSON file in report generation format."
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


def build_user_content(user_text, image_paths):
    image_count = user_text.count("<image>")
    if image_count != len(image_paths):
        raise ValueError(
            f"Mismatch between <image> placeholders ({image_count}) and image paths ({len(image_paths)})."
        )

    cleaned_text = "\n".join(
        line for line in user_text.splitlines() if line.strip() != "<image>"
    ).strip()

    content = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        content.append({"type": "image", "image": image})

    if cleaned_text:
        content.append({"type": "text", "text": cleaned_text})

    return content


def build_messages(sample):
    user_message = sample["messages"][0]
    image_paths = sample["images"]
    return [
        {
            "role": user_message["role"],
            "content": build_user_content(user_message["content"], image_paths),
        }
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
    for sample in tqdm(samples, desc="Running inference", unit="sample"):
        messages = build_messages(sample)
        prediction = run_inference(model, processor, messages, args.max_new_tokens)

        result = copy.deepcopy(sample)
        if len(result["messages"]) > 1 and result["messages"][1]["role"] == "assistant":
            result["messages"][1]["content"] = prediction
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
