import argparse
import base64
import copy
import io
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_NUM_SLICES = 16
WINDOW_CLIPS = [(-1024, 1024), (-135, 215), (0, 80)]


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
        description="Run MedGemma inference on NIfTI abdominal CT volumes listed in a JSON file."
    )
    parser.add_argument("--input_json", default=None, help="Input dataset JSON path.")
    parser.add_argument("--output_json", default=None, help="Output predictions JSON path.")
    parser.add_argument("--model_id", default=None, help="Model directory or Hugging Face model id.")
    parser.add_argument("--max_samples", type=int, default=None, help="Only run the first N samples.")
    parser.add_argument("--num_slices", type=int, default=DEFAULT_NUM_SLICES, help="Number of center slices to use per volume.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of generated tokens.")
    return parser.parse_args()


def load_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {json_path}, but got {type(data).__name__}.")
    return data


def norm(ct_slice, min_value, max_value):
    ct_slice = np.clip(ct_slice, min_value, max_value).astype(np.float32)
    ct_slice -= min_value
    ct_slice /= (max_value - min_value)
    ct_slice *= 255.0
    return ct_slice


def window(ct_slice):
    return np.stack([norm(ct_slice, clip[0], clip[1]) for clip in WINDOW_CLIPS], axis=-1)


def convert_ctrg_slice(ct_slice):
    ct_slice = np.clip(ct_slice, 0, 255).astype(np.uint8)
    return np.stack([ct_slice, ct_slice, ct_slice], axis=-1)


def load_volume_slices(nii_path, num_slices, skip_windowing=False):
    volume = nib.load(str(nii_path)).get_fdata()
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume in {nii_path}, but got shape {volume.shape}.")

    depth = volume.shape[2]
    if depth > num_slices:
        slice_indices = [
            int(round(i / num_slices * (depth - 1)))
            for i in range(1, num_slices + 1)
        ]
    else:
        slice_indices = list(range(depth))

    processed_slices = []
    for slice_idx in slice_indices:
        ct_slice = volume[:, :, slice_idx]
        if skip_windowing:
            processed_slices.append(convert_ctrg_slice(ct_slice))
        else:
            windowed_slice = window(ct_slice)
            processed_slices.append(np.round(windowed_slice, 0).astype(np.uint8))
    return processed_slices


def encode_image(data):
    image_format = "jpeg"
    with io.BytesIO() as img_bytes:
        with Image.fromarray(data) as img:
            img.save(img_bytes, format=image_format)
        img_bytes.seek(0)
        encoded = base64.b64encode(img_bytes.getbuffer()).decode("utf-8")
    return f"data:image/{image_format};base64,{encoded}"


def build_user_content(user_text, video_paths, num_slices, skip_windowing=False):
    if user_text.count("<video>") != len(video_paths):
        raise ValueError(
            f"Mismatch between <video> placeholders ({user_text.count('<video>')}) "
            f"and video paths ({len(video_paths)})."
        )

    cleaned_text = "\n".join(
        line for line in user_text.splitlines() if line.strip() != "<video>"
    ).strip()

    content = []
    for video_path in video_paths:
        slices = load_volume_slices(video_path, num_slices, skip_windowing=skip_windowing)
        for slice_number, ct_slice in enumerate(slices, start=1):
            content.append({"type": "image", "image": encode_image(ct_slice)})
            content.append({"type": "text", "text": f"SLICE {slice_number}"})

    if cleaned_text:
        content.append({"type": "text", "text": cleaned_text})
    return content


def build_messages(sample, num_slices, skip_windowing=False):
    user_message = sample["messages"][0]
    return [
        {
            "role": user_message["role"],
            "content": build_user_content(
                user_message["content"],
                sample["videos"],
                num_slices,
                skip_windowing=skip_windowing,
            ),
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
    # skip_windowing = "CTRG" in str(args.input_json)
    skip_windowing = "ctrg" in str(args.input_json).lower()

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
    for sample in tqdm(samples, desc="Running 3D inference", unit="sample"):
        messages = build_messages(sample, args.num_slices, skip_windowing=skip_windowing)
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
