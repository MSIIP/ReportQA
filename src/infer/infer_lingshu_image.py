import argparse
import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D batch inference for lingshu-medical-mllm/Lingshu-7B on MIMIC-CXR style JSON."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="lingshu-medical-mllm/Lingshu-7B",
        help="HuggingFace model id or local path.",
    )
    parser.add_argument("--data-json", type=str, required=True, help="Input dataset JSON path.")
    parser.add_argument("--output-json", type=str, required=True, help="Output prediction JSON path.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in dataset.")
    parser.add_argument("--max-samples", type=int, default=None, help="Process first N samples from start-index.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model load dtype.",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="sdpa",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention: sdpa 无需 flash-attn。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='整模上单一设备，例如 cuda:0（推荐）。不设则用 device_map="auto"。',
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--save-every", type=int, default=20, help="Dump intermediate results every N samples.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if exists.")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="以 bitsandbytes 4-bit 量化加载权重（省显存，需安装 bitsandbytes）。",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    mapper = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapper[dtype_name]


def dump_json(data: Any, out_path: str) -> None:
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def clean_user_text(content: str) -> str:
    text = str(content or "")
    text = text.replace("<image>", "").replace("<video>", "").strip()
    return text if text else "Please generate a report for the given chest X-ray image."


def get_image_paths(sample: Dict[str, Any]) -> List[str]:
    if isinstance(sample.get("images"), list) and sample["images"]:
        return [str(x) for x in sample["images"]]
    if sample.get("image"):
        return [str(sample["image"])]
    raise ValueError("2D mode requires 'images' list or 'image' field.")


def build_messages(history: List[Dict[str, Any]], image_paths: List[str], user_text: str) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = [{"type": "image", "image": p} for p in image_paths]
    user_content.append({"type": "text", "text": user_text})
    return history + [{"role": "user", "content": user_content}]


def model_input_device(model: torch.nn.Module) -> torch.device:
    """device_map='auto' 时 model.device 常为 CPU，与权重真实设备不一致；以首层参数为准。"""
    return next(model.parameters()).device


def load_model_and_processor(
    model_id: str,
    dtype: torch.dtype,
    attn_impl: str,
    device: Optional[str],
    load_in_4bit: bool = False,
) -> Tuple[Any, Any, Any]:
    try:
        from transformers import AutoProcessor
    except Exception as e:
        raise ImportError("Failed to import AutoProcessor. Please upgrade transformers.") from e

    model_cls = None
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_cls = Qwen2_5_VLForConditionalGeneration
    except Exception:
        try:
            from transformers import AutoModelForVision2Seq

            model_cls = AutoModelForVision2Seq
        except Exception as e:
            raise ImportError(
                "Cannot import Qwen2_5_VLForConditionalGeneration. "
                "Please install transformers>=4.52.1 and qwen-vl-utils."
            ) from e

    try:
        from qwen_vl_utils import process_vision_info
    except Exception as e:
        raise ImportError("Missing qwen-vl-utils. Please run: pip install -U qwen-vl-utils") from e

    load_kw: Dict[str, Any] = {"attn_implementation": attn_impl}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise ImportError("4-bit 加载需要 transformers BitsAndBytesConfig，请升级 transformers。") from e
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
        )
        if device:
            model = model_cls.from_pretrained(model_id, device_map={"": device}, **load_kw)
        else:
            model = model_cls.from_pretrained(model_id, device_map="auto", **load_kw)
    elif device:
        model = model_cls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map={"": device},
            **load_kw,
        )
    else:
        model = model_cls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            **load_kw,
        )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, process_vision_info


def generate_once(
    model: Any,
    processor: Any,
    process_vision_info: Any,
    chat_messages: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    text = processor.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(chat_messages)
    dev = model_input_device(model)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(dev)

    do_sample = temperature > 0
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **gen_kwargs)
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return output_text.strip()


def run(args: argparse.Namespace) -> None:
    if os.path.exists(args.output_json) and not args.overwrite:
        raise FileExistsError(f"{args.output_json} already exists. Use --overwrite.")

    with open(args.data_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("Input JSON must be a list.")
    if args.start_index < 0 or args.start_index >= len(dataset):
        raise ValueError(f"--start-index out of range: {args.start_index}, dataset size={len(dataset)}")

    end_index = len(dataset)
    if args.max_samples is not None:
        end_index = min(end_index, args.start_index + max(0, args.max_samples))
    dataset_slice = dataset[args.start_index:end_index]
    print(f"len(dataset)={len(dataset_slice)}, processing samples from index {args.start_index} to {end_index - 1}")

    dtype = get_torch_dtype(args.dtype)
    print(f"Loading model: {args.model_id}")
    model, processor, process_vision_info = load_model_and_processor(
        model_id=args.model_id,
        dtype=dtype,
        attn_impl=args.attn_impl,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
    )
    dev = model_input_device(model)
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    print(f"model.device (HF属性，auto 时可能不准)={getattr(model, 'device', None)}")
    print(f"实际权重/输入应对齐的设备 (首参)={dev}")
    if dev.type == "cpu" and torch.cuda.is_available():
        print(
            "[WARN] 权重在 CPU 上。若本意用 GPU，请检查 CUDA_VISIBLE_DEVICES，"
            "或显式传入 --device cuda:0"
        )

    outputs: List[Dict[str, Any]] = []
    for idx, sample in enumerate(tqdm(dataset_slice, desc="Infer-2D")):
        sample_out = deepcopy(sample)
        try:
            image_paths = get_image_paths(sample)
            original_messages = sample.get("messages", [])
            if not isinstance(original_messages, list):
                raise ValueError("messages must be a list.")

            pred_messages: List[Dict[str, str]] = []
            history: List[Dict[str, Any]] = []
            for msg in original_messages:
                if msg.get("role") != "user":
                    continue
                prompt = clean_user_text(msg.get("content", ""))
                chat_messages = build_messages(history, image_paths, prompt)
                pred = generate_once(
                    model=model,
                    processor=processor,
                    process_vision_info=process_vision_info,
                    chat_messages=chat_messages,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                pred_messages.append({"role": "user", "content": msg.get("content", "")})
                pred_messages.append({"role": "assistant", "content": pred})

                history.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
                history.append({"role": "assistant", "content": [{"type": "text", "text": pred}]})

            if not pred_messages:
                raise ValueError("No user turn found in messages.")
            sample_out["messages"] = pred_messages
            sample_out["pred_error"] = None
        except Exception as e:
            sample_out["pred_error"] = str(e)
        outputs.append(sample_out)

        if args.save_every > 0 and (idx + 1) % args.save_every == 0:
            dump_json(outputs, args.output_json)

    dump_json(outputs, args.output_json)
    print(f"Done. Saved {len(outputs)} samples to: {args.output_json}")


if __name__ == "__main__":
    run(parse_args())
