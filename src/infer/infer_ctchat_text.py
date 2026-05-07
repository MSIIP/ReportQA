import argparse
import json

import torch
import tqdm

from infer_ctchat_image import (
    conv_templates,
    disable_torch_init,
    get_model_name_from_path,
    infer_runtime_device,
    load_pretrained_model,
)


def infer_conv_mode(model_name: str) -> str:
    name = model_name.lower()
    if "llama-2" in name:
        return "llava_llama_2"
    if "mistral" in name:
        return "mistral_instruct"
    if "v1.6-34b" in name:
        return "chatml_direct"
    if "v1" in name:
        return "llava_v1"
    if "mpt" in name:
        return "mpt"
    return "llama3"


def normalize_user_prompt_text_only(text: str, ensure_provided: bool) -> str:
    """Text-only mode: keep raw text, optionally inject <provided> for llama3 system rules."""
    prompt = text.strip()
    if ensure_provided and "<provided>" not in prompt:
        prompt = "<provided>\n" + prompt
    return prompt


def build_record_like_input(element: dict, pred_messages: list) -> dict:
    out = {}
    for key in ("id", "messages"):
        if key == "messages":
            out["messages"] = pred_messages
        elif key in element:
            out[key] = element[key]
    for key, val in element.items():
        if key not in out:
            out[key] = val
    return out


def decode_new_tokens(tokenizer, output_ids: torch.Tensor, input_len: int) -> str:
    new_token_ids = output_ids[0, input_len:]
    if new_token_ids.numel() > 0:
        new_token_ids = new_token_ids[new_token_ids >= 0]
    if new_token_ids.numel() == 0:
        return ""
    return tokenizer.decode(new_token_ids.tolist(), skip_special_tokens=True).strip()


def main(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    print(f"Detected model name: {model_name}")
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )

    runtime_device = infer_runtime_device(model)
    for module in model.modules():
        if hasattr(module, "inv_freq") and isinstance(module.inv_freq, torch.Tensor):
            if module.inv_freq.device != runtime_device:
                module.inv_freq = module.inv_freq.to(runtime_device)

    with open(args.data_json, "r", encoding="utf-8") as file:
        data_val = json.load(file)

    samples = data_val if args.max_samples <= 0 else data_val[: args.max_samples]
    unit_list = []
    conv_mode = infer_conv_mode(model_name)
    print(f"Auto conv_mode: {conv_mode}")
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] auto conv_mode is {}, while --conv-mode is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
        conv_mode = args.conv_mode

    for element in tqdm.tqdm(samples):
        messages = element.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Sample does not contain valid messages list: keys={list(element.keys())}")

        conv = conv_templates[conv_mode].copy()
        pred_messages = []

        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue

            inp = normalize_user_prompt_text_only(
                msg.get("content", ""),
                ensure_provided=not args.disable_provided_token,
            )
            ref = None
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                ref = messages[i + 1].get("content", "")

            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()
            enc = tokenizer(prompt, return_tensors="pt")
            input_ids = enc.input_ids.to(runtime_device)
            attention_mask = enc.attention_mask.to(input_ids.device) if "attention_mask" in enc else None

            do_sample = args.temperature > 0
            generate_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                do_sample=do_sample,
                attention_mask=attention_mask,
            )
            if do_sample:
                generate_kwargs["temperature"] = args.temperature
                generate_kwargs["top_p"] = args.top_p
            else:
                generate_kwargs["temperature"] = 1.0
                generate_kwargs["top_p"] = 1.0

            with torch.inference_mode():
                output_ids = model.generate(inputs=input_ids, **generate_kwargs)

            outputs = decode_new_tokens(tokenizer, output_ids, input_ids.shape[1])
            conv.messages[-1][-1] = outputs

            pred_messages.append({"role": "user", "content": msg.get("content", "")})
            pred_messages.append({"role": "assistant", "content": outputs})

            if args.debug:
                print("\n", {"prompt": prompt, "reference": ref, "outputs": outputs}, "\n")

        unit_list.append(build_record_like_input(element, pred_messages))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(unit_list, f, ensure_ascii=False, indent=4)
    print(f"Saved predictions to: {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=20, help="<=0 means use all samples")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--data-json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--disable-provided-token",
        action="store_true",
        help="Do not inject <provided> automatically.",
    )
    args = parser.parse_args()
    main(args)
