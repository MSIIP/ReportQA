import copy
import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaForCausalLM, LlamaTokenizer


def unwrap_singleton(value):
    """Unwrap DataLoader batch_size=1 containers."""
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def replace_assistant_content(item, output_text):
    """Keep original structure and replace assistant message content."""
    result = copy.deepcopy(item) if isinstance(item, dict) else {}
    messages = result.get("messages")

    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "assistant":
                message["content"] = output_text
                return result
        messages.append({"role": "assistant", "content": output_text})
        return result

    return {"messages": [{"role": "assistant", "content": output_text}]}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(20)


# ---------------------------------------------------------------------------
# Inlined text_only_multi_dataset from Dataset.multi_dataset_test
# ---------------------------------------------------------------------------


def build_tokenizer_with_image_tokens(text_tokenizer, max_img_size, image_num):
    """Build tokenizer and RadFM image placeholder tokens."""
    image_padding_tokens = []
    if isinstance(text_tokenizer, str):
        text_tokenizer = LlamaTokenizer.from_pretrained(text_tokenizer)
        special_token = {"additional_special_tokens": ["<image>", "</image>"]}

        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image" + str(i * image_num + j) + ">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append(image_token)
            image_padding_tokens.append(image_padding_token)

        text_tokenizer.add_special_tokens(special_token)
        text_tokenizer.pad_token_id = 0
        text_tokenizer.bos_token_id = 1
        text_tokenizer.eos_token_id = 2

    return text_tokenizer, image_padding_tokens


def stack_images(images):
    """Text-only RadFM still passes an all-zero visual placeholder."""
    target_h = 512
    target_w = 512
    target_d = 4
    if len(images) == 0:
        return torch.zeros((1, 3, target_h, target_w, target_d))
    raise ValueError("This standalone script is text-only and expects no images.")


class text_only_multi_dataset(Dataset):
    """
    Text-only report-generation dataset.

    Expected format:
    [
      {
        "id": "...",
        "messages": [
          {"role": "user", "content": "..."},
          {"role": "assistant", "content": "..."}
        ]
      }
    ]
    """

    def __init__(self, text_tokenizer, dataset_json_path, max_img_size=10, image_num=32):
        if not dataset_json_path:
            raise ValueError("text_only_multi_dataset requires dataset_json_path.")
        if not os.path.isfile(dataset_json_path):
            raise FileNotFoundError(f"Dataset JSON not found: {dataset_json_path}")

        self.text_tokenizer, self.image_padding_tokens = build_tokenizer_with_image_tokens(
            text_tokenizer,
            max_img_size,
            image_num,
        )
        self.dataset_json_path = dataset_json_path
        with open(dataset_json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _extract_messages(item):
        question = ""
        answer = ""
        for msg in item.get("messages", []):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = str(msg.get("content", ""))
            if role == "user":
                question = content
            elif role == "assistant":
                answer = content
        return question, answer

    def __getitem__(self, idx):
        item = self.samples[idx]
        question, answer = self._extract_messages(item)
        sample_id = item.get("id", str(idx))

        return {
            "vision_x": stack_images([]),
            "question": question,
            "answer": answer,
            "belong_to": "text_only_messages",
            "id": sample_id,
            "raw_item_json": json.dumps(item, ensure_ascii=False),
        }


# ---------------------------------------------------------------------------
# Text-only RadFM model path
# ---------------------------------------------------------------------------


class TextOnlyRadFMEmbedding(nn.Module):
    """
    Minimal RadFM embedding layer for text-only inference.

    The original MyEmbedding owns many visual modules, but text-only inference
    always passes an all-zero visual placeholder and uses only text embeddings
    plus figure token weights.
    """

    def __init__(self, num_embeddings=32000, embedding_dim=5120):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((num_embeddings, embedding_dim)))
        self.figure_token_weight = nn.Parameter(torch.randn((2, embedding_dim)))
        self.flag = "Text"

    def forward(self, text_input, vision_x=None, key_words_query=None):
        batch_size = text_input.shape[0]
        embedding_weight = torch.cat([self.weight, self.figure_token_weight], dim=0)
        embedding_weight = embedding_weight.unsqueeze(0).repeat(batch_size, 1, 1)
        text_input = F.one_hot(text_input, embedding_weight.shape[1]).to(embedding_weight.dtype).to(
            embedding_weight.device
        )
        output = torch.matmul(text_input, embedding_weight)
        return output, None


class MultiLLaMAForCausalLM(nn.Module):
    """Compact RadFM wrapper for pure text inference."""

    def __init__(self, lang_model_path):
        super().__init__()
        self.lang_model = LlamaForCausalLM.from_pretrained(lang_model_path)
        self.lang_model.gradient_checkpointing_enable()
        self.lang_model.enable_input_require_grads()
        self.embedding_layer = TextOnlyRadFMEmbedding()
        self.embedding_layer.weight = self.lang_model.get_input_embeddings().weight
        self.hidden_dim = 5120
        self.voc_size = 32000

    def generate(self, lang_x, vision_x=None, max_new_tokens=200):
        self.embedding_layer.flag = "Text"
        with torch.no_grad():
            input_embedding, _ = self.embedding_layer(lang_x, vision_x)
            generation = self.lang_model.generate(
                inputs_embeds=input_embedding,
                max_new_tokens=max_new_tokens,
                top_k=50,
            )
        return generation


def load_radfm_text_only_checkpoint(model, ckpt_path):
    """
    Load the shared language-model and text embedding weights from RadFM.

    Visual-only keys are intentionally ignored because this script never runs
    the visual encoder branch.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    load_info = model.load_state_dict(ckpt, strict=False)

    missing = getattr(load_info, "missing_keys", [])
    unexpected = getattr(load_info, "unexpected_keys", [])
    if missing:
        print(f"[WARNING] Missing keys while loading text-only RadFM: {len(missing)}")
    if unexpected:
        print(f"[INFO] Ignored visual/non-text keys from RadFM checkpoint: {len(unexpected)}")
    return model


# ---------------------------------------------------------------------------
# Original infer_RadFm_text.py entrypoint
# ---------------------------------------------------------------------------


@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(
        default=None,
        metadata={"help": "Directory with base LLaMA weights (e.g. MedLLaMA HF folder)."},
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Tokenizer directory. Defaults to lang_encoder_path when omitted.",
        },
    )
    radfm_ckpt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to RadFM pytorch_model.bin checkpoint."},
    )


@dataclass
class DataArguments:
    dataset_json_path: str = field(
        default="",
        metadata={"help": "Path to text-only report JSON."},
    )
    output_json: Optional[str] = field(
        default=None,
        metadata={"help": "Prediction output path."},
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum new tokens for generation."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Run only first N samples when set."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    batch_size_2D: int = field(default=1)
    batch_size_3D: int = field(default=1)
    output_dir: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not data_args.dataset_json_path:
        raise ValueError("--dataset_json_path is required for text-only inference.")
    if not model_args.lang_encoder_path:
        raise ValueError("--lang_encoder_path is required (base LLaMA / MedLLaMA directory).")
    if not model_args.radfm_ckpt_path:
        raise ValueError("--radfm_ckpt_path is required (RadFM pytorch_model.bin).")
    tokenizer_source = model_args.tokenizer_path or model_args.lang_encoder_path

    print("Setup Text-Only Data")
    test_dataset = text_only_multi_dataset(
        text_tokenizer=tokenizer_source,
        dataset_json_path=data_args.dataset_json_path,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=None,
        drop_last=False,
    )

    print("Setup Model")
    model = MultiLLaMAForCausalLM(lang_model_path=model_args.lang_encoder_path)
    model = load_radfm_text_only_checkpoint(model, model_args.radfm_ckpt_path)
    model = model.to("cuda")
    model.eval()
    print("-------------Model loaded successfully---------------")

    if data_args.output_json:
        out_json_path = data_args.output_json
    else:
        dataset_json_path = os.path.abspath(data_args.dataset_json_path)
        stem = os.path.splitext(os.path.basename(dataset_json_path))[0]
        out_json_path = os.path.join(os.path.dirname(dataset_json_path), stem + "_pred_text_only.json")

    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
    print(f"Writing predictions to: {out_json_path}")

    results_list = []
    for sample_idx, sample in enumerate(tqdm.tqdm(test_dataloader)):
        if data_args.max_samples is not None and sample_idx >= data_args.max_samples:
            break

        question = unwrap_singleton(sample["question"])
        raw_item_json = unwrap_singleton(sample.get("raw_item_json"))
        raw_item = None
        if isinstance(raw_item_json, str):
            try:
                raw_item = json.loads(raw_item_json)
            except json.JSONDecodeError:
                raw_item = None

        lang_x = test_dataset.text_tokenizer(question, return_tensors="pt")["input_ids"].to("cuda")
        vision_x = sample["vision_x"].to("cuda")

        try:
            generation = model.generate(
                lang_x,
                vision_x,
                max_new_tokens=data_args.max_new_tokens,
            )
            generated_texts = test_dataset.text_tokenizer.batch_decode(generation, skip_special_tokens=True)
            pred_text = generated_texts[0] if generated_texts else ""
            result_item = replace_assistant_content(raw_item, pred_text)
            results_list.append(result_item)
        except Exception as exc:
            print(f"Skip one sample due to generation error: {exc}")
            continue

    with open(out_json_path, mode="w", encoding="utf-8") as outfile:
        json.dump(results_list, outfile, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
