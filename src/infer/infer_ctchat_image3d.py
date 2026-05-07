import argparse
import dataclasses
import importlib
import json
import math
import os
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import transformers
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many
from torch import einsum
from torch.autograd import grad as torch_grad
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from vector_quantize_pytorch import VectorQuantize


# ---------------------------------------------------------------------------
# Inlined llava.constants
# ---------------------------------------------------------------------------

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

TOKEN_FOR_MULTIPLE_CHOICE = "<multiple_choice>"
TOKEN_FOR_LONG_ANSWER = "<long_answer>"
TOKEN_FOR_SHORT_ANSWER = "<short_answe>"
TOKEN_FOR_REPORT_GENERATION = "<report_generation>"


# ---------------------------------------------------------------------------
# Inlined llava.conversation
# ---------------------------------------------------------------------------


class SeparatorStyle(Enum):
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"
    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if "mmtag" in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""
            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0:
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")
        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )


conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="",
    sep2="</s>",
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llama3 = Conversation(
    system="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n. You are CT-CHAT, an AI assistant specializing in Chest CT imaging, dedicated to providing accurate and relevant information exclusively related to Chest CT scans and associated medical topics. You are equipped to answer questions and offer detailed analyses only when the CT volume/scan/image is provided, indicated by the <provided> token. If this token is not present and users inquire about specific findings, pathologies, or request descriptions related to a Chest CT, respond by requesting the necessary data with the phrase: “Please provide the CT volume.” Once the <provided> token is present in the question, you are authorized to address questions about pathologies, anatomical or clinical findings, diagnostic descriptions, report generation, comparisons, or any other questions regarding the image. If it does not appear in the question, even when special tokens <multiple_choice>, <report_generation>, <long_answer>, and <short_answer> are given, ignore the question and ask for the CT volume. Always look for the <provided> token, even if there are special tokens. If there is a <provided> token in any question (including new and previous ones), never ask for the CT volume again and answer the question. You can ignore the <provided> token check and answer the question directly if and only if the question is about general medical knowledge, not about the provided CT volume (such as typical findings on a Chest CT or management of the patient). For example, “What are the typical imaging findings of acute respiratory distress syndrome (ARDS) on a chest CT?” is a general question not specific. If user asks a CT specific question after non-spesific question, look for the <provided> token as well even if the special tokens are given. It is crucial to avoid discussing topics outside of Chest CT imaging and directly related medical information, ensuring that all responses are clear, concise, and focused on the provided Chest CT data for the highest level of accuracy and relevance. If the user greets you with something like “hello,” respond appropriately.""",
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    version="llama3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|eot_id|>",
)

conv_templates = {
    "llava_llama_2": conv_llava_llama_2,
    "mistral_instruct": conv_mistral_instruct,
    "chatml_direct": conv_chatml_direct,
    "llava_v1": conv_llava_v1,
    "mpt": conv_mpt,
    "llama3": conv_llama3,
}


# ---------------------------------------------------------------------------
# Inlined llava.utils / llava.mm_utils / tokenizer resize
# ---------------------------------------------------------------------------


def disable_torch_init():
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(items, sep):
        return [ele for sublist in zip(items, [sep] * len(items)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    return model_paths[-1]


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    raise NotImplementedError("This standalone inference path does not use anyres image grids.")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


# ---------------------------------------------------------------------------
# Inlined llava multimodal projector and architecture
# ---------------------------------------------------------------------------


class AttentionalPoolProjector(nn.Module):
    def __init__(self, embed_dim, context_dim, projector=None, n_head=8, n_queries=256, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn_pool = AttentionalPooler(
            d_model=embed_dim,
            context_dim=context_dim,
            n_head=n_head,
            n_queries=n_queries,
        )
        self.ln = norm_layer(embed_dim)
        self.proj = projector if projector else nn.Identity()

    def forward(self, x: torch.Tensor):
        tokens = self.attn_pool(x)
        tokens = self.ln(tokens)
        return self.proj(tokens)


class AttentionalPooler(nn.Module):
    def __init__(self, d_model: int, context_dim: int, n_head: int = 8, n_queries: int = 256, norm_layer=nn.LayerNorm):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        dim_head = d_model // n_head
        self.scale = dim_head**-0.5
        self.heads = n_head
        inner_dim = dim_head * n_head
        self.ln_k = norm_layer(context_dim)
        self.ln_q = norm_layer(d_model)
        self.to_q = nn.Linear(d_model, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = rearrange(x, "b n d -> b 1 n d")
        q = repeat(self.query, "n d -> b m n d", b=x.shape[0], m=x.shape[1])
        x = self.ln_k(x)
        q = self.ln_q(q)
        b, m, h = *x.shape[:2], self.heads
        q = self.to_q(q)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out).squeeze(dim=1)


class IdentityMap(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type.startswith("attn_pool"):
        import re

        mlp_projector = projector_type.split("+")[1]
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", mlp_projector)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            projector = nn.Sequential(*modules)
        else:
            projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        return AttentionalPoolProjector(
            embed_dim=config.mm_hidden_size,
            context_dim=config.mm_context_size,
            projector=projector,
        )

    import re

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            self.mm_projector = build_vision_projector(config)
            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower


class LlavaMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        images = images.flatten(1, 3)
        return self.get_model().mm_projector(images)

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        image_sizes=None,
    ):
        if images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            image_features = self.encode_images(images)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            if mm_patch_merge_type != "flat":
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError

        _labels = labels
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            llm_dev = self.get_model().embed_tokens.weight.device
            cur_new_input_embeds = [x.to(llm_dev) for x in cur_new_input_embeds]
            new_input_embeds.append(torch.cat(cur_new_input_embeds))
            new_labels.append(torch.cat(cur_new_labels))

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        llm_dev = self.get_model().embed_tokens.weight.device
        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=llm_dev)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=llm_dev)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=llm_dev)
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=llm_dev)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        new_labels = None if _labels is None else new_labels_padded
        if _attention_mask is not None:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        emb_dev = new_input_embeds.device
        if position_ids is not None and position_ids.device != emb_dev:
            position_ids = position_ids.to(emb_dev)
        if attention_mask is not None and attention_mask.device != emb_dev:
            attention_mask = attention_mask.to(emb_dev)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


def _infer_runtime_device(module: nn.Module) -> torch.device:
    first_device = None
    for p in module.parameters():
        if first_device is None:
            first_device = p.device
        if p.device.type != "cpu":
            return p.device
    for b in module.buffers():
        if first_device is None:
            first_device = b.device
        if b.device.type != "cpu":
            return b.device
    return first_device if first_device is not None else torch.device("cpu")


def _past_kv_seq_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return past_key_values.get_seq_length()
        except Exception:
            pass
    try:
        return past_key_values[0][0].shape[2]
    except Exception:
        return 0


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )

        if inputs_embeds is not None and attention_mask is not None:
            need = inputs_embeds.shape[1]
            if attention_mask.shape[-1] != need:
                attention_mask = torch.ones(
                    (inputs_embeds.shape[0], need),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
        elif inputs_embeds is None and input_ids is not None and attention_mask is not None and past_key_values is not None:
            need = _past_kv_seq_length(past_key_values) + input_ids.shape[1]
            if attention_mask.shape[-1] != need:
                attention_mask = torch.ones(
                    (input_ids.shape[0], need),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

        if inputs_embeds is not None:
            ref = inputs_embeds.device
        elif input_ids is not None and input_ids.device.type != "cpu":
            ref = input_ids.device
        else:
            ref = _infer_runtime_device(self)
        if position_ids is not None and position_ids.device != ref:
            position_ids = position_ids.to(ref)
        if attention_mask is not None and attention_mask.device != ref:
            attention_mask = attention_mask.to(ref)
        if cache_position is not None and isinstance(cache_position, torch.Tensor) and cache_position.device != ref:
            cache_position = cache_position.to(ref)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if inputs is not None and inputs.device.type != "cpu":
            ref = inputs.device
        else:
            ref = _infer_runtime_device(self)
        if position_ids is not None and position_ids.device != ref:
            position_ids = position_ids.to(ref)
        if attention_mask is not None and attention_mask.device != ref:
            attention_mask = attention_mask.to(ref)

        return super().generate(
            inputs=inputs,
            images=images,
            image_sizes=image_sizes,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None and past_key_values is None:
            inputs["images"] = images
        if image_sizes is not None and past_key_values is None:
            inputs["image_sizes"] = image_sizes

        if past_key_values is not None:
            pkv_len = _past_kv_seq_length(past_key_values)
            ids = inputs.get("input_ids")
            if isinstance(ids, torch.Tensor) and ids.shape[1] > 1:
                inputs["input_ids"] = ids[:, -1:]

            need = pkv_len + 1
            mask = inputs.get("attention_mask")
            if isinstance(mask, torch.Tensor) and mask.dim() == 2 and mask.shape[-1] != need:
                inputs["attention_mask"] = torch.ones((mask.shape[0], need), dtype=mask.dtype, device=mask.device)

            pos = inputs.get("position_ids")
            if isinstance(pos, torch.Tensor):
                if pos.shape[-1] > 1:
                    inputs["position_ids"] = pos[:, -1:]
                elif pos.shape[-1] == 1 and pos.numel() == 1 and pos.item() != pkv_len:
                    inputs["position_ids"] = torch.tensor([[pkv_len]], dtype=pos.dtype, device=pos.device)

            cache_pos = inputs.get("cache_position")
            if isinstance(cache_pos, torch.Tensor):
                if cache_pos.shape[-1] > 1:
                    inputs["cache_position"] = cache_pos[-1:].to(cache_pos.device)
                elif cache_pos.shape[-1] == 1 and cache_pos.numel() == 1 and cache_pos.item() != pkv_len:
                    inputs["cache_position"] = torch.tensor([pkv_len], dtype=cache_pos.dtype, device=cache_pos.device)

        dev = _infer_runtime_device(self)
        for key in ("input_ids", "position_ids", "attention_mask"):
            tensor = inputs.get(key)
            if isinstance(tensor, torch.Tensor) and tensor.device != dev:
                inputs[key] = tensor.to(dev)
        return inputs


AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)


# ---------------------------------------------------------------------------
# Inlined llava.model.builder
# ---------------------------------------------------------------------------


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def should_merge_lora(load_8bit: bool, load_4bit: bool) -> bool:
    env = os.getenv("CTCHAT_FORCE_MERGE_LORA")
    if env is not None:
        return str(env).strip().lower() in {"1", "true", "yes", "y", "on"}
    return not (load_8bit or load_4bit)


@contextmanager
def patch_init_weights_skip_non_float(model_cls):
    original_init = getattr(model_cls, "_init_weights", None)
    if original_init is None:
        yield
        return

    def _safe_init(self, module):
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and not torch.is_floating_point(weight):
            return
        return original_init(self, module)

    model_cls._init_weights = _safe_init
    try:
        yield
    finally:
        model_cls._init_weights = original_init


@contextmanager
def patch_dispatch_model_for_quantized_to_error():
    try:
        modeling_utils = importlib.import_module("transformers.modeling_utils")
    except Exception:
        yield
        return

    original_dispatch = getattr(modeling_utils, "dispatch_model", None)
    if original_dispatch is None:
        yield
        return

    big_modeling = None
    original_big_dispatch = None
    try:
        big_modeling = importlib.import_module("accelerate.big_modeling")
        original_big_dispatch = getattr(big_modeling, "dispatch_model", None)
    except Exception:
        big_modeling = None

    def _safe_dispatch_model(model, *args, **kwargs):
        try:
            return original_dispatch(model, *args, **kwargs)
        except ValueError as exc:
            if ".to` is not supported for `4-bit` or `8-bit` bitsandbytes models" in str(exc):
                warnings.warn(
                    "Caught bitsandbytes quantized `.to()` dispatch conflict; skip dispatch_model and continue with model as loaded.",
                    UserWarning,
                )
                return model
            raise

    modeling_utils.dispatch_model = _safe_dispatch_model
    if big_modeling is not None and original_big_dispatch is not None:
        big_modeling.dispatch_model = _safe_dispatch_model

    try:
        yield
    finally:
        modeling_utils.dispatch_model = original_dispatch
        if big_modeling is not None and original_big_dispatch is not None:
            big_modeling.dispatch_model = original_big_dispatch


def load_llava_config_compat(model_path, llava_config_cls):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return llava_config_cls.from_pretrained(model_path)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    rope_scaling = cfg_dict.get("rope_scaling")
    if isinstance(rope_scaling, dict):
        rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
        factor = rope_scaling.get("factor")
        llama3_required = ("low_freq_factor", "high_freq_factor", "original_max_position_embeddings")
        has_llama3_full = rope_type == "llama3" and all(k in rope_scaling for k in llama3_required)
        if not has_llama3_full and rope_type is not None and factor is not None and (
            "rope_type" in rope_scaling or len(rope_scaling.keys()) != 2
        ):
            cfg_dict["rope_scaling"] = {"type": rope_type, "factor": factor}
            warnings.warn(
                f"Detected extended rope_scaling schema; converted to {{'type': '{rope_type}', 'factor': {factor}}} for compatibility.",
                UserWarning,
            )

    try:
        return llava_config_cls.from_dict(cfg_dict)
    except (ValueError, KeyError) as e:
        rope_scaling = cfg_dict.get("rope_scaling")
        if isinstance(rope_scaling, dict):
            rope_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
            factor = rope_scaling.get("factor")
            if rope_type == "llama3" and factor is not None:
                cfg_dict["rope_scaling"] = {"type": "dynamic", "factor": factor}
                warnings.warn(
                    f"Current transformers does not support rope_scaling type 'llama3'; fallback to {{'type': 'dynamic', 'factor': {factor}}}.",
                    UserWarning,
                )
                return llava_config_cls.from_dict(cfg_dict)
        raise e


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda" and not (load_8bit or load_4bit):
        kwargs["device_map"] = {"": device}

    skip_quant_modules = ["mm_projector", "lm_head"]
    if load_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip_quant_modules)
    elif load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_skip_modules=skip_quant_modules,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if (load_8bit or load_4bit) and kwargs.get("device_map") == "auto":
        kwargs.pop("device_map", None)
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    if "llava" not in model_name.lower():
        raise ValueError(f"Standalone CT-CHAT inference only supports LLaVA checkpoints, got {model_name}")

    if "lora" in model_name.lower() and model_base is None:
        warnings.warn("There is `lora` in model name but no `model_base` is provided.")

    if "lora" in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = load_llava_config_compat(model_path, LlavaConfig)
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)

        print("Loading LLaVA from base model...")
        lora_cfg_pretrained.pad_token_id = None
        lora_cfg_pretrained.vocab_size = lora_cfg_pretrained.vocab_size - 1 - 4
        load_kwargs = dict(kwargs)
        if not load_8bit and not load_4bit and load_kwargs.get("device_map") == "auto":
            load_kwargs["device_map"] = {"": device}
        if load_4bit or load_8bit:
            with patch_dispatch_model_for_quantized_to_error(), patch_init_weights_skip_non_float(LlavaLlamaForCausalLM):
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_base,
                    low_cpu_mem_usage=True,
                    config=lora_cfg_pretrained,
                    **load_kwargs,
                )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                config=lora_cfg_pretrained,
                **load_kwargs,
            )

        print("Adding pad token as '<pad>'")
        smart_tokenizer_and_embedding_resize(dict(pad_token="<pad>"), tokenizer=tokenizer, model=model)
        tokenizer.add_tokens(TOKEN_FOR_MULTIPLE_CHOICE, special_tokens=True)
        tokenizer.add_tokens(TOKEN_FOR_LONG_ANSWER, special_tokens=True)
        tokenizer.add_tokens(TOKEN_FOR_SHORT_ANSWER, special_tokens=True)
        tokenizer.add_tokens(TOKEN_FOR_REPORT_GENERATION, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype))

        print("Loading additional LLaVA weights...")
        non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
        if os.path.exists(non_lora_path):
            non_lora_trainables = safe_torch_load(non_lora_path, map_location="cpu")
            print("--------------------non_lora_trainables.bin loaded from local file system-----------------------:")
            for name, param in non_lora_trainables.items():
                print(f"  {name}: {param.shape}")
        else:
            from huggingface_hub import hf_hub_download

            cache_file = hf_hub_download(repo_id=model_path, filename="non_lora_trainables.bin")
            non_lora_trainables = safe_torch_load(cache_file, map_location="cpu")

        non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}

        named_params = dict(model.named_parameters())
        for k, v in non_lora_trainables.items():
            param = named_params.get(k)
            if param is not None:
                non_lora_trainables[k] = v.to(device=param.device, dtype=param.dtype)
            else:
                non_lora_trainables[k] = v.to(device=model.device, dtype=model.dtype)

        try:
            model.load_state_dict(non_lora_trainables, strict=False, assign=True)
        except TypeError:
            model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel

        print("Loading LoRA weights...")
        model = PeftModel.from_pretrained(model, model_path)
        if should_merge_lora(load_8bit=load_8bit, load_4bit=load_4bit):
            print("Merging LoRA weights...")
            model = model.merge_and_unload()
        else:
            print("Keeping LoRA unmerged (PeftModel active). Set CTCHAT_FORCE_MERGE_LORA=1 to force merge.")
        print("Model is loaded...")
    elif model_base is not None:
        print("Loading LLaVA from base model...")
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
        mm_projector_weights = torch.load(os.path.join(model_path, "mm_projector.bin"), map_location="cpu")
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if load_4bit or load_8bit:
            with patch_dispatch_model_for_quantized_to_error(), patch_init_weights_skip_non_float(LlavaLlamaForCausalLM):
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    context_len = model.config.max_sequence_length if hasattr(model.config, "max_sequence_length") else 2048
    return tokenizer, model, None, context_len


# ---------------------------------------------------------------------------
# Inlined transformer_maskgit attention + CTViT encoder path
# ---------------------------------------------------------------------------


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def l2norm(t):
    return F.normalize(t, dim=-1)


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, dropout=0.0):
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PEG(nn.Module):
    def __init__(self, dim, causal=False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    def forward(self, x, shape: Tuple[int, int, int, int] = None):
        needs_shape = x.ndim == 3
        assert not (needs_shape and not exists(shape))
        orig_shape = x.shape
        if needs_shape:
            x = x.reshape(*shape, -1)
        x = rearrange(x, "b ... d -> b d ...")
        frame_padding = (2, 0) if self.causal else (1, 1)
        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value=0.0)
        x = self.dsconv(x)
        x = rearrange(x, "b d ... -> b ... d")
        if needs_shape:
            x = rearrange(x, "b ... d -> b (...) d")
        return x.reshape(orig_shape)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context=None,
        dim_head=64,
        heads=8,
        causal=False,
        num_null_kv=0,
        norm_context=True,
        dropout=0.0,
        scale=8,
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)
        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads=heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()
        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None, context=None, attn_bias=None):
        batch, device = x.shape[0], x.device
        if exists(context):
            context = self.context_norm(context)
        kv_input = default(context, x)
        x = self.norm(x)
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v))
        nk, nv = repeat(self.null_kv, "h (n r) d -> b h n r d", b=batch, r=2).unbind(dim=-2)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        i, j = sim.shape[-2:]
        if exists(attn_bias):
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.0)
            sim = sim + attn_bias
        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
        if self.causal:
            sim = sim + self.rel_pos_bias(sim)
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, "h -> h 1 1")
        self.register_buffer("slopes", slopes, persistent=False)
        self.register_buffer("bias", None, persistent=False)

    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device=device)
        j_arange = torch.arange(j, device=device)
        return -torch.abs(rearrange(j_arange, "j -> 1 1 j") - rearrange(i_arange, "i -> 1 i 1"))

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)
        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: heads - closest_power_of_2]

    def forward(self, sim):
        h, i, j, device = *sim.shape[-3:], sim.device
        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]
        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes.to(device)
        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer("bias", bias, persistent=False)
        return self.bias


class ContinuousPositionBias(nn.Module):
    def __init__(self, *, dim, heads, num_dims=2, layers=2, log_dist=True, cache_rel_pos=False):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist
        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), leaky_relu()))
        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))
        self.net.append(nn.Linear(dim, heads))
        self.cache_rel_pos = cache_rel_pos
        self.register_buffer("rel_pos", None, persistent=False)

    def forward(self, *dimensions, device=torch.device("cpu")):
        if not exists(self.rel_pos) or not self.cache_rel_pos:
            positions = [torch.arange(d, device=device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing="ij"))
            grid = rearrange(grid, "c ... -> (...) c")
            rel_pos = rearrange(grid, "i c -> i 1 c") - rearrange(grid, "j c -> 1 j c")
            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)
            self.register_buffer("rel_pos", rel_pos, persistent=False)
        rel_pos = self.rel_pos.to(torch.float32)
        for layer in self.net:
            rel_pos = layer(rel_pos.float())
        return rearrange(rel_pos, "i j h -> h i j")


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        ff_mult=4,
        peg=False,
        peg_causal=False,
        attn_num_null_kv=2,
        has_cross_attn=False,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PEG(dim=dim, causal=peg_causal) if peg else None,
                        Attention(dim=dim, dim_head=dim_head, heads=heads, causal=causal, dropout=attn_dropout),
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            dim_context=dim_context,
                            heads=heads,
                            causal=False,
                            num_null_kv=attn_num_null_kv,
                            dropout=attn_dropout,
                        )
                        if has_cross_attn
                        else None,
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )
        self.norm_out = LayerNorm(dim)

    def forward(self, x, video_shape: Tuple[int, int, int, int] = None, attn_bias=None, context=None, self_attn_mask=None, cross_attn_context_mask=None):
        for peg, self_attn, cross_attn, ff in self.layers:
            if exists(peg):
                x = peg(x, shape=video_shape) + x
            x = self_attn(x, attn_bias=attn_bias, mask=self_attn_mask) + x
            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context=context, mask=cross_attn_context_mask) + x
            x = ff(x) + x
        return self.norm_out(x)


def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, "vgg")
        if has_vgg:
            vgg = self.vgg
            delattr(self, "vgg")
        out = fn(self, *args, **kwargs)
        if has_vgg:
            self.vgg = vgg
        return out

    return inner


def gradient_penalty(images, output, weight=10):
    gradients = torch_grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = rearrange(gradients, "b ... -> b (...)")
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def safe_div(numer, denom, eps=1e-8):
    return numer / (denom + eps)


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


def grad_layer_wrt_loss(loss, layer):
    return torch_grad(outputs=loss, inputs=layer, grad_outputs=torch.ones_like(loss), retain_graph=True)[0].detach()


def pick_video_frame(video, frame_indices):
    batch = video.shape[0]
    video = rearrange(video, "b c f ... -> b f c ...")
    batch_indices = torch.arange(batch, device=video.device)
    batch_indices = rearrange(batch_indices, "b -> b 1")
    images = video[batch_indices, frame_indices]
    return rearrange(images, "b 1 c ... -> b c ...")


class CTViT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        codebook_size,
        image_size,
        patch_size,
        temporal_patch_size,
        spatial_depth,
        temporal_depth,
        discr_base_dim=16,
        dim_head=64,
        heads=8,
        channels=1,
        use_vgg_and_gan=True,
        vgg=None,
        discr_attn_res_layers=(16,),
        use_hinge_loss=True,
        attn_dropout=0.0,
        ff_dropout=0.0,
    ):
        super().__init__()
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)
        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange("b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(channels * patch_width * patch_height, dim),
            nn.LayerNorm(dim),
        )
        self.to_patch_emb = nn.Sequential(
            Rearrange("b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)", p1=patch_height, p2=patch_width, pt=temporal_patch_size),
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim),
        )

        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,
            peg_causal=True,
        )
        self.enc_spatial_transformer = Transformer(depth=spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **transformer_kwargs)
        from vector_quantize_pytorch import VectorQuantize

        self.vq = VectorQuantize(dim=dim, codebook_size=codebook_size, use_cosine_sim=True)
        self.to_pixels_first_frame = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height),
            Rearrange("b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)", p1=patch_height, p2=patch_width),
        )
        self.to_pixels = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height * temporal_patch_size),
            Rearrange("b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)", p1=patch_height, p2=patch_width, pt=temporal_patch_size),
        )
        self.gen_loss = hinge_gen_loss if use_hinge_loss else None
        self.use_vgg_and_gan = use_vgg_and_gan

    @property
    def image_num_tokens(self):
        return int(self.image_size[0] / self.patch_size[0]) * int(self.image_size[1] / self.patch_size[1])

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def calculate_video_token_mask(self, videos, video_frame_mask):
        *_, h, w = videos.shape
        ph, pw = self.patch_size
        first_frame_mask, rest_frame_mask = video_frame_mask[:, :1], video_frame_mask[:, 1:]
        rest_vq_mask = rearrange(rest_frame_mask, "b (f p) -> b f p", p=self.temporal_patch_size)
        video_mask = torch.cat((first_frame_mask, rest_vq_mask.any(dim=-1)), dim=-1)
        return repeat(video_mask, "b f -> b (f hw)", hw=(h // ph) * (w // pw))

    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        self.load_state_dict(torch.load(str(path)))

    def encode(self, tokens):
        b = tokens.shape[0]
        h, w = self.patch_height_width
        video_shape = tuple(tokens.shape[:-1])
        tokens = rearrange(tokens, "b t h w d -> (b t) (h w) d")
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        tokens = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)
        tokens = rearrange(tokens, "(b t) (h w) d -> b t h w d", b=b, h=h, w=w)
        tokens = rearrange(tokens, "b t h w d -> (b h w) t d")
        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)
        tokens = rearrange(tokens, "(b h w) t d -> b t h w d", b=b, h=h, w=w)
        return tokens

    def forward(
        self,
        video,
        mask=None,
        return_recons=False,
        return_recons_only=False,
        return_discr_loss=False,
        apply_grad_penalty=True,
        return_only_codebook_ids=False,
        return_encoded_tokens=False,
    ):
        assert video.ndim in {4, 5}
        is_image = video.ndim == 4
        if is_image:
            video = rearrange(video, "b c h w -> b c 1 h w")
            assert not exists(mask)
        b, c, f, *image_dims = video.shape
        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f
        tokens = self.to_patch_emb(video)
        *_, h, w, _ = tokens.shape
        tokens = self.encode(tokens)
        tokens, packed_fhw_shape = pack([tokens], "b * d")
        vq_mask = self.calculate_video_token_mask(video, mask) if exists(mask) else None
        tokens, indices, commit_loss = self.vq(tokens, mask=vq_mask)
        if return_only_codebook_ids:
            indices, = unpack(indices, packed_fhw_shape, "b *")
            return indices
        tokens = rearrange(tokens, "b (t h w) d -> b t h w d", h=h, w=w)
        if return_encoded_tokens:
            return tokens
        raise NotImplementedError("Standalone inference only uses CTViT(return_encoded_tokens=True).")


# ---------------------------------------------------------------------------
# Original ctchat_validation_llama.py inference logic
# ---------------------------------------------------------------------------


def module_param_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def infer_runtime_device(module: torch.nn.Module) -> torch.device:
    return _infer_runtime_device(module)


def align_image_tensor_for_mm_projector(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    inner = model.get_model() if hasattr(model, "get_model") else model
    if not hasattr(inner, "mm_projector"):
        inner = model.model
    proj = inner.mm_projector
    p = next(proj.parameters())
    return image_tensor.to(device=p.device, dtype=p.dtype)


def normalize_user_prompt(text: str) -> str:
    t = text.strip()
    if "<video>" in t:
        t = t.replace("<video>", "<image>")
    t = t.strip()
    if t.startswith("\n"):
        t = t.lstrip("\n").strip()

    if "<provided>" not in t:
        if t.startswith("<image>"):
            rest = t[len("<image>") :].lstrip("\n")
            t = "<image>\n<provided>\n" + rest if rest else "<image>\n<provided>"
        else:
            t = "<image>\n<provided>\n" + t
    elif "<image>" not in t:
        t = "<image>\n" + t
    return t


def build_amos_record_like_input(element: dict, pred_messages: list) -> dict:
    out = {}
    for key in ("id", "messages", "videos"):
        if key == "messages":
            out["messages"] = pred_messages
        elif key in element:
            out[key] = element[key]
    for key, val in element.items():
        if key not in out:
            out[key] = val
    return out


def detect_format(element: dict) -> str:
    if "videos" in element and "messages" in element:
        return "amos"
    if "image" in element and "conversations" in element:
        return "legacy"
    raise ValueError(
        "无法识别样本格式：需要 (videos+messages) 或 (image+conversations)，"
        f"当前 keys={list(element.keys())}"
    )


def npz_basename_from_element(element: dict, fmt: str) -> str:
    if fmt == "amos":
        if not element.get("videos"):
            raise ValueError("amos 格式需要非空 videos 列表")
        base = os.path.basename(element["videos"][0])
    else:
        base = os.path.basename(str(element["image"]))
    return base.replace("nii.gz", "npz")


def get_image_path_from_element(element: dict, fmt: str) -> str:
    if fmt == "amos":
        if not element.get("videos"):
            raise ValueError("amos 格式需要非空 videos 列表")
        return element["videos"][0]
    if element.get("image") is None:
        raise ValueError("legacy 格式需要 image 字段")
    return str(element["image"])


def resize_array(array, current_spacing, target_spacing):
    original_shape = array.shape[2:]
    print(f"Original shape: {original_shape}, current spacing: {current_spacing}, target spacing: {target_spacing}")
    scaling_factors = [current_spacing[i] / target_spacing[i] for i in range(len(original_shape))]
    new_shape = [int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))]
    resized_array = F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False).cpu().numpy()
    print(f"Resized shape: {resized_array.shape} with scaling factors: {scaling_factors}")
    return resized_array


def nii_img_to_tensor(path):
    import nibabel as nib

    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata()
    slope, intercept = nii_img.header.get_slope_inter()
    print(f"nii image data shape: {img_data.shape}, dtype: {img_data.dtype}, slope: {slope}, intercept: {intercept}")
    slope = 1.0 if slope is None else float(slope)
    intercept = 0.0 if intercept is None else float(intercept)

    zooms = nii_img.header.get_zooms()
    x_spacing = float(zooms[0]) if len(zooms) > 0 else 1.0
    y_spacing = float(zooms[1]) if len(zooms) > 1 else x_spacing
    z_spacing = float(zooms[2]) if len(zooms) > 2 else 1.0
    xy_spacing = (x_spacing + y_spacing) / 2.0
    print(f"original x_spacing: {x_spacing}, original y_spacing: {y_spacing}, original z_spacing: {z_spacing}")

    target = (1.5, 0.75, 0.75)
    current = (z_spacing, xy_spacing, xy_spacing)

    img_data = slope * img_data + intercept
    img_data = np.clip(img_data, -1000, 1000)
    img_data = img_data.transpose(2, 0, 1)

    tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)
    img_data = resize_array(tensor, current, target)[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))
    img_data = (img_data / 1000).astype(np.float32)
    print(f"img_data: {img_data.shape}")
    tensor = torch.tensor(img_data)
    target_shape = (480, 480, 240)
    h, w, d = tensor.shape
    dh, dw, dd = target_shape

    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before
    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before
    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(
        tensor,
        (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
        value=-1,
    )
    return tensor.permute(2, 0, 1).unsqueeze(0)


def build_image_encoder(encoder_ckpt_path: str, device: str):
    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8,
    ).to(device).eval()

    ckpt = torch.load(encoder_ckpt_path, map_location=device)
    state_dict = ckpt if isinstance(ckpt, dict) and "state_dict" not in ckpt else ckpt.get("state_dict", ckpt)
    has_visual_prefix = any(k.startswith("visual_transformer.") for k in state_dict.keys())
    if has_visual_prefix:
        prefix = "visual_transformer."
        state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

    image_encoder.load_state_dict(state_dict, strict=True)
    print("Visual encoder loaded successfully!")
    return image_encoder


def encode_image_feature(image_path: str, image_encoder: torch.nn.Module):
    enc_dev = module_param_device(image_encoder)
    image_tensor = nii_img_to_tensor(image_path).to(enc_dev)
    with torch.inference_mode():
        image_encoded = image_encoder(image_tensor.unsqueeze(0), return_encoded_tokens=True)
    image_encoded = image_encoded.to(device=enc_dev, dtype=torch.float16)
    return image_encoded, image_encoded.numel()


def main(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    print(f"Detected model name: {model_name}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=args.device,
    )

    model_device = infer_runtime_device(model)
    for module in model.modules():
        if hasattr(module, "inv_freq") and isinstance(module.inv_freq, torch.Tensor):
            if module.inv_freq.device != model_device:
                module.inv_freq = module.inv_freq.to(model_device)

    image_encoder = None
    if not args.use_precomputed_features:
        if not args.encoder_ckpt:
            raise ValueError("在线编码模式需要提供 --encoder-ckpt")
        image_encoder = build_image_encoder(args.encoder_ckpt, args.device)

    with open(args.data_json, "r") as file:
        data_val = json.load(file)

    unit_list = []
    for element in tqdm.tqdm(data_val):
        fmt = args.data_format if args.data_format != "auto" else detect_format(element)

        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llama3"
        print(f"conv_mode: {conv_mode}")
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print(f"[WARNING] the auto inferred conversation mode is {conv_mode}, while `--conv-mode` is {args.conv_mode}, using {args.conv_mode}")
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()

        if args.use_precomputed_features:
            npz_name = npz_basename_from_element(element, fmt)
            image_path = os.path.join(args.encoding_dir, npz_name)
            image = np.load(image_path)["arr"]
            image_size = image.size
            image_tensor = torch.tensor(image).float()
        else:
            image_path = get_image_path_from_element(element, fmt)
            image_tensor, image_size = encode_image_feature(image_path, image_encoder)

        if isinstance(image_tensor, list):
            image_tensor = [align_image_tensor_for_mm_projector(model, t) for t in image_tensor]
        else:
            image_tensor = align_image_tensor_for_mm_projector(model, image_tensor)

        if fmt == "amos":
            pred_messages = []
            messages = element["messages"]
            for i, msg in enumerate(messages):
                if msg.get("role") != "user":
                    continue
                inp = normalize_user_prompt(msg["content"])
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)

                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model_device)

                do_sample = args.temperature > 0
                generate_kwargs = dict(max_new_tokens=args.max_new_tokens, use_cache=True, do_sample=do_sample)
                if do_sample:
                    generate_kwargs["temperature"] = args.temperature
                    generate_kwargs["top_p"] = args.top_p
                else:
                    generate_kwargs["temperature"] = 1.0
                    generate_kwargs["top_p"] = 1.0

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=[image_size],
                        **generate_kwargs,
                    )

                input_token_len = input_ids.shape[1]
                new_token_ids = output_ids[0, input_token_len:]
                if new_token_ids.numel() > 0:
                    new_token_ids = new_token_ids[new_token_ids >= 0]
                outputs = tokenizer.decode(new_token_ids.tolist(), skip_special_tokens=True).strip() if new_token_ids.numel() > 0 else ""
                conv.messages[-1][-1] = outputs

                pred_messages.append({"role": "user", "content": msg["content"]})
                pred_messages.append({"role": "assistant", "content": outputs})

                if args.debug:
                    print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

            unit_list.append(build_amos_record_like_input(element, pred_messages))

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(unit_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data-json", type=str, default=None)
    parser.add_argument("--encoding-dir", type=str, default=None)
    parser.add_argument("--encoder-ckpt", type=str, default=None)
    parser.add_argument("--use-precomputed-features", action="store_true")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--data-format",
        type=str,
        choices=["auto", "legacy", "amos"],
        default="auto",
    )
    args = parser.parse_args()
    if args.output_json is None:
        data_abs = os.path.abspath(args.data_json)
        stem = os.path.splitext(os.path.basename(data_abs))[0]
        args.output_json = os.path.join(os.path.dirname(data_abs), f"{stem}_ctchat_infer.json")
    main(args)
