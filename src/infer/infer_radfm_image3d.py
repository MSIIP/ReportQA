import copy
import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm
import transformers
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many
from PIL import Image
from torch import einsum
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer


def unwrap_singleton(value):
    """Unwrap DataLoader batch_size=1 containers."""
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def replace_assistant_content(item, output_text):
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
# Inlined Dataset.multi_dataset_test + needed radiopaedia datasets
# ---------------------------------------------------------------------------


def build_tokenizer_with_image_tokens(text_tokenizer, max_img_size, image_num):
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
    target_h = 512
    target_w = 512
    target_d = 4

    if len(images) == 0:
        return torch.zeros((1, 3, target_h, target_w, target_d))

    max_d = 4
    d_list = list(range(4, 65, 4))

    for image in images:
        try:
            d = image.shape[3]
            if d > max_d:
                max_d = d
        except Exception:
            continue

    for temp_d in d_list:
        if abs(temp_d - max_d) < abs(target_d - max_d):
            target_d = temp_d

    stacked = []
    for image in images:
        image = torch.tensor(image)
        if len(image.shape) == 3:
            stacked.append(
                torch.nn.functional.interpolate(
                    image.unsqueeze(0).unsqueeze(-1),
                    size=(target_h, target_w, target_d),
                )
            )
        else:
            stacked.append(
                torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=(target_h, target_w, target_d),
                )
            )

    return torch.cat(stacked, dim=0)


def _load_medical_image_as_hwd(path):
    import nibabel as nib

    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".nii", ".gz", ".nrrd", ".mha", ".mhd"):
        return nib.load(path).get_fdata()

    image = np.array(Image.open(path).convert("L"), dtype=np.float32)
    if image.ndim == 2:
        image = image[..., np.newaxis]
    return image


def _normalize_image(image):
    if image.max() > image.min():
        image = (image - image.min()) / (image.max() - image.min())
    if np.isnan(image).any():
        image = np.random.randn(512, 512, 1)
    return image


class RadioCaption_Dataset_test(Dataset):
    def __init__(self, json_path, image_root):
        with open(json_path, "r") as file:
            self.json_data = json.load(file)
        self.image_root = image_root

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        data_index = self.json_data[index]
        question = data_index["conversations"][0]["value"]
        answer = data_index["answer"]
        sample_id = data_index["id"]
        if question.startswith("<image3d>\n"):
            question = question.replace("<image3d>\n", "", 1)

        img_relative_path = data_index["image3d"][0]
        img_path = os.path.join(self.image_root, img_relative_path)

        try:
            image = _load_medical_image_as_hwd(img_path)
        except Exception as exc:
            print(f"Error loading image {img_path}: {exc}")
            image = np.random.randn(512, 512, 10)

        image = _normalize_image(image)
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)
        image = image.expand(3, -1, -1, -1)

        image_dict = []
        if random.random() < 0.5:
            dict_idx = {"image": image, "position": {"question": 0}}
        else:
            dict_idx = {"image": image, "position": {"question": len(question)}}
        image_dict.append(dict_idx)

        return {
            "image_dict": image_dict,
            "question": question,
            "answer": answer,
            "id": sample_id,
        }


class MessagesVideoReportDataset(Dataset):
    """Report-generation JSON with messages plus paths in videos/images/image."""

    def __init__(self, json_path: str, image_root: str = ""):
        with open(json_path, "r") as file:
            self.json_data = json.load(file)
        self.image_root = image_root or ""

    def __len__(self):
        return len(self.json_data)

    @staticmethod
    def _messages_to_qa(data_index):
        question, answer = None, None
        for msg in data_index.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                question = content
            elif role == "assistant":
                answer = content
        return "" if question is None else question, "" if answer is None else answer

    def __getitem__(self, index):
        data_index = self.json_data[index]
        raw_item = data_index
        question, answer = self._messages_to_qa(data_index)
        sample_id = data_index.get("id", "N/A")

        for token in ("<video>\n", "<video>", "<image3d>\n", "<image3d>", "<image>\n", "<image>"):
            question = question.replace(token, "")
        question = question.strip()

        media_paths = []
        if isinstance(data_index.get("videos"), list) and len(data_index["videos"]) > 0:
            media_paths = data_index["videos"]
        elif isinstance(data_index.get("images"), list) and len(data_index["images"]) > 0:
            media_paths = data_index["images"]
        elif isinstance(data_index.get("image"), str) and data_index["image"]:
            media_paths = [data_index["image"]]

        if len(media_paths) == 0:
            raise KeyError('MessagesVideoReportDataset requires one of keys: "videos", "images", or "image".')

        image_dict = []
        for media_rel in media_paths:
            img_path = media_rel if os.path.isabs(media_rel) else os.path.join(self.image_root, media_rel)

            try:
                image = _load_medical_image_as_hwd(img_path)
            except Exception as exc:
                print(f"Error loading image {img_path}: {exc}")
                image = np.random.randn(512, 512, 1)

            image = _normalize_image(image)
            image = torch.from_numpy(image).float()
            if image.dim() == 2:
                image = image.unsqueeze(-1)
            image = image.unsqueeze(0)
            image = image.expand(3, -1, -1, -1)

            image_dict.append({"image": image, "position": {"question": 0}})

        return {
            "image_dict": image_dict,
            "question": question,
            "answer": answer,
            "id": sample_id,
            "raw_item_json": json.dumps(raw_item, ensure_ascii=False),
        }


class multi_dataset(Dataset):
    def __init__(
        self,
        text_tokenizer,
        test_split="close",
        max_seq=2048,
        max_img_size=10,
        image_num=32,
        voc_size=32000,
        dataset_json_path=None,
        dataset_image_root=None,
    ):
        self.text_tokenizer = text_tokenizer
        self.max_img_size = max_img_size
        self.image_num = image_num
        self.max_seq = max_seq
        self.voc_size = voc_size
        self.H = 512
        self.W = 512
        self.image_padding_tokens = []
        self.test_split = test_split

        self.text_tokenizer, self.image_padding_tokens = build_tokenizer_with_image_tokens(
            self.text_tokenizer,
            max_img_size,
            image_num,
        )

        self.data_whole_2D = []
        self.data_whole_3D = []
        self.dataset_reflect = {}

        if dataset_json_path:
            report_ds = MessagesVideoReportDataset(
                json_path=dataset_json_path,
                image_root=dataset_image_root or "",
            )
            self.dataset_reflect["messages_video_report"] = report_ds
            self.data_whole_3D = self.data_whole_3D + [{"messages_video_report": i} for i in range(len(report_ds))]
            print(f"MessagesVideoReportDataset loaded from {dataset_json_path} with {len(report_ds)} samples.")
        else:
            print(f"No dataset json path provided")

        self.data_whole = self.data_whole_2D + self.data_whole_3D

    def __len__(self):
        return len(self.data_whole)

    def __getitem__(self, idx):
        sample = list(self.data_whole[idx].items())[0]
        belong_to = sample[0]
        sample = self.dataset_reflect[sample[0]][sample[1]]

        images = sample["image_dict"]
        if len(images) > 8:
            images = random.sample(images, 8)

        question = str(sample["question"])
        answer = str(sample["answer"])
        sample_id = sample.get("id", "N/A")
        raw_item_json = sample.get("raw_item_json")
        images, question, answer = self.text_add_image(images, question, answer)

        try:
            vision_x = stack_images(images)
        except Exception:
            print(self.data_whole[idx].items())
            input()

        out = {
            "vision_x": vision_x,
            "question": question,
            "answer": answer,
            "belong_to": belong_to,
            "id": sample_id,
        }
        if raw_item_json is not None:
            out["raw_item_json"] = raw_item_json
        return out

    def text_add_image(self, images, question, answer):
        ref_image = []
        question_list = [[] for _ in range(len(str(question)))]
        answer_list = [[] for _ in range(len(str(answer)))]

        for index, image in enumerate(images):
            ref_image.append(image["image"])
            position = image["position"]
            position = list(position.items())[0]

            if position[0] == "question":
                insert_loc = position[1] - 1
                if insert_loc < 0:
                    insert_loc = 0
                question_list[insert_loc].append(index)
            if position[0] == "answer":
                insert_loc = position[1] - 1
                if insert_loc < 0:
                    insert_loc = 0
                answer_list[insert_loc].append(index)

        new_question = ""
        new_answer = ""
        for char_i in range(len(question)):
            if question_list[char_i] == []:
                new_question = new_question + question[char_i]
            else:
                for img_index in question_list[char_i]:
                    try:
                        new_question = new_question + "<image>" + self.image_padding_tokens[img_index] + "</image>"
                    except Exception:
                        print("Error: out of max image input size")
                new_question = new_question + question[char_i]

        for char_i in range(len(answer)):
            if answer_list[char_i] == []:
                new_answer = new_answer + answer[char_i]
            else:
                for img_index in answer_list[char_i]:
                    try:
                        new_answer = new_answer + "<image>" + self.image_padding_tokens[img_index] + "</image>"
                    except Exception:
                        print("Error: out of max image input size")
                new_answer = new_answer + answer[char_i]

        new_answer = new_answer.replace("•", "")
        return ref_image, new_question, new_answer


# ---------------------------------------------------------------------------
# Inlined Model.RadFM helpers, ViT, decoder, embedding, and multimodal model
# ---------------------------------------------------------------------------


def exists(val):
    return val is not None


def perceiver_feed_forward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        h = self.heads
        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=6,
        dim_head=64,
        heads=8,
        num_latents=64,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = nn.Parameter(torch.randn(max_num_frames, dim)) if exists(max_num_frames) else None
        self.media_time_embs = nn.Parameter(torch.randn(max_num_media, 1, dim)) if exists(max_num_media) else None
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        perceiver_feed_forward(dim=dim, mult=ff_mult),
                    ]
                )
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, t, f, v = x.shape[:4]
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:f], "F d -> b T F v d", b=b, T=t, v=v)
            x = x + frame_embs
        x = rearrange(x, "b T F v d -> b T (F v) d")
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:t]
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=t)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)


class PositionEmbeddingLearned3d(nn.Module):
    def __init__(self, num_pos_feats=256, h_patch_num=16, w_patch_num=16, d_patch_num=64):
        super().__init__()
        self.h_patch_num = h_patch_num
        self.w_patch_num = w_patch_num
        self.d_patch_num = d_patch_num
        self.row_embed = nn.Embedding(h_patch_num, num_pos_feats)
        self.col_embed = nn.Embedding(w_patch_num, num_pos_feats)
        self.dep_embed = nn.Embedding(d_patch_num, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.dep_embed.weight)

    def forward(self, batch_size, h, w, d, x):
        i = (torch.arange(h, device=x.device) + 1) * (self.h_patch_num // h) - 1
        j = (torch.arange(w, device=x.device) + 1) * (self.w_patch_num // w) - 1
        k = (torch.arange(d, device=x.device) + 1) * (self.d_patch_num // d) - 1
        x_emb = self.row_embed(i).unsqueeze(1).unsqueeze(2).repeat(1, w, d, 1)
        y_emb = self.col_embed(j).unsqueeze(0).unsqueeze(2).repeat(h, 1, d, 1)
        z_emb = self.dep_embed(k).unsqueeze(0).unsqueeze(1).repeat(h, w, 1, 1)
        pos = torch.cat([x_emb, y_emb, z_emb], dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        return rearrange(pos, "b h w d c -> b (h w d) c")


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViTPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class ViTFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ViTAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ViTTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        ViTPreNorm(dim, ViTAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        ViTPreNorm(dim, ViTFeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0
        assert frames % frame_patch_size == 0
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.frame_patch_size = frame_patch_size

        patch_dim = channels * patch_height * patch_width * frame_patch_size
        assert pool in {"cls", "mean"}
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) (f pf) -> b (h w f) (p1 p2 pf c)",
                p1=patch_height,
                p2=patch_width,
                pf=frame_patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = PositionEmbeddingLearned3d(
            dim // 3,
            (image_height // patch_height),
            (image_width // patch_width),
            (frames // frame_patch_size),
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = ViTTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, video):
        batch_size, channels, height, width, depth = video.shape
        x = self.to_patch_embedding(video)
        pos = self.pos_embedding(
            batch_size,
            height // self.patch_height,
            width // self.patch_width,
            depth // self.frame_patch_size,
            x,
        )
        x += pos
        x = self.dropout(x)
        x = self.transformer(x)
        return x, pos


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        output = tgt
        intermediate = []
        atten_layers = []
        for layer in self.layers:
            output, weights = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
                residual=True,
            )
            atten_layers.append(weights)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output, atten_layers


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
        residual=True,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt)
        tgt2, weights = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, weights

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2, weights = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2, attn_weights = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, attn_weights

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
        residual=True,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
            residual,
        )


class MyEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings=32000,
        embedding_dim=5120,
        perceiver_num=32,
        vis_dim=768,
        patch_size=32,
        frame_patch_size=4,
        seg_channel=256,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.randn((num_embeddings, embedding_dim)))
        self.figure_token_weight = nn.Parameter(torch.randn((2, embedding_dim)))
        self.flag = "Text"
        self.patch_size = patch_size
        self.frame_patch_size = frame_patch_size
        self.seg_channel = seg_channel

        self.bert_tokenizer = AutoTokenizer.from_pretrained("path/to/MedKEBERT")
        self.bert_model = AutoModel.from_pretrained("path/to/MedKEBERT")
        self.bert_projection_fc = nn.Linear(768, vis_dim)

        self.vision_encoder = ViT(
            image_size=512,
            frames=512,
            image_patch_size=patch_size,
            frame_patch_size=frame_patch_size,
            dim=vis_dim,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
        )

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(vis_dim, vis_dim // 4, kernel_size=2, stride=2),
            nn.BatchNorm3d(vis_dim // 4),
            nn.GELU(),
            nn.ConvTranspose3d(vis_dim // 4, vis_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

        decoder_layer = TransformerDecoderLayer(d_model=vis_dim, nhead=8, normalize_before=True)
        decoder_norm = nn.LayerNorm(vis_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=4, norm=decoder_norm)

        self.transformer_decoder_mlp = nn.Sequential(
            nn.Linear(vis_dim, vis_dim // 4),
            nn.GELU(),
            nn.Linear(vis_dim // 4, vis_dim // 8),
            nn.GELU(),
        )
        self.vis_dim = vis_dim
        self.perceiver = PerceiverResampler(dim=self.vis_dim, num_latents=perceiver_num)
        self.fc = nn.Linear(self.vis_dim, self.embedding_dim)
        self.cls_head = nn.Linear(self.vis_dim // 8, 1)

    def forward(self, text_input, vision_x, key_words_query=None):
        if self.flag == "Text":
            if vision_x is None or vision_x.numel() == 0 or torch.count_nonzero(vision_x).item() == 0:
                batch_size = text_input.shape[0]
                embedding_weight = torch.cat([self.weight, self.figure_token_weight], dim=0)
                embedding_weight = embedding_weight.unsqueeze(0).repeat(batch_size, 1, 1)
                text_input = F.one_hot(text_input, embedding_weight.shape[1]).to(embedding_weight.dtype).to(embedding_weight.device)
                out_put = torch.matmul(text_input, embedding_weight)
                return out_put, None

            batch_size, series, channels, height, width, depth = vision_x.shape
            vision_x = rearrange(vision_x, "b S c h w d-> (b S) c h w d")
            vision_x, pos_embedding = self.vision_encoder(vision_x)
            vision_x = rearrange(vision_x, "(b s F) v d -> b s F v d", b=batch_size, s=series, F=1)

            loss_matching = None
            if key_words_query is not None:
                query_words = [item for sublist in key_words_query for item in sublist]
                query_words = list(set(query_words))
                if len(query_words) > 16:
                    random.shuffle(query_words)
                    query_words = query_words[0:16]

                if query_words != []:
                    contrastive_labels = torch.zeros(batch_size, len(query_words))
                    for i, sublist in enumerate(key_words_query):
                        for j, item in enumerate(query_words):
                            if item in sublist:
                                contrastive_labels[i, j] = 1
                    contrastive_labels = contrastive_labels.to(vision_x.dtype).to(vision_x.device)

                    with torch.no_grad():
                        query_words_embedding = self.bert_tokenizer(
                            query_words,
                            padding="max_length",
                            truncation=True,
                            max_length=256,
                            return_tensors="pt",
                        )
                        query_words_embedding = self.bert_model(
                            input_ids=query_words_embedding["input_ids"].to(vision_x.device),
                            attention_mask=query_words_embedding["attention_mask"].to(vision_x.device),
                        )["last_hidden_state"][:, 0, :].to(vision_x.dtype).to(vision_x.device)

                    query_words_embedding = self.bert_projection_fc(query_words_embedding)
                    query_words_embedding = query_words_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
                    _, num_queries, _ = query_words_embedding.shape
                    image_embedding = vision_x.mean(dim=1)
                    image_embedding = rearrange(image_embedding, "b F v d -> b (F v) d")
                    pos_embedding = rearrange(pos_embedding, "(b s) v d -> b s v d", b=batch_size, s=series)[:, 0, :, :]
                    image_embedding = image_embedding.transpose(0, 1)
                    pos_embedding = pos_embedding.transpose(0, 1)
                    query_words_embedding = query_words_embedding.transpose(0, 1)

                    oo_embedding, _ = self.transformer_decoder(query_words_embedding, image_embedding, pos=pos_embedding)
                    oo_embedding = oo_embedding.transpose(0, 1)
                    oo_embedding = rearrange(oo_embedding, "b n d -> (b n) d")
                    oo_embedding = self.transformer_decoder_mlp(oo_embedding)
                    oo_embedding = self.cls_head(oo_embedding).mean(dim=-1)
                    oo_embedding = rearrange(oo_embedding, "(b n) -> b n", b=batch_size, n=num_queries)
                    loss_matching = F.binary_cross_entropy_with_logits(oo_embedding, contrastive_labels)

            vision_x = self.perceiver(vision_x)
            n = vision_x.shape[2]
            vision_x = rearrange(vision_x, "b s n d -> (b s n) d")
            vision_x = self.fc(vision_x)
            vision_x = rearrange(vision_x, "(b T) d -> b T d", b=batch_size, T=n * series)

            embedding_weight = torch.cat([self.weight, self.figure_token_weight], dim=0)
            embedding_weight = embedding_weight.unsqueeze(0).repeat(batch_size, 1, 1)
            embedding_weight = torch.cat([embedding_weight, vision_x], dim=1)
            text_input = F.one_hot(text_input, embedding_weight.shape[1]).to(vision_x.dtype).to(vision_x.device)
            out_put = torch.matmul(text_input, embedding_weight)

        return out_put, loss_matching


class MultiLLaMAForCausalLM(nn.Module):
    def __init__(self, lang_model_path):
        super().__init__()
        self.lang_model = LlamaForCausalLM.from_pretrained(lang_model_path)
        self.lang_model.gradient_checkpointing_enable()
        self.lang_model.enable_input_require_grads()
        self.embedding_layer = MyEmbedding()
        self.embedding_layer.weight = self.lang_model.get_input_embeddings().weight
        self.hidden_dim = 5120
        self.voc_size = 32000

    def forward(self, lang_x, vision_x, attention_mask, labels, loss_reweight, key_words_query):
        if labels.shape == lang_x.shape:
            self.embedding_layer.flag = "Text"
            input_embedding, loss_match = self.embedding_layer(lang_x, vision_x, key_words_query)
            output = self.lang_model(inputs_embeds=input_embedding, attention_mask=attention_mask, labels=labels)
            logits = output["logits"]
            loss_reg = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_loss_reweight = loss_reweight[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                shift_logits = shift_logits.view(-1, self.voc_size)
                shift_labels = shift_labels.view(-1)
                shift_loss_reweight = shift_loss_reweight.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                shift_loss_reweight = shift_loss_reweight.to(shift_logits.device)
                loss_reg = loss_fct(shift_logits, shift_labels)
                loss_reg = torch.sum(shift_loss_reweight * loss_reg) / torch.sum(shift_loss_reweight)
            loss = loss_reg
            if loss_match is not None:
                loss = 0.8 * loss + 0.2 * loss_match
            logits = output["logits"][..., :-1, :].contiguous().detach()
            total = len(labels)
            predictions = torch.argmax(logits, dim=-1)
            labels = labels[..., 1:].contiguous()
            accuracy = torch.sum(torch.all(torch.logical_or(predictions == labels, labels == -100), dim=-1)) / total
            return dict(logits=accuracy, loss=output["loss"])

    def generate(self, lang_x, vision_x, max_new_tokens=200):
        self.embedding_layer.flag = "Text"
        with torch.no_grad():
            input_embedding, _ = self.embedding_layer(lang_x, vision_x)
            generation = self.lang_model.generate(inputs_embeds=input_embedding, max_new_tokens=max_new_tokens, top_k=50)
        return generation


# ---------------------------------------------------------------------------
# Original infer_RadFM_image3d.py entrypoint
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
            "help": "Tokenizer directory (tokenizer.model, tokenizer_config.json). "
            "Defaults to lang_encoder_path when omitted — same folder as the base LM is fine.",
        },
    )
    radfm_ckpt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to RadFM pytorch_model.bin checkpoint."},
    )


@dataclass
class DataArguments:
    Mode: Optional[str] = field(default="Train")
    test_split: Optional[str] = field(default="open")
    dataset_json_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to JSON with messages+videos (report-gen format)."},
    )
    dataset_image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Optional root directory if videos[] paths are relative."},
    )
    output_json: Optional[str] = field(default=None)
    max_new_tokens: int = field(default=512)
    max_samples: Optional[int] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    batch_size_2D: int = field(default=4)
    batch_size_3D: int = field(default=1)
    output_dir: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not model_args.lang_encoder_path:
        raise ValueError("--lang_encoder_path is required (base LLaMA / MedLLaMA directory).")
    if not model_args.radfm_ckpt_path:
        raise ValueError("--radfm_ckpt_path is required (RadFM pytorch_model.bin).")
    tokenizer_source = model_args.tokenizer_path or model_args.lang_encoder_path

    print("Setup Data")
    test_dataset = multi_dataset(
        text_tokenizer=tokenizer_source,
        test_split=data_args.test_split,
        dataset_json_path=data_args.dataset_json_path,
        dataset_image_root=data_args.dataset_image_root,
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

    ckpt = torch.load(model_args.radfm_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    print("-------------Model loaded successfully---------------")
    model = model.to("cuda")
    model.eval()

    if data_args.output_json:
        out_json_path = data_args.output_json
    elif data_args.dataset_json_path:
        dataset_json_path = os.path.abspath(data_args.dataset_json_path)
        stem = os.path.splitext(os.path.basename(dataset_json_path))[0]
        out_json_path = os.path.join(os.path.dirname(dataset_json_path), stem + "_pred.json")
    else:
        out_json_path = "output_whole_2_id_epoch" + data_args.test_split + ".json"

    print(f"Writing predictions to: {out_json_path}")
    os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)

    results_list = []
    for sample_idx, sample in enumerate(tqdm.tqdm(test_dataloader)):
        if data_args.max_samples is not None and sample_idx >= data_args.max_samples:
            break

        question = unwrap_singleton(sample["question"])
        belong_to = unwrap_singleton(sample["belong_to"])
        sample_id = unwrap_singleton(sample["id"])
        answer = unwrap_singleton(sample["answer"])
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
            generation = model.generate(lang_x, vision_x, max_new_tokens=data_args.max_new_tokens)
            generated_texts = test_dataset.text_tokenizer.batch_decode(generation, skip_special_tokens=True)
            pred_text = generated_texts[0] if isinstance(generated_texts, list) and len(generated_texts) > 0 else ""

            if raw_item is not None:
                result_item = replace_assistant_content(raw_item, pred_text)
            else:
                result_item = replace_assistant_content(
                    {
                        "id": sample_id,
                        "belong_to": belong_to,
                        "messages": [
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ],
                    },
                    pred_text,
                )

            results_list.append(result_item)
        except Exception as exc:
            print(f"Skip one sample due to generation error: {exc}")
            continue

    with open(out_json_path, mode="w", encoding="utf-8") as outfile:
        json.dump(results_list, outfile, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
