# Standalone HuluMed inference file.
# This file inlines the local hulumed package pieces needed by inference.py
# so the inference entrypoint does not import from hulumed.* at runtime.

import argparse
import ast
import base64
import copy
import json
import math
import os
import re
import shutil
import sys
import traceback
import warnings
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import cv2
import einops
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as VF
from PIL import Image
from torch.nn.init import _calculate_fan_in_and_fan_out
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPImageProcessor,
    CLIPVisionConfig,
    CLIPVisionModel,
    PretrainedConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3Model,
    SiglipImageProcessor,
    SiglipVisionConfig,
    SiglipVisionModel,
    StoppingCriteria,
)
from transformers.activations import ACT2FN
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation.utils import GenerateOutput
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import TensorType, is_flash_attn_2_available, is_vision_available, logging

VideoInput = ImageInput


def disable_torch_init():
    """Disable redundant default initialization to speed up model creation."""
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)




# ---------------------------------------------------------------------------
# Inlined from hulumed/constants
# ---------------------------------------------------------------------------

CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100

# Image arguments
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# Video arguments
VIDEO_TOKEN_INDEX = -201
DEFAULT_VIDEO_TOKEN = "<video>"
NUM_FRAMES = 128
MAX_FRAMES = 768
NUM_FRAMES_PER_SECOND = 1

# Audio arguments
AUDIO_TOKEN_INDEX = -202
DEFAULT_AUDIO_TOKEN = "<audio>"

# Stream arguments
STREAM_START_TOKEN = "<|stream_start|>"
STREAM_END_TOKEN = "<|stream_end|>"
STREAM_MAX_FRAMES = 400

MODAL_INDEX_MAP = {
    "<image>": -200,
    "<video>": -201,
    "<audio>": -202,
}

subimage_token_num=196



# ---------------------------------------------------------------------------
# Inlined from hulumed/image_processing
# ---------------------------------------------------------------------------

# Adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py.
# Below is the original copyright:
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for HuluMed."""

import math
from typing import Dict, List, Optional, Union

import numpy as np

import torch
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    # VideoInput,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
)
VideoInput = ImageInput  # 兼容性处理

from transformers.utils import TensorType, is_vision_available, logging


logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image


def is_valid_video(video) -> bool:
    if isinstance(video, (list, tuple)):
        return all(is_valid_image(frame) for frame in video)
    elif isinstance(video, np.ndarray):
        return video.ndim == 4
    elif isinstance(video, torch.Tensor):
        return video.ndim == 4
    return False


def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Accepts images in list or nested list format, and makes a list of images for preprocessing.

    Args:
        images (`Union[List[List[ImageInput]], List[ImageInput], ImageInput]`):
            The input image.

    Returns:
        list: A list of images.
    """
    if isinstance(images, (list, tuple)):
        # list of images/videos
        if not all(is_valid_video(image) or is_valid_image(image) for image in images):
            raise ValueError(f"Could not make batched images from {images}")
        return images
    elif is_valid_video(images) or is_valid_image(images):
        # single image/video
        return [images]

    raise ValueError(f"Could not make batched images from {images}")


def simple_batched_resize(
    images, factor: int = 28, min_tokens: int = 4 * 4, max_tokens: int = 16384, input_data_format: str = None
):
    min_pixels = min_tokens * factor * factor
    # max_pixels = max_tokens * factor * factor
    max_pixels = 200704

    num_images = 0
    for image in images:
        if is_valid_video(image):
            num_images += len(image)
        else:
            num_images += 1

    image_sizes = []
    for image in images:
        if is_valid_video(image):
            image = image[0]
        if isinstance(image, Image.Image):
            height, width = image.size
        else:
            height, width = get_image_size(image, channel_dim=input_data_format)
        image_sizes.append([height, width])

    tmp_image_sizes = []
    for height, width in image_sizes:
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > (max_pixels // num_images):
            beta = math.sqrt((height * width) / (max_pixels // num_images))
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        # per image min_pixels
        if h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        tmp_image_sizes.append((h_bar, w_bar))
    image_sizes = tmp_image_sizes
    return image_sizes


def batched_resize(
    images, factors: List[int], min_tokens: int = 4 * 4, max_tokens: int = 16384, input_data_format: str = None
):
    image_sizes = []
    for image in images:
        if is_valid_video(image):
            num_frame = len(image)
            image = image[0]
        else:
            num_frame = 1
        if isinstance(image, Image.Image):
            height, width = image.size
        else:
            height, width = get_image_size(image, channel_dim=input_data_format)
        image_sizes.append([num_frame, height, width])

    # global max_pixels
    smart_scale_factors = 1.0
    total_tokens = 0
    for (num_frame, height, width), factor in zip(image_sizes, factors):
        total_tokens += num_frame * math.ceil(height / factor) * math.ceil(width / factor)

    # TODO: add min_pixels
    if total_tokens > max_tokens:
        beta = math.sqrt(total_tokens / max_tokens)
        tmp_image_sizes = []
        for (_, height, width), factor in zip(image_sizes, factors):
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
            tmp_image_sizes.append((h_bar, w_bar))
        image_sizes = tmp_image_sizes
    else:
        tmp_image_sizes = []
        for (_, height, width), factor in zip(image_sizes, factors):
            height = round(height / factor) * factor
            width = round(width / factor) * factor
            tmp_image_sizes.append((height, width))
        image_sizes = tmp_image_sizes

    return image_sizes


class HulumedImageProcessor(BaseImageProcessor):
    r"""
    Constructs a HuluMed image processor that dynamically resizes images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spacial patch size of the vision encoder.
    """

    model_input_names = ["pixel_values", "grid_sizes", "merge_sizes"]

    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_tokens: int = 4 * 4,
        max_tokens: int = 16384,
        patch_size: int = 14,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.patch_size = patch_size
        self.do_convert_rgb = do_convert_rgb

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        target_size: List[int],
        merge_size: int = 1,
        do_resize: bool = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            target_size (`List[int]`):
                The target size to resize the image to. Should be a list of two integers: [target_height, target_width].
            merge_size (`int`, *optional*, defaults to `1`):
                The merge size after the vision encoder.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = target_size
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        t = patches.shape[0]
        channel = patches.shape[1]
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.reshape(
            t,
            channel,
            grid_h // merge_size,
            merge_size,
            self.patch_size,
            grid_w // merge_size,
            merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 2, 5, 3, 6, 1, 4, 7)
        flatten_patches = patches.reshape(
            t * grid_h * grid_w, channel * self.patch_size * self.patch_size
        )

        return flatten_patches, (t, grid_h, grid_w)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        merge_size: Optional[Union[int, List[int]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = make_batched_images(images)

        if isinstance(merge_size, (list, tuple)):
            assert len(merge_size) == len(images), "Merge size must be the same length as images."
            merge_sizes = merge_size
        else:
            merge_sizes = [merge_size for _ in images]

        if all(merge_size == merge_sizes[0] for merge_size in merge_sizes):
            target_sizes = simple_batched_resize(
                images,
                factor=self.patch_size * merge_sizes[0],
                min_tokens=self.min_tokens,
                max_tokens=self.max_tokens,
                input_data_format=input_data_format,
            )
        else:
            target_sizes = batched_resize(
                images,
                factors=[self.patch_size * merge_size for merge_size in merge_sizes],
                min_tokens=self.min_tokens,
                max_tokens=self.max_tokens,
                input_data_format=input_data_format,
            )

        pixel_values, grid_sizes = [], []
        for image, merge_size, target_size in zip(images, merge_sizes, target_sizes):
            patches, grid_size = self._preprocess(
                image,
                target_size=target_size,
                merge_size=merge_size,
                do_resize=do_resize,
                resample=resample,
                do_rescale=do_rescale,
                rescale_factor=rescale_factor,
                do_normalize=do_normalize,
                image_mean=image_mean,
                image_std=image_std,
                data_format=data_format,
                do_convert_rgb=do_convert_rgb,
                input_data_format=input_data_format,
            )
            pixel_values.append(patches)
            grid_sizes.append(grid_size)

        pixel_values = np.concatenate(pixel_values, axis=0)
        grid_sizes = np.array(grid_sizes)
        merge_sizes = np.array(merge_sizes)

        data = {
            "pixel_values": pixel_values,
            "grid_sizes": grid_sizes,
            "merge_sizes": merge_sizes,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)



# ---------------------------------------------------------------------------
# Inlined from hulumed/vision_config
# ---------------------------------------------------------------------------

# Adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/configuration_siglip.py.
# Below is the original copyright:
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HuluMed vision encoder model configuration."""

from transformers import PretrainedConfig


class HulumedVisionEncoderConfig(PretrainedConfig):

    model_type = "hulumed_vision_encoder"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act



# ---------------------------------------------------------------------------
# Inlined from hulumed/vision_model
# ---------------------------------------------------------------------------

# Adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py.
# Below is the original copyright:
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch HuluMed vision encoder model."""

import importlib.util
import os.path as osp
import math
import warnings
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn.init import _calculate_fan_in_and_fan_out

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_flash_attn_2_available

# if is_flash_attn_2_available():
#     from flash_attn import flash_attn_varlen_func
# else:
#     flash_attn_varlen_func = None


try:
    if is_flash_attn_2_available():
        from flash_attn import flash_attn_varlen_func  # type: ignore[import-not-found]
    else:
        flash_attn_varlen_func = None
except (ImportError, RuntimeError, OSError):
    # flash-attn 已安装但二进制不兼容时，捕获错误并降级
    flash_attn_varlen_func = None

# HulumedVisionEncoderConfig is inlined above.


def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=a, max=b)


def trunc_normal_tf_(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> torch.Tensor:
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \text{mean} \\leq b`.

    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsequently scaled and shifted by the mean and std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    with torch.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def default_flax_embed_init(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class VisionRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs
    

class HulumedVisionEmbeddings(nn.Module):

    def __init__(self, config: HulumedVisionEncoderConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1, self.config.num_channels, self.patch_size, self.patch_size
        )
        patch_embeds = self.patch_embedding(hidden_states)  # shape = [*, width, grid, grid]
        # embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = patch_embeds.view(-1, self.embed_dim)

        return embeddings


class VisionAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        """Input shape: Time x Channel"""

        q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_heads, self.head_dim)

        query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
        key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.zeros([1, q_len, q_len], device=query_states.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)

        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(q_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output


class VisionFlashAttention2(VisionAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # Adapted from transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_heads, self.head_dim)
        query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
        key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
        
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(query_states, key_states, value_states, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            q_len, -1
        )
        attn_output = self.out_proj(attn_output)
        
        return attn_output


class VisionSdpaAttention(VisionAttention):

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(seq_length, self.num_heads, self.head_dim)
        key_states = key_states.view(seq_length, self.num_heads, self.head_dim)
        value_states = value_states.view(seq_length, self.num_heads, self.head_dim)

        query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
        key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = torch.zeros([1, seq_length, seq_length], device=query_states.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True

        query_states = query_states.transpose(0, 1)
        key_states = key_states.transpose(0, 1)
        value_states = value_states.transpose(0, 1)
        attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output


VISION_ATTENTION_CLASSES = {
    "eager": VisionAttention,
    "flash_attention_2": VisionFlashAttention2,
    "sdpa": VisionSdpaAttention,
}


# Copied from transformers.models.clip.modeling_clip.CLIPMLP with CLIP->Hulumed
class HulumedVisionMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class HulumedVisionEncoderLayer(nn.Module):

    def __init__(self, config: HulumedVisionEncoderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = VISION_ATTENTION_CLASSES[config._attn_implementation](config=config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = HulumedVisionMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.layer_norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class HulumedVisionTransformerEncoder(nn.Module):

    def __init__(self, config: HulumedVisionEncoderConfig):
        super().__init__()
        self.config = config
        head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.layers = nn.ModuleList([HulumedVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_sizes, merge_sizes):
        pos_ids = []
        for (t, h, w), merge_size in zip(grid_sizes, merge_sizes):
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_sizes[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)

        return rotary_pos_emb

    # def forward(self, hidden_states, grid_sizes, merge_sizes) -> torch.Tensor:
    #     rotary_pos_emb = self.rot_pos_emb(grid_sizes, merge_sizes)
    #
    #     cu_seqlens = torch.repeat_interleave(grid_sizes[:, 1] * grid_sizes[:, 2], grid_sizes[:, 0]).cumsum(dim=0, dtype=torch.int32)
    #     cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    #
    #     for blk in self.layers:
    #         if self.gradient_checkpointing and self.training:
    #             hidden_states = self._gradient_checkpointing_func(
    #                 blk.__call__,
    #                 hidden_states,
    #                 cu_seqlens,
    #                 rotary_pos_emb
    #             )
    #         else:
    #             hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
    #
    #     return hidden_states

    def forward(self, hidden_states, grid_sizes, merge_sizes, stop_at_layer: Optional[int] = None) -> torch.Tensor:
        rotary_pos_emb = self.rot_pos_emb(grid_sizes, merge_sizes)

        cu_seqlens = torch.repeat_interleave(grid_sizes[:, 1] * grid_sizes[:, 2], grid_sizes[:, 0]).cumsum(dim=0,
                                                                                                           dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        layers_to_run = self.layers[: stop_at_layer + 1] if stop_at_layer is not None else self.layers
        for blk in layers_to_run:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    blk.__call__,
                    hidden_states,
                    cu_seqlens,
                    rotary_pos_emb
                )
            else:
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return hidden_states


class HulumedVisionEncoderModel(PreTrainedModel):

    config_class = HulumedVisionEncoderConfig
    base_model_prefix = "hulumed"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "HulumedVisionEncoderLayer",
        "HulumedVisionEmbeddings",
    ]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: HulumedVisionEncoderConfig):
        super().__init__(config=config)
        embed_dim = config.hidden_size

        self.embeddings = HulumedVisionEmbeddings(config)
        self.encoder = HulumedVisionTransformerEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.post_init()

    def forward(self, pixel_values, grid_sizes, merge_sizes=None, stop_at_layer: Optional[int] = None) -> torch.Tensor:
        hidden_states = self.embeddings(pixel_values)
        # hidden_states = self.encoder(hidden_states, grid_sizes, merge_sizes)
        hidden_states = self.encoder(hidden_states, grid_sizes, merge_sizes, stop_at_layer=stop_at_layer)
        hidden_states = self.post_layernorm(hidden_states)

        if stop_at_layer is not None:
            return hidden_states

        hidden_states_chunks = hidden_states.split(grid_sizes.prod(dim=1).tolist(), dim=0)
        outputs = []

        for hidden_states, grid_size, merge_size in zip(hidden_states_chunks, grid_sizes, merge_sizes):
            # NOTE: previous implementation, which supports downsampling with any factor
            c = hidden_states.shape[-1]
            hidden_states = hidden_states.view(
                grid_size[0], grid_size[1] // merge_size, grid_size[2] // merge_size, merge_size, merge_size,  c
            ).permute(0, 1, 3, 2, 4, 5)
            hidden_states = hidden_states.reshape(
                grid_size[0], grid_size[1], grid_size[2], c
            ).permute(0, 3, 1, 2)
            hidden_states = torch.nn.functional.interpolate(
                hidden_states,
                size=(grid_size[1] // merge_size, grid_size[2] // merge_size),
                mode='bilinear'
            )
            hidden_states = hidden_states.permute(0, 2, 3, 1).view(-1, c)

            # NOTE: simplified implementation, which only supports downsampling with integer factor
            # NOTE: this implementation is mathematically equivalent to the previous one when merge_size is 1 or 2 but may cause slightly different results
            # hidden_states = hidden_states.view(-1, merge_size * merge_size, hidden_states.size(-1))
            # hidden_states = hidden_states.mean(dim=1)

            outputs.append(hidden_states)

        return torch.cat(outputs, dim=0)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, VisionAttention):
            nn.init.xavier_uniform_(module.q_proj.weight)
            nn.init.xavier_uniform_(module.k_proj.weight)
            nn.init.xavier_uniform_(module.v_proj.weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.q_proj.bias)
            nn.init.zeros_(module.k_proj.bias)
            nn.init.zeros_(module.v_proj.bias)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, HulumedVisionMLP):
            nn.init.xavier_uniform_(module.fc1.weight)
            nn.init.xavier_uniform_(module.fc2.weight)
            nn.init.normal_(module.fc1.bias, std=1e-6)
            nn.init.normal_(module.fc2.bias, std=1e-6)
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



# ---------------------------------------------------------------------------
# Inlined from hulumed/projector
# ---------------------------------------------------------------------------

#    Copyright 2024 Alibaba DAMO Academy
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import math
import os
import re

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import TRANSFORMERS_CACHE
except ImportError:
    from transformers.utils import TRANSFORMERS_CACHE



def parse_snapshot_folder(repo_id, cache_dir=None, repo_type="model"):
    revision = "main"
    # 1. parse the downloaded cache folder
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    else:
        cache_dir = cache_dir
    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    # 2. resolve refs (for instance to convert main to the associated commit sha)
    refs_dir = os.path.join(repo_cache, "refs")
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()
    # 3. acquire the snapshot folder
    folder = os.path.join(repo_cache, "snapshots", revision)

    return folder


def load_mm_projector(model_path, cache_dir=None, token=None):
    if os.path.exists(os.path.join(model_path, 'mm_projector.bin')):
        is_local = True
        folder = model_path
    else:
        is_local = False
        folder = parse_snapshot_folder(model_path, cache_dir=cache_dir, repo_type="model")
        if not os.path.exists(os.path.join(folder, 'mm_projector.bin')):
            # downloading from remote repo
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_path, cache_dir=cache_dir, token=token)

    mm_projector_weights = torch.load(os.path.join(folder, 'mm_projector.bin'), map_location='cpu')
    mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    return mm_projector_weights


class IdentityMap(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


def build_mlp(depth, hidden_size, output_hidden_size):
    modules = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        modules.append(nn.GELU())
        modules.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*modules)


class SimSpatialConv(nn.Module):

    def __init__(self, mm_hidden_size, hidden_size, downsample=(2, 2), padding=1, depth=1, mlp_depth=2):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size = mm_hidden_size
        self.output_hidden_size = output_hidden_size = hidden_size
        self.downsample = downsample
        self.padding = padding
        self.sampler = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder_hidden_size,
                out_channels=4 * self.encoder_hidden_size,
                kernel_size=self.downsample,
                stride=self.downsample,
                padding=self.padding,
                bias=True
            ),
            nn.SiLU(),
        )
        self.readout = build_mlp(mlp_depth, 4 * self.encoder_hidden_size, self.output_hidden_size)

    def forward(self, x):
        hw = int(x.size(1) ** 0.5)
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.sampler(x)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)
        return x

    def cal_proj_size(self, input_size):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        height = math.ceil((input_size[0] + self.padding) / self.downsample[0])
        width  = math.ceil((input_size[1] + self.padding) / self.downsample[1])
        return height * width


class MlpGeluProjector(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size, projector_type):
        super().__init__()

        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        mlp_depth = int(mlp_gelu_match.group(1))

        self.readout = build_mlp(mlp_depth, mm_hidden_size, hidden_size)

    def forward(self, x):
        x = self.readout(x)
        return x

    def cal_proj_size(self, input_size):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        height = input_size[0]
        width  = input_size[1]
        return height * width


def build_vision_projector(config, mm_hidden_size, delay_load=False, **kwargs):
    # hulumed projector only support image-wise operation now, i.e., prohibit the temporal aggregation
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    hidden_size = config.hidden_size

    if projector_type == "linear":
        # NOTE: for both linear and mlp2x_gelu projector type, mean pooling is adopted to aggreate video features
        return nn.Linear(mm_hidden_size, hidden_size)
    elif  projector_type == "simp_spatial_conv":
        return SimSpatialConv(mm_hidden_size, hidden_size)
    elif projector_type.startswith("mlp"):
        return MlpGeluProjector(mm_hidden_size, hidden_size, projector_type)
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')



# ---------------------------------------------------------------------------
# Inlined from hulumed/encoder
# ---------------------------------------------------------------------------

import os

import torch
import torch.nn as nn
from transformers import (CLIPImageProcessor, CLIPVisionConfig,
                          CLIPVisionModel, SiglipImageProcessor,
                          SiglipVisionConfig, SiglipVisionModel)



class CLIPVisionEncoder(nn.Module):

    def __init__(self, vision_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = vision_encoder
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.attn_implementation = getattr(args, 'mm_attn_implementation', 'flash_attention_2')
            self.load_model()
        else:
            # uncertain whether flash-attention-2 is supported during inference phase.
            self.attn_implementation = 'sdpa' # 'eager'
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self):
        if self.is_loaded:
            print('Vision tower is already loaded, `load model` call again, skipping.')
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_encoder_name)

        self.vision_encoder = CLIPVisionModel.from_pretrained(self.vision_encoder_name,
                                                            attn_implementation=self.attn_implementation)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images, **kwargs):
        images = torch.cat(images)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_encoder(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_encoder(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size


class SiglipVisionEncoder(nn.Module):

    def __init__(self, vision_encoder, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = vision_encoder
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.attn_implementation = getattr(args, 'mm_attn_implementation', 'flash_attention_2')
            self.load_model()
        else:
            self.attn_implementation = 'sdpa' # 'eager'
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self):
        if self.is_loaded:
            print('Vision tower is already loaded, `load model` call again, skipping.')
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_encoder_name)

        self.vision_encoder = SiglipVisionModel.from_pretrained(self.vision_encoder_name,
                                                              attn_implementation=self.attn_implementation)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images, **kwargs):
        images = torch.cat(images)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_encoder(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_encoder(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def image_size(self):
        return self.config.image_size



class HulumedVisionEncoder(nn.Module):

    def __init__(self, vision_encoder, args, delay_load=False, vision_encoder_config=None):
        super().__init__()

        self.is_loaded = False

        self.vision_encoder_name = vision_encoder
        self.args = args
        # For merged models: the vision encoder config is embedded in the main config
        self._vision_encoder_config = vision_encoder_config

        if not delay_load:
            self.attn_implementation = getattr(args, 'mm_attn_implementation', 'flash_attention_2')
            self.load_model(self.args)
        else:
            self.attn_implementation = 'sdpa' # 'eager'
            if vision_encoder_config is not None:
                cfg_dict = vision_encoder_config if isinstance(vision_encoder_config, dict) else vars(vision_encoder_config)
                self.cfg_only = HulumedVisionEncoderConfig(**cfg_dict)
            else:
                self.cfg_only = HulumedVisionEncoderConfig.from_pretrained(self.vision_encoder_name)

    def load_model(self, args):
        if self.is_loaded:
            print('Vision tower is already loaded, `load model` call again, skipping.')
            return

        if self._vision_encoder_config is not None:
            # Merged model: build architecture from embedded config,
            # weights will be loaded by the main model's from_pretrained
            cfg_dict = self._vision_encoder_config if isinstance(self._vision_encoder_config, dict) else vars(self._vision_encoder_config)
            vec_config = HulumedVisionEncoderConfig(**cfg_dict)
            self.cfg_only = vec_config
            self.image_processor = HulumedImageProcessor.from_pretrained(self.vision_encoder_name)
            print('视觉处理器已经加载完成。')
            self.vision_encoder = HulumedVisionEncoderModel(vec_config)
        else:
            # Separate vision encoder: load from pretrained path
            self.image_processor = HulumedImageProcessor.from_pretrained(self.vision_encoder_name)
            self.cfg_only = HulumedVisionEncoderConfig.from_pretrained(self.vision_encoder_name)
            self.vision_encoder = HulumedVisionEncoderModel.from_pretrained(
                self.vision_encoder_name,
                torch_dtype=args.torch_dtype,
                attn_implementation=self.attn_implementation)

        self.is_loaded = True

    # def forward(self, pixel_values, grid_sizes, merge_sizes, **kwargs):
    #     image_features = self.vision_encoder(pixel_values, grid_sizes, merge_sizes)
    #     return image_features

    def forward(self, pixel_values, grid_sizes, merge_sizes, stop_at_layer=None, **kwargs):
        image_features = self.vision_encoder(
            pixel_values, grid_sizes, merge_sizes,
            stop_at_layer=stop_at_layer if hasattr(self.vision_encoder, 'encoder') else None
        )
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_encoder.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return -1

    @property
    def num_patches_per_side(self):
        return -1

    @property
    def image_size(self):
        return -1


def build_vision_encoder(vision_encoder_cfg, **kwargs):
    vision_encoder = getattr(vision_encoder_cfg, 'mm_vision_encoder', getattr(vision_encoder_cfg, 'vision_encoder', None))

    if vision_encoder is not None:
        # Standard path-based loading (separate vision encoder model)
        if 'clip' in vision_encoder:
            return CLIPVisionEncoder(vision_encoder, args=vision_encoder_cfg, **kwargs)
        elif 'siglip' in vision_encoder:
            return SiglipVisionEncoder(vision_encoder, args=vision_encoder_cfg, **kwargs)
        elif 'navit' in vision_encoder.lower():
            return HulumedVisionEncoder(vision_encoder, args=vision_encoder_cfg, **kwargs)
        else:
            raise ValueError(f'Unknown vision encoder: {vision_encoder}')

    # Fallback: merged model with embedded vision_encoder_config
    vec = getattr(vision_encoder_cfg, 'vision_encoder_config', None)
    if vec is not None:
        model_type = vec.get('model_type', '') if isinstance(vec, dict) else getattr(vec, 'model_type', '')
        print('model_type from vision_encoder_config:', model_type)
        model_path = getattr(vision_encoder_cfg, '_name_or_path', None)
        print('model_path from vision_encoder_config:', model_path)
        if 'hulumed' in model_type and model_path:
            return HulumedVisionEncoder(model_path, args=vision_encoder_cfg, vision_encoder_config=vec, **kwargs)

    raise ValueError(
        f'Cannot determine vision encoder from config. '
        f'Expected mm_vision_encoder, vision_encoder (str path), or vision_encoder_config (dict). '
        f'Got vision_encoder={vision_encoder}, vision_encoder_config={vec}'
    )



# ---------------------------------------------------------------------------
# Inlined from hulumed/arch
# ---------------------------------------------------------------------------

# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import math
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn



def spatial_downsampling(features, grid_thws, stride=2):
    n, c = features.shape

    flatten_grid_thws = torch.cat([grid_thw for batch_grid_thws in grid_thws for grid_thw in batch_grid_thws])
    split_sizes = [grid_thw.prod() for grid_thw in flatten_grid_thws]
    features = torch.split(features, split_sizes)

    new_features = []
    for feature, grid_thw in zip(features, flatten_grid_thws):
        feature = feature.view(grid_thw[0], grid_thw[1] // stride, grid_thw[2] // stride, stride, stride,  c).permute(0, 1, 3, 2, 4, 5)
        feature = feature.reshape(grid_thw[0], grid_thw[1], grid_thw[2], c).permute(0, 3, 1, 2)
        new_feature = torch.nn.functional.interpolate(feature, (math.ceil(grid_thw[1] / stride), math.ceil(grid_thw[2] / stride)), mode='bilinear')
        new_features.append(new_feature.permute(0, 2, 3, 1).view(-1, c))
    new_features = torch.cat(new_features)

    return new_features


class HulumedMetaModel:

    def __init__(self, config):
        super(HulumedMetaModel, self).__init__(config)

        # Check for vision encoder: either a string path or an embedded config dict
        has_vision_encoder = (
            getattr(config, 'mm_vision_encoder', None) is not None
            or getattr(config, 'vision_encoder', None) is not None
            or getattr(config, 'vision_encoder_config', None) is not None
        )
        # if has_vision_encoder:
        #     self.vision_encoder = build_vision_encoder(config, delay_load=False)
        #     self.mm_projector = build_vision_projector(config, self.vision_encoder.hidden_size)
        if has_vision_encoder:
            _wrapper = build_vision_encoder(config, delay_load=False)
            # 拆掉 HulumedVisionEncoder 包装层，直接存储内部的 HulumedVisionEncoderModel
            # 这样权重路径从 model.vision_encoder.vision_encoder.encoder...
            # 变为model.vision_encoder.encoder...（与官方一致）
            if hasattr(_wrapper, '_vision_encoder_config'):
                self.vision_encoder = _wrapper.vision_encoder
                self.vision_encoder.image_processor = _wrapper.image_processor
                hidden_size = _wrapper.hidden_size
            else:
                self.vision_encoder = _wrapper
                hidden_size = _wrapper.hidden_size
            self.mm_projector = build_vision_projector(config, hidden_size)

    def get_vision_encoder(self):
        vision_encoder = getattr(self, 'vision_encoder', None)
        if type(vision_encoder) is list:
            vision_encoder = vision_encoder[0]
        return vision_encoder

    def get_mm_projector(self):
        return self.mm_projector

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_encoder = model_args.vision_encoder
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_projector = model_args.pretrain_mm_projector

        self.config.mm_vision_encoder = vision_encoder

        if self.get_vision_encoder() is None:
            vision_encoder = build_vision_encoder(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_encoder = [vision_encoder]
            else:
                self.vision_encoder = vision_encoder
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_encoder = self.vision_encoder[0]
            else:
                vision_encoder = self.vision_encoder

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_encoder.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_projector is not None:
            if os.path.exists(pretrain_mm_projector):
                is_local = True
                if os.path.isdir(pretrain_mm_projector):
                    mm_projector_weights = load_mm_projector(pretrain_mm_projector)
                else:
                    mm_projector_weights = torch.load(pretrain_mm_projector, map_location='cpu')
            else:
                is_local = False
                pretrain_mm_projector = pretrain_mm_projector.replace('mm_projector.bin', '')
                pretrain_mm_projector = pretrain_mm_projector.strip('/').strip('\\').strip()
                mm_projector_weights = load_mm_projector(pretrain_mm_projector)

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=False)


class HulumedMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_encoder(self):
        return self.get_model().get_vision_encoder()

    def get_mm_projector(self):
        return self.get_model().get_mm_projector()

    def encode_images(
        self,
        pixel_values: torch.FloatTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        stop_at_layer: Optional[int] = None,
    ) -> torch.FloatTensor:
        mm_features = self.get_model().get_vision_encoder()(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
            stop_at_layer=stop_at_layer,
        )
        if stop_at_layer is None:
            mm_features = self.get_model().mm_projector(mm_features)
        return mm_features

    def _get_valid_visual_tokens(
        self,
        mm_features: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
    ):
        valid_masks = []
        for num_patches, modal in zip(batched_num_patches, modals):
            valid_mask = torch.full((num_patches, ), modal != "text", dtype=torch.bool, device=mm_features.device)
            valid_masks.append(valid_mask)
        mm_features = mm_features[torch.cat(valid_masks)]
        return mm_features

    def _maybe_truncate_visual_tokens(
        self,
        mm_features: torch.FloatTensor,
        compression_mask: torch.BoolTensor,
        batched_num_patches: torch.LongTensor,
        modals: List[str],
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
    ):
        if position_ids is None or mm_features.shape[0] == input_ids.eq(self.config.image_token_index).sum():
            return mm_features, compression_mask

        truncation_mask = []
        for num_patches, modal in zip(batched_num_patches, modals):
            if modal == "text":
                truncation_mask.append(torch.ones((0,), dtype=torch.bool, device=input_ids.device))
            else:
                truncation_mask.append(torch.ones((num_patches,), dtype=torch.bool, device=input_ids.device))

        seq_end_indices = torch.nonzero(position_ids == 0)[:, 0]
        seq_end_indices = seq_end_indices[seq_end_indices > 0].tolist()+ [len(input_ids)]
        seq_start_indices = [0] + seq_end_indices[:-1]
        num_visual_tokens = [
            input_ids[start:end].eq(self.config.image_token_index).sum()
            for start, end in zip(seq_start_indices, seq_end_indices)
        ]

        for n, mask in zip(num_visual_tokens, truncation_mask):
            if len(mask) > 0:
                mask[n:] = False
        truncation_mask = torch.cat(truncation_mask)

        return mm_features[truncation_mask], compression_mask[truncation_mask]

    def _get_compression_mask(
        self,
        pixel_values: torch.FloatTensor,
        batched_num_patches: torch.LongTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        modals: List[str],
        threshold: float = 0.1,
        min_tokens: int = 1,
    ) -> torch.BoolTensor:
        batched_images = pixel_values.split(grid_sizes.prod(dim=1).tolist(), dim=0)
        compression_masks = []

        for images, num_patches, grid_size, merge_size, modal in zip(
            batched_images, batched_num_patches, grid_sizes, merge_sizes, modals
        ):
            t, h, w = grid_size
            if modal == "image" or (modal == "video" and t == 1):
                compression_masks.append(torch.ones((num_patches,), dtype=torch.bool, device=images.device))

            elif modal == "video":
                images = images.view(t, (h // merge_size) * (w // merge_size), -1)

                pixel_diff = images[1:] - images[:-1]
                pixel_diff = torch.abs(pixel_diff).mean(dim=-1) * 255
                pixel_diff = torch.cat([torch.full_like(pixel_diff[0:1], threshold + 1), pixel_diff], dim=0)
                mask = (pixel_diff / 255.0) > threshold
                padding_ids = torch.nonzero(mask.sum(dim=1) < min_tokens)[:, 0]
                mask[padding_ids, :min_tokens] = 1
                compression_masks.append(mask.flatten())

            else:
                compression_masks.append(torch.ones((0,), dtype=torch.bool, device=images.device))

        return torch.cat(compression_masks)

    def _compress_visual_tokens(
        self,
        compression_mask: torch.BoolTensor,
        mm_features: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        mm_features = mm_features[compression_mask]
        image_selected = (input_ids == self.config.image_token_index)

        text_masks = torch.logical_not(image_selected)
        text_masks[image_selected] = compression_mask
        input_ids = input_ids[text_masks]

        if attention_mask is not None:
            attention_mask = attention_mask[text_masks]
        if labels is not None:
            labels = labels[text_masks]
        if position_ids is not None:
            position_ids = position_ids[text_masks]
            pos_start = [0] + torch.nonzero(position_ids == 0)[:, 0].tolist()
            pos_end = pos_start[1:] + [len(input_ids)]
            position_ids = torch.cat([torch.arange(end - start, device=input_ids.device) for start, end in zip(pos_start, pos_end)])

        return mm_features, input_ids, attention_mask, position_ids, labels

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        precomputed_mm_features: Optional[torch.FloatTensor] = None,
    ):
        vision_encoder = self.get_vision_encoder()
        if vision_encoder is None or pixel_values is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, position_ids, past_key_values, None, labels, None

        B, N = input_ids.shape
        input_ids = input_ids.view(B * N)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B * N)
        if position_ids is not None:
            position_ids = position_ids.view(B * N)
        if labels is not None:
            labels = labels.view(B * N)

        batched_num_patches = grid_sizes.prod(dim=1).div(merge_sizes ** 2).long()
        # print("batched_num_patches:", batched_num_patches)  # tensor([1599, 1400]
        if precomputed_mm_features is not None:
            mm_features = precomputed_mm_features.to(input_ids.device)
        else:
            mm_features = self.encode_images(pixel_values, grid_sizes, merge_sizes).to(input_ids.device)
        # print("mm_features.shape:", mm_features.shape)  # torch.Size([2999, 2560]
        # print("modals:", modals)  # modals: None
        if modals is None:
            modals = ["image"] * grid_sizes.shape[0]
        mm_features = self._get_valid_visual_tokens(mm_features, batched_num_patches, modals)

        compression_mask = self._get_compression_mask(
            pixel_values, batched_num_patches, grid_sizes, merge_sizes, modals
        )
        mm_features, compression_mask = self._maybe_truncate_visual_tokens(
            mm_features, compression_mask, batched_num_patches, modals, input_ids, position_ids
        )

        if self.config.use_token_compression:
            assert B == 1, "Token compression is only supported for batch_size=1"
            mm_features, input_ids, attention_mask, position_ids, labels = self._compress_visual_tokens(
                compression_mask, mm_features, input_ids, attention_mask, position_ids, labels
            )

        inputs_embeds = self.get_model().embed_tokens(input_ids).clone()

        image_selected = (input_ids == self.config.image_token_index)
        # inputs_embeds[image_selected] = inputs_embeds[image_selected] * 0.0 + mm_features
        inputs_embeds[image_selected] = inputs_embeds[image_selected] * 0.0 + mm_features.to(inputs_embeds.dtype)

        C = inputs_embeds.shape[-1]
        inputs_embeds = inputs_embeds.reshape(B, -1, C)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B, -1)
        if labels is not None:
            labels = labels.view(B, -1)
        if position_ids is not None:
            position_ids = position_ids.view(B, -1)

        # image_token 位置的布尔掩码 [B, L]
        image_mask = image_selected.view(B, -1)

        return None, attention_mask, position_ids, past_key_values, inputs_embeds, labels, image_mask



# ---------------------------------------------------------------------------
# Inlined from hulumed/qwen
# ---------------------------------------------------------------------------

# Adopted from: https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (AutoConfig, AutoModelForCausalLM,
                          Qwen2Config, Qwen2ForCausalLM, Qwen2Model,
                          Qwen3Config, Qwen3ForCausalLM, Qwen3Model)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast



class HulumedQwen2Config(Qwen2Config):
    model_type = "hulumed_qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = kwargs.get("model_type", "hulumed_qwen2")


class HulumedQwen3Config(Qwen3Config):
    model_type = "hulumed_qwen3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = kwargs.get("model_type", "hulumed_qwen3")


class HulumedQwen2Model(HulumedMetaModel, Qwen2Model):
    config_class = HulumedQwen2Config

    def __init__(self, config: HulumedQwen2Config):
        super(HulumedQwen2Model, self).__init__(config)



class HulumedQwen3Model(HulumedMetaModel, Qwen3Model):
    config_class = HulumedQwen3Config

    def __init__(self, config: HulumedQwen3Config):
        super(HulumedQwen3Model, self).__init__(config)


class HulumedQwen2ForCausalLM(Qwen2ForCausalLM, HulumedMetaForCausalLM):
    config_class = HulumedQwen2Config

    def __init__(self, config, **kwargs):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = HulumedQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
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
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        loss = None
        if labels is not None:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            mask = shift_labels != IGNORE_INDEX
            shift_hidden_states = shift_hidden_states[mask]
            shift_labels = shift_labels[mask]
            logits = self.lm_head(shift_hidden_states)
            if "num_items_in_batch" in loss_kwargs:
                loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="sum")
                loss = loss / loss_kwargs["num_items_in_batch"]
            else:
                loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=IGNORE_INDEX)

        else:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        past_key_values = kwargs.pop("past_key_values", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if pixel_values is not None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=None,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


class HulumedQwen3ForCausalLM(Qwen3ForCausalLM, HulumedMetaForCausalLM):
    """Qwen3 主干；须与 Qwen3ForCausalLM 构成 super() MRO，不可再继承 HulumedQwen2ForCausalLM。"""

    config_class = HulumedQwen3Config

    def __init__(self, config, **kwargs):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = HulumedQwen3Model(config)
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
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=labels,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        loss = None
        if labels is not None:
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            mask = shift_labels != IGNORE_INDEX
            shift_hidden_states = shift_hidden_states[mask]
            shift_labels = shift_labels[mask]
            logits = self.lm_head(shift_hidden_states)
            if "num_items_in_batch" in loss_kwargs:
                loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="sum")
                loss = loss / loss_kwargs["num_items_in_batch"]
            else:
                loss = nn.functional.cross_entropy(logits, shift_labels, ignore_index=IGNORE_INDEX)

        else:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)
        past_key_values = kwargs.pop("past_key_values", None)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if pixel_values is not None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                labels=None,
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
                modals=modals,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs


AutoConfig.register("hulumed_qwen2", HulumedQwen2Config)
AutoModelForCausalLM.register(HulumedQwen2Config, HulumedQwen2ForCausalLM)

AutoConfig.register("hulumed_qwen3", HulumedQwen3Config)
AutoModelForCausalLM.register(HulumedQwen3Config, HulumedQwen3ForCausalLM)



# ---------------------------------------------------------------------------
# Inlined from hulumed/loader
# ---------------------------------------------------------------------------

# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import warnings
import shutil

import torch
from transformers import PretrainedConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig



VLLMs = {
    "hulumed_qwen2": HulumedQwen2ForCausalLM,
    "hulumed_qwen3": HulumedQwen3ForCausalLM,
}

VLLMConfigs = {
    "hulumed_qwen2": HulumedQwen2Config,
    "hulumed_qwen3": HulumedQwen3Config,
}



def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="cuda", **kwargs):
    if 'token' in kwargs:
        token = kwargs['token']
    else:
        token = None

    kwargs = {"device_map": device_map, **kwargs}

    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = kwargs.pop('attn_implementation', "flash_attention_2") # default to flash_attention_2

    torch_dtype = config.torch_dtype if hasattr(config, "torch_dtype") else kwargs.pop('torch_dtype', torch.float16)

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        # NOTE: High-version Transformers will report: """ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time."""
        # kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch_dtype

    model_type = config.model_type if hasattr(config, "model_type") else kwargs.pop('model_type', "hulumed_qwen2")

    is_alignment = getattr(config, "tune_mm_mlp_adapter", False) or getattr(config, "is_alignment", False)

    # NOTE: lora/qlora model loading
    if 'lora' in model_name.lower() or 'qlora' in model_name.lower():
        cfg_pretrained = PretrainedConfig.from_pretrained(model_path, token=token)
        # NOTE: AutoConfig will modify `_name_or_path` property to `model_path` if `model_path` is not None.
        # cfg_pretrained = AutoConfig.from_pretrained(model_path, token=token)
        model_base = model_base if model_base is not None else cfg_pretrained._name_or_path

        if hasattr(cfg_pretrained, 'quantization_config'):
            del cfg_pretrained.quantization_config
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, token=token)
        print('Loading HuluMed from base model...')

        if 'qwen2' in model_base.lower():
            model = HulumedQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
        else:
            model = HulumedQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional HuluMed weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename, subfolder=None):
                cache_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    subfolder=subfolder)
                return torch.load(cache_file, map_location='cpu')
            non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')
    elif model_base is not None or '-base' in model_name.lower() or is_alignment:
        print('Loading HuluMed from base model...')
        cfg_pretrained = PretrainedConfig.from_pretrained(model_path, token=token)
        model_base = model_base if model_base is not None else cfg_pretrained._name_or_path

        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, token=token)

        # 8. 修改了 model_type 的检查条件和实例化的类名
        if model_type in ['hulumed', 'hulumed_qwen2']:
            model = HulumedQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)
        else:
            model = HulumedQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=config, **kwargs)

       
        mm_projector_weights = load_mm_projector(model_path, token=token)
        model.load_state_dict(mm_projector_weights, strict=False)
    elif 'hulumed' in model_type:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)

        if model_type in ['hulumed_qwen3']:
            model = HulumedQwen3ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
        elif model_type in ['hulumed_qwen2']:
            model = HulumedQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
        else:
            model = HulumedQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, config=config, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, token=token)
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config, **kwargs)

    processor = None
    
    if "hulumed" in model_type:
        vision_encoder = model.get_vision_encoder()
        processor = vision_encoder.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, processor, context_len




# ---------------------------------------------------------------------------
# Inlined from hulumed/mm_utils
# ---------------------------------------------------------------------------

import ast
import os
import re
import math
import base64
import traceback
from io import BytesIO
from typing import Optional

import torch
import torchvision.transforms.functional as VF
import numpy as np
from transformers import StoppingCriteria

import cv2
from PIL import Image


import nibabel as nib

def slice_nifti_to_images(nifti_path, num_slices=180, axis='axial'):
    try:
        
        axis_map = {'axial': 2, 'coronal': 1, 'sagittal': 0}
        if axis not in axis_map:
            raise ValueError("Axis arugment must be one of 'axial', 'coronal', 'sagittal'")
        
        slice_axis = axis_map[axis]

        nifti_file = nib.load(nifti_path)
        image_data = nifti_file.get_fdata()

        num_total_slices = image_data.shape[slice_axis]
        
        sampled_indices = np.linspace(0, num_total_slices - 1, num_slices, dtype=int)
        
        images = []
        for slice_index in sampled_indices:
            if slice_axis == 0:
                slice_2d = image_data[slice_index, :, :]
            elif slice_axis == 1:
                slice_2d = image_data[:, slice_index, :]
            else:
                slice_2d = image_data[:, :, slice_index]
            if slice_2d.max() > slice_2d.min():
                slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255.0
            slice_2d = slice_2d.astype(np.uint8)
            pil_image = Image.fromarray(slice_2d).convert('RGB')
            images.append(pil_image)
            
        return images

    except ImportError:
        raise ImportError("please run 'pip install nibabel'")
    except Exception as e:
        print(f" NIfTI  {nifti_path} error: {e}")
        return [] 
def chunk_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def grid_divide(image, cell_size):
    grid = []
    width, height = image.size
    for i in range(0, height, cell_size):
        row = []
        for j in range(0, width, cell_size):
            box = (j, i, j + cell_size, i + cell_size)
            row.append(image.crop(box))
        grid.append(row)

    return grid
def load_images(image_path, nii_slice_axis='axial', nii_num_slices=180):
    images = []

    def safe_open(f):
        try:
            with Image.open(f).convert('RGB') as img:
                return img
        except Exception:
            pass

    if isinstance(image_path, str) and (image_path.endswith('.nii') or image_path.endswith('.nii.gz')):
        print(f"Detected NIfTI file, slicing into {nii_num_slices} slices along '{nii_slice_axis}' axis...")
        images = slice_nifti_to_images(image_path, num_slices=nii_num_slices, axis=nii_slice_axis)

    elif isinstance(image_path, str) and os.path.isfile(image_path):
        img = safe_open(image_path)
        if img is not None:
            images.append(img)

    elif isinstance(image_path, str) and os.path.isdir(image_path):
        for f in sorted(os.listdir(image_path)):
            full_path = os.path.join(image_path, f)
            if os.path.isfile(full_path):
                img = safe_open(full_path)
                if img is not None:
                    images.append(img)

    elif isinstance(image_path, list) and isinstance(image_path[0], str):
        for f in image_path:
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                print(f"Detected NIfTI file, slicing into {nii_num_slices} slices along '{nii_slice_axis}' axis...")
                images.extend(slice_nifti_to_images(f, num_slices=nii_num_slices, axis=nii_slice_axis))
            else:
                img = safe_open(f)
                if img is not None:
                    images.append(img)

    elif isinstance(image_path, list) and isinstance(image_path[0], Image.Image):
        images = [img.convert('RGB') for img in image_path]

    elif isinstance(image_path, Image.Image):
        images = [image_path.convert('RGB')]

    else:
        if isinstance(image_path, str):
             raise ValueError(f"Unsupported image path or file type: {image_path}")
        else:
             raise ValueError(f"Unsupported image path type: {type(image_path)}")

    return images


def process_pad_image(image, padding_value=(0, 0, 0)):
    image = expand2square(image, padding_value)

    return [image]


def find_closest_aspect_ratio(src_ratio, tgt_ratios, ori_size, tgt_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = ori_size[0] * ori_size[1]
    for ratio in tgt_ratios:
        tgt_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(src_ratio - tgt_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * tgt_size[0] * tgt_size[1] * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def process_dynamic_image(image, image_size=384, use_thumbnail=True):
    min_num = 1
    max_num = 12

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    ori_size = image.size
    aspect_ratio = ori_size[0] / ori_size[1]

    tgt_ratios = []
    for n in range(min_num, max_num + 1):
        tgt_ratios.extend([(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num])
    tgt_ratios = set(tgt_ratios)
    tgt_ratios = sorted(tgt_ratios, key=lambda x: x[0] * x[1])

    tgt_ratio = find_closest_aspect_ratio(aspect_ratio, tgt_ratios, ori_size, image_size)

    tgt_width = image_size[0] * tgt_ratio[0]
    tgt_height = image_size[1] * tgt_ratio[1]
    resized_img = image.resize((tgt_width, tgt_height))

    image_grid = grid_divide(resized_img, image_size[0])

    if use_thumbnail:
        thumbnail_img = image.resize((image_size[0], image_size[1]))
        image_grid = [[thumbnail_img]] + image_grid

    return image_grid


def process_highres_image(image, image_size=384, use_thumbnail=True, padding_value=(0, 0, 0)):
    grid_width = [1, 2, 3]
    grid_width_real = [x * image_size for x in grid_width]

    longest_side = max(image.size)
    fit_grid_width_real = [x for x in grid_width_real if x >= longest_side]
    if len(fit_grid_width_real) == 0:
        select_size = max(grid_width_real)
    else:
        select_size = min(fit_grid_width_real)

    image_padded = expand2square(image, padding_value)
    image_padded = image_padded.resize((select_size, select_size))
    image_grid = grid_divide(image_padded, image_size)

    if use_thumbnail:
        thumbnail_img = image.resize((image_size, image_size))
        image_grid = [[thumbnail_img]] + image_grid

    return image_grid


def select_best_resolution(original_size, possible_resolutions):
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def process_anyres_image(image, image_size=384, use_thumbnail=True, padding_value=(0, 0, 0)):
    possible_grids = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    possible_resolutions = [(x * image_size, y * image_size) for x, y in possible_grids]

    best_resolution = select_best_resolution(image.size, possible_resolutions)

    nw, nh = best_resolution
    ow, oh = image.size

    scale_factor = min(nw / ow, nh / oh)
    new_size = (int(ow * scale_factor), int(oh * scale_factor))

    image_padded = Image.new("RGB", (nw, nh), padding_value)
    image_padded.paste(image.resize(new_size), ((nw - new_size[0]) // 2, (nh - new_size[1]) // 2))

    image_grid = grid_divide(image_padded, image_size)

    if use_thumbnail:
        thumbnail_img = image.resize((image_size, image_size))
        image_grid = [[thumbnail_img]] + image_grid

    return image_grid


def process_adares_image(image, image_size=384, use_thumbnail=True):
    min_num = 1
    max_num = 12

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    ori_size = image.size
    aspect_ratio = ori_size[0] / ori_size[1]

    tgt_ratios = []
    for n in range(min_num, max_num + 1):
        tgt_ratios.extend([(i, j) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num])
    tgt_ratios = set(tgt_ratios)
    possible_resolutions = [(x * image_size[0], y * image_size[1]) for x, y in tgt_ratios]

    best_resolution = select_best_resolution(ori_size, possible_resolutions)

    resized_img = image.resize((best_resolution[0], best_resolution[1]))

    image_grid = grid_divide(resized_img, image_size[0])

    if use_thumbnail:
        thumbnail_img = image.resize((image_size[0], image_size[1]))
        image_grid = [[thumbnail_img]] + image_grid

    return image_grid


def process_images(image_path, processor, aspect_ratio='anyres', image_size=384, use_thumbnail=True):
    images = load_images(image_path)

    padding_value = tuple(int(x*255) for x in processor.image_mean)

    image_grids = []
    for image in images:
        if aspect_ratio == 'pad':
            image_grid = process_pad_image(image, padding_value=padding_value)
        elif aspect_ratio == 'dynamic':
            image_grid = process_dynamic_image(image, image_size=image_size, use_thumbnail=use_thumbnail)
        elif aspect_ratio == 'highres':
            image_grid = process_highres_image(image, image_size=image_size, use_thumbnail=use_thumbnail, padding_value=padding_value)
        elif aspect_ratio == 'anyres':
            image_grid = process_anyres_image(image, image_size=image_size, use_thumbnail=use_thumbnail, padding_value=padding_value)
        elif aspect_ratio == 'adares':
            image_grid = process_adares_image(image, image_size=image_size, use_thumbnail=use_thumbnail)
        else:
            image_grid = [image]

        image_grid = [processor.preprocess(image_row, return_tensors='pt', num_images=len(images)) for image_row in image_grid]
        image_grids.append(image_grid)

    return image_grids


def frame_sample(duration, mode='uniform', num_frames=None, vid_fps=None, fps=None):
    if mode == 'uniform':
        assert num_frames is not None, "Number of frames must be provided for uniform sampling."
        if duration <= num_frames:
            return np.arange(duration).astype(int)
        return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == 'fps':
        assert vid_fps is not None, "FPS must be provided for FPS sampling."
        fps = fps if fps is not None else NUM_FRAMES_PER_SECOND
        segment_len = min(vid_fps // fps, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ImportError(f'Unsupported frame sampling mode: {mode}')


def load_video_from_ids(video_path, s=None, e=None, fps=None, max_frames=None, temporal_factor=1):
    import imageio  # type: ignore[import-not-found]
    from decord import VideoReader, cpu  # type: ignore[import-not-found]

    if s is not None and e is not None:
        s = s if s >= 0. else 0.
        e = e if e >= 0. else 0.
        if s > e:
            s, e = e, s
        elif s == e:
            e = s + 1

    if os.path.isdir(video_path):
        frame_files = sorted(os.listdir(video_path))

        vid_fps = 1
        num_frames_of_video = len(frame_files)
    elif video_path.endswith('.gif'):
        gif_reader = imageio.get_reader(video_path)

        vid_fps = 25
        num_frames_of_video = len(gif_reader)
    else:
        vreader = VideoReader(video_path, ctx=cpu(0), num_threads=64)
        vid_fps = vreader.get_avg_fps()
        num_frames_of_video = len(vreader)

    f_start = 0                       if s is None else max(int(s * vid_fps) - 1, 0)
    f_end   = num_frames_of_video - 1 if e is None else min(int(e * vid_fps) - 1, num_frames_of_video - 1)
    frame_indices = list(range(f_start, f_end + 1))

    duration = len(frame_indices)
    max_frames = max_frames if max_frames is not None else MAX_FRAMES
    if fps is not None and duration / vid_fps < max_frames:
        sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='fps', vid_fps=vid_fps, fps=fps)]
    else:
        sampled_frame_indices = [frame_indices[i] for i in frame_sample(duration, mode='uniform', num_frames=max_frames)]

    if os.path.isdir(video_path):
        frames = []
        for frame_idx in sampled_frame_indices:
            filepath = os.path.join(video_path, frame_files[frame_idx])
            try:
                with Image.open(filepath).convert('RGB') as img:
                    frames.append(img)
            except Exception as e:
                print(f"Error with {filepath} with {e}")
                pass
    elif video_path.endswith('.gif'):
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for idx, frame in enumerate(gif_reader) if idx in sampled_frame_indices]
    else:
        frames = vreader.get_batch(sampled_frame_indices).asnumpy()

    timestamps = [x / vid_fps for x in sampled_frame_indices]

    if temporal_factor > 1:
        pad_length = temporal_factor - len(frames) % temporal_factor
        frames = np.concatenate([frames, frames[-1:].repeat(pad_length, axis=0)])
        [timestamps.append(timestamps[-1] + 1 / fps) for _ in range(pad_length)]

    return frames, timestamps


def load_video(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    size: Optional[int] = None,
    size_divisible: int = 1,
    temporal_factor: int = 1
):
    from decord import VideoReader, cpu  # type: ignore[import-not-found]

    if isinstance(video_path, list):
        video_path = video_path[0]
    if start_time is not None and end_time is not None and end_time - start_time < 1:
        return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
    if os.path.isdir(video_path):
        return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
    if video_path.endswith('.gif'):
        return load_video_from_ids(video_path, start_time, end_time, fps=fps, max_frames=max_frames)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not exist: {video_path}")

    vr = VideoReader(video_path, ctx=cpu(0))
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / video_fps

    start_frame = 0
    end_frame = total_frames
    if start_time is not None:
        start_frame = int(start_time * video_fps)
        start_frame = max(0, min(start_frame, total_frames - 1))
    if end_time is not None:
        end_frame = int(end_time * video_fps)
        end_frame = max(start_frame, min(end_frame, total_frames))

    frame_indices = list(range(start_frame, end_frame))
    if fps is not None:
        target_frame_rate = fps
        frame_step = max(1, int(video_fps / target_frame_rate))
        frame_indices = frame_indices[::frame_step]

    if max_frames is not None and len(frame_indices) > max_frames:
        frame_indices = np.linspace(start_frame, end_frame - 1, max_frames, dtype=int)

    frames = vr.get_batch(frame_indices).asnumpy()

    if size is not None:
        h, w = frames.shape[1], frames.shape[2]
        scale_factor = size / min(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        new_h = new_h // size_divisible * size_divisible
        new_w = new_w // size_divisible * size_divisible

        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (new_w, new_h))
            resized_frames.append(resized_frame)
        frames = np.array(resized_frames)

    timestamps = [i / video_fps for i in frame_indices]

    if temporal_factor > 1:
        pad_length = temporal_factor - len(frames) % temporal_factor
        frames = np.concatenate([frames, np.repeat(frames[-1:], pad_length, axis=0)])
        timestamps.extend([timestamps[-1] + 1 / video_fps] * pad_length)

    return frames, timestamps


def process_video(video_path, processor, s=None, e=None, aspect_ratio='avt', num_frames=None):
    fps = 1 if num_frames is None else None
    frames, timestamps = load_video(video_path, s, e, fps=fps, max_frames=num_frames)

    assert len(frames) == len(timestamps), "Number of frames and timestamps must match."

    if aspect_ratio == 'pad':
        frames = [expand2square(f, tuple(int(x*255) for x in processor.image_mean)) for f in frames]

    if aspect_ratio == 'avt':
        frames = [processor.preprocess(frame, return_tensors='pt', image_num=len(frames)) for frame in frames]
        grid_frames = [frames]
    else:
        frames = processor.preprocess(frames, return_tensors='pt', image_num=len(frames))
        grid_frames = [[frames]]

    return grid_frames, timestamps


def tokenizer_multimodal_token(prompt, tokenizer, multimodal_token=DEFAULT_IMAGE_TOKEN, return_tensors=None):
    multimodal_token_index = MODAL_INDEX_MAP.get(multimodal_token, None)
    if multimodal_token_index is None:
        input_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    else:
        prompt_chunks = [tokenizer(chunk, add_special_tokens=False).input_ids for idx, chunk in enumerate(prompt.split(multimodal_token))]

        input_ids = []
        for i in range(1, 2 * len(prompt_chunks)):
            if i % 2 == 1:
                input_ids.extend(prompt_chunks[i // 2])
            else:
                input_ids.append(multimodal_token_index)

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)



# ---------------------------------------------------------------------------
# Inlined from hulumed/processor
# ---------------------------------------------------------------------------

# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import math
import warnings
from typing import List, Union, Dict, Optional

import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput



DEFAULT_CHAT_TEMPLATE = """
{%- set identifier = 'im' %}
{% for message in messages %}
    {% if message['role'] == 'stream' %}
        {% set identifier = 'stream' %}
    {% else %}
        {% set identifier = 'im' %}
    {% endif %}
    {{- '<|' + identifier + '_start|>' + message['role'] + '\n' -}}
    {% if message['content'] is string %}
        {{- message['content'] + '<|' + identifier + '_end|>\n' -}}
    {% else %}
        {% for content in message['content'] %}
            {% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}
                {% if 'time' in content %}
                    {{- 'Time ' + content['time'] | round(1) | string + 's: ' -}}
                {% endif %}
"""
DEFAULT_CHAT_TEMPLATE += """
                {{- '%s\n' -}}
""" % DEFAULT_IMAGE_TOKEN
DEFAULT_CHAT_TEMPLATE += """
            {% elif content['type'] == 'video' or 'video' in content or 'video_url' in content %}
                {% for i in range(content['num_frames']) %}
                    {% if 'timestamps' in content %}
                        {{- 'Time ' + content['timestamps'][i] | round(1) | string + 's:' -}}
                    {% endif %}
                    {% if i < content['num_frames'] - 1 %}
"""
DEFAULT_CHAT_TEMPLATE += """
                        {{- '%s,' -}}
""" % DEFAULT_IMAGE_TOKEN
DEFAULT_CHAT_TEMPLATE += """
                    {% else %}
"""
DEFAULT_CHAT_TEMPLATE += """
                        {{- '%s\n' -}}
""" % DEFAULT_IMAGE_TOKEN
DEFAULT_CHAT_TEMPLATE += """
                    {% endif %}
                {% endfor %}
            {% elif content['type'] == 'text' or 'text' in content %}
                {{- content['text'] -}}
            {% endif %}
        {% endfor %}
        {{- '<|' + identifier + '_end|>\n' -}}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' -}}
{% endif %}
"""


class HulumedProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class HulumedProcessor(ProcessorMixin):
    r"""
    Modified from Qwen2VLProcessor
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "Qwen2VLImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        if chat_template is None:
            chat_template = DEFAULT_CHAT_TEMPLATE
        tokenizer.chat_template = chat_template
        self.chat_template = chat_template
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.generation_prompt = self._infer_generation_prompt()
        self.generation_prompt_ids = self.tokenizer.encode(self.generation_prompt, return_tensors="pt")
        self.generation_prompt_length = len(self.generation_prompt_ids[0])
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.eos_token_id = self.tokenizer.eos_token_id

    def get_generation_prompt(self):
        return self.generation_prompt

    def get_generation_prompt_ids(self):
        return self.generation_prompt_ids

    def load_video(self, *args, **kwargs):
        return load_video(*args, **kwargs)

    def load_images(self, *args, **kwargs):
        return load_images(*args, **kwargs)

    def _infer_generation_prompt(self):
        pseudo_message = [{"role": "user", "content": ""}]
        instruction = self.tokenizer.apply_chat_template(pseudo_message, tokenize=False, add_generation_prompt=True)
        conversation = self.tokenizer.apply_chat_template(pseudo_message, tokenize=False, add_generation_prompt=False)
        return instruction.replace(conversation, "")

    def _process_text_with_label(
        self,
        text: List[Dict],
        grid_sizes: torch.Tensor = None,
        **kwargs,
    ):
        assert kwargs.pop("return_tensors", "pt") == "pt", "Only PyTorch tensors are supported when return_labels=True."
        assert isinstance(text[0], dict), "When return_labels=True, text must be a list of messages."

        input_ids_list = []
        targets_list = []
        sample_types_list = []
        image_idx = 0

        for message_idx, message in enumerate(text):
            prompt = self.tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False)
            prompt_chunks = prompt.split(DEFAULT_IMAGE_TOKEN)
            prompt = []
            for chunk_idx in range(len(prompt_chunks) - 1):
                prompt.append(prompt_chunks[chunk_idx])
                thw = grid_sizes[image_idx]
                prompt.append(DEFAULT_IMAGE_TOKEN * thw.prod().long())
                image_idx += 1
            prompt.append(prompt_chunks[-1])
            prompt = "".join(prompt)

            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0]
            input_ids_list.append(input_ids)

            targets = torch.full_like(input_ids, IGNORE_INDEX)
            sample_types = torch.full_like(input_ids, IGNORE_INDEX)
            if message["role"] == "assistant":
                # print("------input_ids:-------\n ", input_ids)
                # print("------prompt---:-------\n ", prompt)
                # print('------token string:-------\n ', self.tokenizer.convert_ids_to_tokens(input_ids))
                targets[self.generation_prompt_length:-1] = input_ids[self.generation_prompt_length:-1].clone()
                # print("------targets:-------\n ", targets)
            elif message["role"] == "stream":
                diff = torch.diff((input_ids == self.image_token_id).float())
                image_end_indices = torch.nonzero(diff < 0)[:, 0]
                targets[image_end_indices + 1] = input_ids[image_end_indices + 1]
                sample_types = targets.clone()
                sample_types[torch.logical_and(sample_types > 0, sample_types != self.eos_token_id)] = 0
                targets[-2] = input_ids[-2]

            targets_list.append(targets)
            sample_types_list.append(sample_types)
        assert len(grid_sizes) == image_idx, "Number of images does not match the number of image tokens in the text."

        targets = torch.cat(targets_list)
        sample_types = torch.cat(sample_types_list)
        types, counts = torch.unique(sample_types[sample_types > -1], return_counts=True)

        if len(types) > 0:
            target_num_samples = counts.amin()

            for type_id, type_count in zip(types, counts):
                if type_count > target_num_samples:
                    indices = torch.nonzero(sample_types == type_id)[:, 0]
                    random_selector = torch.randperm(indices.size(0))[:-target_num_samples]
                    targets[indices[random_selector]] = IGNORE_INDEX
                    sample_types[indices[random_selector]] = -1
        # print("------final targets:-------\n ", targets)
        # print('-------target shape:-------\n ', targets.shape)
        # print('--------------------input_ids shape:--------------\n ', torch.cat(input_ids_list).shape, 'targets shape: ', targets.shape)
        # mask = targets != IGNORE_INDEX
        # if mask.any():
        #     ids_to_compute = targets[mask].tolist()
        #     tokens_to_compute = self.tokenizer.convert_ids_to_tokens(ids_to_compute)
        #     decoded_str = self.tokenizer.decode(ids_to_compute, skip_special_tokens=False)
        #     print("[processor labels] 参与 loss 的 token 数:", mask.sum().item())
        #     print("[processor labels] token ids:", ids_to_compute[:50], "..." if len(ids_to_compute) > 50 else "")
        #     print("[processor labels] tokens:", tokens_to_compute[:50], "..." if len(tokens_to_compute) > 50 else "")
        #     print("[processor labels] decode 文本:", repr(decoded_str[:200]), "..." if len(decoded_str) > 200 else "")

        text_inputs = {
            "input_ids": torch.cat(input_ids_list),
            "labels": targets,
        }

        return text_inputs

    def _process_text_without_label(
        self,
        text: Union[List[str], List[Dict]],
        grid_sizes: torch.Tensor = None,
        **kwargs,
    ):
        if isinstance(text[0], dict):
            warnings.warn("Input text is a list of messages. Automatically convert it to a string with 'apply_chat_template' with generation prompt.")
            text = [self.tokenizer.apply_chat_template(text, tokenize=False, add_generation_prompt=True)]

        image_idx = 0
        for i in range(len(text)):
            while DEFAULT_IMAGE_TOKEN in text[i]:
                thw = grid_sizes[image_idx]
                text[i] = text[i].replace(DEFAULT_IMAGE_TOKEN, "<placeholder>" * thw.prod().long(), 1)
                image_idx += 1
            text[i] = text[i].replace("<placeholder>", DEFAULT_IMAGE_TOKEN)
        assert len(grid_sizes) == image_idx, "Number of images does not match the number of image tokens in the text."

        text_inputs = self.tokenizer(text, **kwargs)
        return text_inputs

    def process_text(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], List[Dict]],
        image_inputs: Dict[str, torch.Tensor] = {},
        return_labels: bool = False,
        **kwargs,
    ):
        kwargs.pop("padding", None)
        kwargs.pop("padding_side", None)

        if not isinstance(text, (list, tuple)):
            text = [text]
        assert len(text), "At least one text must be provided."

        grid_sizes = []
        for grid_size, merge_size in zip(image_inputs.get("grid_sizes", []), image_inputs.get("merge_sizes", [])):
            if not torch.all(grid_size[1:] % merge_size == 0):
                warnings.warn(f"Grid size {grid_size} is not divisible by merge size. Some undesired errors may occur.")
            if grid_size[0] == 1:
                grid_sizes.append(grid_size[1:] / merge_size)
            elif grid_size[0] > 1:
                grid_sizes.extend([grid_size[1:] / merge_size] * grid_size[0])

        if return_labels:
            return self._process_text_with_label(text, grid_sizes, **kwargs)
        return self._process_text_without_label(text, grid_sizes, **kwargs)

    def process_images(
        self,
        images: ImageInput = None,
        merge_size: Optional[int] = 1,
        **kwargs,
    ):
        if images is None:
            return {}
        image_inputs = self.image_processor(images=images, merge_size=merge_size, **kwargs)
        return image_inputs

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput], List[Dict]] = None,
        images: ImageInput = None,
        merge_size: Optional[int] = 1,
        return_labels: bool = False,
        **kwargs: Unpack[HulumedProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            HulumedProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        output_kwargs["text_kwargs"].pop("padding")
        output_kwargs["text_kwargs"].pop("padding_side")

        image_inputs = self.process_images(images, merge_size, **output_kwargs["images_kwargs"])
        text_inputs = self.process_text(text, image_inputs, return_labels, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))



# ---------------------------------------------------------------------------
# Original HuluMed inference entrypoint
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hulu-Med Qwen3 inference for MIMIC-CXR multi-image samples"
    )
    parser.add_argument("--model-path", default=None, help="本地模型目录")
    parser.add_argument("--json-path", default=None, help="输入 JSON 文件")
    parser.add_argument("--image-root", default=None, help="当 JSON 中是相对路径时使用")
    parser.add_argument("--output-path", default=None, help="输出结果 JSON")
    parser.add_argument("--device", default="cuda:0", help='推理设备，如 "cuda:0"')
    parser.add_argument("--attn-implementation", default="sdpa", help='如 "sdpa" 或 "flash_attention_2"')
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.6, help="采样温度")
    parser.add_argument("--do-sample", action="store_true", help="开启采样生成")
    parser.add_argument("--max-samples", type=int, default=None, help="仅调试前 N 条")
    return parser.parse_args()


def normalize_path(path_value, image_root):
    if not isinstance(path_value, str):
        raise ValueError(f"路径字段必须是字符串，当前收到: {type(path_value)}")
    if os.path.isabs(path_value):
        return path_value
    if image_root:
        return os.path.join(image_root, path_value)
    return path_value


def resolve_image_paths(item, image_root):
    if "images" in item:
        value = item["images"]
    elif "image" in item:
        value = item["image"]
    else:
        raise KeyError("未找到图像字段，期望 image/images 之一")

    if isinstance(value, list):
        if not value:
            raise ValueError("images 为空列表")
        return [normalize_path(v, image_root) for v in value]
    return [normalize_path(value, image_root)]


def parse_question(item):
    messages = item.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    for token in ("<image>\n", "<image>", "<video>\n", "<video>", "<3d>\n", "<3d>"):
                        content = content.replace(token, "")
                    text = content.strip()
                    if text:
                        return text
                break
    return item.get("text", "Generate a medical report for the given chest X-ray image(s).")


def build_conversation(question, num_images):
    content = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def load_model_and_processor(model_path, device, attn_implementation):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    processor = HulumedProcessor(image_processor, tokenizer)
    model.config.use_token_compression = False
    model = model.to(device)
    return tokenizer, model, processor


def generate_one_case(model, tokenizer, processor, images, conversation, device, do_sample, temperature, max_new_tokens):
    modal = "image"
    text_input = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        images=[images],
        text=text_input,
        merge_size=1,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    generation_kwargs = {
        "do_sample": do_sample,
        "modals": [modal],
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs.update(
            {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 50,
            }
        )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def replace_assistant_content(item, output_text):
    result = copy.deepcopy(item)
    messages = result.get("messages")

    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "assistant":
                message["content"] = output_text
                return result
        messages.append({"role": "assistant", "content": output_text})
        return result

    return {"messages": [{"role": "assistant", "content": output_text}]}


def main():
    args = parse_args()
    tokenizer, model, processor = load_model_and_processor(
        args.model_path,
        args.device,
        args.attn_implementation,
    )

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples is not None:
        data = data[: args.max_samples]

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results = []
    for idx, item in enumerate(data, start=1):
        try:
            image_paths = resolve_image_paths(item, args.image_root)
            question = parse_question(item)
        except Exception as exc:
            print(f"[{idx}/{len(data)}] 跳过，样本解析失败: {exc}")
            continue

        missing_paths = [p for p in image_paths if not os.path.exists(p)]
        if missing_paths:
            print(f"[{idx}/{len(data)}] 跳过，文件不存在: {missing_paths[0]}")
            continue

        print(f"[{idx}/{len(data)}] 正在处理: {len(image_paths)} 张图")
        images = load_images(image_paths)
        if not images:
            print(f"[{idx}/{len(data)}] 跳过，图像加载失败: {image_paths[:1]}")
            continue

        conversation = build_conversation(question, len(images))
        output_text = generate_one_case(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            images=images,
            conversation=conversation,
            device=args.device,
            do_sample=args.do_sample,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )

        result = replace_assistant_content(item, output_text)
        results.append(result)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"推理完成，结果已保存到: {args.output_path}")


if __name__ == "__main__":
    main()
