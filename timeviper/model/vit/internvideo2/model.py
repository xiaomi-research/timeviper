"""
internvideo2_modules/model.py

InternVideo2 Vision Tower implementation and Image Processor.
"""

import os
from dataclasses import dataclass
from functools import partial
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from transformers.image_processing_utils import get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)

from .vit_scale_clean import (
    PretrainVisionTransformer_clean,
    interpolate_pos_embed_internvideo2,
)


# === InternVideo2 Specific Processor ===
class InternVideo2ImageProcessor:
    def __init__(
        self,
        image_mean=(0.485, 0.456, 0.406),
        image_std=(0.229, 0.224, 0.225),
        size=(224, 224),
        crop_size: Dict[str, int] = None,
        resample=PILImageResampling.BICUBIC,
        rescale_factor=1 / 255,
        data_format=ChannelDimension.FIRST,
    ):
        crop_size = (
            crop_size
            if crop_size is not None
            else {"height": size[0], "width": size[1]}
        )
        crop_size = get_size_dict(
            crop_size, default_to_square=True, param_name="crop_size"
        )

        self.image_mean = image_mean
        self.image_std = image_std
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.data_format = data_format
        self.crop_size = crop_size

    def preprocess(self, images, return_tensors, target_size=None):
        if isinstance(images, Image.Image):
            images = [images]

        # This part is crucial for handling video frames passed as a list of PIL Images
        if isinstance(images, list) and all(isinstance(i, Image.Image) for i in images):
            pass  # Already in the correct list-of-PIL format
        else:
            images = [to_numpy_array(image) for image in images]
            assert isinstance(images, list)

        if target_size is None:
            target_size = self.size

        transforms = [
            convert_to_rgb,
            partial(to_numpy_array),  # PIL -> numpy
            partial(
                resize,
                size=target_size,
                resample=self.resample,
                data_format=self.data_format,
            ),
            partial(to_numpy_array),  # convert back to numpy after resize
            partial(rescale, scale=self.rescale_factor, data_format=self.data_format),
            partial(
                normalize,
                mean=self.image_mean,
                std=self.image_std,
                data_format=self.data_format,
            ),
            partial(
                to_channel_dimension_format,
                channel_dim=self.data_format,
                input_channel_dim=self.data_format,
            ),
        ]

        processed_images = []
        for img in images:
            proc_img = img
            for f in transforms:
                proc_img = f(proc_img)
            processed_images.append(torch.from_numpy(proc_img))

        # Stack to create the final tensor -> (T, C, H, W) for video, (1, C, H, W) for image
        final_tensor = torch.stack(processed_images)

        # For video, the processor in timeviper expects (B, T, C, H, W). We return (T, C, H, W) and batching is handled by collator.
        # For image, we return (1, C, H, W)
        return final_tensor


@dataclass
class InternVideo2VisionConfig:
    """Model hyperparameters for InternVideo2 vision tower."""

    num_frames: int = 4
    hidden_size: int = 1408
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 14
    x_vis_return_idx: int = -2
    sep_image_video_pos_embed: bool = True
    use_checkpoint: bool = True
    checkpoint_num: int = 40
    vision_tower_path: str = "./ckpts/InternVideo2-1B_f4_vision.pt"


# === InternVideo2 Main Model Wrapper (Featurizer) ===
class InternVideo2VisionTower(nn.Module):
    def __init__(self, config: InternVideo2VisionConfig):
        super().__init__()
        self.config = config
        self.vision_tower = self._build_model()
        self.vision_tower.requires_grad_(False)

    def _build_model(self):
        model = PretrainVisionTransformer_clean(
            in_chans=self.config.num_channels,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            embed_dim=self.config.hidden_size,
            depth=self.config.num_hidden_layers,
            num_heads=self.config.num_attention_heads,
            mlp_ratio=48 / 11,
            attn_pool_num_heads=16,
            qkv_bias=False,
            drop_path_rate=0.25,
            init_values=0.00001,
            qk_normalization=True,
            use_flash_attn=True,
            use_fused_rmsnorm=False,
            use_fused_mlp=False,
            fused_mlp_heuristic=1,
            layerscale_no_force_fp32=False,
            num_frames=self.config.num_frames,
            tubelet_size=1,
            sep_pos_embed=False,
            sep_image_video_pos_embed=self.config.sep_image_video_pos_embed,
            use_checkpoint=self.config.use_checkpoint,
            checkpoint_num=self.config.checkpoint_num,
            x_vis_return_idx=self.config.x_vis_return_idx,
            x_vis_only=True,
        )
        return model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (B, T, C, H, W)
        # InternVideo2 expects: (B, C, T, H, W)
        # The forward pass also handles single image case (T=1)
        # import pdb; pdb.set_trace()
        # is_video = pixel_values.ndim == 5
        is_video = pixel_values.shape[1] > 1
        # if not is_video:  # Single image: (B, C, H, W) -> (B, 1, C, H, W)
        # pixel_values = pixel_values.unsqueeze(1)

        # B, T, C, H, W = pixel_values.shape
        # pixel_values = pixel_values.permute(0, 2, 1, 3, 4) # B, C, T, H, W
        # T, B, C, H, W = pixel_values.shape
        B, T, C, H, W = pixel_values.shape
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
        # B, C, T, H, W --> B * T // 4, C, 4, H, W
        if is_video:
            pixel_values = pixel_values.reshape(B * (T // 4), C, 4, H, W)
        # The model returns patch features, excluding the CLS token
        image_embeds = self.vision_tower(pixel_values, use_image=(T == 1))

        # Output: (B, T*num_patches_per_frame, D)
        return image_embeds[:, 1:, :]

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device
