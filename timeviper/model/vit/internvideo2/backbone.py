"""
internvideo2_vit.py

Vision Backbone for InternVideo2 model.
"""

import os
from functools import partial
from typing import Callable, Tuple

import torch
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy

from timeviper.model.vit.base_vision import ImageTransform, VisionBackbone
from timeviper.model.vit.registry import get_vision_backbone_config
from timeviper.utils.train_utils import _load_checkpoint, load_ckpt_from_deepspeed_zero

from .model import (
    InternVideo2ImageProcessor,
    InternVideo2VisionConfig,
    InternVideo2VisionTower,
)
from .vit_scale_clean import Block, interpolate_pos_embed_internvideo2


class InternVideo2ViTBackbone(VisionBackbone):
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        num_frames: int = 4,
        use_zero3: bool = False,
    ) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size)

        self.spec = get_vision_backbone_config(vision_backbone_id)
        if self.spec["type"] != "internvideo2":
            raise ValueError(
                f"Backbone {vision_backbone_id} is not an InternVideo2 backbone config!"
            )

        # Instantiate model config
        self.vision_config = InternVideo2VisionConfig(
            image_size=default_image_size,
            num_frames=num_frames or self.spec.get("num_frames", 4),
            vision_tower_path=self.spec["vision_tower_path"],
        )

        # Instantiate the featurizer (the actual model)
        self.featurizer = InternVideo2VisionTower(self.vision_config)
        self.featurizer.eval()

        # Instantiate the image transform
        self.processor = InternVideo2ImageProcessor(
            size=(default_image_size, default_image_size),
        )
        self.image_transform = self.processor.preprocess
        self.dtype = torch.bfloat16

        self.load_model_weights(use_zero3)

    def load_model_weights(self, use_zero3: bool = False):
        ckpt_path = self.featurizer.config.vision_tower_path
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"InternVideo2 checkpoint not found at {ckpt_path}. Please download it."
            )

        state_dict = torch.load(ckpt_path, map_location="cpu")

        if self.featurizer.config.image_size != 224:
            interpolate_pos_embed_internvideo2(
                state_dict,
                self.featurizer.vision_tower,
                orig_t_size=self.featurizer.config.num_frames,
            )

        if use_zero3:
            load_ckpt_from_deepspeed_zero(
                self.featurizer.vision_tower, state_dict, strict=False
            )
        else:
            message = self.featurizer.vision_tower.load_state_dict(
                state_dict, strict=False
            )
            print(
                f"Loaded InternVideo2 weights with message: {message}. \n It is common that blocks.0.lsx.weight, clip_decoder.x.norm.xx... are not used"
            )

    def get_image_transform(self) -> ImageTransform:
        # The transform needs to know the target tensor type
        return partial(self.image_transform, return_tensors="pt")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP policy that wraps each Transformer block and then the entire featurizer."""
        intern_video_wrap_policy = partial(
            _module_wrap_policy, module_classes={InternVideo2VisionTower}
        )
        transformer_block_policy = partial(_module_wrap_policy, module_classes={Block})
        return partial(
            _or_policy, policies=[intern_video_wrap_policy, transformer_block_policy]
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the featurizer."""
        return self.featurizer(pixel_values)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (3, self.default_image_size, self.default_image_size)

    @property
    def embed_dim(self) -> int:
        return self.vision_config.hidden_size

    @property
    def num_patches(self) -> int:
        # Return num_patches for a single frame, as the VLM handles sequence length dynamically
        return (self.vision_config.image_size // self.vision_config.patch_size) ** 2

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return self.dtype

    @property
    def get_identifier(self) -> str:
        return "internvideo2"
