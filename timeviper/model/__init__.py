#    Copyright 2025 Renmin University of China and Xiaomi Corporation.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
materialize.py

Factory class for initializing Vision Backbones, LLM Backbones, and VLMs from a set registry; provides and exports
individual functions for clear control flow.
"""
from typing import Optional, Tuple

from transformers import PreTrainedTokenizerBase

from timeviper.model.llm.base_llm import LLMBackbone
from timeviper.model.llm.llm_factory import LLMBackboneFactory
from timeviper.model.vit import (
    ImageTransform,
    InternVideo2ViTBackbone,
    TimmCheckpointBackbone,
    VisionBackbone,
)
from timeviper.model.vit.registry import get_vision_backbone_config

from .generic_vlm import GenericTimeViperVLM
from .hybrid_vlm import HybridTimeViperVLM


def get_vision_backbone_and_transform(
    vision_backbone_id: str, image_resize_strategy: str, use_zero3: bool = False
) -> Tuple[VisionBackbone, ImageTransform]:
    """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform."""
    vision_cfg = get_vision_backbone_config(vision_backbone_id)

    if vision_cfg["type"] == "timm":
        vision_backbone = TimmCheckpointBackbone(
            vision_backbone_id, image_resize_strategy, use_zero3=use_zero3
        )
    elif vision_cfg["type"] == "internvideo2":
        vision_backbone = InternVideo2ViTBackbone(
            vision_backbone_id, image_resize_strategy, use_zero3=use_zero3
        )
    else:
        raise ValueError(
            f"Vision Backbone type `{vision_cfg['type']}` is not supported!"
        )

    image_transform = vision_backbone.get_image_transform()
    return vision_backbone, image_transform


def get_llm_backbone_and_tokenizer(
    llm_backbone_id: str,
    llm_max_length: int = None,
    hf_token: Optional[str] = None,
    inference_mode: bool = False,
    attn_implementation: str = "flash_attention_2",
    continue_pretrain_ckpt: Optional[str] = None,
    merge_module: Optional[str] = "no_merge",
    use_pdrop: bool = False,
    pdrop_type: Optional[str] = None,
) -> Tuple[LLMBackbone, PreTrainedTokenizerBase]:
    assert merge_module in [
        "no_merge",
        "CrossAttention",
    ], f"merge_module must be one of None, 'CrossAttention', got {merge_module}"
    try:
        llm_backbone = LLMBackboneFactory(
            llm_backbone_id=llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            attn_implementation=attn_implementation,
            continue_pretrain_ckpt=continue_pretrain_ckpt,
            merge_module=merge_module,
            use_pdrop=use_pdrop,
            pdrop_type=pdrop_type,
        )
        tokenizer = llm_backbone.get_tokenizer()
        return llm_backbone, tokenizer
    except ValueError as e:
        raise ValueError(
            f"LLM Backbone `{llm_backbone_id}` is not supported or not found in the registry!"
        ) from e


def get_vlm(
    model_id: str,
    arch_specifier: str,
    vision_backbone: VisionBackbone,
    llm_backbone: LLMBackbone,
    enable_mixed_precision_training: bool = True,
    visual_token_order="raw",
):
    """Lightweight wrapper around initializing a VLM, mostly for future-proofing (if one wants to add a new VLM)."""
    family = llm_backbone.llm_family
    if family in ["nano"]:
        return HybridTimeViperVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            visual_token_order=visual_token_order,
        )
    elif family in ["qwen2"]:
        return GenericTimeViperVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            visual_token_order=visual_token_order,
        )
    else:
        raise ValueError(
            f"No VLM is configured for the LLM family `{family}` (from backbone `{llm_backbone.identifier}`)!"
        )
