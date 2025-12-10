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
registry.py

Registry for Vision Backbones configurations.
"""

from typing import Any, Dict

VISION_MODEL_REGISTRY = {
    "siglip": {
        "type": "timm",
        "variants": {
            "siglip-vit-b16-224px": {
                "timm_id": "vit_base_patch16_siglip_224",
                "ckpt_path": "./ckpts/vit_base_patch16_siglip_224.pth",
                "default_image_size": 224,
            },
            "siglip-vit-b16-256px": {
                "timm_id": "vit_base_patch16_siglip_256",
                "ckpt_path": "./ckpts/vit_base_patch16_siglip_256.pth",
                "default_image_size": 256,
            },
            "siglip-vit-b16-384px": {
                "timm_id": "vit_base_patch16_siglip_384",
                "ckpt_path": "./ckpts/vit_base_patch16_siglip_384.pth",
                "default_image_size": 384,
            },
            "siglip-vit-so400m": {
                "timm_id": "vit_so400m_patch14_siglip_224",
                "ckpt_path": "./ckpts/vit_so400m_patch14_siglip_224.pth",
                "default_image_size": 224,
            },
            "siglip-vit-so400m-384px": {
                "timm_id": "vit_so400m_patch14_siglip_384",
                "ckpt_path": "./ckpts/vit_so400m_patch14_siglip_384.pth",
                "default_image_size": 384,
            },
        },
    },
    "dinov2": {
        "type": "timm",
        "variants": {
            "dinov2-vit-l": {
                "timm_id": "vit_large_patch14_reg4_dinov2.lvd142m",
                "ckpt_path": "./ckpts/vit_large_patch14_reg4_dinov2.lvd142m",
                "default_image_size": 224,
            },
        },
    },
    "internvideo2": {
        "type": "internvideo2",
        "variants": {
            "internvideo2-1b-16-224px": {
                "default_image_size": 224,
                "num_frames": 4,
                "vision_tower_path": "./ckpts/InternVideo2-1B_f4_vision.pt",
            },
        },
    },
}


def get_vision_backbone_config(vision_backbone_id: str) -> Dict[str, Any]:
    for family, config in VISION_MODEL_REGISTRY.items():
        if vision_backbone_id in config["variants"]:
            variant_config = config["variants"][vision_backbone_id]
            # Merge family config (excluding variants) with variant config
            family_config = {k: v for k, v in config.items() if k != "variants"}
            final_config = {
                **family_config,
                **variant_config,
                "vision_family": family,
                "identifier": family,
            }
            return final_config
    raise ValueError(f"Vision Backbone `{vision_backbone_id}` is not supported!")
