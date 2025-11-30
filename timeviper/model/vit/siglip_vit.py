"""
siglip_vit.py
"""

import torch

from timeviper.model.vit.base_vision import TimmViTBackbone
from timeviper.utils.train_utils import load_ckpt_from_deepspeed_zero

# Registry =>> Supported SigLIP Vision Backbones (from TIMM) =>> Note:: Using SigLIP w/ Patch = 14 (but SO400M Arch)
SIGLIP_VISION_BACKBONES = {
    "siglip-vit-b16-224px": "vit_base_patch16_siglip_224",
    "siglip-vit-b16-256px": "vit_base_patch16_siglip_256",
    "siglip-vit-b16-384px": "vit_base_patch16_siglip_384",
    "siglip-vit-so400m": "vit_so400m_patch14_siglip_224",
    "siglip-vit-so400m-384px": "vit_so400m_patch14_siglip_384",
}


class SigLIPViTBackbone(TimmViTBackbone):
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        use_zero3: bool = False,
    ) -> None:
        super().__init__(
            vision_backbone_id,
            SIGLIP_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
            use_zero3=use_zero3,
        )
        ckpt = torch.load(
            f"./ckpts/{SIGLIP_VISION_BACKBONES[vision_backbone_id]}.pth",
            map_location="cpu",
        )
        if use_zero3:
            load_ckpt_from_deepspeed_zero(self.featurizer, ckpt, strict=True)
        else:
            self.featurizer.load_state_dict(
                ckpt,
                strict=True,
            )
        print(
            f"Loaded pretrained SigLIP ViT Backbone from ./ckpts/{SIGLIP_VISION_BACKBONES[vision_backbone_id]}.pth"
        )

    @property
    def get_identifier(self) -> str:
        return "siglip"
