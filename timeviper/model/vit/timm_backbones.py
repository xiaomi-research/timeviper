"""
TIMM-based ViT backbones that load weights from local checkpoints.
This module now mirrors the structure used by the InternVideo2 backbone
for a consistent code style (config + tower + backbone wrapper).
"""

from timeviper.model.vit.base_vision import TimmViTBackbone
from timeviper.model.vit.registry import get_vision_backbone_config
from timeviper.utils.train_utils import _load_checkpoint


class TimmCheckpointBackbone(TimmViTBackbone):
    """Unified TIMM ViT backbone that loads weights from local checkpoints."""

    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = None,
        use_zero3: bool = False,
    ) -> None:
        self.cfg = get_vision_backbone_config(vision_backbone_id)
        if self.cfg["type"] != "timm":
            raise ValueError(
                f"Backbone {vision_backbone_id} is not a TIMM backbone config!"
            )

        effective_image_size = default_image_size or self.cfg["default_image_size"]

        super().__init__(
            vision_backbone_id,
            self.cfg["timm_id"],
            image_resize_strategy,
            default_image_size=effective_image_size,
            use_zero3=use_zero3,
        )
        self.load_model_weights(use_zero3)

    def load_model_weights(self, use_zero3: bool) -> None:
        _load_checkpoint(self.featurizer, self.cfg, use_zero3)
        print(f"Loaded pretrained {self.identifier} from {self.cfg['ckpt_path']}")

    @property
    def get_identifier(self) -> str:
        return self.cfg.get("identifier", self.identifier)
