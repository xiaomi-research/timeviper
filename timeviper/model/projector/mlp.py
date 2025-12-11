"""
nn_utils.py

Utility functions and PyTorch submodule definitions.
"""

from typing import Dict

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    def __init__(
        self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu_mlp"
    ) -> None:
        super().__init__()
        if mlp_type == "gelu_mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type}` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class MultiMLPProjector(nn.Module):
    """
    MoF feature interleaving
    """

    def __init__(
        self, vision_dims: Dict[str, int], llm_dim: int, mlp_type: str = "gelu_mlp"
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict()
        self.keys = list(vision_dims.keys())
        for key, dim in vision_dims.items():
            if mlp_type == "gelu_mlp":
                self.projectors[key] = MLPProjector(dim, llm_dim, mlp_type)
            else:
                raise ValueError(f"Projector with `{mlp_type}` is not supported!")

    def forward(self, img_patches: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = []
        for key in self.keys:
            outputs.append(self.projectors[key](img_patches[key]))

        if len(outputs) == 2:
            if (
                outputs[0].shape != outputs[1].shape
                and outputs[0].numel() == outputs[1].numel()
            ):
                if outputs[0].shape[0] > outputs[1].shape[0]:
                    outputs[1] = outputs[1].reshape(outputs[0].shape)
                else:
                    outputs[0] = outputs[0].reshape(outputs[1].shape)

        if outputs[0].shape[1] != outputs[1].shape[1]:
            # Use sequential stacking due to different sequence lengths
            return torch.cat(outputs, dim=1)
        else:
            # Use stacking and flattening for same sequence lengths
            return torch.stack(outputs, dim=2).flatten(1, 2)
