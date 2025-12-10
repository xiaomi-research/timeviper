"""
nn_utils.py

Utility functions and PyTorch submodule definitions.
"""

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
