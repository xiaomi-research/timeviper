# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from typing import Callable, Dict, Tuple, Union

import torch
import torch.nn as nn


def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).
    """
    protected = 0

    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    assert r > 0, r

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_add(-2, dst_idx.expand(n, r, c), src)  # , reduce=mode)

        return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge


def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size


class ToMe16_mlp_hd64(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        mlp_type: str = "tome_mlp",
        num_compressed_tokens: int = 16,
        token_order: str = "raw",
    ) -> None:
        super().__init__()
        # self.num_attention_heads = vision_cfg.num_attention_heads
        self.num_attention_heads = 16  # default is 16
        self.num_compressed_tokens = num_compressed_tokens
        if mlp_type == "tome_mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        elif mlp_type == "fused_tome_mlp":
            self.initial_projection_dim = vision_dim * 4
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, self.initial_projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Fused Projector with `{mlp_type}` is not supported!")
        self.token_order = token_order

    def merge_tokens(self, x, target_num_token, token_order):
        r"""
        x = torch.randn(10, 2560, c)
        x = merge_tokens(x, r_merge_list=[1280])
        """
        size = None
        b, p, c = x.shape
        tmp_p = p
        r_merge_list = []
        assert (
            tmp_p > target_num_token
        ), f"{tmp_p} should greater than {target_num_token}"
        while tmp_p != target_num_token:
            if tmp_p - target_num_token <= (tmp_p // 2):
                r_merge_list.append(tmp_p - target_num_token)
                break
            else:
                r_merge_list.append(tmp_p // 2)
                tmp_p = tmp_p - (tmp_p // 2)

        head = self.num_attention_heads

        dim = c // head
        for r in r_merge_list:
            metric = x.reshape(b, p, head, dim).mean(2)  # [b, p, c//head]
            merge, _ = bipartite_soft_matching(metric, r)
            x, size = merge_wavg(merge, x, size)
            _, p, _ = x.shape
        # token reordering
        if token_order in ["ascending", "descending"]:
            descending = token_order == "descending"
            sort_idx = size.squeeze(-1).argsort(dim=1, descending=descending)
            x = x.gather(dim=1, index=sort_idx.unsqueeze(-1).expand(-1, -1, c))
            size = size.gather(dim=1, index=sort_idx.unsqueeze(-1))
        return x

    def forward(self, x, compress=False, local_num_frames=-1):
        dtype = x.dtype
        device = x.device
        if local_num_frames != -1 and local_num_frames != 1:
            assert compress is True
        if compress:
            if local_num_frames != -1:
                num_frames = local_num_frames
                x = x.reshape(x.shape[0], -1, x.shape[-1])
            else:
                num_frames = x.shape[0]
                x = x.reshape(1, -1, x.shape[-1])
            num_tome_tokens = (
                self.num_compressed_tokens * num_frames
            )  # compress to 16 tokens / frame
        else:
            num_tome_tokens = self.num_compressed_tokens * local_num_frames

        x = self.merge_tokens(
            x, target_num_token=num_tome_tokens, token_order=self.token_order
        )
        x = self.projector(x)

        return x


class MultiToMe16_mlp_hd64(nn.Module):
    def __init__(
        self,
        vision_dims: Dict[str, int],
        llm_dim: int,
        mlp_type: str = "tome_mlp",
        num_compressed_tokens: int = 16,
        token_order: str = "raw",
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict()
        self.keys = list(vision_dims.keys())
        for key, dim in vision_dims.items():
            if "tome_mlp" in mlp_type:
                self.projectors[key] = ToMe16_mlp_hd64(
                    dim, llm_dim, mlp_type, num_compressed_tokens, token_order
                )
            else:
                raise ValueError(f"Projector with `{mlp_type}` is not supported!")

    def forward(
        self,
        img_patches: Dict[str, torch.Tensor],
        compress=False,
        local_num_frames: Union[int, Dict[str, int]] = -1,
    ) -> torch.Tensor:
        outputs = []
        for key in self.keys:
            lnf = local_num_frames
            if isinstance(local_num_frames, dict):
                lnf = local_num_frames.get(key, -1)
            outputs.append(
                self.projectors[key](
                    img_patches[key], compress=compress, local_num_frames=lnf
                )
            )

        # two encoders
        if len(outputs) == 2:
            # internvideo2 + another vision encoder
            if (
                outputs[0].shape != outputs[1].shape
                and outputs[0].numel() == outputs[1].numel()
            ):
                if outputs[0].shape[0] > outputs[1].shape[0]:
                    outputs[1] = outputs[1].reshape(outputs[0].shape)
                else:
                    outputs[0] = outputs[0].reshape(outputs[1].shape)
        if outputs[0].shape[1] != outputs[1].shape[1]:
            return torch.cat(outputs, dim=1)
        else:
            return torch.stack(outputs, dim=2).flatten(1, 2)
