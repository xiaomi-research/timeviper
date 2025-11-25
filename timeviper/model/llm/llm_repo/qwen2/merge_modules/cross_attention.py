import math
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

if is_flash_attn_2_available():
    import inspect

    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters
    )

    from transformers.modeling_flash_attention_utils import (  # _flash_attention_forward,
        _upad_input,
    )
else:
    flash_attn_varlen_func = None
    flash_attn_func = None
    _flash_supports_window_size = False


logger = logging.get_logger(__name__)
apply_multimodal_rotary_pos_emb_query_or_key = None
prepare_fa2_from_position_ids = None


# Expand KV if less than H
# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Qwen cross attention
class Qwen2VLCrossAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        # if config.head_dim is not None:
        #     self.head_dim = config.head_dim
        # else:
        #     self.head_dim = config.hidden_size // config.num_attention_heads
        # self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = False
        self.attention_dropout = config.attention_dropout
        # self.rope_scaling = config.rope_scaling

        # if (self.head_dim * self.num_heads) != self.hidden_size:
        #     raise ValueError(
        #         f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
        #         f" and `num_heads`: {self.num_heads})."
        #     )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_position_embeddings: Optional[torch.LongTensor] = None,
        past_key_value_ca: Optional[Cache] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()  # [1, 29, 3584]
        kv_len = encoder_hidden_states.size(1)  # [1, 1296, 3584]

        query_states = self.q_proj(hidden_states)  # [1, 29, 3584] -> [1, 29, 3584]
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # check if in cache
        if past_key_value_ca is not None and past_key_value_ca.check(self.layer_idx):
            key_states, value_states = past_key_value_ca[self.layer_idx]
        else:
            # calculate
            key_states = self.k_proj(
                encoder_hidden_states
            )  # [1296, 3584] -> [1296, 512]
            value_states = self.v_proj(encoder_hidden_states)
            key_states = key_states.view(
                bsz, kv_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, kv_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            if past_key_value_ca is not None:
                key_states, value_states = past_key_value_ca.update(
                    key_states, value_states, self.layer_idx
                )

        if position_embeddings is not None:
            q_cos, q_sin = position_embeddings
            query_states = apply_multimodal_rotary_pos_emb_query_or_key(
                query_states, q_cos, q_sin, self.rope_scaling["mrope_section"]
            )
        if encoder_position_embeddings is not None:
            k_cos, k_sin = encoder_position_embeddings
            key_states = apply_multimodal_rotary_pos_emb_query_or_key(
                key_states, k_cos, k_sin, self.rope_scaling["mrope_section"]
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_len)}, but is"
                f" {attn_weights.size()}"
            )

        if cross_attention_mask is not None:  # no matter the length, we just slice it
            cross_attention_mask = cross_attention_mask.unsqueeze(1).repeat(
                1, self.num_heads, 1, 1
            )
            attn_weights = attn_weights + cross_attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


# Qwen Sdpa cross attention
class Qwen2VLSdpaCrossAttention(Qwen2VLCrossAttention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_position_embeddings: Optional[torch.LongTensor] = None,
        past_key_value_ca: Optional[Cache] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                cross_attention_mask=cross_attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )

        bsz, q_len, _ = hidden_states.size()
        kv_len = encoder_hidden_states.size(1)

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # check if in cache
        if past_key_value_ca is not None and past_key_value_ca.check(self.layer_idx):
            key_states, value_states = past_key_value_ca[self.layer_idx]
        else:
            # calculate
            key_states = self.k_proj(encoder_hidden_states)
            value_states = self.v_proj(encoder_hidden_states)
            key_states = key_states.view(
                bsz, kv_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                bsz, kv_len, self.num_key_value_heads, self.head_dim
            ).transpose(1, 2)
            if past_key_value_ca is not None:
                key_states, value_states = past_key_value_ca.update(
                    key_states, value_states, self.layer_idx
                )

        if position_embeddings is not None:
            q_cos, q_sin = position_embeddings
            query_states = apply_multimodal_rotary_pos_emb_query_or_key(
                query_states, q_cos, q_sin, self.rope_scaling["mrope_section"]
            )
        if encoder_position_embeddings is not None:
            k_cos, k_sin = encoder_position_embeddings
            key_states = apply_multimodal_rotary_pos_emb_query_or_key(
                key_states, k_cos, k_sin, self.rope_scaling["mrope_section"]
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        # is_causal = True if causal_mask is None and q_len > 1 else False
        is_causal = False

        if cross_attention_mask is not None:
            cross_attention_mask = cross_attention_mask.unsqueeze(1).repeat(
                1, self.num_heads, 1, 1
            )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=cross_attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)
        return attn_output, None
