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

import warnings
from functools import partial
from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from timeviper.utils.overwatch import initialize_overwatch

from .llm_registry import get_model_config

# Suppress Hugging Face Deprecation Warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Overwatch for logging
overwatch = initialize_overwatch(__name__)


class GenericLLMBackbone(nn.Module):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: Optional[int] = None,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        attn_implementation: Optional[str] = "sdpa",
        continue_pretrain_ckpt: Optional[str] = None,
        merge_module: Optional[str] = "no_merge",
        use_pdrop: bool = False,
        pdrop_type: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.identifier = llm_backbone_id
        self.inference_mode = inference_mode

        self.model_config = get_model_config(llm_backbone_id)

        llm_family = self.model_config["llm_family"]
        llm_cls = self.model_config["llm_cls"]
        hf_hub_path = self.model_config["hf_hub_path"]
        self.llm_family = llm_family
        self.llm_max_length = (
            llm_max_length
            if llm_max_length is not None
            else self.model_config.get("default_max_length", 2048)
        )

        attn_implementation = (
            attn_implementation
            if attn_implementation is not None
            else self.model_config.get("attn_implementation", "sdpa")
        )
        use_rope_mem = self.model_config.get("use_rope_mem", False)

        if not self.inference_mode:
            overwatch.info(
                f"Loading [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]",
                ctx_level=1,
            )
            if continue_pretrain_ckpt is not None:
                load_model_path = continue_pretrain_ckpt
            else:
                load_model_path = f"./ckpts/{hf_hub_path.split('/')[-1]}"
            overwatch.info(f"Loading model weights from {load_model_path}", ctx_level=2)
            if use_pdrop:
                self.llm: PreTrainedModel = llm_cls.from_pretrained(
                    load_model_path,
                    token=hf_token,
                    attn_implementation=attn_implementation,
                    do_sample=False,
                    temperature=1.0,
                    trust_remote_code=True,
                    top_p=1.0,
                    merge_module=merge_module,
                    use_pdrop=use_pdrop,
                    pdrop_type=pdrop_type,
                )
            else:
                self.llm: PreTrainedModel = llm_cls.from_pretrained(
                    load_model_path,
                    token=hf_token,
                    attn_implementation=attn_implementation,
                    do_sample=False,
                    temperature=1.0,
                    trust_remote_code=True,
                    top_p=1.0,
                )
        else:
            overwatch.info(
                f"Building empty [bold]{llm_family}[/] LLM from [underline]`{hf_hub_path}`[/]",
                ctx_level=1,
            )
            llm_config = AutoConfig.from_pretrained(
                f"./ckpts/{hf_hub_path.split('/')[-1]}", token=hf_token
            )
            llm_config.attn_implementation = attn_implementation
            if use_rope_mem:
                llm_config.use_mem_rope = True
            self.llm: PreTrainedModel = llm_cls._from_config(llm_config)

        self.llm.config.use_cache = self.inference_mode

        if not self.inference_mode:
            self.llm.enable_input_require_grads()

        overwatch.info(
            f"Loading [bold]{llm_family}[/] (Fast) Tokenizer via the AutoTokenizer API",
            ctx_level=1,
        )
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            f"./ckpts/{hf_hub_path.split('/')[-1]}",
            model_max_length=self.llm_max_length,
            token=hf_token,
        )

        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})

        if self.tokenizer.padding_side != "right":
            self.tokenizer.padding_side = "right"
        assert (
            self.tokenizer.padding_side == "right"
        ), "Tokenizer `padding_side` is not set to `right`!"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id

        if "init_hook" in self.model_config:
            self.model_config["init_hook"](self)

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    def get_fsdp_wrapping_policy(self) -> Callable:
        transformer_block_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={self.transformer_layer_cls},
        )
        return transformer_block_policy

    def enable_gradient_checkpointing(self) -> None:
        self.llm.gradient_checkpointing_enable()

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    def allocate_inference_cache(self, *args, **kwargs):
        if hasattr(self.llm, "allocate_inference_cache"):
            return self.llm.allocate_inference_cache(*args, **kwargs)
        else:
            raise NotImplementedError(
                "Inference cache allocation not implemented for this model."
            )

    def forward(self, *args, **kwargs) -> CausalLMOutputWithPast:
        output: CausalLMOutputWithPast = self.llm(*args, **kwargs)
        return output

    @property
    def transformer_layer_cls(
        self,
    ) -> Union[Type[nn.Module], tuple[Type[nn.Module], ...]]:
        return self.model_config["layer_cls"]

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    @property
    def embed_dim(self) -> int:
        return self.llm.config.hidden_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id


def LLMBackboneFactory(llm_backbone_id: str, **kwargs) -> GenericLLMBackbone:
    return GenericLLMBackbone(llm_backbone_id, **kwargs)
