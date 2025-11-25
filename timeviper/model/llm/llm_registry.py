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

from timeviper.model.llm.llm_repo.nano.modeling_nano import (
    NemotronHBlock,
    NemotronHForCausalLM,
)
from timeviper.model.llm.llm_repo.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
)

add_eot_id = lambda self: setattr(
    self,
    "terminators",
    list(
        set(
            [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
        )
    ),
)

add_im_end_id = lambda self: setattr(
    self,
    "terminators",
    list(
        set(
            [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            ]
        )
    ),
)

add_special_12_id = lambda self: setattr(
    self,
    "terminators",
    list(
        set(
            [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<SPECIAL_12>"),
                self.tokenizer.convert_tokens_to_ids("</s>"),
            ]
        )
    ),
)

MODEL_REGISTRY = {
    "qwen2": {
        "llm_cls": Qwen2ForCausalLM,
        "layer_cls": Qwen2DecoderLayer,
        "default_max_length": 128000,
        "variants": {
            "qwen2-7b": {"hf_hub_path": "Qwen/Qwen2-7B"},
            "qwen2-7b-instruct": {"hf_hub_path": "Qwen/Qwen2-7B-Instruct"},
            "qwen2-1.5b": {"hf_hub_path": "Qwen/Qwen2-1.5B"},
            "qwen2-1.5b-instruct": {"hf_hub_path": "Qwen/Qwen2-1.5B-Instruct"},
            "qwen2.5-7b-instruct": {"hf_hub_path": "Qwen/Qwen2.5-7B-Instruct"},
            "qwen2.5-7b-base": {"hf_hub_path": "Qwen/Qwen2.5-7B-Base"},
            "qwen2.5-3b-instruct": {"hf_hub_path": "Qwen/Qwen2.5-3B-Instruct"},
            "qwen2.5-3b-base": {"hf_hub_path": "Qwen/Qwen2.5-3B-Base"},
        },
        "init_hook": add_im_end_id,
    },
    "nano": {
        "llm_cls": NemotronHForCausalLM,
        "layer_cls": NemotronHBlock,
        "default_max_length": 128000,
        "variants": {
            "nano-9b-v2": {"hf_hub_path": "nvidia/NVIDIA-Nemotron-Nano-9B-v2"},
            "nemotron-h-8b-base": {"hf_hub_path": "nvidia/Nemotron-H-8B-Base-8K"},
            "nano-9b-v2-base": {
                "hf_hub_path": "nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base"
            },
            "nano-12b-v2-base": {
                "hf_hub_path": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base"
            },
        },
        "init_hook": add_special_12_id,
    },
}


def get_model_config(llm_backbone_id: str):
    for family, config in MODEL_REGISTRY.items():
        if llm_backbone_id in config["variants"]:
            variant_config = config["variants"][llm_backbone_id]
            final_config = {**config, **variant_config, "llm_family": family}
            return final_config
    raise ValueError(f"Model ID '{llm_backbone_id}' not found in master registry.")
