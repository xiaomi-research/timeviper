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
#

import copy
import os
import re
from typing import Any, Optional

import torch
from transformers.utils import (
    is_torch_cuda_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)
from vllm import LLM, SamplingParams


def get_device_count() -> int:
    r"""Get the number of available GPU or NPU devices."""
    if is_torch_xpu_available():
        return torch.xpu.device_count()
    elif is_torch_npu_available():
        return torch.npu.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0


class vllmTimeViperWrapper:
    def __init__(self, args, model):
        pipeline_parallel_size = args.pipeline_parallel_size
        if pipeline_parallel_size > get_device_count():
            raise ValueError(
                "Pipeline parallel size should be smaller than the number of gpus."
            )
        engine_args = {
            "model": f"{'/'.join(args.model_base.split('/')[:-1])}",
            "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
            "pipeline_parallel_size": pipeline_parallel_size,
            # "max_model_len": args.total_pixels // 28 // 28 + 4096 + args.max_new_tokens,
            "max_model_len": 32768,
            "max_num_batched_tokens": 32768,
            "gpu_memory_utilization": 0.8,
            "disable_mm_preprocessor_cache": True,  # Otherwise, once the cache hits, the number of FPS won't match the number of videos, causing a bug. Our speed bottleneck isn't here.
            "enable_prompt_embeds": True,
            "enforce_eager": False,  # debug use only, later, we can switch on it for further speed up
        }
        self.arch_specifier = model.arch_specifier
        self.vision_backbone = model.vision_backbone
        self.projector = model.projector
        self.default_token_id = model.default_token_id
        self.llm_embedding = model.llm_backbone.llm.get_input_embeddings()
        self.eos_token_ids_to_use = getattr(
            model.llm_backbone,
            "terminators",
            [model.llm_backbone.tokenizer.eos_token_id],
        )
        os.makedirs(
            os.path.join(engine_args["model"], "use_vllm_inference", "llm_ckpts"),
            exist_ok=True,
        )
        if not os.path.exists(
            os.path.join(
                engine_args["model"], "use_vllm_inference/llm_ckpts", "llm_model.bin"
            )
        ):
            torch.save(
                model.llm_backbone.llm.state_dict(),
                os.path.join(
                    engine_args["model"],
                    "use_vllm_inference/llm_ckpts",
                    "llm_model.bin",
                ),
            )
            os.system(
                f"cp -v {engine_args['model']}/*.json {os.path.join(engine_args['model'], 'use_vllm_inference/llm_ckpts')}"
            )
            print("llm ckpt saved")
        else:
            print("Already have llm ckpt")
        engine_args["model"] = os.path.join(
            engine_args["model"], "use_vllm_inference/llm_ckpts"
        )
        del model
        torch.cuda.empty_cache()
        # engine_args["limit_mm_per_prompt"] = {"image": 0, "video": 1, "audio": 0}
        self.model = LLM(**engine_args)
        self.tokenizer = self.model.get_tokenizer()

    def find_answer_token_last_occurrence(self, text: str):
        answer_token = "<answer>"
        idx = text.rfind(answer_token)
        return idx

    @staticmethod
    def extract_timestamps(response):
        number_pattern = r"\d+(?:\.\d+)?"
        matches = re.findall(number_pattern, response)
        output = [float(num) for num in matches[-2:]]
        if len(output) == 2:
            return output[0], output[1]
        return None, None

    @torch.no_grad()
    def get_input_embeddings(
        self, input_ids, pixel_values=None, pixel_values_videos=None
    ):
        with torch.set_grad_enabled(False):
            vision_inputs = (
                pixel_values_videos if pixel_values_videos is not None else pixel_values
            )
            patch_features = self.vision_backbone(vision_inputs)
            if "tome_mlp" in self.arch_specifier:
                # 处理混合编码器情况
                vision_identifier = self.vision_backbone.get_identifier
                if vision_identifier in ["internvideo2siglip"]:
                    len_L = patch_features.shape[1]
                    frame_T = len_L // 2 // 729  # 729 适配 siglip 384 的输入像素
                    dim_D = patch_features.shape[2]
                    visual_embeddings_video_encoder = self.projector(
                        patch_features[:, : len_L // 2, :],
                        compress=True,
                        local_num_frames=4,
                    )
                    visual_embeddings_image_encoder = self.projector(
                        patch_features[:, len_L // 2 :, :].reshape(frame_T, -1, dim_D),
                        compress=True,
                        local_num_frames=1,
                    ).reshape(1, frame_T * self.num_compressed_tokens, -1)
                    visual_embeddings = torch.cat(
                        [
                            visual_embeddings_video_encoder,
                            visual_embeddings_image_encoder,
                        ],
                        dim=1,
                    )
                elif vision_identifier in ["internvideo2"]:
                    visual_embeddings = self.projector(
                        patch_features, compress=True, local_num_frames=4
                    )
                elif vision_identifier in ["siglip", "dinov2siglip", "dinov2"]:
                    visual_embeddings = self.projector(
                        patch_features, compress=True, local_num_frames=1
                    )
            else:
                visual_embeddings = self.projector(patch_features)

            if torch.isnan(visual_embeddings).any():
                print("NaN values found in hidden_states.")

            # 融合视觉和文本嵌入
            vision_positions = (input_ids == self.default_token_id).nonzero(
                as_tuple=False
            )
            embeddings_list = []
            embeddings_list.append(
                self.llm_embedding(input_ids[0:1, 0 : vision_positions[0][1]])
            )
            for i in range(len(vision_positions)):
                batch_idx, seq_idx = vision_positions[i]  # batch size always equals 1
                embeddings_list.append(visual_embeddings[i : i + 1, :, :])
                start = seq_idx + 1
                end = (
                    vision_positions[i + 1][1]
                    if i < len(vision_positions) - 1
                    else input_ids.shape[1]
                )
                if input_ids[0, start] == self.default_token_id:
                    continue
                embeddings_list.append(self.llm_embedding(input_ids[0:1, start:end]))

            fused_embeddings = torch.cat(
                embeddings_list,
                dim=1,
            )

        return fused_embeddings

    @torch.no_grad()
    def generate(
        self,
        inputs: dict[str, Any],
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
        answer_prompt: Optional[str] = None,  # only r1 model needed
    ):
        r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

        Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
        """
        vllm_inputs = []
        for input_ids in zip(
            inputs["input_ids"],
        ):
            vllm_inputs.append(
                {
                    "input_ids": list(input_ids),
                }
            )
        sampling_params = SamplingParams(
            repetition_penalty=repetition_penalty or 1.0,  # repetition_penalty must > 0
            temperature=temperature,
            top_p=top_p,  # top_p must > 0
            top_k=top_k,  # top_k must > 0, ==-1
            stop=None,
            stop_token_ids=self.eos_token_ids_to_use,  # generate_config.json
            max_tokens=max_new_tokens,
            include_stop_str_in_output=True,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
            seed=seed,
        )
        sampling_params = [
            copy.deepcopy(sampling_params) for _ in range(len(vllm_inputs))
        ]
        dtype = self.projector.projector[0].weight.dtype
        device = self.projector.projector[0].weight.device
        if inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                dtype=dtype, device=device
            )
        if inputs["pixel_values_videos"] is not None:
            inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(
                dtype=dtype, device=device
            )
        input_embeddings = self.get_input_embeddings(
            inputs["input_ids"].to(device=device),
            inputs["pixel_values"],
            inputs["pixel_values_videos"],
        )
        if answer_prompt is not None:
            answer_prompt_ids = self.tokenizer(
                answer_prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device)
            answer_prompt_embedding = self.llm_embedding(
                answer_prompt_ids
            ).repeat_interleave(input_embeddings.shape[0], dim=0)
            input_embeddings = torch.cat(
                [input_embeddings, answer_prompt_embedding], dim=1
            )

        prompts_with_embeds = [
            {"prompt_embeds": single_embedding} for single_embedding in input_embeddings
        ]
        results = self.model.generate(
            prompts=prompts_with_embeds, sampling_params=sampling_params, use_tqdm=False
        )

        preds = [result.outputs[0].text for result in results]
        # For MCQ questions, to obtain option letters, we follow the MVBench approach by adding an answer prompt to force the model to print the option in the specified position.
        # This uses string matching, as input_ids vary too much otherwise.

        return preds
