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

from __future__ import annotations

import inspect
from functools import partial
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from torch.utils.checkpoint import checkpoint
from transformers import GenerationMixin, PretrainedConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import CausalLMOutputWithPast

from timeviper.model.llm import GenericLLMBackbone
from timeviper.model.projector import (
    MLPProjector,
    ToMe16_mlp_hd64,
)
from timeviper.model.vit import VisionBackbone
from timeviper.utils.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)
IGNORE_INDEX = -100
DEFAULT_TOKEN = "<image>"


def print_grad(name):
    def hook(grad):
        if grad.isnan().any():
            print(f"NaN gradient detected in {name}!")

    return hook


def _parse_compressed_tokens(arch_specifier: str) -> int:
    """Helper function to parse compressed tokens, moved here to be shared."""
    if "tome_mlp" in arch_specifier:
        parts = arch_specifier.split("-")
        try:
            if len(parts) > 2 and parts[2].isdigit():
                return int(parts[2])
        except (IndexError, ValueError):
            return 16
    return 16


class GenericTimeViperVLM(nn.Module, GenerationMixin):

    supports_gradient_checkpointing = True
    _is_stateful = False

    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: GenericLLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        visual_token_order: str = "raw",
        disable_data_packing: bool = False,
    ) -> None:
        super().__init__()

        # --- 从 VLM 基类合并的属性 ---
        self.model_family = f"{llm_backbone.llm_family}"
        self.model_id = model_id
        self.vision_backbone = vision_backbone
        self.llm_backbone = llm_backbone
        self.enable_mixed_precision_training = enable_mixed_precision_training

        self.generation_config = self.llm_backbone.llm.generation_config
        self.main_input_name = "input_ids"

        self._supports_cache_class = True
        torch.manual_seed(vision_backbone.embed_dim)

        self.arch_specifier = arch_specifier
        assert visual_token_order in [
            "raw",
            "ascending",
            "descending",
        ], f"visual_token_order must be one of 'raw', 'ascending', 'descending', but got {visual_token_order}"
        self._initialize_projector(visual_token_order=visual_token_order)

        self.vision_backbone_requires_grad = False
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector"]
        self.trainable_module_keys = []
        self.eos_token_ids_to_use = getattr(
            self.llm_backbone, "terminators", [self.llm_backbone.tokenizer.eos_token_id]
        )

        # for pdrop
        has_pdrop, module = self.get_pdrop_value(llm_backbone)
        if has_pdrop:
            self.use_pdrop = module.use_pdrop
            self.pdrop_args = {"use_pdrop": self.use_pdrop}
            if self.use_pdrop:
                assert (
                    module.pdrop_types is not None
                ), "use_pdrop is True, but pdrop_type is not set"
                pdrop_types = module.pdrop_types
                assert all(
                    [len(typ) == 3 for typ in pdrop_types]
                ), "pdrop_type should be like 'type_layernum_ratio-...' "
                self.pdrop_args.update(
                    {
                        "pdrop_compress_types": [typ[0] for typ in pdrop_types],
                        "pdrop_layers": [int(typ[1]) for typ in pdrop_types],
                        "pdrop_ratios": [1] + [float(typ[2]) for typ in pdrop_types],
                    }
                )
                overwatch.info(f"Using pdrop with types: {pdrop_types}")
                if hasattr(self.llm_backbone.llm, "set_pdrop_args"):
                    self.llm_backbone.llm.set_pdrop_args(**self.pdrop_args)
                else:
                    raise NotImplementedError(
                        f"llm_backbone {self.llm_backbone} currently does not support pdrop."
                    )
        else:
            self.use_pdrop = False
            self.pdrop_args = {"use_pdrop": self.use_pdrop}
        self.disable_data_packing = disable_data_packing

    def get_pdrop_value(self, llm_backbone: GenericLLMBackbone) -> bool:
        """Helper function to get pdrop value from llm_backbone config."""
        if hasattr(llm_backbone.llm, "backbone") and hasattr(
            llm_backbone.llm.backbone, "use_pdrop"
        ):
            return True, llm_backbone.llm.backbone
        elif hasattr(llm_backbone.llm.model, "use_pdrop") and hasattr(
            llm_backbone.llm.model, "use_pdrop"
        ):
            return True, llm_backbone.llm.model
        else:
            return False, None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.device:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return dtype

    @staticmethod
    def can_generate() -> bool:
        return True

    @property
    def llm_tokenizer(self):
        return self.llm_backbone.tokenizer

    @property
    def default_token_id(self):
        return self.llm_tokenizer.convert_tokens_to_ids(DEFAULT_TOKEN)

    @property
    def config(self) -> PretrainedConfig:
        return self.llm_backbone.llm.config

    def _reorder_cache(self, past_key_values, beam_idx):
        return self.llm_backbone.llm._reorder_cache(past_key_values, beam_idx)

    def _initialize_projector(self, visual_token_order):
        vision_embed_dim = self.vision_backbone.embed_dim
        llm_embed_dim = self.llm_backbone.embed_dim
        if self.arch_specifier.endswith("gelu-mlp"):
            self.projector = MLPProjector(vision_embed_dim, llm_embed_dim)
        elif "tome_mlp" in self.arch_specifier:
            self.num_compressed_tokens = _parse_compressed_tokens(self.arch_specifier)
            self.projector = ToMe16_mlp_hd64(
                vision_embed_dim,
                llm_embed_dim,
                mlp_type=self.arch_specifier.split("+")[-1],
                num_compressed_tokens=self.num_compressed_tokens,
                token_order=visual_token_order,
            )
        else:
            raise ValueError(
                f"GenericTimeViperVLM with projector architecture `{self.arch_specifier}` is not supported!"
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inference_params=None,
        num_last_tokens: int = 0,
        answer_prompt: Optional[str] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        logits_cache: Optional[torch.FloatTensor] = None,
        txt_seq_lens: Optional[List[int]] = None,
        img_seq_lens: Optional[List[int]] = None,
        vid_seq_lens: Optional[List[int]] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        if pixel_values is None and pixel_values_videos is None:
            return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                inference_params=inference_params,
                num_last_tokens=num_last_tokens,
                train_pdrop_args=self.pdrop_args,
            )
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            vision_inputs = (
                pixel_values_videos if pixel_values_videos is not None else pixel_values
            )
            if self.training:
                vision_features = self.vision_backbone(vision_inputs)
            else:
                clips = vision_inputs.split(split_size=256)
                clip_features = []
                for clip in clips:
                    clip_feature = self.projector_forward(self.vision_backbone(clip))
                    clip_features.append(clip_feature)
                visual_embeddings = torch.cat(tensors=clip_features, dim=0)
        if self.training:
            visual_embeddings = self.projector_forward(vision_features)

        assert input_ids is not None
        # ====================  DATA PACKING ====================
        seq_idx = None
        is_packed = txt_seq_lens is not None and len(txt_seq_lens) > 1
        if not is_packed or self.training is False:
            if self.use_pdrop and inputs_embeds is None:
                num_placeholder_tokens = input_ids.eq(self.default_token_id).sum(dim=1)
                first_vision_token_positions = torch.argmax(
                    input_ids.eq(self.default_token_id).int(), dim=1
                )
                text_prompt_lens = [
                    len(ids) - num_p
                    for ids, num_p in zip(input_ids, num_placeholder_tokens)
                ]
                num_vision_tokens = [
                    visual_embeddings.size(0) * visual_embeddings.size(1)
                ]  # assume batch size = 1
                train_pdrop_args = {
                    "first_vision_token_positions": first_vision_token_positions,
                    "text_prompt_lens": text_prompt_lens,
                    "num_vision_tokens": num_vision_tokens,
                    "is_interleaved": False,
                }
                self.pdrop_args.update(train_pdrop_args)
            fused_embeddings, fused_labels = self.get_fused_data_nopacked(
                visual_embeddings, input_ids, labels
            )
            position_ids, fused_attention_mask, fused_cache_position = (
                self.get_attention_mask_nopacked(
                    attention_mask,
                    cache_position,
                    position_ids,
                    visual_embedding_shape=visual_embeddings.shape,
                    txt_seq_len=input_ids.shape[1]
                    - input_ids.eq(self.default_token_id).sum(),
                    fused_embedding_shape=fused_embeddings.shape,
                    total_len=fused_embeddings.shape[1],
                    labels=labels,
                    device=fused_embeddings.device,
                )
            )
        else:
            input_ids_list = torch.split(input_ids.squeeze(0), txt_seq_lens)
            vis_num_list = [
                torch.count_nonzero(input_ids == self.default_token_id)
                for input_ids in input_ids_list
            ]
            vis_seq_lens = [
                vis_num * self.num_compressed_tokens for vis_num in vis_num_list
            ]
            true_text_lengths = [
                txt_len - vis_num
                for txt_len, vis_num in zip(txt_seq_lens, vis_num_list)
            ]
            if self.use_pdrop and inputs_embeds is None:
                first_vision_token_positions_list = [
                    torch.argmax(i_ids.eq(self.default_token_id).int(), dim=0)
                    for i_ids in input_ids_list
                ]
                is_interleaved = False
                train_pdrop_args = {
                    "first_vision_token_positions": first_vision_token_positions_list,
                    "text_prompt_lens": true_text_lengths,
                    "num_vision_tokens": vis_seq_lens,
                    "sample_seq_lens": [
                        t + v for (t, v) in zip(true_text_lengths, vis_seq_lens)
                    ],
                    "is_interleaved": is_interleaved,
                }
                self.pdrop_args.update(train_pdrop_args)
            assert (
                input_ids.shape[0] == 1
            ), "Data packing currently supports batch size 1 only."

            fused_embeddings, fused_labels = self.get_fused_data_packed(
                visual_embeddings,
                input_ids_list,
                labels,
                vis_num_list,
                txt_seq_lens,
            )

            position_ids, fused_attention_mask, fused_cache_position, seq_idx = (
                self.get_attention_mask_packed(
                    visual_lengths=vis_seq_lens,
                    text_lengths=true_text_lengths,
                    total_len=fused_embeddings.shape[1],
                    device=fused_embeddings.device,
                )
            )
        if self.disable_data_packing:
            seq_idx = None
            position_ids = torch.arange(
                fused_embeddings.shape[1], device=fused_embeddings.device
            ).unsqueeze(0)
        # ====================  DATA PACKING ====================
        return self.llm_backbone(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            cache_position=fused_cache_position,
            logits_to_keep=logits_to_keep,
            seq_idx=seq_idx,
            train_pdrop_args=train_pdrop_args if self.use_pdrop else None,
        )

    def projector_forward(self, patch_features):
        if "tome_mlp" in self.arch_specifier:
            vision_identifier = self.vision_backbone.get_identifier
            if vision_identifier in ["siglip", "dinov2siglip", "dinov2"]:
                return self.projector(patch_features, compress=True, local_num_frames=1)
        else:
            visual_embeddings = self.projector(patch_features)
            self.num_compressed_tokens = visual_embeddings.shape[1]
        if torch.isnan(visual_embeddings).any():
            print("NaN values found in hidden_states.")
        return visual_embeddings

    def get_attention_mask_nopacked(
        self,
        attention_mask,
        cache_position,
        position_ids,
        visual_embedding_shape,
        txt_seq_len,
        fused_embedding_shape,
        total_len,
        labels,
        device,
    ):
        """prefilling: attention mask is only for text ids
        decoding: attention_mask is for v+t ids
        """
        fused_cache_position = None
        if self.training is False and attention_mask is not None:
            if visual_embedding_shape[0] != fused_embedding_shape[0]:
                visual_embedding_shape = (
                    fused_embedding_shape[0],
                    visual_embedding_shape[0] * visual_embedding_shape[1],
                )
            visual_attention_mask = torch.ones(
                visual_embedding_shape[:2], dtype=attention_mask.dtype, device=device
            )
            fused_attention_mask = torch.cat(
                [visual_attention_mask, attention_mask], dim=1
            )
            if cache_position is not None:
                num_visual_tokens = visual_embedding_shape[1]
                visual_positions = torch.arange(num_visual_tokens, device=device)
                text_positions = (
                    torch.arange(txt_seq_len, device=device) + num_visual_tokens
                )
                fused_cache_position = torch.cat(
                    [visual_positions, text_positions], dim=0
                )

            if (
                position_ids is not None
                and position_ids.shape[1] != fused_attention_mask.shape[1]
            ):
                num_visual_tokens = visual_embedding_shape[1]
                visual_positions = torch.arange(num_visual_tokens, device=device)
                text_positions = (
                    torch.arange(txt_seq_len, device=device) + num_visual_tokens
                )
                position_ids = (
                    torch.cat([visual_positions, text_positions], dim=0)
                    .unsqueeze(0)
                    .expand(fused_embedding_shape[0], -1)
                )
            if len(fused_attention_mask.shape) == 2:
                fused_attention_mask = fused_attention_mask.unsqueeze(0).unsqueeze(0)
            if fused_attention_mask.dtype != torch.bool:
                fused_attention_mask = fused_attention_mask.to(torch.bool)
            if self.llm_backbone.llm.config._attn_implementation == "flash_attention_2":
                fused_attention_mask = None
            if self.llm_backbone.llm.config._attn_implementation == "eager":
                fused_attention_mask = torch.ones(
                    (1, total_len), dtype=attention_mask.dtype, device=device
                )
                inputs_embeds_for_mask = torch.empty(
                    fused_embedding_shape[:2], dtype=self.dtype, device=device
                )
                fused_attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask=fused_attention_mask,
                    input_shape=fused_embedding_shape[:2],
                    inputs_embeds=inputs_embeds_for_mask,
                    past_key_values_length=0,  # Assuming prefilling, no past_key_values
                )
        else:
            # 2. 显式创建 Position IDs
            fused_attention_mask = None
            position_ids = torch.arange(total_len, device=device).unsqueeze(0)
        return position_ids, fused_attention_mask, fused_cache_position

    def get_fused_data_nopacked(self, visual_embeddings, input_ids, labels):
        vision_positions = (input_ids == self.default_token_id).nonzero(as_tuple=False)
        embeddings_list = []
        labels_list = []
        embeddings_list.append(
            self.llm_backbone.embed_input_ids(
                input_ids[0:1, 0 : vision_positions[0][1]]
            )
        )
        labels_list.append(
            labels[0:1, 0 : vision_positions[0][1]] if labels is not None else None
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
            embeddings_list.append(
                self.llm_backbone.embed_input_ids(input_ids[0:1, start:end])
            )
            labels_list.append(labels[0:1, start:end] if labels is not None else None)

        fused_embeddings = torch.cat(
            embeddings_list,
            dim=1,
        )
        fused_labels = None
        if labels is not None:
            labels = torch.cat(labels_list, dim=1)
            projected_patch_labels = torch.full(
                (
                    1,
                    visual_embeddings.shape[0] * visual_embeddings.shape[1],
                ),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            # Only support multi-round conversations, instead of interleaved vision and text inputs.
            fused_labels = torch.cat([projected_patch_labels, labels], dim=1)

        return fused_embeddings, fused_labels

    def fuse_img_text(self, visual_embedding, text_embeddings, input_ids):
        """replace <image> with visual embeddings
        single image + text fusion
        visual_embedding: shape: (num_frames, num_patches, dim)
        text_embedding: shape: (batch(1), seq_len, dim)
        input_ids: shape: (batch(1), seq_len)
        """
        vision_positions = (input_ids == self.default_token_id).nonzero(as_tuple=False)
        # replace one frame / single image visual embedding by each <image> token
        embeddings_list = []
        embeddings_list.append(text_embeddings[0:1, 0 : vision_positions[0][1], :])
        for i in range(len(vision_positions)):
            batch_idx, seq_idx = vision_positions[i]  # batch size always equals 1
            start = seq_idx + 1
            end = (
                vision_positions[i + 1][1] + 1
                if i < len(vision_positions) - 1
                else text_embeddings.shape[1] + 1
            )
            embeddings_list.append(visual_embedding[i : i + 1, :, :])
            embeddings_list.append(text_embeddings[0:1, start:end, :])
        fused_embeddings = torch.cat(
            embeddings_list,
            dim=1,
        )
        return fused_embeddings

    def get_fused_data_packed(
        self,
        visual_embeddings,
        input_ids_list,
        labels,
        vis_num_list,
        txt_seq_lens,
    ):
        """
        vis_num_list: [4, 6, ...], number of images in each single-round conversation
        """
        num_of_packed_samples = len(txt_seq_lens)
        # text_embeds_list = torch.split(input_embeddings.squeeze(0), txt_seq_lens)
        labels_list = torch.split(labels.squeeze(0), txt_seq_lens)
        num_of_imgs = sum(vis_num_list)
        # case 1: single image/video, multiple conversation packed, duplicate the visual input
        if num_of_imgs > visual_embeddings.shape[0]:
            visual_embeds_list = [visual_embeddings] * num_of_packed_samples
        # case 2: single conversations, multiple images packed, split the visual input
        elif num_of_imgs == visual_embeddings.shape[0]:
            visual_embeds_list = torch.split(visual_embeddings, vis_num_list)
        else:
            raise NotImplementedError(
                f"num_of_imgs {num_of_imgs} and visual_embeddings {visual_embeddings.shape[0]} not match."
            )

        fused_embeddings_list = []
        fused_labels_list = []
        for vis_embeds, input_ids, lbls in zip(
            visual_embeds_list, input_ids_list, labels_list
        ):
            fused_embeddings, fused_labels = self.get_fused_data_nopacked(
                vis_embeds, input_ids.unsqueeze(0), lbls.unsqueeze(0)
            )
            fused_embeddings_list.append(fused_embeddings)

            if labels is not None:
                fused_labels_list.append(fused_labels)
        fused_embeddings = torch.cat(fused_embeddings_list, dim=1)
        fused_labels = torch.cat(fused_labels_list, dim=1)

        return fused_embeddings, fused_labels

    def get_attention_mask_packed(
        self, visual_lengths, text_lengths, total_len, device
    ):
        """attention mask for packed data"""
        fused_cache_position = None
        if self.llm_backbone.llm.config._attn_implementation == "flash_attention_2":
            fused_attention_mask = None
        else:
            fused_attention_mask = torch.zeros(
                (1, 1, total_len, total_len), dtype=torch.bool, device=device
            )
        position_ids = torch.zeros((1, total_len), dtype=torch.int, device=device)
        seq_idx = torch.zeros((1, total_len), dtype=torch.int, device=device)
        current_offset = 0
        for idx, (v_len, t_len) in enumerate(zip(visual_lengths, text_lengths)):
            seq_len = v_len + t_len
            start, end = current_offset, current_offset + seq_len
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
            if fused_attention_mask is not None:
                fused_attention_mask[0, 0, start:end, start:end] = causal_mask

            position_ids[0, start:end] = torch.arange(seq_len)
            seq_idx[0, start:end] = idx
            current_offset += seq_len
        return position_ids, fused_attention_mask, fused_cache_position, seq_idx

    def allocate_inference_cache(self, *args, **kwargs):
        return self.llm_backbone.allocate_inference_cache(*args, **kwargs)

    def freeze_backbones(self, stage: str) -> None:
        if stage == "align":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)
            self.trainable_module_keys = ["projector"]
            self.vision_backbone_requires_grad = False
        elif stage in ["finetune", "full-finetune"]:
            is_full_finetune = stage == "full-finetune"
            self.vision_backbone.requires_grad_(is_full_finetune)
            if is_full_finetune:
                self.vision_backbone.dtype = torch.float32
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
            self.trainable_module_keys = ["projector", "llm_backbone"]
            if is_full_finetune:
                self.trainable_module_keys.append("vision_backbone")
            self.vision_backbone_requires_grad = is_full_finetune
        else:
            raise ValueError(
                f"Stage `{stage}` is not supported! Try < align | finetune | full-finetune >"
            )

    def load_from_checkpoint(
        self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None
    ) -> None:
        assert stage in {
            "align",
            "finetune",
            "full-finetune",
        }, f"Stage {stage} is not supported!"
        if self.arch_specifier.startswith("no-align") or stage == "align":
            return

        overwatch.info(f"Stage `{stage}` requires pretrained weights", ctx_level=1)
        if pretrained_checkpoint is None:
            model, scale, _, seed = run_dir.name.split("+")
            align_dirs = [
                d
                for d in run_dir.parent.iterdir()
                if d.name.startswith(f"{model}+{scale}")
                and d.name.endswith(f"+stage-align+{seed}")
            ]
            assert (
                len(align_dirs) == 1
            ), "Multiple or No Valid Pretrained Directories Exist!"
            pretrained_checkpoint = (
                align_dirs[0] / "checkpoints" / "latest-checkpoint.pt"
            )
            if not pretrained_checkpoint.exists():
                raise FileNotFoundError(
                    f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!"
                )

        overwatch.info(
            f"Loading projector from Checkpoint `{pretrained_checkpoint}`", ctx_level=1
        )
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")[
            "model"
        ]
        self.projector.load_state_dict(model_state_dict["projector"])

    def get_fsdp_wrapping_policy(self) -> Callable:
        vision_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_policy = self.llm_backbone.get_fsdp_wrapping_policy()
        projector_policy = partial(
            _module_wrap_policy,
            module_classes={
                MLPProjector,
                ToMe16_mlp_hd64,
            },
        )
        return partial(
            _or_policy, policies=[vision_policy, llm_policy, projector_policy]
        )

    @torch.inference_mode()
    def generate(self, *args, **kwargs) -> str:
        if "eos_token_id" not in kwargs:
            kwargs["eos_token_id"] = self.eos_token_ids_to_use
        output_ids = super().generate(*args, **kwargs)
        if not kwargs.get("return_dict_in_generate", False):
            input_ids = args[0]
            if input_ids is not None:
                output_ids = output_ids[:, input_ids.shape[1] :]
            output_text = self.llm_backbone.tokenizer.decode(
                output_ids[0], skip_special_tokens=False
            ).strip()
            if self.llm_backbone.llm_family == "zamba" and "assistant\n" in output_text:
                output_text = output_text.split("assistant\n")[-1].strip()
        else:
            output_text = output_ids

        return output_text

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        is_prefill = past_key_values is None or cache_position.shape[0] != 1
        if is_prefill and kwargs.get("answer_prompt"):
            tokenizer = self.llm_backbone.tokenizer
            answer_prompt_ids = tokenizer(
                kwargs["answer_prompt"], add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(input_ids.device)
            answer_prompt_ids = answer_prompt_ids.expand(input_ids.shape[0], -1)
            answer_prompt_mask = torch.ones_like(answer_prompt_ids)
            input_ids = torch.cat([input_ids, answer_prompt_ids], dim=1)
            attention_mask = torch.cat([attention_mask, answer_prompt_mask], dim=1)
            if cache_position is not None:
                cache_position = torch.arange(
                    input_ids.shape[1], device=input_ids.device
                )

        model_inputs = self.llm_backbone.llm.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        if is_prefill:
            model_inputs["pixel_values"] = kwargs.get("pixel_values")
            model_inputs["pixel_values_videos"] = kwargs.get("pixel_values_videos")
        else:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

            if past_key_values is not None:
                layer_idx = 0
                if self.llm_backbone.llm_family == "zamba2":
                    layer_idx = getattr(
                        self.llm_backbone.llm.model, "first_transformer_layer_id", 0
                    )
                    past_length = past_key_values.get_seq_length(layer_idx=layer_idx)
                elif self.llm_backbone.llm_family == "nano":
                    layer_indices = [7, 14]  # 7: nano8b, 14: nano-9b
                    for layer_idx in layer_indices:
                        past_length = past_key_values.get_seq_length(
                            layer_idx=layer_idx
                        )
                        if past_length > 1:
                            break
                else:
                    past_length = past_key_values.get_seq_length(layer_idx=layer_idx)
                num_new_tokens = model_inputs["input_ids"].shape[1]

                model_inputs["attention_mask"] = torch.ones(
                    (input_ids.shape[0], past_length + num_new_tokens),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                model_inputs["cache_position"] = torch.arange(
                    past_length, past_length + num_new_tokens, device=input_ids.device
                )
                position_ids = model_inputs["attention_mask"].long().cumsum(-1) - 1
                position_ids.masked_fill_(model_inputs["attention_mask"] == 0, 1)
                model_inputs["position_ids"] = position_ids[:, -num_new_tokens:]

                if (
                    self.llm_backbone.llm.config._attn_implementation
                    == "flash_attention_2"
                ):
                    model_inputs["attention_mask"] = None
                else:
                    model_inputs["attention_mask"] = (
                        model_inputs["attention_mask"]
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(torch.bool)
                    )

        for key in ["answer_prompt", "multimodal_indices"]:
            model_inputs.pop(key, None)

        return model_inputs

    def embed_textpos_to_video_simplified(self, batch_size, projected_patch_embeddings):
        tokenizer = self.llm_backbone.tokenizer
        bs_frame, num_tokens, num_dimension = projected_patch_embeddings.shape
        n_frames = bs_frame // batch_size
        projected_patch_embeddings = projected_patch_embeddings.view(
            batch_size, n_frames, num_tokens, num_dimension
        )  # 1, 24, 16, 2560
        video_embeddings_with_pos = []
        time_texts = [f"This is the {i:03d}-th frame:" for i in range(n_frames)]
        output = tokenizer(
            time_texts, truncation=False, return_tensors="pt", padding=False
        )
        text_pos_encodings = self.llm_backbone.embed_input_ids(
            output.input_ids.to(self.device)
        )  # bs_frame, num_tokens, 2560
        video_embeddings_with_pos = torch.cat(
            [
                projected_patch_embeddings,
                text_pos_encodings.unsqueeze(0).expand(batch_size, -1, -1, -1),
            ],
            dim=2,
        )  # 1, 24, 16+30, 2560
        return video_embeddings_with_pos  # xx, 2560

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: GenericLLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        visual_token_order="raw",
    ) -> "GenericTimeViperVLM":
        vlm = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            visual_token_order=visual_token_order,
        )

        pretrained_weights = torch.load(pretrained_checkpoint, map_location="cpu")

        vlm.load_state_dict(
            pretrained_weights,
            strict=True,
        )

        vlm.requires_grad_(False)
        vlm.eval()

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        vlm.to(device, dtype=dtype)

        return vlm

    def get_input_embeddings(self):
        return self.llm_backbone.llm.get_input_embeddings()

    def get_output_embeddings(self):
        return self.llm_backbone.llm.get_output_embeddings()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if not self.supports_gradient_checkpointing:
            raise ValueError(
                f"{self.__class__.__name__} does not support gradient checkpointing."
            )

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": True}

        gradient_checkpointing_func = partial(
            checkpoint, **gradient_checkpointing_kwargs
        )

        _is_using_old_format = (
            "value" in inspect.signature(self._set_gradient_checkpointing).parameters
        )

        if not _is_using_old_format:
            self._set_gradient_checkpointing(
                enable=True, gradient_checkpointing_func=gradient_checkpointing_func
            )
        else:
            self.apply(partial(self._set_gradient_checkpointing, value=True))

        if getattr(self, "_hf_peft_config_loaded", False):
            self.enable_input_require_grads()

    def enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)

        self._require_grads_hook = self.get_input_embeddings().register_forward_hook(
            make_inputs_require_grads
        )

    def _set_gradient_checkpointing(
        self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint
    ):
        is_gradient_checkpointing_set = False

        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing."
            )
