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


import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import deepspeed
import regex as re
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from transformers.hf_argparser import HfArgumentParser

import timeviper.utils.torch_load_patch
from hf_mtask_trainer import HfMultiTaskTrainer
from timeviper.data import MultimodalTokenConfig, load_data_from_config
from timeviper.data.conversation import conv_templates
from timeviper.data.image_processing import ImageProcessor
from timeviper.data.processor import Qwen2VLProcessor
from timeviper.model import (
    get_llm_backbone_and_tokenizer,
    get_vision_backbone_and_transform,
    get_vlm,
)
from timeviper.utils.train_utils import (
    _right_pad_inputs_with_attention_mask,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    load_ckpt_from_deepspeed_zero,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)


@dataclass
class TrainingArguments(TrainingArguments):
    group_by_modality_length: bool = field(default=False)
    llm_backbone_lr: Optional[float] = field(
        default=1e-5,
        metadata={"help": "The learning rate for the LLM backbone."},
    )
    projector_lr: Optional[float] = field(
        default=1e-5,
        metadata={"help": "The learning rate for the projector."},
    )
    vision_backbone_lr: Optional[float] = field(
        default=1e-5,
        metadata={"help": "The learning rate for the vision backbone."},
    )
    merge_modules_lr: Optional[float] = field(
        default=None,
        metadata={"help": "The learning rate for the merge modules."},
    )
    use_zero3: Optional[bool] = field(
        default=False,
    )


@dataclass
class DataArguments:
    max_img_seq_len: Optional[int] = field(
        metadata={
            "help": "The maximum number of image sequence length after tokenization. Sequences longer "
            "than this will be truncated.",
            "default": 1024,
            "required": False,
        },
        default=20000,  # actual number
    )
    max_txt_seq_len: Optional[int] = field(
        metadata={
            "help": "The maximum number of text sequence length after tokenization. Sequences longer "
            "than this will be truncated.",
            "default": 1024,
            "required": False,
        },
        default=1024,
    )
    use_video_encoder: Optional[bool] = field(
        metadata={
            "help": "Whether to use video encoder",
            "default": False,
            "required": False,
        },
        default=False,
    )
    video_frames_per_clip: Optional[int] = field(
        metadata={
            "help": "The number of video frames per clip",
            "default": 4,
            "required": False,
        },
        default=4,
    )
    data_config_file: Optional[str] = field(
        metadata={
            "help": "Pretrained config name or path if not the same as model_name",
            "default": None,
            "required": False,
        },
        default=None,
    )
    dataset_balancing: Optional[bool] = field(
        metadata={
            "help": "Whether to balance the dataset",
            "default": True,
            "required": False,
        },
        default=False,
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models",
            "default": "llava-hf/llava-1.5-7b-hf",
            "required": False,
        },
        default="llava-hf/llava-1.5-7b-hf",
    )
    trainable_modules: Optional[str] = field(
        metadata={"help": "The modules to train", "default": "all", "required": False},
        default="all",
    )
    lora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use LoRA", "default": False, "required": False},
        default=False,
    )
    qlora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use QLoRA", "default": False, "required": False},
        default=False,
    )
    dora_enabled: Optional[bool] = field(
        metadata={"help": "Whether to use Dora", "default": False, "required": False},
        default=False,
    )
    lora_r: Optional[int] = field(
        metadata={"help": "LoRA r", "default": 128, "required": False},
        default=128,
    )
    lora_alpha: Optional[float] = field(
        metadata={"help": "LoRA alpha", "default": 256, "required": False},
        default=256,
    )
    lora_dropout: Optional[float] = field(
        metadata={"help": "LoRA dropout", "default": 0.05, "required": False},
        default=0.05,
    )
    lora_bias: Optional[str] = field(
        metadata={"help": "LoRA bias", "default": "none", "required": False},
        default="none",
    )
    attn_implementation: Optional[str] = field(
        metadata={
            "help": "The attention implementation to use, choose from 'eager', 'sdpa', 'flash_attention_2'",
            "default": "flash_attention_2",
            "required": False,
        },
        default="flash_attention_2",
    )
    max_image_size: Optional[str] = field(
        metadata={
            "help": "The maximum image size",
            "default": "(1080,1920)",
            "required": False,
        },
        default="(1080,1920)",
    )
    conv_template: Optional[str] = field(
        metadata={
            "help": "The conversation template to use",
            "default": None,
            "required": False,
        },
        default=None,
    )
    init_cross_attn_weights_from_self_attn: Optional[bool] = field(
        metadata={
            "help": "Whether to initialize cross attention weights from self attention layers",
            "default": False,
            "required": False,
        },
        default=False,
    )
    init_cross_attn_weights_from_nearest_self_attn: Optional[bool] = field(
        metadata={
            "help": "Whether to initialize cross attention weights from the nearest self attention layers",
            "default": False,
            "required": False,
        },
        default=False,
    )
    model_id: Optional[str] = field(
        default="cobra-siglip+3b",
        metadata={"help": "Identifier for the model, used for saving/loading"},
    )
    arch_specifier: Optional[str] = field(default="no-align+gelu-mlp")
    vision_backbone_id: Optional[str] = field(default="siglip-vit-so400m-384px")
    llm_backbone_id: Optional[str] = field(default="mamba-2.8b-zephyr")
    image_resize_strategy: Optional[str] = field(default="resize-naive")
    hf_token: Optional[Union[str, Path]] = field(default=Path(".hf_token"))
    llm_max_length: Optional[int] = field(default=None)
    enable_mixed_precision_training: Optional[bool] = field(
        default=True,
    )
    # for pdrop
    use_pdrop: Optional[bool] = field(default=False)
    pdrop_type: Optional[str] = field(
        metadata={
            "help": "pdrop type, e.g., 'uni_14_0.8-attn_21_0.6-attn_30_0.4-attn_39_0.2'",
        },
        default=None,
    )
    merge_module: Optional[str] = field(default="no_merge")
    visual_token_order: Optional[str] = field(default="raw")


def save_training_artifacts(trainer, model, processor, model_args, output_dir):
    state_dict = None
    non_lora_state_dict = None

    if model_args.lora_enabled:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), model_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )

    if trainer.is_world_process_zero():
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model artifacts to {output_dir}")
        processor.save_pretrained(output_dir)

        if model_args.lora_enabled:
            model.config.save_pretrained(output_dir)
            model.save_pretrained(output_dir, state_dict=state_dict)
            if non_lora_state_dict:
                torch.save(
                    non_lora_state_dict,
                    os.path.join(output_dir, "non_lora_trainables.bin"),
                )
        else:
            trainer.save_model(output_dir=output_dir)

        try:
            if hasattr(model, "llm_backbone"):
                model.llm_backbone.tokenizer.save_pretrained(output_dir)
                model.llm_backbone.llm.save_pretrained(output_dir)
        except Exception as e:
            print(f"Failed to save LLM backbone: {e}")


def find_all_linear_names_llama(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["projector", "vision_backbone"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def load_all_modules(use_zero3):
    # Step 1: load vision tower
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_args.vision_backbone_id,
        image_resize_strategy=model_args.image_resize_strategy,
        use_zero3=use_zero3,
    )

    # Step 2: load llm
    continue_pretrain_ckpt = None
    if (
        model_args.model_name_or_path
        and model_args.model_name_or_path.endswith(".bin")
        and "stage3" in training_args.run_name.lower()
    ):
        continue_pretrain_ckpt = "/".join(model_args.model_name_or_path.split("/")[:-1])

    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_args.llm_backbone_id,
        llm_max_length=model_args.llm_max_length,
        attn_implementation=model_args.attn_implementation,
        continue_pretrain_ckpt=continue_pretrain_ckpt,
        merge_module=model_args.merge_module,
        use_pdrop=model_args.use_pdrop,
        pdrop_type=model_args.pdrop_type,
    )

    model = get_vlm(
        model_args.model_id,
        model_args.arch_specifier,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=model_args.enable_mixed_precision_training,
        visual_token_order=model_args.visual_token_order,
    )

    return model, tokenizer, image_transform


def load_model(model_args, training_args):
    print("Loading model...")

    ds_config = training_args.deepspeed
    if training_args.use_zero3:
        with deepspeed.zero.Init():
            model, tokenizer, image_transform = load_all_modules(use_zero3=True)
    else:
        model, tokenizer, image_transform = load_all_modules(use_zero3=False)

    if model_args.model_name_or_path and model_args.model_name_or_path.endswith(".bin"):
        ckpt = torch.load(model_args.model_name_or_path, map_location="cpu")
        ckpt = {k: v for k, v in ckpt.items() if "projector" in k}
        if training_args.use_zero3:
            load_ckpt_from_deepspeed_zero(model, ckpt, strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    if model_args.init_cross_attn_weights_from_self_attn:
        try:
            model.llm_backbone.llm.init_cross_attn_from_self_attn()
        except Exception as e:
            print(f"Failed to initialize cross attention weights: {e}")
    if model_args.init_cross_attn_weights_from_nearest_self_attn:
        try:
            model.llm_backbone.llm.init_merge_modules_from_nearest_self_attn()
        except Exception as e:
            print(f"Failed to initialize cross attention weights: {e}")

    image_processor = ImageProcessor(patch_size=14, image_transforms=image_transform)
    processor = Qwen2VLProcessor(image_processor, tokenizer, model_config=model.config)

    print("\n[Step 6] Applying freezing/unfreezing logic...")
    for param in model.parameters():
        param.requires_grad = False

    if model_args.lora_enabled:
        llm_family = model.llm_backbone.llm_family
        if llm_family in ["qwen2", "nano"]:
            find_all_linear_names = find_all_linear_names_llama
        else:
            raise ValueError(
                f"LoRA is not supported for {model_args.llm_backbone_id} backbone"
            )
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
            use_dora=model_args.dora_enabled,
        )
        print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        print("Successfully added LoRA adapters")
    if model_args.trainable_modules and model_args.trainable_modules.lower() != "all":
        trainable_modules = [x.strip() for x in model_args.trainable_modules.split(",")]
        print(f"Unfreezing parameters for specified modules: {trainable_modules}")
        for name, param in model.named_parameters():
            if (
                any(trainable_module in name for trainable_module in trainable_modules)
                and "base_layer" not in name
            ):
                param.requires_grad = True
        if "vision_backbone" in model_args.trainable_modules.lower():
            model.vision_backbone_requires_grad = True
    else:
        print("Tuning all modules, unfreezing all parameters.")
        for param in model.parameters():
            param.requires_grad = True

    # Print a summary of trainable parameters for verification
    print("\n--- Trainable Parameters Summary ---")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        print(f"Total parameters: {total_params / 1e9:.2f}G")
        print(f"Trainable parameters: {trainable_params / 1e9:.2f}G")
        print(f"Trainable percentage: {100 * trainable_params / total_params:.4f}%")
    except:
        print("Could not compute parameter summary due to an error.")

    MultimodalTokenConfig.set_config(
        DEFAULT_IMAGE_TOKEN="<image>",
        DEFAULT_VIDEO_TOKEN="<image>",
        MMODAL_TOKEN_SEP="",
        IGNORE_INDEX=-100,
    )

    return model, processor


def main(
    training_args: TrainingArguments,
    data_args: DataArguments,
    model_args: ModelArguments,
):
    training_args.output_dir = (
        Path(training_args.output_dir)
        / model_args.model_name_or_path.split("/")[-1]
        / training_args.run_name
    )
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.output_dir = str(training_args.output_dir)

    training_args.remove_unused_columns = False
    data_args.is_master_worker = training_args.local_rank in [-1, 0]

    data_args.extra_collator_func = _right_pad_inputs_with_attention_mask
    if not training_args.resume_from_checkpoint:
        training_args.resume_from_checkpoint = True
    if training_args.resume_from_checkpoint is True:
        # search for the latest checkpoint
        all_checkpoints = list(Path(training_args.output_dir).glob("checkpoint-*"))
        all_checkpoints = [
            x
            for x in all_checkpoints
            if (x / "trainer_state.json").exists() and not x.name.endswith("final")
        ]
        if len(all_checkpoints) == 0:
            training_args.resume_from_checkpoint = None
            print("No checkpoint found, starting from scratch")
        else:
            all_checkpoints = [str(x) for x in all_checkpoints]
            latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
            training_args.resume_from_checkpoint = latest_checkpoint
            print("Resuming from checkpoint", latest_checkpoint)

    model, processor = load_model(model_args, training_args)

    data_args.model_patch_size = processor.image_processor.patch_size
    data_args.temporal_patch_size = processor.image_processor.temporal_patch_size

    if model_args.conv_template:
        data_args.conv_format = conv_templates[model_args.conv_template]
    else:
        data_args.conv_format = conv_templates["default"]
    if data_args.data_config_file is not None:
        train_dataset, val_dataset, collate_fn = load_data_from_config(
            data_args, processor
        )
    else:
        raise ValueError("Data config file is required")
    trainer = HfMultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor,
        model_args=model_args,
    )
    data_args.is_master_worker = trainer.is_world_process_zero()
    if trainer.is_world_process_zero():
        redirect_output(os.path.join(training_args.output_dir, "train.log"))
        print(f"=============================================")
        print(f"Total length of the training dataset is: {len(train_dataset)}")

        if dist.is_initialized():
            world_size = dist.get_world_size()
            print(f"torch.distributed WORLD_SIZE: {world_size}")
            world_size_args = training_args.world_size
            print(f"TrainingArguments WORLD_SIZE: {world_size_args}")
        else:
            print("Distributed environment is not initialized.")
            world_size = 1

        print(f"=============================================")

    if trainer.is_world_process_zero():
        print("Training arguments:")
        print(training_args)
        print("Data arguments:")
        print(data_args)
        print("Model arguments:")
        print(model_args)

    if training_args.do_train:
        final_checkpoint_dir = os.path.join(
            training_args.output_dir, "checkpoint-final"
        )
        try:
            trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
            save_training_artifacts(
                trainer, model, processor, model_args, final_checkpoint_dir
            )
        except Exception as e:
            print(f"Training interrupted with exception: {e}")
            try:
                save_training_artifacts(
                    trainer, model, processor, model_args, final_checkpoint_dir
                )
                if trainer.is_world_process_zero():
                    torch.save(
                        model.state_dict(),
                        os.path.join(final_checkpoint_dir, "pytorch_model_manual.bin"),
                    )
            except Exception as save_error:
                print(f"Failed to save emergency checkpoint: {save_error}")
            raise e


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()  # Ensure it gets written immediately

    def flush(self):
        for file in self.files:
            file.flush()


def redirect_output(output_file):
    """
    Redirects terminal output (stdout and stderr) to both the terminal and a file.

    Parameters:
    output_file (str): Path to the file where the output will be saved.
    """
    # Open the file in write mode
    log_file = open(output_file, "w")

    # Create a Tee object that writes to both the terminal (sys.__stdout__) and the log file
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = sys.stdout


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, DataArguments, ModelArguments))
    training_args, data_args, model_args = parser.parse_args_into_dataclasses()
    main(training_args, data_args, model_args)
