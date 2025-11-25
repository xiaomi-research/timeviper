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

import glob
import logging

# ==============================================================================
# Part 1: Imports and Configuration
# ==============================================================================
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from decord import VideoReader
from PIL import Image
from transformers import ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

from timeviper.data.conversation import SeparatorStyle
from timeviper.data.data_utils import (
    captioning_templates,
    temporal_video_grounding_templates,
    timestamp_prompt,
)
from timeviper.utils.train_utils import load_json_data

logger = logging.getLogger(__name__)


@dataclass
class MultimodalTokenConfig:
    """Centralized Configuration for Special Tokens."""

    IGNORE_INDEX: int = -100
    DEFAULT_IMAGE_TOKEN: str = "<image>"
    DEFAULT_VIDEO_TOKEN: str = "<video>"
    MMODAL_TOKEN_SEP: str = "\n"

    # Global instance
    @classmethod
    def set_config(cls, **kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k.upper()):
                setattr(cls, k.upper(), v)
                logger.info(f"Updated {k.upper()} to {v}")


# ==============================================================================
# Part 2: Utility Functions
# ==============================================================================


def get_resize_output_image_size(
    height: int, width: int, shortest_edge: int, longest_edge: int
) -> Tuple[int, int]:
    if shortest_edge is None and longest_edge is None:
        return height, width

    aspect_ratio = width / height
    if width >= height and width > longest_edge:
        width = longest_edge
        height = int(width / aspect_ratio)
    elif height > width and height > longest_edge:
        height = longest_edge
        width = int(height * aspect_ratio)

    height = max(height, shortest_edge)
    width = max(width, shortest_edge)
    return height, width


def load_image_from_path(image_path: Union[str, Image.Image]) -> List[Image.Image]:
    if isinstance(image_path, Image.Image):
        return [image_path]
    return [Image.open(image_path).convert("RGB")]


def get_frame_indices(
    num_frames: int,
    vlen: int,
    start_frame: float = None,
    end_frame: float = None,
    sample: str = "rand",
    input_fps: float = 1.0,
    max_num_frames: int = -1,
) -> List[int]:
    """Optimized frame sampling logic."""
    if sample in ["rand", "middle"]:
        # Calculate range
        start_idx = max(0, math.ceil(start_frame)) if start_frame is not None else 0
        end_idx = min(math.floor(end_frame), vlen) if end_frame is not None else vlen

        if start_idx >= end_idx:
            logger.warning(
                f"Invalid frame range: {start_idx} to {end_idx}, using full video."
            )
            start_idx, end_idx = 0, vlen

        acc_samples = min(num_frames, end_idx - start_idx)
        intervals = np.linspace(start_idx, end_idx, acc_samples + 1).astype(int)
        ranges = [
            (intervals[i], intervals[i + 1] - 1) for i in range(len(intervals) - 1)
        ]

        if sample == "rand":
            try:
                indices = [
                    random.choice(range(x[0], x[1] + 1)) if x[1] >= x[0] else x[0]
                    for x in ranges
                ]
            except ValueError:
                indices = sorted(np.random.permutation(vlen)[:acc_samples].tolist())
        else:  # middle
            indices = [(x[0] + x[1]) // 2 for x in ranges]

        # Pad if necessary
        if len(indices) < num_frames:
            indices.extend([indices[-1]] * (num_frames - len(indices)))
        return indices

    elif "fps" in sample:
        output_fps = float(sample[3:])
        duration = (
            (end_frame - start_frame) / input_fps
            if start_frame is not None
            else vlen / input_fps
        )
        delta = 1 / output_fps
        frame_seconds = np.arange(delta / 2, duration + delta / 2, delta)
        indices = np.around(frame_seconds * input_fps).astype(int)
        indices = [idx for idx in indices if idx < vlen]

        if 0 < max_num_frames < len(indices):
            indices = indices[:max_num_frames]
        return list(indices)

    raise ValueError(f"Unknown sample method: {sample}")


# ==============================================================================
# Part 3: Core Dataset Logic (Base Class)
# ==============================================================================


class BaseMultimodalDataset(torch.utils.data.Dataset):
    """
    Base class handling media loading, tokenization, and label creation.
    Concrete subclasses should implement/override prompt generation strategies.
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        data_items: List[Dict],
        data_dir: str,
        conv_format: Any,
        dataset_cfg: Dict,
        logit_cache_path: Optional[str] = None,
    ):
        self.processor = processor
        self.data_dir = data_dir
        self.conv = conv_format.copy()
        self.data = data_items
        self.logit_cache_path = logit_cache_path

        # Unpack config
        self.max_txt_seq_len = dataset_cfg.get("max_txt_seq_len", 2048)
        self.do_resize = dataset_cfg.get("do_resize", True)
        self.img_min_edge = dataset_cfg.get("img_shortest_edge", None)
        self.img_max_edge = dataset_cfg.get("img_longest_edge", None)
        self.num_tries = dataset_cfg.get("num_tries", 5)
        self.dataset_type = dataset_cfg.get("dataset_type", "image")

        # Video params
        self.video_params = {
            "sample_type": dataset_cfg.get("video_sample_type", "rand"),
            "num_frames": dataset_cfg.get("video_num_frames", 8),
            "max_frames": dataset_cfg.get("max_num_frames_fps", 32),
            "min_frames": dataset_cfg.get("min_num_frames_fps", 8),
            "use_encoder": dataset_cfg.get("use_video_encoder", False),
            "frames_per_clip": dataset_cfg.get("video_frames_per_clip", 4),
            "resolution": dataset_cfg.get("resolution", "raw"),
            "temporal_patch_size": dataset_cfg.get("temporal_patch_size", 1),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.process_item(self.data[idx])

    def process_item(self, item: Dict) -> Dict:
        """Central pipeline for processing a single item."""
        # Retry mechanism wrapper
        for _ in range(self.num_tries):
            try:
                return self._process_item_internal(item)
            except Exception as e:
                logger.warning(
                    f"Error processing item {item.get('id', 'unknown')}: {e}. Retrying..."
                )
                item = random.choice(self.data)

        raise RuntimeError(f"Failed to process item after {self.num_tries} attempts.")

    def _process_item_internal(self, item: Dict) -> Dict:
        media_paths = self._get_media_paths(item)
        # 1. Load Media
        durations = None
        if "video" in self.dataset_type:
            # Assumes single video path usually
            frames_list, durations = self._load_video_safe(
                media_paths[0], item.get("video_start"), item.get("video_end")
            )
            media_content = frames_list
            default_token = MultimodalTokenConfig.DEFAULT_VIDEO_TOKEN
            num_units = len(frames_list[0])
        else:
            # Image
            images = self._load_images_safe(media_paths)
            media_content = images
            default_token = MultimodalTokenConfig.DEFAULT_IMAGE_TOKEN
            num_units = len(images)

        # 2. Build Conversation (Delegated to subclass/method)
        conv_messages = self.build_conversation(item, num_units, default_token)

        # 3. Tokenize
        self.conv.messages = conv_messages
        prompt = self.conv.get_prompt()

        encoding = self.processor(
            prompt,
            media_content,
            return_tensors="pt",
            truncation=True,
            do_resize=self.do_resize,
            max_length=self.max_txt_seq_len,
        )

        # Post-processing encoding
        if "image_patches" in encoding:
            encoding.pop("attention_mask", None)
            encoding["image_patches"] = encoding["image_patches"][0]

        # 4. Labeling
        encoding = self._create_labels(encoding, prompt)

        # 5. Optional Logits Cache
        if self.logit_cache_path and "logits" in item:
            encoding["logits_cache"] = torch.load(
                os.path.join(self.logit_cache_path, item["logits"])
            )

        return encoding

    def build_conversation(
        self, item: Dict, num_frames: int, token_str: str
    ) -> List[Tuple[str, str]]:
        """
        To be implemented/overridden by subclasses or mixins.
        Default implementation for standard conversation.
        """
        conv = self.conv.copy()
        conv.messages = []
        token_str = token_str * num_frames if "<video>" in token_str else token_str

        for i, sentence in enumerate(item["conversations"]):
            role = conv.roles[0] if sentence["from"] == "human" else conv.roles[1]
            value = sentence["value"]

            if i == 0 and sentence["from"] == "human":
                if "<image>" in value:
                    value = value.replace("<image>", token_str)
                else:
                    value = (
                        f"{token_str}{MultimodalTokenConfig.MMODAL_TOKEN_SEP}{value}"
                    )
            elif sentence["from"] == "human" and "<image>" in value:
                value = value.replace("<image>", token_str)

            # Auto-append "Describe" if first message is not human (edge case)
            if i == 0 and sentence["from"] != "human":
                conv.append_message(
                    conv.roles[0],
                    f"{token_str}{MultimodalTokenConfig.MMODAL_TOKEN_SEP}Describe the content.",
                )

            conv.append_message(role, value)
        return conv.messages.copy()

    def _get_media_paths(self, item: Dict) -> List[str]:
        if "image" not in item:
            # Fallback for video keys if they differ
            raise KeyError(f"Item {item.get('id')} missing 'image' key for path.")

        # Handle "path1 | path2" format
        paths = [p.strip() for p in item["image"].split(" | ")]
        return [os.path.join(self.data_dir, p) for p in paths]

    def _resize_media(self, image: Image.Image) -> Image.Image:
        """Handles resizing logic for both standard resize and small-image upscaling."""
        w, h = image.size

        # 1. Upscale if too small (minimum 16x16 requirement for many ViTs)
        if w < 16 or h < 16:
            scale = max(16 / w, 16 / h)
            image = image.resize((int(w * scale), int(h * scale))).convert("RGB")
            w, h = image.size  # Update

        # 2. Downscale/Resize if configured and we are handling it manually
        if not self.do_resize and self.img_min_edge and self.img_max_edge:
            h_new, w_new = get_resize_output_image_size(
                h, w, self.img_min_edge, self.img_max_edge
            )
            image = image.resize((w_new, h_new), resample=Image.Resampling.LANCZOS)

        return image

    def _load_images_safe(self, paths: List[str]) -> List[Image.Image]:
        images = []
        for p in paths:
            imgs = load_image_from_path(p)
            images.extend([self._resize_media(img) for img in imgs])
        return images

    def _load_video_safe(
        self, path: str, start: float, end: float
    ) -> Tuple[List[List[Image.Image]], List[float]]:
        # Handle folder of images acting as video
        if os.path.isdir(path):
            return self._load_video_from_folder(path)

        # Handle video file
        try:
            vr = VideoReader(path, num_threads=1)
        except:
            # Fallback logic for downsampled versions could go here
            raise RuntimeError(f"Could not open video {path}")

        vlen = len(vr)
        fps = vr.get_avg_fps()

        # Determine number of frames
        num_frames = self.video_params["num_frames"]
        if num_frames == "auto":
            duration = (
                (end - start)
                if (start is not None and end is not None)
                else (vlen / fps)
            )
            num_frames = int(duration * 1.0)  # 1 fps default
            num_frames = max(
                min(num_frames, self.video_params["max_frames"]),
                self.video_params["min_frames"],
            )
        else:
            num_frames = int(num_frames)

        num_frames = min(num_frames, vlen)
        if self.video_params["use_encoder"]:
            clip_size = self.video_params["frames_per_clip"]
            num_frames = (num_frames // clip_size) * clip_size

        indices = get_frame_indices(
            num_frames,
            vlen,
            start_frame=start * fps if start else None,
            end_frame=end * fps if end else None,
            sample=self.video_params["sample_type"],
            input_fps=fps,
        )

        durations = [i / fps for i in indices]
        frames_np = vr.get_batch(indices).asnumpy()
        frames = [self._resize_media(Image.fromarray(f, mode="RGB")) for f in frames_np]

        return [frames], durations

    def _load_video_from_folder(
        self, path: str
    ) -> Tuple[List[List[Image.Image]], List[float]]:
        # Simplified logic for folder loading
        files = sorted(
            glob.glob(os.path.join(path, "*"))
        )  # Add extension filter in real use
        if not files:
            raise ValueError(f"Empty folder {path}")

        # Uniform sampling
        target_num = 8
        indices = np.linspace(0, len(files) - 1, target_num).astype(int)
        selected_files = [files[i] for i in indices]

        frames = []
        for p in selected_files:
            with Image.open(p) as img:
                frames.append(self._resize_media(img.convert("RGB")))

        return [frames], [float(i) for i in range(len(frames))]

    def _create_labels(self, encoding: BatchFeature, prompt_text: str) -> BatchFeature:
        if "labels" in encoding:
            return encoding

        labels = torch.full_like(
            encoding["input_ids"], MultimodalTokenConfig.IGNORE_INDEX
        )
        input_ids = encoding["input_ids"][0]
        target = labels[0]

        # Masking logic based on separator style (simplified for Qwen2/Nano)
        if self.conv.sep_style in [SeparatorStyle.NANO, SeparatorStyle.QWEN2]:
            sep_id = self.processor.tokenizer.convert_tokens_to_ids(self.conv.sep)
            sep_idxs = torch.nonzero(input_ids == sep_id, as_tuple=True)[0].tolist()

            skip = 0 if self.conv.system else 1  # Skip system prompt if exists

            for i in range(len(sep_idxs)):
                if i % 2 == skip:
                    continue  # Mask user input
                start = sep_idxs[i] + 1
                end = sep_idxs[i + 1] + 1 if i + 1 < len(sep_idxs) else len(input_ids)
                target[start:end] = input_ids[start:end]

        encoding["labels"] = labels
        return encoding


# ==============================================================================
# Part 4: Specific Strategies (Subclasses)
# ==============================================================================


class CaptioningDataset(BaseMultimodalDataset):
    def build_conversation(
        self, item: Dict, num_frames: int, token_str: str
    ) -> List[Tuple[str, str]]:
        conv = self.conv.copy()
        conv.messages = []

        # Adjust token
        token_str = token_str * num_frames if "<video>" in token_str else token_str

        template = random.choice(captioning_templates["user"])

        user_msg = f"{token_str}{MultimodalTokenConfig.MMODAL_TOKEN_SEP}{template.format(self.dataset_type)}"
        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], item["caption"])
        return conv.messages


class TemporalVideoGroundingDataset(BaseMultimodalDataset):
    def __init__(self, *args, **kwargs):
        self.use_template = kwargs.get("dataset_cfg", {}).get("use_template", True)
        super().__init__(*args, **kwargs)

    def build_conversation(
        self, item: Dict, num_frames: int, token_str: str
    ) -> List[Tuple[str, str]]:
        conv = self.conv.copy()
        conv.messages = []
        token_str = token_str * num_frames

        template = (
            random.choice(temporal_video_grounding_templates["user"])
            if self.use_template
            else "{}"
        )

        for i, sentence in enumerate(item["conversations"]):
            role = conv.roles[0] if sentence["from"] == "human" else conv.roles[1]
            value = sentence["value"]

            if i % 2 == 0 and sentence["from"] == "human":
                ts_prompt = timestamp_prompt.format(item["duration"], num_frames)
                if "<image>" in value:
                    val_clean = value.replace("<image>", token_str)
                    value = f"{val_clean}{MultimodalTokenConfig.MMODAL_TOKEN_SEP}{ts_prompt}"
                else:
                    value = f"{token_str}{MultimodalTokenConfig.MMODAL_TOKEN_SEP}{ts_prompt}{template.format(value)}"

            conv.append_message(role, value)
        return conv.messages


class DenseVideoCaptioningDataset(BaseMultimodalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_conversation(
        self, item: Dict, num_frames: int, token_str: str
    ) -> List[Tuple[str, str]]:
        conv = self.conv.copy()
        conv.messages = []
        token_str = token_str * num_frames

        for i, sentence in enumerate(item["conversations"]):
            role = conv.roles[0] if sentence["from"] == "human" else conv.roles[1]
            value = sentence["value"]

            if i % 2 == 0 and sentence["from"] == "human":
                ts_prompt = timestamp_prompt.format(item["duration"], num_frames)
                if "<image>" in value:
                    val_clean = value.replace("<image>", token_str)
                    value = f"{val_clean}{MultimodalTokenConfig.MMODAL_TOKEN_SEP}{ts_prompt}"
                else:
                    value = f"{token_str}{MultimodalTokenConfig.MMODAL_TOKEN_SEP}{ts_prompt}{value}"

            conv.append_message(role, value)
        return conv.messages


# ==============================================================================
# Part 5: Packing Wrapper
# ==============================================================================


class PackedDataset(torch.utils.data.Dataset):
    """
    Wraps ANY BaseMultimodalDataset to support packing (multiple conversations per sample).
    """

    def __init__(self, dataset: BaseMultimodalDataset, pack_size: int):
        self.dataset = dataset
        self.pack_size = pack_size
        self.original_indices = list(range(len(dataset)))
        # Create packed batches of indices
        self.packed_indices = [
            self.original_indices[i : i + pack_size]
            for i in range(0, len(self.original_indices), pack_size)
        ]

    def __len__(self):
        return len(self.packed_indices)

    def __getitem__(self, idx: int) -> BatchFeature:
        indices = self.packed_indices[idx]

        # Process each item
        items = [self.dataset.process_item(self.dataset.data[i]) for i in indices]

        # Merge results
        merged = {}
        for item in items:
            for k, v in item.items():
                if k not in merged:
                    merged[k] = []
                merged[k].append(v)

        final_batch = {}
        for k, v_list in merged.items():
            if k == "input_ids":
                final_batch["txt_seq_lens"] = [v.shape[1] for v in v_list]
            if k in ["pixel_values", "pixel_values_videos"]:
                final_batch[k] = torch.cat(v_list, dim=0)  # Batch dimension stack
            elif k in ["input_ids", "labels", "durations"]:
                final_batch[k] = torch.cat(v_list, dim=1)  # Sequence dimension cat
            # Skip others like attention_mask which are handled by collator usually

        return BatchFeature(final_batch)


# ==============================================================================
# Part 6: Collator
# ==============================================================================


class Qwen2VLCollator:
    def __init__(self, processor):
        self.processor = processor
        self.pad_id = processor.tokenizer.pad_token_id
        self.ignore_id = MultimodalTokenConfig.IGNORE_INDEX

    def __call__(self, batch: List[Dict]) -> Dict:
        # Standardize input to list of lists if coming from packing
        # This handles both packed and non-packed inputs if they follow the shape convention
        input_ids = [
            x["input_ids"] if x["input_ids"].dim() == 2 else x["input_ids"].unsqueeze(0)
            for x in batch
        ]
        labels = [
            x["labels"] if x["labels"].dim() == 2 else x["labels"].unsqueeze(0)
            for x in batch
        ]

        # Flatten for padding
        input_ids = [t.squeeze(0) for t in input_ids]
        labels = [t.squeeze(0) for t in labels]

        # Pad sequence
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_id
        )

        # Create Attention Mask
        attention_mask = input_ids.ne(self.pad_id) if len(batch) == 1 else None

        # Collect Media Tensors
        def collect_tensor(key):
            tensors = [x[key] for x in batch if key in x and x[key] is not None]
            return torch.cat(tensors, dim=0) if tensors else None

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": collect_tensor("pixel_values"),
            "pixel_values_videos": collect_tensor("pixel_values_videos"),
            "image_grid_thw": collect_tensor("image_grid_thw"),
            "video_grid_thw": collect_tensor("video_grid_thw"),
            "logits_cache": None,  # Handle if needed
            "txt_seq_lens": (
                batch[0]["txt_seq_lens"]
                if "txt_seq_lens" in batch[0]
                else input_ids.shape[1]
            ),
        }


# ==============================================================================
# Part 7: Factory
# ==============================================================================


class DatasetFactory:
    _REGISTRY = {
        "caption": CaptioningDataset,
        "conversation": BaseMultimodalDataset,
        "tvg": TemporalVideoGroundingDataset,
        "dvc": DenseVideoCaptioningDataset,
    }

    @staticmethod
    def create(data_args, processor):
        with open(data_args.data_config_file, "r") as f:
            config = yaml.safe_load(f)

        datasets = {"train": [], "val": []}

        for cfg in config["data"]:
            fmt = cfg["format"]
            # Determine base class
            base_cls = DatasetFactory._REGISTRY.get(
                fmt.split("_")[0], BaseMultimodalDataset
            )

            # Load Data
            json_data = load_json_data(cfg["json_path"])
            if cfg.get("max_size"):
                json_data = json_data[: cfg["max_size"]]
            if cfg.get("sample_ratio", 1.0) > 1.0:
                # Logic for upsampling
                pass

            # Create Instance
            ds_instance = base_cls(
                processor=processor,
                data_items=json_data,
                data_dir=cfg["data_path"],
                conv_format=data_args.conv_format,
                dataset_cfg=cfg,
            )

            # Apply Packing if needed
            if "packed" in fmt or cfg.get("pack_size", 1) > 1:
                ds_instance = PackedDataset(
                    ds_instance, pack_size=cfg.get("pack_size", 1)
                )
            split = cfg.get("split", "train")
            datasets[split].append(ds_instance)

        # Combine
        train_ds = (
            torch.utils.data.ConcatDataset(datasets["train"])
            if datasets["train"]
            else None
        )
        val_ds = (
            torch.utils.data.ConcatDataset(datasets["val"]) if datasets["val"] else None
        )

        return train_ds, val_ds, Qwen2VLCollator(processor)


def load_data_from_config(data_args, processor):
    return DatasetFactory.create(data_args, processor)
