import json
import os
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass
from multiprocessing import Manager
from typing import Iterable, List, Literal, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from eval.utils import load_decord
from timeviper.data.conversation import Conversation
from timeviper.data.data_utils import temporal_video_grounding_templates
from timeviper.data.processor import Qwen2VLProcessor

from .data_loader import *

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_TOKEN_ID = None  # should be set when loading the processor
DEFAULT_VIDEO_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN_ID = None  # should be set when loading the processor
MMODAL_TOKEN_SEP = ""


class Qwen2VLMultiInputCollator:
    def __init__(self, processor=None):
        self.processor = processor

    def __call__(self, batch):
        all_input_ids = []
        # all_labels = []
        # all_position_ids = []
        all_attention_masks = []
        max_input_ids_len = max([x["input_ids"].shape[1] for x in batch])
        utils = {}
        for x in batch:
            for key in x.keys():
                if key in [
                    "options",
                    "answer",
                    "qid",
                    "video_paths",
                    "task_type",
                    "duration",
                    "timestamps",
                ]:
                    if key not in utils:
                        utils[key] = []
                    utils[key].append(x[key])
            input_ids = x["input_ids"]
            seq_len = input_ids.shape[1]
            pad_len = max_input_ids_len - seq_len

            all_input_ids.append(
                torch.cat(
                    [
                        self.processor.tokenizer.pad_token_id
                        * torch.ones((1, pad_len), dtype=torch.long),
                        input_ids,
                    ],
                    dim=1,
                )
            )

            attention_mask = torch.cat(
                [
                    torch.zeros((1, pad_len), dtype=torch.long),
                    torch.ones((1, seq_len), dtype=torch.long),
                ],
                dim=1,
            )
            all_attention_masks.append(attention_mask)
        all_input_ids = torch.cat(all_input_ids, dim=0)
        all_attention_masks = (
            torch.cat(all_attention_masks, dim=0)
            if len(all_attention_masks) > 0
            else None
        )

        all_pixel_values = [x["pixel_values"] for x in batch if "pixel_values" in x]
        all_pixel_values_videos = [
            x["pixel_values_videos"] for x in batch if "pixel_values_videos" in x
        ]
        if all_pixel_values[0] is not None:
            pixel_values = (
                torch.cat(all_pixel_values, dim=0)
                if len(all_pixel_values) > 0
                else None
            )
        else:
            pixel_values = None

        if all_pixel_values_videos[0] is not None:
            pixel_values_videos = (
                torch.cat(all_pixel_values_videos, dim=0)
                if len(all_pixel_values_videos) > 0
                else None
            )
        else:
            pixel_values_videos = None
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            **utils,
        }


class LimitedSizeSharedDict(MutableMapping):
    def __init__(self, max_size=8):
        self.manager = Manager()
        self._data = self.manager.dict()
        self._order = self.manager.list()
        self.max_size = max_size
        self.lock = self.manager.Lock()

    def __setitem__(self, key, value):
        with self.lock:
            if key not in self._data:
                if len(self._data) >= self.max_size:
                    oldest_key = self._order.pop(0)
                    del self._data[oldest_key]
                self._data[key] = value
                self._order.append(key)

    def __getitem__(self, key):
        with self.lock:
            return self._data[key]

    def __delitem__(self, key):
        with self.lock:
            del self._data[key]
            self._order.remove(key)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __contains__(self, key):
        return key in self._data

    def get(self, key, default=None):
        try:
            return self._data[key]
        except KeyError:
            return default


class BaseDataset(Dataset, ABC):
    def __init__(
        self,
        processor: Qwen2VLProcessor,
        dataset_name: str,
        conv_format: Conversation,
        split="train",
        already_finished=None,
        curr_idx=0,
        total_idx=1,
        prompt=None,
        sample_fps=1,  # auto means use fps inference
        video_sample_type="fps",
        uniform_sampled_frames="auto",  # when uniform
        max_num_frames=128,  # used for fps sampling
        min_num_frames=1,
        sys_prompt="",
        min_pixels=None,
        total_pixels=None,
        max_frames=None,
        fps=None,
        cache_size=1,  # 0 means no cache
    ):
        self.sys_prompt = sys_prompt
        self.min_pixels = min_pixels
        self.total_pixels = total_pixels
        self.max_frames = max_frames
        self.dataset_name = dataset_name
        self.fps = fps

        if cache_size <= 0:
            self.video_cache = None
        else:
            print(f"Use Video Cache (size={cache_size})")
            self.video_cache = LimitedSizeSharedDict(max_size=cache_size)

        self.processor = processor

        self.conv = conv_format
        if already_finished is None:
            already_finished = set()

        self.data = self._load_data(dataset_name, split=split)
        self.data = [item for item in self.data if item["qid"] not in already_finished]
        self.data = self._split_data(self.data, curr_idx, total_idx)
        self.video_size_list = self._load_videosize(dataset_name)
        print("max_num_frames:", max_num_frames)

        self.sample_config = {
            "sample_type": video_sample_type,
            "output_fps": sample_fps,
            "num_frames": uniform_sampled_frames,
            "max_num_frames": max_num_frames,
            "min_num_frames": min_num_frames,
            "img_shortest_edge": None,
            "img_longest_edge": None,
        }

    def __len__(self):
        return len(self.data)

    def _make_conversation(self, item, DEFAULT_TOKEN, num_frames: int = 1):
        conv = self.conv.copy()
        conv.messages = []
        question = self._build_user_prompt(item, num_frames=num_frames)
        token = DEFAULT_TOKEN * num_frames
        if "<video>" in question:
            question = question.replace("<video>", token)
        elif "<image>" in question:
            question = question.replace("<image>", token)
        else:
            question = f"{token}{question}"
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], "")
        conv_str = conv.get_prompt()
        return conv_str

    @staticmethod
    def _load_data(dataset_name, split):
        if dataset_name == "charades":
            print("Use charades")
            data = load_charades(split=split)
        elif dataset_name == "activitynet":
            print("Use activitynet")
            data = load_activitynet(split=split)
        elif dataset_name == "tvgbench":
            print("Use tvgbench")
            data = load_tvgbench(split=split)
        elif dataset_name == "tvgbench_filter":
            print("Use tvgbench_filter")
            data = load_tvgbench_filter(split=split)
        elif dataset_name == "videomme":
            print("Use videomme")
            data = load_videomme(split=split)
        elif dataset_name == "lvbench":
            print("Use lvbench")
            data = load_lvbench(split=split)
        elif dataset_name == "mlvu":
            print("Use mlvu")
            data = load_mlvu(split=split)
        elif dataset_name == "longvideobench":
            print("Use longvideobench")
            data = load_longvideobench(split=split)
        elif dataset_name == "mvbench":
            print("Use mvbench")
            data = load_mvbench(split=split)
        elif dataset_name == "egoschema":
            print("Use egoschema")
            data = load_egoschema(split=split)
        elif dataset_name == "tempcompass":
            print("Use tempcompass")
            if len(data) > 0 and split == "captioning":
                assert False
            data = load_tempcompass(split=split)
        elif dataset_name == "cgbench":
            print("Use cgbench")
            data = load_cgbench(split=split)
        elif dataset_name == "auroracap":
            print("Use auroracap")
            data = load_auroracap(split=split)
        elif dataset_name == "youcook2":
            print("Use youcook2")
            data = load_youcook2(split=split)
        else:
            raise NotImplementedError(f"{dataset_name} is not implemented yet")
        return data

    def load_media_data_video(self, data_path):
        frames, durations = load_decord(
            src_path=data_path,
            video_size_list=self.video_size_list,
            **self.sample_config,
        )
        return frames, durations

    @staticmethod
    def _load_videosize(dataset_name):
        data = {}
        if dataset_name == "charades":
            pass
        elif dataset_name == "activitynet":
            pass
        elif dataset_name == "tvgbench":
            pass
        elif dataset_name == "tvgbench_filter":
            pass
        elif dataset_name == "videomme":
            print("Use videomme")
            data_path = "./dataset/videomme/videomme/video_size.json"
            with open(data_path, "r") as f:
                data = json.load(f)
        elif dataset_name == "lvbench":
            print("Use lvbench")
            data_path = "./dataset/lvbench/data/video_size.json"
            with open(data_path, "r") as f:
                data = json.load(f)
        elif dataset_name == "mlvu":
            print("Use mlvu")
            data_path = "./dataset/mlvu/MLVU/video_size.json"
            with open(data_path, "r") as f:
                data = json.load(f)
        elif dataset_name == "longvideobench":
            print("Use longvideobench")
            data_path = "./dataset/longvideobench/video_size.json"
            with open(data_path, "r") as f:
                data = json.load(f)
        elif dataset_name == "mvbench":
            pass
        elif dataset_name == "egoschema":
            pass
        elif dataset_name == "tempcompass":
            pass
        elif dataset_name == "cgbench":
            print("Use cgbench")
            data_path = "./dataset/cgbench/video_size.json"
            with open(data_path, "r") as f:
                data = json.load(f)
        elif dataset_name == "auroracap":
            print("Use auroracap")
            data_path = "./dataset/auroracap/video_size.json"
            with open(data_path, "r") as f:
                data = json.load(f)
        elif dataset_name == "youcook2":
            pass
        else:
            raise NotImplementedError(f"{dataset_name} is not implemented yet")
        return data

    @staticmethod
    def _split_data(data, curr_idx, total_idx):
        """
        Zigzag split: split data into 2 * total_idx chunks and select two chunks:
        index = curr_idx and mirror_index = 2*total_idx - 1 - curr_idx.
        Example: total_idx=6, curr_idx=0 -> split into 12 parts, pick parts 0 and 11.
        """
        n = len(data)
        parts = 2 * total_idx
        if parts <= 0 or n == 0:
            return []

        # size of each chunk (ceil)
        chunk_size = (n + parts - 1) // parts

        a = curr_idx
        b = parts - 1 - curr_idx

        selected_ranges = []
        result = []
        for idx in (a, b):
            st = idx * chunk_size
            ed = min(n, (idx + 1) * chunk_size)
            if st < ed:
                selected_ranges.append((st, ed))
                result.extend(data[st:ed])

        print("Selected ranges:", selected_ranges)
        print("Total data len:", len(result))
        return result

    # TODO: 目前没有在用这个
    def _save_video_to_cache(self, video_path, video_ele, video):
        if self.video_cache:
            key = video_path + json.dumps(video_ele)
            self.video_cache[key] = video

    def _load_video_from_cache(self, video_path, video_ele):
        if self.video_cache:
            key = video_path + json.dumps(video_ele)
            return self.video_cache.get(key, None)
        else:
            return None

    @staticmethod
    def _load_video_from_prepared(video_path, video_dirs):
        video_id = video_path.split("/")[-1].split(".")[0]
        for video_dir in video_dirs:
            video_prepared_path = os.path.join(video_dir, video_id + ".pt")
            if os.path.exists(video_prepared_path):
                return torch.load(video_prepared_path)
        return None

    def default_ele(self):
        ele = {}
        if self.min_pixels is not None:
            ele["min_pixels"] = self.min_pixels
        if self.total_pixels is not None:
            ele["total_pixels"] = self.total_pixels
        if self.max_frames is not None:
            ele["max_frames"] = self.max_frames
        if self.fps is not None:
            ele["fps"] = self.fps
        return ele

    def _preprocess(self, item, sub_images):
        """

        Args:
            item (dict)::
                {
                    "video": video_path,
                    "question": question,
                    "options": [option1, option2, ...],
                    "answer": answer,
                    "qid": question_id,
                    "video_start": video_start,
                    "video_end": video_end,
                }
            sub_images (list[Image.Image]): video frames

        Returns:
            dict: input_ids, pixel_values, pixel_values_videos
        """
        DEFAULT_TOKEN = DEFAULT_VIDEO_TOKEN
        conv_str = self._make_conversation(
            item, DEFAULT_TOKEN, num_frames=len(sub_images)
        )

        try:
            encoding = self.processor(
                text=conv_str,
                images=[sub_images],
                return_tensors="pt",
                truncation=True,
                do_resize=False,
                max_length=100000,
            )
        except Exception as e:
            print("Sub_images:", sub_images)
            print("Error when processing", item)
            print("Exception:", e)
            raise e

        # 3. return processed inputs
        return {
            "input_ids": encoding["input_ids"],
            "pixel_values": encoding.get("pixel_values", None),
            "pixel_values_videos": encoding.get("pixel_values_videos", None),
        }

    def _get_inputs(self, item):
        video_path = item["video"]

        if self.video_cache is not None:
            if video_path not in self.video_cache:
                # print(f"Cache miss. Loading video: {video_path}")
                video_frames, _ = self.load_media_data_video(video_path)
                self.video_cache[video_path] = video_frames
            else:
                video_frames = self.video_cache[video_path]
        else:
            video_frames, _ = self.load_media_data_video(video_path)

        inputs = self._preprocess(item, video_frames)
        return inputs


class MultipleChoiceQADataset(BaseDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_prompt = "Best Option: ("

    def _choice_id_to_letter(self, offset):
        return chr(ord("A") + offset)

    def _build_user_prompt(self, item, num_frames: int = 1):
        question = "\n".join(
            [
                item["question"],
                "\n".join(item["options"]),
                "Please provide your answer by stating the letter followed by the full option.",
            ]
        )
        return question

    def _build_options(self, item):
        return [self._choice_id_to_letter(i) for i in range(len(item["options"]))]

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self._get_inputs(item)

        return {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs.get("pixel_values", None),
            "pixel_values_videos": inputs.get("pixel_values_videos", None),
            "options": self._build_options(self.data[idx]),
            "answer": self.data[idx]["answer"],
            "qid": self.data[idx]["qid"],
            "video_paths": self.data[idx]["video"],
            "task_type": self.data[idx].get("task_type", None),
            "duration": self.data[idx]["duration"],
        }


class TemporalGroundingDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_user_prompt(self, item, num_frames: int = 1):
        timestamp_prompt = "The video lasts for {} seconds, and {} frames are uniformly sampled from it."
        return f"{timestamp_prompt.format(item['duration'], num_frames)} {temporal_video_grounding_templates['user'][-6].format(item['sentence'])}"

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self._get_inputs(item)

        return {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs.get("pixel_values", None),
            "pixel_values_videos": inputs.get("pixel_values_videos", None),
            "timestamps": self.data[idx]["timestamp"],
            "qid": self.data[idx]["qid"],
            "video_paths": self.data[idx]["video"],
            "duration": self.data[idx]["duration"],
        }


class VideoCaptionDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_user_prompt(self, item, num_frames: int = 1):
        if self.dataset_name == "auroracap":
            import random

            from train.data_utils import auroracap_prompts

            question = random.choice(auroracap_prompts[item["task_type"]])
            return question
        else:
            timestamp_prompt = "The video lasts for {} seconds, and {} frames are uniformly sampled from it.".format(
                item["duration"], num_frames
            )
            question = (
                "Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences. "
                "The output format of each predicted event should be like: start - end seconds, event description. "
                "A specific example is: 90 - 102 seconds, spread margarine on two slices of white bread in the video."
            )
            return f"{timestamp_prompt} {question}"

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self._get_inputs(item)

        return {
            "input_ids": inputs["input_ids"],
            "pixel_values": inputs.get("pixel_values", None),
            "pixel_values_videos": inputs.get("pixel_values_videos", None),
            "answer": self.data[idx]["answer"],
            "qid": self.data[idx]["qid"],
            "video_paths": self.data[idx]["video"],
            "task_type": self.data[idx].get("task_type", None),
            "duration": self.data[idx].get("duration", None),
        }


def build_dataloader(
    processor: Qwen2VLProcessor,
    datatype: Literal["tg", "mcq", "caption"],
    # ======== for dataset class
    dataset_name: List[str],
    conv_format: Conversation,
    split="train",
    already_finished: Optional[Iterable] = None,
    curr_idx=0,
    total_idx=1,
    prompt: Optional[str] = None,  # not used
    sample_fps=1,  # auto means use fps inference
    video_sample_type: Literal["uniform", "fps"] = "uniform",  # middle, random, or all
    uniform_sampled_frames: (
        int | Literal["auto"]
    ) = "auto",  # when uniform, auto means use fps inference
    max_num_frames=128,
    min_num_frames=1,
    sys_prompt="You are a helpful assistant.",
    min_pixels=16 * 28 * 28,
    total_pixels=3584 * 28 * 28,
    video_cache_size=1,
    # ======== end for dataset class
    batch_size=1,
    num_workers=8,
    return_probs=False,
):

    collate_fn = Qwen2VLMultiInputCollator(processor)

    kwargs = {
        "dataset_name": dataset_name,
        "conv_format": conv_format,
        "split": split,
        "already_finished": already_finished,
        "curr_idx": curr_idx,
        "total_idx": total_idx,
        "sample_fps": sample_fps,  # auto means use fps inference
        "video_sample_type": video_sample_type,
        "uniform_sampled_frames": uniform_sampled_frames,  # when uniform
        "max_num_frames": max_num_frames,
        "min_num_frames": min_num_frames,
        "sys_prompt": sys_prompt,
        "min_pixels": min_pixels,
        "total_pixels": total_pixels,
        "cache_size": video_cache_size,
    }
    if prompt is not None:
        kwargs["prompt"] = prompt

    if datatype == "tg":
        data = TemporalGroundingDataset(processor, **kwargs)
    elif datatype == "mcq" and not return_probs:
        data = MultipleChoiceQADataset(processor, **kwargs)
    elif datatype == "caption":
        data = VideoCaptionDataset(processor, **kwargs)
    else:
        raise NotImplementedError(f"{datatype} is not implemented yet")
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=2,
        pin_memory=True,
    )
    return dataloader
