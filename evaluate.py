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

import argparse
import json
import os
import re
import time

# test
import torch
import torch.nn.functional as F
from tqdm import tqdm

from eval.vllm_inference.data import build_dataloader
from eval.vllm_inference.utils import get_dataset_type
from eval.vllm_inference.vllm_infer import vllmTimeViperWrapper
from timeviper.data.conversation import *
from timeviper.data.image_processing import ImageProcessor
from timeviper.data.processor import Qwen2VLProcessor
from timeviper.model import (
    GenericTimeViperVLM,
    HybridTimeViperVLM,
    get_llm_backbone_and_tokenizer,
    get_vision_backbone_and_transform,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from timeviper.data.conversation import conv_templates


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation for training-free video temporal grounding (Single GPU Version)"
    )
    parser.add_argument(
        "--datatype",
        default="tg",
        type=str,
        help="Specify the dataset.",
        choices=["tg", "mcq", "caption"],
    )
    parser.add_argument(
        "--model_base", type=str, default="../pretrained_models/Qwen2.5-VL-7B-Instruct"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="GPU device to use"
    )
    parser.add_argument(
        "--pipeline_parallel_size", type=int, default=1, help="GPU nodes"
    )
    parser.add_argument("--split", type=str, default="default", help="dataset type")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--curr_idx", type=int, default=0, help="数据分片")
    parser.add_argument("--total_idx", type=int, default=1, help="数据分片")
    parser.add_argument(
        "--total_pixels", type=int, default=3584 * 28 * 28, help="total_pixels"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name",
        choices=[
            "charades",
            "activitynet",
            "videomme",
            "lvbench",
            "longvideobench",
            "mlvu",
            "mvbench",
            "tvgbench_filter",
            "tvgbench",
            "egoschema",
            "tempcompass",
            "cgbench",
            "auroracap",
            "youcook2",
        ],
    )
    parser.add_argument("--use_vllm_inference", action="store_true")
    parser.add_argument(
        "--conv_format", type=str, default="conv_mamba_chat_zephyr", help="Prompt type"
    )
    parser.add_argument(
        "--sample_fps",
        type=int,
        default=1,
        help="Sample fps for video, default is 1, means 1 frame per second",
    )
    parser.add_argument(
        "--video_sample_type",
        type=str,
        default="fps",
        help="Sample type for video sampling, default is 'middle'",
        choices=["uniform", "fps"],
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=128,
        help="Maximum number of frames to sample from video, default is 128",
    )
    parser.add_argument(
        "--min_num_frames",
        type=int,
        default=1,
        help="Minimum number of frames to sample from video, default is 1",
    )
    parser.add_argument(
        "--uniform_sampled_frames",
        type=str,
        default="auto",
        help="When uniform, auto means use fps inference, otherwise use max_num_frames",
    )
    parser.add_argument(
        "--arch_specifier",
        type=str,
        default="no-align+gelu-mlp",
        help="Sample type for video sampling, default is 'middle'",
        # choices=["no-align+gelu-mlp", "no-align+tome_mlp", "no-align+ps_mlp"],
    )
    parser.add_argument(
        "--llm_backbone_id",
        type=str,
        default="mamba-2.8b-zephyr",
        help="Model architecture to use, default is 'mamba-2.8b-zesphyr'",
    )
    parser.add_argument(
        "--vision_backbone_id",
        type=str,
        default="siglip-vit-so400m-384px",
        help="Model architecture to use, default is 'siglip-vit-so400m-384px'",
    )
    parser.add_argument("--no_answer_prompt", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Model architecture to use, default is 'flash_attention_2'",
    )
    parser.add_argument(
        "--video_cache_size",
        type=int,
        default=1,
        help="Number of videos to cache in memory",
    )
    # for pdrop
    parser.add_argument("--use_pdrop", action="store_true", help="Whether to use pdrop")
    parser.add_argument(
        "--pdrop_type",
        type=str,
        default="uni_14_0.8-attn_21_0.6-attn_30_0.4-attn_39_0.2",
        help="pdrop type, e.g., 'uni_14_0.8-attn_21_0.6-attn_30_0.4-attn_39_0.2'",
    )
    parser.add_argument(
        "--merge_module",
        type=str,
        default="no_merge",
        help="Sample type for video sampling, default is 'middle'",
    )
    parser.add_argument("--visual_token_order", type=str, default="raw")
    return parser.parse_args()


def build_model(args):
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        args.vision_backbone_id, image_resize_strategy="resize-naive"
    )

    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        args.llm_backbone_id,
        llm_max_length=None,
        attn_implementation=args.attn_implementation,
        merge_module=args.merge_module,
        use_pdrop=args.use_pdrop,
        pdrop_type=args.pdrop_type,
    )
    family = llm_backbone.llm_family
    if family in ["nano"]:
        model = HybridTimeViperVLM.from_pretrained(
            pretrained_checkpoint=args.model_base,
            model_id="cobra-siglip+3b",  # no use, consider remove this arg
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            arch_specifier=args.arch_specifier,
            visual_token_order=args.visual_token_order,
        )
    elif family in ["qwen2"]:
        model = GenericTimeViperVLM.from_pretrained(
            pretrained_checkpoint=args.model_base,
            model_id="cobra-siglip+3b",  # no use, consider remove this arg
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            arch_specifier=args.arch_specifier,
            visual_token_order=args.visual_token_order,
        )
    else:
        raise ValueError(
            f"No VLM is configured for the LLM family `{family}` (from backbone `{llm_backbone.identifier}`)!"
        )
    image_processor = ImageProcessor(
        patch_size=14,
        image_transforms=image_transform,
    )
    processor = Qwen2VLProcessor(image_processor, tokenizer, model_config=model.config)

    if args.use_vllm_inference:
        model = vllmTimeViperWrapper(args, model)

    return model, processor


@torch.no_grad()
def inference(model, inputs):
    for key in inputs.keys():
        if not isinstance(inputs[key], torch.Tensor):
            continue
        inputs[key] = inputs[key].to(model.device)

    logits = model(**inputs).logits
    bsz, seq_len, _ = logits.shape
    if "attention_mask" in inputs:
        pred_token_indices = torch.sum(inputs["attention_mask"], dim=-1) - 1
    else:
        pred_token_indices = torch.full((bsz,), seq_len - 1, device=logits.device)

    pred_token_logits = logits[
        torch.arange(bsz, device=logits.device), pred_token_indices, :
    ]

    return pred_token_logits


def extract_answer(output_string, datatype):
    if datatype == "tg":
        matches = re.findall(r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", output_string)
        if not matches:
            answer_match = re.search(r"<answer>(.*?)</answer>", output_string)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                answer_matches = re.findall(
                    r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", answer_content
                )
                if answer_matches:
                    last_match = answer_matches[-1]
                    return [float(last_match[0]), float(last_match[2])]
            return [None, None]

        last_match = matches[-1]
        start_time_str = last_match[0]
        end_time_str = last_match[2]

        try:
            start_time = float(start_time_str)
            end_time = float(end_time_str)
            return [start_time, end_time]
        except ValueError:
            return [None, None]

    if datatype == "mcq":
        # matches = re.findall(r"\(([A-Z])\)", output_string)
        try:
            if output_string[0] == "(":
                matches = [output_string[1]]
            else:
                matches = [output_string[0]]
        except Exception:
            matches = ["A"]
        if matches:
            return ord(matches[-1]) - ord("A")
        return None
    if datatype == "caption":
        from eval.vllm_inference.eval_dvc import parse_dvc_prediction

        timestamps, captions = parse_dvc_prediction(output_string)
        return {
            "timestamps": timestamps,
            "captions": captions,
        }

    raise ValueError(f"Unsupported datatype: {datatype}")


@torch.no_grad()
def calc_prob(logits, options_token_ids):
    bsz = logits.shape[0]
    probs = []
    for i in range(bsz):
        logit = logits[i, options_token_ids]
        probs.append(F.softmax(logit, dim=1))
    return probs


@torch.no_grad()
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"{args.curr_idx}_of_{args.total_idx}.jsonl",
    )

    already_finished = set()
    f = open(output_file, "a+")
    try:
        for file in os.listdir(args.output_dir):
            if "jsonl" not in file or "score" in file:
                continue
            file_path = os.path.join(args.output_dir, file)
            already_finished.update(
                [json.loads(line)["qid"] for line in open(file_path)]
            )

    except Exception as e:
        print(e)

    model, processor = build_model(args)

    print("min_num_frames:", args.min_num_frames)
    dataloader_args = {
        "dataset_name": args.dataset,
        "conv_format": conv_templates[args.conv_format],
        "split": args.split,
        "already_finished": already_finished,
        "curr_idx": args.curr_idx,
        "total_idx": args.total_idx,
        "sample_fps": args.sample_fps,  # auto means use fps inference
        "video_sample_type": args.video_sample_type,  # sample type for video sampling
        "uniform_sampled_frames": args.uniform_sampled_frames,  # when uniform, auto means use fps inference
        "max_num_frames": args.max_num_frames,
        "min_num_frames": args.min_num_frames,
        "batch_size": args.batch_size,
        "num_workers": min(8, args.batch_size),
        "total_pixels": args.total_pixels,
        "video_cache_size": args.video_cache_size,
    }

    dataloader = build_dataloader(processor, args.datatype, **dataloader_args)

    program_start_time = time.perf_counter()

    for batch_itm in tqdm(
        dataloader, position=args.curr_idx, desc=f"Eval {args.curr_idx}"
    ):
        if args.datatype == "tg":
            if args.use_vllm_inference:
                output_texts = model.generate(
                    inputs=batch_itm,
                    max_new_tokens=args.max_new_tokens,
                )
                if batch_itm["input_ids"].shape[0] == 1:
                    if not isinstance(output_texts, list):
                        output_texts = [output_texts]
                targets = batch_itm["timestamps"]
                for i in range(len(targets)):
                    pred = extract_answer(output_texts[i], args.datatype)
                    f.write(
                        json.dumps(
                            {
                                "qid": batch_itm["qid"][i],
                                "pred": pred,
                                "target": list(targets[i]),
                                "duration": (
                                    None
                                    if "duration" not in batch_itm
                                    else batch_itm["duration"][i]
                                ),
                                "output_text": output_texts[i],
                            }
                        )
                        + "\n"
                    )
                    f.flush()
            else:
                device = model.device
                dtype = model.llm_backbone.half_precision_dtype

                if (
                    "input_ids" in batch_itm.keys()
                    and batch_itm["input_ids"] is not None
                ):
                    batch_itm["input_ids"] = batch_itm["input_ids"].to(device=device)
                if (
                    "pixel_values" in batch_itm.keys()
                    and batch_itm["pixel_values"] is not None
                ):
                    batch_itm["pixel_values"] = batch_itm["pixel_values"].to(
                        device=device, dtype=dtype
                    )
                if (
                    "pixel_values_videos" in batch_itm.keys()
                    and batch_itm["pixel_values_videos"] is not None
                ):
                    batch_itm["pixel_values_videos"] = batch_itm[
                        "pixel_values_videos"
                    ].to(device=device, dtype=dtype)
                if (
                    "attention_mask" in batch_itm.keys()
                    and batch_itm["attention_mask"] is not None
                ):
                    batch_itm["attention_mask"] = batch_itm["attention_mask"].to(
                        device=device, dtype=torch.long
                    )
                output_texts = model.generate(
                    batch_itm["input_ids"],
                    pixel_values=batch_itm["pixel_values"],
                    pixel_values_videos=batch_itm["pixel_values_videos"],
                    attention_mask=(
                        batch_itm["attention_mask"]
                        if "attention_mask" in batch_itm
                        else None
                    ),
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                    temperature=0,
                    answer_prompt=(
                        None
                        if args.no_answer_prompt is True
                        else dataloader.dataset.answer_prompt
                    ),  # Best Options: (
                )
                if batch_itm["input_ids"].shape[0] == 1:
                    if not isinstance(output_texts, list):
                        output_texts = [output_texts]
                targets = batch_itm["timestamps"]
                for i in range(len(targets)):
                    pred = extract_answer(output_texts[i], args.datatype)
                    f.write(
                        json.dumps(
                            {
                                "qid": batch_itm["qid"][i],
                                "pred": pred,
                                "target": list(targets[i]),
                                "duration": (
                                    None
                                    if "duration" not in batch_itm
                                    else batch_itm["duration"][i]
                                ),
                                "output_text": output_texts[i],
                            }
                        )
                        + "\n"
                    )
                    f.flush()
        elif args.datatype == "mcq":
            # model.projector.projector[0].weight.device
            if args.use_vllm_inference:
                output_texts = model.generate(
                    inputs=batch_itm,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0,
                    answer_prompt=(
                        None
                        if args.no_answer_prompt is True
                        else dataloader.dataset.answer_prompt
                    ),  # Best Options: (
                    # batch_itm["inputs"],
                )
            else:
                device = model.device
                dtype = model.llm_backbone.half_precision_dtype

                if (
                    "input_ids" in batch_itm.keys()
                    and batch_itm["input_ids"] is not None
                ):
                    batch_itm["input_ids"] = batch_itm["input_ids"].to(device=device)
                if (
                    "pixel_values" in batch_itm.keys()
                    and batch_itm["pixel_values"] is not None
                ):
                    batch_itm["pixel_values"] = batch_itm["pixel_values"].to(
                        device=device, dtype=dtype
                    )
                if (
                    "pixel_values_videos" in batch_itm.keys()
                    and batch_itm["pixel_values_videos"] is not None
                ):
                    batch_itm["pixel_values_videos"] = batch_itm[
                        "pixel_values_videos"
                    ].to(device=device, dtype=dtype)
                if (
                    "attention_mask" in batch_itm.keys()
                    and batch_itm["attention_mask"] is not None
                ):
                    batch_itm["attention_mask"] = batch_itm["attention_mask"].to(
                        device=device, dtype=torch.long
                    )
                output_texts = model.generate(
                    batch_itm["input_ids"],
                    pixel_values=batch_itm["pixel_values"],
                    pixel_values_videos=batch_itm["pixel_values_videos"],
                    attention_mask=(
                        batch_itm["attention_mask"]
                        if "attention_mask" in batch_itm
                        else None
                    ),
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                    temperature=0,
                    answer_prompt=(
                        None
                        if args.no_answer_prompt is True
                        else dataloader.dataset.answer_prompt
                    ),  # Best Options: (
                )
            if batch_itm["input_ids"].shape[0] == 1:
                if not isinstance(output_texts, list):
                    output_texts = [output_texts]
            targets = batch_itm["answer"]
            for i in range(len(targets)):
                pred = extract_answer(output_texts[i], args.datatype)
                f.write(
                    json.dumps(
                        {
                            "qid": batch_itm["qid"][i],
                            "pred": pred,
                            "target": targets[i],
                            "duration": (
                                None
                                if "duration" not in batch_itm
                                else batch_itm["duration"][i]
                            ),
                            "output_text": output_texts[i],
                            "task_type": (
                                batch_itm["task_type"][i]
                                if "task_type" in batch_itm
                                else None
                            ),
                        }
                    )
                    + "\n"
                )
                f.flush()
        elif args.datatype == "caption":
            if args.use_vllm_inference:
                output_texts = model.generate(
                    inputs=batch_itm,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0,
                    answer_prompt=(
                        None
                        if args.no_answer_prompt is True
                        else dataloader.dataset.answer_prompt
                    ),
                )
            else:
                device = model.device
                dtype = model.llm_backbone.half_precision_dtype

                if (
                    "input_ids" in batch_itm.keys()
                    and batch_itm["input_ids"] is not None
                ):
                    batch_itm["input_ids"] = batch_itm["input_ids"].to(device=device)
                if (
                    "pixel_values" in batch_itm.keys()
                    and batch_itm["pixel_values"] is not None
                ):
                    batch_itm["pixel_values"] = batch_itm["pixel_values"].to(
                        device=device, dtype=dtype
                    )
                if (
                    "pixel_values_videos" in batch_itm.keys()
                    and batch_itm["pixel_values_videos"] is not None
                ):
                    batch_itm["pixel_values_videos"] = batch_itm[
                        "pixel_values_videos"
                    ].to(device=device, dtype=dtype)
                if (
                    "attention_mask" in batch_itm.keys()
                    and batch_itm["attention_mask"] is not None
                ):
                    batch_itm["attention_mask"] = batch_itm["attention_mask"].to(
                        device=device, dtype=torch.long
                    )
                output_texts = model.generate(
                    batch_itm["input_ids"],
                    pixel_values=batch_itm["pixel_values"],
                    pixel_values_videos=batch_itm["pixel_values_videos"],
                    attention_mask=(
                        batch_itm["attention_mask"]
                        if "attention_mask" in batch_itm
                        else None
                    ),
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                    temperature=0,
                    answer_prompt=(
                        None
                        if args.no_answer_prompt is True
                        else dataloader.dataset.answer_prompt
                    ),
                )
            if batch_itm["input_ids"].shape[0] == 1:
                if not isinstance(output_texts, list):
                    output_texts = [output_texts]
            # remove special tokens
            output_texts_wo_special = model.llm_tokenizer.batch_decode(
                [
                    model.llm_tokenizer.encode(text, add_special_tokens=False)
                    for text in output_texts
                ],
                skip_special_tokens=True,
            )
            targets = batch_itm["answer"]
            for i in range(len(targets)):
                if args.dataset == "auroracap":
                    pred = output_texts_wo_special[i]
                else:
                    pred = extract_answer(output_texts_wo_special[i], args.datatype)
                f.write(
                    json.dumps(
                        {
                            "qid": batch_itm["qid"][i],
                            "pred": pred,
                            "target": targets[i],
                            "duration": (
                                None
                                if "duration" not in batch_itm
                                else batch_itm["duration"][i]
                            ),
                            "output_text": output_texts[i],
                            "task_type": (
                                batch_itm["task_type"][i]
                                if "task_type" in batch_itm
                                else None
                            ),
                        }
                    )
                    + "\n"
                )
                f.flush()
        else:
            logits = inference(model, batch_itm["inputs"])
            options_token_ids = [
                [processor.tokenizer.vocab[word] for word in word_list]
                for word_list in batch_itm["options"]
            ]
            probs = calc_prob(logits, options_token_ids)

            for i in range(len(logits)):
                f.write(
                    json.dumps(
                        {
                            "qid": batch_itm["qid"][i],
                            "pred": probs[i].argmax().item(),
                            "target": batch_itm["answer"][i],
                            "duration": (
                                None
                                if "duration" not in batch_itm
                                else batch_itm["duration"][i]
                            ),
                            "probs": probs[i].cpu().tolist(),
                        }
                    )
                    + "\n"
                )
                f.flush()

    # --- END TOTAL TIME & CALCULATIONS ---
    program_end_time = time.perf_counter()
    total_program_duration = program_end_time - program_start_time

    print("\n--- Timing Summary ---")
    print(f"Total program execution time: {total_program_duration:.2f} seconds")

    output_filename = f"{args.output_dir}/timing_summary_vllm.txt"

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n--- Timing Summary ---\n")
        f.write(f"Total program execution time: {total_program_duration:.2f} seconds\n")
        f.write("Another line of summary using write.\n")


if __name__ == "__main__":
    from eval.vllm_inference.utils import monkey_patch

    monkey_patch()
    args = get_args()
    args.datatype = get_dataset_type(args.dataset)

    main(args)
