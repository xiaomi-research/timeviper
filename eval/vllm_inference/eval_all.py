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
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from data.data_loader import *

from eval.vllm_inference.utils import get_dataset_type

random.seed(42)


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation for training-free video temporal grounding (Single GPU Version)"
    )
    parser.add_argument(
        "--dataset",
        help="Specify the dataset.",
    )
    parser.add_argument("--split", type=str, default="test", help="dataset type")
    parser.add_argument(
        "--model_name",
        type=str,
        default="kl_cot_gaussian_03_iouv2_2500",
        help="model name",
    )
    parser.add_argument(
        "--eval_root",
        type=str,
        default="kl_cot_gaussian_03_iouv2_2500",
        help="model name",
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=64,
        help="maximum number of frames to sample from each video",
    )
    return parser.parse_args()


def compute_IoU(pred, gt):
    """Compute the IoU given predicted and ground truth windows."""
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list:
        pred = [pred]
    if not gt_is_list:
        gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap


def mcq_is_correct(pred, gt):
    gt = chr(gt + ord("A"))
    # matches = re.findall(r"\(([A-Z])\)", pred)
    matches = re.findall(r"([A-Z]\.)", pred)
    if matches:
        return int(matches[0][0] == gt)
    return int(pred[0] == gt)


def load_scored_data(data_dir, dataset_name, split, eval_root=None):
    """Get and compute scores for all data in data_dir."""
    # load all jsonl files in data_dir
    pred_data = []
    for file in os.listdir(data_dir):
        if "jsonl" not in file or "score" in file:
            continue
        file_path = os.path.join(data_dir, file)
        pred_data += [json.loads(line) for line in open(file_path)]

    if dataset_name == "auroracap":
        from eval.vllm_inference.eval_auroracap import get_auroracap_score

        return get_auroracap_score(pred_data, split, eval_root)

    if dataset_name == "youcook2":
        from eval.vllm_inference.eval_dvc import evaluate_youcook2_dvc

        return evaluate_youcook2_dvc(pred_data)

    datatype = get_dataset_type(dataset_name)
    data = {}
    cnt = 0
    for tmp in pred_data:
        cnt += 1

        if datatype == "tg":
            score = (
                compute_IoU(tmp["pred"], tmp["target"])
                if None not in tmp["pred"]
                else 0.0
            )
        elif datatype == "mcq":
            if tmp["pred"] is not None:
                score = int(tmp["pred"] == tmp["target"])
            else:
                score = mcq_is_correct(tmp["output_text"], tmp["target"])
        elif datatype == "caption":
            raise NotImplementedError

        # save different data for different datasets
        if dataset_name in ["videomme", "longvideobench"]:
            data[tmp["qid"]] = {
                "score": score,
                "duration": tmp["duration"],
                "task_type": tmp["task_type"],
            }
        elif dataset_name in ["lvbench", "mlvu", "cgbench"]:
            data[tmp["qid"]] = {
                "score": score,
                "task_type": tmp["task_type"],
            }
        else:
            data[tmp["qid"]] = score
    return data


def calc_score(difficulty_data_dict, datasetname):
    if datasetname in ["youcook2"]:
        scores = {}
        scores["total"] = len(difficulty_data_dict["key"])
        for k, v in difficulty_data_dict.items():
            if k == "key":
                continue
            if "Para_" in k:
                scores[k] = round(v, 1)
            elif k == "n_preds":
                scores[k] = round(sum(v) / len(v), 1)
            else:
                scores[k] = round(sum(v) / len(v) * 100, 1)
        return scores

    data = list(difficulty_data_dict.values())
    if datasetname in ["activitynet", "charades", "tvgbench"]:
        scores = {}
        scores["mIoU"] = np.mean([itm for itm in data]) * 100
        for i in [0.3, 0.5, 0.7]:
            cnt = len([itm for itm in data if itm > i])
            score = cnt / len(difficulty_data_dict) * 100.0
            scores[f"IoU R1@{i}"] = score
        scores["avg"] = sum(scores.values()) / len(scores)
    elif datasetname in ["videomme", "longvideobench"]:
        # save all scores corresponding to different task_types, durations and total scores:
        scores = {}
        scores["total"] = {
            "correct": sum([itm["score"] for itm in data]),
            "total": len(data),
            "avg": round(sum([itm["score"] for itm in data]) / len(data) * 100, 2),
        }
        for itm in data:
            task_type = itm["task_type"]
            duration = itm["duration"]
            if task_type not in scores:
                scores[task_type] = {"correct": 0, "total": 0}
            if duration not in scores:
                scores[duration] = {"correct": 0, "total": 0}
            scores[duration]["correct"] += itm["score"]
            scores[duration]["total"] += 1
            scores[task_type]["correct"] += itm["score"]
            scores[task_type]["total"] += 1
        for key in scores:
            scores[key]["avg"] = round(
                scores[key]["correct"] / scores[key]["total"] * 100, 2
            )
    elif datasetname in ["lvbench", "mlvu", "cgbench"]:  # 按 task_type 分类
        scores = defaultdict(lambda: {"correct": 0, "total": 0})
        for itm in data:
            task_types = itm["task_type"]
            if not isinstance(task_types, list):  # handle mlvu
                task_types = [task_types]
            for task_type in task_types:
                scores["total"]["correct"] += itm["score"]
                scores["total"]["total"] += 1
                scores[task_type]["correct"] += itm["score"]
                scores[task_type]["total"] += 1
        for key in scores:
            scores[key]["avg"] = round(
                scores[key]["correct"] / scores[key]["total"] * 100, 2
            )
    elif datasetname == "auroracap":  # 按 task_type 分类，有 score 和 acc
        scores = defaultdict(lambda: {"total": 0, "score": 0.0, "acc": 0.0})
        for itm in data:
            task_type = itm["task_type"]
            scores["total"]["score"] += itm["score"]
            scores["total"]["acc"] += itm["acc"]
            scores["total"]["total"] += 1
            scores[task_type]["score"] += itm["score"]
            scores[task_type]["acc"] += itm["acc"]
            scores[task_type]["total"] += 1
        for key in scores:
            scores[key]["score"] = round(scores[key]["score"] / scores[key]["total"], 2)
            scores[key]["acc"] = round(
                scores[key]["acc"] / scores[key]["total"] * 100, 2
            )
    else:
        correct = sum([itm for itm in data])
        scores = {
            "correct": correct,
            "total": len(data),
            "avg": round(correct / len(data) * 100, 2),
        }
    return scores


def upload_json_to_server(
    data, api_url="https://validation-server.onrender.com/api/upload/"
):
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url=api_url, headers=headers, json=data)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {"status": "success", "response_text": response.text}

    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "message": str(e),
            "details": f"Failed to upload data to {api_url}",
        }


def eval_egoschema_online(data_dir, original_data):
    qid_to_vid = {}
    for itm in original_data:
        qid, vid = itm["qid"], itm["video"].split("/")[-1].split(".")[0]
        qid_to_vid[qid] = vid

    data = {}
    for file in os.listdir(data_dir):
        if "jsonl" not in file:
            continue
        file_path = os.path.join(data_dir, file)
        for line in open(file_path):
            tmp = json.loads(line)
            matches = re.findall(r"\(([A-Z])\)", tmp["output_text"])
            if matches:
                pred = ord(matches[-1]) - ord("A")
            else:
                pred = ord(random.choice(["A", "B", "C", "D", "E"])) - ord("A")
            data[qid_to_vid[tmp["qid"]]] = pred

    return upload_json_to_server(data)


def main(args):
    dataset = args.dataset
    original_data = None
    if original_data is not None:
        print(f"Original data length: {len(original_data)}")

    for data_dir in [args.eval_root]:
        if dataset == "egoschema":
            results_ego = eval_egoschema_online(data_dir, original_data)
            print(results_ego)
            with open(data_dir + "/scores.json", "w") as f:
                json.dump(results_ego, f, indent=4)
            continue

        difficulty_data_dict = load_scored_data(
            data_dir, dataset, args.split, eval_root=args.eval_root
        )
        if len(difficulty_data_dict) == 0:
            continue
        print(f"len(difficulty_data_dict): {len(difficulty_data_dict)}")

        score_dict = calc_score(difficulty_data_dict, dataset)
        for k, v in score_dict.items():
            print(f"{k}: {v}")
        with open(data_dir + "/scores.json", "w") as f:
            json.dump(score_dict, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args)
