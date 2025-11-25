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

import json
import os

import datasets
import pandas as pd


def load_activitynet(split="default"):
    if split == "default":
        split = "val"
    assert split in ["train", "val", "test"]
    data_root = "./dataset/activitynet"
    data_path = f"{data_root}/annotations/sentence_temporal_grounding/{split}.json"
    data = json.load(open(data_path))
    qid, conv_data = 0, []

    for video_id, meta_data in data.items():
        video_path = None
        for ext in ["mp4", "mkv", "webm"]:
            tmp = os.path.join(f"{data_root}/videos", f"{video_id}.{ext}")
            if os.path.exists(tmp):
                video_path = tmp
                break
        assert video_path is not None

        for i in range(len(meta_data["timestamps"])):
            conv_data.append(
                {
                    "video": video_path,
                    "duration": meta_data["duration"],
                    "timestamp": meta_data["timestamps"][i],
                    "sentence": meta_data["sentences"][i].strip(),
                    "qid": f"activitynet_{qid}",
                }
            )
            qid += 1

    return conv_data


def load_charades(split="default"):
    if split == "default":
        split = "test"
    assert split in ["train", "test"]
    data_root = "./dataset/charades"
    data_path = f"{data_root}/Charades_anno/Charades_sta_{split}.json"
    # data_path = f"{data_root}/Charades_anno/Charades_sta_{split}_debug.json"
    if not os.path.exists(data_path):
        data = {}
        old_data_path = f"{data_root}/Charades_anno/Charades_sta_{split}.txt"
        data_csv = f"{data_root}/Charades_anno/Charades_v1_{split}.csv"
        df = pd.read_csv(data_csv)
        video_to_duration = dict(zip(df["id"], df["length"]))

        for line in open(old_data_path):
            if line.strip() == "":
                continue
            meta_data, sentence = line.split("##")
            video_id, start, end = meta_data.split(" ")
            if video_id not in data:
                data[video_id] = {
                    "duration": video_to_duration[video_id],
                    "timestamps": [],
                    "sentences": [],
                }
            data[video_id]["timestamps"].append([float(start), float(end)])
            data[video_id]["sentences"].append(sentence)
        with open(data_path, "w") as f:
            json.dump(data, f)
    else:
        data = json.load(open(data_path))

    qid, conv_data = 0, []
    for video_id, meta_data in data.items():
        video_path = os.path.join(f"{data_root}/Charades_v1", f"{video_id}.mp4")
        for i in range(len(meta_data["timestamps"])):
            conv_data.append(
                {
                    "video": video_path,
                    "duration": meta_data["duration"],
                    "timestamp": meta_data["timestamps"][i],
                    "sentence": meta_data["sentences"][i].strip(),
                    "qid": f"charades_{qid}",
                }
            )
            qid += 1

    return conv_data


def load_tvgbench_filter(split):
    data_path = split
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qid, conv_data = 0, []
    for meta_data in data:
        video = meta_data["video"]
        duration = meta_data["duration"]
        timestamps = meta_data["timestamp"]
        sentences = meta_data["sentence"]
        qid = meta_data["qid"]
        pred = meta_data["pred"]
        video_start = meta_data["video_start"]
        video_end = meta_data["video_end"]
        conv_data.append(
            {
                "video": video,
                "duration": duration,
                "timestamp": timestamps,
                "pred": pred,
                "sentence": sentences,
                "qid": qid,
                "video_start": video_start,
                "video_end": video_end,
            }
        )

    return conv_data


def load_mlvu(split="default", video_suffix=""):
    data_root = "dataset/mlvu/MLVU"
    data_path = f"{data_root}/json"

    VIDEO_DIR = {
        "plotQA": f"{data_root}/video{video_suffix}/1_plotQA",
        "findNeedle": f"{data_root}/video{video_suffix}/2_needle",
        "ego": f"{data_root}/video{video_suffix}/3_ego",
        "count": f"{data_root}/video{video_suffix}/4_count",
        "order": f"{data_root}/video{video_suffix}/5_order",
        "anomaly_reco": f"{data_root}/video{video_suffix}/6_anomaly_reco",
        "topic_reasoning": f"{data_root}/video{video_suffix}/7_topic_reasoning",
        "subPlot": f"{data_root}/video{video_suffix}/8_sub_scene",
        "summary": f"{data_root}/video{video_suffix}/9_summary",
    }

    conv_data = []
    for file_name in os.listdir(data_path):
        data = json.load(open(os.path.join(data_path, file_name)))
        for qid, itm in enumerate(data):
            video_name = itm["video"]
            task_type = itm["question_type"]
            video_path = os.path.join(VIDEO_DIR[task_type], video_name)
            if "candidates" in itm.keys():
                # only support multi-choice QA tasks, do not support caption task currently
                conv_data.append(
                    {
                        "video": video_path,
                        "question": itm["question"],
                        "options": [
                            f"{chr(65 + i)}. {opt}"
                            for i, opt in enumerate(itm["candidates"])
                        ],
                        "answer": itm["candidates"].index(itm["answer"]),
                        "duration": itm["duration"],
                        "task_type": itm["question_type"],
                        "qid": f"mlvu|{task_type}|{qid}",
                    }
                )

    return conv_data


def load_longvideobench(split="default", video_suffix=""):
    if split == "default":
        split = "test"
    assert split in ["val", "test"]
    data_root = "dataset/longvideobench"
    if split == "test":
        data_path = f"{data_root}/lvb_test_wo_gt.json"
    else:
        data_path = f"{data_root}/lvb_val.json"

    conv_data = []
    data = json.load(open(data_path, "r"))

    duration_dict = {"15": "very short", "60": "short", "600": "medium", "3600": "long"}
    for itm in data:
        video_path = os.path.join(
            f"{data_root}/videos{video_suffix}", itm["video_path"]
        )  # NOTE 视频解压都放到 videos/
        conv_data.append(
            {  # TODO 其他信息
                "video": video_path,
                "question": itm["question"],
                # add "A.", "B.", "C.", "D." to each item of the list itm["candidates"] with size 4
                "options": [
                    f"{chr(65 + i)}. {opt}" for i, opt in enumerate(itm["candidates"])
                ],
                "answer": itm.get("correct_choice", None),  # NOTE test 没有 gt
                "duration": duration_dict[str(itm["duration_group"])],
                "task_type": itm["question_category"],
                "qid": f"longvideobench_{itm['id']}",
            }
        )

    return conv_data


def load_lvbench(split="default", video_suffix=""):
    data_root = "./dataset/lvbench"
    data_path = f"{data_root}/data/video_info.meta.jsonl"
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f.readlines() if line.strip()]
    conv_data = []
    for itm in data:
        video_path = os.path.join(
            f"{data_root}/all_videos{video_suffix}", itm["key"] + ".mp4"
        )
        for qa in itm["qa"]:
            question, *options = qa["question"].split("\n")
            conv_data.append(
                {
                    "video": video_path,
                    "question": question,
                    "options": [op for op in options],
                    "answer": ord(qa["answer"]) - ord("A"),
                    "duration": None,
                    "task_type": qa["question_type"],
                    "qid": f'lvbench_{qa["uid"]}',
                }
            )
    return conv_data


def load_tvgbench(split="default"):
    """
    Load JSON data in TVGBench format.

    Args:
        data_path (str): Path to the JSON file in TVGBench format.

    Returns:
        list: A list containing processed data, where each element is a dictionary
            in the format {'video': str, 'duration': float, 'timestamp': list[float, float], 'sentence': str, 'qid': str}.
            Returns an empty list if the file does not exist or cannot be parsed.
    """
    data_path = "./dataset/trainval/tvgbench.json"

    with open(data_path, "r") as f:
        raw_data = json.load(f)

    qid_counter = 0
    conv_data = []

    for item in raw_data:

        video_path = item["path"]

        if not os.path.exists(video_path):
            continue

        duration_str = item["duration"]
        answer_str = item["answer"]
        question_str = item["question"]
        start = item["start"]
        end = item["end"]
        duration = duration_str

        parts = answer_str.split("-")

        start_time = float(parts[0])
        end_time = float(parts[1])
        timestamp = [start_time, end_time]

        sentence = question_str

        if "source" in item and isinstance(item["source"], str):
            source_filename = os.path.basename(item["source"])
            source_prefix = (
                os.path.splitext(source_filename)[0].replace(".", "_").replace("-", "_")
            )

        qid_str = f"{source_prefix}_{qid_counter}"
        qid_counter += 1

        conv_data.append(
            {
                "video": video_path,
                "duration": duration,
                "timestamp": timestamp,
                "sentence": sentence,
                "qid": qid_str,
                "video_start": start,
                "video_end": end,
            }
        )

    return conv_data


def load_videomme(split="default"):
    if split == "test":
        split = "default"
    assert split in ["short", "medium", "long", "default"]
    data_root = "./dataset/videomme"
    data_path = f"{data_root}/videomme"

    conv_data = []
    data = datasets.load_dataset(
        "parquet", split="test", data_dir=data_path, streaming=True
    )
    # data = pd.read_parquet(data_path)
    for itm in data:
        if split == "default" or itm["duration"] == split:
            video_path = os.path.join(f"{data_root}/data", itm["videoID"] + ".mp4")
            conv_data.append(
                {
                    "video": video_path,
                    "question": itm["question"],
                    # "options": [op[2:].strip() for op in itm["options"]],
                    "options": [op.strip() for op in itm["options"]],
                    "answer": ord(itm["answer"]) - ord("A"),
                    "duration": itm["duration"],
                    "task_type": itm["task_type"],
                    "qid": f'videomme_{itm["question_id"]}',
                }
            )

    return conv_data


def load_egoschema(split="default"):
    assert split in ["default", "subset"]
    data_root = "./dataset/egoschema"
    if split == "subset":
        data_path = f"{data_root}/Subset"
    else:
        data_path = f"{data_root}/MC"

    conv_data = []
    data = datasets.load_dataset(
        "parquet", split="test", data_dir=data_path, streaming=True
    )
    for itm in data:
        video_path = os.path.join(f"{data_root}/videos", itm["video_idx"] + ".mp4")
        conv_data.append(
            {
                "video": video_path,
                "question": itm["question"],
                "options": [op[2:].strip() for op in itm["option"]],
                "answer": itm["answer"],
                "duration": None,
                "qid": f'egoschema_{itm["question_idx"]}',
            }
        )

    return conv_data


def load_tempcompass(split="default"):
    if split in ["default"]:
        split = "multi-choice"
    assert split in ["multi-choice", "captioning", "caption_matching", "yes_no"]
    data_root = "./dataset/tempcompass"
    data_path = f"{data_root}/questions/{split}.json"

    conv_data = []
    for key, value in json.load(open(data_path)).items():
        video_path = os.path.join(f"{data_root}/videos", key + ".mp4")
        for dim in value.keys():
            for idx, itm in enumerate(value[dim]):
                question, options, answer = itm["question"], [], itm["answer"]
                if split == "yes_no":
                    options = ["yes", "no"]
                    answer = options.index(answer)
                if split == "caption_matching":
                    tmp = question.split("\n")
                    question, options, answer = (
                        tmp[0],
                        [],
                        ":".join(answer.split(":")[1:]).strip(),
                    )
                    for i in range(1, len(tmp)):
                        option = ":".join(tmp[i].split(":")[1:]).strip()
                        options.append(option)
                    answer = options.index(answer)
                if split == "multi-choice":
                    tmp = question.split("\n")
                    question, options, answer = tmp[0], [], ord(answer[0]) - ord("A")
                    for i in range(1, len(tmp)):
                        options.append(tmp[i][2:].strip())

                conv_data.append(
                    {
                        "video": video_path,
                        "question": question,
                        "options": options,
                        "answer": answer,
                        "duration": None,
                        "qid": f"tempcompass|{split}|{key}|{dim}|{idx}",
                    }
                )

    return conv_data


def load_mvbench(split="default"):
    data_root = "./dataset/mvbench"
    data_path = f"{data_root}/json"

    DATASET_CONFIG = {
        "action_sequence": f"{data_root}/video/star/Charades_v1_480/",
        "action_prediction": f"{data_root}/video/star/Charades_v1_480/",
        "action_antonym": f"{data_root}/video/ssv2_video/",
        "fine_grained_action": f"{data_root}/video/Moments_in_Time_Raw/videos/",
        "unexpected_action": f"{data_root}/video/FunQA_test/test/",
        "object_existence": f"{data_root}/video/clevrer/video_validation/",
        "object_interaction": f"{data_root}/video/star/Charades_v1_480/",
        "object_shuffle": f"{data_root}/video/perception/videos/",
        "moving_direction": f"{data_root}/video/clevrer/video_validation/",
        "action_localization": f"{data_root}/video/sta/sta_video/",
        "scene_transition": f"{data_root}/video/scene_qa/video/",
        "action_count": f"{data_root}/video/perception/videos/",
        "moving_count": f"{data_root}/video/clevrer/video_validation/",
        "moving_attribute": f"{data_root}/video/clevrer/video_validation/",
        "state_change": f"{data_root}/video/perception/videos/",
        "fine_grained_pose": f"{data_root}/video/nturgbd/",
        "character_order": f"{data_root}/video/perception/videos/",
        "egocentric_navigation": f"{data_root}/video/vlnqa/",
        "episodic_reasoning": f"{data_root}/video/tvqa/output_videos/",
        "counterfactual_inference": f"{data_root}/video/clevrer/video_validation/",
    }

    conv_data = []
    for file_name in os.listdir(data_path):
        data_type = file_name.split(".")[0]
        data = json.load(open(os.path.join(data_path, file_name)))
        for qid, itm in enumerate(data):
            video_name = itm["video"]
            video_path = os.path.join(DATASET_CONFIG[data_type], video_name)
            conv_data.append(
                {
                    "video": video_path,
                    "question": itm["question"],
                    "options": [
                        f"{chr(65 + i)}. {opt}"
                        for i, opt in enumerate(itm["candidates"])
                    ],
                    "answer": itm["candidates"].index(itm["answer"]),
                    "duration": None,
                    "qid": f"mvbench|{data_type}|{qid}",
                }
            )
            if "start" in itm and "end" in itm:
                video_name = (
                    itm["video"].split(".mp4")[0]
                    + "_"
                    + str(itm["start"]).replace(".", "-")
                    + "_"
                    + str(itm["end"]).replace(".", "-")
                    + ".mp4"
                )
                video_path = os.path.join(
                    DATASET_CONFIG[data_type], "split", video_name
                )
                conv_data[-1]["video"] = video_path
            else:
                if "start" in itm:
                    conv_data[-1]["video_start"] = itm["start"]
                if "end" in itm:
                    conv_data[-1]["video_end"] = itm["end"]

    return conv_data


# NOTE cg-bench, cg-bench-mini 两种 split。同样数据可以做 longMCQ、clueMCQ、OE
def load_cgbench(split="default"):
    assert split in ["default", "subset"], f"split {split} not supported"
    data_root = "./dataset/cgbench"
    data_path = (
        f"{data_root}/cgbench_mini.json"
        if split == "subset"
        else f"{data_root}/cgbench.json"
    )
    # data_path = os.path.join(data_root, "cgbench_debug.json")  # debug

    conv_data = []
    data = json.load(open(data_path, "r"))
    for itm in data:
        video_path = os.path.join(
            f"{data_root}/cg_videos_720p", itm["video_uid"] + ".mp4"
        )  # NOTE 官方解压代码放到了 cg_videos_720p/， clue videos 放到 cg_clue_videos，字幕放到 cg_subtitles
        conv_data.append(
            {
                "video": video_path,
                "question": itm["question"],
                "options": [
                    f"{chr(65 + i)}. {opt}" for i, opt in enumerate(itm["choices"])
                ],
                "answer": ord(itm["right_answer"]) - ord("A"),
                "duration": itm["duration"],
                "task_type": itm["sub_category"],
                "qid": f"cgbench|{itm['qid']}",
                # "clue_intervals": itm["clue_intervals"], # clue 所在的秒数区间
            }
        )

    return conv_data


def load_auroracap(split="default"):
    assert split in [
        "default",
        "background",
        "camera",
        "detailed",
        "main_object",
        "short",
    ]
    data_root = "./dataset/auroracap"
    data_path = f"{data_root}/VDC_1k.jsonl"
    # data_path = f"{data_root}/VDC_1k_debug.jsonl" # debug
    tasks = (
        ["background", "camera", "detailed", "main_object", "short"]
        if split == "default"
        else [split]
    )
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f.readlines() if line.strip()]

    conv_data = []
    for itm in data:
        video_path = os.path.join(f"{data_root}/videos/videos", itm["video_name"])
        for task in tasks:
            conv_data.append(
                {
                    "video": video_path,
                    "answer": itm[f"{task}_caption"],
                    "qid": f"auroracap|{task}|{itm['video_id']}",
                    "task_type": task,
                }
            )
    return conv_data


def load_youcook2(split="default"):
    if split == "default":
        split = "val"
    assert split in ["train", "val", "test"]
    data_root = "./dataset/youcook2"
    data_path = (
        f"{data_root}/annotations/youcookii_annotations_test_segments_only.json"
        if split == "test"
        else f"{data_root}/annotations/youcookii_annotations_trainval.json"
    )
    # data_path = f"{data_root}/annotations/youcookii_annotations_trainval_debug.json" # debug
    data = json.load(open(data_path, "r"))["database"]

    split_map = {"train": "training", "val": "validation", "test": "testing"}
    if split in ["train", "val"]:
        data = {k: v for k, v in data.items() if v["subset"] == split_map[split]}

    conv_data = []
    for video_id, item in data.items():
        for ext in ["mp4", "mkv", "webm"]:
            video_path = os.path.join(
                f"{data_root}/raw_videos/{item['subset']}/{item['recipe_type']}",
                f"{video_id}.{ext}",
            )
            if os.path.exists(video_path):
                break

        conv_data.append(
            {
                "video": video_path,
                "duration": item["duration"],
                "answer": item[
                    "annotations"
                ],  # [{"segment": [start, end], "sentence": str, "id": int}, ...]
                "qid": f"youcook2|{video_id}",
            }
        )
    return conv_data
