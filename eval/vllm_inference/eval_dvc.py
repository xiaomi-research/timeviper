"""Tools for evaluating dense captions.

Reimplements evaluation metrics that agree with open-sourced methods at
https://github.com/ranjaykrishna/densevid_eval/blob/master/evaluate.py
"""

import collections
import logging
import pdb
import random
import re
import string

import numpy as np

from eval.metrics.cider import Cider
from eval.metrics.meteor import Meteor
from eval.metrics.ptbtokenizer import PTBTokenizer


def random_string(string_length):
    """Random string generator for unmatched captions."""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(string_length))


def iou(interval_1, interval_2):
    """Compute the IOU between two intervals.

    Args:
        interval_1: A tuple (start, end) containing the first interval.
        interval_2: A tuple (start, end) containing the second interval.

    Returns:
        The IOU of the two intervals.
    """
    start_1, end_1 = float(min(*interval_1)), float(max(*interval_1))
    start_2, end_2 = float(min(*interval_2)), float(max(*interval_2))

    intersection = max(0, min(end_1, end_2) - max(start_1, start_2))
    union = min(
        max(end_1, end_2) - min(start_1, start_2), end_1 - start_1 + end_2 - start_2
    )
    result = float(intersection) / (union + 1e-8)
    return result


def evaluate_detections(
    predicted_segments, gt_segments, splits, iou_thresholds=(0.3, 0.5, 0.7, 0.9)
):
    """Compute the mean P/R between the predicted and ground truth segments.

    Args:
        predicted_segments: A numpy array of shape [K x 2] containing the predicted
            segments.
        gt_segments: A numpy array of shape [S x 2] containing the ground truth
            segments.
        splits: A numpy array of shape [S] indicating the annotation set.
        iou_thresholds: The IOU thresholds to use for Precision/Recall calculations.

    Returns:
        precision: The mean precision of the predictions over the IOU thresholds.
        recall: The mean recall of the predictions over the IOU thresholds.
        best_miou: The mIoU.
        iou_matrices: dictionary mapping each split to the corresponding iou matrix.
    """
    # Recall is the percentage of ground truth that is covered by the predictions.
    # Precision is the percentage of predictions that are valid.

    best_recall = []
    best_precision = []
    iou_matrices = {}

    predicted_shape = predicted_segments.shape[0]

    for split in set(splits):
        metrics = {}
        for threshold in iou_thresholds:
            metrics[str(threshold)] = {
                "gt_covered": set(),
                "pred_covered": set(),
            }
        split_idx = np.where(splits == split)[0]
        split_gt_segments = np.array([gt_segments[idx] for idx in split_idx])

        gt_shape = split_gt_segments.shape[0]

        # Compute the IOUs for the segments.
        iou_matrix = np.zeros((gt_shape, max(predicted_shape, 1)))
        for idx_g, gt_segment in enumerate(split_gt_segments):
            cur_max_iou = 0
            for idx_p, segment in enumerate(predicted_segments):
                sample_iou = iou(segment, gt_segment)
                iou_matrix[idx_g, idx_p] = sample_iou
                cur_max_iou = max(cur_max_iou, sample_iou)
                for threshold in iou_thresholds:
                    if sample_iou > threshold:
                        metrics[str(threshold)]["pred_covered"].add(idx_p)
                        metrics[str(threshold)]["gt_covered"].add(idx_g)

        # Compute the precisions and recalls for each IOU threshold.
        for threshold, m in metrics.items():
            pred_covered = m["pred_covered"]
            gt_covered = m["gt_covered"]

            # Avoid dividing by 0 for precision
            m["precision"] = float(len(pred_covered)) / max(float(predicted_shape), 1.0)
            m["recall"] = float(len(gt_covered)) / float(gt_shape)

        precision = [m["precision"] for m in metrics.values()]
        recall = [m["recall"] for m in metrics.values()]
        if best_precision:
            best_precision = [
                max(precision[i], best_precision[i]) for i in range(len(precision))
            ]
            best_recall = [max(recall[i], best_recall[i]) for i in range(len(recall))]
        else:
            best_precision, best_recall = precision, recall
        iou_matrices[int(split)] = iou_matrix

    return best_precision, best_recall, iou_matrices


def match_captions(
    predicted_segments,
    gt_segments,
    predicted_captions,
    gt_captions,
    iou_thresholds=(0.3, 0.5, 0.7, 0.9),
):
    """Matches the predicted captions to ground truth using the IOU thresholds.

    Args:
     predicted_segments: A numpy array of shape [K x 2] containing the predicted
         segment intervals.
     gt_segments: A numpy array of shape [S x 2] containing the ground truth
         segment intervals.
     predicted_captions: A list of string of shape [K] containing the
         corresponding K predicted captions.
     gt_captions: A list of strings of shape [S] containing the corresponding S
         ground truth captions.
     iou_thresholds: A list of thresholds for IOU to average over.

    Returns:
     ground_truths_filtered: Filtered list of ground truth captions for all
        threshold.
     predictions_filtered: Matching list of predicted captions for all
        threshold.
     isxes: For each threshold, contains lists of isx of matches.
    """

    # Setup a set of dictionaries to hold the results.
    ground_truths_filtered = {str(threshold): {} for threshold in iou_thresholds}
    predictions_filtered = {str(threshold): {} for threshold in iou_thresholds}

    # Create GT lists for each of the IOU thresholds.
    isx = 0
    isxes = {str(threshold): [] for threshold in iou_thresholds}
    for idx_p, segment in enumerate(predicted_segments):
        pc_idxp = predicted_captions[idx_p]
        added = {str(threshold): False for threshold in iou_thresholds}
        for idx_g, gt_segment in enumerate(gt_segments):
            gt_idxg = gt_captions[idx_g]
            sample_iou = iou(segment, gt_segment)
            for threshold in iou_thresholds:
                if sample_iou >= threshold:
                    key = str(isx)
                    isxes[str(threshold)].append(isx)
                    isx += 1
                    ground_truths_filtered[str(threshold)][key] = [{"caption": gt_idxg}]
                    predictions_filtered[str(threshold)][key] = [{"caption": pc_idxp}]
                    added[str(threshold)] = True
        for threshold in iou_thresholds:
            if not added[str(threshold)]:
                key = str(isx)
                isxes[str(threshold)].append(isx)
                isx += 1
                # Set this to a random string with no match to the predictions to
                # get a zero score
                ground_truths_filtered[str(threshold)][key] = [
                    {"caption": random_string(random.randint(10, 20))}
                ]
                predictions_filtered[str(threshold)][key] = [{"caption": pc_idxp}]

    return ground_truths_filtered, predictions_filtered, isxes


def chased_dp_assignment(scores):
    """Run dp matching as https://github.com/fujiso/SODA/blob/master/soda.py."""

    m, n = scores.shape
    dp = -np.ones((m, n))
    path = np.zeros((m, n))

    def transition(i, j):
        if dp[i, j] >= 0:
            return dp[i, j]
        elif i == 0 and j == 0:
            state = [-1, -1, scores[i, j]]
        elif i == 0:
            state = [-1, transition(i, j - 1), scores[i, j]]
        elif j == 0:
            state = [transition(i - 1, j), -1, scores[i, j]]
        else:
            state = [
                transition(i - 1, j),
                transition(i, j - 1),
                transition(i - 1, j - 1) + scores[i, j],
            ]
        dp[i, j] = np.max(state)
        path[i, j] = np.argmax(state)
        return dp[i, j]

    def get_pairs(i, j):
        p = np.where(path[i][: j + 1] == 2)[0]
        # pylint: disable=g-explicit-length-test
        if i != 0 and not len(p):
            return get_pairs(i - 1, j)
        elif i == 0 or p[-1] == 0:
            return [(i, p[-1])]
        else:
            return get_pairs(i - 1, p[-1] - 1) + [(i, p[-1])]

    n, m = scores.shape
    max_score = transition(n - 1, m - 1)
    pairs = get_pairs(n - 1, m - 1)
    return max_score, pairs


def sodac(
    iou_matrices, scorer, predicted_captions, gt_captions, splits, iou_thresholds=(0.0,)
):
    """SODA_c from https://github.com/fujiso/SODA/."""
    if not predicted_captions:
        return {int(split): 0 for split in splits}

    res = {str(index): [p] for index, p in enumerate(predicted_captions)}
    unique_splits = set(splits)
    fs = {int(split): [0] * len(iou_thresholds) for split in unique_splits}
    for split in unique_splits:
        split_idx = np.where(splits == split)[0]
        split_gt_captions = [gt_captions[idx] for idx in split_idx]
        gts = [{index: [x] for index in res} for x in split_gt_captions]
        iou_matrix = iou_matrices[int(split)]
        score_matrix = np.array(
            [np.nan_to_num(scorer.compute_score(res, gt)[1]) for gt in gts]
        )
        for i, threshold in enumerate(iou_thresholds):
            iou_cur = np.copy(iou_matrix)
            iou_cur[iou_cur < threshold] = 0.0
            max_score, _ = chased_dp_assignment(iou_cur * score_matrix)
            (n_g, n_p) = iou_cur.shape
            p = max_score / n_p
            r = max_score / n_g
            fs[int(split)][i] = 2 * p * r / (p + r) if p + r > 0 else 0
    for split in unique_splits:
        fs[int(split)] = np.mean(fs[int(split)])
    return fs


def evaluate_single_dense_captions(
    predicted_segments,
    gt_segments,
    predictions_filtered,
    ground_truths_filtered,
    predicted_captions,
    gt_captions,
    splits,
    keys,
    iou_thresholds=(0.3, 0.5, 0.7, 0.9),
    soda=False,
    tmponly=False,
    scorers=None,
):
    """Compute both the P/R and NLP metrics for the given predictions.

    Args:
     predicted_segments: A numpy arrays, of shape [K x 2]
         containing the predicted segment intervals.
     gt_segments: A numpy arrays, of shape [S x 2]
         containing the ground truth segment intervals.
     predictions_filtered: Matching list of predicted captions for each threshold.
     ground_truths_filtered: Filtered list of ground truth captions for each
        threshold.
     predicted_captions: A list, of string of shape [K]
         containing the corresponding K predicted captions.
     gt_captions: A list, of strings of shape [S] containing the
         corresponding S ground truth captions.
     splits: A numpy array, of shape [S] indicating
         the annotation set (1/2 for ActivityNet).
     keys: A string
     iou_thresholds: A list of thresholds for IOU to average over.
     soda: Whether to compute SODA or not.
     tmponly: In this case do not compute captioning metrics.
     scorers: dictionary mapping strings to scorers.

    Returns:
        (precision, recall): The precision and recall of the detections averaged
        over the IOU thresholds.
        metrics: The NLP metrics of the predictions averaged over the IOU
            thresholds.
    """
    if scorers is None:
        scorers = {}

    # Localization
    detection_precision, detection_recall, iou_matrices = evaluate_detections(
        predicted_segments, gt_segments, splits, iou_thresholds
    )

    # Captions
    n_preds = len(predicted_captions)
    if not tmponly:
        metric_tiou = evaluate_caption_scores(
            ground_truths_filtered, predictions_filtered, iou_thresholds, scorers
        )
        if soda:
            fs = sodac(
                iou_matrices,
                scorers["METEOR"],
                predicted_captions,
                gt_captions,
                splits,
                (0.0,),
            )
    else:
        metric_tiou = {}

    mean_precision = sum(detection_precision) / len(detection_precision)
    mean_recall = sum(detection_recall) / len(detection_recall)
    for j, threshold in enumerate(iou_thresholds):
        metric_tiou[f"Precision@{threshold}"] = float(detection_precision[j])
        metric_tiou[f"Recall@{threshold}"] = float(detection_recall[j])
    metric_tiou["Precision_Mean"] = float(mean_precision)
    metric_tiou["Recall_Mean"] = float(mean_recall)
    metric_tiou["F1_Score"] = (
        2
        * float(mean_recall)
        * float(mean_precision)
        / (float(mean_recall) + float(mean_precision))
        if float(mean_recall) + float(mean_precision) > 0
        else 0
    )
    if soda and not tmponly:
        for split in fs:
            metric_tiou[f"SODA_c_{split}"] = float(fs[split])
    metric_tiou["n_preds"] = n_preds
    metric_tiou["key"] = keys

    return metric_tiou


def evaluate_caption_scores(
    ground_truths_filtered,
    predictions_filtered,
    iou_thresholds=(0.3, 0.5, 0.7, 0.9),
    scorers=None,
):
    """Compute the mean NLP metrics over the given IOU thresholds.

    Args:
     ground_truths_filtered: Filtered list of ground truth captions for each
        threshold.
     predictions_filtered: Matching list of predicted captions for each threshold.
     iou_thresholds: A list of thresholds for IOU to average over.
     scorers: A dictionary of scorers.

    Returns:
     metrics: dictionary with mean captioning score across the threshold set.
    """

    if scorers is None:
        scorers = {}

    # Compute the caption metrics.
    metrics = collections.defaultdict(list)
    for scorer_name, scorer in scorers.items():
        for threshold in iou_thresholds:
            # Handle the case where we have no overlapping truths
            if not ground_truths_filtered[str(threshold)]:
                metrics[scorer_name].append(0.0)
            elif not predictions_filtered[str(threshold)]:
                metrics[scorer_name].append(0.0)
            else:
                score = scorer.compute_score(
                    ground_truths_filtered[str(threshold)],
                    predictions_filtered[str(threshold)],
                )
                score = np.nan_to_num(score[0])
                metrics[scorer_name].append(score)

    # Aggregate the caption metrics.
    for key, value in metrics.items():
        metrics[key] = np.mean(value)

    return metrics


def evaluate_dense_captions(
    predicted_segments,
    gt_segments,
    predicted_captions,
    gt_captions,
    splits,
    keys,
    iou_thresholds=(0.3, 0.5, 0.7, 0.9),
    soda=False,
    tmponly=False,
):
    """Compute both the P/R and NLP metrics for the given predictions.

    This is the same as calling the above functions, however it aggregates the
    metrics generated by evaluate_detections and evaluate_caption_scores across
    a list of inputs.

    Args:
     predicted_segments: A list of numpy arrays, of shape [K x 2]
         containing the predicted segment intervals.
     gt_segments: A list of numpy arrays, of shape [S x 2]
         containing the ground truth segment intervals.
     predicted_captions: A list of lists, of string of shape [K]
         containing the corresponding K predicted captions.
     gt_captions: A list of lists, of strings of shape [S] containing the
         corresponding S ground truth captions.
     splits: A list of numpy arrays, of shape [S] indicating
         the annotation set (1/2 for ActivityNet).
     keys: A list of strings
     iou_thresholds: A list of thresholds for IOU to average over.
     soda: Whether to compute SODA or not.
     tmponly: In this case do not compute captioning metrics.

    Returns:
        (precision, recall): The precision and recall of the detections averaged
        over the IOU thresholds.
        metrics: The NLP metrics of the predictions averaged over the IOU
            thresholds.
    """

    # Handle if these are lists, or single samples.
    assert all([isinstance(p, list) for p in [predicted_segments, gt_segments]])
    # Only construct the scorers once, so that we don't have any issues with
    # overhead when running multiple evaluations.
    scorers = {
        "CIDER": Cider(),
        "METEOR": Meteor(),
    }
    tokenizer = PTBTokenizer()
    metric_tiou = collections.defaultdict(list)
    gts = {str(threshold): {} for threshold in iou_thresholds}
    preds = {str(threshold): {} for threshold in iou_thresholds}
    vid2isx = {str(threshold): {} for threshold in iou_thresholds}

    assert (
        len(predicted_segments)
        == len(gt_segments)
        == len(predicted_captions)
        == len(gt_captions)
        == len(splits)
    )

    # Compute matches
    for pred_seg, gt_seg, pred_cap, gt_cap, key in zip(
        predicted_segments,
        gt_segments,
        predicted_captions,
        gt_captions,
        keys,
    ):
        gt, pred, isxes = match_captions(
            pred_seg, gt_seg, pred_cap, gt_cap, iou_thresholds
        )
        # Flatten for tokenization
        for threshold in iou_thresholds:
            for k, v in gt[str(threshold)].items():
                gts[str(threshold)][key + "_" + str(k)] = v
            for k, v in pred[str(threshold)].items():
                preds[str(threshold)][key + "_" + str(k)] = v
            vid2isx[str(threshold)][key] = isxes[str(threshold)]

    # Call tokenization once
    for threshold in iou_thresholds:
        gts[str(threshold)] = tokenizer.tokenize(gts[str(threshold)])
        preds[str(threshold)] = tokenizer.tokenize(preds[str(threshold)])

    # Tokenize also the original lists for SODA computation
    predicted_captions_dict = {  # pylint: disable=g-complex-comprehension
        keys[i] + "_" + str(j): [{"caption": p}]
        for i, ps in enumerate(predicted_captions)
        for j, p in enumerate(ps)
    }
    gt_captions_dict = {  # pylint: disable=g-complex-comprehension
        keys[i] + "_" + str(j): [{"caption": g}]
        for i, gs in enumerate(gt_captions)
        for j, g in enumerate(gs)
    }
    predicted_captions_tok = tokenizer.tokenize(predicted_captions_dict)
    gt_captions_tok = tokenizer.tokenize(gt_captions_dict)
    predicted_captions_res = []
    gt_captions_res = []
    for i, ps in enumerate(predicted_captions):
        res = [
            predicted_captions_tok[keys[i] + "_" + str(j)][0] for j, _ in enumerate(ps)
        ]
        predicted_captions_res.append(res)
    for i, gs in enumerate(gt_captions):
        res = [gt_captions_tok[keys[i] + "_" + str(j)][0] for j, _ in enumerate(gs)]
        gt_captions_res.append(res)

    # Reshape
    final_gts = {str(threshold): {} for threshold in iou_thresholds}
    final_preds = {str(threshold): {} for threshold in iou_thresholds}
    for threshold in iou_thresholds:
        for key in keys:
            final_gts[str(threshold)][key] = {
                str(k): gts[str(threshold)][key + "_" + str(k)]
                for k in vid2isx[str(threshold)][key]
            }
            final_preds[str(threshold)][key] = {
                str(k): preds[str(threshold)][key + "_" + str(k)]
                for k in vid2isx[str(threshold)][key]
            }

    # Compute dense video captioning metrics at the video level
    for i, key in enumerate(keys):
        pred_filt_i = {str(t): final_preds[str(t)][key] for t in iou_thresholds}
        gt_filt_i = {str(t): final_gts[str(t)][key] for t in iou_thresholds}
        res = evaluate_single_dense_captions(
            predicted_segments[i],
            gt_segments[i],
            pred_filt_i,
            gt_filt_i,
            predicted_captions_res[i],
            gt_captions_res[i],
            splits[i],
            key,
            iou_thresholds,
            soda,
            tmponly,
            scorers,
        )
        for met in res:
            metric_tiou[met].append(res[met])
        if soda:
            if "SODA_c_1" not in res:
                metric_tiou["SODA_c_1"].append(-1)
            if "SODA_c_2" not in res:
                metric_tiou["SODA_c_2"].append(-1)

    logging.info("Closing Meteor")
    with scorers["METEOR"].lock:
        scorers["METEOR"].meteor_p.stdin.close()
        scorers["METEOR"].meteor_p.stdout.close()
        scorers["METEOR"].meteor_p.kill()
        scorers["METEOR"].meteor_p.wait()
    del scorers

    return metric_tiou


def parse_sent(sent):
    """Sentence preprocessor."""
    res = re.sub("[^a-zA-Z]", " ", sent)
    res = res.strip().lower().split()
    return res


def evaluate_para(predicted_captions, gt_captions):
    """Paragraph-level evaluation.

    Args:
     predicted_captions: A list of strings (paragraphs).
     gt_captions: A list of lists (multi-ref) of strings (paragraphs).

    Returns:
        metrics: The NLP metrics of the predictions computed at the corpus level.
    """
    scorers = {
        "CIDER": Cider(),
        "METEOR": Meteor(),
    }
    all_gts = {}
    all_preds = {}
    for i, (preds, gts) in enumerate(zip(predicted_captions, gt_captions)):
        all_preds[str(i)] = [" ".join(parse_sent(preds))]
        all_gts[str(i)] = [" ".join(parse_sent(gt)) for gt in gts]

    metrics = collections.defaultdict(list)
    for scorer_name, scorer in scorers.items():
        score = scorer.compute_score(all_gts, all_preds)
        score = np.nan_to_num(score[0])
        metrics["Para_" + scorer_name] = float(score)

    logging.info("Closing Meteor")
    with scorers["METEOR"].lock:
        scorers["METEOR"].meteor_p.stdin.close()
        scorers["METEOR"].meteor_p.stdout.close()
        scorers["METEOR"].meteor_p.kill()
        scorers["METEOR"].meteor_p.wait()
    del scorers

    return metrics


# =========== parsing ==========
def extract_time_part(time_part):
    radius = 20
    # remove 1. 2. 3. etc.
    extracted_time_part = re.compile(r"\d+\.*\d*\s*-\s*\d+\.*\d*").findall(time_part)
    if len(extracted_time_part) == 0:
        if time_part.count(":") == 1:
            # for 1. The video starts at 0:00.
            extracted_time = re.compile(r"\d+\.*\d*:\d+\.*\d*").findall(time_part)[0]
            extracted_time = int(extracted_time.split(":")[0]) * 60 + int(
                extracted_time.split(":")[1]
            )
            if extracted_time > radius:
                extracted_time_part = [
                    f"{extracted_time - radius} - {extracted_time + radius}"
                ]
            else:
                extracted_time_part = [
                    f"{extracted_time} - {extracted_time + 2*radius}"
                ]
        elif time_part.count(":") == 2:
            # for * Using a wok to cook dishes (from 1:09 to 1:20)
            start, end = re.compile(r"\d+\.*\d*:\d+\.*\d*").findall(time_part)
            start_seconds = int(start.split(":")[0]) * 60 + int(start.split(":")[1])
            end_seconds = int(end.split(":")[0]) * 60 + int(end.split(":")[1])
            extracted_time_part = [f"{start_seconds} - {end_seconds}"]
        else:
            pass
    if len(extracted_time_part) == 0:
        extracted_time_part = re.compile(r"\d+\.*\d*(?!\.)").findall(time_part)
        if len(extracted_time_part) == 1:
            # for start - 180 seconds, Add 1/4 cup of olive oil to the pan
            extracted_time = float(extracted_time_part[0])
            if extracted_time > radius:
                extracted_time_part = [
                    f"{extracted_time - radius} - {extracted_time + radius}"
                ]
            else:
                extracted_time_part = [
                    f"{extracted_time} - {extracted_time + 2 * radius}"
                ]
        elif len(extracted_time_part) == 2:
            # for 10s-38s, Sharpener was instructed to get a pair of scissors and cut off the top of the pineapple.
            extracted_time_part = [
                f"{extracted_time_part[0]} - {extracted_time_part[1]}"
            ]
        else:
            pass
    return extracted_time_part


def extract_time_from_para(paragraph):
    paragraph = paragraph.lower()
    patterns = [
        (
            r"(?:from\s*)?(\d+\.*\d*)\s*(?:-|to)\s*(\d+\.*\d*)",
            r"((?:from\s*)?\d+\.*\d*\s*(?:-|to)\s*\d+\.*\d*)",
        )  # n - m, caption
    ]
    timestamps = []
    captions = []

    # Check for m - n, captions (no seconds)
    for time_pattern, string_pattern in patterns:
        time_matches = re.findall(time_pattern, paragraph, re.IGNORECASE)
        string_matches = re.findall(string_pattern, paragraph, re.IGNORECASE)

        if time_matches:
            # n - m, caption
            timestamps = [[float(start), float(end)] for start, end in time_matches]
            # get captions
            rest_para = paragraph
            for time_string in string_matches:
                rest_para = rest_para.replace(time_string, "\n")
            captions = rest_para.replace("seconds", "").split("\n")
        if len(timestamps) > 0:
            break

    # Check for 'Start time: N seconds' and 'End time: M seconds' format, e.g.
    #   4. Start time: 113 seconds
    #   End time: 116 seconds
    #   Description: Spreading brownies in a pan
    if len(timestamps) == 0:
        start_time_pattern = r"(?:start(?:ing)? time: (\d+\.*\d*)(?:s| seconds)?)"
        end_time_pattern = r"(?:end(?:ing)? time: (\d+\.*\d*)(?:s| seconds)?)"
        end_matches = re.findall(end_time_pattern, paragraph, re.DOTALL | re.IGNORECASE)
        start_matches = re.findall(
            start_time_pattern, paragraph, re.DOTALL | re.IGNORECASE
        )

        if start_matches and end_matches:
            timestamps = [
                [float(start), float(end)]
                for start, end in zip(start_matches, end_matches)
            ]
            captions = re.findall(r"description: (.*)", paragraph)
            if len(captions) == 0:
                captions = re.findall(r"\*\s*(.*)", paragraph)

    # Check for 'start time X.X, end time Y.Y' format
    if len(timestamps) == 0:
        start_end_matches = re.findall(
            r"start time (\d+\.*\d*), end time (\d+\.*\d*)", paragraph
        )
        if start_end_matches:
            timestamps = list(start_end_matches)
            for start, end in start_end_matches:
                paragraph = paragraph.replace(
                    f"start time {start}, end time {end}", "\n"
                )
                captions = paragraph.split("\n")
            if len(timestamps) > 0:
                pdb.set_trace()

    captions = [c.strip().strip(", ").rstrip() for c in captions if len(c) > 5]
    min_len = min(len(timestamps), len(captions))
    timestamps = timestamps[:min_len]
    captions = captions[:min_len]
    assert len(timestamps) == len(
        captions
    ), f"# timestamps {len(timestamps)}, # captions {len(captions)}, para {paragraph}."
    return timestamps, captions


def parse_dvc_prediction(caption):
    """
    Parse the DVC prediction to extract timestamps and captions.

    Args:
        caption (str): The generated caption string from the model.

    Returns:
        tuple:
            - timestamps (List[List[float]]): List of [start, end] timestamps.
            - sents (List[str]): List of corresponding captions.
    """
    timestamps = []
    sents = []
    # type 1: directly detect timestamps in generated paragraph to process multi-lines cases like:
    #   1. Start time: 105
    #   End time: 109
    #   Description: Making brown sugar sandwiches with white bread
    paras = caption
    timestamps, sents = extract_time_from_para(paras)

    # type 2：detect timestamps in splited sentences
    if len(timestamps) == 0:
        caps = []
        if "\n" in caption:
            caps = caption.split("\n")
            caps = [c for c in caps if len(c) > 7]
        if len(caps) <= 1:
            raw_caps = caption.split(". ")
            caps = [c for c in raw_caps if len(c) > 7]
            caps = [c + "." for c in caps]
        for cap in caps:
            try:
                parts = cap.split("seconds")
                parts = [p.strip(",") for p in parts]
                time_part = parts[0]
                extracted_time_part = extract_time_part(time_part)
                if len(extracted_time_part) == 0:
                    continue
                else:
                    time_part = extracted_time_part[0]
                sent_part = parts[-1]
                stime = round(float(time_part.split("-")[0].strip()), 2)
                etime = round(float(time_part.split("-")[1].strip()), 2)
                timestamps.append([stime, etime])
                sents.append(sent_part.strip())
            except:
                continue

    return timestamps, sents


def evaluate_youcook2_dvc(pred_data):
    """
    Evaluate DVC predictions on youcook2 format data.

    Args:
        pred_data: List of prediction dictionaries with format:
            {
                "qid": "youcook2|video_id",
                "pred": {"timestamps": [...], "captions": [...]},
                "target": [{"segment": [start, end], "sentence": str, "id": int}, ...],  # ground truth
                "output_text": predicted_dvc_text
                "duration": video_duration_in_seconds
            }

    Returns:
        Dictionary with evaluation metrics
    """
    gt_segments = []
    gt_captions = []
    predicted_segments = []
    predicted_captions = []
    splits = []
    keys = []
    gt_paras = []
    predicted_paras = []
    for item in pred_data:
        qid = item["qid"]
        keys.append(qid)
        # gt
        gt_segs = [x["segment"] for x in item["target"]]
        gt_caps = [x["sentence"] + "." for x in item["target"]]
        gt_segments.append(np.array(gt_segs))
        gt_captions.append(np.array(gt_caps))
        gt_paras.append([" ".join(gt_caps)])
        splits.append(
            np.array([0] * len(gt_segs))
        )  # youcook2 没有标注员划分，全部设为0
        # prediction
        pred_segs = item["pred"]["timestamps"]
        pred_caps = item["pred"]["captions"]
        predicted_segments.append(np.array(pred_segs))
        predicted_captions.append(np.array(pred_caps))
        predicted_paras.append(". ".join(pred_caps))

    # paragraph video captioning
    print("paragraph video captioning")
    para_results = evaluate_para(predicted_paras, gt_paras)

    # dense video captioning
    print("dense video captioning")
    dvc_results = evaluate_dense_captions(
        predicted_segments,
        gt_segments,
        predicted_captions,
        gt_captions,
        splits,
        keys,
        soda=True,
    )

    return {**para_results, **dvc_results}
