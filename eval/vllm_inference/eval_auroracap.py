import ast
import json
import os

import torch
from sglang import (
    RuntimeEndpoint,
    assistant,
    function,
    gen,
    set_default_backend,
    system,
    user,
)
from sglang.utils import (
    launch_server_cmd,
    print_highlight,
    terminate_process,
    wait_for_server,
)
from tqdm import tqdm

# HACK: set your local LLaMA-3.1-8B-Instruct checkpoint path here
AURORACAP_JUDGE_PATH = (
    os.environ.get("AURORACAP_JUDGE_PATH", None) or "meta-llama/Llama-3.1-8B-Instruct"
)


def start_sglang_server(
    model_path="meta-llama/Llama-3.1-8B-Instruct", dp=1, nnodes=1, node_rank=0
):
    """Starts the SGLang server as a background process."""

    server_process, port = launch_server_cmd(
        f"python3 -m sglang.launch_server --model-path {model_path} --log-level warning --dp {dp} --nnodes {nnodes} --node-rank {node_rank}",
    )

    wait_for_server(f"http://localhost:{port}")

    return server_process, port


@function
def gener_pred_response(s, pred_cap, q):
    s += system(
        "You are an intelligent chatbot designed for providing accurate answers to questions related to the content based on a detailed description of a video or image."
        "Here's how you can accomplish the task:"
        "------"
        "##INSTRUCTIONS: "
        "- Read the detailed description carefully.\n"
        "- Answer the question only based on the detailed description.\n"
        "- The answer should be a short sentence or phrase.\n"
    )
    s += user(
        "Please provide accurate answers to questions related to the content based on a detailed description of a video or image:\n\n"
        f"detailed description: {pred_cap}, question: {q}"
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide short but accurate answer."
    )
    s += assistant(gen("answer_1", max_tokens=256))


@function
def gener_pred_score(s, qa):
    s += system(
        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
        "------"
        "##INSTRUCTIONS: "
        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
        "- Consider synonyms or paraphrases as valid matches.\n"
        "- Evaluate the correctness of the prediction compared to the answer."
    )
    s += user(
        "Please evaluate the following video-based question-answer pair:\n\n"
        f"Question: {qa['question']}\n"
        f"Correct Answer: {qa['answer']}\n"
        f"Predicted Answer: {qa['response']}\n\n"
        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
    )
    s += assistant(gen("answer_1", max_tokens=256))


def get_auroracap_score(pred_dict, split, eval_root=None):
    """
    评估AuroraCap分数,支持断点恢复

    Args:
        pred_dict: 预测结果字典列表
        split: 评估的split类型
        eval_root: 保存中间结果的根目录,如果为None则不保存中间结果
    """
    try:
        # 创建保存目录
        if eval_root:
            os.makedirs(eval_root, exist_ok=True)
            result_file = os.path.join(eval_root, "eval_scores.jsonl")

            # 加载已有的评估结果
            existing_results = {}
            if os.path.exists(result_file):
                print(f"Loading existing results from {result_file}")
                with open(result_file, "r") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            qid = item["qid"]
                            existing_results[qid] = {
                                "score": item["score"],
                                "acc": item["acc"],
                                "task_type": item["task_type"],
                            }
                print(f"Loaded {len(existing_results)} existing results")
        else:
            existing_results = {}
            result_file = None

        print("Starting SGLang server...")
        # nnodes = os.environ.get("NNODES", 1)
        node_rank = os.environ.get("NODE_RANK", 0)
        num_gpus = torch.cuda.device_count()
        # print(f"nnodes: {nnodes}, node_rank: {node_rank}, num_gpus: {num_gpus}")

        server_process, port = start_sglang_server(
            model_path=AURORACAP_JUDGE_PATH, dp=num_gpus, node_rank=node_rank
        )
        print_highlight(f"SGLang server started on port {port}")

        set_default_backend(RuntimeEndpoint(f"http://localhost:{port}"))

        if split == "default":
            split = ["background", "camera", "detailed", "main_object", "short"]
        else:
            split = [split]
        gt_qas = {}
        for task in split:
            with open(f"dataset/auroracap/VDCScore_qa/{task}.jsonl", "r") as f:
                gt_qas[task] = {}
                for line in f:
                    item = json.loads(line)
                    gt_qas[task].update(item)

        results = existing_results.copy()  # 从已有结果开始

        # 过滤出需要处理的项目
        items_to_process = [
            item for item in pred_dict if item["qid"] not in existing_results
        ]
        print(
            f"Total items: {len(pred_dict)}, Already processed: {len(existing_results)}, To process: {len(items_to_process)}"
        )

        # 使用 a+ 模式打开文件,保持文件句柄
        result_fp = None
        if result_file:
            result_fp = open(result_file, "a")

        try:
            for item in tqdm(items_to_process, desc="Processing items"):
                qid = item["qid"]
                video_id = qid.split("|")[-1]
                pred = item["pred"]
                task_type = item["task_type"]
                # answer = item['answer'] # scoring does not need the GT caption

                try:
                    # TP
                    # step 1: generate QA from the ground truth caption
                    if video_id not in gt_qas[task_type].keys():
                        print(
                            f"Warning: video_id {video_id} not found in GT QA for task {task_type}"
                        )
                        continue

                    result_gtqa_list = gt_qas[task_type][video_id]

                    tp_result_dict = {
                        "id": video_id,
                        "pred_caption": pred,
                        # 'gt_caption': answer,
                        "qa_tp_list": [],
                    }
                    # step 2: generate response for each question
                    qa_list = []
                    for qa_dict in result_gtqa_list:
                        temp_dict = {
                            "question": qa_dict["question"],
                            "answer": qa_dict["answer"],
                        }

                        state = gener_pred_response.run(
                            pred_cap=pred,
                            q=qa_dict["question"],
                        )

                        temp_dict["response"] = state["answer_1"]

                        qa_list.append(temp_dict)
                    # step 3: match the generated answers with the ground truth answers
                    for qa in qa_list:
                        state = gener_pred_score.run(
                            qa=qa,
                        )
                        response_dict = ast.literal_eval(state["answer_1"])

                        qa.update(response_dict)

                    # step 4: calculate the final score
                    total_score, total_acc = 0, 0
                    for qa in qa_list:
                        # print('qa', qa) # debug
                        total_score += float(qa["score"])
                        tp_result_dict["qa_tp_list"].append(qa)
                        if qa["pred"] == "yes":
                            total_acc += 1
                    tp_score = total_score / len(qa_list)
                    tp_acc = total_acc / len(qa_list)

                    results[qid] = {
                        "score": tp_score,
                        "acc": tp_acc,
                        "task_type": task_type,
                    }

                    # 保存单个结果到jsonl文件
                    if result_fp:
                        result_item = {
                            "qid": qid,
                            "score": tp_score,
                            "acc": tp_acc,
                            "task_type": task_type,
                        }
                        result_fp.write(json.dumps(result_item) + "\n")
                        result_fp.flush()

                except Exception as e:
                    print(f"Error processing {qid}: {e}")
                    import traceback

                    traceback.print_exc()
        finally:
            if result_fp:
                result_fp.close()

        return results

    finally:
        # close the server
        terminate_process(server_process)
