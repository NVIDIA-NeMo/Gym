# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted material: Apache-2.0, Copyright 2023 VLMEvalKit Authors.
# Source: https://github.com/zhuzil/VTC-Bench/blob/5e0a5533f597b0c56dae7d9705771498202ab19b/eval/eval/VLMEvalKit/vlmeval/dataset/image_vqa.py
# License: https://github.com/zhuzil/VTC-Bench/blob/5e0a5533f597b0c56dae7d9705771498202ab19b/eval/eval/VLMEvalKit/LICENSE
"""VTCBench rules and judge prompts adapted from the public evaluator.

Source: https://github.com/zhuzil/VTC-Bench/blob/
5e0a5533f597b0c56dae7d9705771498202ab19b/eval/eval/VLMEvalKit/vlmeval/dataset/image_vqa.py
"""

MCQ_PROMPT = (
    "You are a information extractor that extracts multiple choice letter answer choices from a "
    "paragraph that contains the answer choice and sometimes explaination of why that choice is "
    "correct to the given question.\n"
    "What letter did the following answer choose? If the answer did not select a letter answer "
    "choice, first try to infer the answer based off the given choices.\n"
    "If it does not seem like the given answer corresponds to an answer choice OR if there is no "
    "selected answer, please just respond with Z.\n"
    "If the answer sentence is incomplete or partial,  simply make a mistake and respond with Z.\n"
    "Make sure you answer with ONLY the letters chosen.\n"
    "Example 1: \n"
    "Question: <start>\n"
    "What is the main object in image?\n"
    "Options: A. teddy bear B. rabbit C. cat D. dog\n"
    "<end>\n"
    "Answer: <start>\n"
    "a cute teddy bear\n"
    "<end>\n"
    "Your output: A\n"
    "Example 2: \n"
    "Question: <start>\n"
    "What is the main object in image?\n"
    "Options: A. teddy bear B. rabbit C. cat D. dog\n"
    "<end>\n"
    "Answer: <start>\n"
    "Spider\n"
    "<end>\n"
    "Your output: Z\n"
    "Question: <start>\n"
    "{}\n"
    "Options: {}<end>\n"
    "Answer: <start>\n"
    "{}\n"
    "<end>\n"
    "Your output: "
)
VQA_PROMPT = (
    "You are an answer evaluator. Your task is to determine if the predicted answer contains or "
    "conveys the same information as the ground truth answer.\n"
    "If the answer sentence is incomplete or partial, just response with 'no'\n"
    "The predicted answer may be:\n"
    "- An exact match (e.g., 'Paris' vs 'Paris')\n"
    "- A complete sentence containing the answer (e.g., 'The capital is Paris.' vs 'Paris')\n"
    "- Semantically equivalent (e.g., '3' vs 'three')\n"
    "- Ignore mathematical units (e.g., '180°' vs '180')\n"
    "- Equivalent mathematical forms (e.g., 'y=2x-6.5' vs '4x-2y-13=0', '2.0' vs '2', '26000' vs "
    "'26,000')\n"
    "\n"
    "Ground Truth: {}\n"
    "Predicted Answer: {}\n"
    "\n"
    "Does the predicted answer correctly contain or express the ground truth? Answer with ONLY "
    "'yes' or 'no'.\n"
)


def rule_score(task: dict, prediction: str) -> int:
    """Match the public evaluator's rules on the whole visible response."""
    prediction = prediction.strip().lower()
    answer = str(task["answer"]).strip().lower()
    options = {key: str(value).strip().lower() for key, value in task["options"].items() if key in "ABCDEFG"}
    if options or (len(answer) == 1 and answer in "abcd"):
        first = prediction[:1]
        if len(answer) == 1 and answer in "abcdefg":
            return int(first == answer and (len(prediction) == 1 or prediction[1] in ". -"))
        if first and first in "abcdefg":
            return int(options.get(first.upper()) == answer)
        return int(any(option in prediction and option == answer for option in options.values()))
    pred_normalized = " ".join(prediction.split())
    answer_normalized = " ".join(answer.split())
    return int(
        pred_normalized == answer_normalized
        or (len(answer_normalized.split()) < 5 and answer_normalized in pred_normalized)
    )


def judge_prompt(task: dict, prediction: str) -> str:
    """Use the public MCQ extraction or free-answer equivalence prompt."""
    options = task["options"]
    if options:
        options_text = "".join(f"{key}. {options[key]} " for key in "ABCDEFG" if key in options)
        return MCQ_PROMPT.format(task["question"], options_text, prediction)
    return VQA_PROMPT.format(task["answer"], prediction)


def judge_score(task: dict, verdict: str) -> int | None:
    """Return None for an invalid verdict, preserving the public parsing rules."""
    if "Failed to obtain answer via API." in verdict:
        return None
    if task["options"]:
        if not verdict.isupper() or not verdict.isalpha():
            return None
        reference = "".join(sorted(str(task["answer"]).lower().split(", ")))
        return int("".join(sorted(verdict.lower())) == reference)
    verdict = verdict.strip().lower()
    return int(verdict == "yes") if verdict in ("yes", "no") else None
