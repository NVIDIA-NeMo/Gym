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

import re

from resources_servers.opencode_multimodal_benchmarks.mcq import infer_option
from resources_servers.opencode_multimodal_benchmarks.task_data import reference_options


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


def final_answer_text(prediction: str) -> str:
    """Use the last explicit Answer line, or the final nonempty line."""
    lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    for line in reversed(lines):
        match = re.fullmatch(r"(?:final\s+)?answer\s*:\s*(.*)", line, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return lines[-1] if lines else ""


def rule_score(task: dict, prediction: str) -> int:
    """Grade the selected final answer; ambiguous prose requires the judge."""
    prediction = final_answer_text(prediction)
    answer = str(task["answer"]).strip()
    if not answer:
        raise ValueError("VTCBench requires a nonempty reference answer")
    options = {key: str(value).strip() for key, value in task["options"].items() if key in "ABCDEFG"}
    if options:
        reference = reference_options(answer, options)
        selected = infer_option(prediction, options)
        if len(reference) == 1:
            return int(selected == reference)
        if re.fullmatch(r"[A-Ga-g](?:\s*,\s*[A-Ga-g])+", prediction):
            letters = [letter.strip().upper() for letter in prediction.split(",")]
            return int("".join(sorted(letters)) == reference)
        return 0
    return int(" ".join(prediction.casefold().split()) == " ".join(answer.casefold().split()))


def judge_prompt(task: dict, prediction: str) -> str:
    """Use the public MCQ extraction or free-answer equivalence prompt."""
    options = task["options"]
    if options:
        options_text = "".join(f"{key}. {options[key]} " for key in "ABCDEFG" if key in options)
        return MCQ_PROMPT.format(task["question"], options_text, prediction)
    return VQA_PROMPT.format(task["answer"], prediction)


def judge_score(task: dict, verdict: str) -> int | None:
    """Return None for an invalid verdict, preserving the public parsing rules."""
    if not isinstance(verdict, str):
        return None
    verdict = verdict.strip()
    if "Failed to obtain answer via API." in verdict:
        return None
    if task["options"]:
        if not verdict.isupper() or not verdict.isalpha():
            return None
        reference = reference_options(str(task["answer"]), task["options"])
        return int("".join(sorted(verdict)) == reference)
    verdict = verdict.strip().lower()
    return int(verdict == "yes") if verdict in ("yes", "no") else None
