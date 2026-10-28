# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted material: Apache-2.0, Copyright 2023 VLMEvalKit Authors.
# Source: https://github.com/open-compass/VLMEvalKit/blob/ebac2d9bda8fa28dc426a3b2ccc8f53c886dc0c5/vlmeval/utils/matching_util.py
# Source: https://github.com/open-compass/VLMEvalKit/blob/ebac2d9bda8fa28dc426a3b2ccc8f53c886dc0c5/vlmeval/dataset/utils/multiple_choice.py
# Full license: resources_servers/opencode_multimodal_benchmarks/third-party-notices.md
"""Reference-blind MCQ extraction; transport belongs to the Gym resource."""

import re


MCQ_SCORER = "mcq-vlmeval-llm-v2"
_VERBOSE_ANSWER = re.compile(r"(?i)(?:correct\s+)?answer\s+is\s+\**([ABCD])\**")


def visible_answer(prediction: str) -> str:
    """Exclude private reasoning, including an unfinished reasoning block, from grading."""
    answer = re.sub(r"<(think|thinking)>.*?</\1>", "", prediction, flags=re.DOTALL)
    return re.split(r"<(?:think|thinking)>", answer, maxsplit=1)[0].strip()


def infer_option(prediction: str, choices: dict[str, str]) -> str | None:
    """Extract an option letter, falling back to a short matching option text."""
    answer = str(prediction)
    failed = "Failed to obtain answer via API" in answer
    if not failed:
        rejections = (
            "Sorry, I can't help with images of people yet.",
            "I can't process this file.",
            "I'm sorry, but without the image provided",
            "Cannot determine the answer",
        )
        if any(message in answer for message in rejections):
            return "Z"
        modified = answer
        for char in ".()[],:;!*#{}":
            modified = modified.replace(char, " ")
        tokens = modified.split()
        letters = [letter for letter in choices if letter in tokens]
        if len(letters) == 1 and tokens.index(letters[0]) > len(tokens) - 5:
            return letters[0]
        if not letters and "Z" in tokens:
            return "Z"
        match = _VERBOSE_ANSWER.search(answer)
        if match and match.group(1).upper() in choices:
            return match.group(1).upper()
    if len(answer) <= 2 * sum(len(str(value)) for value in choices.values()):
        matches = [letter for letter, value in choices.items() if str(value).lower() in answer.lower()]
        if len(matches) == 1:
            return matches[0]
    return None


def matcher_prompt(task: dict, prediction: str) -> str:
    options = "There are several options: \n" + "".join(
        f"{letter}. {value}\n" for letter, value in sorted(task["options"].items())
    )
    template = BUILD_PROMPT_CN if re.search(r"[\u4e00-\u9fff]", task["question"]) else BUILD_PROMPT
    return template.format(task["question"], options, prediction)


BUILD_PROMPT = "You are an AI assistant who will help me to match an answer with several options of a single-choice question. You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer. If the meaning of all options are significantly different from the answer, output Z. Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \nExample 1: \nQuestion: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\nAnswer: a cute teddy bear\nYour output: A\nExample 2: \nQuestion: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\nAnswer: Spider\nYour output: Z\nExample 3: \nQuestion: {}?\nOptions: {}\nAnswer: {}\nYour output: "

BUILD_PROMPT_CN = "你是一个帮助我匹配答案与单选题中多个选项的 AI 助手。你会被提供：一个问题，多个选项，一个答案。你的任务是找到与答案意义最相近的选项。如果所有选项的意义都与答案显著不同，则输出 Z。你应该输出一个单个的大写字母，例如 A, B, C, D（如果它们是有效选项），或 Z。例 1:问题: 图中最主要的物体是什么?\n选项: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n答案: 一只可爱的泰迪熊\n输出: A\n例 2: \n问题: 图中最主要的物体是什么?\n选项: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n答案: 蜘蛛\n输出: Z\n例 3: \n问题: {}?\n选项: {}\n答案: {}\n输出: "
