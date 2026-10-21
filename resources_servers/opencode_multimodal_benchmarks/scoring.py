# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted material: Apache-2.0, Copyright 2023 VLMEvalKit Authors.
# Source: https://github.com/open-compass/VLMEvalKit/blob/ebac2d9bda8fa28dc426a3b2ccc8f53c886dc0c5/vlmeval/dataset/babyvision.py
# Full license: resources_servers/opencode_multimodal_benchmarks/third-party-notices.md
"""BabyVision judge prompt and boxed-answer extraction."""

import re


# Original BabyVision judge prompt:
# https://github.com/UniPat-AI/BabyVision/blob/7f92fd4b1dc1c68b7b936a9bc09c68b4a944a55a/babyvision_eval/utils.py
# Also distributed in VLMEvalKit:
# https://github.com/open-compass/VLMEvalKit/blob/ebac2d9bda8fa28dc426a3b2ccc8f53c886dc0c5/vlmeval/dataset/babyvision.py
BABYVISION_JUDGE_PROMPT = """You are a careful and strict evaluator. You will be given:

1. **Question**
2. **Ground Truth Answer** (correct answer)
3. **Model Output** (answer from another model)

**Your goal:** Determine if the Model Output **accurately matches** the Ground Truth Answer in meaning.

* Matching means: the facts, entities, and key details are equivalent, even if phrasing differs.
* Not matching means: the Model Output is wrong, incomplete, contains extra incorrect facts, or changes the meaning.

**Process (internal reasoning):**

1. Read and understand the Question, Ground Truth Answer, and Model Output.
2. Ignore small wording differences, formatting, or synonyms.
3. If all factual content matches, conclude `1`. Otherwise, conclude `0`.

**Important:**

* Think through your decision step-by-step **internally** before responding.
* In your final output, return **only** True or False, with no extra text or explanation.

**Output format:**

True

or

False

**Input:**

Question: {question},
Ground Truth Answer: {groundtruth},
Model Output: {modeloutput}
"""


def boxed_answer(text: str) -> str | None:
    """Return the last balanced box, including nested LaTeX braces."""
    boxes = []
    for marker in re.finditer(r"\\boxed\s*\{", text):
        start = marker.end()
        end = start
        depth = 1
        while end < len(text) and depth:
            depth += int(text[end] == "{") - int(text[end] == "}")
            end += 1
        if depth == 0:
            boxes.append(text[start : end - 1])
    alternative = re.findall(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", text, re.DOTALL)
    return boxes[-1] if boxes else (alternative[-1] if alternative else None)
