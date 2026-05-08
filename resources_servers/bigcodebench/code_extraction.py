# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def preprocess_code_completion(completion: str, language: str = "python", strip_whitespace: bool = True) -> str:
    r"""Port of NeMo-Skills' nemo_skills.evaluation.evaluator.code.preprocess_code.

    Skills uses this to extract a Python solution from a model rollout before
    running the bigcodebench unittest. We replicate the logic byte-for-byte so
    parity numbers between Skills and Gym aren't muddied by extractor drift.

    Behaviour:
      1. Drop everything up to and including the first ``</think>`` (model
         reasoning trace). If a ``<think>`` opens but never closes, return ``""``.
      2. Find the LAST fenced block (``\`\`\`python`` preferred, falls back to
         the generic ``\`\`\``). Strict mode: if the opener has no closer,
         return ``""``.
      3. Optional strip of surrounding whitespace.
    """
    completion = (completion or "").replace("\r", "")

    if "</think>" in completion:
        _, separator, post_thought = completion.partition("</think>")
        if separator:
            completion = post_thought
        else:
            return ""

    specific_fence = f"```{language}"
    generic_fence = "```"
    start_index = completion.rfind(specific_fence)
    fence_len = len(specific_fence)

    if start_index == -1:
        start_index = completion.rfind(generic_fence)
        fence_len = len(generic_fence)

    if start_index != -1:
        content_start = start_index + fence_len
        completion = completion[content_start:]
        end_index = completion.find(generic_fence)
        if end_index != -1:
            completion = completion[:end_index]
        else:
            completion = ""

    if strip_whitespace:
        completion = completion.strip()

    return completion
