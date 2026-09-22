# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def preprocess_code_completion(completion: str, language: str = "python", strip_whitespace: bool = True) -> str:
    r"""Port of NeMo-Skills' nemo_skills.evaluation.evaluator.code.preprocess_code.

    Duplicated from ``resources_servers/bigcodebench/code_extraction.py``, where the
    two files currently differ only in this docstring. Keeping them identical is the
    intent — it is what stops a score difference between the two servers being an
    extractor artifact — but **nothing enforces it**: there is no shared import, no
    symlink and no test comparing the two, so an edit to either file silently breaks
    the invariant. Change both, or make them share one module.

    Behaviour:
      1. Drop everything up to and including the first ``</think>`` (model
         reasoning trace). Matching is on that closing tag alone: no ``<think>``
         opener is required, and because the tag is known to be present the
         ``return ""`` branch below is unreachable. Both quirks are inherited
         verbatim from NeMo-Skills and are left as-is to hold score parity with
         BigCodeBench.
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
