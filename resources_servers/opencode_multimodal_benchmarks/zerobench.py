# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Public scoring example: MIT, Copyright (c) 2025 Jonathan Roberts.
# Source: https://github.com/jonathan-roberts1/zerobench/blob/0debbe43deddfe54f9965a6fdfd4f70a20d8ab62/README.md#code
# Full license: resources_servers/opencode_multimodal_benchmarks/third-party-notices.md
"""Deterministic scoring from the public ZeroBench evaluation example."""

import re


ANSWER_INSTRUCTION = (
    "\n\n\nLet's think step by step and give the final answer in curly braces, like this: {final answer}"
)


def score(prediction: str, answer: str) -> tuple[str, bool]:
    """Match the public last-brace extraction and case-insensitive exact comparison."""
    matches = re.findall(r"\{(.*?)\}", prediction.strip().lower())
    extracted = matches[-1] if matches else ""
    correct = extracted[: len(answer)].lower() == answer.strip().lower() and len(extracted) == len(answer.strip())
    return extracted, correct
