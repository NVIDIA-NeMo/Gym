# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from extract_answer in mmrl/artifacts.py in NVIDIA's visual-coding RL research code.
import re


def extract_answer(text: str, language: str) -> str:
    """Use final answer only; never salvage an unfinished thinking/code fragment."""
    text = text.strip()
    if "<think>" in text:
        match = re.fullmatch(r"<think>[\s\S]*?</think>\s*([\s\S]*)", text)
        if not match:
            raise ValueError("Unfinished thinking")
        text = match[1].strip()
    fence = re.fullmatch(r"```([\w+-]*)\s*\n([\s\S]*?)\n\s*```", text)
    if fence:
        aliases = {
            "python": {"python", "py"},
            "svg": {"svg", "xml"},
            "html": {"html"},
            "diff": {"diff", "patch"},
            "latex": {"latex", "tex"},
        }
        if fence[1].lower() not in aliases[language]:
            raise ValueError("Wrong code fence")
        text = fence[2]
    if not text.strip() or "```" in text or "<think>" in text or "</think>" in text:
        raise ValueError("Expected one complete artifact")
    return text
