# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Code-extraction helpers adapted from OSWorld
# (https://github.com/xlang-ai/OSWorld, Apache-2.0).
"""Parse pyautogui actions out of model output.

OSWorld's `pyautogui` action space expects each step to either be a block of
runnable pyautogui code or one of the special control tokens ``WAIT``,
``DONE``, ``FAIL``. We accept the same shape so model outputs stay
interchangeable between this wrapper and the upstream `PromptAgent`.
"""

from __future__ import annotations

import re
from typing import List


_SPECIAL = {"WAIT", "DONE", "FAIL"}

_THINK_PATTERN = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE_PATTERN = re.compile(r"```(?:\w+\s+)?(.*?)```", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove ``<think>``/``<thinking>`` blocks emitted by reasoning models."""
    return _THINK_PATTERN.sub("", text or "").strip()


def parse_actions(model_output: str) -> List[str]:
    """Extract a list of pyautogui code blocks (or WAIT/DONE/FAIL tokens).

    Empty list means the model produced no usable action — caller should
    treat that as a no-op step rather than crashing.
    """
    if not model_output:
        return []

    text = strip_thinking(model_output)

    stripped = text.strip()
    if stripped in _SPECIAL:
        return [stripped]

    matches = _CODE_FENCE_PATTERN.findall(text)
    actions: List[str] = []
    for match in matches:
        # Preserve the complete fence body. Python already handles
        # semicolon-separated statements, and comments may contain semicolons.
        block = match.strip()
        if not block:
            continue
        if block in _SPECIAL:
            actions.append(block)
            continue
        last_line = block.split("\n")[-1].strip()
        if last_line in _SPECIAL and len(block.split("\n")) > 1:
            # `code\nDONE` — execute the code then signal DONE.
            actions.append("\n".join(block.split("\n")[:-1]))
            actions.append(last_line)
        else:
            actions.append(block)

    return actions
