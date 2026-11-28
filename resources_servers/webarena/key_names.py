# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Key-name normalization of the real WebArena eval harness, for the ``key_press`` action.

The harness (``osworld_internal/webarena/nvidia/nemotron_toolcall_agent.py``,
``_normalize_key``) lower-cases every key name and maps a few aliases before calling
pyautogui. Without it, pyautogui on X11 silently drops ``cmd``, ``command``,
``control`` and ``option`` (no X11 binding), so ``["cmd", "t"]`` types a plain "t"
in training while the eval opens a new tab; and ``["ctrl", "A"]`` becomes
Ctrl+Shift+A instead of select-all. Our checkpoints pressed ``cmd`` 77 times in
6,917 held-out episodes (cmd+t 38, cmd+left 16, cmd+a 9).
"""

from __future__ import annotations

from collections.abc import Iterable


HARNESS_KEY_ALIASES: dict[str, str] = {
    "cmd": "ctrl",
    "command": "ctrl",
    "control": "ctrl",
    "return": "enter",
    "escape": "esc",
    "option": "alt",
}


def normalize_key_names(keys: Iterable[object]) -> list[str]:
    """The harness's ``_normalize_key`` applied to each key of a ``key_press`` action."""
    return [HARNESS_KEY_ALIASES.get(str(key).lower(), str(key).lower()) for key in keys]
