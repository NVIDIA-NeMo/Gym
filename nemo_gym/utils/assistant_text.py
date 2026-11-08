# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select the grading text from a Responses trajectory."""

from typing import Any


def final_assistant_text(output: list[dict[str, Any]]) -> str:
    """Return the last assistant answer, stopping at an unfinished reasoning or tool turn."""
    for item in reversed(output):
        if item.get("type") in {"reasoning", "function_call"}:
            break
        if item.get("type") == "message" and item.get("role") == "assistant":
            return "".join(part["text"] for part in item["content"] if part["type"] == "output_text")
    return ""
