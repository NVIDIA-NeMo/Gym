# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Outcome reward for synthetic GeoBenchX-style tasks.

solve:   1 when the final answer is within tolerance of a ground-truth value
         computed by replaying a reference plan through the real tools, and the
         model did not reject the task.
reject:  1 when the model's last tool call is reject_task.
control: 1 when the answer is correct and no tool was called.
"""

from __future__ import annotations

import math
import re
from typing import Any


_ANSWER = re.compile(r"answer\s*[:=]\s*\**\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)


def parse_output(output: list[Any]) -> tuple[list[str], str]:
    """Tool names in call order and the text of the final assistant message."""
    calls, text = [], ""
    for item in output:
        item = item if isinstance(item, dict) else item.model_dump()
        if item.get("type") == "function_call":
            calls.append(item.get("name", ""))
        elif item.get("type") == "message" and item.get("role") == "assistant":
            parts = item.get("content") or []
            text = (
                "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                if isinstance(parts, list)
                else str(parts)
            )
    return calls, text


def parse_answer(text: str) -> float | None:
    matches = _ANSWER.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def close_enough(value: float, targets: list[float], rel_tol: float, abs_tol: float) -> bool:
    return any(math.isclose(value, t, rel_tol=rel_tol, abs_tol=abs_tol) for t in targets)


def score(task: dict[str, Any], output: list[Any]) -> dict[str, Any]:
    calls, text = parse_output(output)
    answer = parse_answer(text)
    rejected = "reject_task" in calls
    kind = task["task_kind"]
    if kind == "reject":
        correct = bool(calls) and calls[-1] == "reject_task"
    elif kind == "control":
        correct = (
            not calls
            and answer is not None
            and close_enough(answer, task["answers"], task["rel_tol"], task["abs_tol"])
        )
    else:
        correct = (
            not rejected
            and answer is not None
            and close_enough(answer, task["answers"], task["rel_tol"], task["abs_tol"])
        )
    return {
        "reward": 1.0 if correct else 0.0,
        "answer": answer,
        "rejected": rejected,
        "tool_calls": len(calls),
    }
