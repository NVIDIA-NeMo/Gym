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

"""The vocabulary of the comparison.

The task key candidates, the embedded verifier fields, the sidecar flag, the file name
constants, the Category and Side literals, the Roots alias, the input error and the three
result dataclasses: one side of a task, one compared task and the whole result. Every other
module of the package imports from here and this module imports from none of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, TypeVar


_T = TypeVar("_T")

Category = Literal["identical", "flipped", "flipped_agent_error", "masked", "missing"]
Side = Literal["old", "new"]

# Where a row may name its task, in detection order. Dotted paths descend into nested dicts.
TASK_KEY_CANDIDATES: tuple[str, ...] = (
    "_ng_task_id",
    "task_name",
    "task_id",
    "verifier_metadata.task_id",
    "verifier_metadata.task_name",
    "verifier_metadata.instance_id",
    "verifier_metadata.id",
)
# Verifier stdout embedded in the row itself (legacy servers), in preference order.
EMBEDDED_VERIFIER_FIELDS: tuple[str, ...] = ("test_output", "verifier_stdout", "verifier_output")
# File read under a row's `verifier_logs_dir` (the Harbor environment server writes it there).
VERIFIER_STDOUT_FILENAME = "test-stdout.txt"
FAILURES_SUFFIX = "_failures.jsonl"
# Set on rows folded in from a failures sidecar. A failure recorded before the task was known carries no task id;
# such a row is listed as masked under an `unknown (<side>) #<n>` label instead of breaking key detection for
# the side, and it never joins anything on the other side.
SIDECAR_FLAG = "_ng_compare_sidecar"
# Roots a relative `verifier_logs_dir` is resolved against: one list for both sides, or one list per side (the
# losing row's own run directory first, so two runs with the same relative layout do not shadow each other).
Roots = list[Path] | dict[str, list[Path]]


class CompareInputError(Exception):
    """An input file could not be read or does not look like rollouts."""


@dataclass
class TaskSide:
    """One side's rows for a task, reduced to what the comparison needs."""

    rows: list[dict[str, Any]]
    reward: Optional[float]
    masked: Optional[str]  # why this side is masked, or None
    agent_error: Optional[str]  # an agent-level error marker (legacy `error`, `*_error_type` metadata), or None
    reward_rows: int = 0  # rows that contributed to `reward` (fewer than `rows` when some are masked)
    failure_kind: Optional[str] = None  # a `failure_kind` on a counted (unmasked) row: a scored failure, or None

    @property
    def repeated(self) -> bool:
        return len(self.rows) > 1

    def reward_text(self) -> str:
        if self.reward is None:
            return "none"
        text = str(round(self.reward, 4))
        if not self.repeated:
            return text
        if self.reward_rows != len(self.rows):
            return f"{text} (mean of {self.reward_rows} of {len(self.rows)} rows)"
        return f"{text} (mean of {len(self.rows)} rows)"


@dataclass
class TaskComparison:
    task: str
    category: Category
    old: Optional[TaskSide] = None
    new: Optional[TaskSide] = None
    winner: Optional[Side] = None  # for flips: the side with the higher reward

    @property
    def loser(self) -> Optional[Side]:
        if self.winner is None:
            return None
        return "new" if self.winner == "old" else "old"


@dataclass
class CompareResult:
    tasks: list[TaskComparison]
    old_key: str
    new_key: str
    old_rows: int
    new_rows: int
    old_masked_rows: int = 0
    new_masked_rows: int = 0
    old_prefix_stripped: Optional[str] = None
    new_prefix_stripped: Optional[str] = None
    keyed_by_taskset: bool = False  # tasks are labelled `taskset/task_id`: both sides span several tasksets
    notes: list[str] = field(default_factory=list)

    def by_category(self, category: Category) -> list[TaskComparison]:
        return [t for t in self.tasks if t.category == category]

    def counts(self) -> dict[str, int]:
        flips = self.by_category("flipped")
        masked = self.by_category("masked")
        missing = self.by_category("missing")
        return {
            "tasks": len(self.tasks),
            "identical": len(self.by_category("identical")),
            "flipped": len(flips),
            "flipped_old_win": sum(1 for t in flips if t.winner == "old"),
            "flipped_new_win": sum(1 for t in flips if t.winner == "new"),
            "flipped_agent_error": len(self.by_category("flipped_agent_error")),
            "masked": len(masked),
            "masked_old": sum(1 for t in masked if t.old is not None and t.old.masked),
            "masked_new": sum(1 for t in masked if t.new is not None and t.new.masked),
            "missing": len(missing),
            "missing_old_only": sum(1 for t in missing if t.new is None),
            "missing_new_only": sum(1 for t in missing if t.old is None),
        }
