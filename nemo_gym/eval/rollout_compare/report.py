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

"""Render a CompareResult as the text report and the JSON summary.

Finds the losing side's verifier output, from a file under the row's verifier logs directory
resolved against the roots or from a field embedded in the row, tails it, and lays out the
header, the category table and the flip, masked and missing sections. summary_dict is the
JSON shape the command writes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

from nemo_gym.eval.rollout_compare.rows import _finite_reward_of
from nemo_gym.eval.rollout_compare.schema import (
    EMBEDDED_VERIFIER_FIELDS,
    VERIFIER_STDOUT_FILENAME,
    CompareResult,
    Roots,
    Side,
    TaskComparison,
    TaskSide,
)


def _resolve_logs_path(logs_dir: str, roots: list[Path]) -> Optional[Path]:
    candidate = Path(logs_dir)
    candidates = [candidate] if candidate.is_absolute() else [root / candidate for root in roots]
    for path in candidates:
        if path.is_file():
            return path
        if (path / VERIFIER_STDOUT_FILENAME).is_file():
            return path / VERIFIER_STDOUT_FILENAME
    return None


def verifier_output(row: dict[str, Any], roots: list[Path]) -> tuple[Optional[str], str]:
    """(text, source) for a row's verifier stdout: the file under `verifier_logs_dir` when it exists, else the
    text embedded in the row; `(None, reason)` when neither is available."""
    logs_dir = row.get("verifier_logs_dir")
    if logs_dir:
        path = _resolve_logs_path(str(logs_dir), roots)
        if path is not None:
            try:
                return path.read_text(errors="replace"), str(path)
            except OSError as exc:
                return None, f"unavailable: cannot read {path}: {exc.strerror or exc}"
    for name in EMBEDDED_VERIFIER_FIELDS:
        text = row.get(name)
        if isinstance(text, str) and text.strip():
            return text, f"row field {name!r}"
    if logs_dir:
        return None, f"unavailable: {logs_dir} not found under {', '.join(str(r) for r in roots)}"
    return None, "unavailable: row has no verifier_logs_dir and no embedded verifier output"


def tail_lines(text: str, count: int) -> list[str]:
    lines = text.rstrip("\n").splitlines()
    return lines[-count:] if count else []


def _losing_row(side: TaskSide) -> dict[str, Any]:
    """The worst row on the losing side (meaningful when a side has repeats)."""
    return min(side.rows, key=lambda row: _finite_reward_of(row) if _finite_reward_of(row) is not None else math.inf)


# --------------------------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------------------------


def _roots_for(roots: Roots, side: Side) -> list[Path]:
    return roots[side] if isinstance(roots, dict) else roots


def _flip_block(task: TaskComparison, roots: Roots, tail: int) -> list[str]:
    assert task.old is not None and task.new is not None and task.loser is not None
    loser_side = task.old if task.loser == "old" else task.new
    lines = [f"### {task.task}: old={task.old.reward_text()} new={task.new.reward_text()} -> {task.winner} wins"]
    for side_name, side in (("old", task.old), ("new", task.new)):
        if side.agent_error:
            lines.append(f"    {side_name} agent error: {side.agent_error}")
        if side.failure_kind:
            lines.append(f"    {side_name} scored with failure_kind={side.failure_kind} (not masked)")
    # Resolved against the losing row's own run first: both runs may carry the same relative `verifier_logs_dir`.
    text, source = verifier_output(_losing_row(loser_side), _roots_for(roots, task.loser))
    if text is None:
        lines.append(f"    {task.loser} verifier output {source}")
    else:
        lines.append(f"    {task.loser} verifier output (last {tail} lines of {source}):")
        lines.extend(f"    | {line}" for line in tail_lines(text, tail))
    return lines


def render_report(result: CompareResult, roots: Roots, tail: int = 15) -> str:
    counts = result.counts()

    def side_rows(total: int, masked: int) -> str:
        return f"{total} rows" + (f" incl. {masked} masked" if masked else "")

    lines = [
        f"gym dev compare: {counts['tasks']} tasks (old {side_rows(result.old_rows, result.old_masked_rows)}, "
        f"new {side_rows(result.new_rows, result.new_masked_rows)})",
        "join key: old="
        + result.old_key
        + (f" (prefix {result.old_prefix_stripped!r} stripped)" if result.old_prefix_stripped else "")
        + ", new="
        + result.new_key
        + (f" (prefix {result.new_prefix_stripped!r} stripped)" if result.new_prefix_stripped else ""),
    ]
    lines.extend(f"note: {note}" for note in result.notes)
    lines += [
        "",
        f"{'category':<24}{'count':>6}",
        f"{'identical':<24}{counts['identical']:>6}",
        f"{'flipped':<24}{counts['flipped']:>6}   old-win {counts['flipped_old_win']}, "
        f"new-win {counts['flipped_new_win']}",
        f"{'flipped (agent error)':<24}{counts['flipped_agent_error']:>6}   rewards differ and a side reports an "
        "agent error; rerun before counting",
        f"{'masked':<24}{counts['masked']:>6}   old {counts['masked_old']}, new {counts['masked_new']}",
        f"{'missing':<24}{counts['missing']:>6}   old-only {counts['missing_old_only']}, "
        f"new-only {counts['missing_new_only']}",
    ]

    flips = result.by_category("flipped")
    lines += ["", f"## Flips ({len(flips)})"]
    for task in flips:
        lines.extend(_flip_block(task, roots, tail))
    agent_error_flips = result.by_category("flipped_agent_error")
    if agent_error_flips:
        lines += ["", f"## Flips with agent errors ({len(agent_error_flips)})"]
        for task in agent_error_flips:
            lines.extend(_flip_block(task, roots, tail))

    masked = result.by_category("masked")
    lines += ["", f"## Masked ({len(masked)})"]
    for task in masked:
        # An anonymous sidecar failure has one side only; the other is shown as absent.
        rewards = [
            f"{name}={side.reward_text() if side is not None else 'absent'}"
            for name, side in (("old", task.old), ("new", task.new))
        ]
        present = [(name, side) for name, side in (("old", task.old), ("new", task.new)) if side is not None]
        reasons = [f"{name} {side.masked}" for name, side in present if side.masked]
        lines.append(f"- {task.task}: {' '.join(rewards)}; {'; '.join(reasons)}")

    missing = result.by_category("missing")
    lines += ["", f"## Missing ({len(missing)})"]
    for task in missing:
        present = "old" if task.new is None else "new"
        side = task.old if task.new is None else task.new
        assert side is not None
        lines.append(f"- {task.task}: only on {present} side (reward {side.reward_text()})")
    return "\n".join(lines) + "\n"


def summary_dict(result: CompareResult) -> dict[str, Any]:
    def side_dict(side: Optional[TaskSide]) -> Optional[dict[str, Any]]:
        if side is None:
            return None
        return {
            "reward": side.reward,
            "rows": len(side.rows),
            "masked": side.masked,
            "agent_error": side.agent_error,
            "failure_kind": side.failure_kind,
        }

    return {
        "counts": result.counts(),
        "join": {
            "old_key": result.old_key,
            "new_key": result.new_key,
            "old_prefix_stripped": result.old_prefix_stripped,
            "new_prefix_stripped": result.new_prefix_stripped,
            "keyed_by_taskset": result.keyed_by_taskset,
        },
        "notes": result.notes,
        "tasks": [
            {
                "task": t.task,
                "category": t.category,
                "winner": t.winner,
                "old": side_dict(t.old),
                "new": side_dict(t.new),
            }
            for t in result.tasks
        ],
    }
