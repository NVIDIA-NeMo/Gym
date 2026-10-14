# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarise a gym dataset validate run from its rollout and failure files.

Reads the rows a validation run wrote, keys them by task, reports which tasks passed, which
scored below one and which were never run, and returns the text and the overall verdict as a
ValidationReport. The task digest lookup the resume checks share lives here as well. cli.py
imports it for validate_target and the eval run resume conflict check.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_gym.path_utils import failures_path_for


@dataclass(frozen=True)
class ValidationReport:
    text: str
    ok: bool


def _dig(mapping: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(mapping, dict):
            return None
        mapping = mapping.get(key)
    return mapping


def _task_name(row: dict[str, Any]) -> str:
    identity = row.get("_ng_task_id") or row.get("task_id") or {}
    return str(identity.get("task_id", "?")) if isinstance(identity, dict) else str(identity)


def _validation_key(row: dict[str, object]) -> tuple[str, str, object, object]:
    identity = row.get("_ng_task_id") or row.get("task_id") or {}
    taskset = str(identity.get("taskset", "")) if isinstance(identity, dict) else ""
    return taskset, _task_name(row), row.get("_ng_task_index"), row.get("_ng_rollout_index")


def _validation_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_validation(
    rollouts: Path,
    unvalidated: list[str],
    *,
    expected: list[dict[str, object]] | None = None,
    skipped: dict[str, str] | None = None,
) -> ValidationReport:
    """Report every selected task/repeat, reconciling results with failed attempts.

    The collector stores verify responses directly and keeps historical failed attempts in a
    sidecar. A result supersedes failures for the same task/repeat, but never another repeat.
    ``expected`` contains the collector's selected, expanded input rows (after limit/repeats).
    """
    lines = ["task_id\tstatus\treward"]
    ok = True
    selected = {_validation_key(row): row for row in expected} if expected is not None else None
    results = {_validation_key(row): row for row in _validation_rows(rollouts)}
    failures = {_validation_key(row): row for row in _validation_rows(failures_path_for(rollouts))}
    keys = selected if selected is not None else dict.fromkeys([*results, *failures])
    for key in keys:
        _, task_id, _, repeat = key
        label = f"{task_id} [repeat {repeat}]" if repeat is not None else task_id
        if task_id in unvalidated:
            continue
        row = results.get(key)
        if row is None:
            ok = False
            failure = failures.get(key)
            if failure is None:
                lines.append(f"{label}\tmissing result\t-")
            else:
                message = failure.get("_ng_failure_message") or failure.get("_ng_failure_class") or "failed"
                lines.append(f"{label}\tfailed: {str(message)[:120]}\t-")
            continue
        oracle = _dig(row, "response", "metadata", "oracle")
        reward = row.get("reward")
        if row.get("mask_sample"):
            ok = False
            lines.append(f"{label}\tmasked ({row.get('failure_kind')})\t-")
        elif oracle == "unvalidated":
            # A selected reference solution disappearing is not a successful validation.
            if selected is not None:
                ok = False
            lines.append(f"{label}\tunvalidated (no solution/)\t-")
        else:
            # Only the oracle agent stamps `response.metadata.oracle` (``solved`` or ``failed`` here). A row
            # without it came from another producer, such as a model run that wrote to this file, and a
            # status the agent never emits is no validation either: neither may pass the task.
            if oracle not in ("solved", "failed"):
                oracle = "missing" if oracle is None else f"status {oracle!r}"
                oracle += " (not an oracle result)"
            scored = isinstance(reward, (int, float)) and not isinstance(reward, bool) and math.isfinite(reward)
            if oracle != "solved" or not scored or reward < 1.0:
                ok = False
            kind = row.get("failure_kind")
            note = f" ({kind})" if kind else ""
            lines.append(f"{label}\toracle {oracle or 'ran'}{note}\t{reward}")
    if selected is None and not rollouts.is_file():
        ok = False
        lines.append(f"-\tno rollouts written at {rollouts}\t-")
    for task_id in unvalidated:
        lines.append(f"{task_id}\tunvalidated (no solution/)\t-")
    for task_id, reason in (skipped or {}).items():
        ok = False
        lines.append(f"{task_id}\tskipped: {reason}\t-")
    return ValidationReport(text="\n".join(lines), ok=ok)
