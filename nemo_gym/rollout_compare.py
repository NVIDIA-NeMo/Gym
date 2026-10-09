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
"""Per-task parity report between two rollout JSONL files (`gym dev compare`).

A temporary migration aid: before an old resources server is deleted in favour of its Harbor-path replacement,
both are run on the same tasks and this report is the gate. It joins the rows by task, classifies every task
(identical, flipped, masked, missing) and prints the losing side's verifier output for every flip, replacing the
ad-hoc audit scripts used for Terminal Bench 2.1 and Terminal Bench 4.

Rows from either path are accepted: the Harbor path carries `_ng_task_id`, legacy servers carry `task_name` or a
task id inside `verifier_metadata`. Masked infrastructure failures that the Harbor path writes to the sibling
`<name>_failures.jsonl` are folded back in as masked rows.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, TypeVar

from pydantic import Field

from nemo_gym.config_types import BaseNeMoGymCLIConfig


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
# File read under a row's `verifier_logs_dir` (the Harbor resources server writes it there).
VERIFIER_STDOUT_FILENAME = "test-stdout.txt"
FAILURES_SUFFIX = "_failures.jsonl"
# Set on rows folded in from a failures sidecar. A failure recorded before the task was known carries no task id;
# such a row is listed as masked under an `unknown#<n>` label instead of breaking key detection for the side.
SIDECAR_FLAG = "_ng_compare_sidecar"


class CompareInputError(Exception):
    """An input file could not be read or does not look like rollouts."""


class RolloutCompareConfig(BaseNeMoGymCLIConfig):
    """
    Compare two rollout JSONL files task by task: count identical, flipped, masked and missing tasks and print the
    losing side's verifier output for every flip. A report, not a gate: the exit code is 0 for a report and 1 when
    an input cannot be read or the `--json` summary cannot be written.

    Examples:

    ```bash
    # Old server vs Harbor path on the same tasks
    gym dev compare old/rollouts.jsonl new/rollouts.jsonl

    # Resolve relative `verifier_logs_dir` paths against the Harbor server directory, keep 30 log lines per flip,
    # and also write a machine-readable summary
    gym dev compare old.jsonl new.jsonl --logs-root resources_servers/harbor --tail 30 --json compare.json

    # Join on an explicit field when auto-detection picks the wrong one
    gym dev compare old.jsonl new.jsonl --key verifier_metadata.instance_id
    ```
    """

    old_rollouts: str = Field(description="Rollouts JSONL from the old (reference) side.")
    new_rollouts: str = Field(description="Rollouts JSONL from the new (candidate) side.")
    key: Optional[str] = Field(
        default=None,
        description="Dotted field holding the task id, applied to both sides. Default: detected per side from "
        + ", ".join(TASK_KEY_CANDIDATES)
        + ".",
    )
    tail: int = Field(default=15, ge=0, description="Lines of the losing side's verifier output to print per flip.")
    json_output: Optional[str] = Field(default=None, description="Also write a machine-readable summary here.")
    json_stdout: bool = Field(
        default=False,
        description="Print the JSON summary on stdout instead of the report (the root `gym --json` toggle).",
    )
    logs_root: Optional[str] = Field(
        default=None,
        description="Extra directory against which a relative `verifier_logs_dir` is resolved (tried after the "
        "current directory and each JSONL file's directory).",
    )


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


# --------------------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise CompareInputError(f"cannot read {path}: {exc.strerror or exc}") from exc
    rows: list[dict[str, Any]] = []
    # "\n" only: str.splitlines would also break a line at U+2028, U+2029 or U+0085 inside a string field.
    for lineno, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise CompareInputError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise CompareInputError(f"{path}:{lineno}: expected a JSON object per line")
        try:
            _reward_of(row)
        except CompareInputError as exc:
            raise CompareInputError(f"{path}:{lineno}: {exc} (task {_row_task_label(row)})") from exc
        rows.append(row)
    return rows


def _row_task_label(row: dict[str, Any]) -> str:
    """Best-effort task name for an error about one row, before the join key is known."""
    for candidate in TASK_KEY_CANDIDATES:
        if _lookup(row, candidate) is not None:
            return task_id_of(row, candidate, with_taskset=True)
    return "unknown"


def _failure_as_masked_row(failure: dict[str, Any]) -> dict[str, Any]:
    """A `<name>_failures.jsonl` record becomes a masked row with no reward."""
    row = dict(failure)
    row[SIDECAR_FLAG] = True
    row.setdefault("reward", None)
    row["mask_sample"] = True
    row.setdefault("failure_kind", failure.get("_ng_failure_class") or "failure")
    row.setdefault("failure_reason", failure.get("_ng_failure_message"))
    return row


def load_rollouts(path: str | Path) -> list[dict[str, Any]]:
    """Read a rollouts JSONL plus, when present, its sibling `<name>_failures.jsonl` of masked failures."""
    path = Path(path)
    rows = _read_jsonl(path)
    if path.suffix == ".jsonl":
        failures_path = path.with_name(path.stem + FAILURES_SUFFIX)
        if failures_path.is_file():
            rows.extend(_failure_as_masked_row(f) for f in _read_jsonl(failures_path))
    return rows


# --------------------------------------------------------------------------------------------------------------
# Task identity
# --------------------------------------------------------------------------------------------------------------


def _lookup(row: dict[str, Any], dotted: str) -> Any:
    value: Any = row
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _has_any_key(row: dict[str, Any]) -> bool:
    return any(_lookup(row, candidate) is not None for candidate in TASK_KEY_CANDIDATES)


def detect_task_key(rows: list[dict[str, Any]]) -> str:
    """The first candidate field every row carries. Sidecar failure rows that name no task at all are left out
    of the vote; they cannot be joined and are listed as masked under an `unknown#<n>` label."""
    voters = [row for row in rows if not (row.get(SIDECAR_FLAG) and not _has_any_key(row))]
    for candidate in TASK_KEY_CANDIDATES:
        if voters and all(_lookup(row, candidate) is not None for row in voters):
            return candidate
    raise CompareInputError(
        "no task id field found; rows carry none of " + ", ".join(TASK_KEY_CANDIDATES) + " (use --key)"
    )


def _task_identity(row: dict[str, Any], key: str) -> tuple[Optional[str], str]:
    """(taskset, task id) named by `key`; the taskset is only known for a dict `_ng_task_id`."""
    value = _lookup(row, key)
    if value is None:
        raise CompareInputError(f"row has no {key!r} field (keys: {sorted(row)[:12]})")
    if not isinstance(value, dict):
        return None, str(value)
    # The Harbor path's `_ng_task_id` is {"taskset": ..., "task_id": ...}.
    task_id = value.get("task_id", value.get("id"))
    if task_id is None:
        return None, json.dumps(value, sort_keys=True)
    taskset = value.get("taskset")
    return (None if taskset is None else str(taskset)), str(task_id)


def task_id_of(row: dict[str, Any], key: str, *, with_taskset: bool = False) -> str:
    """The label a row is joined and listed under: the plain task id, or `taskset/task_id` with `with_taskset`
    (used when a side spans several tasksets, where the same task id may recur)."""
    taskset, task_id = _task_identity(row, key)
    return f"{taskset}/{task_id}" if with_taskset and taskset else task_id


def _unkeyed_sidecar(row: dict[str, Any], key: str) -> bool:
    return bool(row.get(SIDECAR_FLAG)) and _lookup(row, key) is None


def tasksets_of(rows: list[dict[str, Any]], key: str) -> set[str]:
    identities = (_task_identity(row, key) for row in rows if not _unkeyed_sidecar(row, key))
    return {taskset for taskset, _ in identities if taskset}


def group_by_task(
    rows: list[dict[str, Any]], key: str, *, with_taskset: bool = False
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    unknown = 0
    for row in rows:
        if _unkeyed_sidecar(row, key):
            unknown += 1
            grouped.setdefault(f"unknown#{unknown}", []).append(row)
            continue
        grouped.setdefault(task_id_of(row, key, with_taskset=with_taskset), []).append(row)
    return grouped


def _common_prefix(ids: list[str]) -> Optional[str]:
    """A `<prefix>/` every id shares (e.g. `terminal-bench/`), or None."""
    prefixes = {task_id.split("/", 1)[0] + "/" for task_id in ids if "/" in task_id}
    if len(prefixes) == 1 and all("/" in task_id for task_id in ids):
        return prefixes.pop()
    return None


def _strip_prefix(grouped: dict[str, _T], prefix: str) -> dict[str, _T]:
    return {task_id[len(prefix) :]: rows for task_id, rows in grouped.items()}


def _join_sides(
    old: dict[str, _T], new: dict[str, _T]
) -> tuple[dict[str, _T], dict[str, _T], Optional[str], Optional[str]]:
    """Strip a `<prefix>/` namespace from whichever side carries one when that is what keeps the ids apart:
    legacy servers often namespace the task (`terminal-bench/x`) where the Harbor path does not."""
    old_prefix = new_prefix = None
    if old and new and not (set(old) & set(new)):
        old_prefix, new_prefix = _common_prefix(list(old)), _common_prefix(list(new))
        if old_prefix and not new_prefix:
            old = _strip_prefix(old, old_prefix)
        elif new_prefix and not old_prefix:
            new = _strip_prefix(new, new_prefix)
        else:
            old_prefix = new_prefix = None
    return old, new, old_prefix, new_prefix


Groups = dict[str, list[dict[str, Any]]]


def _match_plain_to_keyed(
    plain: Groups, plain_key: str, keyed: Groups, keyed_key: str
) -> tuple[Groups, Groups, Optional[str], Optional[str], dict[str, list[str]]]:
    """Join a side of plain task ids to one keyed `taskset/task_id`. A keyed group is relabelled with its plain
    task id when that id lives in exactly one taskset (or in the plain side's own taskset); an id the plain side
    carries that recurs across tasksets is ambiguous: its groups keep their taskset labels and are reported.
    Returns the two sides, the prefix stripped from each, and the ambiguous ids with their taskset labels."""
    labels_by_id: dict[str, list[str]] = {}
    taskset_of_label: dict[str, Optional[str]] = {}
    for label, rows in keyed.items():
        taskset, task_id = _task_identity(rows[0], keyed_key)
        labels_by_id.setdefault(task_id, []).append(label)
        taskset_of_label[label] = taskset
    plain, labels_by_id, plain_prefix, keyed_prefix = _join_sides(plain, labels_by_id)
    matched: Groups = {}
    ambiguous: dict[str, list[str]] = {}
    for task_id, labels in labels_by_id.items():
        chosen = labels
        if task_id in plain and len(labels) > 1:
            own_taskset = _task_identity(plain[task_id][0], plain_key)[0]
            chosen = [label for label in labels if taskset_of_label[label] == own_taskset] or labels
        if len(chosen) == 1:
            matched[task_id] = keyed[chosen[0]]
            matched.update((label, keyed[label]) for label in labels if label != chosen[0])
        else:
            matched.update((label, keyed[label]) for label in labels)
            if task_id in plain:
                ambiguous[task_id] = labels
    return plain, matched, plain_prefix, keyed_prefix, ambiguous


# --------------------------------------------------------------------------------------------------------------
# Per-row signals
# --------------------------------------------------------------------------------------------------------------


def mask_reason(row: dict[str, Any]) -> Optional[str]:
    """Why a row does not count: `mask_sample`, no reward, or a NaN/infinite reward.

    A `failure_kind` alone does not mask: with `mask_sample` false it labels a measured reward (e.g. the Harbor
    server's scored zero for a verifier timeout), which counts like any other reward; see `measured_failure_kind`.
    """
    if row.get("mask_sample"):
        kind = row.get("failure_kind")
        return f"mask_sample (failure_kind={kind})" if kind else "mask_sample"
    reward = row.get("reward")
    if reward is None:
        return "no reward"
    if isinstance(reward, float) and not math.isfinite(reward):
        return f"non-finite reward ({reward!r})"
    return None


def measured_failure_kind(row: dict[str, Any]) -> Optional[str]:
    """The `failure_kind` of a row that still counts (not masked): a scored failure the report labels but does
    not set aside, so a flip it causes is a real flip."""
    kind = row.get("failure_kind")
    return str(kind) if kind and not mask_reason(row) else None


def agent_error(row: dict[str, Any]) -> Optional[str]:
    """An agent-level error the row reports without being masked: the legacy agent's `error` field or an
    `*error_type` entry in the response metadata (e.g. Terminus's `terminus2_error_type`)."""
    error = row.get("error")
    if error:
        last_line = str(error).strip().splitlines()[-1] if str(error).strip() else str(error)
        return f"error: {last_line[:120]}"
    metadata = (row.get("response") or {}).get("metadata") or {}
    if isinstance(metadata, dict):
        for name, value in metadata.items():
            if name.endswith("error_type") and value:
                return f"{name}={value}"
    return None


def _reward_of(row: dict[str, Any]) -> Optional[float]:
    """The row's reward as a float (None when absent; NaN/inf pass through for `mask_reason` to catch). Booleans
    and strings are rejected rather than coerced: a `"1.0"` string means the producer is broken."""
    reward = row.get("reward")
    if reward is None:
        return None
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise CompareInputError(f"reward {reward!r} is not a number")
    return float(reward)


def _finite_reward_of(row: dict[str, Any]) -> Optional[float]:
    reward = _reward_of(row)
    return reward if reward is not None and math.isfinite(reward) else None


def summarize_side(rows: list[dict[str, Any]]) -> TaskSide:
    reasons = [reason for reason in (mask_reason(row) for row in rows) if reason]
    rewards = [reward for reward in (_finite_reward_of(row) for row in rows) if reward is not None]
    errors = [err for err in (agent_error(row) for row in rows) if err]
    kinds = [kind for kind in (measured_failure_kind(row) for row in rows) if kind]
    masked = None
    if reasons:
        masked = reasons[0] if len(rows) == 1 else f"{len(reasons)}/{len(rows)} rows masked: {reasons[0]}"
    return TaskSide(
        rows=rows,
        reward=statistics.fmean(rewards) if rewards else None,
        masked=masked,
        agent_error=errors[0] if errors else None,
        reward_rows=len(rewards),
        failure_kind=kinds[0] if kinds else None,
    )


# --------------------------------------------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------------------------------------------


def compare_rollouts(
    old_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]], key: Optional[str] = None
) -> CompareResult:
    if not old_rows and not new_rows:
        raise CompareInputError("no rows in either input")
    if key:
        old_key = new_key = key
    else:
        # An empty side (everything missing) borrows the other side's detected key.
        old_key = detect_task_key(old_rows) if old_rows else detect_task_key(new_rows)
        new_key = detect_task_key(new_rows) if new_rows else old_key
    # A side spanning several tasksets may repeat a task id, so on that side the taskset becomes part of the key
    # and heading. A side with no taskset (legacy rows) or a single one keeps plain ids; against a keyed side it
    # joins on the task id where that is unambiguous.
    old_tasksets, new_tasksets = tasksets_of(old_rows, old_key), tasksets_of(new_rows, new_key)
    old_keyed, new_keyed = len(old_tasksets) > 1, len(new_tasksets) > 1
    old = group_by_task(old_rows, old_key, with_taskset=old_keyed)
    new = group_by_task(new_rows, new_key, with_taskset=new_keyed)
    ambiguous: dict[str, list[str]] = {}
    if old_keyed and not new_keyed:
        new, old, new_prefix, old_prefix, ambiguous = _match_plain_to_keyed(new, new_key, old, old_key)
    elif new_keyed and not old_keyed:
        old, new, old_prefix, new_prefix, ambiguous = _match_plain_to_keyed(old, old_key, new, new_key)
    else:
        old, new, old_prefix, new_prefix = _join_sides(old, new)

    tasks: list[TaskComparison] = []
    repeated: list[str] = []
    for task in sorted(set(old) | set(new)):
        old_side = summarize_side(old[task]) if task in old else None
        new_side = summarize_side(new[task]) if task in new else None
        if (old_side and old_side.repeated) or (new_side and new_side.repeated):
            repeated.append(task)
        if old_side is None or new_side is None:
            # A sidecar failure that named no task has nothing to be missing from: it is a masked row.
            one_sided_mask = task.startswith("unknown#") and (old_side or new_side).masked
            tasks.append(TaskComparison(task, "masked" if one_sided_mask else "missing", old_side, new_side))
            continue
        if old_side.masked or new_side.masked:
            tasks.append(TaskComparison(task, "masked", old_side, new_side))
            continue
        assert old_side.reward is not None and new_side.reward is not None  # unmasked sides carry rewards
        if math.isclose(old_side.reward, new_side.reward, rel_tol=0.0, abs_tol=1e-9):
            tasks.append(TaskComparison(task, "identical", old_side, new_side))
            continue
        winner: Side = "old" if old_side.reward > new_side.reward else "new"
        category: Category = "flipped_agent_error" if (old_side.agent_error or new_side.agent_error) else "flipped"
        tasks.append(TaskComparison(task, category, old_side, new_side, winner))

    notes = []
    if old_keyed and new_keyed:
        notes.append(
            "both sides span several tasksets ("
            + ", ".join(sorted(old_tasksets | new_tasksets))
            + "); tasks are keyed and listed as taskset/task_id"
        )
    elif old_keyed or new_keyed:
        side, tasksets = ("old", old_tasksets) if old_keyed else ("new", new_tasksets)
        notes.append(
            f"the {side} side spans several tasksets (" + ", ".join(sorted(tasksets)) + "); its tasks are matched "
            "to the other side's plain ids on task_id"
        )
        if ambiguous:
            notes.append(
                f"{len(ambiguous)} task id(s) recur across tasksets on the {side} side and cannot be matched "
                "(listed as missing under taskset/task_id): "
                + ", ".join(f"{task_id} ({', '.join(labels)})" for task_id, labels in sorted(ambiguous.items()))
            )
    if repeated:
        notes.append(
            f"{len(repeated)} task(s) have several rows on a side; they are compared on the mean reward: "
            + ", ".join(repeated)
        )
    return CompareResult(
        tasks=tasks,
        old_key=old_key,
        new_key=new_key,
        old_rows=len(old_rows),
        new_rows=len(new_rows),
        old_masked_rows=sum(1 for row in old_rows if mask_reason(row)),
        new_masked_rows=sum(1 for row in new_rows if mask_reason(row)),
        old_prefix_stripped=old_prefix,
        new_prefix_stripped=new_prefix,
        keyed_by_taskset=old_keyed and new_keyed,
        notes=notes,
    )


# --------------------------------------------------------------------------------------------------------------
# Verifier output
# --------------------------------------------------------------------------------------------------------------


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


def _flip_block(task: TaskComparison, roots: list[Path], tail: int) -> list[str]:
    assert task.old is not None and task.new is not None and task.loser is not None
    loser_side = task.old if task.loser == "old" else task.new
    lines = [f"### {task.task}: old={task.old.reward_text()} new={task.new.reward_text()} -> {task.winner} wins"]
    for side_name, side in (("old", task.old), ("new", task.new)):
        if side.agent_error:
            lines.append(f"    {side_name} agent error: {side.agent_error}")
        if side.failure_kind:
            lines.append(f"    {side_name} scored with failure_kind={side.failure_kind} (not masked)")
    text, source = verifier_output(_losing_row(loser_side), roots)
    if text is None:
        lines.append(f"    {task.loser} verifier output {source}")
    else:
        lines.append(f"    {task.loser} verifier output (last {tail} lines of {source}):")
        lines.extend(f"    | {line}" for line in tail_lines(text, tail))
    return lines


def render_report(result: CompareResult, roots: list[Path], tail: int = 15) -> str:
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
        assert task.old is not None and task.new is not None
        reasons = [f"{name} {side.masked}" for name, side in (("old", task.old), ("new", task.new)) if side.masked]
        lines.append(f"- {task.task}: old={task.old.reward_text()} new={task.new.reward_text()}; {'; '.join(reasons)}")

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


# --------------------------------------------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------------------------------------------


def run_compare(config: RolloutCompareConfig, out=None) -> int:
    """Load both sides, print the report (or the JSON summary with `json_stdout`) and write the JSON file when
    asked. Exit code: 0 for a report, 1 for unreadable input or unwritable output."""
    out = out or sys.stdout
    try:
        old_rows = load_rollouts(config.old_rollouts)
        new_rows = load_rollouts(config.new_rollouts)
        result = compare_rollouts(old_rows, new_rows, key=config.key)
    except CompareInputError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    roots = [Path.cwd(), Path(config.new_rollouts).parent.resolve(), Path(config.old_rollouts).parent.resolve()]
    if config.logs_root:
        roots.append(Path(config.logs_root))
    summary = json.dumps(summary_dict(result), indent=2) + "\n"
    if config.json_output:
        # Written before the report so a bad path fails with the one error line and nothing half-printed.
        try:
            Path(config.json_output).write_text(summary)
        except OSError as exc:
            print(f"error: cannot write {config.json_output}: {exc.strerror or exc}", file=sys.stderr)
            return 1
    if config.json_stdout:
        out.write(summary)
    else:
        out.write(render_report(result, roots, tail=config.tail))
    if config.json_output:
        # Keeps stdout machine-readable under `--json`: the note goes to stderr there.
        print(f"\nwrote {config.json_output}", file=sys.stderr if config.json_stdout else out)
    return 0
