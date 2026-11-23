# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream one side's rollout files into per-task counts.

A side is the output of one `gym eval run`: `<name>.jsonl` with the scored rows, `<name>_failures.jsonl` with the
rollouts that produced no result, and `<name>_materialized_inputs.jsonl` with one row per expected rollout. Each file
is read line by line and every row is reduced at once to the few fields the comparison needs, so trajectories are
never held in memory.

Row semantics are those of `gym dev compare` (nemo_gym.eval.rollout_compare), applied to the reduced rows: a row is
masked when `mask_sample` is set or its reward is missing or not finite; a `failure_kind` on an unmasked row labels a
measured reward and still counts; only the last failures-sidecar record per (task index, rollout index) stands, and
none whose rollout was scored after all. Unlike `gym dev compare`, which keeps every main-file row, only the last
main-file row per (task index, rollout index) stands, so a resumed run that rewrites a row is counted once.

Imports: `nemo_gym.config` and `nemo_gym.rollouts` names come from their public APIs. `nemo_gym.eval` declares no
public API of its own, so the row rules come from `nemo_gym.eval.rollout_compare`, the subpackage that re-exports
them as its public names for `gym dev compare`; importing them (not copying them) keeps the two tools' masking rules
the same. `is_non_superseding_rollout_row` is not exported by `nemo_gym.rollouts`, so its three-line rule is carried
below as `_is_non_superseding`.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from nemo_gym.config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.eval.rollout_compare import agent_error, current_failures, mask_reason, measured_failure_kind
from nemo_gym.rollouts import materialized_path_for


ROLLOUTS_FILENAME = "rollouts.jsonl"
FAILURES_SUFFIX = "_failures.jsonl"
SUMMARY_SUFFIX = "_run_summary.json"
# The Harbor path stamps `_ng_task_id` on every record, while its materialized input row carries the same value
# under `task_id`; a key on the first is looked up on the second when an input row is the only source.
RECORD_TASK_ID = "_ng_task_id"
INPUT_TASK_ID = "task_id"
KEY_FIELD = "_parity_key"
REWARD_KEYS_FIELD = "_parity_reward_keys"
_TEXT_LIMIT = 300
# Sidecar markers the rollout collector writes (nemo_gym.rollouts.rollout_collection).
_OMITTED = "_ng_omitted_rollout"
_DISPATCH_DRAINED = "_ng_dispatch_drained"
_UNROUTED_RUN_ERROR = "_ng_unrouted_run_error"
_KEPT_FIELDS = (
    "reward",
    "mask_sample",
    "failure_kind",
    "_ng_failure_class",
    _OMITTED,
    _DISPATCH_DRAINED,
    _UNROUTED_RUN_ERROR,
    TASK_INDEX_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
)

Attempt = tuple[Any, Any]


class ParityInputError(Exception):
    """An input cannot be read, a row cannot be keyed, or the two sides cannot be matched."""


@dataclass
class TaskCounts:
    """One side's rollouts of one task."""

    task_indices: set[Any] = field(default_factory=set)
    expected: set[Attempt] = field(default_factory=set)  # from the materialized inputs, when present
    seen: set[Attempt] = field(default_factory=set)
    unindexed: int = 0  # rows without a (task index, rollout index): each is one rollout
    rewards: list[float] = field(default_factory=list)  # one per scored (unmasked) rollout
    # Scored rollouts by their rollout index (the repeat), so repeat means do not depend on file order.
    rewards_by_repeat: dict[Any, float] = field(default_factory=dict)
    masked: int = 0  # rows in the main file that do not count: mask_sample, no reward, non-finite reward
    failed: int = 0  # standing failures-sidecar records: the rollout produced no result
    measured_failures: int = 0  # scored rows that carry a failure_kind (a measured reward, e.g. a verifier timeout)
    agent_errors: int = 0
    failure_kinds: Counter = field(default_factory=Counter)

    @property
    def rollouts(self) -> int:
        """Rollouts this side was meant to run: the materialized inputs, else every rollout that left a record."""
        return len(self.expected | self.seen) + self.unindexed

    @property
    def missing(self) -> int:
        """Expected rollouts that left no record at all (an interrupted or unfinished run)."""
        return len(self.expected - self.seen)

    @property
    def scored(self) -> int:
        return len(self.rewards)

    @property
    def mean(self) -> Optional[float]:
        return math.fsum(self.rewards) / len(self.rewards) if self.rewards else None


@dataclass
class SideRun:
    """One side of the comparison, read from its files."""

    label: str
    key: str
    rollouts_path: Path
    failures_path: Optional[Path]
    materialized_path: Optional[Path]
    tasks: dict[str, TaskCounts]
    reward_keys: Counter
    summary: Optional[dict[str, Any]]


def resolve_rollouts_path(path: str | Path) -> Path:
    """A rollouts JSONL itself, or a run directory that holds exactly one `rollouts.jsonl` (directly or one level
    down, e.g. `results/harbor/<taskset>/`)."""
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise ParityInputError(f"{path}: no such file or directory")
    if (path / ROLLOUTS_FILENAME).is_file():
        return path / ROLLOUTS_FILENAME
    found = sorted(path.glob(f"*/{ROLLOUTS_FILENAME}"))
    if len(found) == 1:
        return found[0]
    if not found:
        raise ParityInputError(f"{path}: no {ROLLOUTS_FILENAME} here or one level down; pass the file itself")
    raise ParityInputError(
        f"{path}: several {ROLLOUTS_FILENAME} one level down ({', '.join(str(p) for p in found)}); pass one"
    )


def lookup(row: dict[str, Any], dotted: str) -> Any:
    """The value at a dotted path (`verifier_metadata.instance_id`), or None."""
    value: Any = row
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _input_lookup(row: dict[str, Any], key: str) -> Any:
    value = lookup(row, key)
    if value is None and (key == RECORD_TASK_ID or key.startswith(RECORD_TASK_ID + ".")):
        value = lookup(row, INPUT_TASK_ID + key[len(RECORD_TASK_ID) :])
    return value


def key_text(value: Any) -> Optional[str]:
    """The task label a key value stands for. A `{"taskset", "task_id"}` dict reads as `taskset/task_id`."""
    if value is None:
        return None
    if isinstance(value, dict):
        if "task_id" in value:
            return f"{value['taskset']}/{value['task_id']}" if value.get("taskset") else str(value["task_id"])
        return json.dumps(value, sort_keys=True)
    return str(value)


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """(line number, row) for each non-blank line, read one line at a time."""
    try:
        handle = path.open(encoding="utf-8", errors="replace", newline="\n")
    except OSError as exc:
        raise ParityInputError(f"cannot read {path}: {exc.strerror or exc}") from exc
    with handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ParityInputError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise ParityInputError(f"{path}:{lineno}: expected a JSON object per line")
            yield lineno, row


def _check_reward(row: dict[str, Any], where: str) -> None:
    reward = row.get("reward")
    if reward is not None and (isinstance(reward, bool) or not isinstance(reward, (int, float))):
        raise ParityInputError(f"{where}: reward {reward!r} is not a number")


def _short(value: Any) -> Any:
    return value[-_TEXT_LIMIT:] if isinstance(value, str) else value


def reduce_row(row: dict[str, Any], key: str) -> dict[str, Any]:
    """The fields of a rollout row the comparison reads; everything else (trajectories, responses) is dropped."""
    reduced = {name: row[name] for name in _KEPT_FIELDS if name in row}
    reduced[KEY_FIELD] = key_text(lookup(row, key))
    if row.get("error"):
        reduced["error"] = _short(str(row["error"]).strip())
    metadata = (row.get("response") or {}).get("metadata") if isinstance(row.get("response"), dict) else None
    if isinstance(metadata, dict):
        errors = {name: value for name, value in metadata.items() if name.endswith("error_type") and value}
        if errors:
            reduced["response"] = {"metadata": errors}
    rewards = row.get("verifier_rewards")
    if isinstance(rewards, dict):
        reduced[REWARD_KEYS_FIELD] = sorted(str(name) for name in rewards)
    return reduced


def _attempt(row: dict[str, Any]) -> Optional[Attempt]:
    if TASK_INDEX_KEY_NAME in row and ROLLOUT_INDEX_KEY_NAME in row:
        return row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]
    return None


def _is_non_superseding(row: dict[str, Any]) -> bool:
    """A sidecar record readers skip before picking a rollout's latest attempt: a rollout never started, or a failed
    /run kept only as evidence. The same rule as `is_non_superseding_rollout_row` in the rollout collector."""
    omitted = bool(row.get(_OMITTED))
    return omitted and (bool(row.get(_DISPATCH_DRAINED)) or row.get(_UNROUTED_RUN_ERROR) is True)


def _read_materialized(path: Path, key: str) -> tuple[dict[Any, Optional[str]], dict[Any, set[Attempt]]]:
    """(task index -> key label, task index -> expected attempts) from the materialized inputs."""
    labels: dict[Any, Optional[str]] = {}
    expected: dict[Any, set[Attempt]] = {}
    for _, row in iter_jsonl(path):
        attempt = _attempt(row)
        if attempt is None:
            continue
        label = key_text(_input_lookup(row, key))
        if label is not None or attempt[0] not in labels:
            labels[attempt[0]] = label
        expected.setdefault(attempt[0], set()).add(attempt)
    return labels, expected


def _label_of(row: dict[str, Any], key: str, labels: dict[Any, Optional[str]], where: str) -> str:
    label = row.get(KEY_FIELD)
    if label is None and row.get(TASK_INDEX_KEY_NAME) in labels:
        label = labels[row[TASK_INDEX_KEY_NAME]]
    if label is None:
        index = row.get(TASK_INDEX_KEY_NAME)
        hint = f" nor on its materialized input row (task index {index})" if index is not None else ""
        raise ParityInputError(f"{where}: row has no {key!r} field{hint}")
    return label


def read_side(
    path: str | Path,
    key: str,
    *,
    label: str,
    id_map: Optional[dict[str, str]] = None,
) -> SideRun:
    """Read one side. `key` is a dotted field naming the task on the row, or on the row's materialized input when
    the row lacks it. `id_map` renames this side's task labels (to the other side's ids) before matching."""
    rollouts_path = resolve_rollouts_path(path)
    failures_path = rollouts_path.with_name(rollouts_path.stem + FAILURES_SUFFIX)
    materialized_path = materialized_path_for(rollouts_path)
    summary_path = rollouts_path.with_name(rollouts_path.stem + SUMMARY_SUFFIX)

    labels: dict[Any, Optional[str]] = {}
    expected: dict[Any, set[Attempt]] = {}
    if materialized_path.is_file():
        labels, expected = _read_materialized(materialized_path, key)

    # Main file: the last row per attempt stands (a resumed run may rewrite one).
    scored: dict[Attempt, dict[str, Any]] = {}
    unindexed: list[dict[str, Any]] = []
    for lineno, row in iter_jsonl(rollouts_path):
        where = f"{rollouts_path}:{lineno}"
        _check_reward(row, where)
        reduced = reduce_row(row, key)
        reduced[KEY_FIELD] = _label_of(reduced, key, labels, where)
        attempt = _attempt(reduced)
        if attempt is None:
            unindexed.append(reduced)
        else:
            scored[attempt] = reduced

    failures: list[dict[str, Any]] = []
    if failures_path.is_file():
        attempts = []
        for lineno, row in iter_jsonl(failures_path):
            if _is_non_superseding(row):
                continue
            reduced = reduce_row(row, key)
            reduced["_where"] = f"{failures_path}:{lineno}"
            attempts.append(reduced)
        # An omitted-rollout record still supersedes older failures but leaves its rollout missing.
        failures = [
            record for record in current_failures(attempts, [*scored.values(), *unindexed]) if not record.get(_OMITTED)
        ]
        for record in failures:
            record[KEY_FIELD] = _label_of(record, key, labels, record.pop("_where"))

    tasks: dict[str, TaskCounts] = {}
    reward_keys: Counter = Counter()

    def counts_for(raw_label: str) -> TaskCounts:
        return tasks.setdefault((id_map or {}).get(raw_label, raw_label), TaskCounts())

    for index, task_label in labels.items():
        if task_label is not None:
            counts = counts_for(task_label)
            counts.task_indices.add(index)
            counts.expected |= expected.get(index, set())

    def count_attempt(counts: TaskCounts, row: dict[str, Any]) -> Optional[Attempt]:
        attempt = _attempt(row)
        if attempt is None:
            counts.unindexed += 1
            # A row with a task index but no rollout index still names its dataset row for the uniqueness check.
            if TASK_INDEX_KEY_NAME in row:
                counts.task_indices.add(row[TASK_INDEX_KEY_NAME])
        else:
            counts.task_indices.add(attempt[0])
            counts.seen.add(attempt)
        return attempt

    for row in [*scored.values(), *unindexed]:
        counts = counts_for(row[KEY_FIELD])
        attempt = count_attempt(counts, row)
        if row.get(REWARD_KEYS_FIELD) is not None:
            reward_keys[",".join(row[REWARD_KEYS_FIELD]) or "(empty)"] += 1
        if agent_error(row):
            counts.agent_errors += 1
        if mask_reason(row):
            counts.masked += 1
            counts.failure_kinds[str(row.get("failure_kind") or "masked")] += 1
            continue
        counts.rewards.append(float(row["reward"]))
        if attempt is not None:
            counts.rewards_by_repeat[attempt[1]] = float(row["reward"])
        kind = measured_failure_kind(row)
        if kind:
            counts.measured_failures += 1
            counts.failure_kinds[kind] += 1

    for record in failures:
        counts = counts_for(record[KEY_FIELD])
        count_attempt(counts, record)
        counts.failed += 1
        counts.failure_kinds[str(record.get("failure_kind") or record.get("_ng_failure_class") or "failure")] += 1

    if not tasks:
        raise ParityInputError(
            f"{label}: {rollouts_path} has no rollout rows (and no materialized inputs or failure records)"
        )
    duplicated = {name: sorted(c.task_indices, key=str) for name, c in tasks.items() if len(c.task_indices) > 1}
    if duplicated:
        listed = "; ".join(f"{name}: task indices {indices}" for name, indices in sorted(duplicated.items())[:10])
        raise ParityInputError(
            f"{label}: key {key!r} is not unique, {len(duplicated)} value(s) name several dataset rows ({listed}). "
            "Pick a field that identifies one task."
        )

    summary = None
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summary = None
    return SideRun(
        label=label,
        key=key,
        rollouts_path=rollouts_path,
        failures_path=failures_path if failures_path.is_file() else None,
        materialized_path=materialized_path if materialized_path.is_file() else None,
        tasks=tasks,
        reward_keys=reward_keys,
        summary=summary if isinstance(summary, dict) else None,
    )
