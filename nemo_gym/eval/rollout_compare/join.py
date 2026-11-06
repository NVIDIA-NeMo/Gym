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

"""How rows on the two sides are matched to the same task.

Key detection by vote across rows, the taskset and task id behind a Harbor row's task id,
grouping rows by task with anonymous sidecar rows kept apart, stripping a namespace prefix one
side carries and the other does not, and matching plain ids to keyed labels while reporting
the ambiguous ones.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from nemo_gym.eval.rollout_compare.schema import _T, SIDECAR_FLAG, TASK_KEY_CANDIDATES, CompareInputError


def _lookup(row: dict[str, Any], dotted: str) -> Any:
    value: Any = row
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _has_any_key(row: dict[str, Any]) -> bool:
    return any(_lookup(row, candidate) is not None for candidate in TASK_KEY_CANDIDATES)


def _names_a_task(row: dict[str, Any]) -> bool:
    """Whether a row has a say in key detection: a sidecar failure that names no task at all does not."""
    return not (row.get(SIDECAR_FLAG) and not _has_any_key(row))


def has_keyed_rows(rows: list[dict[str, Any]]) -> bool:
    """Whether a side names any task at all. A side that is empty, or holds only anonymous sidecar failures, has
    no key of its own to detect."""
    return any(_names_a_task(row) for row in rows)


def detect_task_key(rows: list[dict[str, Any]]) -> str:
    """The first candidate field every row carries. Sidecar failure rows that name no task at all are left out
    of the vote; they cannot be joined and are listed as masked under an `unknown (<side>) #<n>` label."""
    voters = [row for row in rows if _names_a_task(row)]
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
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """(rows grouped by task label, anonymous rows). An anonymous row is a sidecar failure that names no task:
    it is kept out of the join entirely and listed as masked on its own side."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    anonymous: list[dict[str, Any]] = []
    for row in rows:
        if _unkeyed_sidecar(row, key):
            anonymous.append(row)
            continue
        grouped.setdefault(task_id_of(row, key, with_taskset=with_taskset), []).append(row)
    return grouped, anonymous


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
