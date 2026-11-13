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
"""The task ids a run selects (``task_ids`` or ``task_ids_fpath``), their digest, and the record a subset run keeps.

Nothing here writes; the run that collects the rows writes the record (``rollouts_task_selection.json``).
"""

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.config.config_types import ConfigError, ConfigPathNotFoundError


class TaskSelection(NamedTuple):
    """The ids a run selects, as strings, and where they came from: ``task_ids`` or the id file's path."""

    task_ids: list[str]
    source: str


def task_ids_digest(task_ids: Sequence[str]) -> str:
    """The sha256 of the selected ids, sorted and unique, one per line: the same set always gives the same digest."""
    return hashlib.sha256("\n".join(sorted(set(task_ids))).encode()).hexdigest()


def task_ids_as_strings(value: Any) -> Any:
    """Inline ids as strings: Hydra reads an unquoted ``7`` as an int and ``true`` as a bool. A float is refused, since
    its text (``1e3``, ``1.50``) is gone; anything that is not a list is returned for the caller to reject."""
    if value is None or isinstance(value, (str, Mapping)) or not isinstance(value, Iterable):
        return value
    ids = []
    for item in value:
        if isinstance(item, bool):
            ids.append("true" if item else "false")
        elif isinstance(item, int):
            ids.append(str(item))
        elif isinstance(item, float):
            raise ValueError(
                f"task_ids: {item!r} was read as a number, so its text is lost; quote the id or use task_ids_fpath."
            )
        else:
            ids.append(item)
    return ids


def read_task_ids_file(fpath: str) -> list[str]:
    """The ids a task id file lists, one per line; blank lines and lines starting with ``#`` are skipped."""
    path = _resolve_under_cwd_or_install(fpath)
    if not path.is_file():
        raise ConfigPathNotFoundError(f"Task id file not found: '{fpath}' (task_ids_fpath).")
    task_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not task_ids:
        raise ConfigError(f"Task id file '{fpath}' lists no task ids.")
    return task_ids


def task_selection(task_ids: Any, task_ids_fpath: str | None) -> TaskSelection | None:
    """The run's selection from its ``task_ids`` or ``task_ids_fpath`` setting; None when it selects no tasks by id."""
    if task_ids_fpath:
        return TaskSelection(read_task_ids_file(task_ids_fpath), task_ids_fpath)
    if task_ids is not None:
        return TaskSelection([str(task_id) for task_id in task_ids_as_strings(task_ids)], "task_ids")
    return None


def task_selection_record(selection: TaskSelection, rows: int) -> dict[str, Any]:
    """What ``rollouts_task_selection.json`` holds: the source, the counts, the sorted ids and their digest."""
    unique_ids = sorted(set(selection.task_ids))
    return {
        "source": selection.source,
        "tasks": len(unique_ids),
        "rows": rows,
        "task_ids_sha256": task_ids_digest(unique_ids),
        "task_ids": unique_ids,
    }


def read_task_selection_record(path: Path) -> dict[str, Any] | None:
    """A saved selection record; None when the run selected no tasks (no record). An unreadable record is a subset
    whose source and digest are unknown, never a run of every task."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        return {"source": None, "task_ids_sha256": None}
    if not isinstance(record, dict):
        return {"source": None, "task_ids_sha256": None}
    return record
