# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select input rows by the task id each row carries."""

import json
from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from nemo_gym.config_types import ConfigError
from nemo_gym.episode_types import TaskId, is_materialized_task_row
from nemo_gym.task_materialization import TASK_ID_FIELDS


def row_task_id(row: Mapping[str, JsonValue]) -> str | None:
    """The id a row carries itself: a materialized row's TaskId, else its first of ``TASK_ID_FIELDS``."""
    if is_materialized_task_row(row):
        return TaskId.model_validate(row["task_id"]).task_id
    return next((str(row[key]) for key in TASK_ID_FIELDS if row.get(key) is not None), None)


def _canonical(row: Mapping[str, JsonValue]) -> str:
    """The row without Gym's per-rollout bookkeeping (``_ng_*``), which differs between repeats of one task."""
    return json.dumps(
        {k: v for k, v in row.items() if not k.startswith("_ng_")}, sort_keys=True, separators=(",", ":")
    )


def select_tasks(rows: Sequence[Mapping[str, JsonValue]], task_ids: Sequence[str]) -> list[int]:
    """Indices of the rows to run: the first row of each selected task, in input order.

    Each task is returned once even when the input repeats its row, so the caller's repeat count alone
    decides the number of samples. Rows are selected only by an id stored in the row, never by position.
    """
    first_index: dict[str, int] = {}
    without_id: list[int] = []
    conflicting: set[str] = set()
    for index, row in enumerate(rows):
        task_id = row_task_id(row)
        if task_id is None:
            without_id.append(index)
        elif task_id not in first_index:
            first_index[task_id] = index
        elif _canonical(row) != _canonical(rows[first_index[task_id]]):
            conflicting.add(task_id)
    if without_id:
        raise ConfigError(
            f"{len(without_id)} of {len(rows)} input rows have no task id (none of {list(TASK_ID_FIELDS)}), "
            f"e.g. rows {without_id[:5]}; tasks can only be selected by an id stored in the row."
        )
    if conflicting:
        raise ConfigError(f"Different rows share a task id: {sorted(conflicting)[:10]}")
    unknown = sorted(set(task_ids) - first_index.keys())
    if unknown:
        raise ConfigError(f"No input rows for task ids {unknown[:10]}")
    return sorted(first_index[task_id] for task_id in set(task_ids))
