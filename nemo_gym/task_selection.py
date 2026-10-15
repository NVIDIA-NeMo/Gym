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


def select_tasks(rows: Sequence[Mapping[str, JsonValue]], task_ids: Sequence[str]) -> list[int]:
    """Indices of the rows to run: the first row of each selected task, in input order.

    Each task is returned once even when the input repeats its row, so the caller's repeat count alone
    decides the number of samples. Rows are selected only by an id stored in the row, never by position.
    Every row must carry an id; only the selected tasks' rows must agree with each other, so a caller that
    selects one task per call does not re-check the whole input each time.

    For example, a collated input that already repeats each task four times (``a, a, a, a, b, b, b, b``)
    with ``task_ids=["a"]`` gives one index, ``[0]``. Run with ``num_repeats=2``, that is two rollouts
    of ``a``, not eight.
    """
    _require_task_id_in_every_row(rows)
    first_row_of_task = _first_row_of_each_task(rows)
    _require_task_ids_in_input(task_ids, first_row_of_task)
    _require_repeated_rows_to_be_identical(rows, first_row_of_task, set(task_ids))
    return sorted(first_row_of_task[task_id] for task_id in set(task_ids))


def _require_task_id_in_every_row(rows: Sequence[Mapping[str, JsonValue]]) -> None:
    rows_without_id = [index for index, row in enumerate(rows) if row_task_id(row) is None]
    if rows_without_id:
        raise ConfigError(
            f"{len(rows_without_id)} of {len(rows)} input rows have no task id (none of {list(TASK_ID_FIELDS)}), "
            f"e.g. rows {rows_without_id[:5]}; tasks can only be selected by an id stored in the row."
        )


def _first_row_of_each_task(rows: Sequence[Mapping[str, JsonValue]]) -> dict[str, int]:
    first_row_of_task: dict[str, int] = {}
    for index, row in enumerate(rows):
        first_row_of_task.setdefault(row_task_id(row), index)
    return first_row_of_task


def _require_repeated_rows_to_be_identical(
    rows: Sequence[Mapping[str, JsonValue]], first_row_of_task: Mapping[str, int], task_ids: set[str]
) -> None:
    first_row_content = {
        task_id: _content_without_bookkeeping(rows[first_row_of_task[task_id]]) for task_id in task_ids
    }
    rows_of_selected_tasks = [(row_task_id(row), row) for row in rows if row_task_id(row) in task_ids]
    tasks_with_differing_rows = {
        task_id
        for task_id, row in rows_of_selected_tasks
        if _content_without_bookkeeping(row) != first_row_content[task_id]
    }
    if tasks_with_differing_rows:
        raise ConfigError(f"Different rows share a task id: {sorted(tasks_with_differing_rows)[:10]}")


def _require_task_ids_in_input(task_ids: Sequence[str], first_row_of_task: Mapping[str, int]) -> None:
    unknown = sorted(set(task_ids) - first_row_of_task.keys())
    if unknown:
        raise ConfigError(f"No input rows for task ids {unknown[:10]}")


def _content_without_bookkeeping(row: Mapping[str, JsonValue]) -> str:
    """The row as canonical JSON without Gym's per-rollout bookkeeping (``_ng_*``), which differs between repeats."""
    return json.dumps(
        {k: v for k, v in row.items() if not k.startswith("_ng_")}, sort_keys=True, separators=(",", ":")
    )
