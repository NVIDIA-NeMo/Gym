# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wrap flat dataset rows with environment-neutral task identity."""

from collections.abc import Mapping

from pydantic import JsonValue

from nemo_gym.episode_types import TaskId
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    SKILLS_REF_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
)


def materialize_task(
    row: Mapping[str, JsonValue], *, taskset: str, task_index: int | None = None
) -> dict[str, JsonValue]:
    """Preserve legacy task identity and task fields without embedding runtime routing.

    Rows without an explicit ID use the collector's ``_ng_task_index``, or a source-row
    index across all datasets of the same type in the collation, before repeats.
    Adding or reordering other datasets shifts these positional IDs; editing prompt text does not.
    Prepared source files are never modified by this conversion.
    """
    if "task_input" in row or isinstance(row.get("task_id"), Mapping):
        raise ValueError("Expected a flat dataset row, not an already materialized task")
    task_id = next(
        (str(row[key]) for key in ("task_id", "problem_id", "instance_id") if row.get(key) is not None),
        None,
    )
    if task_id is None:
        index = row.get(TASK_INDEX_KEY_NAME, task_index)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("A flat row without a task ID requires a non-negative task_index")
        task_id = str(index)
    excluded = {AGENT_REF_KEY_NAME, TASK_SOURCE_KEY_NAME, SKILLS_REF_KEY_NAME}
    return {
        "task_id": TaskId(taskset=taskset, task_id=task_id).model_dump(mode="json"),
        "task_input": {key: value for key, value in row.items() if key not in excluded and not key.startswith("_ng_")},
    }
