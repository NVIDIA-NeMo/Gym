# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert flat dataset rows into the built-in single-agent-turn task contract."""

from collections.abc import Mapping

from pydantic import JsonValue

from nemo_gym.episode_types import MaterializedTask, TaskId
from nemo_gym.global_config import TASK_INDEX_KEY_NAME
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput


def materialize_single_agent_task(
    row: Mapping[str, JsonValue], *, taskset: str, task_index: int | None = None
) -> MaterializedTask[SingleAgentTurnTaskInput]:
    """Preserve legacy task identity and task fields without embedding runtime routing.

    Rows without an explicit ID use their source index, not a hash of mutable prompt text.
    Collation supplies that index before repeating rows; the legacy adapter already has it.
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
    excluded = {"responses_create_params", "agent_ref", "task_source", "skills_ref"}
    return MaterializedTask[SingleAgentTurnTaskInput](
        task_id=TaskId(taskset=taskset, task_id=task_id),
        task_input=SingleAgentTurnTaskInput(
            responses_create_params=row["responses_create_params"],
            task_data={key: value for key, value in row.items() if key not in excluded and not key.startswith("_ng_")},
        ),
    )
