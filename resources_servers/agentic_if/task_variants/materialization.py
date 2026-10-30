# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task input of a sealed row, as Gym's episode routing carries it."""

from collections.abc import Mapping


def sealed_task_input(task: Mapping[str, object]) -> object:
    """Return a materialized task's sealed content.

    Gym collation of a flat row (``nemo_gym.task_materialization.materialize_task``) takes the row's
    ``task_id`` as the task identity and keeps that field inside ``task_input``. The sealed content, and
    therefore the content digest, is everything else.
    """
    content = task.get("task_input")
    if isinstance(content, Mapping) and "task_id" in content:
        identity = task.get("task_id")
        if not isinstance(identity, Mapping) or content["task_id"] != identity.get("task_id"):
            raise ValueError("materialized task identity receipt does not match its content")
        return {key: value for key, value in content.items() if key != "task_id"}
    return content
