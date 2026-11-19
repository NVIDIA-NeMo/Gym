# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The task input single_agent_turn_legacy's episode request validates.

Its ``/run`` also takes a flat row.
It materializes the row with ``materialize_task(row, taskset=task_source)`` and validates the result as this ``TaskInput``.

Collation and ``gym env validate`` load this module without the server's dependencies (see ``nemo_gym/data/task_input.py``).
"""

from nemo_gym.server_runtime.single_agent_turn_types import SingleAgentTurnTaskInput as TaskInput


ACCEPTS_FLAT_ROWS = True

__all__ = ["ACCEPTS_FLAT_ROWS", "TaskInput"]
