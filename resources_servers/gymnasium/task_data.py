# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the gymnasium base server.

Abstract family base, not a standalone server: ``GymnasiumServer.verify()`` raises
NotImplementedError — reward comes from /step, driven by gymnasium_agent. Rows reach concrete
environments completely untyped as ``model_extra`` on ``EnvResetRequest``/``EnvStepRequest``
(``base.py:33/43``, ``extra="allow"``), which is why the base declares no task fields: the base
itself consumes nothing from the row, and each concrete environment (grl_sokoban, grl_tetris,
tales, openair_congestion, blackjack) documents its own reset()-consumed fields in its own
``task_data.py``. The committed ``data/example.jsonl`` is a minimal smoke row carrying only
framework keys. Heirs deliberately do NOT import this module: ``resources_servers/gymnasium/
__init__.py`` imports the server class (fastapi/nemo_gym), so a package import from another
server's task_data would violate the dependency-light rule — and there are no shared fields to
inherit anyway.
"""

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    """No task-owned fields: concrete environments define their own reset()-time fields."""

    model_config = ConfigDict(extra="allow")
