# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the blackjack server.

Gymnasium-family environment (see ``resources_servers/gymnasium/base.py``): there is no /verify —
reward (+1 win, 0 draw, -1 loss) comes from /step, driven by gymnasium_agent. The task lives
entirely out-of-row: game state is dealt from a fresh unseeded ``random.Random()`` per session at
reset(), and reset()/step() ignore the row-extras metadata dict completely. Rows carry only
framework keys (``responses_create_params`` + ``agent_ref``), so the schema is empty. The empty
shape is shared with the gymnasium base and openenv, but is redefined here rather than imported:
the gymnasium package ``__init__.py`` pulls fastapi/nemo_gym, which the dependency-light rule
forbids, and there are no fields to share.
"""

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    """No task-owned fields: every row is an identical prompt; the deal is random per session."""

    model_config = ConfigDict(extra="allow")
