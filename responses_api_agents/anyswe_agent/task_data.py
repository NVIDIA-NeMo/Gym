# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the self-contained anyswe_agent (no resources server).

Rows are prompt-only: the committed example dataset carries nothing beyond
``responses_create_params``. ``AnySweRunRequest`` (app.py) declares no task fields and is
``extra="allow"``, so any additional row keys ride through the wire unread by the agent.
"""

from pydantic import BaseModel, ConfigDict


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")
