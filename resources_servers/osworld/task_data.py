# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset metadata for the stateful OSWorld resources server.

The example rows mirror osworld_agent's task metadata. The agent sends the
task specification separately as ``task_config`` on /seed_session or /reset;
/verify evaluates the existing session and requires no task-owned row fields.
These metadata fields are therefore optional, not new HTTP requirements.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str | None = Field(default=None, description="OSWorld task identifier for provenance.")
    domain: str | None = Field(default=None, description="OSWorld application domain for provenance.")
    osworld_task: dict[str, Any] | None = Field(
        default=None,
        description="Canonical task specification passed unchanged by the agent to session initialization.",
    )
