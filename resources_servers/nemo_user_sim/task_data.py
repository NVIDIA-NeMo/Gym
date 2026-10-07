# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-data schema for materialized NeMo UserSim rows."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


UserSimAgentRole = Literal["user", "assistant", "judge", "summary", "tool_simulation"]


class UserSimResponseCreateParams(BaseModel):
    """Dependency-light Responses API payload retained for Environment Server dispatch."""

    model_config = ConfigDict(extra="allow")


class TaskData(BaseModel):
    """Fully resolved, provenance-pinned input loaded from one prepared task row."""

    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
    resolved_row: dict[str, Any]
    role_request_params: dict[UserSimAgentRole, UserSimResponseCreateParams] = Field(default_factory=dict)
