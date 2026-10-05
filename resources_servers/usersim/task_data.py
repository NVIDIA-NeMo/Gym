# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-data schema for materialized NeMo UserSim rows."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


UserSimAgentRole = Literal["user", "assistant", "judge", "summary", "tool_simulation"]


class UserSimSamplingRequest(BaseModel):
    """Dataset-owned inputs used to select one replayable scenario."""

    model_config = ConfigDict(extra="forbid")

    locale: str = Field("en_US", pattern=r"^[A-Za-z0-9_]+$")
    seed: int
    probe_type: str | None = None


class UserSimResponseCreateParams(BaseModel):
    """Dependency-light Responses API payload retained for Environment Server dispatch."""

    model_config = ConfigDict(extra="allow")


class TaskData(BaseModel):
    """Durable input loaded from one UserSim task row."""

    model_config = ConfigDict(extra="forbid")

    sampling: UserSimSamplingRequest
    probe_data: dict[str, Any] = Field(default_factory=dict)
    responses_create_params: dict[UserSimAgentRole, UserSimResponseCreateParams] = Field(default_factory=dict)
