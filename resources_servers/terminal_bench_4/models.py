# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TB4 episode results and persisted verification records."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse


class SessionRequest(BaseModel):
    session_id: str


class AgentTermination(BaseModel):
    reason: Literal["completed", "timeout", "nonzero_exit", "cancelled", "infrastructure_error"]
    exit_code: int | None = None
    detail: str | None = None
    artifacts: list[str] = Field(default_factory=list)


class SandboxedVerifyRequest(BaseVerifyRequest, SessionRequest):
    termination: AgentTermination


class SandboxedVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    session_id: str
    evaluation_completed: bool
    termination: AgentTermination
    infrastructure_error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    timings: dict[str, Any] = Field(default_factory=dict)
