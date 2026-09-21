# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TB4 episode results and persisted verification records."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse


class SessionRequest(BaseModel):
    session_id: str


class AgentTermination(BaseModel):
    reason: Literal["completed", "timeout", "nonzero_exit", "cancelled", "infrastructure_error"]
    exit_code: int | None = None
    detail: str | None = None
    artifacts: list[str] = Field(default_factory=list)


class SandboxedVerifyRequest(BaseVerifyRequest, SessionRequest):
    termination: AgentTermination
    agent_started: bool = False
    agent_timings: dict[str, dict[str, str]] = Field(default_factory=dict)
    harness_metadata: dict[str, Any] = Field(default_factory=dict)


class SandboxedVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    session_id: str
    evaluation_completed: bool
    termination: AgentTermination
    infrastructure_error: str | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    timings: dict[str, Any] = Field(default_factory=dict)


class TerminalBench4RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_name: str
    task_ref: str
    dataset_ref: str
    rollout_id: str = Field(min_length=1, max_length=256)
    client_session_id: str | None = Field(default=None, min_length=1, max_length=256)
    artifact_directory: str | None = Field(default=None, min_length=1)


class SeedSessionResponse(SessionRequest):
    sandbox_descriptor: dict[str, Any] | None = None
    sandbox_provider: dict[str, Any] = Field(default_factory=dict)
    instruction: str = ""
    user: str | int | None = None
    agent_timeout_sec: float = Field(default=28800, gt=0)
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    skills_dir: str | None = None
    termination: AgentTermination | None = None
    verified_response: SandboxedVerifyResponse | None = None
