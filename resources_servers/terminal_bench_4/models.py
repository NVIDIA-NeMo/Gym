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


class SandboxedVerifyRequest(BaseVerifyRequest):
    """Agent-side verification payload.

    The mini-SWE agent binds its own session and reports termination. The unmodified OpenCode agent posts only the
    row fields plus ``response`` (extra fields kept), so ``session_id`` and ``termination`` are optional at the wire
    and the resources server binds them from the resources session cookie before finalization.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str | None = None
    termination: AgentTermination | None = None
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
    # Optional at the wire so the server can answer a missing value with 422 in both contracts; OpenCode rows must
    # carry one distinct value per attempt (it also keys the episode owner when the agent sends no client session).
    rollout_id: str | None = Field(default=None, min_length=1, max_length=256)
    client_session_id: str | None = Field(default=None, min_length=1, max_length=256)
    artifact_directory: str | None = Field(default=None, min_length=1)


class SeedSessionResponse(SessionRequest):
    task_id: str | None = None
    # Provider sandbox id for agents that reconnect by handle (the OpenCode sandboxed agent).
    sandbox_handle: str | None = None
    sandbox_descriptor: dict[str, Any] | None = None
    sandbox_provider: dict[str, Any] = Field(default_factory=dict)
    instruction: str = ""
    user: str | int | None = None
    execution_mode: Literal["miniswe", "oracle"] = "miniswe"
    agent_timeout_sec: float = Field(default=28800, gt=0)
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    skills_dir: str | None = None
    termination: AgentTermination | None = None
    verified_response: SandboxedVerifyResponse | None = None
