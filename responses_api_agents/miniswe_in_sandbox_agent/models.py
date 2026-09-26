# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-owned views of the resources seed/verify wire protocol (TB4 split contract)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse


class RunRequest(BaseRunRequest):
    """Forward task-specific fields to resources without interpreting them."""

    model_config = ConfigDict(extra="allow")


class VerifyResponse(BaseVerifyResponse):
    """Preserve resource-specific results without requiring benchmark fields."""

    model_config = ConfigDict(extra="allow")


class Termination(BaseModel):
    reason: Literal["completed", "timeout", "nonzero_exit", "cancelled", "infrastructure_error"]
    exit_code: int | None = None
    detail: str | None = None
    artifacts: list[str] = Field(default_factory=list)


class SeedSessionResponse(BaseModel):
    """Execution inputs supplied by a resource-owned session."""

    session_id: str
    task_id: str | None = None
    sandbox_descriptor: dict[str, JsonValue] | None = None
    sandbox_provider: dict[str, JsonValue] = Field(default_factory=dict)
    instruction: str = ""
    user: str | int | None = None
    execution_mode: str = "miniswe"
    agent_timeout_sec: float = Field(default=28800, gt=0)
    mcp_servers: list[dict[str, JsonValue]] = Field(default_factory=list)
    skills_dir: str | None = None
    termination: Termination | None = None
    verified_response: VerifyResponse | None = None


class AgentExecutionResult(BaseVerifyRequest):
    """Agent output, independent of resource verification and reward semantics."""

    termination: Termination
    agent_started: bool = False
    agent_timings: dict[str, dict[str, str]] = Field(default_factory=dict)
    harness_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SandboxedVerifyRequest(AgentExecutionResult):
    """Bind an execution result to its resource-owned session for verification."""

    session_id: str
