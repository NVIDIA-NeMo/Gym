# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-owned views of the legacy seed/verify wire protocol.

Resources servers validate their own models at the HTTP boundary. Keep this
adapter local until Gym's shared agent/task session contracts are available.
"""

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from responses_api_agents.miniswe_sandboxed_agent.harness import HarnessOutcome


class MiniSWERunRequest(BaseRunRequest):
    """Forward task-specific fields to resources without interpreting them."""

    model_config = ConfigDict(extra="allow")


class MiniSWEVerifyResponse(BaseVerifyResponse):
    """Preserve resource-specific results without requiring benchmark fields."""

    model_config = ConfigDict(extra="allow")


class SeedSessionResponse(BaseModel):
    """Execution inputs supplied by a resource-owned session."""

    session_id: str
    task_id: str | None = None
    sandbox_descriptor: dict[str, JsonValue] | None = None
    sandbox_provider: dict[str, JsonValue] = Field(default_factory=dict)
    instruction: str = ""
    workdir: str | None = None
    user: str | int | None = None
    agent_timeout_sec: float = Field(default=28800, gt=0)
    mcp_servers: list[dict[str, JsonValue]] = Field(default_factory=list)
    skills_dir: str | None = None
    termination: HarnessOutcome | None = None
    verified_response: MiniSWEVerifyResponse | None = None


class AgentExecutionResult(BaseVerifyRequest):
    """Agent output, independent of resource verification and reward semantics."""

    termination: HarnessOutcome
    agent_started: bool = False
    agent_timings: dict[str, dict[str, str]] = Field(default_factory=dict)
    harness_metadata: dict[str, JsonValue] = Field(default_factory=dict)


class SandboxedVerifyRequest(AgentExecutionResult):
    """Bind an execution result to its resource-owned session for verification."""

    session_id: str
