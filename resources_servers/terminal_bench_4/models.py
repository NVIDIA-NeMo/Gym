# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TB4 episode results and persisted verification records."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse


class SessionRequest(BaseModel):
    session_id: str


class AgentTermination(BaseModel):
    reason: Literal["completed", "timeout", "nonzero_exit", "killed", "cancelled", "infrastructure_error"]
    exit_code: int | None = None
    detail: str | None = None
    artifacts: list[str] = Field(default_factory=list)


class SandboxedVerifyRequest(BaseVerifyRequest, SessionRequest):
    termination: AgentTermination
    agent_started: bool = False
    agent_timings: dict[str, dict[str, str]] = Field(default_factory=dict)
    harness_metadata: dict[str, Any] = Field(default_factory=dict)


class VerifierTestCounts(BaseModel):
    """Reported test outcomes; null counts mean no usable test report was available."""

    num_tests_passed: int | None = Field(default=None, ge=0, strict=True)
    num_tests_total: int | None = Field(default=None, ge=0, strict=True)
    num_tests_failed: int | None = Field(default=None, ge=0, strict=True)
    num_tests_skipped: int | None = Field(default=None, ge=0, strict=True)


class SandboxResourceLimit(BaseModel):
    """Provider evidence that the agent sandbox exceeded its allocated resources."""

    resource: Literal["memory", "storage"]
    detail: str


class SandboxedVerifyResponse(BaseVerifyResponse, VerifierTestCounts):
    model_config = ConfigDict(extra="allow")

    session_id: str
    evaluation_completed: bool
    termination: AgentTermination
    infrastructure_error: str | None = None
    sandbox_resource_limit: SandboxResourceLimit | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    timings: dict[str, Any] = Field(default_factory=dict)
    # Fraction of the reported tests that passed, never below the official reward
    # (which it equals without usable test counts). Reported beside, never instead
    # of, the official reward; 0 like the reward when the official evaluation did not complete.
    partial_credit: float = 0.0


class TerminalBench4RunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task_name: str
    task_ref: str
    dataset_ref: str
    rollout_id: str = Field(min_length=1, max_length=256)
    client_session_id: str | None = Field(default=None, min_length=1, max_length=256)
    artifact_directory: str | None = Field(default=None, min_length=1)


class TaskIdentity(BaseModel):
    """The pinned package a task row names; Environment Server seeds carry it in ``task_data``."""

    task_name: str
    task_ref: str
    dataset_ref: str


class ResourcesSessionVerifyRequest(BaseVerifyRequest, TaskIdentity):
    """Environment Server verification: the task row plus the agent's response.

    The resources-session cookie, not a field, names the episode. The agent has already been
    closed, so the request carries no termination; it is derived from ``response.status``.
    """

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def not_a_tb4_session_request(cls, data: Any) -> Any:
        # TB4's own verification names its TB4 session; keep such bodies off this path.
        if isinstance(data, dict) and "session_id" in data:
            raise ValueError("A request with session_id is a TB4 session verification")
        return data


class SeedSessionResponse(SessionRequest):
    task_id: str | None = None
    sandbox_descriptor: dict[str, Any] | None = None
    sandbox_provider: dict[str, Any] = Field(default_factory=dict)
    instruction: str = ""
    user: str | int | None = None
    agent_timeout_sec: float = Field(default=28800, gt=0)
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    skills_dir: str | None = None
    termination: AgentTermination | None = None
    verified_response: SandboxedVerifyResponse | None = None
