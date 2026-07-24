# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed request, result, and trace models for policy/tool generation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse


PolicyToolProfile = Literal["general", "proactive"]
ModelRole = Literal["policy", "judge"]
CallPhase = Literal[
    "policy_v1",
    "tools_v1",
    "policy_refine",
    "tools_refine",
    "cohesion_judge",
    "golden_judge",
]


class PolicyToolDomain(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    applications: list[Any] = Field(default_factory=list)


class PolicyToolGenerationRunRequest(BaseRunRequest):
    """One domain and one prompt profile per Gym rollout."""

    model_config = ConfigDict(extra="allow")

    profile: PolicyToolProfile
    domain: PolicyToolDomain


class TraceMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class ModelCallTrace(BaseModel):
    role: ModelRole
    phase: CallPhase
    attempt: int = Field(ge=1)
    ordinal: int = Field(default=0, ge=0)
    messages: list[TraceMessage]
    response: dict[str, Any]
    parsed: Any = None
    generated_target_index: int | None = None


class AttemptTrace(BaseModel):
    attempt: int = Field(ge=1)
    timestamp: str | None = None
    policy_tool_reference_order: list[int] = Field(default_factory=list)
    policy_reference_order: list[int] = Field(default_factory=list)
    unused_tools_reference_order: list[int] = Field(default_factory=list)
    golden_reference_order: list[int] = Field(default_factory=list)
    calls: list[ModelCallTrace] = Field(default_factory=list)
    tool_validation_passed: bool | None = None
    cohesion_failure_fraction: float | None = None
    golden_failure_fraction: float | None = None
    accepted: bool = False
    failure_stage: str | None = None
    failure_detail: str | None = None


class PolicyToolGenerationTrace(BaseModel):
    profile: PolicyToolProfile
    domain_name: str
    max_attempts: int = Field(ge=1)
    attempts: list[AttemptTrace] = Field(default_factory=list)


class PolicyToolGenerationResult(BaseModel):
    accepted: Literal[True] = True
    profile: PolicyToolProfile
    domain: PolicyToolDomain
    attempt_count: int = Field(ge=1)
    policy_md: str
    tools: list[dict[str, Any]]
    tools_jsonl: str


class PolicyToolGenerationVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    result: PolicyToolGenerationResult
    generation_trace: PolicyToolGenerationTrace
