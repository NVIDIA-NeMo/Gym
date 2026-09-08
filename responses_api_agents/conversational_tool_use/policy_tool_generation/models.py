# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    source_artifacts: dict[str, Any] = Field(default_factory=dict)


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
    use_refinement: bool = True
    initial_reference_count: int = Field(default=8, ge=0)
    policy_refine_reference_count: int = Field(default=8, ge=0)
    minimum_tool_count: int = Field(default=0, ge=0)
    cohesion_judge_count: int = Field(default=3, ge=0)
    cohesion_max_failure_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    golden_reference_count: int = Field(default=2, ge=0)
    golden_max_failure_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    max_judge_concurrency: int | None = Field(default=None, ge=1)
    random_seed: int | None = None
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
