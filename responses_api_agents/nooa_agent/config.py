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

from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict, Field, field_validator, model_validator

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from responses_api_agents.nooa_agent.invocation import (
    NOOAInvocationAdapter as NOOAInvocationAdapter,
)
from responses_api_agents.nooa_agent.invocation import (
    NOOAInvocationConfig as NOOAInvocationConfig,
)
from responses_api_agents.nooa_agent.invocation import (
    load_agent_class as load_agent_class,
)
from responses_api_agents.nooa_agent.invocation import (
    load_invocation_adapter as load_invocation_adapter,
)
from responses_api_agents.nooa_agent.invocation import (
    validate_invocation as validate_invocation,
)


class NOOAAgentConfig(BaseResponsesAPIAgentConfig):
    """Gym server configuration for the NOOA adapter."""

    model_config = ConfigDict(extra="forbid")

    resources_server: ResourcesServerRef | None = Field(
        default=None,
        description="Dataset/schema discovery metadata; runtime access comes from environment session grants.",
    )
    model_server: ModelServerRef
    runtime_wheelhouse: Path | None = Field(
        default=None,
        description="Controller-prepared offline runtime bundle; when set, task setup uses no package-network access.",
    )
    runtime_wheelhouse_required: bool = Field(
        default=False, description="Require an explicitly prepared offline runtime before starting this agent."
    )
    nooa: NOOAInvocationConfig
    max_policy_calls: int | None = Field(
        default=None, gt=0, description="Optional shared model-call limit per rollout; null disables the limit."
    )
    context_window: int | None = Field(
        default=None, gt=0, description="Known policy context window used by NOOA's context planner."
    )
    trace_jsonl: bool = Field(
        default=False,
        description=(
            "Write NOOA's OTLP JSONL trace in the sandbox for each rollout and copy it to "
            "<artifacts_dir>/<rollout_id>/traces.jsonl on this host. A debugging artifact, not evidence: Gym's own "
            "capture and the rollout record remain authoritative."
        ),
    )
    artifacts_dir: Path = Field(
        default=Path("nooa-artifacts"),
        description="Host directory for per-rollout NOOA artifacts such as the trace file; relative to the server's cwd.",
    )

    sandbox_runner_timeout_seconds: float = Field(
        default=21420,
        gt=0,
        allow_inf_nan=False,
        description="Finite sandbox worker ceiling in seconds; separate from the environment's episode timeout.",
    )
    session_close_timeout_seconds: float = Field(
        default=30,
        gt=0,
        allow_inf_nan=False,
        description="Budget in seconds for each sandbox close phase and for joining the remaining activation.",
    )

    @field_validator("runtime_wheelhouse")
    @classmethod
    def normalize_runtime_wheelhouse(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        expanded = value.expanduser()
        if not expanded.is_absolute():
            raise ValueError(
                "runtime_wheelhouse must be an absolute path (or ~/path); relative server paths are ambiguous"
            )
        return expanded.resolve()

    @model_validator(mode="after")
    def require_prepared_runtime(self) -> NOOAAgentConfig:
        if self.runtime_wheelhouse_required and self.runtime_wheelhouse is None:
            raise ValueError(
                "This recipe requires an offline NOOA runtime. Set NOOA_SWE_RUNTIME_WHEELHOUSE to an absolute "
                "bundle path and prepare it with python -m responses_api_agents.nooa_agent.offline_runtime "
                "--output PATH --target x86_64/gnu before starting servers."
            )
        return self
