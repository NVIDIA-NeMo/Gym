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
from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from nemo_gym.agents import BaseResponsesAPIAgentConfig, SandboxNetworkAccess
from nemo_gym.config import BenchmarkDatasetConfig, DatasetConfig, ModelServerRef, ResourcesServerRef
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
from responses_api_agents.nooa_agent.policy_settings import MediaInput, PolicySettings, ReasoningEffort


class NOOAAgentConfig(BaseResponsesAPIAgentConfig):
    """Gym server configuration for the NOOA adapter."""

    model_config = ConfigDict(extra="forbid")

    resources_server: ResourcesServerRef | None = Field(
        default=None,
        description="Dataset/schema discovery metadata; runtime access comes from environment session grants.",
    )
    model_server: ModelServerRef
    sandbox_provider: str | None = None
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    network_access: SandboxNetworkAccess = Field(
        default="inherit", description="Egress policy for a sandbox created by this agent; borrowed sandboxes inherit."
    )
    datasets: list[DatasetConfig | BenchmarkDatasetConfig] | None = Field(
        default=None,
        description="Rows rollout collection reads for this agent, such as a Harbor taskset's; this server does not.",
    )
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
    # The policy model as NOOA must know it. Every field is required and has no default: a run declares the model
    # it evaluates (the Eval Factory fragment or the benchmark config sets them), or the config fails to load.
    context_window: int = Field(
        gt=0, description="The served context window in tokens; NOOA's context planner budget."
    )
    max_output_tokens: int = Field(
        gt=0, description="Reply cap sent as max_output_tokens on every model call the request does not cap itself."
    )
    temperature: float = Field(
        ge=0, description="Sampling temperature sent on every model call the request does not set."
    )
    top_p: float = Field(gt=0, le=1, description="Nucleus sampling sent on every model call the request does not set.")
    reasoning_effort: ReasoningEffort = Field(
        description=(
            "Reasoning effort sent as reasoning.effort on every model call the request does not set; "
            "`none` sends no reasoning setting."
        )
    )
    media_inputs: tuple[MediaInput, ...] = Field(
        description=(
            "Media the policy model accepts, as NOOA's media_inputs; an empty list is text only and NOOA replaces "
            "other media with a short note instead of sending it."
        )
    )
    artifacts_dir: Path = Field(
        default=Path("nooa-artifacts"),
        description=(
            "Host directory for per-rollout NOOA artifacts, relative to the server's cwd. Every rollout's NOOA trace "
            "(OpenTelemetry spans plus the message journal that reconstructs each model call) is copied to "
            "<artifacts_dir>/<rollout_id>/traces.jsonl; a rollout whose trace could not be captured says so in its "
            "observation gaps."
        ),
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

    @property
    def policy_settings(self) -> PolicySettings:
        """The declared policy model, as one value the runners and the sandbox worker receive."""
        return PolicySettings(
            context_window=self.context_window,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            reasoning_effort=self.reasoning_effort,
            media_inputs=self.media_inputs,
        )

    @model_validator(mode="after")
    def require_prepared_runtime(self) -> NOOAAgentConfig:
        if self.runtime_wheelhouse_required and self.runtime_wheelhouse is None:
            raise ValueError(
                "This recipe requires an offline NOOA runtime. Set NOOA_SWE_RUNTIME_WHEELHOUSE to an absolute "
                "bundle path and prepare it with python -m responses_api_agents.nooa_agent.offline_runtime "
                "--output PATH --target x86_64/gnu before starting servers."
            )
        return self
