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

from pydantic import ConfigDict, Field, field_validator

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig
from nemo_gym.config_types import ModelServerRef
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

    model_server: ModelServerRef
    nooa: NOOAInvocationConfig
    runtime_requirements_file: Path | None = Field(
        default=None,
        description="Explicit task-runtime requirements profile; default retains the agent's original NOOA pin.",
    )
    max_policy_calls: int | None = Field(
        default=None, gt=0, description="Optional shared model-call limit per rollout; null disables the limit."
    )
    context_window: int | None = Field(
        default=None, gt=0, description="Known policy context window used by NOOA's context planner."
    )

    @field_validator("num_workers")
    @classmethod
    def require_single_worker(cls, value: int | None) -> int | None:
        if value not in (None, 1):
            raise ValueError("NOOA sessions require num_workers=1")
        return value
