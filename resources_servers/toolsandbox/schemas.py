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
"""Schemas shared by the ToolSandbox resources server and its agent harness.

ToolSandbox is a multi-turn, tool-using benchmark. Each scenario is an
:class:`env` held by the resources server: the agent-under-test (the gym policy
model) converses with an internal user-simulator LLM while issuing Python tool
calls against a stateful sandbox. The flow mirrors the aviary env pattern:
``seed_session -> obs + tools``, ``/step(action) -> obs, reward, done``,
``/close``, then a pure ``/verify`` that returns the milestone similarity.
"""

from typing import List, Optional

from openai.types.responses import FunctionToolParam
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
)


class ToolSandboxResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the ToolSandbox resources server."""

    # The gym model server that plays the user simulator (analogous to
    # rolemrc's judge_model_server). Its /v1/chat/completions endpoint is
    # called by the internal USER role during /step.
    user_model_server: ModelServerRef

    # Model id to send in the user-sim chat-completions request. When None the
    # request omits it and the model server falls back to its configured model.
    user_model_name: Optional[str] = None

    # User-simulator sampling. None => omit the parameter (server default).
    user_temperature: Optional[float] = None
    user_top_p: Optional[float] = None
    user_max_tokens: Optional[int] = None
    # Toggle reasoning for the user simulator (vLLM chat_template_kwargs /
    # OpenAI reasoning_effort, dispatched by model id in the vendored role).
    user_enable_thinking: Optional[bool] = None

    # Optional override of the per-scenario message cap. None => use each
    # scenario's own ``max_messages`` (default 30). The cap is on the raw
    # sandbox message index, enforced cumulatively across /step calls.
    max_messages: Optional[int] = None

    # Which tool backend to prefer when tool names collide (default / rapid_api).
    preferred_tool_backend: str = "default"

    # Restrict to a subset of scenario names; None => all scenarios.
    scenarios: Optional[List[str]] = None

    # Use the small smoke-test scenario subset instead of the full set.
    test_mode: bool = False

    # Controls ``additionalProperties`` in seeded tool JSON schemas.
    #   True  (default) => leave it unset — matches upstream ToolSandbox, which
    #                      sends bare Chat-Completions tool schemas.
    #   False           => inject ``additionalProperties: False`` (a strict
    #                      Responses-API convention upstream never sends).
    tool_schema_additional_properties: bool = True


class ToolSandboxSeedSessionRequest(BaseSeedSessionRequest):
    task_idx: int


class ToolSandboxSeedSessionResponse(BaseSeedSessionResponse):
    env_id: str
    # Scenario name, for logging / debugging.
    scenario: str
    # Initial agent-facing conversation head (system prompt + first user turn).
    obs: List[NeMoGymEasyInputMessage]
    # Tools the agent may call, in Responses-API FunctionToolParam form.
    tools: List[FunctionToolParam]


class ToolSandboxStepRequest(BaseModel):
    env_id: str
    # The agent's tool calls this turn (empty when the agent replied in text).
    function_calls: List[NeMoGymResponseFunctionToolCall] = Field(default_factory=list)
    # The agent's natural-language reply to the user (used only when there are
    # no tool calls, matching the vendored agent's either/or behaviour).
    text: Optional[str] = None


class ToolSandboxStepResponse(BaseModel):
    # New messages addressed to the agent: tool outputs (function_call_output)
    # and/or the user simulator's reply (user message).
    obs: List[NeMoGymFunctionCallOutput | NeMoGymEasyInputMessage]
    # Per-step reward is always 0.0; the real reward is the end-of-episode
    # milestone similarity computed in /close and returned by /verify.
    reward: float = 0.0
    done: bool


class ToolSandboxCloseRequest(BaseModel):
    env_id: str


class ToolSandboxCloseResponse(BaseModel):
    message: str
    success: bool


class ToolSandboxNeMoGymResponse(NeMoGymResponse):
    env_id: str
    group_id: str
    # Discriminates the `output` union below: True => `output` is a list of
    # per-step agent-state snapshots (transitions); False => a single flat
    # trajectory. Mirrors AviaryNeMoGymResponse so trajectory consumers can
    # interpret the shape without guessing.
    contains_transitions: bool
    output: List[NeMoGymResponseOutputItem] | List[List[NeMoGymResponseOutputItem]]


class ToolSandboxVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    response: ToolSandboxNeMoGymResponse


# MRO so ToolSandboxVerifyRequest.response supersedes BaseVerifyResponse.response.
class ToolSandboxVerifyResponse(ToolSandboxVerifyRequest, BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
