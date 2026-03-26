# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains a set of schemas used by both the AviaryResourcesServer and the AviaryAgent.
"""

from typing import Literal

from openai.types.responses import FunctionToolParam
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
)


class AviaryResourcesServerConfig(BaseResourcesServerConfig):
    pass


class AviarySeedSessionRequest(BaseSeedSessionRequest):
    task_idx: int


class AviaryEnvStateEasyInputMessage(NeMoGymEasyInputMessage):
    """Special subclass so we can identify messages representing environment state."""

    is_env_state: Literal[True] = True


class AviarySeedSessionResponse(BaseSeedSessionResponse):
    env_id: str
    obs: list[NeMoGymEasyInputMessage | AviaryEnvStateEasyInputMessage]
    tools: list[FunctionToolParam]


class AviaryStepRequest(BaseModel):
    env_id: str
    action: list[NeMoGymResponseFunctionToolCall]


class AviaryStepResponse(BaseModel):
    obs: list[NeMoGymFunctionCallOutput | NeMoGymEasyInputMessage | AviaryEnvStateEasyInputMessage]
    reward: float
    done: bool


class AviaryNeMoGymResponse(NeMoGymResponse):
    env_id: str
    group_id: str
    contains_transitions: bool
    output: list[NeMoGymResponseOutputItem] | list[list[NeMoGymResponseOutputItem]]


class AviaryAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    response: AviaryNeMoGymResponse


# Use this MRO so AviaryAgentVerifyRequest.response supersedes BaseVerifyResponse.response
class AviaryAgentVerifyResponse(AviaryAgentVerifyRequest, BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class AviaryCloseRequest(BaseModel):
    env_id: str


class AviaryCloseResponse(BaseModel):
    message: str
    success: bool
