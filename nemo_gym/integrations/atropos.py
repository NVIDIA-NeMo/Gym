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
from typing import Any, Optional
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
    NeMoGymResponse,
)


class AtroposResourcesServerConfig(BaseResourcesServerConfig):
    pass


class AtroposSeedSessionRequest(BaseSeedSessionRequest):
    task_idx: int


class AtroposSeedSessionResponse(BaseSeedSessionResponse):
    env_id: str
    obs: list[NeMoGymEasyInputMessage]
    system_prompt: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class AtroposStepRequest(BaseModel):
    env_id: str
    action: str


class AtroposStepResponse(BaseModel):
    obs: list[NeMoGymEasyInputMessage]
    reward: float
    done: bool
    info: Optional[dict[str, Any]] = None


class AtroposNeMoGymResponse(NeMoGymResponse):
    env_id: str


class AtroposAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    response: AtroposNeMoGymResponse


class AtroposAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class AtroposCloseRequest(BaseModel):
    env_id: str


class AtroposCloseResponse(BaseModel):
    message: str
    success: bool
