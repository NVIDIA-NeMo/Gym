# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from abc import abstractmethod
from typing import Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import BaseVerifyRequest, SimpleResourcesServer
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY


class EnvResetRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class EnvResetResponse(BaseModel):
    observation: Optional[str] = None
    info: dict = {}


class EnvStepRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    response: NeMoGymResponse


class EnvStepResponse(BaseModel):
    observation: Optional[str] = None
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict = {}


def extract_text(response: NeMoGymResponse) -> str:
    """Extract the first text content from a NeMoGymResponse."""
    for item in response.output:
        if item.type == "message":
            content = item.content
            if isinstance(content, str):
                return content
            for c in content:
                if c.type == "output_text":
                    return c.text
    return ""


class GymnasiumServer(SimpleResourcesServer):
    """Gymnasium-style base class. Pair with gymnasium_agent.

    step() returns (observation, reward, terminated, truncated, info).
    """

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/reset")(self._reset_endpoint)
        app.post("/step")(self._step_endpoint)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    async def _reset_endpoint(self, body: EnvResetRequest, request: Request) -> EnvResetResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        obs, info = await self.reset(body.model_extra or {}, session_id)
        return EnvResetResponse(observation=obs, info=info)

    async def _step_endpoint(self, body: EnvStepRequest, request: Request) -> EnvStepResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        obs, reward, terminated, truncated, info = await self.step(body.response, body.model_extra or {}, session_id)
        return EnvStepResponse(observation=obs, reward=reward, terminated=terminated, truncated=truncated, info=info)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        return None, {}

    @abstractmethod
    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]: ...

    async def verify(self, body: BaseVerifyRequest) -> None:  # type: ignore[override]
        raise NotImplementedError("GymnasiumServer uses /step instead of /verify. Pair with gymnasium_agent.")
