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
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI
from pydantic import BaseModel, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.single_agent_episode_types import ResponsesResourcesVerifyRequest
from nemo_gym.verifier_fixture import VerifierFixture


class SimpleWeatherResourcesServerConfig(BaseResourcesServerConfig):
    pass


class GetWeatherRequest(BaseModel):
    city: str


class GetWeatherResponse(BaseModel):
    city: str
    weather_description: str


class SimpleWeatherVerifier:
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        reward = float(
            any(item.type == "function_call" and item.name == "get_weather" for item in body.response.output)
        )
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


class SimpleWeatherResourcesServer(SimpleWeatherVerifier, SimpleResourcesServer):
    config: SimpleWeatherResourcesServerConfig
    _native_sessions: dict[str, tuple[EpisodeId, TaskId]] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/get_weather")(self.get_weather)
        app.post("/close_session")(self.close_session)

        return app

    async def seed_session(
        self,
        body: ResourcesSeedSessionRequest | BaseSeedSessionRequest,
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        if not isinstance(body, ResourcesSeedSessionRequest):
            return BaseSeedSessionResponse()

        resources_session_id = f"resources-session-{uuid4().hex}"
        self._native_sessions[resources_session_id] = (body.episode_id, body.task_id)
        return ResourcesSeedSessionResponse(resources_session_id=resources_session_id)

    async def verify(
        self,
        body: ResponsesResourcesVerifyRequest | BaseVerifyRequest,
    ) -> BaseVerifyResponse:
        if isinstance(body, ResponsesResourcesVerifyRequest):
            body = BaseVerifyRequest(
                responses_create_params=body.verification_input.responses_create_params,
                response=body.verification_input.response,
            )
        return await SimpleWeatherVerifier.verify(self, body)

    async def close_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        try:
            episode_id, _ = self._native_sessions[body.resources_session_id]
        except KeyError as error:
            raise ValueError(f"Unknown resources_session_id: {body.resources_session_id}") from error
        if body.episode_id != episode_id:
            raise ValueError("episode_id does not match the seeded resources session")

        del self._native_sessions[body.resources_session_id]
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    async def get_weather(self, body: GetWeatherRequest) -> GetWeatherResponse:
        return GetWeatherResponse(city=body.city, weather_description=f"The weather in {body.city} is cold.")


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=SimpleWeatherVerifier,
    request_model=BaseVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    SimpleWeatherResourcesServer.run_webserver()
