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
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
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
    ray_enabled = False
    # Tool results depend only on the request, so a continuation needs no session state.
    checkpoint_mode = "stateless"
    config: SimpleWeatherResourcesServerConfig
    _session_episodes: dict[str, tuple[EpisodeId, TaskId]] = PrivateAttr(default_factory=dict)
    _closed_session_ids: set[str] = PrivateAttr(default_factory=set)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/get_weather")(self.get_weather)

        return app

    async def seed_session(
        self,
        body: ResourcesSeedSessionRequest | BaseSeedSessionRequest,
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        if not isinstance(body, ResourcesSeedSessionRequest):
            return BaseSeedSessionResponse()

        resources_session_id = body.resources_session_id
        # Nothing below awaits, so no other request can run between the checks and the update.
        if resources_session_id in self._closed_session_ids:
            raise ValueError(f"Resources session is already closed: {resources_session_id}")
        identity = self._session_episodes.setdefault(resources_session_id, (body.episode_id, body.task_id))
        if identity != (body.episode_id, body.task_id):
            raise ValueError("resources_session_id is already bound to another episode or task")
        return ResourcesSeedSessionResponse(resources_session_id=resources_session_id)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        # Sessions are keyed by resources_session_id, not the cookie's session id, so the body names the session.
        resources_session_id = body.resources_session_id
        # Nothing below awaits, so no other request can run between the check and the update.
        identity = self._session_episodes.get(resources_session_id)
        if identity is not None and body.episode_id != identity[0]:
            raise ValueError("episode_id does not match the seeded resources session")
        self._session_episodes.pop(resources_session_id, None)
        self._closed_session_ids.add(resources_session_id)
        return ResourcesCloseSessionResponse(resources_session_id=resources_session_id)

    async def get_weather(self, body: GetWeatherRequest) -> GetWeatherResponse:
        return GetWeatherResponse(city=body.city, weather_description=f"The weather in {body.city} is cold.")


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=SimpleWeatherVerifier,
    request_model=BaseVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    SimpleWeatherResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = SimpleWeatherResourcesServer.run_webserver()  # noqa: F401
