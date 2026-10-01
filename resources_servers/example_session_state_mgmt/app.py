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
from typing import Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

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
from nemo_gym.server_utils import SESSION_ID_KEY


class StatefulCounterResourcesServerConfig(BaseResourcesServerConfig):
    pass


class IncrementCounterRequest(BaseModel):
    count: int


class IncrementCounterResponse(BaseModel):
    success: bool


class GetCounterValueResponse(BaseModel):
    count: int


class StatefulCounterVerifyRequest(BaseVerifyRequest):
    expected_count: int


class StatefulCounterSeedSessionRequest(BaseSeedSessionRequest):
    initial_count: int


class StatefulCounterResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: StatefulCounterResourcesServerConfig
    session_id_to_counter: Dict[str, int] = Field(default_factory=dict)
    native_sessions: dict[str, ResourcesSeedSessionRequest] = Field(default_factory=dict)
    closed_sessions: dict[str, ResourcesCloseSessionRequest] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/increment_counter")(self.increment_counter)
        app.post("/get_counter_value")(self.get_counter_value)

        return app

    async def seed_session(
        self, request: Request, body: StatefulCounterSeedSessionRequest | ResourcesSeedSessionRequest
    ) -> BaseSeedSessionResponse | ResourcesSeedSessionResponse:
        if isinstance(body, ResourcesSeedSessionRequest):
            session_id = body.resources_session_id
            if session_id in self.closed_sessions:
                raise HTTPException(409, "Resources session is closed")
            previous = self.native_sessions.get(session_id)
            if previous is not None and previous != body:
                raise HTTPException(409, "Resources session has different task data or identity")
            initial_count = StatefulCounterSeedSessionRequest.model_validate(body.task_data).initial_count
            self.native_sessions[session_id] = body.model_copy(deep=True)
            request.session[SESSION_ID_KEY] = session_id
            self.session_id_to_counter.setdefault(session_id, initial_count)
            return ResourcesSeedSessionResponse(resources_session_id=session_id)
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_counter.setdefault(session_id, body.initial_count)
        return BaseSeedSessionResponse()

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        previous = self.native_sessions.get(body.resources_session_id)
        closed = self.closed_sessions.get(body.resources_session_id)
        if (previous is not None and previous.episode_id != body.episode_id) or (
            closed is not None and closed != body
        ):
            raise HTTPException(409, "Resources session belongs to another episode")
        self.session_id_to_counter.pop(body.resources_session_id, None)
        self.native_sessions.pop(body.resources_session_id, None)
        self.closed_sessions[body.resources_session_id] = body.model_copy(deep=True)
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    async def increment_counter(self, request: Request, body: IncrementCounterRequest) -> IncrementCounterResponse:
        session_id = request.session[SESSION_ID_KEY]
        if session_id in self.closed_sessions:
            raise HTTPException(409, "Resources session is closed")
        counter = self.session_id_to_counter.setdefault(session_id, 0)

        counter += body.count

        self.session_id_to_counter[session_id] = counter

        return IncrementCounterResponse(success=True)

    async def get_counter_value(self, request: Request) -> GetCounterValueResponse:
        session_id = request.session[SESSION_ID_KEY]
        if session_id in self.closed_sessions:
            raise HTTPException(409, "Resources session is closed")
        counter = self.session_id_to_counter.setdefault(session_id, 0)
        return GetCounterValueResponse(count=counter)

    async def verify(self, request: Request, body: StatefulCounterVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]

        reward = 0.0
        if session_id in self.session_id_to_counter:
            counter = self.session_id_to_counter[session_id]
            reward = float(body.expected_count == counter)

        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    StatefulCounterResourcesServer.run_webserver()
