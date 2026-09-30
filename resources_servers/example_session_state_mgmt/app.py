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

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, JsonValue

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
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
    checkpoint_mode = "exported"
    config: StatefulCounterResourcesServerConfig
    session_id_to_counter: Dict[str, int] = Field(default_factory=dict)

    async def export_session_states(self, session_ids: list[str]) -> dict[str, JsonValue]:
        return {session_id: {"count": self.session_id_to_counter.get(session_id, 0)} for session_id in session_ids}

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        counters = {}
        for session_id, state in states.items():
            count = state.get("count") if isinstance(state, dict) else None
            if not isinstance(count, int):
                raise ValueError(f"invalid counter state for session {session_id!r}: {state!r}")
            counters[session_id] = count
        self.session_id_to_counter.update(counters)

    async def retire_session_state(self, session_id: str) -> None:
        self.session_id_to_counter.pop(session_id, None)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/increment_counter")(self.increment_counter)
        app.post("/get_counter_value")(self.get_counter_value)

        return app

    async def seed_session(self, request: Request, body: StatefulCounterSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_counter.setdefault(session_id, body.initial_count)
        return BaseSeedSessionResponse()

    async def increment_counter(self, request: Request, body: IncrementCounterRequest) -> IncrementCounterResponse:
        session_id = request.session[SESSION_ID_KEY]
        counter = self.session_id_to_counter.setdefault(session_id, 0)

        counter += body.count

        self.session_id_to_counter[session_id] = counter

        return IncrementCounterResponse(success=True)

    async def get_counter_value(self, request: Request) -> GetCounterValueResponse:
        session_id = request.session[SESSION_ID_KEY]
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
