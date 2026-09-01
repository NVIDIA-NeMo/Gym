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
from pydantic import BaseModel, Field

from nemo_gym._checkpoint import ResourceSnapshot
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.rollout_correlation import current_attempt_index, current_logical_rollout_id
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
    config: StatefulCounterResourcesServerConfig
    session_id_to_counter: Dict[str, int] = Field(default_factory=dict)
    execution_to_session: Dict[tuple[str, int], str] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        app.post("/increment_counter")(self.increment_counter)
        app.post("/get_counter_value")(self.get_counter_value)

        return app

    async def seed_session(self, request: Request, body: StatefulCounterSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_counter.setdefault(session_id, body.initial_count)
        identity = self._current_identity()
        if identity is not None:
            self.execution_to_session[identity] = session_id
        return BaseSeedSessionResponse()

    async def increment_counter(self, request: Request, body: IncrementCounterRequest) -> IncrementCounterResponse:
        session_id = self._session_id(request)
        counter = self.session_id_to_counter.setdefault(session_id, 0)

        counter += body.count

        self.session_id_to_counter[session_id] = counter

        return IncrementCounterResponse(success=True)

    async def get_counter_value(self, request: Request) -> GetCounterValueResponse:
        session_id = self._session_id(request)
        counter = self.session_id_to_counter.setdefault(session_id, 0)
        return GetCounterValueResponse(count=counter)

    async def verify(self, request: Request, body: StatefulCounterVerifyRequest) -> BaseVerifyResponse:
        session_id = self._session_id(request)

        reward = 0.0
        if session_id in self.session_id_to_counter:
            counter = self.session_id_to_counter[session_id]
            reward = float(body.expected_count == counter)

        return BaseVerifyResponse(**body.model_dump(), reward=reward)

    @staticmethod
    def _current_identity() -> tuple[str, int] | None:
        rollout_id = current_logical_rollout_id()
        attempt_index = current_attempt_index()
        if rollout_id is None or attempt_index is None:
            return None
        return rollout_id, attempt_index

    def _session_id(self, request: Request) -> str:
        identity = self._current_identity()
        if identity is not None and identity in self.execution_to_session:
            return self.execution_to_session[identity]
        return request.session[SESSION_ID_KEY]

    def checkpoint_state_enabled(self) -> bool:
        return True

    async def export_checkpoint_state(self, rollout_id: str, attempt_index: int) -> dict:
        session_id = self.execution_to_session[(rollout_id, attempt_index)]
        return {"counter": self.session_id_to_counter[session_id]}

    async def restore_checkpoint_states(self, snapshots: list[ResourceSnapshot]) -> None:
        counters = dict(self.session_id_to_counter)
        index = dict(self.execution_to_session)
        for snapshot in snapshots:
            session_id = f"checkpoint:{snapshot.rollout_id}:a{snapshot.attempt_index}"
            counters[session_id] = int(snapshot.state["counter"])
            index[(snapshot.rollout_id, snapshot.attempt_index)] = session_id
        self.session_id_to_counter = counters
        self.execution_to_session = index


if __name__ == "__main__":
    StatefulCounterResourcesServer.run_webserver()
