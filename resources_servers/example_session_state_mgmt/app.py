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

from fastapi import Request
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
    gym_tool,
)
from nemo_gym.server_utils import SESSION_ID_KEY


class StatefulCounterResourcesServerConfig(BaseResourcesServerConfig):
    pass


class IncrementCounterResponse(BaseModel):
    success: bool


class GetCounterValueResponse(BaseModel):
    count: int


class StatefulCounterVerifyRequest(BaseVerifyRequest):
    expected_count: int


class StatefulCounterSeedSessionRequest(BaseSeedSessionRequest):
    initial_count: int


class StatefulCounterResourcesServer(SimpleResourcesServer):
    """Per-rollout session state example: each ``@gym_tool`` method is served over both HTTP and MCP,
    with ``session_id`` injected by the base class; ``seed_session`` and ``verify`` stay harness endpoints."""

    config: StatefulCounterResourcesServerConfig
    session_id_to_counter: Dict[str, int] = Field(default_factory=dict)

    async def seed_session(self, request: Request, body: StatefulCounterSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_counter.setdefault(session_id, body.initial_count)
        return BaseSeedSessionResponse()

    @gym_tool
    async def increment_counter(self, session_id: str, count: int) -> IncrementCounterResponse:
        """Add `count` to this session's counter."""
        counter = self.session_id_to_counter.setdefault(session_id, 0)

        counter += count

        self.session_id_to_counter[session_id] = counter

        return IncrementCounterResponse(success=True)

    @gym_tool
    async def get_counter_value(self, session_id: str) -> GetCounterValueResponse:
        """Get this session's current counter value."""
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
