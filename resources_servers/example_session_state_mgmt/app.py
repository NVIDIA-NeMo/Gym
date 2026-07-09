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


class StatefulCounterSeedSessionRequest(BaseSeedSessionRequest):
    initial_count: int


class StatefulCounterVerifyRequest(BaseVerifyRequest):
    expected_count: int


class StatefulCounterResourcesServer(SimpleResourcesServer):
    """Canonical example of per-rollout session state behind Gym tools.

    Every rollout gets its own counter, keyed by the Gym session id. The two ``@gym_tool`` methods
    below are model-facing tools: the base class serves each one over BOTH transports — an HTTP
    ``POST /<tool_name>`` route and an MCP tool on ``/mcp`` — from this single declaration.
    ``seed_session`` and ``verify`` are harness endpoints (the agent calls them directly around the
    rollout), so they stay plain methods and are never exposed to the model as tools.
    """

    config: StatefulCounterResourcesServerConfig

    # Per-rollout state: one counter per Gym session id. In-memory state like this is fine for a
    # single-process server; anything multi-worker needs external storage.
    session_id_to_counter: Dict[str, int] = Field(default_factory=dict)

    async def seed_session(self, request: Request, body: StatefulCounterSeedSessionRequest) -> BaseSeedSessionResponse:
        # Harness endpoint: the agent seeds each rollout's counter from the task row before the
        # model takes its first turn. Harness endpoints keep the FastAPI ``request`` parameter and
        # read the session id off the session cookie themselves.
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_counter.setdefault(session_id, body.initial_count)
        return BaseSeedSessionResponse()

    @gym_tool
    async def increment_counter(self, session_id: str, count: int) -> IncrementCounterResponse:
        """Add `count` to this session's counter."""
        # ``session_id`` is injected by the base on both transports (from the session cookie on
        # HTTP, from the signed session token on MCP) and is never visible in the model-facing
        # tool schema. The remaining typed parameters ARE the tool's input schema.
        counter = self.session_id_to_counter.setdefault(session_id, 0) + count
        self.session_id_to_counter[session_id] = counter
        return IncrementCounterResponse(success=True)

    @gym_tool
    async def get_counter_value(self, session_id: str) -> GetCounterValueResponse:
        """Get this session's current counter value."""
        return GetCounterValueResponse(count=self.session_id_to_counter.setdefault(session_id, 0))

    async def verify(self, request: Request, body: StatefulCounterVerifyRequest) -> BaseVerifyResponse:
        # Harness endpoint: score the rollout by comparing the session's final counter value to the
        # task's expected count. An unseeded session scores 0.
        session_id = request.session[SESSION_ID_KEY]

        reward = 0.0
        if session_id in self.session_id_to_counter:
            reward = float(body.expected_count == self.session_id_to_counter[session_id])

        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    StatefulCounterResourcesServer.run_webserver()
