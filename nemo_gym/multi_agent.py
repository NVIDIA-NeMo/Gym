# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Alternating-turn multi-agent resources-server primitives.

The environment owns turn order and hidden state.  An orchestrator receives only
the active agent's observation and sends back that agent's action, preventing it
from accidentally forwarding one agent's private observation to another.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseVerifyRequest, SimpleResourcesServer
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_correlation import RolloutContextMiddleware
from nemo_gym.server_utils import SESSION_ID_KEY


class MultiAgentResetRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class MultiAgentResetResponse(BaseModel):
    active_agent: str
    observation: str
    info: dict[str, Any] = Field(default_factory=dict)


class MultiAgentStepRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    agent_id: str
    action: str


class MultiAgentStepResponse(BaseModel):
    active_agent: Optional[str] = None
    observation: Optional[str] = None
    rewards: dict[str, float] = Field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = Field(default_factory=dict)


class MultiAgentCloseResponse(BaseModel):
    closed: bool = True


class AgentTurn(BaseModel):
    observation: str
    action: str


class MultiAgentResourcesServer(SimpleResourcesServer):
    """Base server for one-active-agent-at-a-time environments."""

    session_state: Dict[str, Any] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.add_middleware(RolloutContextMiddleware)
        app.post("/reset")(self._reset_endpoint)
        app.post("/step")(self._step_endpoint)
        app.post("/close")(self._close_endpoint)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    async def _reset_endpoint(self, body: MultiAgentResetRequest, request: Request) -> MultiAgentResetResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        return await self.reset(body.model_extra or {}, session_id)

    async def _step_endpoint(self, body: MultiAgentStepRequest, request: Request) -> MultiAgentStepResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        result = await self.step(body.agent_id, body.action, body.model_extra or {}, session_id)
        if result.terminated or result.truncated:
            await self.close_session(session_id)
        return result

    async def _close_endpoint(self, request: Request) -> MultiAgentCloseResponse:
        await self.close_session(request.session.get(SESSION_ID_KEY))
        return MultiAgentCloseResponse()

    @abstractmethod
    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> MultiAgentResetResponse:
        """Create an episode and return the first active agent's private observation."""

    @abstractmethod
    async def step(
        self,
        agent_id: str,
        action: str,
        metadata: dict,
        session_id: Optional[str] = None,
    ) -> MultiAgentStepResponse:
        """Apply the active agent's action and return the next private observation."""

    async def close_session(self, session_id: Optional[str]) -> None:
        self.session_state.pop(session_id, None)

    async def verify(self, body: BaseVerifyRequest) -> None:  # type: ignore[override]
        raise NotImplementedError("MultiAgentResourcesServer uses /step instead of /verify.")
