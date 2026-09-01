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

import json
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import BaseVerifyRequest, SimpleResourcesServer
from nemo_gym.checkpoint import ResourceSnapshot
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
)
from nemo_gym.rollout_correlation import current_attempt_index, current_logical_rollout_id
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
    """Extract all text content from a NeMoGymResponse."""
    parts = []
    for item in response.output:
        if item.type == "message":
            content = item.content
            if isinstance(content, str):
                parts.append(content)
            else:
                for c in content:
                    if c.type == "output_text":
                        parts.append(c.text)
    return "".join(parts)


class GymnasiumServer(SimpleResourcesServer):
    """Gymnasium-style base class. Used with gymnasium_agent.

    step() returns (observation, reward, terminated, truncated, info).
    """

    session_state: Dict[str, Any] = Field(default_factory=dict)
    execution_to_session: Dict[tuple[str, int], str] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/reset")(self._reset_endpoint)
        app.post("/step")(self._step_endpoint)
        return app

    async def _reset_endpoint(self, body: EnvResetRequest, request: Request) -> EnvResetResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        identity = self._current_identity()
        if identity is not None and session_id is not None:
            self.execution_to_session[identity] = session_id
        obs, info = await self.reset(body.model_extra or {}, session_id)
        return EnvResetResponse(observation=obs, info=info)

    async def _step_endpoint(self, body: EnvStepRequest, request: Request) -> EnvStepResponse:
        identity = self._current_identity()
        session_id = self.execution_to_session.get(identity) if identity is not None else None
        if session_id is None:
            session_id = request.session.get(SESSION_ID_KEY)
        obs, reward, terminated, truncated, info = await self.step(body.response, body.model_extra or {}, session_id)
        if terminated or truncated:
            await self.close_session(session_id)
            if identity is not None:
                self.execution_to_session.pop(identity, None)
                if self._checkpoint_participant is not None:
                    self.checkpoint_participant().mark_terminal_after_request(*identity)
        return EnvStepResponse(observation=obs, reward=reward, terminated=terminated, truncated=truncated, info=info)

    @staticmethod
    def _current_identity() -> Optional[tuple[str, int]]:
        rollout_id = current_logical_rollout_id()
        attempt_index = current_attempt_index()
        if rollout_id is None or attempt_index is None:
            return None
        return rollout_id, attempt_index

    def checkpoint_state_enabled(self) -> bool:
        return True

    def serialize_session_state(self, state: Any) -> dict[str, Any]:
        if not isinstance(state, dict):
            raise TypeError(f"Gymnasium session state must be a dictionary, got {type(state).__name__}")
        # Fail during prepare if the environment did not provide a JSON-safe
        # logical snapshot. Live Python objects are never written implicitly.
        return json.loads(json.dumps(state))

    def deserialize_session_state(self, state: dict[str, Any]) -> Any:
        return deepcopy(state)

    async def export_checkpoint_state(self, rollout_id: str, attempt_index: int) -> dict[str, Any]:
        session_id = self.execution_to_session[(rollout_id, attempt_index)]
        return self.serialize_session_state(self.session_state[session_id])

    async def restore_checkpoint_states(self, snapshots: list[ResourceSnapshot]) -> None:
        replacement_state = dict(self.session_state)
        replacement_index = dict(self.execution_to_session)
        for snapshot in snapshots:
            session_id = f"checkpoint:{snapshot.rollout_id}:a{snapshot.attempt_index}"
            replacement_state[session_id] = self.deserialize_session_state(snapshot.state)
            replacement_index[(snapshot.rollout_id, snapshot.attempt_index)] = session_id
        self.session_state = replacement_state
        self.execution_to_session = replacement_index

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        return None, {}

    @abstractmethod
    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]: ...

    async def close_session(self, session_id: Optional[str]) -> None:
        self.session_state.pop(session_id, None)

    @staticmethod
    def tool_output(call: NeMoGymResponseFunctionToolCall, result: Any) -> dict:
        return {"call_id": call.call_id, "output": json.dumps(result, default=str)}

    async def verify(self, body: BaseVerifyRequest) -> None:  # type: ignore[override]
        raise NotImplementedError("GymnasiumServer uses /step instead of /verify. Use with gymnasium_agent.")
