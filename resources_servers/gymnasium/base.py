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
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from nemo_gym.base_resources_server import BaseVerifyRequest, SimpleResourcesServer
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
)
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

    A session starts at ``/reset`` and ends at a terminal ``/step``. Sessions are ``restart_only`` for
    checkpoints unless a subclass sets ``checkpoint_mode = "exported"``. The default hooks then export and
    restore ``session_state`` one session at a time through ``export_env_state`` and ``restore_env_state``.
    """

    session_state: Dict[str, Any] = Field(default_factory=dict)

    def register_protocol_routes(self, app: FastAPI) -> None:
        # Replaces /seed_session, /close_session and /verify: gymnasium_agent drives episodes through these.
        app.post("/reset")(self._reset_endpoint)
        app.post("/step")(self._step_endpoint)

    async def _reset_endpoint(self, body: EnvResetRequest, request: Request) -> EnvResetResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        obs, info = await self.reset(body.model_extra or {}, session_id)
        self.checkpoint_session_started(request)
        return EnvResetResponse(observation=obs, info=info)

    async def _step_endpoint(self, body: EnvStepRequest, request: Request) -> EnvStepResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        obs, reward, terminated, truncated, info = await self.step(body.response, body.model_extra or {}, session_id)
        if terminated or truncated:
            await self.close_session(session_id)
            self.checkpoint_session_ended(request)
        return EnvStepResponse(observation=obs, reward=reward, terminated=terminated, truncated=truncated, info=info)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        return None, {}

    @abstractmethod
    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]: ...

    async def close_session(self, session_id: Optional[str]) -> None:
        self._cleanup_session(session_id)

    def _cleanup_session(self, session_id: Optional[str]) -> None:
        """Release state owned by a session. Safe to call repeatedly."""
        self.session_state.pop(session_id, None)

    async def export_session_states(self, session_ids: list[str]) -> dict[str, JsonValue]:
        return {
            session_id: self.export_env_state(self.session_state[session_id])
            for session_id in session_ids
            if session_id in self.session_state
        }

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        # Convert every state before installing any, so an invalid one leaves the server untouched.
        restored = {session_id: self.restore_env_state(state) for session_id, state in states.items()}
        self.session_state.update(restored)

    async def retire_session_state(self, session_id: str) -> None:
        await self.close_session(session_id)

    def export_env_state(self, state: Any) -> JsonValue:
        """Return one session's ``session_state`` entry as JSON for a checkpoint.

        The default accepts state that is already JSON and copies it, so later steps cannot change what was exported.
        Override it for state that holds objects, such as a random number generator.
        """
        return json.loads(json.dumps(state))

    def restore_env_state(self, state: JsonValue) -> Any:
        """Rebuild one session's ``session_state`` entry from ``export_env_state``'s output.

        Raise if the state is invalid; no session is restored then.
        The default returns it unchanged.
        """
        return state

    @staticmethod
    def tool_output(call: NeMoGymResponseFunctionToolCall, result: Any) -> dict:
        return {"call_id": call.call_id, "output": json.dumps(result, default=str)}

    async def verify(self, body: BaseVerifyRequest) -> None:  # type: ignore[override]
        raise NotImplementedError("GymnasiumServer uses /step instead of /verify. Use with gymnasium_agent.")
