# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.grl_tetris.tetris_env import TetrisEnv


DEFAULT_GRID_LOOKUP = {0: "_", 1: "#", 2: "X"}
DEFAULT_ACTION_LOOKUP = {0: "Left", 1: "Right", 2: "Down"}


class GrlTetrisResourcesServerConfig(BaseResourcesServerConfig):
    env_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "grid_lookup": DEFAULT_GRID_LOOKUP,
            "action_lookup": DEFAULT_ACTION_LOOKUP,
            "render_mode": "text",
            "dim_x": 4,
            "dim_y": 4,
            "box_type": 3,
        }
    )


class GrlTetrisSeedSessionRequest(BaseSeedSessionRequest):
    seed: Optional[int] = None


class GrlTetrisSeedSessionResponse(BaseSeedSessionResponse):
    observation: str


class GrlTetrisStepRequest(BaseModel):
    actions: List[Union[str, int]] = Field(default_factory=list)


class GrlTetrisStepTrace(BaseModel):
    action_id: int
    action_label: str
    reward: float
    done: bool
    info: Dict[str, Any]


class GrlTetrisStepResponse(BaseModel):
    observation: str
    reward: float
    total_reward: float
    done: bool
    steps: List[GrlTetrisStepTrace]
    history: List[GrlTetrisStepTrace] = Field(default_factory=list)


class GrlTetrisVerifyResponse(BaseVerifyResponse):
    success: bool


@dataclass
class TetrisSessionState:
    env: Any
    observation: str
    total_reward: float = 0.0
    done: bool = False
    last_info: Dict[str, Any] = field(default_factory=dict)
    history: List[GrlTetrisStepTrace] = field(default_factory=list)


class GrlTetrisResourcesServer(SimpleResourcesServer):
    config: GrlTetrisResourcesServerConfig
    server_client: ServerClient
    session_id_to_state: Dict[str, TetrisSessionState] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/step")(self.step)
        return app

    def _create_env(self) -> TetrisEnv:
        return TetrisEnv(self.config.env_config)

    async def seed_session(self, request: Request, body: GrlTetrisSeedSessionRequest) -> GrlTetrisSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        env = self._create_env()
        observation = env.reset(seed=body.seed)

        self.session_id_to_state[session_id] = TetrisSessionState(
            env=env,
            observation=observation,
        )
        return GrlTetrisSeedSessionResponse(observation=observation)

    async def step(self, request: Request, body: GrlTetrisStepRequest) -> GrlTetrisStepResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id is None or session_id not in self.session_id_to_state:
            raise HTTPException(status_code=400, detail="Session not initialized. Call /seed_session first.")

        session_state = self.session_id_to_state[session_id]
        env = session_state.env

        reverse_lookup = {label.lower(): idx for idx, label in env.ACTION_LOOKUP.items()}
        total_step_reward = 0.0
        steps: List[GrlTetrisStepTrace] = []

        if session_state.done:
            return GrlTetrisStepResponse(
                observation=session_state.observation,
                reward=0.0,
                total_reward=session_state.total_reward,
                done=True,
                steps=[],
                history=list(session_state.history),
            )

        for action in body.actions:
            action_id = self._parse_action(action, reverse_lookup)
            if action_id not in env.ACTION_LOOKUP:
                raise HTTPException(status_code=400, detail=f"Invalid action identifier: {action}")

            next_obs, reward, done, info = env.step(action_id)
            info = self._to_python_types(info)
            total_step_reward += reward
            session_state.total_reward += reward
            session_state.observation = next_obs
            session_state.last_info = info
            session_state.done = bool(done)

            step = GrlTetrisStepTrace(
                action_id=action_id,
                action_label=env.ACTION_LOOKUP[action_id],
                reward=reward,
                done=session_state.done,
                info=info,
            )
            session_state.history.append(step)
            steps.append(step)

            if session_state.done:
                break

        return GrlTetrisStepResponse(
            observation=session_state.observation,
            reward=total_step_reward,
            total_reward=session_state.total_reward,
            done=session_state.done,
            steps=steps,
            history=list(session_state.history),
        )

    async def verify(self, request: Request, body: BaseVerifyRequest) -> GrlTetrisVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        session_state = self.session_id_to_state.get(session_id)

        success = False
        reward = 0.0
        if session_state is not None:
            success = bool(session_state.last_info.get("success"))
            reward = session_state.total_reward

        if session_id in self.session_id_to_state:
            try:
                session_state.env.close()  # type: ignore[union-attr]
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            del self.session_id_to_state[session_id]

        return GrlTetrisVerifyResponse(
            **body.model_dump(),
            reward=reward,
            success=success,
        )

    @staticmethod
    def _parse_action(action: Union[str, int], reverse_lookup: Dict[str, int]) -> int:
        if isinstance(action, int):
            return action

        candidate = action.strip()
        lower_candidate = candidate.lower()
        if lower_candidate in reverse_lookup:
            return reverse_lookup[lower_candidate]

        try:
            return int(candidate)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Unable to parse action: {action}") from exc

    @staticmethod
    def _to_python_types(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: GrlTetrisResourcesServer._to_python_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [GrlTetrisResourcesServer._to_python_types(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        return obj


if __name__ == "__main__":
    GrlTetrisResourcesServer.run_webserver()
