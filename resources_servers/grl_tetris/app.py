# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import HTTPException
from pydantic import Field

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.grl_tetris.tetris_env import TetrisEnv
from resources_servers.gymnasium import GymnasiumServer, extract_text


DEFAULT_GRID_LOOKUP = {0: "_", 1: "#", 2: "X"}
DEFAULT_ACTION_LOOKUP = {0: "Left", 1: "Right", 2: "Down"}
ACTION_TAG_PATTERN = re.compile(r"<action>\s*(left|right|down|[0-2])\s*</action>", re.IGNORECASE)


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


@dataclass
class TetrisSessionState:
    env: Any
    observation: str
    total_reward: float = 0.0
    done: bool = False
    last_info: Dict[str, Any] = field(default_factory=dict)


class GrlTetrisResourcesServer(GymnasiumServer):
    config: GrlTetrisResourcesServerConfig
    session_id_to_state: Dict[str, TetrisSessionState] = Field(default_factory=dict)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        if session_id is None:
            raise HTTPException(status_code=400, detail="Missing session id.")

        self._close_env(session_id)

        env = TetrisEnv(self._env_config_from_metadata(metadata))
        observation = env.reset(seed=metadata.get("seed"))
        self.session_id_to_state[session_id] = TetrisSessionState(env=env, observation=observation)
        return self._format_observation(observation), {}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        if session_id is None or session_id not in self.session_id_to_state:
            raise HTTPException(status_code=400, detail="Session not initialized. Call /reset first.")

        session_state = self.session_id_to_state[session_id]
        if session_state.done:
            return session_state.observation, 0.0, True, False, dict(session_state.last_info)

        env = session_state.env
        action_ids = self._parse_actions(action, env.ACTION_LOOKUP)
        if not action_ids:
            raise HTTPException(status_code=400, detail="No <action>Left|Right|Down</action> tags found in response.")

        step_reward = 0.0
        applied: List[Dict[str, Any]] = []
        for action_id in action_ids:
            next_obs, reward, done, info = env.step(action_id)
            info = self._to_python_types(info)
            step_reward += reward
            session_state.total_reward += reward
            session_state.observation = next_obs
            session_state.last_info = info
            applied.append(
                {
                    "action_id": action_id,
                    "action_label": env.ACTION_LOOKUP[action_id],
                    "reward": reward,
                    "info": info,
                }
            )
            if done:
                session_state.done = True
                break

        final_info = dict(session_state.last_info) | {
            "total_reward": session_state.total_reward,
            "num_actions_applied": len(applied),
            "actions": applied,
        }
        return (
            self._format_observation(session_state.observation)
            if not session_state.done
            else session_state.observation,
            step_reward,
            session_state.done,
            False,
            final_info,
        )

    def _env_config_from_metadata(self, metadata: dict) -> Dict[str, Any]:
        env_config = dict(self.config.env_config)
        for key in ("grid_lookup", "action_lookup", "render_mode", "dim_x", "dim_y", "box_type"):
            if key in metadata:
                env_config[key] = metadata[key]
        return env_config

    def _close_env(self, session_id: str) -> None:
        session_state = self.session_id_to_state.pop(session_id, None)
        if session_state is None:
            return
        try:
            session_state.env.close()
        except Exception:
            pass

    @staticmethod
    def _format_observation(observation: str) -> str:
        return (
            "Tetris board:\n"
            f"{observation}\n\n"
            "Legend: _=empty, #=settled block, X=falling piece.\n"
            "Respond with one or more moves using <action>Left</action>, <action>Right</action>, "
            "or <action>Down</action>. Multiple action tags in one turn are applied in order."
        )

    @staticmethod
    def _parse_actions(response: NeMoGymResponse, action_lookup: Dict[int, str]) -> List[int]:
        text = extract_text(response)
        reverse_lookup = {label.lower(): idx for idx, label in action_lookup.items()}
        action_ids: List[int] = []
        for match in ACTION_TAG_PATTERN.finditer(text):
            token = match.group(1).strip().lower()
            if token.isdigit():
                action_id = int(token)
            else:
                action_id = reverse_lookup.get(token, -1)
            if action_id in action_lookup:
                action_ids.append(action_id)
        return action_ids

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
