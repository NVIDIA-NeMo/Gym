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

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fastapi import HTTPException
from pydantic import Field

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.base_gymnasium import GymnasiumServer, extract_text
from resources_servers.grl_sokoban.sokoban_env import SokobanEnv


DEFAULT_GRID_LOOKUP = {0: "#", 1: "_", 2: "O", 3: "√", 4: "X", 5: "P", 6: "S"}
DEFAULT_ACTION_LOOKUP = {1: "Up", 2: "Down", 3: "Left", 4: "Right"}
ACTION_TAG_PATTERN = re.compile(r"<action>\s*(up|down|left|right|[1-4])\s*</action>", re.IGNORECASE)
ACTION_WORD_PATTERN = re.compile(r"\b(up|down|left|right|[1-4])\b", re.IGNORECASE)


class GrlSokobanResourcesServerConfig(BaseResourcesServerConfig):
    env_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "grid_lookup": DEFAULT_GRID_LOOKUP,
            "action_lookup": DEFAULT_ACTION_LOOKUP,
            "search_depth": 100,
            "dim_room": (6, 6),
            "max_steps": 100,
            "num_boxes": 1,
            "render_mode": "text",
        }
    )


@dataclass
class SokobanSessionState:
    env: Any
    observation: str
    total_reward: float = 0.0
    done: bool = False
    last_info: Dict[str, Any] = field(default_factory=dict)


class GrlSokobanResourcesServer(GymnasiumServer):
    config: GrlSokobanResourcesServerConfig
    session_id_to_state: Dict[str, SokobanSessionState] = Field(default_factory=dict)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        if session_id is None:
            raise HTTPException(status_code=400, detail="Missing session id.")

        self._close_env(session_id)

        env = SokobanEnv(self._env_config_from_metadata(metadata))
        observation = env.reset(seed=metadata.get("seed"))
        self.session_id_to_state[session_id] = SokobanSessionState(env=env, observation=observation)
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
        action_id = self._parse_action(action, env.ACTION_LOOKUP)
        next_obs, reward, done, info = env.step(action_id)

        session_state.total_reward += reward
        session_state.observation = next_obs
        session_state.last_info = info | {
            "action_id": action_id,
            "action_label": env.ACTION_LOOKUP[action_id],
            "total_reward": session_state.total_reward,
        }
        session_state.done = bool(done)

        return (
            self._format_observation(next_obs) if not session_state.done else next_obs,
            reward,
            session_state.done,
            False,
            dict(session_state.last_info),
        )

    def _env_config_from_metadata(self, metadata: dict) -> dict[str, Any]:
        env_config = dict(self.config.env_config)
        for key in (
            "grid_lookup",
            "action_lookup",
            "search_depth",
            "dim_room",
            "max_steps",
            "num_boxes",
            "render_mode",
        ):
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
            "Sokoban board:\n"
            f"{observation}\n\n"
            "Legend: #=wall, _=floor, O=target, X=box, √=box on target, P=player, S=player on target.\n"
            "Respond with exactly one move using <action>Up</action>, <action>Down</action>, "
            "<action>Left</action>, or <action>Right</action>."
        )

    @staticmethod
    def _parse_action(response: NeMoGymResponse, action_lookup: Dict[int, str]) -> int:
        text = extract_text(response).strip()
        match = ACTION_TAG_PATTERN.search(text) or ACTION_WORD_PATTERN.search(text)
        if match is None:
            raise HTTPException(status_code=400, detail=f"Unable to parse action from response: {text!r}")

        token = match.group(1).strip()
        if token.isdigit():
            action_id = int(token)
            if action_id in action_lookup:
                return action_id
            raise HTTPException(status_code=400, detail=f"Invalid action identifier: {action_id}")

        reverse_lookup = {label.lower(): idx for idx, label in action_lookup.items()}
        action_id = reverse_lookup.get(token.lower())
        if action_id is None:
            raise HTTPException(status_code=400, detail=f"Invalid action identifier: {token}")
        return action_id


GrlSokobanEnv = GrlSokobanResourcesServer


if __name__ == "__main__":
    GrlSokobanResourcesServer.run_webserver()
