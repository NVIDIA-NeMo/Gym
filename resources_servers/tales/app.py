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

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import gymnasium as gym
from fastapi import HTTPException
from pydantic import Field

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.gymnasium import GymnasiumServer, extract_text


class TALESResourcesServerConfig(BaseResourcesServerConfig):
    expose_admissible_commands: bool = False
    framework: str = "alfworld"
    task_no: int = 0
    seed: int = 0
    split: str = "train"
    max_episode_steps: int = 25


@dataclass
class TALESSessionState:
    env: Any
    framework: str
    observation: str
    max_episode_steps: int
    last_score: float = 0.0
    total_score: float = 0.0
    step_count: int = 0
    done: bool = False
    last_info: Dict[str, Any] = field(default_factory=dict)


class TALESResourcesServer(GymnasiumServer):
    config: TALESResourcesServerConfig
    session_id_to_state: Dict[str, TALESSessionState] = Field(default_factory=dict)

    async def reset(self, metadata: dict, session_id: Optional[str] = None) -> tuple[Optional[str], dict]:
        if session_id is None:
            raise HTTPException(status_code=400, detail="Missing session id.")

        framework = self._resolve(metadata, "framework")
        task_no = self._resolve(metadata, "task_no")
        split = self._resolve(metadata, "split")
        seed = self._resolve(metadata, "seed")
        max_episode_steps = self._resolve(metadata, "max_episode_steps")

        try:
            framework_module = importlib.import_module(f"tales.{framework}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not load TALES framework '{framework}': {e!r}")

        if split == "train":
            envs = getattr(framework_module, "train_environments", None) or framework_module.environments
        else:
            envs = framework_module.environments
        if task_no < 0 or task_no >= len(envs):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task number {task_no} for framework '{framework}' (split '{split}'). "
                f"Choose 0..{len(envs) - 1}.",
            )

        self._close_env(session_id)

        task = envs[task_no]
        env_key = f"{task[0]}-{task[1]}"
        make_kwargs: dict[str, Any] = {"disable_env_checker": True, "admissible_commands": True}
        if framework == "scienceworld":
            make_kwargs["split"] = split
        try:
            env = gym.make(id=f"tales/{env_key}", **make_kwargs)
            obs, info = env.reset(seed=seed)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not launch TALES env 'tales/{env_key}': {e!r}")

        self.session_id_to_state[session_id] = TALESSessionState(
            env=env,
            framework=framework,
            observation=obs,
            max_episode_steps=max_episode_steps,
        )
        return obs, self._build_info(info)

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        if session_id is None or session_id not in self.session_id_to_state:
            raise HTTPException(status_code=400, detail="Session not initialized. Call /reset first.")

        state = self.session_id_to_state[session_id]
        if state.done:
            return state.observation, 0.0, True, False, dict(state.last_info)

        command = extract_text(action).strip()
        obs, score, done, info = state.env.step(command)

        if state.framework == "textworld":
            reward = float(score - state.last_score)
        else:
            reward = float(score)
        state.last_score = float(score)
        state.total_score += reward
        state.step_count += 1
        state.observation = obs
        state.done = bool(done)

        terminated = state.done
        truncated = (not terminated) and state.step_count >= state.max_episode_steps

        state.last_info = self._build_info(info) | {
            "score": score,
            "total_score": state.total_score,
            "step_count": state.step_count,
        }
        return obs, reward, terminated, truncated, dict(state.last_info)

    def _resolve(self, metadata: dict, key: str) -> Any:
        value = metadata.get(key)
        return value if value is not None else getattr(self.config, key)

    def _build_info(self, info: dict) -> dict:
        info = dict(info or {})
        if not self.config.expose_admissible_commands:
            info.pop("admissible_commands", None)
        return info

    def _close_env(self, session_id: str) -> None:
        state = self.session_id_to_state.pop(session_id, None)
        if state is None:
            return
        try:
            state.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    TALESResourcesServer.run_webserver()
