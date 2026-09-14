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

import asyncio
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
    framework: str = "textworld"
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
    highscore: float = 0.0
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

        await self._close_env(session_id)

        task = envs[task_no]
        env_key = f"{task[0]}-{task[1]}"
        make_kwargs: dict[str, Any] = {"disable_env_checker": True, "admissible_commands": True}
        if framework == "scienceworld":
            make_kwargs["split"] = split
        try:
            env = gym.make(id=f"tales/{env_key}", **make_kwargs)
            obs, info = await asyncio.to_thread(env.reset, seed=seed)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not launch TALES env 'tales/{env_key}': {e!r}")

        self.session_id_to_state[session_id] = TALESSessionState(
            env=env,
            framework=framework,
            observation=obs,
            max_episode_steps=max_episode_steps,
        )
        return obs, self._build_info(info) | {"framework": framework}

    async def step(
        self, action: NeMoGymResponse, metadata: dict, session_id: Optional[str] = None
    ) -> tuple[Optional[str], float, bool, bool, dict]:
        if session_id is None or session_id not in self.session_id_to_state:
            raise HTTPException(status_code=400, detail="Session not initialized. Call /reset first.")

        state = self.session_id_to_state[session_id]
        if state.done:
            return state.observation, 0.0, True, False, dict(state.last_info)

        command = extract_text(action).strip()
        obs, score, done, info = await asyncio.to_thread(state.env.step, command)

        if state.framework == "textworld":
            reward = float(score - state.last_score)
        else:
            reward = float(score)
        state.last_score = float(score)
        state.total_score += reward
        state.step_count += 1
        state.observation = obs
        state.done = bool(done)

        # TALES' upstream wrappers report the cumulative game score in info["score"] for every
        # framework (the paper's benchmark.py scores from it); the positional `score` is per-step
        # for scienceworld/textworld_express and cumulative elsewhere, so it cannot be used
        # cross-family. The paper metric is the running highscore normalized by max_score.
        game_score = float((info or {}).get("score", score))
        state.highscore = max(state.highscore, game_score)
        max_score = (info or {}).get("max_score")
        normalized_highscore = state.highscore / float(max_score) if max_score else None

        terminated = state.done
        truncated = (not terminated) and state.step_count >= state.max_episode_steps

        state.last_info = self._build_info(info) | {
            "framework": state.framework,
            "step_score": score,
            "game_score": game_score,
            "highscore": state.highscore,
            "normalized_highscore": normalized_highscore,
            "total_score": state.total_score,
            "step_count": state.step_count,
        }
        return obs, reward, terminated, truncated, dict(state.last_info)

    def compute_metrics(self, tasks: list[list[Dict[str, Any]]]) -> Dict[str, Any]:
        """Per-family normalized-highscore metrics matching the TALES paper's aggregation.

        Paper metric (benchmark.py): per episode, running highscore / max_score; mean over
        episodes of a game; mean over games of a framework; equal-weight macro-average across
        frameworks. Reported on the paper's 0-100 scale. A pooled mean over raw rewards is
        meaningless here (per-family reward scales are incomparable).
        """
        family_task_scores: Dict[str, list[float]] = {}
        family_task_wins: Dict[str, list[float]] = {}
        skipped = 0
        for task_rollouts in tasks:
            scores, wins, framework = [], [], None
            for rollout in task_rollouts:
                info = rollout.get("info") or {}
                normalized = info.get("normalized_highscore")
                if info.get("framework") is None or normalized is None:
                    skipped += 1
                    continue
                framework = info["framework"]
                scores.append(float(normalized) * 100.0)
                wins.append(1.0 if info.get("won") else 0.0)
            if framework is None:
                continue
            family_task_scores.setdefault(framework, []).append(sum(scores) / len(scores))
            family_task_wins.setdefault(framework, []).append(sum(wins) / len(wins))

        metrics: Dict[str, Any] = {}
        for framework in sorted(family_task_scores):
            task_scores = family_task_scores[framework]
            task_wins = family_task_wins[framework]
            metrics[f"tales/{framework}/normalized_highscore"] = sum(task_scores) / len(task_scores)
            metrics[f"tales/{framework}/success_rate"] = sum(task_wins) / len(task_wins)
        if family_task_scores:
            family_means = [metrics[f"tales/{fw}/normalized_highscore"] for fw in family_task_scores]
            family_win_means = [metrics[f"tales/{fw}/success_rate"] for fw in family_task_scores]
            metrics["tales/macro_avg/normalized_highscore"] = sum(family_means) / len(family_means)
            metrics["tales/macro_avg/success_rate"] = sum(family_win_means) / len(family_win_means)
            metrics["tales/num_families"] = len(family_task_scores)
        if skipped:
            metrics["tales/rollouts_missing_score_fields"] = skipped
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key_metrics = {k: v for k, v in agent_metrics.items() if k.startswith("tales/")}
        return key_metrics or {k: v for k, v in agent_metrics.items() if k.startswith("mean/")}

    def _resolve(self, metadata: dict, key: str) -> Any:
        value = metadata.get(key)
        return value if value is not None else getattr(self.config, key)

    def _build_info(self, info: dict) -> dict:
        info = dict(info or {})
        if not self.config.expose_admissible_commands:
            info.pop("admissible_commands", None)
        return info

    async def close_session(self, session_id: Optional[str]) -> None:
        await self._close_env(session_id)
        await super().close_session(session_id)

    async def _close_env(self, session_id: str) -> None:
        state = self.session_id_to_state.pop(session_id, None)
        if state is None:
            return
        try:
            await asyncio.to_thread(state.env.close)
        except Exception:
            pass


if __name__ == "__main__":
    TALESResourcesServer.run_webserver()
