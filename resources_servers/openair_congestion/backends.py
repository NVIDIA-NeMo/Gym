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

"""Replay and recorded-data backends for OpenAir congestion episodes."""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Optional


# Import guard for an incomplete checkout: app.py and dataset_backend.py import
# this module first, so the diagnostic covers them too.
try:
    import openair_congestion  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only when unpackaged
    if exc.name != "openair_congestion":
        raise
    raise ImportError(
        "Could not import the colocated 'openair_congestion' domain package. "
        "Verify that resources_servers/openair_congestion/openair_congestion "
        "is present in this checkout."
    ) from exc

from openair_congestion.replay_env import ACTION_EFFECT_VERSION, ReplayEnv  # noqa: E402
from openair_congestion.rewards import (  # noqa: E402
    DEFAULT_WEIGHTS,
    PRB_PRESSURE_THRESHOLD,
    REWARD_VERSION,
)
from openair_congestion.schemas import EpisodeMeta, Observation, ToolCall  # noqa: E402


class Backend(ABC):
    @abstractmethod
    def reset(
        self, task_params: dict[str, Any], *, live_episode_ids: Optional[set[str]] = None
    ) -> tuple[Observation, EpisodeMeta]:
        """Start an episode, reaping allocations not owned by live sessions."""

    @abstractmethod
    def step(self, episode_id: str, tool_call: ToolCall) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Apply one action and return its transition."""

    @abstractmethod
    def close(self, episode_id: str) -> dict[str, Any]:
        """Release an episode slot."""

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Describe the backend's training guarantees."""

    def reward_contract(self, tier: str) -> dict[str, Any]:
        """Return the effective scoring configuration exposed to clients."""

        if tier != "replay":
            raise ValueError(f"tier {tier!r} is not supported by this contribution")
        return {
            "reward_profile": REWARD_VERSION,
            "reward_weights": asdict(DEFAULT_WEIGHTS),
            "prb_pressure_threshold": PRB_PRESSURE_THRESHOLD,
        }


class ReplayBackend(Backend):
    """Deterministic synthetic backend with orphaned-episode reclamation."""

    def __init__(
        self,
        *,
        pool_size: int = 32,
        max_steps_default: int = 60,
    ) -> None:
        self._env = ReplayEnv(
            pool_size=pool_size,
            max_steps_default=max_steps_default,
        )
        # Episode ids created here and not yet closed, for the leak reaper.
        self._open_episode_ids: set[str] = set()
        self._track_lock = threading.Lock()

    def reset(
        self, task_params: dict[str, Any], *, live_episode_ids: Optional[set[str]] = None
    ) -> tuple[Observation, EpisodeMeta]:
        try:
            first_obs, meta = self._reset_env(task_params)
        except RuntimeError as exc:
            if "pool exhausted" not in str(exc):
                raise
            # Reap episodes no session owns anymore (crashed rollouts), retry once.
            self._reap_leaked(live_episode_ids or set())
            first_obs, meta = self._reset_env(task_params)
        with self._track_lock:
            self._open_episode_ids.add(meta.episode_id)
        return first_obs, meta

    def _reset_env(self, task_params: dict[str, Any]) -> tuple[Observation, EpisodeMeta]:
        # Keys map 1:1 to ReplayEnv.reset() kwargs; defaults mirror the env's.
        return self._env.reset(
            seed=int(task_params.get("seed", 0)),
            difficulty=float(task_params.get("difficulty", 0.5)),
            regime_mix=task_params.get("regime_mix"),
            scenario_id=task_params.get("scenario_id"),
            tier=str(task_params.get("tier", "replay")),
            max_steps=task_params.get("max_steps"),
        )

    def _reap_leaked(self, live_episode_ids: set[str]) -> None:
        with self._track_lock:
            leaked = [eid for eid in self._open_episode_ids if eid not in live_episode_ids]
        for episode_id in leaked:
            try:
                self._env.close(episode_id)
            except KeyError:
                pass  # already gone inside the env
            with self._track_lock:
                self._open_episode_ids.discard(episode_id)

    def step(self, episode_id: str, tool_call: ToolCall) -> tuple[Observation, float, bool, dict[str, Any]]:
        return self._env.step(episode_id, tool_call)

    def close(self, episode_id: str) -> dict[str, Any]:
        try:
            summary = self._env.close(episode_id)
        except KeyError:
            with self._track_lock:
                self._open_episode_ids.discard(episode_id)
            raise
        with self._track_lock:
            self._open_episode_ids.discard(episode_id)
        return summary

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "replay",
            "dynamics_mode": ACTION_EFFECT_VERSION,
            "action_affects_observation": True,
            "causal_action_effects": True,
            "training_usable": True,
            "diagnostic_only": False,
        }


def select_backend(config: Any) -> Backend:
    """Build the configured backend, honoring the environment override."""
    name = os.environ.get("OPENAIR_CONGESTION_BACKEND") or getattr(config, "backend", None) or "replay"
    name = name.strip().lower()
    if name == "replay":
        return ReplayBackend(
            pool_size=getattr(config, "pool_size", 32),
            max_steps_default=getattr(config, "max_steps_default", 60),
        )
    if name == "dataset_replay":
        # Local import so the default replay path never pays for (or fails
        # on) ingestion code.
        from resources_servers.openair_congestion.dataset_backend import (
            DatasetReplayBackend,
        )

        return DatasetReplayBackend(
            dataset_path=getattr(config, "dataset_path", "data/fixtures/sample_provided.jsonl"),
            pool_size=getattr(config, "pool_size", 32),
            max_steps_default=getattr(config, "max_steps_default", 60),
            cell_capacity_mbps=getattr(config, "cell_capacity_mbps", 60.0),
            reward_weights=getattr(config, "reward_weights", None),
        )
    if name == "oai_collector":
        raise ValueError(
            "backend 'oai_collector' is not implemented in this contribution; "
            "supported backends are 'replay' and diagnostic-only 'dataset_replay'"
        )
    raise ValueError(f"unknown backend {name!r}; supported backends: 'replay', 'dataset_replay'")
