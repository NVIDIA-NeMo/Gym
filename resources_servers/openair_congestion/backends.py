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

"""Backend abstraction for the openair_congestion resources server.

The gymnasium-style server in ``app.py`` talks to a :class:`Backend`, which
owns episode lifecycles:

    reset(task_params, live_episode_ids=...) -> (Observation, EpisodeMeta)
    step(episode_id, tool_call)              -> (Observation, reward, terminated, info)
    close(episode_id)                        -> summary dict

Two drivers implement the contract:

- :class:`ReplayBackend` ('replay', the default): offline and deterministic.
  Wraps the colocated ``openair_congestion.replay_env.ReplayEnv``; no 5G
  lab, no GPU, no KPI exporter, no wall-clock sleeps.
- ``DatasetReplayBackend`` ('dataset_replay', in ``dataset_backend.py``):
  offline replay of recorded KPI snapshots instead of seed-synthesized
  trajectories.
The unfinished live OAI collector is deliberately not selectable. Live
control belongs in a later contribution with its own lab evidence.

Selection is via :func:`select_backend`. The YAML ``backend:`` field is the
canonical switch; the ``OPENAIR_CONGESTION_BACKEND`` env var overrides it for
local development.

Episode slots are finite (``pool_size``) and normally free via close(). If a
rollout dies between /reset and its terminal /step the slot would leak, so
``reset()`` accepts ``live_episode_ids`` — the episode ids still owned by
live sessions — and backends reap orphaned episodes when the pool is
exhausted.

Rewards are not touched here: ``ReplayEnv.step()`` computes the per-step
reward internally via ``rewards.compute_breakdown()`` and this layer passes
its total through unchanged.
"""

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

from openair_congestion.replay_env import ReplayEnv, action_effect_version  # noqa: E402
from openair_congestion.reward_profiles import select_reward_profile  # noqa: E402
from openair_congestion.rewards import DEFAULT_WEIGHTS  # noqa: E402
from openair_congestion.schemas import EpisodeMeta, Observation, ToolCall  # noqa: E402


class Backend(ABC):
    """Episode-oriented environment driver behind the gymnasium server.

    ``task_params`` is the plain dict of scenario controls taken from the
    task row (seed / difficulty / regime_mix / scenario_id / tier /
    max_steps); keys map 1:1 onto ``ReplayEnv.reset()`` keyword arguments.
    """

    @abstractmethod
    def reset(
        self, task_params: dict[str, Any], *, live_episode_ids: Optional[set[str]] = None
    ) -> tuple[Observation, EpisodeMeta]:
        """Start a new episode; returns (first Observation, EpisodeMeta).

        ``meta.episode_id`` is the handle for subsequent step()/close() calls.
        ``live_episode_ids`` is the set of episode ids still owned by live
        sessions; backends may use it to reap orphaned episodes when their
        pool is exhausted.
        """

    @abstractmethod
    def step(self, episode_id: str, tool_call: ToolCall) -> tuple[Observation, float, bool, dict[str, Any]]:
        """Apply one action; returns (next_obs, reward, terminated, info).

        ``reward`` is the per-step total already computed inside the env
        (rewards.compute_breakdown), passed through unchanged. ``info``
        carries guardrail_accepted / rejection_reason / step_idx /
        reward_terms / reward_measurements / kpi_source / dynamics_mode.
        """

    @abstractmethod
    def close(self, episode_id: str) -> dict[str, Any]:
        """Release the episode slot; returns a summary like {ok, n_steps}."""

    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        """Describe whether actions causally affect served transitions."""

    def reward_contract(self, tier: str) -> dict[str, Any]:
        """Return the effective scoring configuration exposed to clients."""

        profile = select_reward_profile(tier)
        return {
            "reward_profile": profile.version,
            "reward_weights": asdict(DEFAULT_WEIGHTS),
            "prb_pressure_threshold": profile.prb_pressure_threshold,
        }


class ReplayBackend(Backend):
    """Offline deterministic driver wrapping ``ReplayEnv`` (the default).

    Standalone: no lab, no GPU, no exporter. One shared ReplayEnv instance
    manages all episodes by episode_id; it is internally locked (per-episode
    threading.RLock), so a single instance is safe across concurrent sessions.

    Leak safety: this backend tracks every episode id it creates. If
    ``ReplayEnv.reset()`` raises its pool-exhausted RuntimeError, episodes not
    referenced by any live session (``live_episode_ids``) are closed as leaked
    and the reset is retried exactly once.
    """

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
            "dynamics_mode": action_effect_version(),
            "action_affects_observation": True,
            "causal_action_effects": True,
            "training_usable": True,
            "diagnostic_only": False,
        }


def select_backend(config: Any) -> Backend:
    """Build the configured Backend (dependency injection point for app.py).

    Precedence: OPENAIR_CONGESTION_BACKEND env var > config.backend > 'replay'.
    ``config`` is duck-typed (the app's config object) so this module never
    imports app.py.
    """
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
