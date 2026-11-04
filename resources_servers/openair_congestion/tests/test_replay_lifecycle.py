# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import threading

import pytest
from openair_congestion import replay_env
from openair_congestion.replay_env import ReplayEnv
from openair_congestion.schemas import ToolCall


_ACTION = ToolCall(
    name="set_ul_power_control",
    arguments={"cell_id": 0, "p0_dbm": 10.0, "alpha": 1.0},
)


def _make_episode() -> tuple[ReplayEnv, str]:
    env = ReplayEnv(pool_size=1, max_steps_default=4)
    _, meta = env.reset(
        seed=555,
        difficulty=0.95,
        regime_mix={"prb_exhaustion": 1.0},
        scenario_id="lifecycle",
        tier="replay",
        max_steps=4,
    )
    return env, meta.episode_id


def _snapshot_episode(env: ReplayEnv, episode_id: str) -> dict:
    episode = env._episodes[episode_id]
    return {
        "step_idx": episode.step_idx,
        "last_action": copy.deepcopy(episode.last_action),
        "last_reward": episode.last_reward,
        "last_rejection": episode.last_rejection,
        "history": copy.deepcopy(episode.history),
        "action_state": copy.deepcopy(episode.action_state),
        "trajectory": [observation.model_dump(by_alias=True) for observation in episode.trajectory],
        "closed": episode.closed,
    }


def test_reward_exception_leaves_episode_state_unchanged(monkeypatch):
    env, episode_id = _make_episode()
    before = _snapshot_episode(env, episode_id)

    def _raise_reward_error(*args, **kwargs):
        raise RuntimeError("reward failure")

    monkeypatch.setattr(
        replay_env._rewards,
        "compute_breakdown",
        _raise_reward_error,
    )

    with pytest.raises(RuntimeError, match="reward failure"):
        env.step(episode_id, _ACTION)

    assert _snapshot_episode(env, episode_id) == before


def test_close_waits_for_inflight_step(monkeypatch):
    env, episode_id = _make_episode()
    reward_entered = threading.Event()
    release_reward = threading.Event()
    step_done = threading.Event()
    close_done = threading.Event()
    step_errors: list[BaseException] = []
    close_errors: list[BaseException] = []
    original = replay_env._rewards.compute_breakdown

    def _blocking_reward(*args, **kwargs):
        reward_entered.set()
        assert release_reward.wait(timeout=5.0)
        return original(*args, **kwargs)

    def _step():
        try:
            env.step(episode_id, _ACTION)
        except BaseException as exc:  # pragma: no cover - assertion reports it
            step_errors.append(exc)
        finally:
            step_done.set()

    def _close():
        try:
            env.close(episode_id)
        except BaseException as exc:  # pragma: no cover - assertion reports it
            close_errors.append(exc)
        finally:
            close_done.set()

    monkeypatch.setattr(
        replay_env._rewards,
        "compute_breakdown",
        _blocking_reward,
    )
    step_thread = threading.Thread(target=_step)
    close_thread = threading.Thread(target=_close)
    step_thread.start()
    assert reward_entered.wait(timeout=5.0)
    close_thread.start()

    assert not close_done.wait(timeout=0.1)
    release_reward.set()
    step_thread.join(timeout=5.0)
    close_thread.join(timeout=5.0)

    assert step_done.is_set()
    assert close_done.is_set()
    assert not step_errors
    assert not close_errors
    with pytest.raises(KeyError):
        env.step(episode_id, _ACTION)
