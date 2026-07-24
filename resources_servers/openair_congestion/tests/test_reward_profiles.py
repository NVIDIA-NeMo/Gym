# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from openair_congestion.replay_env import ReplayEnv
from openair_congestion.reward_profiles import (
    RewardProfile,
    select_reward_profile,
)
from openair_congestion.rewards import compute_breakdown
from openair_congestion.schemas import ToolCall


def _reset(env: ReplayEnv, *, tier: str):
    return env.reset(
        seed=555,
        difficulty=0.95,
        regime_mix={"prb_exhaustion": 1.0},
        scenario_id=f"reward-profile-{tier}",
        tier=tier,
        max_steps=2,
    )


def test_reward_profile_contract():
    assert select_reward_profile("T2") == RewardProfile(
        version="openair_t2_v3",
        prb_pressure_threshold=0.85,
    )
    assert select_reward_profile(
        "T1",
        connected_t1_runner=True,
    ) == RewardProfile(
        version="openair_v1",
        prb_pressure_threshold=0.08,
    )
    assert select_reward_profile("replay") == RewardProfile(
        version="openair_v1",
        prb_pressure_threshold=0.85,
    )


def test_t2_replay_reports_v3_service_terms():
    env = ReplayEnv(pool_size=1, max_steps_default=2)
    first, meta = _reset(env, tier="T2")
    episode = env._episodes[meta.episode_id]
    for cell in first.cells:
        for ue in cell.ues:
            key = (cell.cell_id, ue.ue_id)
            episode.action_state.service.requested_mbps[key] = 2.0 * ue.offered_mbps
            episode.action_state.service.admitted_mbps[key] = ue.offered_mbps

    _, _, _, info = env.step(
        meta.episode_id,
        ToolCall(name="noop", arguments={}),
    )
    env.close(meta.episode_id)

    assert info["reward_version"] == "openair_t2_v3"
    assert info["reward_terms"]["service_denial"] < 0.0
    assert info["service_accounting"]["unadmitted_service_mbps"] > 0.0


def test_v1_replay_matches_frozen_default_contract():
    env = ReplayEnv(pool_size=1, max_steps_default=2)
    first, meta = _reset(env, tier="replay")
    second, reward, _, info = env.step(
        meta.episode_id,
        ToolCall(name="noop", arguments={}),
    )
    env.close(meta.episode_id)
    expected = compute_breakdown(
        prev_obs=first,
        curr_obs=second,
        action=ToolCall(name="noop", arguments={}),
    )

    assert info["reward_version"] == "openair_v1"
    assert reward == pytest.approx(expected["total"])
    assert info["reward_terms"] == pytest.approx(expected["terms"])
