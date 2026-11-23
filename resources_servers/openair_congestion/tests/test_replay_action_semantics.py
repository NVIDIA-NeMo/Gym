# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import pytest
from openair_congestion.render import to_policy_text
from openair_congestion.replay_env import (
    ReplayActionState,
    ReplayEnv,
    apply_action_effect,
    build_trajectory,
)
from openair_congestion.schemas import Observation, ToolCall


@dataclass(frozen=True)
class _StepResult:
    observation: Observation
    reward: float
    info: dict


def _reset(
    env: ReplayEnv,
    *,
    regime_mix: dict[str, float] | None = None,
    tier: str = "replay",
):
    return env.reset(
        seed=555,
        difficulty=0.95,
        regime_mix=regime_mix or {"prb_exhaustion": 0.6, "interference": 0.4},
        scenario_id="action-semantics",
        tier=tier,
        max_steps=4,
    )


def _run_one(
    action: ToolCall,
    *,
    regime_mix: dict[str, float] | None = None,
) -> _StepResult:
    env = ReplayEnv(pool_size=2, max_steps_default=4)
    _, meta = _reset(env, regime_mix=regime_mix)
    observation, reward, _, info = env.step(meta.episode_id, action)
    env.close(meta.episode_id)
    return _StepResult(observation=observation, reward=reward, info=info)


def _cell_payload(observation: Observation, cell_id: int = 0) -> dict:
    cell = next(cell for cell in observation.cells if cell.cell_id == cell_id)
    return cell.model_dump(by_alias=True)


def _cell_delivery(observation: Observation, cell_id: int = 0) -> float:
    cell = next(cell for cell in observation.cells if cell.cell_id == cell_id)
    return sum(float(ue.delivered_mbps) for ue in cell.ues)


def test_self_contained_replay_examples_are_congested_and_regime_distinct():
    """The fallback used by a clean checkout must exercise all example regimes."""

    regimes = (
        "prb_exhaustion",
        "bursty",
        "interference",
        "prach_storm",
        "qos_competition",
    )
    first_policy_text: dict[str, str] = {}
    for regime in regimes:
        observations, _ = build_trajectory(
            seed=7001,
            difficulty=0.6,
            regime_mix={regime: 1.0},
            tier="replay",
            n_steps=4,
        )
        assert max(cell.prb_util_dl_p99 for observation in observations for cell in observation.cells) >= 0.85
        first_policy_text[regime] = to_policy_text(observations[0])

    assert len(set(first_policy_text.values())) == len(regimes)


def test_synthetic_replay_never_delivers_more_than_offered():
    observations, _ = build_trajectory(
        seed=0,
        difficulty=0.1,
        regime_mix={"prb_exhaustion": 1.0},
        tier="replay",
        n_steps=16,
    )

    for observation in observations:
        for cell in observation.cells:
            for ue in cell.ues:
                assert ue.delivered_mbps <= ue.offered_mbps

    env = ReplayEnv(pool_size=1, max_steps_default=4)
    _, meta = env.reset(
        seed=0,
        difficulty=0.1,
        regime_mix={"prb_exhaustion": 1.0},
        scenario_id="delivery-bound",
        tier="replay",
        max_steps=4,
    )
    adjusted, _, _, _ = env.step(
        meta.episode_id,
        ToolCall(
            name="set_scheduler_policy",
            arguments={"cell_id": 0, "policy": "MaxCI"},
        ),
    )
    env.close(meta.episode_id)
    for cell in adjusted.cells:
        for ue in cell.ues:
            assert ue.delivered_mbps <= ue.offered_mbps


def test_synthetic_replay_trajectory_respects_per_cell_capacity():
    observations, fingerprint = build_trajectory(
        seed=928,
        difficulty=1.0,
        regime_mix={"qos_competition": 1.0},
        tier="replay",
        n_steps=2,
    )

    for observation in observations:
        for cell in observation.cells:
            assert sum(float(ue.delivered_mbps) for ue in cell.ues) <= (fingerprint.cell_capacity_mbps + 1e-9)


def test_action_effect_caps_shared_cell_capacity_and_recomputes_derived_kpis():
    env = ReplayEnv(pool_size=1, max_steps_default=4)
    first_obs, meta = _reset(env)
    base_next = env._episodes[meta.episode_id].trajectory[1]

    adjusted = apply_action_effect(
        prev_obs=first_obs,
        base_next_obs=base_next,
        action=ToolCall(
            name="set_scheduler_policy",
            arguments={"cell_id": 0, "policy": "MaxCI"},
        ),
        state=ReplayActionState(),
        cell_capacity_mbps=1.0,
    )
    env.close(meta.episode_id)

    for cell in adjusted.cells:
        delivered = [float(ue.delivered_mbps) for ue in cell.ues]
        assert sum(delivered) <= 1.0 + 1e-9
        squared = sum(value * value for value in delivered)
        expected_fairness = 1.0 if squared <= 1e-12 else sum(delivered) ** 2 / (len(delivered) * squared)
        assert cell.fairness_jain == pytest.approx(expected_fairness)
        assert cell.rrc_connected_ues == len(cell.ues)
        assert cell.sla_violations_last_window == sum(int(ue.pdb_violations) for ue in cell.ues)
        for ue in cell.ues:
            assert ue.buffer_occupancy_kb == pytest.approx((ue.offered_mbps - ue.delivered_mbps) * 50.0)
            assert ue.pdb_violations == int(ue.buffer_occupancy_kb > 500.0)


def test_scheduler_policies_expose_real_tradeoffs():
    noop = _run_one(ToolCall(name="noop", arguments={}))
    rr = _run_one(
        ToolCall(
            name="set_scheduler_policy",
            arguments={"cell_id": 0, "policy": "RR"},
        )
    )
    max_ci = _run_one(
        ToolCall(
            name="set_scheduler_policy",
            arguments={"cell_id": 0, "policy": "MaxCI"},
        )
    )

    noop_cell = next(cell for cell in noop.observation.cells if cell.cell_id == 0)
    rr_cell = next(cell for cell in rr.observation.cells if cell.cell_id == 0)
    max_ci_cell = next(cell for cell in max_ci.observation.cells if cell.cell_id == 0)

    assert rr_cell.fairness_jain > noop_cell.fairness_jain
    assert rr_cell.sched_latency_ms_p99 > noop_cell.sched_latency_ms_p99
    assert _cell_delivery(rr.observation) < _cell_delivery(noop.observation)
    assert max_ci_cell.fairness_jain < noop_cell.fairness_jain
    assert max_ci_cell.sched_latency_ms_p99 < noop_cell.sched_latency_ms_p99
    assert _cell_delivery(max_ci.observation) > _cell_delivery(noop.observation)


def test_ul_power_extremes_produce_distinct_kpis_and_rewards():
    low = _run_one(
        ToolCall(
            name="set_ul_power_control",
            arguments={"cell_id": 0, "p0_dbm": -126.0, "alpha": 0.0},
        )
    )
    high = _run_one(
        ToolCall(
            name="set_ul_power_control",
            arguments={"cell_id": 0, "p0_dbm": 23.0, "alpha": 1.0},
        )
    )

    assert _cell_payload(low.observation) != _cell_payload(high.observation)
    assert low.reward != pytest.approx(high.reward)


def test_max_ul_power_is_not_an_unconditional_relief_action():
    high_power = ToolCall(
        name="set_ul_power_control",
        arguments={"cell_id": 0, "p0_dbm": 23.0, "alpha": 1.0},
    )
    low_interference = {"prb_exhaustion": 1.0}
    high_interference = {"interference": 1.0}

    low_interference_noop = _run_one(ToolCall(name="noop", arguments={}), regime_mix=low_interference)
    low_interference_high_power = _run_one(high_power, regime_mix=low_interference)
    high_interference_noop = _run_one(ToolCall(name="noop", arguments={}), regime_mix=high_interference)
    high_interference_high_power = _run_one(high_power, regime_mix=high_interference)

    low_noop_cell = next(cell for cell in low_interference_noop.observation.cells if cell.cell_id == 0)
    low_high_cell = next(cell for cell in low_interference_high_power.observation.cells if cell.cell_id == 0)
    assert _cell_delivery(low_interference_high_power.observation) == pytest.approx(
        _cell_delivery(low_interference_noop.observation)
    )
    assert sum(ue.sinr_db for ue in low_high_cell.ues) > sum(ue.sinr_db for ue in low_noop_cell.ues)
    assert sum(ue.bler for ue in low_high_cell.ues) < sum(ue.bler for ue in low_noop_cell.ues)
    assert _cell_delivery(high_interference_high_power.observation) < _cell_delivery(
        high_interference_noop.observation
    )


def test_handover_extremes_produce_distinct_kpis_and_rewards():
    aggressive = _run_one(
        ToolCall(
            name="set_handover_trigger",
            arguments={"cell_id": 0, "a3_offset_db": -24.0, "ttt_ms": 0},
        )
    )
    conservative = _run_one(
        ToolCall(
            name="set_handover_trigger",
            arguments={"cell_id": 0, "a3_offset_db": 24.0, "ttt_ms": 5120},
        )
    )

    assert _cell_payload(aggressive.observation) != _cell_payload(conservative.observation)
    assert aggressive.reward != pytest.approx(conservative.reward)


def test_aggressive_handover_is_conditioned_on_cell_edge_pressure():
    aggressive = ToolCall(
        name="set_handover_trigger",
        arguments={"cell_id": 0, "a3_offset_db": -24.0, "ttt_ms": 0},
    )
    low_pressure = {"prb_exhaustion": 1.0}
    high_pressure = {"interference": 1.0}

    low_noop = _run_one(ToolCall(name="noop", arguments={}), regime_mix=low_pressure)
    low_aggressive = _run_one(aggressive, regime_mix=low_pressure)
    high_noop = _run_one(ToolCall(name="noop", arguments={}), regime_mix=high_pressure)
    high_aggressive = _run_one(aggressive, regime_mix=high_pressure)

    assert _cell_delivery(low_aggressive.observation) < _cell_delivery(low_noop.observation)
    assert _cell_delivery(high_aggressive.observation) > _cell_delivery(high_noop.observation)


def test_high_mcs_is_not_rewarded_when_sinr_cannot_support_it():
    high_mcs = _run_one(
        ToolCall(
            name="set_mcs_bounds",
            arguments={
                "cell_id": 0,
                "mcs_min": 27,
                "mcs_max": 27,
                "target_bler": 0.1,
            },
        ),
        regime_mix={"interference": 1.0},
    )
    noop = _run_one(
        ToolCall(name="noop", arguments={}),
        regime_mix={"interference": 1.0},
    )

    assert _cell_delivery(high_mcs.observation) < _cell_delivery(noop.observation)


def test_reapplying_same_scheduler_setpoint_is_idempotent():
    env = ReplayEnv(pool_size=1, max_steps_default=4)
    first_obs, meta = _reset(env)
    episode = env._episodes[meta.episode_id]
    base_next = episode.trajectory[1]
    state = ReplayActionState()
    action = ToolCall(
        name="set_scheduler_policy",
        arguments={"cell_id": 0, "policy": "MaxCI"},
    )

    first = apply_action_effect(
        prev_obs=first_obs,
        base_next_obs=base_next,
        action=action,
        state=state,
        cell_capacity_mbps=episode.fingerprint.cell_capacity_mbps,
    )
    second = apply_action_effect(
        prev_obs=first_obs,
        base_next_obs=base_next,
        action=action,
        state=state,
        cell_capacity_mbps=episode.fingerprint.cell_capacity_mbps,
    )

    assert second.model_dump(by_alias=True) == first.model_dump(by_alias=True)


def test_default_pf_scheduler_setpoint_does_not_create_kpi_credit():
    env = ReplayEnv(pool_size=1, max_steps_default=4)
    first_obs, meta = _reset(env)
    episode = env._episodes[meta.episode_id]
    base_next = episode.trajectory[1]

    pf = apply_action_effect(
        prev_obs=first_obs,
        base_next_obs=base_next,
        action=ToolCall(
            name="set_scheduler_policy",
            arguments={"cell_id": 0, "policy": "PF"},
        ),
        state=ReplayActionState(),
        cell_capacity_mbps=episode.fingerprint.cell_capacity_mbps,
    )
    noop = apply_action_effect(
        prev_obs=first_obs,
        base_next_obs=base_next,
        action=ToolCall(name="noop", arguments={}),
        state=ReplayActionState(),
        cell_capacity_mbps=episode.fingerprint.cell_capacity_mbps,
    )
    env.close(meta.episode_id)

    assert pf.model_dump(by_alias=True) == noop.model_dump(by_alias=True)


def test_admission_ledger_matches_emitted_topology():
    baseline_env = ReplayEnv(pool_size=1, max_steps_default=4)
    baseline, baseline_meta = _reset(baseline_env)
    baseline_cell = next(cell for cell in baseline.cells if cell.cell_id == 0)
    baseline_count = len(baseline_cell.ues)
    baseline_env.close(baseline_meta.episode_id)

    result = _run_one(
        ToolCall(
            name="set_admission_policy",
            arguments={
                "cell_id": 0,
                "accept_threshold_pct": 50.0,
                "slice_reservation": {},
            },
        )
    )
    cell = next(cell for cell in result.observation.cells if cell.cell_id == 0)

    assert len(cell.ues) == baseline_count // 2
    assert cell.rrc_connected_ues == len(cell.ues)
    assert result.observation.global_.n_ues_total == sum(len(item.ues) for item in result.observation.cells)


def test_guardrail_uses_current_topology_after_admission_change():
    env = ReplayEnv(pool_size=1, max_steps_default=4)
    _, meta = _reset(env)
    env.step(
        meta.episode_id,
        ToolCall(
            name="set_admission_policy",
            arguments={
                "cell_id": 0,
                "accept_threshold_pct": 50.0,
                "slice_reservation": {},
            },
        ),
    )

    _, _, _, info = env.step(
        meta.episode_id,
        ToolCall(
            name="set_prb_cap",
            arguments={
                "cell_id": 0,
                "target": "ue",
                "target_id": 1,
                "max_prb": 100,
            },
        ),
    )
    env.close(meta.episode_id)

    assert info["guardrail_accepted"] is False
    assert "target_id=1 not present in cell 0" in info["rejection_reason"]
