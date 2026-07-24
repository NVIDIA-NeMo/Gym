# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import pytest
from openair_congestion.replay_env import ReplayActionState, ReplayEnv, apply_action_effect
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

    assert _cell_delivery(low_interference_high_power.observation) > _cell_delivery(low_interference_noop.observation)
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
    accounting = result.info["service_accounting"]

    assert len(cell.ues) == baseline_count // 2
    assert cell.rrc_connected_ues == len(cell.ues)
    assert result.observation.global_.n_ues_total == sum(len(item.ues) for item in result.observation.cells)
    assert accounting["requested_service_mbps"] >= accounting["admitted_service_mbps"]
    assert accounting["unadmitted_service_mbps"] == pytest.approx(
        accounting["requested_service_mbps"] - accounting["admitted_service_mbps"]
    )
    assert accounting["step_forced_termination_events"] == baseline_count - len(cell.ues)
    assert accounting["step_forced_terminated_service_mbps"] > 0.0


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


def test_globally_numbered_ue_id_listed_in_cell_is_accepted():
    env = ReplayEnv(pool_size=1, max_steps_default=2)
    _, meta = env.reset(
        seed=7001,
        difficulty=0.8,
        regime_mix={"prb_exhaustion": 1.0},
        scenario_id="global-ue-ids",
        tier="T2",
        max_steps=2,
    )
    episode = env._episodes[meta.episode_id]
    remapped = []
    for observation in episode.trajectory:
        cells = []
        for cell in observation.cells:
            offset = cell.cell_id * 8
            cells.append(
                cell.model_copy(
                    update={"ues": [ue.model_copy(update={"ue_id": offset + ue.ue_id}) for ue in cell.ues]}
                )
            )
        remapped.append(observation.model_copy(update={"cells": cells}))
    episode.trajectory = remapped

    try:
        _, _, _, info = env.step(
            meta.episode_id,
            ToolCall(
                name="set_prb_cap",
                arguments={
                    "cell_id": 2,
                    "target": "ue",
                    "target_id": 19,
                    "max_prb": 200,
                },
            ),
        )
    finally:
        env.close(meta.episode_id)

    assert info["guardrail_accepted"] is True
