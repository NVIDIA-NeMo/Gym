# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from openair_congestion.render import to_policy_text
from openair_congestion.replay_env import (
    ReplayActionState,
    ReplayEnv,
    apply_action_effect,
    build_trajectory,
)
from openair_congestion.schemas import Observation, ToolCall
from openair_congestion.tools import get_parameters_schema


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
    seed: int = 555,
    difficulty: float = 0.95,
):
    return env.reset(
        seed=seed,
        difficulty=difficulty,
        regime_mix=regime_mix or {"prb_exhaustion": 0.6, "interference": 0.4},
        scenario_id="action-semantics",
        tier=tier,
        max_steps=4,
    )


def _run_one(
    action: ToolCall,
    *,
    regime_mix: dict[str, float] | None = None,
    seed: int = 555,
    difficulty: float = 0.95,
) -> _StepResult:
    env = ReplayEnv(pool_size=2, max_steps_default=4)
    _, meta = _reset(
        env,
        regime_mix=regime_mix,
        seed=seed,
        difficulty=difficulty,
    )
    observation, reward, _, info = env.step(meta.episode_id, action)
    env.close(meta.episode_id)
    return _StepResult(observation=observation, reward=reward, info=info)


def _cell_payload(observation: Observation, cell_id: int = 0) -> dict:
    cell = next(cell for cell in observation.cells if cell.cell_id == cell_id)
    return cell.model_dump(by_alias=True)


def _cell_delivery(observation: Observation, cell_id: int = 0) -> float:
    cell = next(cell for cell in observation.cells if cell.cell_id == cell_id)
    return sum(float(ue.delivered_mbps) for ue in cell.ues)


def _jain(values: list[float]) -> float:
    if not values:
        return 1.0
    squared = sum(value * value for value in values)
    if squared <= 1e-12:
        return 1.0
    return (sum(values) ** 2) / (len(values) * squared)


def _assert_delivery_accounting(observation: Observation, *, cell_capacity_mbps: float) -> None:
    for cell in observation.cells:
        delivered = [float(ue.delivered_mbps) for ue in cell.ues]
        assert sum(delivered) <= cell_capacity_mbps + 1e-9
        assert cell.rrc_connected_ues == len(cell.ues)
        assert cell.fairness_jain == pytest.approx(_jain(delivered))
        assert cell.sla_violations_last_window == sum(int(ue.pdb_violations) for ue in cell.ues)
        for ue in cell.ues:
            assert 0.0 <= ue.delivered_mbps <= ue.offered_mbps
            assert ue.buffer_occupancy_kb == pytest.approx((ue.offered_mbps - ue.delivered_mbps) * 50.0)
            assert ue.pdb_violations == int(ue.buffer_occupancy_kb > 500.0)


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


def test_synthetic_replay_respects_shared_cell_capacity_and_derived_kpis():
    for seed in (0, 7, 493):
        for regime in (
            "prb_exhaustion",
            "bursty",
            "interference",
            "prach_storm",
            "qos_competition",
        ):
            observations, fingerprint = build_trajectory(
                seed=seed,
                difficulty=1.0,
                regime_mix={regime: 1.0},
                tier="replay",
                n_steps=8,
            )
            for observation in observations:
                _assert_delivery_accounting(
                    observation,
                    cell_capacity_mbps=fingerprint.cell_capacity_mbps,
                )


@pytest.mark.parametrize(
    "action",
    [
        pytest.param(ToolCall(name="noop", arguments={}), id="noop"),
        pytest.param(
            ToolCall(name="set_scheduler_policy", arguments={"cell_id": 0, "policy": "MaxCI"}),
            id="scheduler",
        ),
        pytest.param(
            ToolCall(
                name="set_prb_cap",
                arguments={"cell_id": 0, "target": "ue", "target_id": 0, "max_prb": 137},
            ),
            id="prb-cap",
        ),
        pytest.param(
            ToolCall(
                name="set_mcs_bounds",
                arguments={"cell_id": 0, "mcs_min": 0, "mcs_max": 14, "target_bler": 0.1},
            ),
            id="mcs",
        ),
        pytest.param(
            ToolCall(name="set_qos_weights", arguments={"cell_id": 0, "weights": {"1": 5.0, "9": 2.0}}),
            id="qos",
        ),
        pytest.param(
            ToolCall(
                name="set_admission_policy",
                arguments={"cell_id": 0, "accept_threshold_pct": 50.0, "slice_reservation": {}},
            ),
            id="admission",
        ),
        pytest.param(
            ToolCall(
                name="set_handover_trigger",
                arguments={"cell_id": 0, "a3_offset_db": -6.0, "ttt_ms": 160},
            ),
            id="handover",
        ),
        pytest.param(
            ToolCall(name="set_ul_power_control", arguments={"cell_id": 0, "p0_dbm": -80.0, "alpha": 0.8}),
            id="ul-power",
        ),
        pytest.param(
            ToolCall(
                name="set_prb_cap",
                arguments={"cell_id": 0, "target": "ue", "target_id": 0, "max_prb": 0},
            ),
            id="guardrail-rejection",
        ),
    ],
)
def test_synthetic_replay_actions_respect_shared_cell_capacity(action: ToolCall):
    env = ReplayEnv(pool_size=1, max_steps_default=4)
    _, meta = env.reset(
        seed=493,
        difficulty=1.0,
        regime_mix={"qos_competition": 1.0},
        scenario_id="delivery-bound",
        tier="replay",
        max_steps=4,
    )
    try:
        for step_idx in range(4):
            adjusted, _, done, info = env.step(
                meta.episode_id,
                action if step_idx == 0 else ToolCall(name="noop", arguments={}),
            )
            _assert_delivery_accounting(adjusted, cell_capacity_mbps=60.0)
            aggregate = sum(float(ue.delivered_mbps) for cell in adjusted.cells for ue in cell.ues)
            assert info["reward_measurements"]["aggregate_delivered_mbps"] == pytest.approx(aggregate)
            if done:
                break
    finally:
        env.close(meta.episode_id)


def test_equal_qos_weights_do_not_create_throughput():
    noop = _run_one(ToolCall(name="noop", arguments={}), regime_mix={"qos_competition": 1.0})
    equal_weights = _run_one(
        ToolCall(
            name="set_qos_weights",
            arguments={"cell_id": 0, "weights": {"1": 5.0, "9": 5.0}},
        ),
        regime_mix={"qos_competition": 1.0},
    )

    assert _cell_delivery(equal_weights.observation) == pytest.approx(_cell_delivery(noop.observation))


def test_scheduler_action_respects_delivery_accounting():
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
    _assert_delivery_accounting(adjusted, cell_capacity_mbps=60.0)


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
    assert _cell_delivery(low_interference_high_power.observation) <= 60.0
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


def test_mcs_bounds_constrain_independent_next_state_without_manufacturing_relief():
    observations, fingerprint = build_trajectory(
        seed=7001,
        difficulty=0.95,
        regime_mix={"interference": 1.0},
        tier="replay",
        n_steps=2,
    )
    previous_payload = observations[0].model_dump(by_alias=True)
    next_payload = observations[1].model_dump(by_alias=True)
    for ue in previous_payload["cells"][0]["ues"]:
        ue.update({"mcs_mean": 25.0, "bler": 0.1, "sinr_db": -20.0})
    for ue in next_payload["cells"][0]["ues"]:
        ue.update({"mcs_mean": 2.0, "bler": 0.1, "sinr_db": -20.0})
    previous = Observation.model_validate(previous_payload)
    base_next = Observation.model_validate(next_payload)

    bounded = apply_action_effect(
        prev_obs=previous,
        base_next_obs=base_next,
        action=ToolCall(
            name="set_mcs_bounds",
            arguments={"cell_id": 0, "mcs_min": 10, "mcs_max": 14, "target_bler": 0.1},
        ),
        state=ReplayActionState(),
        cell_capacity_mbps=fingerprint.cell_capacity_mbps,
    )
    noop = apply_action_effect(
        prev_obs=previous,
        base_next_obs=base_next,
        action=ToolCall(name="noop", arguments={}),
        state=ReplayActionState(),
        cell_capacity_mbps=fingerprint.cell_capacity_mbps,
    )

    bounded_cell = next(cell for cell in bounded.cells if cell.cell_id == 0)
    noop_cell = next(cell for cell in noop.cells if cell.cell_id == 0)
    assert all(10 <= ue.mcs_mean <= 14 for ue in bounded_cell.ues)
    assert sum(ue.bler for ue in bounded_cell.ues) > sum(ue.bler for ue in noop_cell.ues)
    assert _cell_delivery(bounded) < _cell_delivery(noop)


def test_admission_schema_exposes_only_replay_supported_neutral_policy():
    validator = Draft202012Validator(get_parameters_schema("set_admission_policy"))
    validator.validate(
        {
            "cell_id": 0,
            "accept_threshold_pct": 100.0,
            "slice_reservation": {},
        }
    )

    with pytest.raises(ValidationError):
        validator.validate(
            {
                "cell_id": 0,
                "accept_threshold_pct": 99.0,
                "slice_reservation": {},
            }
        )


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


def test_admission_noop_setpoint_preserves_emitted_topology():
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
                "accept_threshold_pct": 100.0,
                "slice_reservation": {},
            },
        )
    )
    cell = next(cell for cell in result.observation.cells if cell.cell_id == 0)

    assert result.info["guardrail_accepted"] is True
    assert len(cell.ues) == baseline_count
    assert cell.rrc_connected_ues == len(cell.ues)
    assert result.observation.global_.n_ues_total == sum(len(item.ues) for item in result.observation.cells)


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        pytest.param(
            ToolCall(
                name="set_admission_policy",
                arguments={
                    "cell_id": 0,
                    "accept_threshold_pct": 50.0,
                    "slice_reservation": {},
                },
            ),
            "denied-demand accounting",
            id="admission-reduction",
        ),
        pytest.param(
            ToolCall(
                name="set_prb_cap",
                arguments={
                    "cell_id": 0,
                    "target": "ue",
                    "target_id": 0,
                    "max_prb": 100,
                },
            ),
            "equal-share floor",
            id="below-fair-share-cap",
        ),
    ],
)
def test_synthetic_replay_rejects_unmodeled_traffic_shedding(action: ToolCall, reason: str):
    noop = _run_one(ToolCall(name="noop", arguments={}), regime_mix={"prb_exhaustion": 1.0})
    result = _run_one(action, regime_mix={"prb_exhaustion": 1.0})

    assert result.info["guardrail_accepted"] is False
    assert reason in result.info["rejection_reason"]
    assert result.reward < noop.reward
    assert result.observation.cells == noop.observation.cells
    assert result.observation.global_ == noop.observation.global_


def test_equal_share_prb_cap_uses_estimated_prb_consumption_not_delivery_fraction():
    noop = _run_one(ToolCall(name="noop", arguments={}), regime_mix={"prb_exhaustion": 1.0})
    capped = _run_one(
        ToolCall(
            name="set_prb_cap",
            arguments={
                "cell_id": 0,
                "target": "ue",
                "target_id": 0,
                "max_prb": 138,
            },
        ),
        regime_mix={"prb_exhaustion": 1.0},
    )

    noop_target = next(ue for ue in noop.observation.cells[0].ues if ue.ue_id == 0)
    capped_target = next(ue for ue in capped.observation.cells[0].ues if ue.ue_id == 0)
    assert capped.info["guardrail_accepted"] is True
    assert capped_target.delivered_mbps >= 0.95 * noop_target.delivered_mbps


def test_prb_cap_fails_closed_when_displaced_service_cannot_be_reassigned():
    task = {"qos_competition": 1.0}
    noop = _run_one(ToolCall(name="noop", arguments={}), regime_mix=task, seed=0)
    capped = _run_one(
        ToolCall(
            name="set_prb_cap",
            arguments={
                "cell_id": 0,
                "target": "ue",
                "target_id": 1,
                "max_prb": 137,
            },
        ),
        regime_mix=task,
        seed=0,
    )

    assert capped.info["guardrail_accepted"] is False
    assert "cannot reassign" in capped.info["rejection_reason"]
    assert capped.reward < noop.reward
    assert capped.observation.cells == noop.observation.cells
    assert capped.observation.global_ == noop.observation.global_


def test_rejected_unreassignable_prb_cap_is_not_persisted():
    task = {"qos_competition": 1.0}

    def run(first_action: ToolCall) -> tuple[Observation, dict]:
        env = ReplayEnv(pool_size=1, max_steps_default=4)
        _, meta = _reset(env, regime_mix=task, seed=0)
        _, _, _, first_info = env.step(meta.episode_id, first_action)
        observation, _, _, _ = env.step(meta.episode_id, ToolCall(name="noop", arguments={}))
        env.close(meta.episode_id)
        return observation, first_info

    capped, capped_info = run(
        ToolCall(
            name="set_prb_cap",
            arguments={
                "cell_id": 0,
                "target": "ue",
                "target_id": 1,
                "max_prb": 137,
            },
        )
    )
    control, _ = run(ToolCall(name="noop", arguments={}))

    assert capped_info["guardrail_accepted"] is False
    assert capped.cells == control.cells
    assert capped.global_ == control.global_


def test_persistent_prb_cap_suspends_when_later_headroom_disappears():
    base, _ = build_trajectory(
        seed=0,
        difficulty=0.95,
        regime_mix={"qos_competition": 1.0},
        tier="replay",
        n_steps=1,
    )

    def with_delivery(*, recipient_offered: float) -> Observation:
        payload = base[0].model_dump(by_alias=True)
        cell = payload["cells"][0]
        cell["prb_util_dl_p50"] = 1.0
        cell["prb_util_dl_p99"] = 1.0
        recipient, target = cell["ues"]
        recipient.update({"offered_mbps": recipient_offered, "delivered_mbps": 10.0})
        target.update({"offered_mbps": 40.0, "delivered_mbps": 40.0})
        return Observation.model_validate(payload)

    cap = ToolCall(
        name="set_prb_cap",
        arguments={"cell_id": 0, "target": "ue", "target_id": 1, "max_prb": 137},
    )
    state = ReplayActionState()
    first_base = with_delivery(recipient_offered=30.0)
    first = apply_action_effect(
        prev_obs=first_base,
        base_next_obs=first_base,
        action=cap,
        accepted=True,
        state=state,
        cell_capacity_mbps=60.0,
    )
    assert state.last_prb_cap_diagnostics[0]["effect_applied"] is True
    assert _cell_delivery(first) == pytest.approx(50.0)

    later_base = with_delivery(recipient_offered=10.0)
    later = apply_action_effect(
        prev_obs=first,
        base_next_obs=later_base,
        action=ToolCall(name="noop", arguments={}),
        accepted=True,
        state=state,
        cell_capacity_mbps=60.0,
    )
    assert state.last_prb_cap_diagnostics[0]["effect_applied"] is False
    assert _cell_delivery(later) == pytest.approx(50.0)


def test_rejected_admission_reduction_cannot_change_a_persistent_cap():
    def run(first_action: ToolCall) -> tuple[Observation, dict]:
        env = ReplayEnv(pool_size=1, max_steps_default=4)
        _, meta = env.reset(
            seed=493,
            difficulty=1.0,
            regime_mix={"qos_competition": 1.0},
            scenario_id="cap-admission-order",
            tier="replay",
            max_steps=4,
        )
        env.step(meta.episode_id, first_action)
        _, _, _, admission_info = env.step(
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
        observation, _, _, info = env.step(meta.episode_id, ToolCall(name="noop", arguments={}))
        env.close(meta.episode_id)
        assert admission_info["guardrail_accepted"] is False
        return observation, info

    capped, capped_info = run(
        ToolCall(
            name="set_prb_cap",
            arguments={
                "cell_id": 0,
                "target": "ue",
                "target_id": 1,
                "max_prb": 200,
            },
        )
    )
    control, _ = run(ToolCall(name="noop", arguments={}))

    assert capped.cells == control.cells
    assert capped.global_ == control.global_
    assert capped_info.get("prb_cap_dynamics", {})
