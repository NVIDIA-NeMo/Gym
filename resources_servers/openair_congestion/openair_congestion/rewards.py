# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reward function (``docs/PLAN.md`` §5.2).

::

    r = w_sla       * (-Δ sla_violations_last_window)
      + w_tput      * (Δ delivered_aggregate_mbps / cell_capacity)
      + w_fair      * (Δ jain_fairness)
      - w_sla_level * current_sla_violation_fraction
      - w_prb_level * current_prb_pressure
      - w_access_level * current_access_pressure
      - w_fair_level* current_fairness_deficit
      - w_buffer    * current_buffer_pressure
      - w_action    * action_l1_norm * 1{not guardrail_rejected}
      - w_reject    * 1{guardrail_rejected}

Delta terms are computed against the **previous** observation. Level terms
score the current observation so persistent improvements keep earning credit
by avoiding congestion penalties. The first step's reward uses zero-deltas
(``prev is None``), but level penalties can still fire if the episode starts
in a congested state.

The ``action_l1_norm`` term discourages the policy from churning every
step. We use a coarse "one knob changed" approximation: ``noop`` is 0,
any actuator is 1.0 (independent of how big the parameter swing was).
The intent is to penalise *any* change, not its magnitude — magnitudes
are hard to compare across heterogeneous tools (PRB count vs. dB).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import Observation, ToolCall


@dataclass(frozen=True)
class RewardWeights:
    """Per-step reward coefficients.

    Defaults were recalibrated in M7.3 against ``runner_snapshot`` T1
    distributions. Delta throughput/fairness terms need higher gain than the
    replay-era weights because live offered load is a small fraction of the
    nominal 250 Mbps cell-capacity label.
    """

    w_sla: float = 1.0
    w_tput: float = 2.0
    w_fair: float = 5.0
    w_buffer: float = 0.15
    w_sla_level: float = 0.8
    w_prb_level: float = 0.4
    w_access_level: float = 0.3
    w_fair_level: float = 0.35
    w_action: float = 0.0
    w_reject: float = 0.5


DEFAULT_WEIGHTS = RewardWeights()
T2_SERVICE_DENIAL_WEIGHT = 2.0
T2_FORCED_TERMINATION_WEIGHT = 5.0
T2_V3_DENIAL_WEIGHT = 1.0
T2_V3_DELIVERY_GAP_WEIGHT = 1.25
T2_V3_ELASTIC_FAIRNESS_WEIGHT = 0.25
T2_V3_SLA_WEIGHT = 2.0
T2_V3_FORCED_EVENT_WEIGHT = 5.0
T2_V3_FORCED_RATIO_WEIGHT = 2.0
T2_V3_ACTION_WEIGHT = 0.005


def _aggregate_delivered_mbps(obs: Observation) -> float:
    return float(sum(ue.delivered_mbps for c in obs.cells for ue in c.ues))


def _aggregate_offered_mbps(obs: Observation) -> float:
    return float(sum(ue.offered_mbps for c in obs.cells for ue in c.ues))


def _service_cost_measurements(
    curr_obs: Observation,
    service_accounting: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Normalize additive service accounting without affecting reward terms.

    These measurements are deliberately evaluation-only in ``openair_v1``.
    Adding a service-denial penalty would change the frozen T1 objective and
    therefore requires an explicitly versioned reward in a later change.
    """
    supplied = service_accounting or {}

    def metric(name: str, default: float) -> float:
        try:
            value = float(supplied.get(name, default))
        except (TypeError, ValueError):
            return max(0.0, default)
        if not math.isfinite(value):
            return max(0.0, default)
        return max(0.0, value)

    observed_offered = _aggregate_offered_mbps(curr_obs)
    observed_delivered = _aggregate_delivered_mbps(curr_obs)
    requested = metric("requested_service_mbps", observed_offered)
    admitted = metric("admitted_service_mbps", observed_offered)
    delivered = metric("delivered_service_mbps", observed_delivered)
    forced = metric("forced_terminated_service_mbps", 0.0)
    cumulative_forced = metric(
        "cumulative_forced_terminated_service_mbps",
        forced,
    )
    events = metric("forced_termination_events", 0.0)
    step_forced = metric("step_forced_terminated_service_mbps", 0.0)
    step_events = metric("step_forced_termination_events", 0.0)
    unadmitted = metric(
        "unadmitted_service_mbps",
        max(0.0, requested - admitted),
    )
    undelivered_admitted = metric(
        "undelivered_admitted_service_mbps",
        max(0.0, admitted - delivered),
    )
    return {
        "requested_service_mbps": requested,
        "admitted_service_mbps": admitted,
        "delivered_service_mbps": delivered,
        "forced_terminated_service_mbps": forced,
        "cumulative_forced_terminated_service_mbps": cumulative_forced,
        "forced_termination_events": events,
        "step_forced_terminated_service_mbps": step_forced,
        "step_forced_termination_events": step_events,
        "unadmitted_service_mbps": unadmitted,
        "undelivered_admitted_service_mbps": undelivered_admitted,
    }


def _mean_jain(obs: Observation) -> float:
    if not obs.cells:
        return 1.0
    return float(sum(c.fairness_jain for c in obs.cells) / len(obs.cells))


def _mean_elastic_jain(obs: Observation) -> float:
    """Mean per-cell Jain fairness over elastic (5QI-9) delivered service."""

    if not obs.cells:
        return 1.0
    values: list[float] = []
    for cell in obs.cells:
        delivered = [max(0.0, float(ue.delivered_mbps)) for ue in cell.ues if int(ue.qos_5qi) == 9]
        if not delivered or sum(delivered) <= 0.0:
            values.append(1.0)
            continue
        total = sum(delivered)
        squares = sum(value * value for value in delivered)
        values.append((total * total) / max(1e-9, len(delivered) * squares))
    return float(sum(values) / len(values))


def _sla_count(obs: Observation) -> int:
    return int(sum(c.sla_violations_last_window for c in obs.cells))


def _n_ues(obs: Observation) -> int:
    return max(1, sum(len(c.ues) for c in obs.cells))


def _mean_prb_pressure(obs: Observation, *, threshold: float = 0.85) -> float:
    if not obs.cells:
        return 0.0
    denom = max(1e-6, 1.0 - threshold)
    return float(sum(max(0.0, c.prb_util_dl_p99 - threshold) / denom for c in obs.cells) / len(obs.cells))


def _mean_access_pressure(obs: Observation, *, threshold: float = 0.05) -> float:
    if not obs.cells:
        return 0.0
    denom = max(1e-6, 0.5 - threshold)
    return float(sum(max(0.0, c.prach_collision_rate - threshold) / denom for c in obs.cells) / len(obs.cells))


def _mean_fairness_deficit(obs: Observation, *, target: float = 0.80) -> float:
    if not obs.cells:
        return 0.0
    denom = max(1e-6, target)
    return float(sum(max(0.0, target - c.fairness_jain) / denom for c in obs.cells) / len(obs.cells))


def _mean_buffer_pressure(obs: Observation, *, buffer_capacity_kb: float) -> float:
    ues = [ue for c in obs.cells for ue in c.ues]
    if not ues:
        return 0.0
    denom = max(1e-6, buffer_capacity_kb)
    return float(sum(max(0.0, (ue.buffer_occupancy_kb / denom) - 0.7) for ue in ues) / len(ues))


def _action_l1_norm(action: ToolCall) -> float:
    return 0.0 if action.name == "noop" else 1.0


def _delta_terms(
    prev_obs: Observation | None,
    curr_obs: Observation,
    *,
    rejected: bool,
) -> tuple[int, float, float]:
    if prev_obs is None:
        return 0, 0.0, 0.0
    d_sla = _sla_count(prev_obs) - _sla_count(curr_obs)
    d_tput = _aggregate_delivered_mbps(curr_obs) - _aggregate_delivered_mbps(prev_obs)
    d_fair = _mean_jain(curr_obs) - _mean_jain(prev_obs)
    if rejected:
        d_sla = min(0, d_sla)
        d_tput = min(0.0, d_tput)
        d_fair = min(0.0, d_fair)
    return d_sla, d_tput, d_fair


def compute_breakdown(
    prev_obs: Observation | None,
    curr_obs: Observation,
    action: ToolCall,
    *,
    rejected: bool = False,
    weights: RewardWeights = DEFAULT_WEIGHTS,
    cell_capacity_mbps: float = 60.0,
    buffer_capacity_kb: float = 1024.0,
    prb_pressure_threshold: float = 0.85,
    service_accounting: Mapping[str, Any] | None = None,
    reward_version: str = "openair_v1",
) -> dict[str, dict[str, float] | float]:
    """Return raw KPI measurements, weighted reward terms, and total.

    ``openair_v1`` keeps service accounting measurement-only for frozen T1.
    ``openair_t2_v2`` adds explicit service-denial/forced-termination costs.
    ``openair_t2_v3`` is an absolute-state objective: no transition deltas or
    uncontrollable PRB/PRACH proxies can create terminal-step exploits.
    """
    d_sla, d_tput, d_fair = _delta_terms(prev_obs, curr_obs, rejected=rejected)
    cell_capacity_total = max(1e-6, cell_capacity_mbps * max(1, curr_obs.global_.n_cells))
    n_ues = _n_ues(curr_obs)
    measurements: dict[str, float] = {
        "delta_sla_violations": float(d_sla),
        "delta_delivered_mbps": float(d_tput),
        "delta_jain_fairness": float(d_fair),
        "sla_violations": float(_sla_count(curr_obs)),
        "aggregate_delivered_mbps": float(_aggregate_delivered_mbps(curr_obs)),
        "mean_jain_fairness": float(_mean_jain(curr_obs)),
        "mean_elastic_jain_fairness": float(_mean_elastic_jain(curr_obs)),
        "prb_pressure": float(_mean_prb_pressure(curr_obs, threshold=prb_pressure_threshold)),
        "access_pressure": float(_mean_access_pressure(curr_obs)),
        "fairness_deficit": float(_mean_fairness_deficit(curr_obs)),
        "buffer_pressure": float(
            _mean_buffer_pressure(
                curr_obs,
                buffer_capacity_kb=buffer_capacity_kb,
            )
        ),
        "action_l1_norm": float(_action_l1_norm(action)),
        "cell_capacity_mbps_total": float(cell_capacity_total),
        "n_ues": float(n_ues),
    }
    measurements.update(_service_cost_measurements(curr_obs, service_accounting))
    terms: dict[str, float] = {
        "delta_sla": weights.w_sla * d_sla,
        "delta_tput": weights.w_tput * (d_tput / cell_capacity_total),
        "delta_fair": weights.w_fair * d_fair,
        "level_sla": -weights.w_sla_level * (_sla_count(curr_obs) / n_ues),
        "level_prb": -weights.w_prb_level * _mean_prb_pressure(curr_obs, threshold=prb_pressure_threshold),
        "level_access": -weights.w_access_level * _mean_access_pressure(curr_obs),
        "level_fair": -weights.w_fair_level * _mean_fairness_deficit(curr_obs),
        "level_buffer": -weights.w_buffer
        * _mean_buffer_pressure(
            curr_obs,
            buffer_capacity_kb=buffer_capacity_kb,
        ),
        "action": 0.0,
        "reject": 0.0,
        "service_denial": 0.0,
        "forced_termination": 0.0,
    }
    if not rejected:
        terms["action"] = -weights.w_action * _action_l1_norm(action)
    if rejected:
        terms["reject"] = -weights.w_reject
    if reward_version == "openair_t2_v2":
        requested = measurements["requested_service_mbps"]
        if requested > 0.0:
            denial_ratio = min(1.0, measurements["unadmitted_service_mbps"] / requested)
            forced_ratio = min(
                1.0,
                measurements["step_forced_terminated_service_mbps"] / requested,
            )
        else:
            denial_ratio = 0.0
            forced_ratio = 0.0
        terms["service_denial"] = -T2_SERVICE_DENIAL_WEIGHT * denial_ratio
        terms["forced_termination"] = -T2_FORCED_TERMINATION_WEIGHT * (
            measurements["step_forced_termination_events"] + forced_ratio
        )
    elif reward_version == "openair_t2_v3":
        requested = measurements["requested_service_mbps"]
        if requested > 0.0:
            denial_ratio = min(1.0, measurements["unadmitted_service_mbps"] / requested)
            delivery_gap_ratio = min(
                1.0,
                measurements["undelivered_admitted_service_mbps"] / requested,
            )
            forced_ratio = min(
                1.0,
                measurements["step_forced_terminated_service_mbps"] / requested,
            )
        else:
            denial_ratio = 0.0
            delivery_gap_ratio = 0.0
            forced_ratio = 0.0
        elastic_fairness_deficit = max(0.0, 1.0 - measurements["mean_elastic_jain_fairness"])
        # Replace every delta/proxy term with directly auditable service
        # utility. A persistent cap is rewarded or penalized on every step for
        # its persistent effect; acting on the terminal step has no windfall.
        terms.update(
            {
                "delta_sla": 0.0,
                "delta_tput": 0.0,
                "delta_fair": 0.0,
                "level_sla": -T2_V3_SLA_WEIGHT * (_sla_count(curr_obs) / n_ues),
                "level_prb": 0.0,
                "level_access": 0.0,
                "level_fair": 0.0,
                "level_buffer": 0.0,
                "action": (-T2_V3_ACTION_WEIGHT * _action_l1_norm(action) if not rejected else 0.0),
                "reject": -weights.w_reject if rejected else 0.0,
                "service_denial": -T2_V3_DENIAL_WEIGHT * denial_ratio,
                "delivery_gap": -T2_V3_DELIVERY_GAP_WEIGHT * delivery_gap_ratio,
                "elastic_fairness": (-T2_V3_ELASTIC_FAIRNESS_WEIGHT * elastic_fairness_deficit),
                "forced_termination": -(
                    T2_V3_FORCED_EVENT_WEIGHT * measurements["step_forced_termination_events"]
                    + T2_V3_FORCED_RATIO_WEIGHT * forced_ratio
                ),
            }
        )
    elif reward_version != "openair_v1":
        raise ValueError(f"unknown reward_version {reward_version!r}")
    total = float(sum(terms.values()))
    terms["total"] = total
    return {"measurements": measurements, "terms": terms, "total": total}


def compute_terms(
    prev_obs: Observation | None,
    curr_obs: Observation,
    action: ToolCall,
    *,
    rejected: bool = False,
    weights: RewardWeights = DEFAULT_WEIGHTS,
    cell_capacity_mbps: float = 60.0,
    buffer_capacity_kb: float = 1024.0,
    prb_pressure_threshold: float = 0.85,
    service_accounting: Mapping[str, Any] | None = None,
    reward_version: str = "openair_v1",
) -> dict[str, float]:
    """Return versioned per-term reward components for calibration diagnostics."""
    breakdown = compute_breakdown(
        prev_obs,
        curr_obs,
        action,
        rejected=rejected,
        weights=weights,
        cell_capacity_mbps=cell_capacity_mbps,
        buffer_capacity_kb=buffer_capacity_kb,
        prb_pressure_threshold=prb_pressure_threshold,
        service_accounting=service_accounting,
        reward_version=reward_version,
    )
    return breakdown["terms"]  # type: ignore[return-value]


def compute(
    prev_obs: Observation | None,
    curr_obs: Observation,
    action: ToolCall,
    *,
    rejected: bool = False,
    weights: RewardWeights = DEFAULT_WEIGHTS,
    cell_capacity_mbps: float = 60.0,
    buffer_capacity_kb: float = 1024.0,
    prb_pressure_threshold: float = 0.85,
    service_accounting: Mapping[str, Any] | None = None,
    reward_version: str = "openair_v1",
) -> float:
    """Compute the versioned per-step reward. See the module formula."""
    terms = compute_terms(
        prev_obs,
        curr_obs,
        action,
        rejected=rejected,
        weights=weights,
        cell_capacity_mbps=cell_capacity_mbps,
        buffer_capacity_kb=buffer_capacity_kb,
        prb_pressure_threshold=prb_pressure_threshold,
        service_accounting=service_accounting,
        reward_version=reward_version,
    )
    return terms["total"]


__all__ = [
    "RewardWeights",
    "DEFAULT_WEIGHTS",
    "compute",
    "compute_breakdown",
    "compute_terms",
]
