# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decomposed congestion-control reward.

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

from dataclasses import dataclass

from .schemas import Observation, ToolCall


@dataclass(frozen=True)
class RewardWeights:
    """Per-step reward coefficients.

    Delta and level terms balance short-term improvement with persistent
    congestion cost. Values are part of the versioned ``openair_v1`` contract.
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


def _aggregate_delivered_mbps(obs: Observation) -> float:
    return float(sum(ue.delivered_mbps for c in obs.cells for ue in c.ues))


def _mean_jain(obs: Observation) -> float:
    if not obs.cells:
        return 1.0
    return float(sum(c.fairness_jain for c in obs.cells) / len(obs.cells))


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
) -> dict[str, dict[str, float] | float]:
    """Return raw KPI measurements, weighted reward terms, and total."""
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
    }
    if not rejected:
        terms["action"] = -weights.w_action * _action_l1_norm(action)
    if rejected:
        terms["reject"] = -weights.w_reject
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
) -> dict[str, float]:
    """Return per-term reward components for calibration diagnostics."""
    breakdown = compute_breakdown(
        prev_obs,
        curr_obs,
        action,
        rejected=rejected,
        weights=weights,
        cell_capacity_mbps=cell_capacity_mbps,
        buffer_capacity_kb=buffer_capacity_kb,
        prb_pressure_threshold=prb_pressure_threshold,
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
) -> float:
    """Compute the per-step reward. See the module formula."""
    terms = compute_terms(
        prev_obs,
        curr_obs,
        action,
        rejected=rejected,
        weights=weights,
        cell_capacity_mbps=cell_capacity_mbps,
        buffer_capacity_kb=buffer_capacity_kb,
        prb_pressure_threshold=prb_pressure_threshold,
    )
    return terms["total"]


__all__ = [
    "RewardWeights",
    "DEFAULT_WEIGHTS",
    "compute",
    "compute_breakdown",
    "compute_terms",
]
