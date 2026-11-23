# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic dynamics for ``openair_congestion_v1``.

The environment builds a seeded KPI trajectory at reset, then applies accepted
tool calls through a deterministic counterfactual model before computing the
next reward. It is fast enough for local training and evaluation, but it does
not model a physical RAN or issue live OpenAirInterface/FlexRIC controls.

Replay observations do not include UE-to-slice membership, so ``set_prb_cap``
supports UE targets only; unsupported targets are rejected explicitly.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import guardrail as _guardrail
from . import rewards as _rewards
from .env import (
    ObservationContext,
    _build_observation,
    _sample_scenario,
    _ScenarioFingerprint,
)
from .reward_profiles import select_reward_profile
from .schemas import EpisodeMeta, Observation, ToolCall
from .telemetry import KpiSnapshot
from .tools import P0_DBM_RANGE, PRB_MAX, TTT_MS_VALUES


LOG = logging.getLogger("openair_congestion.replay_env")


# --- Synthetic dynamics ----------------------------------------------------

ACTION_EFFECT_VERSION = "synthetic_action_effect_v6_shared_capacity"


@dataclass
class ReplayActionBias:
    """One independently replaceable action's KPI bias."""

    cell_biases: dict[tuple[int, str], float] = field(default_factory=dict)
    ue_biases: dict[tuple[int, int, str], float] = field(default_factory=dict)


@dataclass
class ReplayActionState(ReplayActionBias):
    """Persistent absolute control setpoints for synthetic replay."""

    prb_cap_setpoints: dict[tuple[int, str, int], int] = field(default_factory=dict)
    scheduler_policy: dict[int, str] = field(default_factory=dict)
    handover_setpoint: dict[int, tuple[float, int]] = field(default_factory=dict)
    ul_power_setpoint: dict[int, tuple[float, float]] = field(default_factory=dict)
    mcs_setpoint: dict[int, tuple[int, int, float]] = field(default_factory=dict)
    qos_weights: dict[int, dict[str, float]] = field(default_factory=dict)
    admission_threshold: dict[int, float] = field(default_factory=dict)
    last_prb_cap_diagnostics: dict[int, dict[str, Any]] = field(default_factory=dict)


def action_effect_version() -> str:
    """Return the replay action-effect model identifier for provenance."""

    return ACTION_EFFECT_VERSION


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _regime_weight(fingerprint: _ScenarioFingerprint, regime: str) -> float:
    return float((fingerprint.regime_mix or {}).get(regime, 0.0))


def _cell_payload(data: dict[str, Any], cell_id: int) -> dict[str, Any] | None:
    for cell in data.get("cells", []):
        if int(cell.get("cell_id", -1)) == cell_id:
            return cell
    return None


def _ue_payload(cell: dict[str, Any], ue_id: int) -> dict[str, Any] | None:
    for ue in cell.get("ues", []):
        if int(ue.get("ue_id", -1)) == ue_id:
            return ue
    return None


def _set_cell_delta(
    cell: dict[str, Any],
    field_name: str,
    delta: float,
    *,
    integer: bool = False,
) -> None:
    current = float(cell.get(field_name, 0.0))
    value = current + float(delta)
    if integer:
        cell[field_name] = int(round(value))
    else:
        cell[field_name] = value


def _set_ue_delta(
    ue: dict[str, Any],
    field_name: str,
    delta: float,
    *,
    integer: bool = False,
) -> None:
    current = float(ue.get(field_name, 0.0))
    value = current + float(delta)
    if integer:
        ue[field_name] = int(round(value))
    else:
        ue[field_name] = value


def _add_bias(mapping: dict[Any, float], key: Any, delta: float) -> None:
    mapping[key] = float(mapping.get(key, 0.0) + delta)


def _add_cell_bias(
    state: ReplayActionBias,
    cell_id: int,
    field_name: str,
    delta: float,
) -> None:
    _add_bias(state.cell_biases, (cell_id, field_name), delta)


def _add_ue_bias(
    state: ReplayActionBias,
    cell_id: int,
    ue_id: int,
    field_name: str,
    delta: float,
) -> None:
    _add_bias(state.ue_biases, (cell_id, ue_id, field_name), delta)


def _apply_action_biases(data: dict[str, Any], state: ReplayActionBias) -> None:
    cell_integer_fields = {"rrc_connected_ues", "sla_violations_last_window"}
    ue_integer_fields = {"pdb_violations"}

    for (cell_id, field_name), delta in state.cell_biases.items():
        cell = _cell_payload(data, cell_id)
        if cell is None:
            continue
        _set_cell_delta(
            cell,
            field_name,
            delta,
            integer=field_name in cell_integer_fields,
        )

    for (cell_id, ue_id, field_name), delta in state.ue_biases.items():
        cell = _cell_payload(data, cell_id)
        if cell is None:
            continue
        ue = _ue_payload(cell, ue_id)
        if ue is None:
            continue
        _set_ue_delta(
            ue,
            field_name,
            delta,
            integer=field_name in ue_integer_fields,
        )


def _jain_fairness(values: list[float]) -> float:
    if not values:
        return 1.0
    total = sum(values)
    squared = sum(value * value for value in values)
    if squared <= 1e-12:
        return 1.0
    return _clip((total * total) / (len(values) * squared), 0.0, 1.0)


def _waterfill_recipient_headroom(
    *,
    remaining: float,
    headroom: dict[int, float],
) -> dict[int, float]:
    """Deterministically redistribute no more than shed demand/headroom."""

    boosts = {ue_id: 0.0 for ue_id in headroom}
    active = {ue_id: max(0.0, float(value)) for ue_id, value in headroom.items() if value > 1e-12}
    remaining = max(0.0, float(remaining))
    while remaining > 1e-12 and active:
        share = remaining / len(active)
        distributed = 0.0
        exhausted: list[int] = []
        for ue_id in sorted(active):
            amount = min(active[ue_id], share)
            boosts[ue_id] += amount
            active[ue_id] -= amount
            distributed += amount
            if active[ue_id] <= 1e-12:
                exhausted.append(ue_id)
        if distributed <= 1e-12:
            break
        remaining -= distributed
        for ue_id in exhausted:
            del active[ue_id]
    return boosts


def _apply_collective_prb_caps(
    data: dict[str, Any],
    state: ReplayActionState,
    *,
    cell_capacity_mbps: float,
) -> None:
    """Apply all persistent caps jointly without manufacturing throughput."""

    state.last_prb_cap_diagnostics = {}
    capacity = max(1e-6, float(cell_capacity_mbps))
    for cell in data.get("cells", []):
        cell_id = int(cell.get("cell_id", -1))
        setpoints = {
            target_id: max_prb
            for (cap_cell, target, target_id), max_prb in state.prb_cap_setpoints.items()
            if cap_cell == cell_id and target == "ue" and max_prb < PRB_MAX
        }
        if not setpoints:
            continue
        ues = {int(ue.get("ue_id", -1)): ue for ue in cell.get("ues", [])}
        active = {ue_id: int(max_prb) for ue_id, max_prb in setpoints.items() if ue_id in ues}
        if not active:
            continue

        base_delivered = {ue_id: max(0.0, float(ue.get("delivered_mbps", 0.0))) for ue_id, ue in ues.items()}
        offered = {ue_id: max(0.0, float(ue.get("offered_mbps", 0.0))) for ue_id, ue in ues.items()}
        # A cap may never increase a target above the base or its offered load.
        final = {ue_id: min(base_delivered[ue_id], offered[ue_id]) for ue_id in ues}
        target_shed: dict[int, float] = {}
        for ue_id, max_prb in active.items():
            before = final[ue_id]
            after = before * (float(max_prb) / float(PRB_MAX))
            target_shed[ue_id] = max(0.0, before - after)
            final[ue_id] = after

        total_shed = sum(target_shed.values())
        base_total = sum(base_delivered.values())
        cell_total_limit = min(base_total, capacity)
        current_total = sum(final.values())
        # Synthetic noise should already respect capacity, but fail safe by
        # scaling down rather than ever publishing created/over-cap delivery.
        if current_total > cell_total_limit + 1e-12:
            scale = cell_total_limit / current_total
            final = {ue_id: value * scale for ue_id, value in final.items()}
            current_total = sum(final.values())

        recipient_headroom = {ue_id: max(0.0, offered[ue_id] - final[ue_id]) for ue_id in ues if ue_id not in active}
        redistributable = min(
            total_shed,
            sum(recipient_headroom.values()),
            max(0.0, cell_total_limit - current_total),
        )
        recipient_boosts = _waterfill_recipient_headroom(
            remaining=redistributable,
            headroom=recipient_headroom,
        )
        for ue_id, boost in recipient_boosts.items():
            final[ue_id] += boost
        redistributed = sum(recipient_boosts.values())
        final_total = sum(final.values())

        if redistributed > total_shed + 1e-9:
            raise RuntimeError("PRB cap dynamics redistributed more than shed")
        if any(boost > recipient_headroom[ue_id] + 1e-9 for ue_id, boost in recipient_boosts.items()):
            raise RuntimeError("PRB cap dynamics exceeded recipient headroom")
        if final_total > base_total + 1e-9 or final_total > capacity + 1e-9:
            raise RuntimeError("PRB cap dynamics created or exceeded capacity")

        for ue_id, ue in ues.items():
            ue["delivered_mbps"] = final[ue_id]
            ue["buffer_occupancy_kb"] = max(
                0.0,
                (offered[ue_id] - final[ue_id]) * 50.0,
            )
        cell["fairness_jain"] = _jain_fairness(list(final.values()))
        unredistributed = max(0.0, total_shed - redistributed)
        # PRB pressure falls only when capped throughput is not reassigned.
        relief = unredistributed / capacity
        cell["prb_util_dl_p50"] = float(cell.get("prb_util_dl_p50", 0.0)) - relief
        cell["prb_util_dl_p99"] = float(cell.get("prb_util_dl_p99", 0.0)) - relief
        state.last_prb_cap_diagnostics[cell_id] = {
            "active_setpoints": dict(sorted(active.items())),
            "base_delivered_mbps": base_total,
            "cell_capacity_mbps": capacity,
            "target_shed_mbps": total_shed,
            "recipient_boost_mbps": redistributed,
            "recipient_headroom_mbps": sum(recipient_headroom.values()),
            "unredistributed_shed_mbps": unredistributed,
            "final_delivered_mbps": final_total,
        }


def _apply_state_biases(
    data: dict[str, Any],
    state: ReplayActionState,
    *,
    cell_capacity_mbps: float,
) -> None:
    _apply_action_biases(data, state)
    _apply_collective_prb_caps(
        data,
        state,
        cell_capacity_mbps=cell_capacity_mbps,
    )


def _synthetic_replay_guardrail(
    action: ToolCall,
    result: _guardrail.GuardrailResult,
) -> _guardrail.GuardrailResult:
    """Reject action shapes the synthetic replay cannot model truthfully."""

    if result.accepted and action.name == "set_prb_cap" and action.arguments.get("target") != "ue":
        return _guardrail.GuardrailResult(
            accepted=False,
            reason=(
                "synthetic replay supports only UE-targeted set_prb_cap; "
                "slice membership is not present in replay observations"
            ),
        )
    if result.accepted and action.name == "set_admission_policy" and bool(action.arguments.get("slice_reservation")):
        return _guardrail.GuardrailResult(
            accepted=False,
            reason=(
                "synthetic replay cannot apply slice_reservation because "
                "slice membership is not present in replay observations"
            ),
        )
    return result


def _clamp_observation_payload(
    data: dict[str, Any],
    *,
    cell_capacity_mbps: float,
) -> dict[str, Any]:
    """Clamp fields and enforce the synthetic per-cell throughput capacity."""

    capacity = max(1e-6, float(cell_capacity_mbps))
    for cell in data.get("cells", []):
        cell["prb_util_dl_p50"] = _clip(float(cell.get("prb_util_dl_p50", 0.0)), 0.0, 1.0)
        cell["prb_util_dl_p99"] = _clip(float(cell.get("prb_util_dl_p99", 0.0)), 0.0, 1.0)
        cell["prb_util_dl_p99"] = max(
            float(cell["prb_util_dl_p50"]),
            float(cell["prb_util_dl_p99"]),
        )
        cell["prb_util_ul_p50"] = _clip(float(cell.get("prb_util_ul_p50", 0.0)), 0.0, 1.0)
        cell["sched_latency_ms_p99"] = max(0.0, float(cell.get("sched_latency_ms_p99", 0.0)))
        cell["rrc_connected_ues"] = int(max(0, min(24, round(float(cell.get("rrc_connected_ues", 0))))))
        cell["prach_collision_rate"] = _clip(
            float(cell.get("prach_collision_rate", 0.0)),
            0.0,
            1.0,
        )
        cell["fairness_jain"] = _clip(float(cell.get("fairness_jain", 1.0)), 0.0, 1.0)
        ues = cell.get("ues", [])
        sla_count = 0
        for ue in ues:
            ue["offered_mbps"] = max(0.0, float(ue.get("offered_mbps", 0.0)))
            ue["delivered_mbps"] = _clip(
                float(ue.get("delivered_mbps", 0.0)),
                0.0,
                ue["offered_mbps"],
            )
            ue["bler"] = _clip(float(ue.get("bler", 0.0)), 0.0, 1.0)
            ue["mcs_mean"] = _clip(float(ue.get("mcs_mean", 0.0)), 0.0, 27.0)
            ue["sinr_db"] = _clip(float(ue.get("sinr_db", 0.0)), -20.0, 40.0)
            ue["buffer_occupancy_kb"] = max(
                0.0,
                float(ue.get("buffer_occupancy_kb", 0.0)),
            )
            ue["pdb_violations"] = 1 if ue["buffer_occupancy_kb"] > 500.0 else 0
            sla_count += int(ue["pdb_violations"])
            ue["5qi"] = int(max(1, min(127, round(float(ue.get("5qi", 9))))))

        delivered_total = sum(float(ue["delivered_mbps"]) for ue in ues)
        if delivered_total > capacity + 1e-12:
            scale = capacity / delivered_total
            delivered_values: list[float] = []
            sla_count = 0
            for ue in ues:
                ue["delivered_mbps"] = float(ue["delivered_mbps"]) * scale
                delivered = float(ue["delivered_mbps"])
                delivered_values.append(delivered)
                ue["buffer_occupancy_kb"] = max(
                    0.0,
                    (float(ue["offered_mbps"]) - delivered) * 50.0,
                )
                ue["pdb_violations"] = 1 if ue["buffer_occupancy_kb"] > 500.0 else 0
                sla_count += int(ue["pdb_violations"])
            cell["fairness_jain"] = _jain_fairness(delivered_values)
        cell["sla_violations_last_window"] = sla_count

    return data


def _prev_cell(obs: Observation, cell_id: int):
    return next((c for c in obs.cells if c.cell_id == cell_id), None)


def _apply_admission_setpoints(
    data: dict[str, Any],
    state: ReplayActionState,
) -> None:
    for cell in data.get("cells", []):
        cell_id = int(cell.get("cell_id", -1))
        threshold = state.admission_threshold.get(cell_id)
        ues = sorted(
            cell.get("ues", []),
            key=lambda ue: int(ue.get("ue_id", -1)),
        )
        cell["rrc_connected_ues"] = len(ues)
        if threshold is None:
            continue
        admission_ratio = _clip(float(threshold) / 100.0, 0.0, 1.0)
        admitted_count = min(len(ues), max(1, int(len(ues) * admission_ratio))) if ues and admission_ratio > 0.0 else 0
        retained = ues[:admitted_count]
        cell["ues"] = retained
        cell["rrc_connected_ues"] = len(retained)

    global_data = data.get("global")
    if isinstance(global_data, dict):
        global_data["n_ues_total"] = sum(len(cell.get("ues", [])) for cell in data.get("cells", []))


def _rebuild_action_biases(
    *,
    state: ReplayActionState,
    prev_obs: Observation,
) -> None:
    """Derive KPI biases from absolute setpoints without accumulation."""

    state.cell_biases.clear()
    state.ue_biases.clear()

    for cell_id, policy in state.scheduler_policy.items():
        prev_cell = _prev_cell(prev_obs, cell_id)
        if prev_cell is None:
            continue
        if policy == "PF":
            # PF is the baseline scheduler represented by the synthetic
            # trajectory, so selecting it must not manufacture KPI credit.
            continue
        elif policy == "MaxCI":
            _add_cell_bias(state, cell_id, "fairness_jain", -0.08)
            _add_cell_bias(state, cell_id, "sched_latency_ms_p99", -1.0)
            if prev_cell.ues:
                strongest = max(prev_cell.ues, key=lambda ue: ue.sinr_db)
                for ue in prev_cell.ues:
                    delivery_delta = 0.8 if ue.ue_id == strongest.ue_id else -0.15
                    _add_ue_bias(
                        state,
                        cell_id,
                        ue.ue_id,
                        "delivered_mbps",
                        delivery_delta,
                    )
                    _add_ue_bias(
                        state,
                        cell_id,
                        ue.ue_id,
                        "buffer_occupancy_kb",
                        -50.0 * delivery_delta,
                    )
        else:
            _add_cell_bias(state, cell_id, "fairness_jain", +0.05)
            _add_cell_bias(state, cell_id, "sched_latency_ms_p99", +1.5)
            for ue in prev_cell.ues:
                _add_ue_bias(state, cell_id, ue.ue_id, "delivered_mbps", -0.15)
                _add_ue_bias(state, cell_id, ue.ue_id, "buffer_occupancy_kb", +7.5)

    for cell_id, weights in state.qos_weights.items():
        prev_cell = _prev_cell(prev_obs, cell_id)
        if prev_cell is None:
            continue
        strengths: list[float] = []
        for ue in prev_cell.ues:
            strength = float(weights.get(str(ue.qos_5qi), 1.0)) - 1.0
            strengths.append(strength)
            _add_ue_bias(state, cell_id, ue.ue_id, "delivered_mbps", strength)
            _add_ue_bias(state, cell_id, ue.ue_id, "buffer_occupancy_kb", -500.0 * strength)
        if strengths:
            _add_cell_bias(
                state,
                cell_id,
                "fairness_jain",
                0.03 * (sum(strengths) / len(strengths)),
            )

    for cell_id, (mcs_min, mcs_max, target_bler) in state.mcs_setpoint.items():
        prev_cell = _prev_cell(prev_obs, cell_id)
        if prev_cell is None:
            continue
        for ue in prev_cell.ues:
            current_mcs = float(ue.mcs_mean)
            # Outer-loop link adaptation lowers MCS when observed BLER exceeds
            # the requested target and permits a raise only when the link has
            # explicit SINR headroom. A forced minimum above that headroom is
            # therefore harmful instead of a free throughput increase.
            target_mcs = current_mcs - 8.0 * (float(ue.bler) - float(target_bler))
            bounded_mcs = _clip(target_mcs, float(mcs_min), float(mcs_max))
            mcs_delta = bounded_mcs - current_mcs
            supported_mcs = _clip(1.2 * (float(ue.sinr_db) + 5.0) + 2.0, 0.0, 27.0)
            safe_raise = min(max(0.0, mcs_delta), max(0.0, supported_mcs - current_mcs))
            unsafe_raise = max(0.0, mcs_delta - safe_raise)
            lower = max(0.0, -mcs_delta)
            delivery_delta = 0.08 * safe_raise - 0.20 * unsafe_raise - 0.04 * lower
            bler_delta = 0.004 * safe_raise + 0.020 * unsafe_raise - 0.008 * lower
            _add_ue_bias(state, cell_id, ue.ue_id, "mcs_mean", mcs_delta)
            _add_ue_bias(state, cell_id, ue.ue_id, "bler", bler_delta)
            _add_ue_bias(
                state,
                cell_id,
                ue.ue_id,
                "delivered_mbps",
                delivery_delta,
            )

    for cell_id, (a3_offset_db, ttt_ms) in state.handover_setpoint.items():
        prev_cell = _prev_cell(prev_obs, cell_id)
        offset_aggression = (24.0 - _clip(a3_offset_db, -24.0, 24.0)) / 48.0
        ttt_index = TTT_MS_VALUES.index(ttt_ms)
        ttt_aggression = 1.0 - (ttt_index / max(1, len(TTT_MS_VALUES) - 1))
        aggressiveness = 2.0 * ((offset_aggression + ttt_aggression) / 2.0 - 0.5)
        if prev_cell is not None and prev_cell.ues:
            sinr_pressure = sum(_clip((5.0 - float(ue.sinr_db)) / 10.0, 0.0, 1.0) for ue in prev_cell.ues) / len(
                prev_cell.ues
            )
            bler_pressure = sum(_clip((float(ue.bler) - 0.10) / 0.25, 0.0, 1.0) for ue in prev_cell.ues) / len(
                prev_cell.ues
            )
            cell_edge_pressure = max(sinr_pressure, bler_pressure)
            # Aggressive handover is useful at the cell edge but creates churn
            # when radio conditions are already healthy. Conservative settings
            # have the inverse trade-off.
            effect = aggressiveness * _clip(
                4.0 * (cell_edge_pressure - 0.25),
                -1.0,
                1.0,
            )
            worst = max(prev_cell.ues, key=lambda u: u.buffer_occupancy_kb)
            _add_ue_bias(state, cell_id, worst.ue_id, "delivered_mbps", 1.2 * effect)
            _add_ue_bias(state, cell_id, worst.ue_id, "buffer_occupancy_kb", -250.0 * effect)
            _add_cell_bias(state, cell_id, "fairness_jain", 0.05 * effect)
            _add_cell_bias(state, cell_id, "prb_util_dl_p99", -0.02 * effect)

    for cell_id, (p0_dbm, alpha) in state.ul_power_setpoint.items():
        prev_cell = _prev_cell(prev_obs, cell_id)
        if prev_cell is None:
            continue
        p0_norm = (_clip(float(p0_dbm), P0_DBM_RANGE[0], P0_DBM_RANGE[1]) - P0_DBM_RANGE[0]) / (
            P0_DBM_RANGE[1] - P0_DBM_RANGE[0]
        )
        alpha_norm = _clip(float(alpha), 0.0, 1.0)
        sinr_pressure = sum(_clip((5.0 - float(ue.sinr_db)) / 10.0, 0.0, 1.0) for ue in prev_cell.ues) / max(
            1, len(prev_cell.ues)
        )
        bler_pressure = sum(_clip((float(ue.bler) - 0.10) / 0.25, 0.0, 1.0) for ue in prev_cell.ues) / max(
            1, len(prev_cell.ues)
        )
        interference_pressure = max(sinr_pressure, bler_pressure)
        signal_delta = 4.0 * (p0_norm - 0.5) + 2.0 * (alpha_norm - 0.5) - 9.0 * interference_pressure * p0_norm
        for ue in prev_cell.ues:
            _add_ue_bias(state, cell_id, ue.ue_id, "sinr_db", signal_delta)
            _add_ue_bias(state, cell_id, ue.ue_id, "bler", -0.025 * signal_delta)
            _add_ue_bias(state, cell_id, ue.ue_id, "delivered_mbps", 0.35 * signal_delta)
            _add_ue_bias(state, cell_id, ue.ue_id, "buffer_occupancy_kb", -70.0 * signal_delta)
        _add_cell_bias(
            state,
            cell_id,
            "prb_util_ul_p50",
            0.03 * interference_pressure * (p0_norm - 0.5),
        )


def _record_action_effect(
    *,
    state: ReplayActionState,
    prev_obs: Observation,
    action: ToolCall,
) -> None:
    args = action.arguments or {}
    try:
        cell_id = int(args.get("cell_id", 0))
    except (TypeError, ValueError):
        return

    if action.name == "set_prb_cap":
        max_prb = int(args.get("max_prb", PRB_MAX))
        target = str(args.get("target", "ue"))
        try:
            target_id = int(args.get("target_id", 0))
        except (TypeError, ValueError):
            target_id = 0
        cap_key = (cell_id, target, target_id)
        if max_prb >= PRB_MAX:
            state.prb_cap_setpoints.pop(cap_key, None)
        else:
            # Setpoint replacement is collective: no per-cap positive bias is
            # accumulated here. All active caps are solved jointly below.
            state.prb_cap_setpoints[cap_key] = max_prb
        return

    if action.name == "set_scheduler_policy":
        state.scheduler_policy[cell_id] = str(args.get("policy", "PF"))
        return

    if action.name == "set_qos_weights":
        weights = args.get("weights") or {}
        state.qos_weights[cell_id] = {str(key): float(value) for key, value in weights.items()}
        return

    if action.name == "set_admission_policy":
        state.admission_threshold[cell_id] = float(args.get("accept_threshold_pct", 80.0))
        return

    if action.name == "set_mcs_bounds":
        state.mcs_setpoint[cell_id] = (
            int(args.get("mcs_min", 0)),
            int(args.get("mcs_max", 27)),
            float(args.get("target_bler", 0.1)),
        )
        return

    if action.name == "set_handover_trigger":
        state.handover_setpoint[cell_id] = (
            float(args.get("a3_offset_db", 0.0)),
            int(args.get("ttt_ms", 160)),
        )
        return

    if action.name == "set_ul_power_control":
        state.ul_power_setpoint[cell_id] = (
            float(args.get("p0_dbm", -90.0)),
            float(args.get("alpha", 0.8)),
        )


def apply_action_effect(
    *,
    prev_obs: Observation,
    base_next_obs: Observation,
    action: ToolCall,
    accepted: bool = True,
    state: ReplayActionState | None = None,
    cell_capacity_mbps: float = 60.0,
) -> Observation:
    """Apply deterministic action-conditioned replay effects to ``base_next_obs``.

    The effect model is deliberately coarse. It encodes monotonic control
    directions for synthetic-data generation without pretending to be a radio
    physics model. ``noop`` and rejected actions preserve the original
    action-blind replay transition.
    """

    if state is None and (not accepted or action.name == "noop"):
        return base_next_obs

    state = state or ReplayActionState()
    if accepted and action.name != "noop":
        _record_action_effect(state=state, prev_obs=prev_obs, action=action)
    _rebuild_action_biases(state=state, prev_obs=prev_obs)
    data = base_next_obs.model_dump(by_alias=True)
    _apply_state_biases(
        data,
        state,
        cell_capacity_mbps=cell_capacity_mbps,
    )
    _apply_admission_setpoints(data, state)
    obs = Observation.model_validate(
        _clamp_observation_payload(
            data,
            cell_capacity_mbps=cell_capacity_mbps,
        )
    )
    return obs


def _synthesize_kpi_snapshot(
    rng: np.random.Generator,
    fingerprint: _ScenarioFingerprint,
    step_idx: int,
) -> KpiSnapshot:
    """Build a deterministic :class:`KpiSnapshot` for replay step ``step_idx``.

    The synthetic dynamics are intentionally simple:

    - PRB util drifts around the cell's offered/capacity ratio with a small
      seeded gaussian.
    - Per-UE delivered throughput is offered, attenuated by overload
      (``max(0, load_ratio - 1)``), with a small noise term.
    - SINR is a per-UE base + gaussian noise.
    - BLER scales weakly with PRB util.

    The coefficients are chosen to keep the local reward signal non-degenerate.
    """
    snap = KpiSnapshot()
    snap.source_mode = "replay"

    # Reproducibility: derive each step's generator sequentially from the
    # parent RNG. ReplayEnv always builds the complete trajectory at reset, so
    # identical seeds and inputs consume the same sequence and reproduce the
    # same trajectory.
    sub = np.random.default_rng(rng.bit_generator.random_raw())

    # Cell-level offered load -> baseline PRB utilisation.
    cells_offered: dict[int, float] = {}
    cells_n_ues: dict[int, int] = {}
    for (cell_id, _ue_id), offered in fingerprint.offered_mbps.items():
        cells_offered[cell_id] = cells_offered.get(cell_id, 0.0) + offered
        cells_n_ues[cell_id] = cells_n_ues.get(cell_id, 0) + 1
    if not cells_offered:
        # Fallback: 2 cells, 2 UEs each, 5 Mbps offered.
        for cell_id in range(max(1, fingerprint.n_cells)):
            cells_offered[cell_id] = 10.0
            cells_n_ues[cell_id] = max(1, fingerprint.n_ues_total // max(1, fingerprint.n_cells))

    capacity = max(1.0, fingerprint.cell_capacity_mbps)

    for cell_id, offered_total in cells_offered.items():
        load_ratio = offered_total / capacity
        # Seeded noise; std small enough that values stay realistic.
        prb_util = float(np.clip(load_ratio + sub.normal(0.0, 0.04), 0.0, 1.0))
        snap.prb_util[cell_id] = prb_util
        snap.active_ue_count[cell_id] = int(cells_n_ues.get(cell_id, 0))

        # Global "headroom" knob: above 1.0 the cell is over-subscribed and
        # delivered drops proportionally to overload.
        if load_ratio > 1.0:
            attenuation = 1.0 / load_ratio
        else:
            attenuation = 1.0

        for ue_id in range(cells_n_ues.get(cell_id, 0)):
            offered = float(fingerprint.offered_mbps.get((cell_id, ue_id), 1.0))
            delivered = _clip(
                offered * attenuation + float(sub.normal(0.0, 0.5)),
                0.0,
                offered,
            )
            sinr_db = float(8.0 + 2.0 * ue_id + sub.normal(0.0, 1.5))
            bler = float(np.clip(0.05 + 0.10 * prb_util + sub.normal(0.0, 0.02), 0.0, 1.0))

            interference = _regime_weight(fingerprint, "interference")
            if interference > 0.0:
                # The scenario sampler records NLOS pulses, but the replay
                # fingerprint intentionally keeps only the slim fields needed
                # by the env. Encode the regime as deterministic KPI pressure
                # so interference rollouts can naturally exercise handover,
                # UL-power, and MCS tools.
                affected = 1.0 if ue_id % 2 == 0 else 0.5
                strength = interference * affected
                delivered = max(0.0, delivered * (1.0 - 0.55 * strength))
                sinr_db -= 12.0 * strength
                bler = float(np.clip(bler + 0.22 * strength, 0.0, 1.0))

            snap.throughput_mbps[(cell_id, ue_id)] = delivered
            snap.sinr_db[(cell_id, ue_id)] = sinr_db
            snap.bler[(cell_id, ue_id)] = bler

    return snap


def build_trajectory(
    *,
    seed: int,
    difficulty: float = 0.5,
    regime_mix: Optional[dict[str, float]] = None,
    tier: str = "replay",
    n_steps: int = 60,
) -> tuple[list[Observation], _ScenarioFingerprint]:
    """Standalone helper: produce the full observation trajectory for a seed.

    Used by :class:`ReplayEnv`, which slices the trajectory across ``step``
    calls, and by offline callers that need the same seeded observations.
    """
    fingerprint = _sample_scenario(
        seed=seed,
        difficulty=difficulty,
        regime_mix=regime_mix,
        tier=tier,
    )
    # Use one PRNG seeded by `seed`; `_synthesize_kpi_snapshot` consumes one
    # child seed per trajectory step in a fixed order.
    rng = np.random.default_rng(seed)

    # We need a temporary "episode" to feed _build_observation. Its only
    # role is providing the episode_id and meta context for the obs.
    placeholder_meta = EpisodeMeta(
        episode_id=f"trj_{seed}",
        seed=seed,
        difficulty=difficulty,
        regime_mix=fingerprint.regime_mix,
        tier=tier,
        max_steps=n_steps,
    )
    placeholder = ObservationContext(
        episode_id=placeholder_meta.episode_id,
        meta=placeholder_meta,
        fingerprint=fingerprint,
    )

    obs_list: list[Observation] = []
    for t in range(n_steps):
        snap = _synthesize_kpi_snapshot(rng, fingerprint, t)
        placeholder.step_idx = t
        obs = _build_observation(
            snapshot=snap,
            episode=placeholder,
            t_s=float(t),
        )
        obs = Observation.model_validate(
            _clamp_observation_payload(
                obs.model_dump(by_alias=True),
                cell_capacity_mbps=fingerprint.cell_capacity_mbps,
            )
        )
        obs_list.append(obs)

    return obs_list, fingerprint


# --- Episode + Env ---------------------------------------------------------


@dataclass
class ReplayEpisode:
    episode_id: str
    meta: EpisodeMeta
    fingerprint: _ScenarioFingerprint
    trajectory: list[Observation] = field(default_factory=list)
    step_idx: int = 0
    last_action: Optional[ToolCall] = None
    last_reward: Optional[float] = None
    last_rejection: Optional[str] = None
    history: list[_guardrail.HistoryEntry] = field(default_factory=list)
    action_state: ReplayActionState = field(default_factory=ReplayActionState)
    closed: bool = False
    created_at: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


class ReplayEnv:
    """Deterministic env with pre-built synthetic observation trajectories."""

    def __init__(
        self,
        *,
        pool_size: int = 32,
        max_steps_default: int = 60,
    ) -> None:
        self.pool_size = int(pool_size)
        self.max_steps_default = int(max_steps_default)
        self._lock = threading.Lock()
        self._episodes: dict[str, ReplayEpisode] = {}
        self._pending_resets = 0

    def reset(
        self,
        *,
        seed: int = 0,
        difficulty: float = 0.5,
        regime_mix: Optional[dict[str, float]] = None,
        scenario_id: Optional[str] = None,
        tier: str = "replay",
        max_steps: Optional[int] = None,
    ) -> tuple[Observation, EpisodeMeta]:
        with self._lock:
            self._episodes = {k: v for k, v in self._episodes.items() if not v.closed}
            if len(self._episodes) + self._pending_resets >= self.pool_size:
                raise RuntimeError(f"replay env pool exhausted ({self.pool_size} slots all busy)")
            self._pending_resets += 1

        try:
            n_steps = int(max_steps if max_steps is not None else self.max_steps_default)
            episode_id = f"ep_replay_{uuid.uuid4().hex[:8]}"

            trajectory, fingerprint = build_trajectory(
                seed=seed,
                difficulty=difficulty,
                regime_mix=regime_mix,
                tier=tier,
                n_steps=n_steps + 1,  # +1 so /step always has a "next" obs to return
            )

            meta = EpisodeMeta(
                episode_id=episode_id,
                seed=seed,
                difficulty=difficulty,
                regime_mix=fingerprint.regime_mix,
                tier=tier,
                scenario_id=scenario_id,
                max_steps=n_steps,
            )

            episode = ReplayEpisode(
                episode_id=episode_id,
                meta=meta,
                fingerprint=fingerprint,
                trajectory=trajectory,
            )

            # Re-stamp the trajectory with the real episode_id so observations
            # don't leak the placeholder ``trj_<seed>``.
            episode.trajectory = [obs.model_copy(update={"episode_id": episode_id}) for obs in trajectory]

            with self._lock:
                self._episodes[episode_id] = episode
        finally:
            with self._lock:
                self._pending_resets = max(0, self._pending_resets - 1)

        first_obs = episode.trajectory[0]
        return first_obs, meta

    def step(
        self,
        episode_id: str,
        action: ToolCall,
    ) -> tuple[Observation, float, bool, dict[str, Any]]:
        with self._lock:
            episode = self._episodes.get(episode_id)
        if episode is None:
            raise KeyError(f"unknown episode_id {episode_id!r}")
        with episode.lock:
            if episode.closed:
                raise RuntimeError(f"episode {episode_id!r} is closed")

            logical_now_s = float(episode.step_idx)
            prev_obs = episode.trajectory[episode.step_idx]
            gr = _guardrail.check(
                action,
                history=episode.history,
                n_cells=len(prev_obs.cells),
                n_ues=max(1, prev_obs.global_.n_ues_total),
                ue_ids_by_cell={cell.cell_id: {ue.ue_id for ue in cell.ues} for cell in prev_obs.cells},
                now_s=logical_now_s,
            )
            # Never report a synthetic action as accepted unless replay can
            # apply it causally. Slice membership is not observed, so mapping
            # a slice cap to affected UEs would invent topology.
            gr = _synthetic_replay_guardrail(action, gr)
            rejected = not gr.accepted

            # Advance to the next pre-baked observation. step_idx is incremented
            # to N AFTER the step that returns trajectory[N].
            next_idx = min(episode.step_idx + 1, len(episode.trajectory) - 1)
            base_next_obs = episode.trajectory[next_idx]
            candidate_state = copy.deepcopy(episode.action_state)
            new_obs = apply_action_effect(
                prev_obs=prev_obs,
                base_next_obs=base_next_obs,
                action=action,
                accepted=gr.accepted,
                state=candidate_state,
                cell_capacity_mbps=episode.fingerprint.cell_capacity_mbps,
            )

            reward_profile = select_reward_profile(episode.fingerprint.tier)
            reward_breakdown = _rewards.compute_breakdown(
                prev_obs=prev_obs,
                curr_obs=new_obs,
                action=action,
                rejected=rejected,
                cell_capacity_mbps=episode.fingerprint.cell_capacity_mbps,
                prb_pressure_threshold=(reward_profile.prb_pressure_threshold),
            )
            reward = float(reward_breakdown["total"])

            candidate_history = list(episode.history)
            if not rejected:
                candidate_history.append(
                    _guardrail.HistoryEntry(
                        action=action,
                        t_s=logical_now_s,
                    )
                )
                if len(candidate_history) > 64:
                    candidate_history = candidate_history[-32:]

            # Stamp agent_aux so policy and training consumers receive the
            # action/reward context from the preceding transition.
            from .schemas import AgentAux, LastActionEcho

            aux = AgentAux(
                last_action=LastActionEcho(name=action.name, arguments=action.arguments),
                last_reward=reward,
                last_rejection=gr.reason,
                step_idx=next_idx,
            )
            new_obs = new_obs.model_copy(update={"agent_aux": aux})

            # Commit only after action dynamics, accounting, reward, and
            # observation construction have all succeeded.
            episode.action_state = candidate_state
            episode.step_idx = next_idx
            episode.last_action = action
            episode.last_rejection = gr.reason
            episode.last_reward = reward
            episode.history = candidate_history
            episode.trajectory[next_idx] = new_obs

            done = episode.step_idx >= episode.meta.max_steps
            info = {
                "guardrail_accepted": gr.accepted,
                "rejection_reason": gr.reason,
                "step_idx": episode.step_idx,
                "kpi_source": "replay",
                "dynamics_mode": ACTION_EFFECT_VERSION,
                "reward_measurements": reward_breakdown["measurements"],
                "reward_terms": reward_breakdown["terms"],
                "reward_version": reward_profile.version,
                "prb_cap_dynamics": candidate_state.last_prb_cap_diagnostics,
            }
            return new_obs, reward, done, info

    def close(self, episode_id: str) -> dict[str, Any]:
        with self._lock:
            episode = self._episodes.get(episode_id)
        if episode is None:
            raise KeyError(f"unknown episode_id {episode_id!r}")
        with episode.lock:
            if episode.closed:
                raise KeyError(f"unknown episode_id {episode_id!r}")
            episode.closed = True
            with self._lock:
                if self._episodes.get(episode_id) is not episode:
                    raise KeyError(f"unknown episode_id {episode_id!r}")
                del self._episodes[episode_id]
            return {"ok": True, "n_steps": episode.step_idx}


__all__ = [
    "ReplayEnv",
    "ReplayEpisode",
    "ReplayActionBias",
    "ReplayActionState",
    "ACTION_EFFECT_VERSION",
    "action_effect_version",
    "apply_action_effect",
    "build_trajectory",
    "_synthesize_kpi_snapshot",
]
