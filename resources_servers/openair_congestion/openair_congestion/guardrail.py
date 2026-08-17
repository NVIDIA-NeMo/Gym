# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic validation for congestion-control tool calls.

Every action passes through :func:`check` before being dispatched to the
actuator path. Rejected actions short-circuit actuator dispatch:

- delta-based positive reward terms are suppressed and ``w_reject`` is charged
  in :mod:`openair_congestion.rewards`,
- no backend state changes are applied,
- the next observation surfaces ``agent_aux.last_rejection`` so the LLM policy
  sees the rejection signal.

The check matrix:

1. Out-of-range numeric parameters (cell_id, mcs, prb cap, p0, etc.).
2. Actions targeting non-existent cell/UE IDs (configurable via the exact
   ``cell_ids`` / ``ue_ids_by_cell`` sets, with ``n_cells`` / ``n_ues``
   retained as the replay-compatible fallback).
3. Rate-limited identical-action repeats within ``rate_limit_s`` (default 2 s).
4. Catastrophic combinations (e.g. ``mcs_max == 0`` for an actuator that
   would otherwise zero out throughput).

The guardrail is deliberately **deterministic and stateless across calls**;
the only state is the optional ``history`` argument the env provides. This
keeps it cheap to call from /step and easy to unit-test exhaustively.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import ToolCall
from .tools import (
    A3_OFFSET_DB_RANGE,
    ALPHA_VALUES,
    MCS_MAX,
    P0_DBM_RANGE,
    POLICY_VALUES,
    PRB_CAP_TARGETS,
    PRB_MAX,
    TTT_MS_VALUES,
)


DEFAULT_RATE_LIMIT_S: float = 2.0


@dataclass(frozen=True)
class GuardrailResult:
    accepted: bool
    reason: str | None = None  # populated iff accepted is False


@dataclass
class HistoryEntry:
    action: ToolCall
    t_s: float


def _reject(reason: str) -> GuardrailResult:
    return GuardrailResult(accepted=False, reason=reason)


def _integral_json_number(value: object) -> int | None:
    """Normalize JSON-Schema ``integer`` values without accepting bools."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        return None
    return int(value)


def _finite_json_number(value: object) -> float | None:
    """Normalize JSON-Schema ``number`` values without accepting bools."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def check(
    action: ToolCall,
    *,
    history: list[HistoryEntry] | None = None,
    n_cells: int = 2,
    cell_ids: set[int] | None = None,
    n_ues: int = 4,
    n_ues_by_cell: dict[int, int] | None = None,
    ue_ids_by_cell: dict[int, set[int]] | None = None,
    now_s: float | None = None,
    rate_limit_s: float = DEFAULT_RATE_LIMIT_S,
) -> GuardrailResult:
    """Validate a single action. Returns :class:`GuardrailResult`."""
    args = action.arguments

    if action.name == "noop":
        return GuardrailResult(accepted=True)

    # --- Cell-id existence check (every actuator carries one) --------------
    if "cell_id" not in args:
        return _reject(f"{action.name}: missing cell_id")
    raw_cell_id = args["cell_id"]
    cell_id = _integral_json_number(raw_cell_id)
    if cell_ids is not None:
        if cell_id is None or cell_id not in cell_ids:
            return _reject(f"{action.name}: cell_id={raw_cell_id!r} not present; valid ids={sorted(cell_ids)}")
    elif cell_id is None or cell_id < 0 or cell_id >= n_cells:
        return _reject(f"{action.name}: cell_id={raw_cell_id!r} out of range [0,{n_cells})")

    # --- Per-tool checks ----------------------------------------------------

    if action.name == "set_scheduler_policy":
        policy = args.get("policy")
        if policy not in POLICY_VALUES:
            return _reject(f"set_scheduler_policy: policy={policy!r} not in {POLICY_VALUES}")

    elif action.name == "set_prb_cap":
        target = args.get("target")
        if target not in PRB_CAP_TARGETS:
            return _reject(f"set_prb_cap: target={target!r} not in {PRB_CAP_TARGETS}")
        raw_target_id = args.get("target_id")
        target_id = _integral_json_number(raw_target_id)
        if target == "ue" and ue_ids_by_cell is not None:
            valid_ids = ue_ids_by_cell.get(cell_id, set())
            if target_id is None or target_id not in valid_ids:
                return _reject(
                    f"set_prb_cap: target_id={raw_target_id!r} not present in cell {cell_id}; "
                    f"valid ids={sorted(valid_ids)}"
                )
            max_id = None
        elif target == "ue" and n_ues_by_cell is not None:
            max_id = n_ues_by_cell.get(cell_id, 0) - 1
        else:
            max_id = (n_ues if target == "ue" else 8) - 1
        if max_id is not None and (target_id is None or target_id < 0 or target_id > max_id):
            return _reject(f"set_prb_cap: target_id={raw_target_id!r} out of range [0,{max_id}]")
        raw_max_prb = args.get("max_prb")
        max_prb = _integral_json_number(raw_max_prb)
        if max_prb is None or max_prb < 0 or max_prb > PRB_MAX:
            return _reject(f"set_prb_cap: max_prb={raw_max_prb!r} out of [0,{PRB_MAX}]")
        if max_prb == 0:
            return _reject(f"set_prb_cap: max_prb=0 starves the {target} entirely (catastrophic)")

    elif action.name == "set_mcs_bounds":
        raw_mcs_min = args.get("mcs_min")
        raw_mcs_max = args.get("mcs_max")
        mcs_min = _integral_json_number(raw_mcs_min)
        mcs_max = _integral_json_number(raw_mcs_max)
        raw_target_bler = args.get("target_bler")
        target_bler = _finite_json_number(raw_target_bler)
        for label, val, raw_val in (
            ("mcs_min", mcs_min, raw_mcs_min),
            ("mcs_max", mcs_max, raw_mcs_max),
        ):
            if val is None or val < 0 or val > MCS_MAX:
                return _reject(f"set_mcs_bounds: {label}={raw_val!r} out of [0,{MCS_MAX}]")
        assert mcs_min is not None and mcs_max is not None
        if mcs_min > mcs_max:
            return _reject(f"set_mcs_bounds: mcs_min ({mcs_min}) > mcs_max ({mcs_max})")
        if mcs_max == 0:
            return _reject("set_mcs_bounds: mcs_max=0 zeroes throughput (catastrophic)")
        if target_bler is None or not 0.0 <= target_bler <= 0.5:
            return _reject(f"set_mcs_bounds: target_bler={raw_target_bler!r} out of [0,0.5]")

    elif action.name == "set_qos_weights":
        weights = args.get("weights")
        if not isinstance(weights, dict) or not weights:
            return _reject("set_qos_weights: weights must be a non-empty mapping")
        normalized_weights: list[float] = []
        for k, v in weights.items():
            try:
                key_int = int(k)
            except (TypeError, ValueError):
                return _reject(f"set_qos_weights: 5QI key {k!r} not an integer")
            if key_int < 1 or key_int > 127:
                return _reject(f"set_qos_weights: 5QI {key_int} out of [1,127]")
            weight = _finite_json_number(v)
            if weight is None or weight < 0.0 or weight > 10.0:
                return _reject(f"set_qos_weights[{key_int}]: weight={v!r} out of [0,10]")
            normalized_weights.append(weight)
        if all(weight == 0.0 for weight in normalized_weights):
            return _reject("set_qos_weights: all weights zero (catastrophic)")

    elif action.name == "set_admission_policy":
        accept = _finite_json_number(args.get("accept_threshold_pct"))
        if accept != 100.0:
            return _reject(
                "set_admission_policy: synthetic replay supports only 100% admission "
                "until denied-demand accounting is modeled"
            )
        slice_res = args.get("slice_reservation", {})
        if not isinstance(slice_res, dict):
            return _reject("set_admission_policy: slice_reservation must be a mapping")
        if slice_res:
            return _reject("set_admission_policy: slice_reservation must be empty")

    elif action.name == "set_handover_trigger":
        raw_a3 = args.get("a3_offset_db")
        a3 = _finite_json_number(raw_a3)
        raw_ttt = args.get("ttt_ms")
        ttt = _integral_json_number(raw_ttt)
        if a3 is None or not A3_OFFSET_DB_RANGE[0] <= a3 <= A3_OFFSET_DB_RANGE[1]:
            return _reject(f"set_handover_trigger: a3_offset_db={raw_a3!r} out of {A3_OFFSET_DB_RANGE}")
        if ttt is None or ttt not in TTT_MS_VALUES:
            return _reject(f"set_handover_trigger: ttt_ms={raw_ttt!r} not in 38.331 set")

    elif action.name == "set_ul_power_control":
        raw_p0 = args.get("p0_dbm")
        raw_alpha = args.get("alpha")
        p0 = _finite_json_number(raw_p0)
        alpha = _finite_json_number(raw_alpha)
        if p0 is None or not P0_DBM_RANGE[0] <= p0 <= P0_DBM_RANGE[1]:
            return _reject(f"set_ul_power_control: p0_dbm={raw_p0!r} out of {P0_DBM_RANGE}")
        if alpha is None or alpha not in ALPHA_VALUES:
            return _reject(f"set_ul_power_control: alpha={raw_alpha!r} not in {ALPHA_VALUES}")

    else:
        return _reject(f"unknown tool {action.name!r} reached guardrail")

    # --- Rate-limit: reject identical action within window ------------------

    if history and rate_limit_s > 0.0:
        if now_s is None:
            raise ValueError("now_s is required when rate-limit history is provided")
        # Look at most-recent entries first; stop after the first older than window.
        for entry in reversed(history):
            if entry.t_s < now_s - rate_limit_s:
                break
            if entry.action.name == action.name and entry.action.arguments == args:
                return _reject(f"{action.name}: identical-action repeat within {rate_limit_s:g}s window")

    return GuardrailResult(accepted=True)


__all__ = [
    "DEFAULT_RATE_LIMIT_S",
    "GuardrailResult",
    "HistoryEntry",
    "check",
]
