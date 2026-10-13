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
2. Actions targeting non-existent cell/UE IDs (configurable via the
   ``n_cells`` / ``n_ues`` knobs which the env owns).
3. Rate-limited identical-action repeats within ``rate_limit_s`` (default 2 s).
4. Catastrophic combinations (e.g. ``mcs_max == 0`` for an actuator that
   would otherwise zero out throughput).

The guardrail is deliberately **deterministic and stateless across calls**;
the only state is the optional ``history`` argument the env provides. This
keeps it cheap to call from /step and easy to unit-test exhaustively.
"""

from __future__ import annotations

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


def check(
    action: ToolCall,
    *,
    history: list[HistoryEntry] | None = None,
    n_cells: int = 2,
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
    cell_id = args["cell_id"]
    if not isinstance(cell_id, int) or cell_id < 0 or cell_id >= n_cells:
        return _reject(f"{action.name}: cell_id={cell_id!r} out of range [0,{n_cells})")

    # --- Per-tool checks ----------------------------------------------------

    if action.name == "set_scheduler_policy":
        policy = args.get("policy")
        if policy not in POLICY_VALUES:
            return _reject(f"set_scheduler_policy: policy={policy!r} not in {POLICY_VALUES}")

    elif action.name == "set_prb_cap":
        target = args.get("target")
        if target not in PRB_CAP_TARGETS:
            return _reject(f"set_prb_cap: target={target!r} not in {PRB_CAP_TARGETS}")
        target_id = args.get("target_id")
        if target == "ue" and ue_ids_by_cell is not None:
            valid_ids = ue_ids_by_cell.get(cell_id, set())
            if not isinstance(target_id, int) or target_id not in valid_ids:
                return _reject(
                    f"set_prb_cap: target_id={target_id!r} not present in cell {cell_id}; "
                    f"valid ids={sorted(valid_ids)}"
                )
            max_id = None
        elif target == "ue" and n_ues_by_cell is not None:
            max_id = n_ues_by_cell.get(cell_id, 0) - 1
        else:
            max_id = (n_ues if target == "ue" else 8) - 1
        if max_id is not None and (not isinstance(target_id, int) or target_id < 0 or target_id > max_id):
            return _reject(f"set_prb_cap: target_id={target_id!r} out of range [0,{max_id}]")
        max_prb = args.get("max_prb")
        if not isinstance(max_prb, int) or max_prb < 0 or max_prb > PRB_MAX:
            return _reject(f"set_prb_cap: max_prb={max_prb!r} out of [0,{PRB_MAX}]")
        if max_prb == 0:
            return _reject(f"set_prb_cap: max_prb=0 starves the {target} entirely (catastrophic)")

    elif action.name == "set_mcs_bounds":
        mcs_min = args.get("mcs_min")
        mcs_max = args.get("mcs_max")
        target_bler = args.get("target_bler")
        for label, val in (("mcs_min", mcs_min), ("mcs_max", mcs_max)):
            if not isinstance(val, int) or val < 0 or val > MCS_MAX:
                return _reject(f"set_mcs_bounds: {label}={val!r} out of [0,{MCS_MAX}]")
        if mcs_min > mcs_max:
            return _reject(f"set_mcs_bounds: mcs_min ({mcs_min}) > mcs_max ({mcs_max})")
        if mcs_max == 0:
            return _reject("set_mcs_bounds: mcs_max=0 zeroes throughput (catastrophic)")
        if not isinstance(target_bler, (int, float)) or not 0.0 <= float(target_bler) <= 0.5:
            return _reject(f"set_mcs_bounds: target_bler={target_bler!r} out of [0,0.5]")

    elif action.name == "set_qos_weights":
        weights = args.get("weights")
        if not isinstance(weights, dict) or not weights:
            return _reject("set_qos_weights: weights must be a non-empty mapping")
        for k, v in weights.items():
            try:
                key_int = int(k)
            except (TypeError, ValueError):
                return _reject(f"set_qos_weights: 5QI key {k!r} not an integer")
            if key_int < 1 or key_int > 127:
                return _reject(f"set_qos_weights: 5QI {key_int} out of [1,127]")
            if not isinstance(v, (int, float)) or float(v) < 0.0 or float(v) > 10.0:
                return _reject(f"set_qos_weights[{key_int}]: weight={v!r} out of [0,10]")
        if all(float(v) == 0.0 for v in weights.values()):
            return _reject("set_qos_weights: all weights zero (catastrophic)")

    elif action.name == "set_admission_policy":
        accept = args.get("accept_threshold_pct")
        if not isinstance(accept, (int, float)) or not 0.0 <= float(accept) <= 100.0:
            return _reject(f"set_admission_policy: accept_threshold_pct={accept!r} out of [0,100]")
        if float(accept) <= 0.0:
            return _reject("set_admission_policy: accept_threshold_pct=0 rejects all admissions (catastrophic)")
        slice_res = args.get("slice_reservation", {})
        if not isinstance(slice_res, dict):
            return _reject("set_admission_policy: slice_reservation must be a mapping")
        for k, v in slice_res.items():
            try:
                slice_id = int(k)
            except (TypeError, ValueError):
                return _reject(f"set_admission_policy: slice id {k!r} not an integer")
            if slice_id < 0 or slice_id > 8:
                return _reject(f"set_admission_policy: slice id {slice_id} out of [0,8]")
            if not isinstance(v, int) or v < 0 or v > PRB_MAX:
                return _reject(f"set_admission_policy: slice {k} reservation={v!r} out of [0,{PRB_MAX}]")

    elif action.name == "set_handover_trigger":
        a3 = args.get("a3_offset_db")
        ttt = args.get("ttt_ms")
        if not isinstance(a3, (int, float)) or not A3_OFFSET_DB_RANGE[0] <= float(a3) <= A3_OFFSET_DB_RANGE[1]:
            return _reject(f"set_handover_trigger: a3_offset_db={a3!r} out of {A3_OFFSET_DB_RANGE}")
        if not isinstance(ttt, int) or ttt not in TTT_MS_VALUES:
            return _reject(f"set_handover_trigger: ttt_ms={ttt!r} not in 38.331 set")

    elif action.name == "set_ul_power_control":
        p0 = args.get("p0_dbm")
        alpha = args.get("alpha")
        if not isinstance(p0, (int, float)) or not P0_DBM_RANGE[0] <= float(p0) <= P0_DBM_RANGE[1]:
            return _reject(f"set_ul_power_control: p0_dbm={p0!r} out of {P0_DBM_RANGE}")
        if not isinstance(alpha, (int, float)) or float(alpha) not in ALPHA_VALUES:
            return _reject(f"set_ul_power_control: alpha={alpha!r} not in {ALPHA_VALUES}")

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
