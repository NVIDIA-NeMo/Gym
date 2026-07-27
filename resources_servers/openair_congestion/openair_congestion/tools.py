# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Action space for ``openair_congestion_v1`` — 7 actuator tools + ``noop``.

Schemas are emitted in OpenAI function-calling shape (``{"type": "function",
"function": {...}}``) so vLLM + Qwen3-Instruct can emit valid tool calls
out of the box. The same dicts are consumed by:

- ``schemas.ActionToolCall`` validation in :mod:`openair_congestion.schemas`
- The deterministic ``guardrail.py`` (M4.2) that rejects out-of-range values
  before they hit the backend. Replay mode uses a deterministic synthetic
  action-effect model; synthetic-live actions are logged only; connected T1
  only wires ``set_admission_policy`` to traffic-side stream suppression.
  FlexRIC RC / OAI telnet actuation is future work.

Numeric bounds follow 3GPP / OAI realistic ranges (see ``docs/PLAN.md`` §4.3
and §4.5). Bounds are intentionally narrower than the protocol spec to keep
the action space tractable for SFT / GRPO; widen later if the policy plateaus.

The full list of canonical tool names is exported as ``TOOL_NAMES``; the
OpenAI-shaped list as ``TOOLS``; per-name lookup via ``TOOL_SCHEMA_BY_NAME``.
"""

from __future__ import annotations

from typing import Any


# --- Operational bounds (single source of truth for guardrail + tests) ------

MAX_CELLS = 4  # T1=2, T2=3, T3=4 (PLAN.md §4.7)
MAX_UES = 24  # T3 = 4 cells * up to 6 UEs/cell, with headroom

PRB_MAX = 273  # 100 MHz BW, μ=1, FR1 (3GPP 38.211 Table 4.1.1-2 row μ=1)
MCS_MAX = 27  # 3GPP 38.214 Table 5.1.3.1-1 (mcs index 0..27 + reserved)

POLICY_VALUES = ("PF", "RR", "MaxCI")
PRB_CAP_TARGETS = ("ue",)
A3_OFFSET_DB_RANGE = (-24.0, 24.0)
TTT_MS_VALUES = (
    0,
    40,
    64,
    80,
    100,
    128,
    160,
    256,
    320,
    480,
    512,
    640,
    1024,
    1280,
    2560,
    5120,
)  # 3GPP 38.331 Table TimeToTrigger
P0_DBM_RANGE = (-126.0, 23.0)
ALPHA_VALUES = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# --- Tool schemas (OpenAI function-calling shape) ---------------------------


def _func(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
    *,
    additional_properties: bool = False,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": additional_properties,
            },
        },
    }


_CELL_ID_PROP = {
    "type": "integer",
    "minimum": 0,
    "maximum": MAX_CELLS - 1,
    "description": "Cell identifier (0-indexed; valid 0..n_cells-1 at runtime).",
}

_ACTION_CONNECTIVITY_NOTE = (
    " Current deliverable #1 action provenance: replay mode uses a deterministic "
    "synthetic action-effect model; synthetic-live mode logs accepted actions "
    "only; connected T1 mode wires only set_admission_policy to traffic-side "
    "simulator stream suppression. This is not FlexRIC RC or OAI telnet RAN "
    "control."
)

_ADMISSION_CONNECTIVITY_NOTE = (
    " Current deliverable #1 action provenance: in connected T1 mode this "
    "suppresses one generated UE traffic stream when current estimated load "
    "exceeds accept_threshold_pct. In replay it uses the deterministic "
    "synthetic action-effect model; in synthetic-live it is logged only. This "
    "is not FlexRIC RC or OAI telnet RAN control."
)


TOOLS: list[dict[str, Any]] = [
    _func(
        "set_scheduler_policy",
        "Switch the per-cell MAC scheduler between proportional fair (PF), "
        "round-robin (RR), and max-CI (MaxCI). PF is the default upstream OAI "
        "policy; RR favours fairness over throughput; MaxCI maximises sum-rate "
        "but starves cell-edge UEs." + _ACTION_CONNECTIVITY_NOTE,
        {
            "cell_id": _CELL_ID_PROP,
            "policy": {
                "type": "string",
                "enum": list(POLICY_VALUES),
                "description": "Target scheduler policy.",
            },
        },
        ["cell_id", "policy"],
    ),
    _func(
        "set_prb_cap",
        "Cap the maximum PRBs allocated per scheduling round to a single UE. "
        "Used to throttle a hog UE that is starving the rest of the "
        "cell." + _ACTION_CONNECTIVITY_NOTE,
        {
            "cell_id": _CELL_ID_PROP,
            "target": {
                "type": "string",
                "enum": list(PRB_CAP_TARGETS),
                "description": "The cap applies to one observed UE.",
            },
            "target_id": {
                "type": "integer",
                "minimum": 0,
                "maximum": MAX_UES - 1,
                "description": "UE id exactly as listed under the selected cell.",
            },
            "max_prb": {
                "type": "integer",
                "minimum": 0,
                "maximum": PRB_MAX,
                "description": "Hard cap on PRBs per slot for this target.",
            },
        },
        ["cell_id", "target", "target_id", "max_prb"],
    ),
    _func(
        "set_mcs_bounds",
        "Constrain the MCS index range the scheduler may select per UE, plus "
        "an outer-loop link-adaptation BLER target. Useful when channel "
        "estimates are noisy and the default MCS picker oscillates." + _ACTION_CONNECTIVITY_NOTE,
        {
            "cell_id": _CELL_ID_PROP,
            "mcs_min": {
                "type": "integer",
                "minimum": 0,
                "maximum": MCS_MAX,
                "description": "Lower bound on MCS index (3GPP 38.214 Table 5.1.3.1-1).",
            },
            "mcs_max": {
                "type": "integer",
                "minimum": 0,
                "maximum": MCS_MAX,
                "description": "Upper bound on MCS index. Must be >= mcs_min.",
            },
            "target_bler": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 0.5,
                "description": "Outer-loop target block-error rate (typical 0.1).",
            },
        },
        ["cell_id", "mcs_min", "mcs_max", "target_bler"],
    ),
    _func(
        "set_qos_weights",
        "Re-weight the per-5QI scheduler priority. Larger weights win more "
        "PRBs for that 5QI. Use to defend low-latency 5QIs (e.g. 5QI=1 voice) "
        "during congestion." + _ACTION_CONNECTIVITY_NOTE,
        {
            "cell_id": _CELL_ID_PROP,
            "weights": {
                "type": "object",
                "description": "Mapping of 5QI (string-keyed integer 1..127) → weight (0..10).",
                "additionalProperties": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 10.0,
                },
                "minProperties": 1,
            },
        },
        ["cell_id", "weights"],
    ),
    _func(
        "set_admission_policy",
        "Tighten or loosen RRC admission. Lowering the accept_threshold "
        "rejects new RRC setup requests early during PRACH storms; "
        "slice_reservation guarantees a minimum PRB pool per slice." + _ADMISSION_CONNECTIVITY_NOTE,
        {
            "cell_id": _CELL_ID_PROP,
            "accept_threshold_pct": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 100.0,
                "description": "RRC accept threshold as a percentage of cell load.",
            },
            "slice_reservation": {
                "type": "object",
                "description": "Mapping of slice id (string-keyed integer) → reserved PRBs.",
                "additionalProperties": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": PRB_MAX,
                },
            },
        },
        ["cell_id", "accept_threshold_pct", "slice_reservation"],
    ),
    _func(
        "set_handover_trigger",
        "Adjust the A3-event handover trigger: a3_offset_db is the dB margin "
        "the neighbour must beat the serving cell by, ttt_ms is the dwell time. "
        "Tighter (lower offset, shorter ttt) → more handovers; looser → "
        "ping-pong protection." + _ACTION_CONNECTIVITY_NOTE,
        {
            "cell_id": _CELL_ID_PROP,
            "a3_offset_db": {
                "type": "number",
                "minimum": A3_OFFSET_DB_RANGE[0],
                "maximum": A3_OFFSET_DB_RANGE[1],
                "description": "A3-event offset in dB.",
            },
            "ttt_ms": {
                "type": "integer",
                "enum": list(TTT_MS_VALUES),
                "description": "Time-to-trigger from 3GPP 38.331 TimeToTrigger set.",
            },
        },
        ["cell_id", "a3_offset_db", "ttt_ms"],
    ),
    _func(
        "set_ul_power_control",
        "Configure uplink fractional path-loss compensation: p0_dbm is the "
        "target receive power in dBm, alpha (0..1) is the path-loss compensation "
        "fraction. Lowering p0 reduces UE Tx power (good for inter-cell "
        "interference), raising alpha hurts cell-edge UEs less." + _ACTION_CONNECTIVITY_NOTE,
        {
            "cell_id": _CELL_ID_PROP,
            "p0_dbm": {
                "type": "number",
                "minimum": P0_DBM_RANGE[0],
                "maximum": P0_DBM_RANGE[1],
                "description": "Target receive power per PRB in dBm.",
            },
            "alpha": {
                "type": "number",
                "enum": list(ALPHA_VALUES),
                "description": "Path-loss compensation fraction (3GPP 38.213 Table 7.1.1-1).",
            },
        },
        ["cell_id", "p0_dbm", "alpha"],
    ),
    _func(
        "noop",
        "Explicitly choose to take no configuration action this step. Required "
        "in the action space so the policy can deliberately stand pat instead "
        "of guessing a low-impact change.",
        {},
        [],
    ),
]


TOOL_NAMES: tuple[str, ...] = tuple(t["function"]["name"] for t in TOOLS)
TOOL_SCHEMA_BY_NAME: dict[str, dict[str, Any]] = {t["function"]["name"]: t for t in TOOLS}


def get_parameters_schema(tool_name: str) -> dict[str, Any]:
    """Return the JSON-Schema ``parameters`` block for a tool, or raise KeyError."""
    return TOOL_SCHEMA_BY_NAME[tool_name]["function"]["parameters"]


__all__ = [
    "TOOLS",
    "TOOL_NAMES",
    "TOOL_SCHEMA_BY_NAME",
    "get_parameters_schema",
    "MAX_CELLS",
    "MAX_UES",
    "PRB_MAX",
    "MCS_MAX",
    "POLICY_VALUES",
    "PRB_CAP_TARGETS",
    "A3_OFFSET_DB_RANGE",
    "TTT_MS_VALUES",
    "P0_DBM_RANGE",
    "ALPHA_VALUES",
]
