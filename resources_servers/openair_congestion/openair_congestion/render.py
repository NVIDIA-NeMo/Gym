# SPDX-License-Identifier: Apache-2.0
"""Observation → ChatML render (``docs/PLAN.md`` §4.4).

The training pipeline feeds the policy LLM (Qwen3-1.7B-Instruct) one ChatML
``user`` message per step, summarising the rolling 5 s observation window in
compact natural language. The system message is constant; the user message is
deterministic given the observation.

Two output formats:

- ``to_chatml(obs)`` — full ChatML messages list ``[{role, content}, ...]``,
  ready to feed to vLLM via the ``/v1/chat/completions`` API.
- ``to_ascii(obs)`` — single-line-per-cell summary for the
  ``GET /render?format=ascii`` endpoint and humans tailing logs.
"""

from __future__ import annotations

import json
from typing import Any

from .schemas import Observation
from .t2_policy_features import (
    T2_CELL_CAPACITY_MILLI_MBPS,
    build_t2_decision_features,
    format_deci_kb,
    format_milli_mbps,
    format_t2_cell_row,
    format_t2_policy_row,
    quantize_deci_kb,
    quantize_milli_mbps,
)


T2_OBSERVATION_RENDER = "t2_compact_pipe_v2"


SYSTEM_PROMPT = (
    "You are an OpenAirInterface 5G operator. Each user message contains "
    "rolling 5-second cell and UE telemetry. Choose exactly one tool call "
    "per turn from the configured action space to relieve congestion or "
    "stand pat with `noop`. Respect the parameter ranges; the env rejects "
    "out-of-range arguments and penalizes the same step's reward. Current "
    "action connectivity: replay mode uses a deterministic synthetic "
    "action-effect model; synthetic-live mode logs accepted actions only; "
    "connected T1 mode wires only set_admission_policy to traffic-side "
    "simulator stream suppression. No tool is real FlexRIC RC or OAI telnet "
    "RAN control in deliverable #1."
)

T2_SYSTEM_PROMPT = (
    "You are an OpenAirInterface 5G operator. Each user message contains "
    "rolling 5-second cell and UE telemetry. Choose exactly one tool call "
    "per turn from the configured action space to relieve congestion or "
    "stand pat with `noop`. Respect the parameter ranges; the env rejects "
    "out-of-range arguments and penalizes the same step's reward. For this "
    "T2 experiment, emit only one of these exact shapes: "
    '<tool_call>{"name":"noop","arguments":{}}</tool_call> or '
    '<tool_call>{"name":"set_prb_cap",'
    '"arguments":{"cell_id":0,"target":"ue","target_id":0,'
    '"max_prb":210}}'
    "</tool_call>. Replace cell_id with 0..2, target_id with a UE id listed "
    "under that cell, and max_prb with an integer 200..273. Do not emit "
    "other tools. set_prb_cap is a persistent traffic-side admitted-demand "
    "setpoint, not a real PRB scheduler command; denied service is penalized "
    "and max_prb=273 fully releases a cap. "
    "set_admission_policy forcibly terminates active service; "
    "MCS/UL effects depend on placeholder radio fields; handover/QoS/"
    "scheduler are log-only. runner_snapshot KPIs are traffic-side estimates "
    "and SINR/BLER are placeholders, not radio measurements. Burst load is "
    "duty-averaged, PRACH is modeled arrival pressure, and NLOS is a modeled "
    "traffic impairment. No tool controls "
    "the real OAI RAN, FlexRIC RC, or gNB telnet interface. Compact telemetry "
    "rows use: T|time|step|tier|source; "
    "P|cell_capacity_mbps|expert_cadence_wait_steps|expert_ready; "
    "C|cell|prb50|prb99|ul_prb|latency99_ms|jain|prach|rrc_ues|sla; "
    "U|cell/ue|5qi|requested|admitted|delivered|cap|max_prb|sinr|bler|mcs|"
    "buffer_kb|pdb; D|cell|requested_total|admitted_total|overload|"
    "candidate_ue|candidate_max_prb|fairness_gain_ppm|active_caps|release_ue; "
    "L|tool|arguments|rejection. P reports the conservative hybrid-controller "
    "cadence, not guardrail state. When expert_ready=0, the visible controller "
    "forces noop. When expert_ready=1, noop or any intervention represented "
    "by a D row is permitted: release_ue maps to max_prb=273, and "
    "candidate_ue/candidate_max_prb is that cell's cap option. Prefer a "
    "release before a new cap; otherwise the candidate with largest overload, "
    "then largest fairness_gain_ppm, then lowest cell_id is the expert default. "
    "Other listed choices are exploration support whose preference is learned "
    "from reward. Never invent a non-noop action absent from D. When no D-row "
    "intervention exists, the visible controller forces noop. Under overload, "
    "an active cap may only be "
    "tightened; relax it only through release_ue. Repeating an exact active "
    "setpoint is ineffective, and the guardrail rejects an identical action "
    "within two logical steps."
)


def system_prompt_for_tier(tier: str) -> str:
    """Return the prompt whose actuator claims match the selected tier."""

    return T2_SYSTEM_PROMPT if tier.upper() == "T2" else SYSTEM_PROMPT


def to_ascii(obs: Observation) -> str:
    """Compact single-block ASCII summary suitable for humans + ``/render``."""
    lines: list[str] = []
    g = obs.global_
    lines.append(
        f"t={obs.t_s:5.1f}s  episode={obs.episode_id}  step={obs.agent_aux.step_idx}  "
        f"tier={g.tier}  "
        f"cells={g.n_cells}  ues={g.n_ues_total}  "
        f"kpi_source={obs.kpi_source_mode}"
    )
    for c in obs.cells:
        lines.append(
            f"  cell {c.cell_id}: prb_dl_p50={c.prb_util_dl_p50:.2f} "
            f"p99={c.prb_util_dl_p99:.2f}  prb_ul={c.prb_util_ul_p50:.2f}  "
            f"latency_p99={c.sched_latency_ms_p99:.1f}ms  "
            f"jain={c.fairness_jain:.2f}  sla_viol={c.sla_violations_last_window}  "
            f"prach_coll={c.prach_collision_rate:.2f}  ues={c.rrc_connected_ues}"
        )
        for ue in c.ues:
            lines.append(
                f"    ue {ue.ue_id} (5qi={ue.qos_5qi}): "
                f"offered={ue.offered_mbps:.1f}Mbps  delivered={ue.delivered_mbps:.1f}Mbps  "
                f"sinr={ue.sinr_db:5.1f}dB  bler={ue.bler:.3f}  "
                f"mcs_mean={ue.mcs_mean:.1f}  buffer={ue.buffer_occupancy_kb:.0f}kB  "
                f"pdb_viol={ue.pdb_violations}"
            )
    aux = obs.agent_aux
    if aux.last_action is not None:
        lines.append(
            f"  prev: tool={aux.last_action.name}  reward={aux.last_reward}  reject={aux.last_rejection or 'no'}"
        )
    return "\n".join(lines)


def to_user_text(obs: Observation) -> str:
    """Compact natural-language summary, fed as the ChatML user message."""
    lines: list[str] = []
    g = obs.global_
    lines.append(f"5G RAN telemetry @ t={obs.t_s:.1f}s (step {obs.agent_aux.step_idx}, tier {g.tier}):")
    lines.append(_source_caveat(obs))
    for c in obs.cells:
        lines.append(
            f"- Cell {c.cell_id}: DL PRB util p50={c.prb_util_dl_p50:.0%}, "
            f"p99={c.prb_util_dl_p99:.0%}; UL PRB util p50={c.prb_util_ul_p50:.0%}; "
            f"sched latency p99 {c.sched_latency_ms_p99:.0f}ms; "
            f"Jain fairness {c.fairness_jain:.2f}; "
            f"PRACH collision rate {c.prach_collision_rate:.0%}; "
            f"{c.rrc_connected_ues} UEs RRC-connected; "
            f"{c.sla_violations_last_window} SLA violation(s) in last 5s."
        )
        for ue in c.ues:
            sla = "SLA-VIOLATION" if ue.pdb_violations > 0 else "ok"
            lines.append(
                f"    UE {ue.ue_id} (5QI {ue.qos_5qi}): offered "
                f"{ue.offered_mbps:.1f} Mbps, delivered {ue.delivered_mbps:.1f} Mbps, "
                f"SINR {ue.sinr_db:.1f} dB, BLER {ue.bler:.0%}, mean MCS "
                f"{ue.mcs_mean:.0f}, buffer {ue.buffer_occupancy_kb:.0f} kB, "
                f"PDB violations {ue.pdb_violations} ({sla})."
            )
    aux = obs.agent_aux
    if aux.last_action is not None:
        rej = f", REJECTED ({aux.last_rejection})" if aux.last_rejection else ""
        lines.append(
            f"Last action: {aux.last_action.name}({aux.last_action.arguments}); last reward: {aux.last_reward}{rej}."
        )
    lines.append("Choose one tool call (or noop) to address congestion now. Output only the tool call.")
    return "\n".join(lines)


def to_compact_user_text(
    obs: Observation,
    *,
    capacity_milli_mbps: int = T2_CELL_CAPACITY_MILLI_MBPS,
) -> str:
    """Lossless compact telemetry for large T2 observations.

    T2 has 24 UE rows per turn.  Repeating prose around every scalar exhausts
    Qwen3-1.7B's native 40,960-token context before a 24-step episode ends.
    This format retains the same decision-relevant fields with stable labels.
    """

    g = obs.global_
    features = build_t2_decision_features(
        obs,
        capacity_milli_mbps=capacity_milli_mbps,
    )
    features_by_cell = {cell.cell_id: cell for cell in features.cells}
    lines = [
        f"T|{obs.t_s:.1f}|{obs.agent_aux.step_idx}|{g.tier}|{obs.kpi_source_mode}",
        format_t2_policy_row(features),
    ]
    for cell in obs.cells:
        lines.append(
            f"C|{cell.cell_id}|{cell.prb_util_dl_p50:.3f}|"
            f"{cell.prb_util_dl_p99:.3f}|{cell.prb_util_ul_p50:.3f}|"
            f"{cell.sched_latency_ms_p99:.1f}|{cell.fairness_jain:.3f}|"
            f"{cell.prach_collision_rate:.3f}|{cell.rrc_connected_ues}|"
            f"{cell.sla_violations_last_window}"
        )
        for ue in cell.ues:
            requested = ue.requested_mbps if ue.requested_mbps is not None else ue.offered_mbps
            admitted = ue.admitted_mbps if ue.admitted_mbps is not None else ue.offered_mbps
            max_prb = ue.prb_cap_max_prb if ue.prb_cap_max_prb is not None else 273
            lines.append(
                f"U|{cell.cell_id}/{ue.ue_id}|{ue.qos_5qi}|"
                f"{format_milli_mbps(quantize_milli_mbps(requested))}|"
                f"{format_milli_mbps(quantize_milli_mbps(admitted))}|"
                f"{format_milli_mbps(quantize_milli_mbps(ue.delivered_mbps))}|"
                f"{'on' if max_prb < 273 else 'off'}|{max_prb}|"
                f"{ue.sinr_db:.2f}|{ue.bler:.3f}|{ue.mcs_mean:.1f}|"
                f"{format_deci_kb(quantize_deci_kb(ue.buffer_occupancy_kb))}|"
                f"{ue.pdb_violations}"
            )
        decision = features_by_cell[cell.cell_id]
        lines.append(format_t2_cell_row(decision))
    aux = obs.agent_aux
    if aux.last_action is not None:
        last_arguments = json.dumps(
            aux.last_action.arguments,
            sort_keys=True,
            separators=(",", ":"),
        )
        lines.append(f"L|{aux.last_action.name}|{last_arguments}|{aux.last_rejection or 'none'}")
    lines.append("A|one_tool_call_or_noop")
    return "\n".join(lines)


def to_policy_text(obs: Observation) -> str:
    """Render the policy input while preserving the frozen T1 representation."""

    if obs.global_.tier.upper() == "T2":
        return to_compact_user_text(obs)
    return to_user_text(obs)


def _source_caveat(obs: Observation) -> str:
    mode = obs.kpi_source_mode
    if mode == "runner_snapshot":
        return (
            "KPI source: runner_snapshot. PRB/UE/throughput fields are "
            "traffic-side estimates from the scenario runner; SINR and BLER "
            "are placeholders, not radio measurements."
        )
    if mode in {"replay", "synthetic", "synthetic_fallback"}:
        return (
            f"KPI source: {mode}. Telemetry is synthetic and should be treated "
            "as benchmark data, not measured OAI/FlexRIC KPM."
        )
    return f"KPI source: {mode}. Check kpi_provenance before treating fields as measured radio KPIs."


def to_chatml(obs: Observation) -> list[dict[str, Any]]:
    """Return the ChatML messages list ``[system, user]``."""
    return [
        {"role": "system", "content": system_prompt_for_tier(obs.global_.tier)},
        {"role": "user", "content": to_policy_text(obs)},
    ]


__all__ = [
    "SYSTEM_PROMPT",
    "T2_SYSTEM_PROMPT",
    "T2_OBSERVATION_RENDER",
    "system_prompt_for_tier",
    "to_chatml",
    "to_user_text",
    "to_compact_user_text",
    "to_policy_text",
    "to_ascii",
]
