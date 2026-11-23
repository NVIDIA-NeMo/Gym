# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render one compact, deterministic policy message per observation."""

from __future__ import annotations

from .schemas import Observation


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


def to_policy_text(obs: Observation) -> str:
    """Render the supported replay policy input."""

    return to_user_text(obs)


def _source_caveat(obs: Observation) -> str:
    mode = obs.kpi_source_mode
    if mode in {"replay", "synthetic", "synthetic_fallback"}:
        return (
            f"KPI source: {mode}. Telemetry is synthetic and should be treated "
            "as benchmark data, not measured OAI/FlexRIC KPM."
        )
    return f"KPI source: {mode}. Check kpi_provenance before treating fields as measured radio KPIs."


__all__ = [
    "to_user_text",
    "to_policy_text",
]
