# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic scenario and observation helpers for synthetic replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import SCHEMA_VERSION
from .schemas import (
    AgentAux,
    CellObservation,
    EpisodeMeta,
    LastActionEcho,
    Observation,
    ToolCall,
    UEObservation,
    kpi_provenance_for_source_mode,
)
from .telemetry import KpiSnapshot
from .tools import MAX_CELLS, MAX_UES


# --- Episode state ----------------------------------------------------------


@dataclass
class _ScenarioFingerprint:
    """Slim view of the sampler output we actually need at /step time."""

    n_cells: int = 2
    n_ues_total: int = 4
    tier: str = "replay"
    regime_mix: dict[str, float] = field(default_factory=dict)
    # Immutable per-UE demand used to construct synthetic observations.
    offered_mbps: dict[tuple[int, int], float] = field(default_factory=dict)
    qos_5qi: dict[tuple[int, int], int] = field(default_factory=dict)
    cell_capacity_mbps: float = 60.0


@dataclass
class ObservationContext:
    """Minimal episode context required to render one synthetic observation."""

    episode_id: str
    meta: EpisodeMeta
    fingerprint: _ScenarioFingerprint
    step_idx: int = 0
    last_action: Optional[ToolCall] = None
    last_reward: Optional[float] = None
    last_rejection: Optional[str] = None


# --- Self-contained scenario sampler --------------------------------------


def _sample_scenario(
    *,
    seed: int,
    difficulty: float,
    regime_mix: Optional[dict[str, float]],
    tier: str,
) -> _ScenarioFingerprint:
    """Build the deterministic fingerprint used by synthetic replay.

    This stays dependency-free so identical task rows produce identical
    episodes regardless of unrelated packages in the host environment.
    """
    n_cells, n_ues_total = _tier_dims(tier)
    fp = _ScenarioFingerprint(
        n_cells=n_cells,
        n_ues_total=n_ues_total,
        tier=tier,
        regime_mix=regime_mix or {"prb_exhaustion": 0.6, "bursty": 0.4},
    )
    fp.cell_capacity_mbps = 60.0
    normalized_difficulty = max(0.0, min(1.0, float(difficulty)))
    regime = fp.regime_mix
    for cell_id in range(n_cells):
        ues_in_cell = max(1, n_ues_total // max(1, n_cells))
        # Low difficulty stays below capacity; the checked-in medium/high
        # examples are oversubscribed enough to make control meaningful.
        load_ratio = 0.55 + 0.7 * normalized_difficulty
        load_ratio += 0.15 * float(regime.get("prb_exhaustion", 0.0))
        load_ratio += 0.08 * float(regime.get("bursty", 0.0))
        load_ratio += 0.03 * float(regime.get("qos_competition", 0.0))
        offered_per_ue = fp.cell_capacity_mbps * load_ratio / ues_in_cell
        for ue_idx in range(ues_in_cell):
            key = (cell_id, ue_idx)
            # Alternating demand/classes makes QoS competition observable while
            # preserving the cell's total requested service.
            qos_weight = float(regime.get("qos_competition", 0.0))
            skew = (0.6 if ue_idx % 2 == 0 else 1.4) if qos_weight > 0.0 else 1.0
            offered = offered_per_ue * (1.0 + qos_weight * (skew - 1.0))
            fp.offered_mbps[key] = offered
            fp.qos_5qi[key] = 1 if qos_weight > 0.0 and ue_idx % 2 == 0 else 9
    return fp


def _tier_dims(tier: str) -> tuple[int, int]:
    """Return the synthetic topology for a supported task tier."""

    try:
        return {"replay": (2, 4)}[tier]
    except KeyError as exc:
        raise ValueError(f"unsupported tier {tier!r}") from exc


# --- Observation builder ---------------------------------------------------


def _jain(values: list[float]) -> float:
    if not values or all(v <= 0.0 for v in values):
        return 1.0
    s = sum(values)
    n = len(values)
    sq = sum(v * v for v in values)
    return float((s * s) / max(1e-9, n * sq))


def _build_observation(
    *,
    snapshot: KpiSnapshot,
    episode: ObservationContext,
    t_s: float,
) -> Observation:
    fp = episode.fingerprint
    cell_ids = snapshot.cell_ids() or list(range(fp.n_cells))
    cell_ids = [c for c in cell_ids if 0 <= c < MAX_CELLS]
    if not cell_ids:
        cell_ids = list(range(min(MAX_CELLS, max(1, fp.n_cells))))

    cells: list[CellObservation] = []
    for cell_id in cell_ids:
        prb_dl_p50 = float(snapshot.prb_util.get(cell_id, 0.0))
        prb_dl_p50 = max(0.0, min(1.0, prb_dl_p50))
        prb_dl_p99 = max(prb_dl_p50, min(1.0, prb_dl_p50 * 1.15 + 0.02))
        prb_ul_p50 = max(0.0, min(1.0, prb_dl_p50 * 0.4))
        sched_latency_ms_p99 = 5.0 + 20.0 * prb_dl_p99

        snap_n_ues = max(
            0,
            min(MAX_UES, int(snapshot.active_ue_count.get(cell_id, 0))),
        )
        ue_ids_in_cell = snapshot.ues_in_cell(cell_id)
        explicit_zero_ues = cell_id in snapshot.active_ue_count and snap_n_ues == 0
        if explicit_zero_ues:
            ue_ids_in_cell = []
        elif not ue_ids_in_cell:
            # Fallback: derive from fingerprint
            ue_ids_in_cell = sorted(u for (c, u) in fp.offered_mbps.keys() if c == cell_id)
        ue_ids_in_cell = [u for u in ue_ids_in_cell if 0 <= u < MAX_UES]
        if not ue_ids_in_cell and not explicit_zero_ues:
            ue_ids_in_cell = [0]

        prach_collision_rate = 0.0 if snap_n_ues < 8 else min(0.5, 0.01 * (snap_n_ues - 8) ** 2)
        prach_weight = max(
            0.0,
            min(1.0, float(fp.regime_mix.get("prach_storm", 0.0))),
        )
        if prach_weight > 0.0:
            planned_arrivals = int(8 + 24 * float(episode.meta.difficulty))
            planned_pressure = min(0.5, 0.01 * max(0, planned_arrivals - 8) ** 2)
            prach_collision_rate = max(
                prach_collision_rate,
                prach_weight * planned_pressure,
            )

        ues: list[UEObservation] = []
        thru: list[float] = []
        for ue_id in ue_ids_in_cell:
            delivered = float(snapshot.ue_throughput(cell_id, ue_id, default=0.0))
            sinr = float(snapshot.ue_sinr(cell_id, ue_id, default=10.0))
            bler = float(snapshot.ue_bler(cell_id, ue_id, default=0.0))
            offered = float(fp.offered_mbps.get((cell_id, ue_id), max(delivered, 1.0)))
            mcs_mean = max(0.0, min(27.0, (max(sinr, -10.0) + 5.0) * 1.2))
            buffer_occupancy_kb = max(0.0, (offered - delivered) * 50.0)
            pdb_violations = 1 if buffer_occupancy_kb > 500.0 else 0
            qos_5qi = int(fp.qos_5qi.get((cell_id, ue_id), 9))

            ues.append(
                UEObservation.model_validate(
                    {
                        "ue_id": ue_id,
                        "offered_mbps": offered,
                        "delivered_mbps": delivered,
                        "bler": max(0.0, min(1.0, bler)),
                        "mcs_mean": mcs_mean,
                        "sinr_db": max(-20.0, min(40.0, sinr)),
                        "buffer_occupancy_kb": buffer_occupancy_kb,
                        "pdb_violations": pdb_violations,
                        "5qi": qos_5qi,
                    }
                )
            )
            thru.append(delivered)

        cells.append(
            CellObservation(
                cell_id=cell_id,
                prb_util_dl_p50=prb_dl_p50,
                prb_util_dl_p99=prb_dl_p99,
                prb_util_ul_p50=prb_ul_p50,
                sched_latency_ms_p99=sched_latency_ms_p99,
                rrc_connected_ues=len(ues),
                prach_collision_rate=prach_collision_rate,
                fairness_jain=_jain(thru),
                sla_violations_last_window=sum(1 for u in ues if u.pdb_violations > 0),
                ues=ues,
            )
        )

    aux = AgentAux(
        last_action=(
            LastActionEcho(name=episode.last_action.name, arguments=episode.last_action.arguments)
            if episode.last_action is not None
            else None
        ),
        last_reward=episode.last_reward,
        last_rejection=episode.last_rejection,
        step_idx=episode.step_idx,
    )

    return Observation.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "t_s": float(t_s),
            "episode_id": episode.episode_id,
            "cells": [c.model_dump(by_alias=True) for c in cells],
            "global": {
                "n_cells": len(cells),
                "n_ues_total": sum(len(c.ues) for c in cells),
                "difficulty": episode.meta.difficulty,
                "regime_mix": episode.fingerprint.regime_mix,
                "tier": episode.meta.tier,
            },
            "kpi_source_mode": snapshot.source_mode or "unknown",
            "kpi_provenance": kpi_provenance_for_source_mode(snapshot.source_mode or "unknown"),
            "agent_aux": aux.model_dump(),
        }
    )


__all__ = [
    "ObservationContext",
    "_ScenarioFingerprint",
    "_build_observation",
    "_sample_scenario",
    "_tier_dims",
]
