# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic schemas for ``openair_congestion_v1``.

Models are frozen against ``schema_version="1.0.0"``. Bump that string and the
environment name suffix for any breaking contract change.

Highlights:

- ``ToolCall`` accepts both the simple OpenAI shape (``{"name": ..., "arguments": {...}}``)
  and the chat-completions shape (``{"id": ..., "type": "function", "function":
  {"name": ..., "arguments": "..."}}``). The latter is what vLLM emits when the
  policy uses chat-completions; we normalise to the simple shape internally.
- ``UEObservation.qos_5qi`` uses the JSON alias ``5qi`` while remaining a valid
  Python identifier.
- Numeric fields carry explicit bounds so malformed observations fail fast.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    field_validator,
    model_validator,
)

from . import SCHEMA_VERSION
from .tools import MAX_CELLS, MAX_UES, TOOL_NAMES


KPI_PROVENANCE_VERSION: str = "1.1.0"
SUPPORTED_REGIMES: tuple[str, ...] = (
    "prb_exhaustion",
    "bursty",
    "interference",
    "prach_storm",
    "qos_competition",
)

# Observation KPI provenance. Synthetic replay and recorded datasets share one
# stable shape; source-mode stamping distinguishes generated from supplied KPIs.
KPI_PROVENANCE_V1: dict[str, dict[str, str]] = {
    "cells[].prb_util_dl_p50": {
        "kind": "raw_exporter",
        "source": "source observation or synthetic replay snapshot",
        "notes": "DL PRB utilization supplied by the selected backend.",
    },
    "cells[].prb_util_dl_p99": {
        "kind": "derived",
        "source": "prb_util_dl_p50",
        "notes": "Congestion heuristic: max(p50, min(1, p50 * 1.15 + 0.02)).",
    },
    "cells[].prb_util_ul_p50": {
        "kind": "derived",
        "source": "prb_util_dl_p50",
        "notes": "UL/DL heuristic: min(1, prb_util_dl_p50 * 0.4).",
    },
    "cells[].sched_latency_ms_p99": {
        "kind": "derived",
        "source": "prb_util_dl_p99",
        "notes": "Scheduler-delay heuristic: 5 + 20 * prb_util_dl_p99; not FlexRIC DRB delay.",
    },
    "cells[].rrc_connected_ues": {
        "kind": "raw_exporter",
        "source": "source observation or synthetic replay snapshot",
        "notes": "Active UE count supplied by the selected backend.",
    },
    "cells[].prach_collision_rate": {
        "kind": "derived",
        "source": "rrc_connected_ues",
        "notes": "Load heuristic: 0 below 8 active UEs, then capped quadratic growth.",
    },
    "cells[].fairness_jain": {
        "kind": "derived",
        "source": "cells[].ues[].delivered_mbps",
        "notes": "Jain index over per-UE delivered throughput in the cell.",
    },
    "cells[].sla_violations_last_window": {
        "kind": "derived",
        "source": "cells[].ues[].pdb_violations",
        "notes": "Count of per-UE packet-delay-budget violations in the observation window.",
    },
    "cells[].ues[].offered_mbps": {
        "kind": "scenario",
        "source": "task scenario or recorded dataset",
        "notes": "Requested traffic load sampled at reset; not a measured KPI.",
    },
    "cells[].ues[].delivered_mbps": {
        "kind": "raw_exporter",
        "source": "source observation or synthetic replay snapshot",
        "notes": "Delivered DL throughput supplied by the selected backend.",
    },
    "cells[].ues[].bler": {
        "kind": "raw_exporter",
        "source": "source observation or synthetic replay snapshot",
        "notes": "Block error rate supplied by the selected backend.",
    },
    "cells[].ues[].mcs_mean": {
        "kind": "derived",
        "source": "cells[].ues[].sinr_db",
        "notes": "MCS heuristic from SINR; replace with scheduler/KPM data when available.",
    },
    "cells[].ues[].sinr_db": {
        "kind": "raw_exporter",
        "source": "source observation or synthetic replay snapshot",
        "notes": "DL SINR supplied by the selected backend.",
    },
    "cells[].ues[].buffer_occupancy_kb": {
        "kind": "derived",
        "source": "offered_mbps - delivered_mbps",
        "notes": "Backlog heuristic: max(0, offered - delivered) * 50.",
    },
    "cells[].ues[].pdb_violations": {
        "kind": "derived",
        "source": "cells[].ues[].buffer_occupancy_kb",
        "notes": "Packet-delay-budget violation heuristic: 1 when buffer exceeds 500 kB.",
    },
    "cells[].ues[].5qi": {
        "kind": "scenario",
        "source": "task scenario or recorded dataset",
        "notes": "QoS class sampled at reset; not a measured KPI.",
    },
}


def default_kpi_provenance() -> dict[str, dict[str, str]]:
    """Return a deep copy so Observation instances cannot share mutable state."""

    return {field: dict(entry) for field, entry in KPI_PROVENANCE_V1.items()}


def kpi_provenance_for_source_mode(source_mode: str) -> dict[str, dict[str, str]]:
    """Return source-mode-aware provenance for the observation contract.

    This helper marks synthetic replay fields explicitly and treats unknown
    dataset source labels conservatively as estimates/placeholders.
    """

    provenance = default_kpi_provenance()
    mode = (source_mode or "unknown").strip().lower()
    raw_fields = (
        "cells[].prb_util_dl_p50",
        "cells[].rrc_connected_ues",
        "cells[].ues[].delivered_mbps",
        "cells[].ues[].bler",
        "cells[].ues[].sinr_db",
    )

    if mode in {"replay", "synthetic", "synthetic_fallback"}:
        for field in raw_fields:
            provenance[field]["kind"] = "synthetic"
            provenance[field]["notes"] = (
                f"{provenance[field]['notes']} Source mode {mode!r} is generated "
                "for local/replay testing, not a lab-measured RAN KPI."
            )
        return provenance

    if mode in {"unknown", ""} or mode not in {"measured", "recorded"}:
        for field in (
            "cells[].prb_util_dl_p50",
            "cells[].rrc_connected_ues",
            "cells[].ues[].delivered_mbps",
        ):
            provenance[field]["kind"] = "estimate"
            provenance[field]["notes"] = (
                f"{provenance[field]['notes']} Source mode {mode!r} is not "
                "recognized; treat this field as an estimate until the exporter "
                "advertises a known mode."
            )
        for field in ("cells[].ues[].bler", "cells[].ues[].sinr_db"):
            provenance[field]["kind"] = "placeholder"
            provenance[field]["notes"] = (
                f"{provenance[field]['notes']} Source mode {mode!r} is not "
                "recognized; do not treat this as a measured radio KPI."
            )
        return provenance

    return provenance


# --- Frozen baseclass -------------------------------------------------------


class _Base(BaseModel):
    """Pydantic v2 base: strict types, no extra fields, JSON-serialisable."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )


class KPIProvenanceEntry(_Base):
    kind: Literal[
        "raw_exporter",
        "derived",
        "scenario",
        "synthetic",
        "estimate",
        "placeholder",
    ]
    source: str = Field(..., min_length=1)
    notes: str = Field(..., min_length=1)


# --- Observation: per-UE / per-cell / global / agent_aux --------------------


class UEObservation(_Base):
    ue_id: NonNegativeInt = Field(..., le=MAX_UES - 1)
    offered_mbps: NonNegativeFloat
    delivered_mbps: NonNegativeFloat
    bler: float = Field(..., ge=0.0, le=1.0)
    mcs_mean: float = Field(..., ge=0.0, le=27.0)
    sinr_db: float = Field(..., ge=-20.0, le=40.0)
    buffer_occupancy_kb: NonNegativeFloat
    pdb_violations: NonNegativeInt
    qos_5qi: int = Field(..., ge=1, le=127, alias="5qi")


class CellObservation(_Base):
    cell_id: NonNegativeInt = Field(..., le=MAX_CELLS - 1)
    prb_util_dl_p50: float = Field(..., ge=0.0, le=1.0)
    prb_util_dl_p99: float = Field(..., ge=0.0, le=1.0)
    prb_util_ul_p50: float = Field(..., ge=0.0, le=1.0)
    sched_latency_ms_p99: NonNegativeFloat
    rrc_connected_ues: NonNegativeInt = Field(..., le=MAX_UES)
    prach_collision_rate: float = Field(..., ge=0.0, le=1.0)
    fairness_jain: float = Field(..., ge=0.0, le=1.0)
    sla_violations_last_window: NonNegativeInt
    ues: list[UEObservation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_p99_ge_p50(self) -> "CellObservation":
        if self.prb_util_dl_p99 + 1e-6 < self.prb_util_dl_p50:
            raise ValueError(
                "prb_util_dl_p99 must be >= prb_util_dl_p50 "
                f"(got p50={self.prb_util_dl_p50}, p99={self.prb_util_dl_p99})"
            )
        ue_ids = [ue.ue_id for ue in self.ues]
        if len(ue_ids) != len(set(ue_ids)):
            raise ValueError(f"cell {self.cell_id} contains duplicate ue_id values")
        if self.rrc_connected_ues != len(self.ues):
            raise ValueError(
                f"cell {self.cell_id} rrc_connected_ues="
                f"{self.rrc_connected_ues} does not match "
                f"len(ues)={len(self.ues)}"
            )
        return self


def _validate_regime_mix(v: dict[str, float]) -> dict[str, float]:
    if not v:
        return v
    cleaned: dict[str, float] = {}
    for name, w in v.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"regime_mix key must be a non-empty string: {name!r}")
        if name not in SUPPORTED_REGIMES:
            raise ValueError(f"unknown regime_mix key {name!r}; valid: {SUPPORTED_REGIMES}")
        try:
            weight = float(w)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"regime_mix weight must be numeric: {w!r}") from exc
        if not math.isfinite(weight) or weight < 0.0 or weight > 1.0:
            raise ValueError(f"regime_mix weight out of [0,1]: {w}")
        cleaned[name] = weight
    s = sum(cleaned.values())
    if s <= 0.0:
        raise ValueError("regime_mix must contain at least one positive weight")
    if abs(s - 1.0) > 1e-3:
        raise ValueError(f"regime_mix must sum to 1.0 (got {s})")
    return cleaned


class GlobalObservation(_Base):
    n_cells: NonNegativeInt = Field(..., ge=1, le=MAX_CELLS)
    n_ues_total: NonNegativeInt = Field(..., le=MAX_UES)
    difficulty: float = Field(..., ge=0.0, le=1.0)
    regime_mix: dict[str, float] = Field(default_factory=dict)
    tier: Literal["replay"] = "replay"

    @field_validator("regime_mix")
    @classmethod
    def _mix_sums_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        return _validate_regime_mix(v)


class LastActionEcho(_Base):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentAux(_Base):
    last_action: Optional[LastActionEcho] = None
    last_reward: Optional[float] = None
    last_rejection: Optional[str] = None
    step_idx: NonNegativeInt = 0


class Observation(_Base):
    schema_version: str = SCHEMA_VERSION
    t_s: NonNegativeFloat
    episode_id: str = Field(..., min_length=1, max_length=64)
    cells: list[CellObservation]
    global_: GlobalObservation = Field(..., alias="global")
    kpi_source_mode: str = Field(default="unknown", min_length=1, max_length=64)
    kpi_provenance_version: str = KPI_PROVENANCE_VERSION
    kpi_provenance: dict[str, KPIProvenanceEntry] = Field(
        default_factory=lambda: kpi_provenance_for_source_mode("unknown"),
        validate_default=True,
    )
    agent_aux: AgentAux = Field(default_factory=AgentAux)

    @model_validator(mode="before")
    @classmethod
    def _fill_source_mode_provenance(cls, data: Any) -> Any:
        if isinstance(data, dict) and "kpi_provenance" not in data:
            source_mode = str(data.get("kpi_source_mode") or "unknown")
            data = dict(data)
            data["kpi_provenance"] = kpi_provenance_for_source_mode(source_mode)
        return data

    @field_validator("schema_version")
    @classmethod
    def _pin(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(f"schema_version mismatch: got {v!r}, expected {SCHEMA_VERSION!r}")
        return v

    @field_validator("kpi_provenance_version")
    @classmethod
    def _pin_provenance_version(cls, v: str) -> str:
        if v != KPI_PROVENANCE_VERSION:
            raise ValueError(f"kpi_provenance_version mismatch: got {v!r}, expected {KPI_PROVENANCE_VERSION!r}")
        return v

    @field_validator("kpi_provenance")
    @classmethod
    def _provenance_has_contract_keys(
        cls,
        v: dict[str, KPIProvenanceEntry],
    ) -> dict[str, KPIProvenanceEntry]:
        missing = set(KPI_PROVENANCE_V1) - set(v)
        if missing:
            raise ValueError(f"kpi_provenance missing keys: {sorted(missing)}")
        extra = set(v) - set(KPI_PROVENANCE_V1)
        if extra:
            raise ValueError(f"kpi_provenance unknown keys: {sorted(extra)}")
        return v

    @model_validator(mode="after")
    def _provenance_matches_source_mode(self) -> "Observation":
        expected = kpi_provenance_for_source_mode(self.kpi_source_mode)
        for field in (
            "cells[].prb_util_dl_p50",
            "cells[].rrc_connected_ues",
            "cells[].ues[].delivered_mbps",
            "cells[].ues[].bler",
            "cells[].ues[].sinr_db",
        ):
            expected_kind = expected[field]["kind"]
            actual_kind = self.kpi_provenance[field].kind
            if actual_kind != expected_kind:
                raise ValueError(
                    f"kpi_provenance[{field!r}].kind={actual_kind!r} "
                    f"does not match kpi_source_mode={self.kpi_source_mode!r}; "
                    f"expected {expected_kind!r}"
                )
        return self

    @model_validator(mode="after")
    def _topology_is_consistent(self) -> "Observation":
        cell_ids = [cell.cell_id for cell in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("observation contains duplicate cell_id values")
        if self.global_.n_cells != len(self.cells):
            raise ValueError(f"global.n_cells={self.global_.n_cells} does not match len(cells)={len(self.cells)}")
        n_ues = sum(len(cell.ues) for cell in self.cells)
        if self.global_.n_ues_total != n_ues:
            raise ValueError(f"global.n_ues_total={self.global_.n_ues_total} does not match observed UE count {n_ues}")
        return self


# --- Action ----------------------------------------------------------------


class ToolCall(_Base):
    """Normalised tool call. Accepts both OpenAI chat-completions shape and the
    simple ``{name, arguments}`` shape.

    The chat-completions shape is::

        {"id": "call_0", "type": "function",
         "function": {"name": "set_scheduler_policy",
                      "arguments": "{\\"cell_id\\": 0, \\"policy\\": \\"PF\\"}"}}

    where ``arguments`` is a JSON-encoded string. We normalise it to a dict.
    """

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_known(cls, v: str) -> str:
        if v not in TOOL_NAMES:
            raise ValueError(f"unknown tool {v!r}; valid: {TOOL_NAMES}")
        return v

    @model_validator(mode="before")
    @classmethod
    def _from_openai_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "function" in data and isinstance(data["function"], dict):
            fn = data["function"]
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"tool_calls[].function.arguments must be a JSON object string or a dict; got {args!r} ({e})"
                    ) from e
            return {"name": fn.get("name"), "arguments": args}
        return data


# --- Episode metadata -------------------------------------------------------


class EpisodeMeta(_Base):
    episode_id: str
    seed: int
    difficulty: float = Field(..., ge=0.0, le=1.0)
    regime_mix: dict[str, float] = Field(default_factory=dict)
    tier: Literal["replay"] = "replay"
    scenario_id: Optional[str] = None
    created_at: float = Field(default_factory=lambda: time.time())
    max_steps: NonNegativeInt = 60


__all__ = [
    "UEObservation",
    "CellObservation",
    "GlobalObservation",
    "AgentAux",
    "LastActionEcho",
    "Observation",
    "ToolCall",
    "EpisodeMeta",
]
