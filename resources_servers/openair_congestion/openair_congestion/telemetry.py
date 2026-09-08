# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neutral telemetry container used by deterministic OpenAir replay."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KpiSnapshot:
    """One synthetic KPI snapshot indexed by cell and UE identifiers."""

    prb_util: dict[int, float] = field(default_factory=dict)
    sinr_db: dict[tuple[int, int], float] = field(default_factory=dict)
    throughput_mbps: dict[tuple[int, int], float] = field(default_factory=dict)
    active_ue_count: dict[int, int] = field(default_factory=dict)
    bler: dict[tuple[int, int], float] = field(default_factory=dict)
    source_mode: str = "unknown"

    def cell_ids(self) -> list[int]:
        return sorted(set(self.prb_util) | set(self.active_ue_count))

    def ues_in_cell(self, cell_id: int) -> list[int]:
        return sorted({ue_id for candidate_cell_id, ue_id in self.sinr_db if candidate_cell_id == cell_id})

    def ue_throughput(self, cell_id: int, ue_id: int, default: float = 0.0) -> float:
        return self.throughput_mbps.get((cell_id, ue_id), default)

    def ue_sinr(self, cell_id: int, ue_id: int, default: float = -10.0) -> float:
        return self.sinr_db.get((cell_id, ue_id), default)

    def ue_bler(self, cell_id: int, ue_id: int, default: float = 0.0) -> float:
        return self.bler.get((cell_id, ue_id), default)


__all__ = ["KpiSnapshot"]
