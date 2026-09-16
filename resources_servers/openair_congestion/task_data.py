# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the openair_congestion server.

Gymnasium-family environment: no /verify — reward accrues per /step and gymnasium_agent sums the
episode return. Task fields are consumed at reset() (``app.py:246``), where they arrive untyped
as ``EnvResetRequest.model_extra`` (``resources_servers/gymnasium/base.py:33``, ``extra="allow"``)
and are filtered via ``metadata.get(key) is not None`` — so every field is Optional on the wire.
reset() hand-validates ``seed`` (non-negative int, bool rejected) and ``max_steps`` (positive
int, bool rejected, then clamped to config.agent_max_steps); those value constraints are mirrored
here. ``regime_mix`` is a dict whose single key varies per row with ``scenario_id``
(prb_exhaustion / bursty / interference) — same schema, dynamic key, deliberately left open.
"""

from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    seed: Optional[int] = Field(
        default=None,
        ge=0,
        description="Episode seed forwarded to the backend; reset() rejects negatives and bools.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    scenario_id: Optional[str] = Field(
        default=None,
        description="Congestion scenario selector, e.g. prb_exhaustion, bursty, interference.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    tier: Optional[str] = Field(
        default=None,
        description='Backend tier; committed data uses "replay". Also drives the per-tier reward contract.',
        json_schema_extra={"consumed_by": ["verify"]},
    )
    difficulty: Optional[float] = Field(
        default=None,
        description="Scenario difficulty knob forwarded to the backend.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    regime_mix: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            'Traffic-regime weighting, regime name -> weight (e.g. {"prb_exhaustion": 1.0}). The '
            "key set is dynamic per scenario_id; do not close it."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    max_steps: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Episode step budget; reset() rejects non-positive ints and bools, falls back to "
            "config.max_steps_default when absent, and caps at config.agent_max_steps."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
