# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the grl_sokoban server.

Gymnasium-family environment: no /verify (the base raises NotImplementedError) — reward comes
from /step. Task fields are consumed at reset() time, where they arrive completely untyped as
``EnvResetRequest.model_extra`` (``resources_servers/gymnasium/base.py:33``, ``extra="allow"``);
the wire therefore requires nothing, so every field here is Optional. ``seed`` is read via
``metadata.get("seed")`` and the env-config overrides via an ``if key in metadata`` whitelist
(``app.py:104``). The full whitelist is declared below even though committed data only carries
``seed``/``dim_room``/``num_boxes``: rows MAY legitimately override the rest per task.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    level_id: Optional[int] = Field(
        default=None,
        description="Row identifier; never read by the server.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    seed: Optional[int] = Field(
        default=None,
        description="RNG seed passed to SokobanEnv.reset(); unseeded when absent.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    dim_room: Optional[List[int]] = Field(
        default=None,
        description="Board dimensions [rows, cols] overriding the config default (6, 6).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    num_boxes: Optional[int] = Field(
        default=None,
        description="Number of boxes to place, overriding the config default (1).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    max_steps: Optional[int] = Field(
        default=None,
        description="Env step budget override (config default 100). Accepted per-row; absent in committed data.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    search_depth: Optional[int] = Field(
        default=None,
        description="Level-generation search depth override (config default 100). Absent in committed data.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    render_mode: Optional[str] = Field(
        default=None,
        description='Render mode override (config default "text"). Absent in committed data.',
        json_schema_extra={"consumed_by": ["verify"]},
    )
    grid_lookup: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Cell-id -> glyph map override (JSON object keys are stringified ints, "
            'e.g. {"0": "#"}). Absent in committed data.'
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    action_lookup: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Action-id -> label map override (JSON object keys are stringified ints, "
            'e.g. {"1": "Up"}). Absent in committed data.'
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
