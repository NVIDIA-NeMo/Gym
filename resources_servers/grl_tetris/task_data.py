# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the grl_tetris server.

Gymnasium-family environment: no /verify (the base raises NotImplementedError) — reward comes
from /step. Task fields are consumed at reset() time, arriving untyped as
``EnvResetRequest.model_extra`` (``resources_servers/gymnasium/base.py:33``, ``extra="allow"``);
the wire requires nothing, so every field is Optional. ``seed`` is read via ``metadata.get`` and
env-config overrides via the whitelist in ``app.py:125`` (grid_lookup, action_lookup,
render_mode, dim_x, dim_y, box_type).

KNOWN DATA/CODE MISMATCH: committed rows carry ``dim_board`` (e.g. [4, 4] / [5, 5]) but the
whitelist only forwards ``dim_x``/``dim_y``, and TetrisEnv reads ``config.get("dim_x", 4)`` /
``config.get("dim_y", 4)`` — so ``dim_board`` is silently IGNORED and boards fall back to the
4x4 config default. It is declared below as a deprecated provenance field so validation keeps
passing without codifying the dead field as meaningful; fixing the row/env plumbing is a
separate follow-up.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    game_id: Optional[int] = Field(
        default=None,
        description="Row identifier; never read by the server.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    seed: Optional[int] = Field(
        default=None,
        description="RNG seed passed to TetrisEnv.reset(); unseeded when absent.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    dim_board: Optional[List[int]] = Field(
        default=None,
        deprecated=(
            "Silently ignored: the env-config whitelist only forwards dim_x/dim_y, so boards fall "
            "back to the 4x4 config default regardless of this value."
        ),
        description=(
            "Intended board dimensions [x, y]; present in committed data but NOT consumed by the "
            "server (see module docstring). Use dim_x/dim_y to actually change the board."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    box_type: Optional[int] = Field(
        default=None,
        description="Number of tetromino piece types to sample from, overriding the config default (3).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    dim_x: Optional[int] = Field(
        default=None,
        description="Board width override (config default 4). Absent in committed data.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    dim_y: Optional[int] = Field(
        default=None,
        description="Board height override (config default 4). Absent in committed data.",
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
            'e.g. {"0": "_"}). Absent in committed data.'
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    action_lookup: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Action-id -> label map override (JSON object keys are stringified ints, "
            'e.g. {"0": "Left"}). Absent in committed data.'
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
