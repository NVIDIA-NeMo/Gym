# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the gui_coordinate server.

Ground truth rides as top-level row fields; there is no verifier_metadata. Mirrors
``GuiCoordinateRunRequest`` (app.py): ``expected_answer`` wire-required, ``max_dist`` optional
with the wire's 0.15 default, ``metadata`` a free-form optional dict. Rows carry no agent_ref and
are large (~28KB): the screenshot travels inside ``responses_create_params.input`` as a base64
``input_image`` part, which is framework-owned and not typed here.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_answer: str = Field(
        description=(
            "Gold click target as 'x,y' with coordinates normalized to 0-1 floats (parsed by "
            "_parse_gt). Compared against the model's <point>(x,y)</point> output (which is "
            "divided by 1000) under the max_dist euclidean threshold."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    max_dist: float = Field(
        default=0.15,
        description=(
            "Zero-reward cutoff: reward is 0.0 when the euclidean distance (normalized 0-1 "
            "space) reaches max_dist, else shaped as (1 - dist/max_dist)^2 (1.0 only at exact "
            "match). Optional on the wire but present in all committed rows."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Free-form provenance dict, never read by verify(). Observed subkeys in committed "
            "data: target_color (str), source (str)."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
