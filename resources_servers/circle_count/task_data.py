# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the circle_count server.

Task fields ride at the row top level (no verifier_metadata). Mirrors ``CircleCountVerifyRequest``
(app.py): both fields carry permissive wire defaults (empty list / empty string), so neither is
required here. Wire-identical to circle_click's TaskData but kept separate: consumption differs
(verify() reads only each circle's ``color`` to compute the expected count — a circle dict missing
'color' raises KeyError — while x/y/radius exist because they rendered the task image at data-gen
time) and the guidance dedup groups do not pair the two servers.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    circles: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Circles rendered in the task image, each {'x': int, 'y': int, 'radius': int, "
            "'color': str}. verify() counts elements whose 'color' equals target_color "
            "(hard-indexed); x/y/radius are data-gen provenance for the rendered image."
        ),
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    target_color: str = Field(
        default="",
        description=(
            "Color whose circle count the agent must report via \\boxed{N}; compared against each circle's 'color'."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
