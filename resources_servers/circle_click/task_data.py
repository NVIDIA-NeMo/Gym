# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the circle_click server.

Task fields ride at the row top level (no verifier_metadata). Mirrors
``CircleClickVerifyRequest`` (app.py): both fields carry permissive wire defaults (empty list /
empty string), so neither is required here — a row missing them silently verifies to reward 0
rather than erroring. Circle list items stay ``Dict[str, Any]`` exactly like the wire; the
per-circle ``{x, y, radius, color}`` shape is enforced only by verify()'s runtime indexing
(``_point_in_circle`` hard-indexes x/y/radius, the color match hard-indexes color).
"""

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    circles: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Circles rendered in the task image, each {'x': int, 'y': int, 'radius': int, "
            "'color': str}. verify() hard-indexes all four keys to hit-test the clicked point "
            "against circles of the target color."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    target_color: str = Field(
        default="",
        description="Color of the circle the agent must click; compared against each circle's 'color'.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
