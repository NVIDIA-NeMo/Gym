# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset fields consumed by Vision2Code's reference-image verifier.

The verifier requires these fields inside ``verifier_metadata``. The schema describes
their flattened form for Gym's dataset validation while preserving that wire location.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """Reference image identity, scoring rubric, dimensions, and integrity checksum."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(
        description="Task identifier, also copied to the row top level by preparation.",
        json_schema_extra={"legacy_location": "verifier_metadata"},
    )
    image_path: str = Field(
        description="Reference image path accessible to the verifier, outside the rendering sandbox.",
        json_schema_extra={"legacy_location": "verifier_metadata"},
    )
    rubric: Literal[
        "ChartQA",
        "dvqa",
        "figureqa",
        "ChemVQA-2K",
        "ChemVQA-2K-spacing-v1",
        "DocVQA",
        "DocVQA-v2",
        "EEE-Bench",
        "EEE-Bench-v2",
        "GEOQA_8K_R1V",
        "Geoperception",
        "geometry3k",
        "Graph-Algorithms",
        "GraphVQA-Swift",
        "matplotlib",
        "OlympiadBench",
        "Physics",
        "spatialvlm_qa",
        "generic",
    ] = Field(
        description="Rubric used to select the judge prompt and score aggregation.",
        json_schema_extra={"legacy_location": "verifier_metadata"},
    )
    width: int = Field(
        gt=0,
        description="Expected reference and output image width in pixels.",
        json_schema_extra={"legacy_location": "verifier_metadata"},
    )
    height: int = Field(
        gt=0,
        description="Expected reference and output image height in pixels.",
        json_schema_extra={"legacy_location": "verifier_metadata"},
    )
    sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="SHA-256 of the reference bytes, checked before rendering.",
        json_schema_extra={"legacy_location": "verifier_metadata"},
    )
