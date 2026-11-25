# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the audio_eval server.

Mirrors ``AudioEvalVerifyRequest`` (app.py). Rows are built by the benchmark's ``prepare.py``: the immutable source
row (``ng_source_row``) carries everything the scorer reads and ``ng_source_row_sha256`` binds a verify request to
it. Any other top-level key (profile-specific descriptive fields copied from the source row) is passed through.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """One prepared audio-benchmark row: the profile, the sample identity and the digest-bound source row."""

    model_config = ConfigDict(extra="allow")

    benchmark_profile: str = Field(
        description="Profile name in the benchmark's profiles.json; selects the scorer and its settings.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    sample_id: str = Field(
        description="``<profile>:<1-based index>``; unique within the benchmark file.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    task_id: str = Field(
        description="Stable task identity, equal to ``sample_id``: Gym keys resume and comparison on it, not on the "
        "row position, which shifts when a benchmark is re-prepared, narrowed or reordered.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    ng_source_row: dict = Field(
        description="Immutable source row the scorer reads (reference, subset, scorer fields).",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    ng_source_row_sha256: str = Field(
        description="SHA-256 of the canonical JSON of ``ng_source_row``; verify rejects a mismatch.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
