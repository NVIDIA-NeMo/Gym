# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the perceptionbench resources server."""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str | None = Field(
        default=None,
        description="Stable upstream task identifier (perceptionbench_<index>).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    question: str = Field(
        min_length=1,
        description="Raw upstream problem text with <|image_N|> placeholders intact, passed to the judge prompt.",
        json_schema_extra={"consumed_by": ["prompt", "verify"], "legacy_location": "verifier_metadata"},
    )
    answer: str = Field(
        min_length=1,
        description="Upstream reference answer passed to the judge prompt.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    category: str = Field(
        min_length=1,
        description="Atomic visual-perception error category used for per-category accuracy breakdown.",
        json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"},
    )
    source_bmk: str | None = Field(
        default=None,
        description="Source benchmark from which the item was decomposed, or 'NA' for newly authored items.",
        json_schema_extra={"consumed_by": ["metrics", "provenance"], "legacy_location": "verifier_metadata"},
    )
    index: int | None = Field(
        default=None,
        description="Upstream integer item index (0..2999).",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    source_idx: int | str | None = Field(
        default=None,
        description="Upstream source item identifier when decomposed from another benchmark.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
