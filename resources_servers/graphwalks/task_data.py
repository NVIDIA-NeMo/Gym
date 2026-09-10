# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the graphwalks server.

Task fields ride at the row top level (no verifier_metadata in data or code). ``problem_type`` is
wire-required (GraphWalksVerifyRequest) and drives the per-subset metrics breakdown.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_answer: str = Field(
        description=(
            "JSON-encoded list of expected node-name strings, e.g. '[\"node_1\", \"node_2\"]' or '[]'. "
            "verify() json.loads it into a set for F1 scoring; parse failure falls back to the empty set."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    n_tokens: Optional[int] = Field(
        default=None,
        description="Prompt size in tokens; wire-optional passthrough, never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    problem_type: str = Field(
        description=(
            "Task family, e.g. 'parents' or 'bfs'. Wire-required (GraphWalksVerifyRequest) but unread by "
            "verify(); compute_metrics() uses it as the per-subset breakdown key."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    prompt_chars: Optional[int] = Field(
        default=None,
        description="Prompt size in characters; wire-optional passthrough, never read.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
