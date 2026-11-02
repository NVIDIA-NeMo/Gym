# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the graphwalks server.

Task fields ride at the row top level (no verifier_metadata in data or code). ``GraphWalksCore``
holds the fields shared with the lc_niah server, whose rows are derived from graphwalks data;
lc_niah's TaskData imports it instead of redefining. ``problem_type`` is deliberately NOT part of
the core because its required-ness differs per wire: GraphWalksVerifyRequest requires it, while
lc_niah's wire only sees it as an untyped extra.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class GraphWalksCore(BaseModel):
    """Fields shared by graphwalks and its lc_niah derivative."""

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


class TaskData(GraphWalksCore):
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
