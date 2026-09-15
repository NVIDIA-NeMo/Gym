# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the lc_niah server.

Rows are graphwalks-derived long-context needle-in-a-haystack tasks. Only ``expected_answer`` is
typed on the wire (LCNIAHRunRequest, extra='allow'); every other task field arrives as an untyped
extra and is therefore Optional here even though committed rows always carry it.
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
    problem_type: Optional[str] = Field(
        default=None,
        description="Task family ('parents' | 'bfs'); untyped wire extra, unread by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    question: Optional[str] = Field(
        default=None,
        description="The bare question text (also embedded in the prompt); untyped wire extra, unread.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source: Optional[str] = Field(
        default=None,
        description="Origin dataset tag, e.g. 'graphwalks'; untyped wire extra, unread.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
