# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the hotpotqa_qa server.

Parent of the required-expected_answer QA family (hotpotqa_qa, equivalence_rule): the wire
requires a single ``expected_answer: str`` — the only field ``verify()`` consumes (here graded
with SQuAD-normalized exact match and token-overlap F1) — and everything else rides through
``extra="allow"``. Heirs import ``ExpectedAnswerTaskDataCore`` instead of redefining the field.
Required-ness mirrors ``HotpotQAQARunRequest`` (app.py): ``expected_answer`` is the sole
wire-declared field; question/id/type/level are HotpotQA provenance passthrough that nothing
reads (no subset metrics on type/level).
"""

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class ExpectedAnswerTaskDataCore(BaseModel):
    """Shared core of the family: a required ground-truth answer string, sole verify() input."""

    model_config = ConfigDict(extra="allow")

    expected_answer: str = Field(
        description="Required ground-truth answer string; the only row field verify() consumes.",
        json_schema_extra={"consumed_by": ["verify"]},
    )


class TaskData(ExpectedAnswerTaskDataCore):
    question: Optional[str] = Field(
        default=None,
        description="Original HotpotQA question text; provenance only (the prompt lives in the input messages).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    id: Optional[Union[int, str]] = Field(
        default=None,
        description="HotpotQA _id (24-hex string in committed data); never read by verify() or metrics.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    type: Optional[str] = Field(
        default=None,
        description="HotpotQA question type ('comparison' or 'bridge'); provenance only, kept str to stay open.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    level: Optional[str] = Field(
        default=None,
        description="HotpotQA difficulty level (e.g. 'hard'); provenance only, kept str to stay open.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
