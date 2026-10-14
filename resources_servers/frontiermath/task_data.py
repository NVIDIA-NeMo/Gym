# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-owned fields for FrontierMath's flat dataset rows and verify/run requests."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str = Field(description="Translated problem statement.", json_schema_extra={"consumed_by": ["prompt"]})
    expected_answer: str = Field(
        description="Exact reference answer, excluded from model input.", json_schema_extra={"consumed_by": ["verify"]}
    )
    answer_type: Literal["integer", "expression"] = Field(
        description="Exact comparison mode.", json_schema_extra={"consumed_by": ["verify"]}
    )
    row_id: str = Field(
        description="Stable source problem ID shared across languages.",
        json_schema_extra={"consumed_by": ["metrics", "provenance"]},
    )
    language_code: str = Field(
        description="Language code used for grouped scores.", json_schema_extra={"consumed_by": ["metrics"]}
    )
    tier: int = Field(description="Source FrontierMath tier.", json_schema_extra={"consumed_by": ["provenance"]})
    judge_pass_stage: str = Field(
        description="Translation-review status, independent of answer grading.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    human_evaluation_pending: bool = Field(
        description="Whether human translation review is pending.", json_schema_extra={"consumed_by": ["metrics"]}
    )
    answer_key_version: str = Field(
        description="Version of the reference answer key.", json_schema_extra={"consumed_by": ["provenance"]}
    )

    row_index: int | None = Field(
        default=None, description="Source problem index.", json_schema_extra={"consumed_by": ["provenance"]}
    )
    problem: str | None = Field(
        default=None, description="Original translated statement.", json_schema_extra={"consumed_by": ["provenance"]}
    )
    language: str | None = Field(
        default=None,
        description="Language name inserted into the prompt.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    human_review_reason: str | None = Field(
        default=None,
        description="Reason for human translation review.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    task_id: str | None = Field(
        default=None, description="Language-specific task ID.", json_schema_extra={"consumed_by": ["provenance"]}
    )
