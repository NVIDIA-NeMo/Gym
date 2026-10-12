# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light schema for JEEBench rows and flattened verifier metadata.

Required fields mirror JeebenchMetadata. The live verifier additionally checks
canonical gold answers and binds human reviews to the exact response fingerprint.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        min_length=1,
        description="Stable language-specific question identity.",
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["provenance"]},
    )
    language: str = Field(
        min_length=1,
        description="Language code used for metric breakdowns.",
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["metrics"]},
    )
    subject: str = Field(
        min_length=1,
        description="Subject used for metric breakdowns.",
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["metrics"]},
    )
    question_type: Literal["MCQ", "MCQ(multiple)", "Integer", "Numeric"] = Field(
        description="Selects answer extraction and the published scoring rule.",
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["verify", "metrics"]},
    )
    gold: str = Field(
        description="Canonical reference answer, supplied only to the scorer.",
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["verify"]},
    )
    description: str = Field(
        default="",
        description="Original exam and paper name.",
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["provenance"]},
    )
    index: int | str | None = Field(
        default=None,
        description="Question number within the original paper.",
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["provenance"]},
    )
    question: str = Field(
        default="",
        description="Original question, including options used for unambiguous option-text extraction.",
        json_schema_extra={"consumed_by": ["prompt", "verify"]},
    )
    instruction: str | None = Field(
        default=None,
        description="Prepared question with the authors' type-specific CoT instructions.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    uuid: str | None = None
    question_id: str | None = None
    language_name: str | None = None
    subset_for_metrics: str | None = None
    judge_pass_stage: str | None = None
    source_id: str | None = None
    source_revision: str | None = None
    reviewed_answer: str | None = Field(
        default=None,
        description="Optional human-extracted answer; the verifier requires matching review provenance.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    answer_review_id: str | None = None
    reviewed_response_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
