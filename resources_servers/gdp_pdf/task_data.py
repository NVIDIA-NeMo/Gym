# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the GDP.pdf environment."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class RubricCriterion(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    criterion: str
    criterion_type: Optional[str] = None
    criterion_severity: Optional[Any] = None
    criterion_implicitness: Optional[Any] = None
    criterion_subjectiveness: Optional[Any] = None
    criterion_failure_mode: Optional[str] = None


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: Optional[str] = Field(
        default=None,
        description="Prompt-template copy of task_prompt used to seed responses_create_params.input.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    task_id: str = Field(
        description="Stable source task identifier.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    task_prompt: str = Field(
        description="Original professional task shown to the policy and each rubric judge.",
        json_schema_extra={"consumed_by": ["prompt", "verify"], "legacy_location": "verifier_metadata"},
    )
    domain: str = Field(
        description="GDP.pdf professional domain used for domain Mean Pass metrics.",
        json_schema_extra={"consumed_by": ["metrics", "provenance"], "legacy_location": "verifier_metadata"},
    )
    rubric_criteria: list[RubricCriterion] = Field(
        min_length=1,
        description="Hidden independently judged rubric criteria.",
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    document_manifest: Optional[str] = Field(
        default=None,
        description="Path relative to the agent's documents_base_dir containing page text and image paths.",
        json_schema_extra={"consumed_by": ["prompt"], "legacy_location": "verifier_metadata"},
    )
    source_pdf: Optional[str] = Field(
        default=None,
        description="Pinned source PDF path inside the public dataset snapshot.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
    source_revision: Optional[str] = Field(
        default=None,
        description="Pinned Hugging Face dataset revision.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
