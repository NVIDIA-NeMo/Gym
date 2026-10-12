# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SciKnowEval source metadata and per-task judge rubrics."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    answer_type: Literal[
        "mcq-4-choices", "mcq-2-choices", "true_or_false", "filling", "relation_extraction", "open-ended-qa"
    ] = Field(json_schema_extra={"legacy_location": "verifier_metadata"})
    expected_answer: str = Field(min_length=1, json_schema_extra={"legacy_location": "verifier_metadata"})
    id: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    domain: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    level: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    task: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    subtask: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    source: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    subset_for_metrics: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    original_instruction: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    letters: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    judge_system: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    judge_prefix: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    judge_suffix: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    judge_scale: Literal["score", "T/F", "MCQ"] | None = Field(
        default=None, json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    judge_rubric: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})

    @model_validator(mode="after")
    def validate_grading_fields(self) -> Self:
        if self.answer_type in ("relation_extraction", "open-ended-qa"):
            if any(
                getattr(self, key) is None for key in ("judge_system", "judge_prefix", "judge_suffix", "judge_scale")
            ):
                raise ValueError("Judged tasks require judge_system, judge_prefix, judge_suffix, and judge_scale")
        if self.answer_type.startswith("mcq-"):
            allowed = "AB" if self.answer_type == "mcq-2-choices" else "ABCD"
            if self.expected_answer not in allowed or len(self.expected_answer) != 1:
                raise ValueError("MCQ expected_answer must be one of the available option letters")
        if self.answer_type == "true_or_false" and self.expected_answer.strip() not in ("Yes", "No"):
            raise ValueError("True/false expected_answer must be Yes or No")
        return self
