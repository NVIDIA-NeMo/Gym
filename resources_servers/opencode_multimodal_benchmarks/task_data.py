# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-owned fields shared by the five visual benchmark datasets."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MetadataString = Annotated[str, Field(json_schema_extra={"legacy_location": "verifier_metadata"})]


class TaskData(BaseModel):
    """Flat schema; legacy rows carry these fields in verifier_metadata.

    Extra fields are retained for dataset provenance and caller annotations.
    References remain task data, never policy messages.
    """

    model_config = ConfigDict(extra="allow")

    benchmark: Literal["VStarBench", "HRBench8K", "VTCBench", "AgentX_GoalAcc", "ZeroBench"] = Field(
        json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    index: MetadataString
    question: MetadataString
    answer: MetadataString
    options: dict[str, str] = Field(default_factory=dict, json_schema_extra={"legacy_location": "verifier_metadata"})
    category: MetadataString = ""
    l2_category: MetadataString = ""
    ans_type: MetadataString = ""
    cycle_category: MetadataString = ""
    split: MetadataString = ""
    expected_samples: int = Field(gt=0, json_schema_extra={"legacy_location": "verifier_metadata"})
    dataset_revision: MetadataString = ""
    source_index: MetadataString = ""
    GT_json: MetadataString = ""
    query_type: MetadataString = ""
    query_type_raw: MetadataString = ""
    modality: MetadataString = ""
    source_media_sha256: dict[str, str] = Field(
        default_factory=dict, json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    video_sampling: dict[str, object] = Field(
        default_factory=dict, json_schema_extra={"legacy_location": "verifier_metadata"}
    )

    @field_validator("index", "question", "answer")
    @classmethod
    def require_nonempty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Task index, question and reference answer must be nonempty")
        return value

    @model_validator(mode="after")
    def require_benchmark_fields(self) -> "TaskData":
        if self.benchmark in {"VStarBench", "HRBench8K"}:
            if not self.options or self.answer.strip().upper() not in self.options:
                raise ValueError("MCQ tasks require options and a reference option letter")
            if any(not value.strip() for value in self.options.values()):
                raise ValueError("MCQ option text must be nonempty")
        if self.benchmark == "HRBench8K" and not self.cycle_category.strip():
            raise ValueError("HRBench8K requires cycle_category")
        if self.benchmark == "AgentX_GoalAcc":
            if not self.GT_json.strip() or not self.query_type_raw.strip():
                raise ValueError("Agent-X requires GT_json and query_type_raw")
            if self.query_type not in {"objective", "subjective"} or self.modality not in {"Image", "Video"}:
                raise ValueError("Agent-X requires query_type and modality")
        return self
