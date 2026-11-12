# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-owned fields shared by the four visual benchmark datasets."""

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MetadataString = Annotated[str, Field(json_schema_extra={"legacy_location": "verifier_metadata"})]


def reference_options(answer: str, options: dict[str, str]) -> str:
    """Resolve an MCQ reference to sorted letters without guessing ambiguous option text."""
    answer = answer.strip()
    if re.fullmatch(r"[A-Ga-g](?:\s*,\s*[A-Ga-g])*", answer):
        letters = [letter.strip().upper() for letter in answer.split(",")]
        if len(set(letters)) != len(letters) or any(letter not in options for letter in letters):
            raise ValueError("VTCBench reference must select distinct available options")
        return "".join(sorted(letters))
    matches = [letter for letter, text in options.items() if text.strip().casefold() == answer.casefold()]
    if len(matches) != 1:
        raise ValueError("VTCBench reference text must match exactly one option")
    return matches[0]


class TaskData(BaseModel):
    """Flat schema; legacy rows carry these fields in verifier_metadata.

    Extra fields are retained for dataset provenance and caller annotations.
    References remain task data, never policy messages.
    """

    model_config = ConfigDict(extra="allow")

    benchmark: Literal["VStarBench", "HRBench8K", "VTCBench", "ZeroBench"] = Field(
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
        if self.benchmark == "VTCBench" and self.options:
            if any(key not in "ABCDEFG" or len(key) != 1 or not value.strip() for key, value in self.options.items()):
                raise ValueError("VTCBench options require letters A-G and nonempty text")
            reference_options(self.answer, self.options)
        return self
