# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ChemBench grading fields and provenance, carried inside verifier_metadata."""

import math
import re
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    question_type: Literal["mcq", "numeric"] = Field(json_schema_extra={"legacy_location": "verifier_metadata"})
    expected_answer: str = Field(json_schema_extra={"legacy_location": "verifier_metadata"})
    relative_tolerance: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="Upstream numeric absolute error threshold; None defaults to 0.01 * target.",
        json_schema_extra={"legacy_location": "verifier_metadata"},
    )
    uuid: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    name: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    subset_for_metrics: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    subfield: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    keywords: list[str] = Field(default_factory=list, json_schema_extra={"legacy_location": "verifier_metadata"})
    in_human_subset: bool = Field(default=False, json_schema_extra={"legacy_location": "verifier_metadata"})
    options: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})

    @model_validator(mode="after")
    def validate_expected_answer(self) -> Self:
        if self.question_type == "mcq":
            if not re.fullmatch(r"[A-Z](?:(?:\s*,\s*|\s+)[A-Z])*", self.expected_answer.strip()):
                raise ValueError("MCQ expected_answer must contain one or more uppercase option letters")
        elif not math.isfinite(float(self.expected_answer)):
            raise ValueError("Numeric expected_answer must be finite")
        return self
