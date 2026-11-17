# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schemas of open-ended question rows graded against a weighted-points rubric.

Mirrors the wire: ``VideoQARubricVerifyRequest`` reads ``task_id`` and ``setting`` top-level and ``question`` and
``rubric`` from ``verifier_metadata``. A rubric is checked against ``Rubric`` but kept as the row has it, because the
judge sees it unchanged; criterion guidance, evidence and an ``answer_policy`` are free-form.
"""

import math
from typing import Annotated, Any, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


_NESTED = {"legacy_location": "verifier_metadata"}

Points = Annotated[float, Field(strict=True, ge=0, le=100, allow_inf_nan=False)]
Text = Annotated[str, Field(min_length=1, pattern=r"\S")]


class Criterion(BaseModel):
    """One requirement of a correct answer, worth up to ``weight_points``."""

    model_config = ConfigDict(extra="allow")
    id: Text
    requirement: Text
    weight_points: Annotated[Points, Field(gt=0)]


class Contradiction(BaseModel):
    """A statement that contradicts the correct answer; without ``penalty_points`` it subtracts nothing, and without
    ``score_cap_points`` it caps nothing."""

    model_config = ConfigDict(extra="allow")
    id: Text
    condition: Text
    penalty_points: Points = 0
    score_cap_points: Points | None = None


class Rubric(BaseModel):
    """A question's rubric: criteria whose weights add up to 100, and optional contradictions."""

    model_config = ConfigDict(extra="allow")
    criteria: list[Criterion] = Field(min_length=1)
    contradictions: list[Contradiction] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids_and_full_weight(self) -> Self:
        for items in (self.criteria, self.contradictions):
            if len({item.id for item in items}) != len(items):
                raise ValueError("criterion and contradiction IDs must be unique")
        if not math.isclose(sum(criterion.weight_points for criterion in self.criteria), 100, abs_tol=1e-6):
            raise ValueError("the weight_points of the criteria must add up to 100")
        return self


def validate_rubric(rubric: dict[str, Any]) -> dict[str, Any]:
    """Check RUBRIC against ``Rubric`` and return it unchanged."""
    Rubric.model_validate(rubric)
    return rubric


RubricDict = Annotated[dict[str, Any], AfterValidator(validate_rubric)]


class TaskData(BaseModel):
    """One open-ended question; extra keys are kept, as the verify request keeps them, and reported."""

    model_config = ConfigDict(extra="allow")

    task_id: Text = Field(json_schema_extra={"consumed_by": ["provenance"]})
    setting: Literal["single_turn", "agentic"] = Field(json_schema_extra={"consumed_by": ["verify"]})
    question: Text = Field(json_schema_extra={**_NESTED, "consumed_by": ["verify"]})
    rubric: RubricDict = Field(json_schema_extra={**_NESTED, "consumed_by": ["verify"]})
