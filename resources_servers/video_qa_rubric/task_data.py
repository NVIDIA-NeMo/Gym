# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schemas of open-ended question rows graded by a judge model.

Mirrors the wire: ``VideoQARubricVerifyRequest`` reads ``task_id`` and ``setting`` top-level and the grading data from
``verifier_metadata``. ``grading`` names the protocol: ``weighted_points`` (the default, when a row has no
``grading``) checks the ``rubric`` against ``Rubric``; ``reference_levels`` (reference_levels.py) also needs a
``reference_answer`` and takes any JSON object as its rubric. Either rubric is kept as the row has it, because the
judge sees it unchanged; criterion guidance, evidence and an ``answer_policy`` are free-form. ``subset`` names the
part of a question set a row belongs to, for per-subset metrics. An agentic row may carry a ``video_binding`` (the
video a lending resources server puts into the episode's sandbox) or a ``preloaded_media`` (media its sandbox image
already holds): video_mcqa's shapes.
"""

import math
from typing import Annotated, Any, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from resources_servers.video_mcqa.task_data import PreloadedMedia, VideoBinding


_NESTED = {"legacy_location": "verifier_metadata"}

Points = Annotated[float, Field(strict=True, ge=0, le=100, allow_inf_nan=False)]
Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
Grading = Literal["weighted_points", "reference_levels"]


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
    grading: Grading = Field(default="weighted_points", json_schema_extra={**_NESTED, "consumed_by": ["verify"]})
    question: Text = Field(json_schema_extra={**_NESTED, "consumed_by": ["verify"]})
    # The weighted-points format (Rubric), or any JSON object for reference_levels.
    rubric: dict[str, Any] = Field(json_schema_extra={**_NESTED, "consumed_by": ["verify"]})
    # Required by reference_levels, absent otherwise.
    reference_answer: Text | None = Field(default=None, json_schema_extra={**_NESTED, "consumed_by": ["verify"]})
    subset: Text | None = Field(default=None, json_schema_extra={**_NESTED, "consumed_by": ["metrics"]})
    video_binding: VideoBinding | None = None
    preloaded_media: PreloadedMedia | None = None

    @model_validator(mode="after")
    def grading_data(self) -> Self:
        if self.grading == "weighted_points":
            validate_rubric(self.rubric)
            if self.reference_answer is not None:
                raise ValueError("A weighted_points row has no reference_answer")
        elif self.reference_answer is None:
            raise ValueError("A reference_levels row needs its reference_answer")
        if self.video_binding is not None and self.preloaded_media is not None:
            raise ValueError("A row takes a video_binding or preloaded_media, not both")
        return self
