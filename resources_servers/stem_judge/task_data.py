# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Task-data schema for the stem_judge server.

This is the row contract itself: ``StemJudgeRunRequest`` in app.py inherits ``TaskData``, so
dataset tooling and the server validate rows with the same fields and rules. Two row kinds share
one open model:

- Answer rows carry a reference answer, ``expected_answer`` (or a scalar
  ``metadata.expected_answer``), and score 1 or 0 from the judge's verdict.
- Rubric rows carry a ``rubric`` list and score earned over total positive points; the reference
  is optional.

Other row fields (dataset labels, ids) are allowed and carried into the verify response.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RubricCriterion(BaseModel):
    id: str = Field(min_length=1, description="Criterion id the judge marks true or false; unique within the rubric.")
    points: float = Field(
        allow_inf_nan=False,
        description="Points earned when the criterion is met; negative points mark a penalty. Never 0.",
    )
    criterion: str = Field(min_length=1, description="What the judge checks in the student's solution.")

    @model_validator(mode="after")
    def _points_nonzero(self) -> RubricCriterion:
        if self.points == 0:
            raise ValueError(f"rubric criterion {self.id!r} has 0 points")
        return self


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[Union[str, int]] = Field(
        default=None,
        description="Row identifier; carried into the verify response and the generation log.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    uuid: Optional[Union[str, int]] = Field(
        default=None,
        description="Alternative row identifier, used by the generation log when id is absent.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    expected_answer: Optional[str] = Field(
        default=None,
        description="Reference answer shown to the judge. Required on answer rows (or metadata.expected_answer).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Open bucket; a scalar metadata.expected_answer is the reference when expected_answer is absent or null.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    answer_format: Optional[Literal["boxed", "free"]] = Field(
        default=None,
        description="boxed: the final answer needs a non-empty \\boxed{}; free: judged on the whole final answer. Defaults to the server's default_answer_format.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    answer_format_regex: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Full-match final-answer contract with a named 'answer' capture; overrides boxed/free selection.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    rubric: Optional[list[RubricCriterion]] = Field(
        default=None,
        description="Makes this a rubric row: the judge marks every criterion and the server sums the points.",
        json_schema_extra={"consumed_by": ["verify"]},
    )

    @field_validator("answer_format_regex")
    @classmethod
    def validate_answer_format_regex(cls, value: str | None) -> str | None:
        """Reject malformed contracts before a row can be graded."""
        if value is not None:
            try:
                pattern = re.compile(value)
            except re.error as exc:
                raise ValueError("Invalid answer_format_regex") from exc
            if "answer" not in pattern.groupindex:
                raise ValueError("answer_format_regex requires a named 'answer' capture")
        return value

    @model_validator(mode="after")
    def _row_is_gradable(self) -> TaskData:
        if self.rubric is None:
            # Without a reference the judge would grade from its own opinion.
            if not reference_of(self).strip():
                raise ValueError("an answer row needs a non-empty expected_answer")
            return self
        if not self.rubric:
            raise ValueError("rubric is an empty list; leave it out for an answer row")
        ids = [c.id for c in self.rubric]
        if len(set(ids)) != len(ids):
            raise ValueError(f"rubric criterion ids are not unique: {ids}")
        if not any(c.points > 0 for c in self.rubric):
            raise ValueError("rubric has no criterion with positive points")
        return self


def reference_of(row: TaskData) -> str:
    """The row's reference answer: ``expected_answer``, else a scalar ``metadata["expected_answer"]``.

    Only an absent or null ``expected_answer`` falls back to metadata; a list or other container
    there counts as no reference.
    """
    if row.expected_answer is not None:
        return row.expected_answer
    value = (row.metadata or {}).get("expected_answer")
    return str(value) if isinstance(value, (str, int, float)) else ""
