# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the math_with_judge server.

Family parent for the library-judge math servers: math_with_autograder, physics_judge, polymath,
ugphysics_judge, and finance_sec_search all inherit or mirror this two-field core on the wire
(``LibraryJudgeMathRunRequest`` at app.py:54: ``question`` + ``expected_answer``, both required).
verify() runs math-verify symbolically against ``expected_answer`` first, then falls back to an
Arena-Hard-style bidirectional judge whose prompt is filled from ``question`` and
``expected_answer``. Rows carry both fields at the top level; no ``verifier_metadata`` bucket.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str = Field(
        description="The math problem statement; fills the judge prompt on a symbolic-check miss.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_answer: str = Field(
        description=(
            "Ground-truth answer, often LaTeX wrapped in \\(...\\) or $...$ (delimiters are stripped "
            "before math-verify); also fills the judge prompt."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
