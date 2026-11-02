# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the imo_gradingbench server.

Ground truth rides as top-level row fields; there is no verifier_metadata anywhere. Mirrors
``ImoGradingBenchRunRequest`` (app.py, extra='allow'): all three fields are wire-Optional even
though verify() effectively needs ``expected_answer`` (an absent or unrecognized grade word just
yields reward 0 via normalize_expected_grade, never a 422). Deterministic verifier — regex
last-grade-word extraction, no LLM judge.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_answer: Optional[str] = Field(
        default=None,
        description=(
            "Gold grade word, one of {correct, almost, partial, incorrect} "
            "(normalize_expected_grade rejects anything else -> reward 0). Compared against the "
            "last grade word regex-extracted from the model output; also feeds the high/low "
            "bucket binarized_accuracy metric."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    grading_id: Optional[str] = Field(
        default=None,
        description=(
            "Identifier of the (problem, candidate-solution) grading instance; carried through "
            "the verify response via model_dump, never read."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    problem_id: Optional[str] = Field(
        default=None,
        description="Identifier of the underlying math problem; identity/passthrough only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
