# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the imo_proofbench_judge server.

Olympiad proof problems graded by an LLM judge. ``problem``/``reference_solution``/``rubric`` fill
the judge prompt template placeholders; ``expected_answer`` feeds a sympy ``math_equal`` symbolic
short-circuit before the judge and may legitimately be empty/whitespace. All fields are Optional
mirroring ``ImoProofBenchRunRequest`` (app.py), which types every task field ``Optional[str]``
with ``extra="allow"``.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    problem_id: Optional[str] = Field(
        default=None,
        description="Stable problem identifier (e.g. 'PB-Basic-001'); pass-through only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    problem: Optional[str] = Field(
        default=None,
        description="Problem statement (LaTeX/markdown); fills the judge prompt's {problem} placeholder.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    reference_solution: Optional[str] = Field(
        default=None,
        description="Reference solution prose; fills the judge prompt's {reference_solution} placeholder.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    rubric: Optional[str] = Field(
        default=None,
        description=(
            "Grading rubric as plain markdown/text prose (NOT structured); fills the judge prompt's "
            "{rubric} placeholder."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_answer: Optional[str] = Field(
        default=None,
        description=(
            "Final answer for the sympy math_equal symbolic short-circuit before the LLM judge. "
            "May be empty or whitespace-only for proof-only problems."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    category: Optional[str] = Field(
        default=None,
        description="Problem category (e.g. 'Algebra'); used by compute_metrics for subset breakdowns.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    level: Optional[str] = Field(
        default=None,
        description="Difficulty level (e.g. 'IMO-easy', 'pre-IMO'); used by compute_metrics for subset breakdowns.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    source: Optional[str] = Field(
        default=None,
        description="Original competition source attribution (e.g. '(Modified) IMO 2019, P1').",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
