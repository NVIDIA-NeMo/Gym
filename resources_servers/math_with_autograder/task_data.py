# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the math_with_autograder server.

Heir of math_with_judge: the server subclasses ``LibraryJudgeMathResourcesServer`` and reuses its
request models verbatim (question + expected_answer, both required), only swapping the judge for a
Skills-style unidirectional autograder. Committed rows additionally carry IMO-bench provenance
columns (problem_id/category/subcategory/source) that today's wire model silently drops via
Pydantic's default extra="ignore"; they are declared here as optional provenance so they survive
validation and the row-format migration.
"""

from typing import Optional

from pydantic import Field

from resources_servers.math_with_judge.task_data import TaskData as MathWithJudgeTaskData


class TaskData(MathWithJudgeTaskData):
    problem_id: Optional[str] = Field(
        default=None,
        description="Source benchmark problem identifier, e.g. 'imo-bench-algebra-001'.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    category: Optional[str] = Field(
        default=None,
        description="Problem category, e.g. 'Algebra', 'Geometry'.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    subcategory: Optional[str] = Field(
        default=None,
        description="Problem subcategory, e.g. 'Operation', '3d_geometry'.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source: Optional[str] = Field(
        default=None,
        description="Original competition source, e.g. 'IMO Shortlist 2021'; null in some rows.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
