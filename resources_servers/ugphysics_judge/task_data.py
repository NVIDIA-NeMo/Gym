# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the ugphysics_judge server.

Heir of math_with_judge: ``UGPhysicsJudgeRunRequest`` (app.py:138, extra="allow") extends the
parent's required question + expected_answer with an optional reference ``solution`` (fills the
judge prompt's {solution} placeholder) and an optional ``subject`` (echoed through the verify
response and used by compute_metrics as the per-subject pass@k subset key). ``subject`` is one of
13 UGPhysics subjects, e.g. 'ClassicalElectromagnetism'.
"""

from typing import Optional

from pydantic import Field

from resources_servers.math_with_judge.task_data import TaskData as MathWithJudgeTaskData


class TaskData(MathWithJudgeTaskData):
    solution: str = Field(
        default="",
        description="Reference solution walkthrough; fills the judge prompt's {solution} placeholder.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    subject: Optional[str] = Field(
        default=None,
        description="UGPhysics subject (e.g. 'ClassicalMechanics'); subset key for per-subject metrics.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
