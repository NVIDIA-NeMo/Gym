# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the polymath server.

Heir of math_with_judge: ``PolyMathVerifyRequest`` (app.py:59) extends the parent's required
question + expected_answer with two optional metrics fields. verify() forwards ``weight`` and
``language`` onto the verify response; aggregation reads ``weight`` for difficulty-weighted
pass/majority metrics (default 1.0) and uses ``language`` as a per-language subset key. Rows also
carry a ``difficulty`` label that server code never reads — ``weight`` already encodes it
(low=1, medium=2, high=4, top=8).
"""

from typing import Optional

from pydantic import Field

from resources_servers.math_with_judge.task_data import TaskData as MathWithJudgeTaskData


class TaskData(MathWithJudgeTaskData):
    weight: Optional[float] = Field(
        default=None,
        description="Difficulty weight for weighted pass/majority aggregation; treated as 1.0 when absent.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    language: Optional[str] = Field(
        default=None,
        description="Problem language code (e.g. 'en'); subset key for per-language metrics.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    difficulty: Optional[str] = Field(
        default=None,
        description="Difficulty label ('low'/'medium'/'high'/'top'); unread — weight encodes it.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
