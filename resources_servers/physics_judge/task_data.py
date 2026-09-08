# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the physics_judge server.

Heir of math_with_judge: physics_judge defines no request model of its own — verify() and the
wire contract are the inherited ``LibraryJudgeMathVerifyRequest`` (question + expected_answer,
both required). Committed rows add physics metadata on top: ``domain`` is read defensively at
metrics-aggregation time (compute_metrics, app.py:240-255) for per-domain subsets, while
``difficulty``/``answer_type``/``language``/``solution`` are pass-through provenance written by
benchmarks/physics/prepare.py and never read by server code.
"""

from typing import Optional

from pydantic import Field

from resources_servers.math_with_judge.task_data import TaskData as MathWithJudgeTaskData


class TaskData(MathWithJudgeTaskData):
    domain: Optional[str] = Field(
        default=None,
        description="Physics domain (e.g. 'mechanics', 'thermodynamics'); per-domain subset metrics, skipped if absent.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    difficulty: Optional[str] = Field(
        default=None,
        description="Difficulty label (e.g. 'easy', 'medium'); unread by server code.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    answer_type: Optional[str] = Field(
        default=None,
        description="Expected answer kind (e.g. 'expression'); unread by server code.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    language: Optional[str] = Field(
        default=None,
        description="Problem language code (e.g. 'en'); unread by server code.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    solution: Optional[str] = Field(
        default=None,
        description="Reference solution walkthrough; unread by server code.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
