# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the competitive_coding_challenges server.

Rows carry only identifiers at the top level (no verifier_metadata): the heavy test data (~27 GB)
is loaded server-side by CCCEvaluator, keyed on problem_id/competition_id. Mirrors
``CompetitiveCodingChallengesVerifyRequest`` (app.py:39, extra='allow'): ``problem_id`` is the
only wire-required field; everything else is Optional. ``subtask_score`` is int in committed data
and Optional[float] on the wire (pydantic coerces), so float here too.

``total_time`` is deliberately NOT declared: verify() reads it via ``payload.get('total_time')``
through the wire's extra='allow', but it appears in no committed row and is a rollout-time
passthrough, not a dataset column — it stays documented on the wire model in app.py only.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    problem_id: str = Field(
        description=(
            "Problem key into the server-side test metadata (e.g. 'makingmexes'); the only "
            "wire-required field. Also used as the display name fallback."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    competition_id: Optional[str] = Field(
        default=None,
        description="Competition the problem belongs to (e.g. 'usaco25'); scopes the evaluator lookup.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    subtask: Optional[str] = Field(
        default=None,
        description="Subtask selector for grading (e.g. 'all_tests').",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    name: Optional[str] = Field(
        default=None,
        description="Human-readable problem name (e.g. 'A-makingmexes'); falls back to problem_id when absent.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    subtask_score: Optional[float] = Field(
        default=None,
        description=(
            "Max score of the subtask, used to normalize the reward; int in committed data "
            "(coerced to float, matching the wire). When None, verify() falls back to the "
            "evaluator's metadata lookup."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    id: Optional[int] = Field(
        default=None,
        description=(
            "Row ordinal in the source export. Never read by server code; survives transit only "
            "via the wire model's extra='allow'."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
