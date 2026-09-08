# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the critpt server.

Verification is remote: verify() buffers submissions until a full batch of 70 unique problem_ids
(Challenge_1_main..Challenge_70_main), ships them to the ArtificialAnalysis API, and every
rollout in the batch receives the AA aggregate accuracy as its reward. Mirrors
``CritPtRunRequest``/``CritPtVerifyRequest`` (app.py:191): ``problem_id`` is the only wire field
and is required. ``problem``/``code_template``/``uuid`` are prompt-construction provenance baked
into responses_create_params at data-prep time; undeclared on the wire (dropped in transit by
pydantic's default extra='ignore') and never read by the server.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    problem_id: str = Field(
        description=(
            "Canonical CritPt challenge id (e.g. 'Challenge_1_main'); slots the submission into "
            "the 70-problem ArtificialAnalysis batch. Values must span the canonical "
            "Challenge_1_main..Challenge_70_main set for a batch to complete."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    problem: Optional[str] = Field(
        default=None,
        description=(
            "Full problem statement; already baked into responses_create_params at data-prep "
            "time, never read by the server (dropped in transit)."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    code_template: Optional[str] = Field(
        default=None,
        description=(
            "Python answer-function skeleton the model must fill in; prompt-side provenance, "
            "never read by the server (dropped in transit)."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    uuid: Optional[str] = Field(
        default=None,
        description="Row identifier (equals problem_id in committed rows); never read (dropped in transit).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
