# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the math_proof_judgement server (binary proof-correctness judging).

Task fields ride at the row top level; there is no verifier_metadata. Required-ness mirrors
``MathProofJudgementRunRequest`` (app.py:128, ``extra="allow"``): every task field is Optional —
verify() reads only ``expected_judgement`` (regex-parsed 'Judgement: Yes/No'; a None gold yields
reward 0) plus the assistant message text. ``metadata`` is undeclared on today's wire but passes
through via ``extra="allow"`` and is declared here as provenance.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_judgement: Optional[str] = Field(
        default=None,
        description=(
            "Gold verdict string, e.g. 'Judgement: Yes' or 'Judgement: No'; regex-parsed to Yes/No. "
            "None (or unparseable) gold gives reward 0. compute_metrics() also re-reads it "
            "defensively from rollouts[0] (app.py:335) for the tp/fp/fn/tn breakdown."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    problem_id: Optional[str] = Field(
        default=None,
        description="Row identifier; declared on the wire purely for row re-identification, never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    problem: Optional[str] = Field(
        default=None,
        description="The math problem statement whose proof is being judged; never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    proof: Optional[str] = Field(
        default=None,
        description="The candidate proof text under judgement; never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Arbitrary provenance bucket (committed rows carry {expert_rating: int, model_id: str}). "
            "Undeclared on today's wire but passed through to the verify response via extra='allow'."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
