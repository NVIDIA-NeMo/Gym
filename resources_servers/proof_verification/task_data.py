# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the proof_verification server (grade a proof-verdict against ground truth).

Mirrors ``ProofVerificationVerifyRequest`` (app.py): ``problem`` and ``proof`` are optional with
empty-string defaults, while ``ground_truth_judgement`` and ``ground_truth_verify_score`` are
wire-required. All four feed the meta-verifier judge prompt; the reward is
``1 - |predicted - ground_truth_verify_score|``, and a ground-truth score outside {0, 0.5, 1}
yields reward 0 with reason ``invalid_ground_truth_score`` (enforced at runtime, not here).
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    problem: str = Field(
        default="",
        description="Problem statement; fills the meta-verifier judge prompt.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    proof: str = Field(
        default="",
        description="Candidate proof under judgement; fills the judge prompt (length-capped at 40000 chars).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    ground_truth_judgement: str = Field(
        description="Reference judgement text; fills the meta-verifier prompt.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    ground_truth_verify_score: float = Field(
        description=(
            "Ground-truth verification score, semantically an enum {0, 0.5, 1}; reward is "
            "1 - |predicted - ground_truth|. Values outside the enum zero the reward at runtime."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
