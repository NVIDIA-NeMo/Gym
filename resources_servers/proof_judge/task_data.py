# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the proof_judge server (judge-scored proof generation).

Reward is judge-driven (verifier + optional meta-verifier), so the only task datum is
``problem``. Mirrors ``ProofWithJudgeVerifyRequest`` (app.py), where ``problem`` is required and
nonempty: ``/verify`` answers 422 without it. When ``config.zero_reward_incorrect_groups`` is
enabled, ``problem`` doubles as the rollout-group identity key.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    problem: str = Field(
        min_length=1,
        description=(
            "Problem statement; fills the verifier/meta-verifier judge prompts and serves as the grouping key "
            "in zero_reward_incorrect_groups mode. Required and nonempty."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
