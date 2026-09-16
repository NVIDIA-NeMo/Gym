# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the arena_judge server.

Rows produced by ``benchmarks/arena_hard_v2/prepare.py`` carry the task fields at the top level;
there is no verifier_metadata. Mirrors ``ArenaJudgeRunRequest`` (app.py): every field is
Optional[str] with ``extra="allow"``, and verify() reads all of them defensively
(``body.question or ""``, ``body.category or config.default_category``). ``category`` is a
config-driven enum in practice (the keys of ``judge_prompt_paths``, e.g. ``hard_prompt`` /
``creative_writing``) with a logged fallback to the default for unknown values, so it stays a
plain ``str`` here.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: Optional[str] = Field(
        default=None,
        description="The arena prompt both answers respond to; interpolated into the pairwise judge prompts.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    baseline_answer: Optional[str] = Field(
        default=None,
        description="The frozen baseline model's answer the candidate is judged against (both A/B orderings).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    category: Optional[str] = Field(
        default=None,
        description=(
            "Arena-hard-v2 category selecting the judge prompt (config judge_prompt_paths key, e.g. "
            "'hard_prompt', 'creative_writing'); unknown or missing values fall back to config.default_category. "
            "Also grouped on by compute_metrics for per-category Elo."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    uid: Optional[str] = Field(
        default=None,
        description="Stable task identifier from the source benchmark; never read by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
