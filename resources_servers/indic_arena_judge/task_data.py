# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light task schema for Indic Arena Hard v2."""

from pydantic import Field

from resources_servers.arena_judge.task_data import TaskData as ArenaTaskData


class TaskData(ArenaTaskData):
    baseline_answer: str | None = Field(
        default=None,
        description="Matching-language reference, required for pairwise mode and unused for pointwise scoring.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    language: str | None = Field(
        default=None,
        description="Language code used for pointwise per-language aggregation.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
