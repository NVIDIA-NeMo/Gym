# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Question sets: cohorts of video multiple-choice questions that come from no benchmark, such as a training set."""

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from resources_servers.video_mcqa.dataset import SOURCES
from resources_servers.video_mcqa.scoring import AnswerPolicy
from resources_servers.video_mcqa.task import Setting


class QuestionSetCohort(BaseModel):
    """Freeze a question set: its identity and task IDs, with no benchmark categories.

    ``question_set`` names the metrics. A question may have from two to 26 options, and no VideoMME-v2 level or
    group. Task IDs never use a benchmark's ``<benchmark>:`` scope, so benchmark and question-set rows cannot pass
    for one another.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    question_set: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    dataset_source: str = Field(min_length=1, pattern=r"\S")
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    setting: Setting
    answer_policy: AnswerPolicy = "boxed_only_v1"
    task_ids: tuple[str, ...] = Field(min_length=1)
    num_repeats: int = Field(default=1, ge=1, strict=True)

    @model_validator(mode="after")
    def valid_cohort(self) -> Self:
        if len(set(self.task_ids)) != len(self.task_ids) or any(not identity.strip() for identity in self.task_ids):
            raise ValueError("Expected unique, nonempty task IDs")
        if any(identity.split(":", 1)[0] in SOURCES for identity in self.task_ids):
            raise ValueError("Question-set task IDs must not use a benchmark's scope")
        return self


def load_question_set(path: Path) -> QuestionSetCohort | None:
    """The question set in the cohort manifest at PATH, or None when the manifest names no ``question_set`` (a
    benchmark cohort)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "question_set" not in payload:
        return None
    return QuestionSetCohort.model_validate(payload)
