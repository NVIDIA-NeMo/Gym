# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw accuracy and official four-question nonlinear metrics, on a 0–100 scale."""

import ast
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from resources_servers.videomme_v2.scoring import AccuracySummary, AnswerPolicy, QuestionScore, aggregate_scores
from resources_servers.videomme_v2.task import Setting


GroupType = Literal["relevance", "logic"]
_LOGIC_WEIGHTS = {
    "[1, 2, 3, 4]": (0.0, 6.25, 25.0, 56.25, 100.0),
    "[1, [2, 3], 4]": (0.0, 100.0 / 12, 100.0 * 4 / 12, 100.0 * 7 / 12, 100.0),
    "[[1, 2], 3, 4]": (0.0, 10.0, 20.0, 50.0, 100.0),
}


def _logic_structure(value: str | None) -> str:
    if not isinstance(value, str) or len(value) > 128:
        raise ValueError("Logic groups require a supported group_structure string")
    try:
        structure = repr(ast.literal_eval(value))
    except (SyntaxError, ValueError) as error:
        raise ValueError("Invalid logic group_structure") from error
    if structure not in _LOGIC_WEIGHTS:
        raise ValueError("Unsupported logic group_structure")
    return structure


class QuestionGroup(BaseModel):
    """Four ordered source task IDs; grouping information is verifier-only."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    group_id: str
    task_ids: tuple[str, str, str, str]
    group_type: GroupType
    level: int = Field(strict=True, ge=1, le=3)
    group_structure: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def valid_group(self) -> Self:
        if any(not value.strip() or "\x00" in value for value in (self.group_id, *self.task_ids)):
            raise ValueError("Group and task IDs must be nonempty and contain no NUL bytes")
        if len(set(self.task_ids)) != 4:
            raise ValueError("Each group requires four distinct task IDs in source question order")
        if self.group_type == "logic":
            _logic_structure(self.group_structure)
        return self


def score_group(correct: Sequence[bool], *, group_type: GroupType, group_structure: str | None = None) -> float:
    """Apply the official relevance or dependency-aware logic score to four answers."""
    if len(correct) != 4 or any(type(value) is not bool for value in correct):
        raise ValueError("A group score requires exactly four correctness booleans")
    if group_type == "relevance":
        return 100.0 * sum(correct) ** 2 / 16
    if group_type != "logic":
        raise ValueError("Unknown group_type")
    structure = _logic_structure(group_structure)
    prefix = 0
    for value in correct:
        if not value:
            break
        prefix += 1
    if structure == "[1, [2, 3], 4]" and prefix == 1 and correct[2]:
        prefix += 1
    elif structure == "[[1, 2], 3, 4]" and prefix == 0 and correct[1]:
        prefix += 1
    return _LOGIC_WEIGHTS[structure][prefix]


def validate_metric_cohort(
    task_ids: Collection[str], *, groups: Collection[QuestionGroup], task_levels: Mapping[str, int]
) -> dict[str, int]:
    """Validate a complete group partition and optional complete per-task level map."""
    expected = set(task_ids)
    levels = dict(task_levels)
    if levels and set(levels) != expected:
        raise ValueError("task_levels must cover exactly the frozen task IDs")
    if any(type(level) is not int or level not in (1, 2, 3) for level in levels.values()):
        raise ValueError("Task levels must be integers from one through three")
    grouped_levels: dict[str, int] = {}
    identities: set[tuple[int, str]] = set()
    for group in groups:
        identity = (group.level, group.group_id)
        if identity in identities:
            raise ValueError("Duplicate group identity within a level")
        identities.add(identity)
        for task_id in group.task_ids:
            if task_id not in expected or task_id in grouped_levels:
                raise ValueError("Groups contain an unexpected task or overlap")
            grouped_levels[task_id] = group.level
    if groups and set(grouped_levels) != expected:
        raise ValueError("Groups must partition every frozen task into complete four-question groups")
    if levels and grouped_levels and levels != grouped_levels:
        raise ValueError("Task and group levels disagree")
    return grouped_levels or levels


@dataclass(frozen=True, slots=True)
class MetricSlice:
    """Sufficient statistics for one level or the overall cohort, across repeats."""

    expected: int
    correct: int
    nonlinear_total: float = 0.0
    expected_groups: int = 0
    completed_groups: int = 0

    @classmethod
    def combine(cls, slices: Iterable["MetricSlice"]) -> "MetricSlice":
        """Merge counts and group-score totals without mixing question-level answers."""
        values = tuple(slices)
        return cls(
            expected=sum(value.expected for value in values),
            correct=sum(value.correct for value in values),
            nonlinear_total=sum(value.nonlinear_total for value in values),
            expected_groups=sum(value.expected_groups for value in values),
            completed_groups=sum(value.completed_groups for value in values),
        )

    def metrics(self) -> dict[str, float | int | bool]:
        """Do not emit a numeric nonlinear score when group metadata is unavailable."""
        values: dict[str, float | int | bool] = {
            "expected": self.expected,
            "correct": self.correct,
            "raw_acc_available": self.expected > 0,
            "nonlinear_available": self.expected_groups > 0,
            "expected_groups": self.expected_groups,
            "completed_groups": self.completed_groups,
        }
        if self.expected:
            values["raw_acc"] = 100.0 * self.correct / self.expected
        if self.expected_groups:
            values.update(
                nonlinear_score=self.nonlinear_total / self.expected_groups,
                group_coverage=self.completed_groups / self.expected_groups,
                nonlinear_complete=self.completed_groups == self.expected_groups,
            )
        return values


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """A single repeat's fixed-denominator accuracy and grouped score breakdowns."""

    accuracy: AccuracySummary
    overall: MetricSlice
    levels: dict[int, MetricSlice]


def aggregate_benchmark_scores(
    scores: Iterable[QuestionScore],
    *,
    expected_task_ids: Collection[str],
    dataset_revision: str,
    setting: Setting,
    answer_policy: AnswerPolicy = "boxed_only_v1",
    groups: Collection[QuestionGroup] = (),
    task_levels: Mapping[str, int] | None = None,
) -> BenchmarkSummary:
    """Score one repeat by explicit IDs, never by response arrival order.

    Missing and inconclusive attempts remain zero in the frozen denominators;
    accuracy coverage and completed_groups identify provisional measurements.
    """
    attempts = tuple(scores)
    accuracy = aggregate_scores(
        attempts,
        expected_task_ids=expected_task_ids,
        dataset_revision=dataset_revision,
        setting=setting,
        answer_policy=answer_policy,
    )
    levels = validate_metric_cohort(expected_task_ids, groups=groups, task_levels=task_levels or {})
    correct = {score.task_id for score in attempts if score.outcome == "correct"}
    completed = {score.task_id for score in attempts if score.evaluation_completed}
    ratings = {
        (group.level, group.group_id): score_group(
            [task_id in correct for task_id in group.task_ids],
            group_type=group.group_type,
            group_structure=group.group_structure,
        )
        for group in groups
    }

    def make_slice(identities: Collection[str], selected_groups: Collection[QuestionGroup]) -> MetricSlice:
        return MetricSlice(
            expected=len(identities),
            correct=len(correct.intersection(identities)),
            nonlinear_total=sum(ratings[(group.level, group.group_id)] for group in selected_groups),
            expected_groups=len(selected_groups),
            completed_groups=sum(all(task_id in completed for task_id in group.task_ids) for group in selected_groups),
        )

    return BenchmarkSummary(
        accuracy=accuracy,
        overall=make_slice(expected_task_ids, groups),
        levels={
            level: make_slice(
                [task_id for task_id, task_level in levels.items() if task_level == level],
                [group for group in groups if group.level == level],
            )
            for level in (1, 2, 3)
        },
    )
