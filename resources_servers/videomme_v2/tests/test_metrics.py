# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic truth tables, group identity, missing attempts and repeat isolation."""

from dataclasses import replace
from itertools import product

import pytest
from pydantic import ValidationError

from resources_servers.videomme_v2.metrics import (
    MetricSlice,
    QuestionGroup,
    aggregate_benchmark_scores,
    score_group,
    validate_metric_cohort,
)
from resources_servers.videomme_v2.scoring import grade_answer
from resources_servers.videomme_v2.tests.test_scoring import metadata


REVISION = "a" * 40
PATTERNS = list(product((False, True), repeat=4))
TABLES = [
    ("relevance", None, [0, 6.25, 6.25, 25, 6.25, 25, 25, 56.25, 6.25, 25, 25, 56.25, 25, 56.25, 56.25, 100]),
    ("logic", "[1,2,3,4]", [0] * 8 + [6.25] * 4 + [25, 25, 56.25, 100]),
    ("logic", "[1,[2,3],4]", [0] * 8 + [100 / 12] * 2 + [100 / 3] * 4 + [700 / 12, 100]),
    ("logic", "[[1,2],3,4]", [0] * 4 + [10] * 8 + [20, 20, 50, 100]),
]


@pytest.mark.parametrize(
    ("group_type", "structure", "pattern", "expected"),
    [
        (kind, structure, pattern, expected)
        for kind, structure, values in TABLES
        for pattern, expected in zip(PATTERNS, values)
    ],
)
def test_official_group_truth_tables(group_type, structure, pattern, expected) -> None:
    assert score_group(pattern, group_type=group_type, group_structure=structure) == pytest.approx(expected)


def group(name: str = "one", level: int = 1, **updates: object) -> QuestionGroup:
    return QuestionGroup.model_validate(
        {
            "group_id": name,
            "task_ids": [f"synthetic/{level}/{name}/{index}" for index in range(4)],
            "level": level,
            "group_type": "relevance",
            **updates,
        }
    )


def attempts(value: QuestionGroup, pattern=PATTERNS[-1], *, setting="single_turn"):
    return [
        grade_answer(r"\boxed{B}" if correct else r"\boxed{A}", metadata(task_id), setting=setting)
        for task_id, correct in zip(value.task_ids, pattern)
    ]


def summarize(rows, groups, *, setting="single_turn", **updates):
    return aggregate_benchmark_scores(
        rows,
        expected_task_ids=[task_id for value in groups for task_id in value.task_ids],
        dataset_revision=REVISION,
        setting=setting,
        groups=groups,
        **updates,
    )


@pytest.mark.parametrize("setting", ["single_turn", "agentic"])
def test_raw_and_nonlinear_overall_and_levels(setting) -> None:
    groups = [
        group(),
        group("two", 2, group_type="logic", group_structure="[1,2,3,4]"),
        group("three", 3, group_type="logic", group_structure="[1,2,3,4]"),
    ]
    patterns = [(True, True, False, False), (False, True, True, True), (True, True, False, False)]
    rows = [row for value, pattern in zip(groups, patterns) for row in attempts(value, pattern, setting=setting)]
    result = summarize(reversed(rows), groups, setting=setting)
    assert result.accuracy.avg_accuracy == pytest.approx(7 / 12)
    assert result.overall.metrics()["raw_acc"] == pytest.approx(100 * 7 / 12)
    assert result.overall.metrics()["nonlinear_score"] == pytest.approx(50 / 3)
    assert [result.levels[level].metrics()["raw_acc"] for level in (1, 2, 3)] == [50, 75, 50]
    assert [result.levels[level].metrics()["nonlinear_score"] for level in (1, 2, 3)] == [25, 0, 25]
    assert result.overall.metrics()["group_coverage"] == 1
    assert result.overall.metrics()["nonlinear_complete"] is True


def test_missing_and_inconclusive_answers_keep_group_denominator() -> None:
    value = group()
    rows = attempts(value)[:2]
    rows[1] = grade_answer(None, metadata(value.task_ids[1]), setting="single_turn", failure="infrastructure_failure")
    result = summarize(rows, [value])
    assert result.overall.metrics()["raw_acc"] == 25
    assert result.overall.metrics()["nonlinear_score"] == 6.25
    assert result.overall.metrics()["group_coverage"] == 0
    assert result.overall.metrics()["nonlinear_complete"] is False
    assert result.accuracy.coverage == 0.25 and not result.accuracy.is_complete


def test_completed_failures_are_wrong_but_measured() -> None:
    value = group()
    rows = attempts(value)
    rows[0] = grade_answer(None, metadata(value.task_ids[0]), setting="single_turn", failure="execution_failure")
    rows[1] = grade_answer(None, metadata(value.task_ids[1]), setting="single_turn")
    result = summarize(rows, [value])
    assert result.overall.metrics()["raw_acc"] == 50
    assert result.overall.metrics()["nonlinear_score"] == 25
    assert result.overall.metrics()["group_coverage"] == 1
    assert result.accuracy.is_complete


def test_group_order_not_response_arrival_order() -> None:
    value = group(group_type="logic", group_structure="[1,2,3,4]")
    rows = attempts(value, (True, False, True, True))
    assert summarize(rows, [value]) == summarize(reversed(rows), [value])
    assert summarize(rows, [value]).overall.metrics()["nonlinear_score"] == 6.25


def test_repeats_are_not_combined_into_a_fictitious_all_correct_group() -> None:
    value = group()
    first = summarize(attempts(value, (True, True, False, False)), [value])
    second = summarize(attempts(value, (False, False, True, True)), [value])
    combined = MetricSlice.combine([first.overall, second.overall]).metrics()
    assert (combined["expected"], combined["expected_groups"]) == (8, 2)
    assert (combined["raw_acc"], combined["nonlinear_score"]) == (50, 25)


def test_no_group_metadata_does_not_report_a_fake_nonlinear_score() -> None:
    rows = [grade_answer(r"\boxed{B}", metadata("a"), setting="single_turn")]
    result = aggregate_benchmark_scores(
        rows,
        expected_task_ids=["a", "b", "c"],
        dataset_revision=REVISION,
        setting="single_turn",
        task_levels={"a": 1, "b": 1, "c": 2},
    )
    assert result.overall.metrics()["raw_acc"] == pytest.approx(100 / 3)
    assert result.levels[1].metrics()["raw_acc"] == 50
    assert result.levels[2].metrics()["raw_acc"] == 0
    assert result.levels[3].metrics()["raw_acc_available"] is False
    assert "raw_acc" not in result.levels[3].metrics()
    assert result.overall.metrics()["nonlinear_available"] is False
    assert "nonlinear_score" not in result.overall.metrics()


def test_legacy_cohort_without_levels_only_reports_overall() -> None:
    result = aggregate_benchmark_scores([], expected_task_ids=["a"], dataset_revision=REVISION, setting="single_turn")
    assert result.overall.metrics()["raw_acc"] == 0
    assert all(not value.metrics()["raw_acc_available"] for value in result.levels.values())
    assert MetricSlice.combine([]).metrics()["nonlinear_available"] is False


@pytest.mark.parametrize(
    "updates",
    [
        {"group_id": " "},
        {"group_id": "invalid\x00"},
        {"task_ids": ["a", "a", "b", "c"]},
        {"task_ids": ["a", "b", "c"]},
        {"task_ids": ["a", "b", "c", " "]},
        {"level": True},
        {"level": 1.0},
        {"level": 4},
        {"group_type": "unknown"},
        {"group_type": "logic"},
        {"group_type": "logic", "group_structure": "bad syntax"},
        {"group_type": "logic", "group_structure": "[1,2,4,3]"},
        {"group_type": "logic", "group_structure": "[True,2,3,4]"},
        {"group_type": "logic", "group_structure": "x" * 129},
        {"unknown_field": 1},
    ],
)
def test_invalid_group_metadata(updates) -> None:
    with pytest.raises(ValidationError):
        group(**updates)


@pytest.mark.parametrize("pattern", [[True] * 3, [True] * 5, [1, 0, 1, 0]])
def test_invalid_correctness_vector(pattern) -> None:
    with pytest.raises(ValueError):
        score_group(pattern, group_type="relevance")


def test_invalid_direct_group_scorer_inputs() -> None:
    with pytest.raises(ValueError, match="group_type"):
        score_group([True] * 4, group_type="unknown")
    for structure in (None, "x" * 129, "{", "[1,2,4,3]"):
        with pytest.raises(ValueError):
            score_group([True] * 4, group_type="logic", group_structure=structure)


def test_invalid_group_partitions_and_level_maps() -> None:
    first, second = group(), group("two", 2)
    identities = [*first.task_ids, *second.task_ids]
    invalid = [
        ([first], {}),
        ([first, first], {}),
        ([first, group("other", 2, task_ids=second.task_ids[:3] + first.task_ids[:1])], {}),
        ([first, group("other", 2, task_ids=second.task_ids[:3] + ("unexpected",))], {}),
        ([], {identities[0]: 1}),
        ([], {task_id: True for task_id in identities}),
        ([first, second], {task_id: 3 for task_id in identities}),
    ]
    for groups, levels in invalid:
        with pytest.raises(ValueError):
            validate_metric_cohort(identities, groups=groups, task_levels=levels)


def test_group_ids_can_be_scoped_by_level() -> None:
    groups = [group("same", 1), group("same", 2)]
    result = summarize([row for value in groups for row in attempts(value)], groups)
    assert result.overall.metrics()["nonlinear_score"] == 100


def test_known_levels_must_match_group_levels() -> None:
    value = group()
    levels = {task_id: 1 for task_id in value.task_ids}
    assert validate_metric_cohort(value.task_ids, groups=[value], task_levels=levels) == levels


def test_underlying_protocol_validation_is_preserved() -> None:
    value = group()
    row = attempts(value)[0]
    for rows in ([row, row], [replace(row, dataset_revision="c" * 40)], [replace(row, setting="agentic")]):
        with pytest.raises(ValueError):
            summarize(rows, [value])
