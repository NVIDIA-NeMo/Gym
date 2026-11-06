# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task materialization and producer evidence must agree on trajectory identity."""

import pytest
from pydantic import ValidationError

from nemo_gym.config.keys import (
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.data.task_materialization import materialize_task
from nemo_gym.rollouts.trajectory_builder import _build_trajectory_record, _trajectory_identity
from nemo_gym.trajectory.trajectory_record import TrajectoryRecord, TrajectoryTurn


def _materialized_row() -> dict:
    return materialize_task({"problem_id": "task-7", "prompt": "synthetic"}, taskset="test:tasks") | {
        TASK_INDEX_KEY_NAME: 7,
        ROLLOUT_INDEX_KEY_NAME: 2,
    }


def test_materialized_task_keeps_matching_producer_and_turn_identity() -> None:
    row = _materialized_row() | {ROLLOUT_ID_KEY_NAME: "capture-physical", ATTEMPT_INDEX_KEY_NAME: 3}
    producer = TrajectoryRecord(
        task_id="task-7",
        rollout_id="capture-physical-a3",
        turns=[
            TrajectoryTurn(
                invocation_id="root",
                task_id="task-7",
                rollout_id="capture-physical-a3",
                turn_no=1,
                timestamp=1.0,
                step_count=1,
                answer="retained answer",
            )
        ],
    )

    trajectory = _build_trajectory_record(row, {"ng_trajectory": producer.model_dump(mode="json")})

    assert (trajectory.task_id, trajectory.rollout_id) == ("task-7", "capture-physical-a3")
    assert trajectory.turns == producer.turns
    assert "producer_trajectory_identity_mismatch" not in {gap.code for gap in trajectory.gaps}


@pytest.mark.parametrize("attempt, expected", [(0, "7-2"), (2, "7-2-a2")])
def test_materialized_identity_retains_index_based_rollout_attempt(attempt: int, expected: str) -> None:
    assert _trajectory_identity(_materialized_row() | {ATTEMPT_INDEX_KEY_NAME: attempt}) == ("task-7", expected)


def test_explicit_semantic_identity_wins_before_materialized_task_validation() -> None:
    row = _materialized_row() | {
        "task_id": {},
        "trajectory_identity": {"task_id": "semantic-task", "rollout_id": "semantic-rollout"},
        ROLLOUT_ID_KEY_NAME: "capture-physical",
        ATTEMPT_INDEX_KEY_NAME: 3,
    }
    assert _trajectory_identity(row) == ("semantic-task", "semantic-rollout")


@pytest.mark.parametrize("identity", [{}, {"task_id": "partial"}, {"task_id": "", "rollout_id": "semantic"}])
def test_incomplete_semantic_identity_falls_back_to_materialized_task(identity: dict) -> None:
    assert _trajectory_identity(_materialized_row() | {"trajectory_identity": identity}) == ("task-7", "7-2")


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({"task_id": 0, "problem_id": "problem", "instance_id": "instance"}, "0"),
        ({"task_id": None, "problem_id": "problem", "instance_id": "instance"}, "problem"),
        ({"problem_id": None, "instance_id": "instance"}, "instance"),
        ({}, "7"),
    ],
)
def test_legacy_identity_keeps_field_precedence_and_index_fallback(fields: dict, expected: str) -> None:
    row = {TASK_INDEX_KEY_NAME: 7, ROLLOUT_INDEX_KEY_NAME: 2, ATTEMPT_INDEX_KEY_NAME: 1, **fields}
    assert _trajectory_identity(row) == (expected, "7-2-a1")


@pytest.mark.parametrize("task_input", [None, [], "malformed"])
def test_materialized_identity_does_not_validate_task_input(task_input: object) -> None:
    assert _trajectory_identity(_materialized_row() | {"task_input": task_input}) == ("task-7", "7-2")


@pytest.mark.parametrize("task_id", [{}, {"taskset": "test:tasks"}, {"taskset": "test:tasks", "task_id": ""}])
def test_invalid_materialized_task_identity_fails_validation(task_id: dict) -> None:
    with pytest.raises(ValidationError):
        _trajectory_identity(_materialized_row() | {"task_id": task_id})


@pytest.mark.parametrize("field", ["task_id", "problem_id", "instance_id"])
def test_legacy_identity_with_capture_id_does_not_require_indices(field: str) -> None:
    row = {field: "task-without-index", ROLLOUT_ID_KEY_NAME: "capture-without-index", ATTEMPT_INDEX_KEY_NAME: 2}
    assert _trajectory_identity(row) == ("task-without-index", "capture-without-index-a2")


def test_materialized_identity_with_capture_id_does_not_require_indices() -> None:
    row = materialize_task({"problem_id": "task-7"}, taskset="test:tasks") | {ROLLOUT_ID_KEY_NAME: "capture-id"}
    assert _trajectory_identity(row) == ("task-7", "capture-id")


def test_explicit_semantic_identity_does_not_require_indices() -> None:
    row = {"trajectory_identity": {"task_id": "semantic-task", "rollout_id": "semantic-rollout"}}
    assert _trajectory_identity(row) == ("semantic-task", "semantic-rollout")


def test_capture_id_without_task_identity_still_requires_task_index() -> None:
    with pytest.raises(KeyError, match=TASK_INDEX_KEY_NAME):
        _trajectory_identity({ROLLOUT_ID_KEY_NAME: "capture-id"})
