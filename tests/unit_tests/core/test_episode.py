# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of episode identity in `nemo_gym.core.episode_types`."""

import pytest
from pydantic import ValidationError

from nemo_gym.core.episode_types import BaseEpisodeRequest, EpisodeId, MaterializedTask, RolloutCoordinates, TaskId


def test_capture_key_qualifies_retries() -> None:
    assert EpisodeId(rollout_id="r").capture_key == "r"
    assert EpisodeId(rollout_id="r", attempt=2).capture_key == "r-a2"
    with pytest.raises(ValidationError, match="reserved attempt suffix"):
        EpisodeId(rollout_id="r-a2")


def test_task_id_contains_only_logical_identity() -> None:
    task_id = TaskId(taskset="swebench_pro:test", task_id="instance-1")
    assert task_id.model_dump() == {
        "taskset": "swebench_pro:test",
        "task_id": "instance-1",
    }
    with pytest.raises(ValidationError, match="revision"):
        TaskId(taskset="swebench_pro:test", task_id="instance-1", revision="v1")


def test_rollout_coordinates_read_the_rows_collector_and_group_keys() -> None:
    row = {
        "_ng_task_index": 4,
        "_ng_rollout_index": 0,
        "_ng_group_id": "cohort",
        "_ng_group_attempt": 2,
        "_ng_attempt_index": 1,  # A rollout's own retry counter is the episode attempt, not a coordinate.
        "instance_id": "task",
    }

    assert RolloutCoordinates.from_row(row) == RolloutCoordinates(
        task_index=4, rollout_index=0, group_id="cohort", group_attempt=2
    )
    # A row carries only the keys its caller set; the rest stay None.
    assert RolloutCoordinates.from_row({"_ng_task_index": 4, "_ng_rollout_index": 1}) == RolloutCoordinates(
        task_index=4, rollout_index=1
    )
    assert RolloutCoordinates.from_row({"instance_id": "task", "_ng_group_id": None}) is None


@pytest.mark.parametrize(
    "row",
    [
        {"_ng_rollout_index": True},
        {"_ng_rollout_index": "1"},
        {"_ng_task_index": -1},
        {"_ng_group_attempt": -1},
        {"_ng_group_id": ""},
        {"_ng_group_id": 7},
    ],
)
def test_rollout_coordinates_refuse_values_that_are_not_indices(row: dict) -> None:
    with pytest.raises(ValidationError):
        RolloutCoordinates.from_row(row)


def test_an_episode_request_without_coordinates_dumps_as_before() -> None:
    request = BaseEpisodeRequest[TaskId](
        episode_id=EpisodeId(rollout_id="0-1"),
        task=MaterializedTask[TaskId](
            task_id=TaskId(taskset="t", task_id="0"), task_input=TaskId(taskset="t", task_id="0")
        ),
    )
    assert "rollout_coordinates" not in request.model_dump(mode="json")

    with_coordinates = request.model_copy(
        update={"rollout_coordinates": RolloutCoordinates(task_index=0, rollout_index=1)}
    )
    dumped = with_coordinates.model_dump(mode="json")
    assert dumped["rollout_coordinates"] == {
        "task_index": 0,
        "rollout_index": 1,
        "group_id": None,
        "group_attempt": None,
    }
    assert BaseEpisodeRequest[TaskId].model_validate(dumped) == with_coordinates
