# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests of episode identity in `nemo_gym.core.episode_types`."""

import pytest
from pydantic import ValidationError

from nemo_gym.core.episode_context import current_session_key, rollout_context
from nemo_gym.core.episode_types import EpisodeId, TaskId


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


def test_there_is_no_session_outside_a_rollout() -> None:
    assert current_session_key() is None


@pytest.mark.parametrize("capture_key", ["3-1", "3-1-a2", "swe-task.7-0"])
def test_a_session_is_the_rollout_attempt(capture_key: str) -> None:
    with rollout_context(capture_key):
        assert current_session_key() == capture_key
    assert current_session_key() is None


def test_a_retry_attempt_is_its_own_session() -> None:
    with rollout_context(EpisodeId(rollout_id="3-1").capture_key):
        first = current_session_key()
    with rollout_context(EpisodeId(rollout_id="3-1", attempt=1).capture_key):
        retry = current_session_key()

    assert first != retry
