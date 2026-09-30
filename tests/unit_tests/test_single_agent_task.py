# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
from pydantic import JsonValue, ValidationError

from nemo_gym.single_agent_task import materialize_single_agent_task


@pytest.mark.parametrize(
    "identity, expected",
    [
        ({"task_id": 0, "problem_id": "problem", "instance_id": "instance"}, "0"),
        ({"task_id": None, "problem_id": "problem", "instance_id": "instance"}, "problem"),
        ({"problem_id": None, "instance_id": "instance"}, "instance"),
        ({"_ng_task_index": 0}, "0"),
        ({}, "5"),
    ],
)
def test_identity_precedence_and_prompt_edit_stability(identity: dict[str, JsonValue], expected: str) -> None:
    row = {**identity, "responses_create_params": {"input": "original prompt"}}
    task = materialize_single_agent_task(row, taskset="test", task_index=5)
    assert task.task_id.task_id == expected

    row["responses_create_params"] = {"input": "edited prompt"}
    assert materialize_single_agent_task(row, taskset="test", task_index=5).task_id == task.task_id


def test_conversion_preserves_model_and_verifier_fields_without_mutating_source() -> None:
    row = {
        "task_id": "task",
        "responses_create_params": {"input": "fix it", "temperature": 0.0, "instructions": None},
        "run_script": "verifier\nscript\n",
        "verifier_metadata": {"answer": None, "enabled": False},
        "agent_ref": {"name": "agent"},
        "skills_ref": "skills",
        "task_source": "source",
        "_ng_task_index": 7,
        "_ng_rollout_index": 2,
        "_ng_attempt_index": 1,
        "_ng_rollout_id": "rollout",
        "_ng_custom_metadata": "collector-only",
    }
    original = deepcopy(row)
    task = materialize_single_agent_task(row, taskset="test").model_dump(mode="json", exclude_unset=True)

    assert task["task_input"] == {
        "responses_create_params": row["responses_create_params"],
        "task_data": {key: row[key] for key in ("task_id", "run_script", "verifier_metadata")},
    }
    assert row == original


@pytest.mark.parametrize("index", [None, -1, True, "5"])
def test_missing_identity_requires_valid_source_index(index: JsonValue) -> None:
    with pytest.raises(ValueError, match="non-negative task_index"):
        materialize_single_agent_task(
            {"_ng_task_index": index, "responses_create_params": {"input": "task"}}, taskset="test"
        )


def test_requires_identity_when_source_index_is_unavailable() -> None:
    with pytest.raises(ValueError, match="task_index"):
        materialize_single_agent_task({"responses_create_params": {"input": "task"}}, taskset="test")


def test_already_materialized_task_is_not_converted_twice() -> None:
    task = materialize_single_agent_task(
        {"task_id": "task", "responses_create_params": {"input": "task"}}, taskset="test"
    )
    with pytest.raises(ValueError, match="already materialized"):
        materialize_single_agent_task(task.model_dump(mode="json"), taskset="test")


def test_validates_response_parameters() -> None:
    with pytest.raises(ValidationError):
        materialize_single_agent_task({"task_id": "task", "responses_create_params": {"input": 123}}, taskset="test")
