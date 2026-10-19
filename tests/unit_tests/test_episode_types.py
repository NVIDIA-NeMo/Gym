# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel, ValidationError

from nemo_gym.episode_types import (
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    EpisodeFailure,
    EpisodeId,
    MaterializedTask,
    TaskId,
    is_materialized_task_row,
)


class _TaskInput(BaseModel):
    value: str


class _Request(BaseEpisodeRequest[_TaskInput]):
    pass


class _Response(BaseEpisodeResponse[str]):
    pass


def _request() -> _Request:
    return _Request(
        episode_id=EpisodeId(rollout_id="rollout", attempt=1),
        task=MaterializedTask(
            task_id=TaskId(taskset="test", task_id="task"),
            task_input=_TaskInput(value="result"),
        ),
    )


def test_episode_response_requires_exactly_one_outcome() -> None:
    request = _request()
    with pytest.raises(ValidationError, match="exactly one"):
        _Response(episode_id=request.episode_id, task_id=request.task.task_id)
    with pytest.raises(ValidationError, match="exactly one"):
        _Response(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result="result",
            failure=EpisodeFailure(failure_reason="failure", terminal=True),
        )


def test_native_trajectory_is_typed_envelope_evidence() -> None:
    request = _request()
    response = _Response(
        episode_id=request.episode_id,
        task_id=request.task.task_id,
        result="result",
        agent_trajectory={"task_id": "task", "rollout_id": "rollout-a1"},
    )
    assert response.agent_trajectory["rollout_id"] == "rollout-a1"
    assert _Response.model_validate_json(response.model_dump_json()) == response
    with pytest.raises(ValidationError):
        _Response.model_validate(response.model_dump() | {"agent_trajectory": {"turns": "invalid"}})


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


def test_failure_reason_round_trips_with_optional_metadata() -> None:
    current = EpisodeFailure.model_validate({"failure_reason": "Lost resources session", "terminal": False})
    assert current.failure_reason == "Lost resources session"
    assert current.failure_kind is None and current.stage is None
    expected = {"failure_reason": "Lost resources session", "terminal": False}
    assert current.model_dump() == expected
    assert EpisodeFailure.model_validate_json(current.model_dump_json()) == current


def test_failure_reason_length_limit() -> None:
    with pytest.raises(ValidationError, match="2000"):
        EpisodeFailure(failure_reason="x" * 2001, terminal=True)


@pytest.mark.parametrize("stage", ["admission", "seed", "agent", "verification", "cleanup"])
def test_single_agent_protocol_inherits_shared_failure_metadata(stage: str) -> None:
    from nemo_gym.single_agent_turn_types import SingleAgentTurnFailure

    failure = SingleAgentTurnFailure(
        failure_reason="Participant unavailable",
        terminal=False,
        failure_kind="transport_unreachable",
        stage=stage,
    )
    saved = failure.model_dump(mode="json", exclude={"partial_response", "agent_observations"})
    shared = EpisodeFailure.model_validate(saved)
    assert shared.stage == stage and shared.failure_kind == "transport_unreachable"
    assert shared.terminal is False


def test_shared_failure_kind_accepts_extensions_and_warns_on_legacy_names(caplog) -> None:
    custom = EpisodeFailure(failure_reason="Custom failure", terminal=True, failure_kind="example:tool_crashed")
    assert custom.failure_kind == "example:tool_crashed"
    assert not caplog.records

    legacy = EpisodeFailure(failure_reason="Legacy failure", terminal=False, failure_kind="legacy_contract_test_error")
    assert legacy.failure_kind == "legacy_contract_test_error"
    assert "unregistered failure_kind" in caplog.text


def test_unknown_execution_stage_is_optional_but_stage_aliases_are_rejected() -> None:
    assert EpisodeFailure(failure_reason="Lost reply", terminal=False, failure_kind="transport_timeout").stage is None
    with pytest.raises(ValidationError, match="stage"):
        EpisodeFailure(failure_reason="Failed scoring", terminal=False, stage="verifier")


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"task_id": {"taskset": "test", "task_id": "1"}, "task_input": {}}, True),
        ({"task_id": {"taskset": "test", "task_id": "1"}}, False),
        ({"task_id": {"taskset": "test", "task_id": "1"}, "task_input": None}, True),
        ({"task_id": None, "task_input": {}}, False),
        ({"task_id": "test:1", "task_input": {}}, False),
        ({"task_id": {"taskset": "test", "task_id": "1"}, "task_input": []}, True),
    ],
)
def test_materialized_task_row_identifies_routing_shape(row: dict[str, object], expected: bool) -> None:
    assert is_materialized_task_row(row) is expected


@pytest.mark.parametrize("failed", [False, True])
def test_episode_envelope_projects_native_trajectory_without_reserved_result_keys(failed) -> None:
    from nemo_gym.rollout_collection import _episode_record

    trajectory = {"task_id": "a", "rollout_id": "rollout", "turns": [], "invocations": []}
    outcome = {"failure": {"failure_reason": "interrupted", "terminal": True}} if failed else {"result": {"reward": 1}}
    reply = {"task_id": {"taskset": "test", "task_id": "a"}, **outcome, "agent_trajectory": trajectory}
    record = _episode_record(reply)
    assert record["ng_trajectory"] == trajectory
    assert "agent_trajectory" not in record
    with pytest.raises(ValueError, match="reserved"):
        _episode_record({"task_id": reply["task_id"], "result": {"reward": 1, "ng_trajectory": trajectory}})


def test_materialized_trajectory_keeps_logical_task_identity() -> None:
    from nemo_gym.rollout_collection import _build_trajectory_record

    row = {
        "task_id": {"taskset": "test", "task_id": "task"},
        "task_input": {},
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "_ng_rollout_id": "rollout",
    }
    trajectory = _build_trajectory_record(
        row, {"ng_trajectory": {"task_id": "task", "rollout_id": "rollout", "turns": []}}
    )
    assert trajectory.task_id == "task"
    assert all(gap.code != "producer_trajectory_identity_mismatch" for gap in trajectory.gaps)
