# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import ValidationError

from nemo_gym.episode_types import BaseEpisodeResponse, EpisodeFailure, EpisodeId, TaskId
from nemo_gym.rollout_outcomes import RolloutFailure


def test_same_failure_survives_wire_and_persisted_record_json() -> None:
    wire = BaseEpisodeResponse[str](
        episode_id=EpisodeId(rollout_id="task-7", attempt=2),
        task_id=TaskId(taskset="eval", task_id="7"),
        failure=EpisodeFailure(
            message="Judge timed out", terminal=False, failure_kind="judge_failed", stage="verification"
        ),
    )
    received = BaseEpisodeResponse[str].model_validate_json(wire.model_dump_json())
    record = RolloutFailure(
        episode_id=received.episode_id,
        run_id="eval-1",
        source="environment",
        delivery="delivered",
        failure=received.failure,
    )
    saved = RolloutFailure.model_validate_json(record.model_dump_json())
    assert saved.failure == received.failure
    assert saved.episode_id == received.episode_id
    assert saved.episode_id.capture_key == "task-7-a2"
    assert saved.run_id == "eval-1"
    assert saved.source == "environment" and saved.delivery == "delivered"
    assert "reward" not in json.loads(record.model_dump_json())


@pytest.mark.parametrize("terminal", [False, True])
def test_collector_observation_does_not_invent_an_execution_stage_or_retry_decision(terminal: bool) -> None:
    record = RolloutFailure(
        episode_id=EpisodeId(rollout_id="task-7"),
        run_id="eval-1",
        source="collector",
        delivery="possibly_delivered",
        failure=EpisodeFailure(message="No reply", terminal=terminal, failure_kind="transport_timeout"),
        http_status=504,
        exception_type="TimeoutError",
    )
    saved = RolloutFailure.model_validate_json(record.model_dump_json())
    assert saved.failure.stage is None
    assert saved.failure.terminal is terminal
    assert saved.delivery == "possibly_delivered"
    assert saved.http_status == 504 and saved.exception_type == "TimeoutError"
    assert saved.episode_id.capture_key == "task-7"


def test_protocol_diagnostics_remain_outside_the_serialized_failure_record() -> None:
    class DiagnosticFailure(EpisodeFailure):
        partial_response: dict[str, str]

    failure = DiagnosticFailure(
        message="Judge unavailable",
        terminal=False,
        failure_kind="judge_failed",
        stage="verification",
        partial_response={"answer": "42"},
    )
    assert failure.model_dump()["partial_response"] == {"answer": "42"}
    record = RolloutFailure(
        episode_id=EpisodeId(rollout_id="task-7"),
        run_id="eval-1",
        source="environment",
        delivery="delivered",
        failure=failure,
    )
    payload = json.loads(record.model_dump_json())
    assert payload["failure"] == {
        "message": "Judge unavailable",
        "terminal": False,
        "failure_kind": "judge_failed",
        "stage": "verification",
    }
    for extra in ({"reward": 0}, {"response": {}}, {"messages": []}, {"token_ids": []}):
        with pytest.raises(ValidationError, match="Extra inputs"):
            RolloutFailure.model_validate(payload | extra)
        with pytest.raises(ValidationError, match="Extra inputs"):
            EpisodeFailure.model_validate(payload["failure"] | extra)


def test_saved_record_requires_explicit_run_and_delivery_evidence() -> None:
    payload = {
        "episode_id": {"rollout_id": "task-7", "attempt": 0},
        "run_id": "eval-1",
        "source": "collector",
        "delivery": "not_sent",
        "failure": {"message": "Input could not be sent", "terminal": True},
    }
    assert RolloutFailure.model_validate(payload).delivery == "not_sent"
    for field in ("run_id", "source", "delivery"):
        with pytest.raises(ValidationError, match=field):
            RolloutFailure.model_validate({key: value for key, value in payload.items() if key != field})


def test_public_response_schema_keeps_the_explicit_failure_fields() -> None:
    schema = RolloutFailure.model_json_schema(mode="serialization")
    failure = schema["$defs"]["EpisodeFailure"]
    assert failure["additionalProperties"] is False
    assert failure["required"] == ["message", "terminal"]
    assert failure["properties"]["message"]["maxLength"] == 2000
    assert failure["properties"]["stage"]["anyOf"][0]["enum"] == ["seed", "agent", "verification", "cleanup"]
    assert "failure_kind" in failure["properties"]
