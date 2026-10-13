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
            failure_reason="Judge timed out", terminal=False, failure_kind="judge_failed", stage="verification"
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
        failure=EpisodeFailure(failure_reason="No reply", terminal=terminal, failure_kind="transport_timeout"),
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
        failure_reason="Judge unavailable",
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
        "failure_reason": "Judge unavailable",
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
        "failure": {"failure_reason": "Input could not be sent", "terminal": True},
    }
    assert RolloutFailure.model_validate(payload).delivery == "not_sent"
    for field in ("run_id", "source", "delivery"):
        with pytest.raises(ValidationError, match=field):
            RolloutFailure.model_validate({key: value for key, value in payload.items() if key != field})


def test_public_response_schema_keeps_the_explicit_failure_fields() -> None:
    schema = RolloutFailure.model_json_schema(mode="serialization")
    failure = schema["$defs"]["EpisodeFailure"]
    assert schema["properties"]["schema_version"]["const"] == 1
    properties = {
        "RolloutFailure": set(schema["properties"]),
        "EpisodeFailure": set(failure["properties"]),
        "EpisodeId": set(schema["$defs"]["EpisodeId"]["properties"]),
    }
    assert properties == {
        "RolloutFailure": {
            "schema_version",
            "episode_id",
            "run_id",
            "source",
            "delivery",
            "failure",
            "http_status",
            "exception_type",
        },
        "EpisodeFailure": {"failure_reason", "terminal", "failure_kind", "stage"},
        "EpisodeId": {"rollout_id", "attempt"},
    }, "Changing version-1 saved fields requires reviewing the RolloutFailure schema_version bump policy"
    assert failure["additionalProperties"] is False
    assert failure["required"] == ["failure_reason", "terminal"]
    assert failure["properties"]["failure_reason"]["maxLength"] == 2000
    assert "message" not in failure["properties"]
    assert failure["properties"]["stage"]["anyOf"][0]["enum"] == [
        "admission",
        "seed",
        "agent",
        "verification",
        "cleanup",
    ]
    assert "failure_kind" in failure["properties"]


@pytest.mark.parametrize("schema_version", [None, 1, 2])
def test_saved_failure_record_versions(schema_version: int | None) -> None:
    payload = {
        "episode_id": {"rollout_id": "task-7"},
        "run_id": "eval-1",
        "source": "collector",
        "delivery": "not_sent",
        "failure": {"failure_reason": "Input could not be sent", "terminal": True},
    }
    if schema_version is not None:
        payload["schema_version"] = schema_version
    if schema_version == 2:
        with pytest.raises(ValidationError, match="schema_version"):
            RolloutFailure.model_validate(payload)
    else:
        record = RolloutFailure.model_validate(payload)
        saved = json.loads(record.model_dump_json())
        assert saved["schema_version"] == 1
        assert saved["failure"]["failure_reason"] == "Input could not be sent"
        assert RolloutFailure.model_validate_json(record.model_dump_json()) == record
        with pytest.raises(ValidationError, match="Extra inputs"):
            RolloutFailure.model_validate(saved | {"future_transport_field": "unknown"})


@pytest.mark.parametrize("http_status", [None, 100, 599, 99, 600])
def test_saved_failure_http_status_bounds(http_status: int | None) -> None:
    payload = {
        "episode_id": {"rollout_id": "task-7"},
        "run_id": "eval-1",
        "source": "collector",
        "delivery": "possibly_delivered",
        "failure": {"failure_reason": "No reply", "terminal": False},
        "http_status": http_status,
    }
    if http_status is not None and not 100 <= http_status <= 599:
        with pytest.raises(ValidationError, match="http_status"):
            RolloutFailure.model_validate(payload)
    else:
        record = RolloutFailure.model_validate(payload)
        assert RolloutFailure.model_validate_json(record.model_dump_json()).http_status == http_status


@pytest.mark.parametrize("source", ["environment", "collector"])
@pytest.mark.parametrize("delivery", ["not_sent", "possibly_delivered", "delivered"])
def test_environment_failure_requires_established_delivery(source: str, delivery: str) -> None:
    payload = {
        "episode_id": {"rollout_id": "task-7"},
        "run_id": "eval-1",
        "source": source,
        "delivery": delivery,
        "failure": {"failure_reason": "Request failed", "terminal": False},
    }
    if source == "environment" and delivery != "delivered":
        with pytest.raises(ValidationError, match="requires delivery='delivered'"):
            RolloutFailure.model_validate_json(json.dumps(payload))
    else:
        record = RolloutFailure.model_validate_json(json.dumps(payload))
        assert record.source == source and record.delivery == delivery
        assert RolloutFailure.model_validate_json(record.model_dump_json()) == record
