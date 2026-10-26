# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock

import pytest

from nemo_gym.episode_types import EpisodeFailure
from nemo_gym.failure_kinds import MODEL_REQUEST_ERROR, VERIFIER_ERROR
from nemo_gym.path_utils import failures_path_for
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.typed_exceptions import (
    ModelFailure,
    ResServerFailure,
    boundary_failure,
    encode_events_header,
    failure_event_map,
)
from tests.unit_tests.test_rollout_collection import FakeResponse, install_fake_server_client


@pytest.mark.parametrize("body_dropped, header_dropped", [(0, 0), (2, 3), (3, 3)])
@pytest.mark.parametrize("forward_recovered", [False, True, "same", "projection"])
async def test_collector_unions_body_failure_and_recovered_header_events(
    tmp_path, monkeypatch, forward_recovered, body_dropped, header_dropped
):
    earlier = ModelFailure(MODEL_REQUEST_ERROR, "recovered model request", http_status=429)
    current = ResServerFailure(
        VERIFIER_ERROR, chr(30028) * 1900 if forward_recovered == "projection" else "current verifier failure"
    )
    own = failure_event_map(current)
    failure = EpisodeFailure(
        failure_reason="current verifier failure",
        terminal=True,
        failure_kind=VERIFIER_ERROR,
        failure_id=current.event_id,
        failure_events=own,
        failure_events_dropped=body_dropped or None,
    )
    reply = FakeResponse(
        200,
        {
            "episode_id": {"rollout_id": "r", "attempt": 0},
            "task_id": {"taskset": "t:d", "task_id": "1"},
            "failure": failure.model_dump(mode="json"),
        },
    )
    reply.headers = (
        encode_events_header(own if forward_recovered == "same" else failure_event_map(earlier))
        if forward_recovered
        else {}
    )
    if header_dropped:
        reply.headers["X-NG-Failure-Events-Dropped"] = str(header_dropped)
    if forward_recovered == "projection":
        from nemo_gym.typed_exceptions import FAILURE_EVENT_HEADER, decode_header, encode_header

        projected = decode_header({FAILURE_EVENT_HEADER: encode_header(failure)})
        reply.headers.update(encode_events_header(projected.failure_events))
    post = AsyncMock(return_value=reply)
    client = install_fake_server_client(monkeypatch, post)
    monkeypatch.setattr("nemo_gym.rollout_collection.get_global_config_dict", lambda: client.global_config_dict)
    inputs = tmp_path / "input.jsonl"
    output = tmp_path / "rollouts.jsonl"
    inputs.write_text(
        json.dumps({"task_id": {"taskset": "t:d", "task_id": "1"}, "task_input": {"value": "task"}}) + "\n"
    )
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(inputs),
        output_jsonl_fpath=str(output),
        disable_aggregation=True,
        disable_health_check=True,
        route_failures_to_sidecar=True,
        environment_server_routes={"t:d": "my_environment_server"},
    )
    with pytest.raises(RuntimeError, match="None of the"):
        await RolloutCollectionHelper().run_from_config(config)
    saved = [json.loads(x) for x in failures_path_for(output).read_text().splitlines()]
    assert len(saved) == 1 and post.await_count == 1
    row = saved[0]
    assert row["_ng_failure_terminal"] is True and row["_ng_failure_kind"] == VERIFIER_ERROR
    assert "reward" not in row and "response" not in row
    expected = {current.event_id} | ({earlier.event_id} if forward_recovered is True else set())
    assert row["_ng_failure_id"] == current.event_id
    assert row.get("_ng_failure_events_dropped", 0) == max(body_dropped, header_dropped)
    assert set(row["_ng_failure_events"]) == expected
    assert row["_ng_failure_events"][current.event_id]["failure_reason"] == current.failure_reason


def test_foreign_outer_keeps_its_typed_explicit_cause_event():
    inner = ModelFailure(MODEL_REQUEST_ERROR, "model request failed", http_status=503)
    try:
        raise RuntimeError("foreign boundary wrapper") from inner
    except RuntimeError as outer:
        failure = boundary_failure(outer, server="agent")
        repeated = boundary_failure(outer, server="agent")
        assert repeated.failure_id == failure.failure_id
        assert repeated.failure_events == failure.failure_events
    events = failure.failure_events or {}
    assert inner.event_id in events
    assert events[inner.event_id].failure_kind == MODEL_REQUEST_ERROR
    outer_events = [event for event in events.values() if event.exception_type == "RuntimeError"]
    assert len(outer_events) == 1 and outer_events[0].cause_id == inner.event_id
    assert outer_events[0].failure_kind is None

    assert failure.failure_id == outer_events[0].event_id
    assert failure.failure_kind == MODEL_REQUEST_ERROR
    assert failure.failure_reason == "model request failed"
    assert failure.terminal is False
    assert events[inner.event_id].http_status == 503


def test_foreign_implicit_context_does_not_acquire_a_typed_cause():
    try:
        try:
            raise ModelFailure(MODEL_REQUEST_ERROR, "earlier request")
        except ModelFailure:
            raise RuntimeError("unrelated handler problem")
    except RuntimeError as outer:
        failure = boundary_failure(outer, server="agent")
    assert failure.failure_events is None
    assert failure.failure_id is None
    assert failure.failure_kind is None


@pytest.mark.parametrize("reverse", [False, True])
def test_rich_body_and_bounded_header_keep_one_event(reverse):
    from nemo_gym.rollout_collection import _merge_observation_fields
    from nemo_gym.typed_exceptions import FAILURE_EVENT_HEADER, decode_header, encode_header, row_fields

    error = ModelFailure(MODEL_REQUEST_ERROR, chr(30028) * 1900, http_status=503)
    full = boundary_failure(error, server="model")
    partial = decode_header({FAILURE_EVENT_HEADER: encode_header(full)})
    assert partial.failure_events[error.event_id].reason_truncated
    left, right = row_fields(full), row_fields(partial)
    if reverse:
        left, right = right, left
    row = _merge_observation_fields(left, right)
    assert row["_ng_failure_id"] == error.event_id
    assert row["_ng_failure_events"][error.event_id]["failure_reason"] == error.failure_reason
    assert not row.get("_ng_failure_events_conflicted")


@pytest.mark.parametrize("reverse", [False, True])
def test_missing_cause_projection_merges_without_hiding_a_real_conflict(reverse):
    from nemo_gym.rollout_collection import _merge_observation_fields
    from nemo_gym.typed_exceptions import (
        bounded_failure_events,
        decode_events_header,
        encode_events_header,
        events_row_value,
    )

    inner = ModelFailure(MODEL_REQUEST_ERROR, "i" * 1900, http_status=401)
    outer = ResServerFailure(VERIFIER_ERROR, "outer")
    outer.__cause__ = inner
    full = failure_event_map(outer)
    shortened, dropped = bounded_failure_events(full, keep=outer.event_id, max_bytes=800)
    assert dropped == 1 and shortened[outer.event_id].cause_truncated
    shortened = decode_events_header(encode_events_header(shortened))
    left = {"_ng_failure_events": events_row_value(full), "_ng_failure_id": outer.event_id}
    right = {"_ng_failure_events": events_row_value(shortened)}
    if reverse:
        left, right = right, left
    row = _merge_observation_fields(left, right)
    assert set(row["_ng_failure_events"]) == set(full)
    assert row["_ng_failure_events"][outer.event_id]["cause_id"] == inner.event_id
    assert not row.get("_ng_failure_events_conflicted")
    contradictory = dict(full)
    contradictory[inner.event_id] = full[inner.event_id].model_copy(update={"http_status": 503})
    bad = _merge_observation_fields(row, {"_ng_failure_events": events_row_value(contradictory)})
    assert inner.event_id not in bad["_ng_failure_events"]
    assert inner.event_id in bad["_ng_failure_events_conflicted"]
    assert bad["_ng_failure_events"][outer.event_id]["cause_truncated"]
    # A subsequent valid copy must not resurrect an ID already quarantined.
    again = _merge_observation_fields(bad, {"_ng_failure_events": events_row_value(full)})
    assert inner.event_id not in again["_ng_failure_events"]


@pytest.mark.parametrize("changed", [{"http_status": 401}, {"failure_kind": VERIFIER_ERROR}])
def test_source_truncation_is_explicit_and_never_hides_different_facts(changed):
    from nemo_gym.typed_exceptions import (
        FAILURE_EVENT_HEADER,
        conflicting_failure_ids,
        decode_header,
        encode_header,
        merge_failure_events,
    )

    error = ModelFailure(MODEL_REQUEST_ERROR, chr(30028) * 4500, http_status=503)
    full = boundary_failure(error, server="model")
    event = full.failure_events[error.event_id]
    assert event.reason_truncated and len(event.failure_reason) == 2000
    assert event.failure_reason.endswith(" [truncated]")
    partial = decode_header({FAILURE_EVENT_HEADER: encode_header(full)})
    assert not conflicting_failure_ids(full.failure_events, partial.failure_events)
    assert error.event_id in merge_failure_events(full.failure_events, partial.failure_events)
    bad = {error.event_id: partial.failure_events[error.event_id].model_copy(update=changed)}
    assert conflicting_failure_ids(full.failure_events, bad) == {error.event_id}
    assert error.event_id not in merge_failure_events(full.failure_events, bad)


def test_event_map_saved_record_roundtrip_retains_ids_and_version_one():
    from nemo_gym.rollout_outcomes import RolloutFailure

    error = ModelFailure(MODEL_REQUEST_ERROR, "provider failed", http_status=503)
    failure = boundary_failure(error, server="model")
    record = RolloutFailure.model_validate(
        {
            "episode_id": {"rollout_id": "r", "attempt": 0},
            "run_id": "run",
            "source": "environment",
            "delivery": "delivered",
            "failure": failure,
        }
    )
    restored = RolloutFailure.model_validate_json(record.model_dump_json())
    assert restored == record and restored.schema_version == 1
    assert restored.failure.failure_id == error.event_id
    assert restored.failure.failure_events[error.event_id].http_status == 503


def test_bounded_conflict_ids_report_loss_without_changing_outcome():
    from nemo_gym.rollout_collection import _events_row_fields, _merge_observation_fields
    from nemo_gym.typed_exceptions import (
        MAX_CONFLICTED_IDS,
        ObservedFailures,
        decode_conflicted_header,
        decode_dropped_header,
        encode_events_header,
    )

    observed = ObservedFailures()
    observed.conflicted.update(f"id-{i:02d}" for i in range(20))
    headers = encode_events_header(observed)
    assert len(decode_conflicted_header(headers)) == MAX_CONFLICTED_IDS
    assert decode_dropped_header(headers) > 0
    body = {"reward": 0.0, "mask_sample": False}
    result = _merge_observation_fields(body, _events_row_fields(headers))
    assert result is body and result["reward"] == 0.0 and result["mask_sample"] is False
    assert len(result["_ng_failure_events_conflicted"]) == MAX_CONFLICTED_IDS
    assert result["_ng_failure_events_dropped"] > 0


def test_non_ascii_conflicted_ids_are_bounded_by_serialized_header_bytes():
    from nemo_gym.typed_exceptions import (
        FAILURE_EVENTS_CONFLICTED_HEADER,
        ObservedFailures,
        decode_conflicted_header,
        decode_dropped_header,
        encode_events_header,
    )

    events = ObservedFailures()
    events.conflicted.update(chr(30028) * 125 + str(i) for i in range(16))
    headers = encode_events_header(events)
    wire = headers[FAILURE_EVENTS_CONFLICTED_HEADER]
    assert wire.isascii() and len(wire.encode("ascii")) <= 4096
    kept = decode_conflicted_header(headers)
    assert 0 < len(kept) < 16 and kept <= events.conflicted
    assert decode_dropped_header(headers) == 16 - len(kept)
