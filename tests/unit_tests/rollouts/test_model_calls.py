# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest
from omegaconf import OmegaConf

from nemo_gym.models.model_call_capture import CaptureStore
from nemo_gym.rollouts.model_calls import _build_captured_model_call_record


def test_observability_enabled_from_config(tmp_path):
    from nemo_gym.rollouts.model_calls import observability_enabled_from_config

    assert observability_enabled_from_config({}) is False

    global_config = OmegaConf.create({"observability_enabled": True, "model_call_capture_dir": str(tmp_path)})
    assert observability_enabled_from_config(global_config) is True

    with pytest.raises(ValueError, match="required"):
        observability_enabled_from_config({"observability_enabled": True})


def _capture_exchange(dialect, model_server, usage, response):
    return {
        "model_call_id": f"call-{model_server}",
        "dialect": dialect,
        "model_ref": {"type": "responses_api_models", "name": model_server},
        "started_at": 100.0,
        "completed_at": 100.01,
        "latency_ms": 1.0,
        "status_code": 200,
        "error_category": None,
        "request": {"input": "hi"},
        "response": {"model": "m", "usage": usage, **response},
    }


def test_merge_capture_attaches_metrics_without_raw_payloads(tmp_path):
    from nemo_gym.models.model_call_capture import CaptureStore
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    exchange = _capture_exchange(
        "responses",
        "A",
        {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        {
            "id": "resp-A",
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
        },
    )
    exchange["request_raw"] = "malformed request"
    exchange["response_raw"] = "malformed response"
    store.record("0-0", exchange)

    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "reward": 1.0,
        "response": {"harness": "A"},
        "ng_agent_observations": {
            "source": "test",
            "records": [
                {
                    "kind": "agent_invocation",
                    "invocation_id": "root",
                    "model_calls": [
                        {
                            "model_ref": {"type": "responses_api_models", "name": "A"},
                            "response_id": "resp-A",
                        }
                    ],
                }
            ],
            "gaps": [{"code": "model_call_ownership_unavailable", "detail": "capture:stale"}],
        },
    }
    merge_model_call_capture_into_record(record, [tmp_path])

    capture = record["ng_model_call_capture"]
    assert set(capture) == {"rollout_id", "metrics", "calls"}
    assert capture["rollout_id"] == "0-0"
    assert capture["metrics"]["num_calls"] == 1
    attached_call = capture["calls"][0]
    assert attached_call["model_call_id"] == "call-A"
    assert attached_call["response_id"] == "resp-A"
    assert attached_call["model_ref"] == {"type": "responses_api_models", "name": "A"}
    assert attached_call["model"] == "m"
    assert attached_call["response_status"] == "completed"
    assert attached_call["started_at"] == 100.0 and attached_call["completed_at"] == 100.01
    assert attached_call["tokens_in"] == 3
    assert {"request", "response", "request_raw", "response_raw"}.isdisjoint(attached_call)
    assert record["response"] == {"harness": "A"} and record["reward"] == 1.0
    [joined_ref] = record["ng_agent_observations"]["records"][0]["model_calls"]
    assert joined_ref["model_call_id"] == "call-A"
    assert record["ng_agent_observations"]["gaps"] == []

    with_payloads = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    merge_model_call_capture_into_record(with_payloads, [tmp_path], include_payloads=True)
    attached_call = with_payloads["ng_model_call_capture"]["calls"][0]
    assert attached_call["request"] == exchange["request"]
    assert attached_call["response"] == exchange["response"]
    assert attached_call["request_raw"] == "malformed request"
    assert attached_call["response_raw"] == "malformed response"


def test_merge_capture_owns_opencode_calls_by_client_session(tmp_path):
    from nemo_gym.models.model_call_capture import CaptureStore
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    exchange = _capture_exchange(
        "chat",
        "A",
        {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        {"id": "resp-A", "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]},
    )
    exchange["client_session_id"] = "opencode-root"
    store.record("0-0", exchange)
    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "ng_agent_observations": {
            "source": "opencode",
            "records": [{"kind": "agent_invocation", "invocation_id": "opencode-root"}],
        },
    }

    merge_model_call_capture_into_record(record, [tmp_path])

    [reference] = record["ng_agent_observations"]["records"][0]["model_calls"]
    assert reference["model_call_id"] == "call-A"
    assert record["ng_agent_observations"]["gaps"] == []


def test_merge_capture_reports_observation_join_failure(tmp_path, monkeypatch):
    from nemo_gym.models.model_call_capture import CaptureStore
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    exchange = _capture_exchange("chat", "A", {}, {"id": "resp-A"})
    exchange["client_session_id"] = "opencode-root"
    store.record("0-0", exchange)
    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "ng_agent_observations": {
            "source": "opencode",
            "records": [{"kind": "agent_invocation", "invocation_id": "opencode-root"}],
        },
    }

    def fail_association(*_args, **_kwargs):
        raise RuntimeError("association failed")

    monkeypatch.setattr("nemo_gym.rollouts.model_calls.join_agent_model_calls", fail_association)
    merge_model_call_capture_into_record(record, [tmp_path])

    assert [gap["code"] for gap in record["ng_model_call_capture"]["gaps"]] == ["agent_observation_join_failed"]


def test_merge_capture_reports_missing_capture(tmp_path):
    from nemo_gym.models.model_call_capture import CaptureStore
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    rec = {
        "_ng_task_index": 9,
        "_ng_rollout_index": 9,
        "reward": 1.0,
        "ng_agent_observations": {
            "source": "test",
            "records": [
                {
                    "kind": "agent_invocation",
                    "invocation_id": "root",
                    "model_calls": [{"model_call_id": "missing"}],
                }
            ],
            "gaps": [{"code": "model_call_ownership_unavailable"}],
        },
    }
    merge_model_call_capture_into_record(rec, [tmp_path])  # no capture file for 9-9
    assert rec["ng_model_call_capture"]["calls"] == []
    assert [gap["code"] for gap in rec["ng_model_call_capture"]["gaps"]] == ["model_call_capture_no_records"]
    assert {gap["code"] for gap in rec["ng_agent_observations"]["gaps"]} == {
        "model_call_ownership_unavailable",
        "model_call_reference_unmatched",
    }

    CaptureStore(tmp_path).path_for("8-8").touch()
    empty = {"_ng_task_index": 8, "_ng_rollout_index": 8}
    merge_model_call_capture_into_record(empty, [tmp_path])
    assert [gap["code"] for gap in empty["ng_model_call_capture"]["gaps"]] == ["model_call_capture_no_records"]


def test_merge_capture_preserves_valid_records_around_malformed_data(tmp_path):
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    first = _capture_exchange("responses", "A", {}, {"id": "resp-A"})
    third = _capture_exchange("responses", "B", {}, {"id": "resp-B"})
    store.path_for("9-8").write_bytes(orjson.dumps(first) + b"\n{not-json}\n" + orjson.dumps(third) + b"\n")
    record = {"_ng_task_index": 9, "_ng_rollout_index": 8}

    merge_model_call_capture_into_record(record, [tmp_path])

    assert [call["call_index"] for call in record["ng_model_call_capture"]["calls"]] == [0, 2]
    assert [call["response_id"] for call in record["ng_model_call_capture"]["calls"]] == ["resp-A", "resp-B"]
    assert record["ng_model_call_capture"]["gaps"] == [
        {
            "code": "model_call_capture_records_unreadable",
            "detail": "count=1",
        }
    ]


def test_merge_capture_reports_partial_write_loss(tmp_path):
    from nemo_gym.models.model_call_capture import CaptureStore
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    store.record("4-2", _capture_exchange("responses", "A", {}, {"id": "resp-A"}))
    store.mark_incomplete("4-2")
    record = {"_ng_task_index": 4, "_ng_rollout_index": 2}

    merge_model_call_capture_into_record(record, [tmp_path])

    assert len(record["ng_model_call_capture"]["calls"]) == 1
    assert [gap["code"] for gap in record["ng_model_call_capture"]["gaps"]] == ["model_call_capture_incomplete"]


def test_clear_model_call_captures_for_rollouts_run_scoping(tmp_path, monkeypatch):
    import nemo_gym.rollouts.model_calls as obs
    from nemo_gym.rollouts.model_calls import clear_model_call_captures_for_rollouts

    store = CaptureStore(tmp_path)
    store.record("0-0", {"dialect": "chat", "request": {}, "response": {}})
    store.record("1-0", {"dialect": "chat", "request": {}, "response": {}})
    store.mark_incomplete("0-0")
    assert store.read("0-0") and store.read("1-0")

    # Clears only the rollout ids about to be (re)run; rows without indices are skipped, others stay.
    clear_model_call_captures_for_rollouts([{"_ng_task_index": 0, "_ng_rollout_index": 0}, {"no": "id"}], [tmp_path])
    assert store.read("0-0") == [] and not store.is_incomplete("0-0") and store.read("1-0")
    clear_model_call_captures_for_rollouts([{"_ng_task_index": 1, "_ng_rollout_index": 0}], [])  # no dirs -> no-op
    assert store.read("1-0")

    # A stale-capture cleanup failure must be visible rather than mixing old and new calls.
    def _boom(_directory):
        raise OSError("cannot open")

    monkeypatch.setattr(obs, "CaptureStore", _boom)
    with pytest.raises(OSError, match="cannot open"):
        clear_model_call_captures_for_rollouts([{"_ng_task_index": 1, "_ng_rollout_index": 0}], [tmp_path])


@pytest.mark.asyncio
async def test_await_quiescence_waits_for_a_detached_write_to_land(tmp_path):
    import asyncio

    from nemo_gym.rollouts.model_calls import await_model_call_capture_quiescence

    store = CaptureStore(tmp_path)
    store.mark_pending("2-0", "call-1")

    async def late_write():
        await asyncio.sleep(0.2)
        store.record("2-0", _capture_exchange("responses", "A", {}, {"id": "resp-A"}))
        store.clear_pending("2-0", "call-1")

    task = asyncio.create_task(late_write())
    # The race is real at this point: snapshotting now would miss the row.
    assert store.read("2-0") == []

    quiesced = await await_model_call_capture_quiescence("2-0", [tmp_path])
    await task

    assert quiesced is True
    assert len(store.read("2-0")) == 1


@pytest.mark.asyncio
async def test_await_quiescence_gives_up_on_a_stranded_claim(tmp_path):
    from nemo_gym.rollouts.model_calls import await_model_call_capture_quiescence

    CaptureStore(tmp_path).mark_pending("3-0", "call-1")

    assert await await_model_call_capture_quiescence("3-0", [tmp_path], timeout_s=0.05, poll_s=0.01) is False


def test_merge_capture_flags_a_claim_that_never_landed(tmp_path):
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    store.record("4-0", _capture_exchange("responses", "A", {}, {"id": "resp-A"}))
    store.mark_pending("4-0", "call-2")

    record = {"_ng_task_index": 4, "_ng_rollout_index": 0}
    merge_model_call_capture_into_record(record, [tmp_path])

    assert "model_call_capture_pending" in [gap["code"] for gap in record["ng_model_call_capture"]["gaps"]]


def test_clear_captures_for_rollouts_drops_stale_pending_claims(tmp_path):
    from nemo_gym.rollouts.model_calls import clear_model_call_captures_for_rollouts

    store = CaptureStore(tmp_path)
    store.mark_pending("5-0", "call-1")

    clear_model_call_captures_for_rollouts([{"_ng_task_index": 5, "_ng_rollout_index": 0}], [tmp_path])

    assert store.has_pending("5-0") is False


def _cancelled_exchange(response_raw, *, status_code=200, dialect="responses"):
    """A call the capture recorded as cancelled: no parsed response, the body only as raw text."""
    return {
        "model_call_id": "call-cancelled",
        "dialect": dialect,
        "model_ref": {"type": "responses_api_models", "name": "policy_model"},
        "started_at": 100.0,
        "completed_at": 100.5,
        "latency_ms": 500.0,
        "status_code": status_code,
        "error_category": "cancelled",
        "request": {"input": "hi"},
        "response": None,
        "response_raw": response_raw,
    }


_COMPLETE_RESPONSES_BODY = {
    "id": "resp-cancelled",
    "model": "m",
    "status": "completed",
    "usage": {
        "input_tokens": 12,
        "output_tokens": 7,
        "total_tokens": 19,
        "output_tokens_details": {"reasoning_tokens": 5},
    },
    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
}


def test_cancelled_call_with_a_complete_body_recovers_its_response():
    exchange = _cancelled_exchange(orjson.dumps(_COMPLETE_RESPONSES_BODY).decode())

    record = _build_captured_model_call_record(exchange, call_index=0)

    assert record.response == _COMPLETE_RESPONSES_BODY
    assert record.response_raw is None
    assert record.response_id == "resp-cancelled"
    assert record.response_status == "completed"
    assert (record.tokens_in, record.tokens_out, record.tokens_total, record.tokens_reasoning) == (12, 7, 19, 5)
    # The cancellation itself is still reported.
    assert record.error_category == "cancelled"
    assert record.status_code == 200


def test_cancelled_chat_call_with_a_complete_body_recovers_its_response():
    body = {
        "id": "chatcmpl-1",
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
    }

    record = _build_captured_model_call_record(
        _cancelled_exchange(orjson.dumps(body).decode(), dialect="chat"), call_index=0
    )

    assert record.response == body
    assert record.response_id == "chatcmpl-1"
    assert record.finish_reason == "stop"
    assert (record.tokens_in, record.tokens_out, record.tokens_total) == (9, 3, 12)


@pytest.mark.parametrize(
    "response_raw, status_code",
    [
        pytest.param('{"id": "resp-1", "usage": {"input_t', 200, id="body cut before it was complete"),
        pytest.param("", 200, id="empty body"),
        pytest.param(None, 200, id="no body"),
        pytest.param("[1, 2, 3]", 200, id="JSON that is not an object"),
        pytest.param(orjson.dumps(_COMPLETE_RESPONSES_BODY).decode(), None, id="no status"),
        pytest.param('{"error": {"message": "overloaded"}}', 500, id="error status"),
    ],
)
def test_cancelled_call_without_a_complete_answer_keeps_what_was_captured(response_raw, status_code):
    record = _build_captured_model_call_record(
        _cancelled_exchange(response_raw, status_code=status_code), call_index=0
    )

    assert record.response is None
    assert record.response_raw == response_raw
    assert record.response_id is None
    assert (record.tokens_in, record.tokens_out) == (None, None)
    assert record.error_category == "cancelled"


def test_only_cancelled_calls_have_their_raw_body_parsed():
    exchange = _cancelled_exchange(orjson.dumps(_COMPLETE_RESPONSES_BODY).decode())
    exchange["error_category"] = "capture_parse_error"

    record = _build_captured_model_call_record(exchange, call_index=0)

    assert record.response is None
    assert record.response_raw == exchange["response_raw"]
    assert record.tokens_out is None


def test_a_parsed_response_is_never_replaced_by_its_raw_body():
    exchange = _cancelled_exchange('{"id": "from-raw"}')
    exchange["response"] = {"id": "parsed", "usage": {"input_tokens": 1, "output_tokens": 2}}
    exchange["error_category"] = None

    record = _build_captured_model_call_record(exchange, call_index=0)

    assert record.response_id == "parsed"
    assert record.response == exchange["response"]
    assert record.response_raw == '{"id": "from-raw"}'


def test_merge_capture_binds_a_cancelled_call_with_a_complete_body_to_its_agent_turn(tmp_path):
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    CaptureStore(tmp_path).record("0-0", _cancelled_exchange(orjson.dumps(_COMPLETE_RESPONSES_BODY).decode()))
    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "ng_agent_observations": {
            "source": "test",
            "records": [
                {
                    "kind": "agent_invocation",
                    "invocation_id": "root",
                    "model_calls": [
                        {
                            "model_ref": {"type": "responses_api_models", "name": "policy_model"},
                            "response_id": "resp-cancelled",
                        }
                    ],
                }
            ],
            "gaps": [],
        },
    }

    merge_model_call_capture_into_record(record, [tmp_path], include_payloads=True)

    capture = record["ng_model_call_capture"]
    [call] = capture["calls"]
    assert call["response_id"] == "resp-cancelled"
    assert (call["tokens_in"], call["tokens_out"]) == (12, 7)
    assert call["response"] == _COMPLETE_RESPONSES_BODY
    assert "response_raw" not in call or call["response_raw"] is None
    assert capture["metrics"]["tokens_out"] == 7
    [joined_ref] = record["ng_agent_observations"]["records"][0]["model_calls"]
    assert joined_ref["model_call_id"] == "call-cancelled"
    assert record["ng_agent_observations"]["gaps"] == []


def _sse(*events):
    """Server-sent events: each is (event name or None, payload as a dict or a literal string)."""
    blocks = []
    for name, payload in events:
        data = payload if isinstance(payload, str) else orjson.dumps(payload).decode()
        blocks.append((f"event: {name}\n" if name else "") + f"data: {data}\n\n")
    return "".join(blocks)


def test_cancelled_streamed_responses_call_with_a_complete_stream_recovers_its_response():
    raw = _sse(
        ("response.created", {"type": "response.created", "response": {"id": "resp-cancelled"}}),
        ("response.completed", {"type": "response.completed", "response": _COMPLETE_RESPONSES_BODY}),
    )

    record = _build_captured_model_call_record(_cancelled_exchange(raw), call_index=0)

    assert record.response == _COMPLETE_RESPONSES_BODY
    assert record.response_raw is None
    assert (record.response_id, record.tokens_in, record.tokens_out) == ("resp-cancelled", 12, 7)


def test_cancelled_streamed_chat_call_with_a_complete_stream_recovers_its_response():
    chunk = {"id": "chatcmpl-1", "model": "m"}
    raw = _sse(
        (None, {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "o"}}]}),
        (None, {**chunk, "choices": [{"index": 0, "delta": {"content": "k"}, "finish_reason": "stop"}]}),
        (None, {**chunk, "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}}),
        (None, "[DONE]"),
    )

    record = _build_captured_model_call_record(_cancelled_exchange(raw, dialect="chat"), call_index=0)

    assert record.response["choices"][0]["message"]["content"] == "ok"
    assert record.response_raw is None
    assert (record.response_id, record.finish_reason, record.tokens_in, record.tokens_out) == (
        "chatcmpl-1",
        "stop",
        9,
        3,
    )


@pytest.mark.parametrize(
    "raw, dialect",
    [
        pytest.param(
            _sse(("response.created", {"type": "response.created", "response": {"id": "resp-1"}})),
            "responses",
            id="stream cut before its terminal event",
        ),
        pytest.param(
            _sse(("response.failed", {"type": "response.failed", "response": {"id": "resp-1"}})),
            "responses",
            id="stream ended with a failure",
        ),
        pytest.param(
            _sse(("response.completed", {"type": "response.completed", "response": _COMPLETE_RESPONSES_BODY})),
            "unknown",
            id="unknown dialect",
        ),
        pytest.param(
            _sse(("response.completed", {"type": "response.completed", "response": _COMPLETE_RESPONSES_BODY})),
            None,
            id="no dialect",
        ),
    ],
)
def test_cancelled_streamed_call_without_a_complete_stream_keeps_its_raw_body(raw, dialect):
    exchange = _cancelled_exchange(raw)
    exchange["dialect"] = dialect

    record = _build_captured_model_call_record(exchange, call_index=0)

    assert record.response is None
    assert record.response_raw == raw
    assert record.tokens_out is None


def test_recovering_a_complete_cancelled_stream_keeps_its_failure_report(tmp_path):
    from nemo_gym.core.failure_records import FailureReport
    from nemo_gym.core.failure_transport import FailureHeader, FailureReportRow
    from nemo_gym.rollouts.model_calls import merge_model_call_capture_into_record

    # An omitted report is evidence too; recovery must not erase it with the raw SSE body.
    raw = _sse(
        (
            "response.completed",
            {
                "type": "response.completed",
                "response": _COMPLETE_RESPONSES_BODY,
                FailureHeader.STREAM_FIELD: FailureReport(omitted=True).model_dump(mode="json"),
            },
        )
    )
    store = CaptureStore(tmp_path)
    store.record("0-0", _cancelled_exchange(raw))
    record = {"_ng_task_index": 0, "_ng_rollout_index": 0}

    merge_model_call_capture_into_record(record, [tmp_path], include_payloads=True)

    assert FailureReportRow.report(record).omitted
    [call] = record["ng_model_call_capture"]["calls"]
    assert call["response"] == _COMPLETE_RESPONSES_BODY
    assert call["response_raw"] is None
