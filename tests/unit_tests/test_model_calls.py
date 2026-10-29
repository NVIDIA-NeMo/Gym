# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest
from omegaconf import OmegaConf

from nemo_gym.base_responses_api_model import CaptureStore
from nemo_gym.observability.model_calls import build_model_call_record


def test_build_model_call_record_from_exchange():
    exchange = {
        "model_call_id": "call-1",
        "client_session_id": "session-1",
        "client_assistant_message_id": "assistant-1",
        "dialect": "responses",
        "model_ref": {"type": "responses_api_models", "name": "srv"},
        "started_at": 100.0,
        "completed_at": 100.02,
        "latency_ms": 18.4,
        "request": {"input": "hi"},
        "response": {
            "id": "resp-1",
            "model": "m",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "output_tokens_details": {"reasoning_tokens": 3},
                "prompt_tokens_details": {"cached_tokens": 4},
            },
            "output": [
                {"type": "reasoning", "summary": [{"text": "thinking..."}]},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]},
                {"type": "function_call", "call_id": "c1", "name": "calc", "arguments": '{"x": 1}'},
            ],
        },
    }
    rec = build_model_call_record(exchange, call_index=3)
    assert rec.model_call_id == "call-1"
    assert rec.response_id == "resp-1"
    assert rec.client_session_id == "session-1"
    assert rec.client_assistant_message_id == "assistant-1"
    assert rec.call_index == 3
    assert rec.model_ref is not None and rec.model_ref.name == "srv"
    assert rec.model == "m"
    assert rec.dialect == "responses"
    assert rec.started_at == 100.0 and rec.completed_at == 100.02
    assert (rec.tokens_in, rec.tokens_out, rec.tokens_total, rec.tokens_reasoning) == (10, 5, 15, 3)
    assert rec.cache_hit is True and rec.cached_tokens == 4
    assert rec.reasoning_content == "thinking..."
    assert rec.tool_calls == [{"call_id": "c1", "name": "calc", "arguments": {"x": 1}}]
    assert rec.latency_total_ms == 18.4
    assert build_model_call_record({"response": {"id": 123}}, call_index=0).response_id is None
    empty = build_model_call_record({"request": {}, "response": {}}, call_index=0)
    assert empty.request == {}
    assert empty.response == {}
    assert empty.client_assistant_message_id is None
    assert (
        build_model_call_record({"client_assistant_message_id": 123}, call_index=0).client_assistant_message_id is None
    )
    assert {
        "model_call_id",
        "response_id",
        "client_session_id",
        "client_assistant_message_id",
        "call_index",
        "model_ref",
        "model",
        "dialect",
        "status_code",
        "response_status",
        "finish_reason",
        "started_at",
        "completed_at",
        "tokens_in",
        "tokens_out",
        "tokens_reasoning",
        "tokens_total",
        "request",
        "response",
        "request_raw",
        "response_raw",
        "tool_calls",
        "reasoning_content",
        "cache_hit",
        "cached_tokens",
        "cache_creation_tokens",
        "error_category",
        "latency_total_ms",
        "latency_ttft_ms",
    } <= type(rec).model_json_schema()["properties"].keys()

    aliases = {"cached_input_tokens": 4, "reasoning_output_tokens": 3}
    aliased = build_model_call_record({"response": {"usage": aliases}}, call_index=0)
    assert (aliased.cached_tokens, aliased.tokens_reasoning) == (4, 3)


@pytest.mark.parametrize(
    "response,expected_finish_reason,expected_response_status",
    [
        ({"choices": [{"finish_reason": "tool_calls"}]}, "tool_calls", None),
        ({"stop_reason": "end_turn"}, "end_turn", None),
        ({"incomplete_details": {"reason": "max_output_tokens"}}, "max_output_tokens", None),
        ({"status": "completed"}, None, "completed"),
        ({"choices": {"finish_reason": "invalid"}, "incomplete_details": []}, None, None),
    ],
)
def test_build_model_call_record_normalizes_termination(response, expected_finish_reason, expected_response_status):
    record = build_model_call_record({"response": response}, call_index=0)
    assert (record.finish_reason, record.response_status) == (expected_finish_reason, expected_response_status)


def test_build_model_call_record_tolerates_malformed_nested_shapes():
    record = build_model_call_record(
        {
            "response": {
                "usage": {"input_tokens": "invalid", "prompt_tokens_details": {"cached_tokens": []}},
                "choices": [{"message": "invalid"}],
                "output": [{"type": "reasoning", "summary": [{"text": {}}]}],
            }
        },
        call_index=0,
    )

    assert record.tokens_total is None
    assert record.tool_calls == []


def test_observability_enabled_from_config(tmp_path):
    from nemo_gym.observability.model_calls import observability_enabled_from_config

    assert observability_enabled_from_config({}) is False

    global_config = OmegaConf.create({"observability_enabled": True, "model_call_capture_dir": str(tmp_path)})
    assert observability_enabled_from_config(global_config) is True

    with pytest.raises(ValueError, match="required"):
        observability_enabled_from_config({"observability_enabled": True})


def test_as_arguments():
    from nemo_gym.observability.model_calls import _as_arguments

    assert _as_arguments({"x": 1}) == {"x": 1}
    assert _as_arguments('{"y": 2}') == {"y": 2}
    assert _as_arguments("not json") == {"_raw": "not json"}
    assert _as_arguments(123) == {}


def test_cache_signal():
    from nemo_gym.observability.model_calls import _cache_signal

    assert _cache_signal(None) == (None, None)
    assert _cache_signal({"prompt_tokens_details": {"cached_tokens": 4}}) == (True, 4)
    assert _cache_signal({"input_tokens_details": {"cached_tokens": 0}}) == (False, 0)
    assert _cache_signal(
        {
            "prompt_tokens_details": {"cached_tokens": 4},
            "input_tokens_details": {"cached_tokens": 9},
        }
    ) == (True, 4)
    assert _cache_signal({"cache_read_input_tokens": 5}) == (True, 5)  # Anthropic
    assert _cache_signal({"unrelated": 1}) == (None, None)


def test_extract_token_stats_chat_fallback():
    from nemo_gym.observability.model_calls import extract_token_stats

    assert extract_token_stats(None)["tokens_total"] is None
    stats = extract_token_stats(
        {"prompt_tokens": 5, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 1}}
    )
    assert (stats["tokens_in"], stats["tokens_out"], stats["tokens_total"], stats["tokens_reasoning"]) == (5, 3, 8, 1)


def test_tool_calls_and_reasoning_chat_and_anthropic():
    from nemo_gym.observability.model_calls import _tool_calls_and_reasoning

    chat = {
        "choices": [
            {
                "message": {
                    "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": '{"a": 1}'}}],
                    "reasoning_content": "think",
                }
            }
        ]
    }
    assert _tool_calls_and_reasoning(chat) == ([{"call_id": "c1", "name": "f", "arguments": {"a": 1}}], "think")

    anthropic = {
        "content": [
            {"type": "tool_use", "id": "t1", "name": "g", "input": {"b": 2}},
            {"type": "thinking", "thinking": "hmm"},
        ]
    }
    assert _tool_calls_and_reasoning(anthropic) == ([{"call_id": "t1", "name": "g", "arguments": {"b": 2}}], "hmm")
    assert _tool_calls_and_reasoning({"unrelated": 1}) == ([], None)


def test_tool_calls_and_reasoning_accepts_reasoning_alias():
    from nemo_gym.observability.model_calls import _tool_calls_and_reasoning

    # vLLM / newer servers emit `reasoning` (no `reasoning_content`).
    assert _tool_calls_and_reasoning({"choices": [{"message": {"content": "hi", "reasoning": "because"}}]}) == (
        [],
        "because",
    )
    # `reasoning_content` wins when both are present.
    both = {"choices": [{"message": {"reasoning_content": "rc", "reasoning": "r"}}]}
    assert _tool_calls_and_reasoning(both) == ([], "rc")


def test_tool_calls_and_reasoning_skips_non_dict_output_items():
    from nemo_gym.observability.model_calls import _tool_calls_and_reasoning

    # A Responses `output` list may contain non-dict junk; it must be skipped, not crash.
    resp = {"output": [None, "junk", {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}]}
    assert _tool_calls_and_reasoning(resp) == ([{"call_id": "c", "name": "f", "arguments": {}}], None)
    # Same guard on the Anthropic content blocks.
    anth = {"content": [None, "junk", {"type": "tool_use", "id": "t", "name": "g", "input": {"b": 2}}]}
    assert _tool_calls_and_reasoning(anth) == ([{"call_id": "t", "name": "g", "arguments": {"b": 2}}], None)


def test_tool_calls_and_reasoning_records_custom_tool_calls():
    from nemo_gym.observability.model_calls import _tool_calls_and_reasoning

    patch = "*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch\n"
    resp = {
        "output": [
            {"type": "custom_tool_call", "call_id": "c1", "name": "apply_patch", "input": patch},
            {"type": "function_call", "call_id": "c2", "name": "shell", "arguments": '{"command": "ls"}'},
        ]
    }
    assert _tool_calls_and_reasoning(resp) == (
        [
            {"call_id": "c1", "name": "apply_patch", "arguments": {"input": patch}},
            {"call_id": "c2", "name": "shell", "arguments": {"command": "ls"}},
        ],
        None,
    )


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
    from nemo_gym.base_responses_api_model import CaptureStore
    from nemo_gym.observability.model_calls import merge_model_call_capture_into_record

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
    from nemo_gym.base_responses_api_model import CaptureStore
    from nemo_gym.observability.model_calls import merge_model_call_capture_into_record

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
    from nemo_gym.base_responses_api_model import CaptureStore
    from nemo_gym.observability.model_calls import merge_model_call_capture_into_record

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

    monkeypatch.setattr("nemo_gym.observability.model_calls.join_agent_model_calls", fail_association)
    merge_model_call_capture_into_record(record, [tmp_path])

    assert [gap["code"] for gap in record["ng_model_call_capture"]["gaps"]] == ["agent_observation_join_failed"]


def test_merge_capture_reports_missing_capture(tmp_path):
    from nemo_gym.base_responses_api_model import CaptureStore
    from nemo_gym.observability.model_calls import merge_model_call_capture_into_record

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
    from nemo_gym.observability.model_calls import merge_model_call_capture_into_record

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
    from nemo_gym.base_responses_api_model import CaptureStore
    from nemo_gym.observability.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    store.record("4-2", _capture_exchange("responses", "A", {}, {"id": "resp-A"}))
    store.mark_incomplete("4-2")
    record = {"_ng_task_index": 4, "_ng_rollout_index": 2}

    merge_model_call_capture_into_record(record, [tmp_path])

    assert len(record["ng_model_call_capture"]["calls"]) == 1
    assert [gap["code"] for gap in record["ng_model_call_capture"]["gaps"]] == ["model_call_capture_incomplete"]


def test_clear_model_call_captures_for_rollouts_run_scoping(tmp_path, monkeypatch):
    import nemo_gym.observability.model_calls as obs
    from nemo_gym.observability.model_calls import clear_model_call_captures_for_rollouts

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


def test_aggregate_model_call_records_sums_and_counts():
    from nemo_gym.observability.model_calls import ModelCallRecord, aggregate_model_call_records

    calls = [
        ModelCallRecord(
            call_index=0, tokens_in=10, tokens_out=5, tokens_total=15, cached_tokens=4, latency_total_ms=2.0
        ),
        ModelCallRecord(
            call_index=1, tokens_in=20, tokens_out=3, tokens_total=23, cached_tokens=0, latency_total_ms=1.0
        ),
    ]
    agg = aggregate_model_call_records(calls)
    assert (agg["tokens_in"], agg["tokens_out"], agg["tokens_total"]) == (30, 8, 38)
    assert agg["cached_tokens"] == 4
    assert agg["latency_total_ms"] == 3.0 and agg["num_calls"] == 2
    # empty -> all-None totals but a well-formed shape (num_calls 0)
    assert aggregate_model_call_records([]) == {
        "tokens_in": None,
        "tokens_out": None,
        "tokens_reasoning": None,
        "tokens_total": None,
        "cached_tokens": None,
        "latency_total_ms": None,
        "num_calls": 0,
    }


def test_extract_token_stats_anthropic_cache_fold():
    from nemo_gym.observability.model_calls import extract_token_stats

    # Anthropic native: input_tokens is the uncached remainder; cache-read + cache-creation fold in.
    stats = extract_token_stats(
        {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 30,
        }
    )
    assert stats["tokens_in"] == 330  # 100 + 200 + 30
    assert stats["tokens_out"] == 50
    assert stats["tokens_total"] == 380  # 330 + 50 (Anthropic sends no total_tokens)
    assert stats["cache_creation_tokens"] == 30


def test_extract_token_stats_anthropic_fully_cached_zero_base():
    from nemo_gym.observability.model_calls import extract_token_stats

    # A fully-cached Anthropic response omits input_tokens; a 0 base preserves the folded prompt
    # size instead of leaving tokens_in null.
    stats = extract_token_stats(
        {
            "output_tokens": 12,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 0,
        }
    )
    assert stats["tokens_in"] == 500  # 0 base + cache_read 500 + cache_creation 0
    assert stats["tokens_out"] == 12
    assert stats["cache_creation_tokens"] == 0
    empty = extract_token_stats({"output_tokens": 12, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})
    assert (empty["tokens_in"], empty["tokens_total"], empty["cache_creation_tokens"]) == (None, None, 0)


# One row per usage shape: (usage, tokens_in, tokens_total, cache_creation_tokens).
_USAGE_SHAPES = {
    "responses": ({"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}, 100, 120, None),
    "responses cached subset": (
        {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 80}, "output_tokens": 20},
        100,
        120,
        None,
    ),
    "chat completions": ({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}, 100, 120, None),
    "chat completions cached subset": (
        {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 80}, "completion_tokens": 20},
        100,
        120,
        None,
    ),
    "anthropic messages with cache": (
        {"input_tokens": 20, "cache_read_input_tokens": 70, "cache_creation_input_tokens": 10, "output_tokens": 20},
        100,
        120,
        10,
    ),
    "anthropic messages fully cached": ({"cache_read_input_tokens": 100, "output_tokens": 20}, 100, 120, None),
    "anthropic messages zero cache, no input count": (
        {"cache_read_input_tokens": 0, "output_tokens": 20},
        None,
        None,
        None,
    ),
    # NVIDIA-NeMo/Gym#4721: a gateway adds Anthropic-style cache fields to Chat Completions usage.
    "mixed, cache read": (
        {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cache_read_input_tokens": 80},
        100,
        120,
        None,
    ),
    "mixed, cache write": (
        {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cache_creation_input_tokens": 80},
        100,
        120,
        80,
    ),
    "mixed, prompt and input counts": (
        {"prompt_tokens": 100, "input_tokens": 20, "cache_read_input_tokens": 80, "completion_tokens": 20},
        100,
        120,
        None,
    ),
    "anthropic messages, nothing cached": (
        {"input_tokens": 100, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "output_tokens": 20},
        100,
        120,
        0,
    ),
    # A count that is not a non-negative integer is ignored rather than trusted.
    "malformed prompt count": (
        {"prompt_tokens": "100", "input_tokens": 20, "cache_read_input_tokens": 80, "output_tokens": 20},
        100,
        120,
        None,
    ),
    "no usage": ({}, None, None, None),
    "usage is not a mapping": (None, None, None, None),
}


@pytest.mark.parametrize(
    ("usage", "tokens_in", "tokens_total", "cache_creation"), _USAGE_SHAPES.values(), ids=_USAGE_SHAPES
)
def test_extract_token_stats_counts_the_cache_once_in_every_usage_shape(
    usage, tokens_in, tokens_total, cache_creation
):
    from nemo_gym.observability.model_calls import extract_token_stats

    stats = extract_token_stats(usage)

    assert (stats["tokens_in"], stats["tokens_total"], stats["cache_creation_tokens"]) == (
        tokens_in,
        tokens_total,
        cache_creation,
    )


def test_extract_token_stats_openai_cached_not_double_counted():
    from nemo_gym.observability.model_calls import extract_token_stats

    # OpenAI: input_tokens already includes cached tokens (cached_tokens is a subset) -> no fold.
    stats = extract_token_stats(
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "prompt_tokens_details": {"cached_tokens": 4}}
    )
    assert stats["tokens_in"] == 10
    assert stats["tokens_total"] == 15
    assert stats["cache_creation_tokens"] is None


@pytest.mark.asyncio
async def test_await_quiescence_waits_for_a_detached_write_to_land(tmp_path):
    import asyncio

    from nemo_gym.observability.model_calls import await_model_call_capture_quiescence

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
    from nemo_gym.observability.model_calls import await_model_call_capture_quiescence

    CaptureStore(tmp_path).mark_pending("3-0", "call-1")

    assert await await_model_call_capture_quiescence("3-0", [tmp_path], timeout_s=0.05, poll_s=0.01) is False


def test_merge_capture_flags_a_claim_that_never_landed(tmp_path):
    from nemo_gym.observability.model_calls import merge_model_call_capture_into_record

    store = CaptureStore(tmp_path)
    store.record("4-0", _capture_exchange("responses", "A", {}, {"id": "resp-A"}))
    store.mark_pending("4-0", "call-2")

    record = {"_ng_task_index": 4, "_ng_rollout_index": 0}
    merge_model_call_capture_into_record(record, [tmp_path])

    assert "model_call_capture_pending" in [gap["code"] for gap in record["ng_model_call_capture"]["gaps"]]


def test_clear_captures_for_rollouts_drops_stale_pending_claims(tmp_path):
    from nemo_gym.observability.model_calls import clear_model_call_captures_for_rollouts

    store = CaptureStore(tmp_path)
    store.mark_pending("5-0", "call-1")

    clear_model_call_captures_for_rollouts([{"_ng_task_index": 5, "_ng_rollout_index": 0}], [tmp_path])

    assert store.has_pending("5-0") is False
