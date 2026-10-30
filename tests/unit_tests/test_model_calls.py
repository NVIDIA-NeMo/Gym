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

    record = build_model_call_record(exchange, call_index=0)

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

    record = build_model_call_record(_cancelled_exchange(orjson.dumps(body).decode(), dialect="chat"), call_index=0)

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
    record = build_model_call_record(_cancelled_exchange(response_raw, status_code=status_code), call_index=0)

    assert record.response is None
    assert record.response_raw == response_raw
    assert record.response_id is None
    assert (record.tokens_in, record.tokens_out) == (None, None)
    assert record.error_category == "cancelled"


def test_only_cancelled_calls_have_their_raw_body_parsed():
    exchange = _cancelled_exchange(orjson.dumps(_COMPLETE_RESPONSES_BODY).decode())
    exchange["error_category"] = "capture_parse_error"

    record = build_model_call_record(exchange, call_index=0)

    assert record.response is None
    assert record.response_raw == exchange["response_raw"]
    assert record.tokens_out is None


def test_a_parsed_response_is_never_replaced_by_its_raw_body():
    exchange = _cancelled_exchange('{"id": "from-raw"}')
    exchange["response"] = {"id": "parsed", "usage": {"input_tokens": 1, "output_tokens": 2}}
    exchange["error_category"] = None

    record = build_model_call_record(exchange, call_index=0)

    assert record.response_id == "parsed"
    assert record.response == exchange["response"]
    assert record.response_raw == '{"id": "from-raw"}'


def test_merge_capture_binds_a_cancelled_call_with_a_complete_body_to_its_agent_turn(tmp_path):
    from nemo_gym.observability.model_calls import merge_model_call_capture_into_record

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


def _covered_record():
    return {
        "ng_model_call_capture": {
            "rollout_id": "0-0",
            "metrics": {},
            "calls": [{"model_call_id": "c1"}, {"model_call_id": "c2"}],
        },
        "ng_trajectory": {
            "model_calls": [
                {"model_call_id": "c1", "request": {"input": "hi"}, "response": {"id": "r1"}},
                # A raw body kept as text still counts as a response.
                {"model_call_id": "c2", "request": {"input": "again"}, "response": '{"id": "r2"'},
            ],
            "gaps": [],
        },
    }


def _redact_payloads(record):
    record["ng_trajectory"]["gaps"].append({"code": "multimodal_history_redacted"})
    for call in record["ng_trajectory"]["model_calls"]:
        call["request"] = call["response"] = None


def test_uncovered_reasons_is_empty_when_every_call_has_its_payload():
    from nemo_gym.observability.model_calls import model_call_capture_uncovered_reasons

    assert model_call_capture_uncovered_reasons(_covered_record()) == []


@pytest.mark.parametrize(
    "mutate, expected",
    [
        pytest.param(lambda r: r.pop("ng_model_call_capture"), ["no_capture_attachment"], id="no capture attachment"),
        pytest.param(
            lambda r: r["ng_model_call_capture"].update(gaps=[{"code": "model_call_capture_pending"}]),
            ["capture_gap:model_call_capture_pending"],
            id="capture gap",
        ),
        pytest.param(lambda r: r.pop("ng_trajectory"), ["no_trajectory"], id="no trajectory"),
        pytest.param(
            lambda r: r["ng_trajectory"]["gaps"].append({"code": "multimodal_history_redacted"}),
            ["payload_gap:multimodal_history_redacted"],
            id="redacted history in the trajectory",
        ),
        pytest.param(
            lambda r: _redact_payloads(r),
            ["payload_gap:multimodal_history_redacted"],
            id="redacted history leaves no payloads, reported once",
        ),
        pytest.param(
            lambda r: r.update(ng_agent_observations={"gaps": [{"code": "multimodal_history_redacted"}]}),
            ["payload_gap:multimodal_history_redacted"],
            id="redacted history in the agent observations",
        ),
        pytest.param(
            lambda r: r["ng_trajectory"]["model_calls"].pop(),
            ["call_without_payload:c2"],
            id="call missing from the trajectory",
        ),
        pytest.param(
            lambda r: r["ng_trajectory"]["model_calls"][0].update(response=None),
            ["call_without_payload:c1"],
            id="call without a response",
        ),
        pytest.param(
            lambda r: r["ng_trajectory"]["model_calls"][1].update(request=None),
            ["call_without_payload:c2"],
            id="call without a request",
        ),
        pytest.param(
            lambda r: r["ng_model_call_capture"]["calls"].append({"model_call_id": None}),
            ["call_without_id"],
            id="call without an id",
        ),
    ],
)
def test_uncovered_reasons_names_what_the_rollout_lacks(mutate, expected):
    from nemo_gym.observability.model_calls import model_call_capture_uncovered_reasons

    record = _covered_record()
    mutate(record)

    assert model_call_capture_uncovered_reasons(record) == expected


def test_uncovered_reasons_covers_a_merged_capture_once_the_trajectory_is_attached(tmp_path):
    from nemo_gym.observability.model_calls import (
        merge_model_call_capture_into_record,
        model_call_capture_uncovered_reasons,
    )
    from nemo_gym.rollout_collection import _attach_trajectory_record

    row = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    store = CaptureStore(tmp_path)
    store.record("0-0", _cancelled_exchange(orjson.dumps(_COMPLETE_RESPONSES_BODY).decode()))
    record = dict(row)
    merge_model_call_capture_into_record(record, [tmp_path], include_payloads=True)
    _attach_trajectory_record(row, record)

    assert model_call_capture_uncovered_reasons(record) == []

    store.mark_pending("0-0", "call-late")
    pending = dict(row)
    merge_model_call_capture_into_record(pending, [tmp_path], include_payloads=True)
    _attach_trajectory_record(row, pending)

    assert model_call_capture_uncovered_reasons(pending) == ["capture_gap:model_call_capture_pending"]


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

    record = build_model_call_record(_cancelled_exchange(raw), call_index=0)

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

    record = build_model_call_record(_cancelled_exchange(raw, dialect="chat"), call_index=0)

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

    record = build_model_call_record(exchange, call_index=0)

    assert record.response is None
    assert record.response_raw == raw
    assert record.tokens_out is None


def _covered_capture(tmp_path, rollout_id="0-0"):
    """A capture file of one complete call, and a rollout that holds that call with its payload."""
    store = CaptureStore(tmp_path)
    exchange = _capture_exchange("responses", "A", {"input_tokens": 3, "output_tokens": 2}, {"id": "resp-A"})
    store.record(rollout_id, exchange)
    record = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    return store, record


def _merge_and_attach(tmp_path, record):
    """Collect the capture into the record the way rollout collection does."""
    from nemo_gym.observability.model_calls import merge_model_call_capture_into_record
    from nemo_gym.rollout_collection import _attach_trajectory_record

    merge_model_call_capture_into_record(record, [tmp_path], include_payloads=True)
    _attach_trajectory_record({"_ng_task_index": 0, "_ng_rollout_index": 0}, record)
    return record


def test_capture_store_size_is_none_without_a_file(tmp_path):
    store = CaptureStore(tmp_path)

    assert store.size("0-0") is None
    store.record("0-0", {"model_call_id": "c"})
    assert store.size("0-0") == store.path_for("0-0").stat().st_size > 0


def test_delete_if_unchanged_deletes_a_file_that_has_not_grown(tmp_path):
    store = CaptureStore(tmp_path)
    store.record("0-0", {"model_call_id": "c"})

    assert store.delete_if_unchanged("0-0", store.size("0-0")) is None
    assert not store.path_for("0-0").exists()


@pytest.mark.parametrize(
    "setup, reason",
    [
        pytest.param(lambda store: store.record("0-0", {"model_call_id": "late"}), "capture_file_changed", id="grown"),
        pytest.param(lambda store: store.mark_pending("0-0", "late"), "write_pending", id="pending write"),
    ],
)
def test_delete_if_unchanged_keeps_a_file_that_may_hold_unread_calls(tmp_path, setup, reason):
    store = CaptureStore(tmp_path)
    store.record("0-0", {"model_call_id": "c"})
    size = store.size("0-0")
    setup(store)

    assert store.delete_if_unchanged("0-0", size) == reason
    assert store.path_for("0-0").exists()


def test_pending_rollout_ids_lists_every_rollout_with_a_claimed_write(tmp_path):
    store = CaptureStore(tmp_path)
    store.record("9-0", {"model_call_id": "not pending"})
    store.mark_pending("0-0", "call-a")
    store.mark_pending("0-0", "call-b")
    store.mark_pending("1-0-a1", "call-c")
    # A rollout id may contain dots; the claim token after the last dot does not.
    store.mark_pending("step7.0-0", "call-d")

    assert store.pending_rollout_ids() == {"0-0", "1-0-a1", "step7.0-0"}

    store.clear_pending("0-0", "call-a")
    assert "0-0" in store.pending_rollout_ids()
    store.clear_pending("0-0", "call-b")
    assert "0-0" not in store.pending_rollout_ids()


def test_pending_rollout_ids_agrees_with_has_pending_for_rollouts_whose_ids_share_a_prefix(tmp_path):
    store = CaptureStore(tmp_path)
    store.mark_pending("1-0-a1", "call")

    pending = store.pending_rollout_ids()

    for rollout_id in ("1-0", "1-0-a1", "1-0-a11", "1", "2-0"):
        assert (rollout_id in pending) is store.has_pending(rollout_id)


def test_delete_if_unchanged_uses_the_pending_ids_it_is_given_instead_of_scanning(tmp_path, monkeypatch):
    store = CaptureStore(tmp_path)
    store.record("0-0", {"model_call_id": "c"})
    size = store.size("0-0")
    monkeypatch.setattr(CaptureStore, "has_pending", lambda self, rollout_id: pytest.fail("scanned the directory"))

    assert store.delete_if_unchanged("0-0", size, pending_rollout_ids={"0-0"}) == "write_pending"
    assert store.path_for("0-0").exists()
    assert store.delete_if_unchanged("0-0", size, pending_rollout_ids={"1-0"}) is None
    assert not store.path_for("0-0").exists()


def test_delete_if_unchanged_reports_a_missing_file(tmp_path):
    assert CaptureStore(tmp_path).delete_if_unchanged("0-0", 0) == "capture_file_missing"


def _retire(rollout_id, record, snapshot, **add_kwargs):
    """Queue one persisted rollout and flush, as rollout collection does; return the batch."""
    from nemo_gym.observability.model_calls import CaptureRetirementBatch

    batch = CaptureRetirementBatch()
    batch.add(rollout_id, record, snapshot, **add_kwargs)
    batch.flush(lambda: None)
    return batch


def test_retirement_deletes_a_capture_file_that_the_persisted_rollout_covers(tmp_path):
    from nemo_gym.observability.model_calls import snapshot_model_call_capture

    store, record = _covered_capture(tmp_path)
    snapshot = snapshot_model_call_capture("0-0", [tmp_path])
    _merge_and_attach(tmp_path, record)

    batch = _retire("0-0", record, snapshot)

    assert (batch.retired, dict(batch.kept)) == (1, {})
    assert not store.path_for("0-0").exists()


def test_retirement_judges_coverage_from_the_rollout_it_is_given(tmp_path):
    from nemo_gym.observability.model_calls import snapshot_model_call_capture

    store, record = _covered_capture(tmp_path)
    snapshot = snapshot_model_call_capture("0-0", [tmp_path])
    _merge_and_attach(tmp_path, record)
    record["ng_trajectory"]["gaps"].append({"code": "multimodal_history_redacted"})
    record["ng_trajectory"]["model_calls"][0]["request"] = None

    batch = _retire("0-0", record, snapshot)

    assert (batch.retired, dict(batch.kept)) == (0, {"payload_gap": 1})
    assert store.path_for("0-0").exists()


def test_retirement_keeps_a_capture_file_that_grew_after_the_snapshot(tmp_path):
    from nemo_gym.observability.model_calls import snapshot_model_call_capture

    store, record = _covered_capture(tmp_path)
    snapshot = snapshot_model_call_capture("0-0", [tmp_path])
    store.record("0-0", _capture_exchange("responses", "B", {}, {"id": "resp-B"}))
    _merge_and_attach(tmp_path, record)

    batch = _retire("0-0", record, snapshot)

    assert dict(batch.kept) == {"capture_file_changed": 1}
    assert store.path_for("0-0").exists()


def test_retirement_keeps_a_capture_file_that_a_rollout_shares_with_another(tmp_path):
    from nemo_gym.observability.model_calls import snapshot_model_call_capture

    store, record = _covered_capture(tmp_path)
    snapshot = snapshot_model_call_capture("0-0", [tmp_path])
    _merge_and_attach(tmp_path, record)

    batch = _retire("0-0", record, snapshot, shares_capture_key=True)

    assert dict(batch.kept) == {"rollout_id_shared": 1}
    assert store.path_for("0-0").exists()


def test_retirement_without_a_snapshot_has_no_file_to_delete():
    batch = _retire("0-0", {}, None)

    assert dict(batch.kept) == {"no_capture_attachment": 1, "no_capture_file": 1}


def test_retirement_only_touches_the_rollout_attempt_it_was_given(tmp_path):
    from nemo_gym.observability.model_calls import snapshot_model_call_capture

    store = CaptureStore(tmp_path)
    store.record("0-0", _capture_exchange("responses", "A", {}, {"id": "resp-A"}))
    store.record("0-0-a1", _capture_exchange("responses", "B", {}, {"id": "resp-B"}))
    snapshot = snapshot_model_call_capture("0-0-a1", [tmp_path])
    record = {
        "ng_model_call_capture": {"calls": [{"model_call_id": "call-B"}]},
        "ng_trajectory": {"model_calls": [{"model_call_id": "call-B", "request": {}, "response": {}}]},
    }

    batch = _retire("0-0-a1", record, snapshot)

    assert batch.retired == 1
    assert store.path_for("0-0").exists() and not store.path_for("0-0-a1").exists()


def test_a_capture_file_that_cannot_be_deleted_is_left_in_place_without_failing_the_run(tmp_path, monkeypatch):
    from nemo_gym.observability.model_calls import snapshot_model_call_capture

    store, record = _covered_capture(tmp_path)
    snapshot = snapshot_model_call_capture("0-0", [tmp_path])
    _merge_and_attach(tmp_path, record)

    def fail(self, rollout_id, expected_size, **kwargs):
        raise PermissionError("read-only capture directory")

    monkeypatch.setattr(CaptureStore, "delete_if_unchanged", fail)

    batch = _retire("0-0", record, snapshot)

    assert (batch.retired, dict(batch.kept)) == (0, {"delete_failed": 1})
    assert store.path_for("0-0").exists()


def test_retiring_covered_captures_is_opt_in_and_needs_observability():
    from nemo_gym.observability.model_calls import retire_covered_model_call_captures_from_config

    base = {"observability_enabled": True, "model_call_capture_dir": "/captures"}
    assert retire_covered_model_call_captures_from_config(base) is False
    assert retire_covered_model_call_captures_from_config({**base, "retire_covered_model_call_captures": True}) is True
    assert retire_covered_model_call_captures_from_config({"retire_covered_model_call_captures": True}) is False


def test_shared_capture_rollout_ids_names_keys_that_several_rows_map_to():
    from nemo_gym.observability.model_calls import shared_capture_rollout_ids

    rows = [
        {"_ng_task_index": 0, "_ng_rollout_index": 0},
        {"_ng_task_index": 0, "_ng_rollout_index": 0},
        {"_ng_task_index": 0, "_ng_rollout_index": 1},
        # A later attempt of the same rollout has its own capture key.
        {"_ng_task_index": 1, "_ng_rollout_index": 0},
        {"_ng_task_index": 1, "_ng_rollout_index": 0, "_ng_attempt_index": 1},
        {"_ng_rollout_id": "explicit-1"},
        {"_ng_rollout_id": "explicit-1"},
        # A row without a usable id cannot share one.
        {"_ng_rollout_id": "not valid!"},
        {"unrelated": True},
    ]

    assert shared_capture_rollout_ids(rows) == {"0-0", "explicit-1"}


_COVERED_ROLLOUT = {
    "ng_model_call_capture": {"calls": [{"model_call_id": "a"}]},
    "ng_trajectory": {"model_calls": [{"model_call_id": "a", "request": {}, "response": {}}]},
}


def test_retirement_batch_makes_rollouts_durable_before_deleting_any_capture_file(tmp_path):
    from nemo_gym.observability.model_calls import CaptureRetirementBatch, snapshot_model_call_capture

    store = CaptureStore(tmp_path)
    batch = CaptureRetirementBatch()
    for rollout_id in ("0-0", "1-0"):
        store.record(rollout_id, {"model_call_id": rollout_id})
        batch.add(rollout_id, _COVERED_ROLLOUT, snapshot_model_call_capture(rollout_id, [tmp_path]))
    files_present_when_durable = []

    batch.flush(lambda: files_present_when_durable.append([store.path_for(r).exists() for r in ("0-0", "1-0")]))

    assert files_present_when_durable == [[True, True]]
    assert not store.path_for("0-0").exists() and not store.path_for("1-0").exists()


def test_retirement_batch_deletes_nothing_when_making_rollouts_durable_fails(tmp_path):
    from nemo_gym.observability.model_calls import CaptureRetirementBatch, snapshot_model_call_capture

    store = CaptureStore(tmp_path)
    store.record("0-0", {"model_call_id": "0-0"})
    batch = CaptureRetirementBatch()
    batch.add("0-0", _COVERED_ROLLOUT, snapshot_model_call_capture("0-0", [tmp_path]))

    def fail():
        raise OSError("fsync failed")

    with pytest.raises(OSError, match="fsync failed"):
        batch.flush(fail)

    assert store.path_for("0-0").exists()
    assert batch.retired == 0


def test_retirement_batch_counts_a_rollout_once_under_each_distinct_kept_reason(tmp_path):
    from nemo_gym.observability.model_calls import CaptureRetirementBatch, snapshot_model_call_capture

    store = CaptureStore(tmp_path)
    store.record("0-0", {"model_call_id": "0-0"})
    # Two calls lack a payload and a write is still pending: three reasons, two distinct kinds.
    rollout = {
        "ng_model_call_capture": {
            "calls": [{"model_call_id": "a"}, {"model_call_id": "b"}],
            "gaps": [{"code": "model_call_capture_pending"}],
        },
        "ng_trajectory": {"model_calls": []},
    }
    batch = CaptureRetirementBatch()
    batch.add("0-0", rollout, snapshot_model_call_capture("0-0", [tmp_path]))

    batch.flush(lambda: None)

    assert batch.kept == {"call_without_payload": 1, "capture_gap": 1}
    assert batch.retired == 0
    assert store.path_for("0-0").exists()


def test_retirement_batch_does_not_make_anything_durable_when_nothing_is_queued():
    from nemo_gym.observability.model_calls import CaptureRetirementBatch

    calls = []

    CaptureRetirementBatch().flush(lambda: calls.append("durable"))

    assert calls == []


def test_retirement_batch_scans_for_pending_writes_once_per_directory_not_once_per_file(tmp_path, monkeypatch):
    from nemo_gym.observability.model_calls import CaptureRetirementBatch, snapshot_model_call_capture

    store = CaptureStore(tmp_path)
    batch = CaptureRetirementBatch()
    for rollout_id in ("0-0", "1-0", "2-0"):
        store.record(rollout_id, {"model_call_id": rollout_id})
        batch.add(rollout_id, _COVERED_ROLLOUT, snapshot_model_call_capture(rollout_id, [tmp_path]))
    scans = []
    original = CaptureStore.pending_rollout_ids
    monkeypatch.setattr(CaptureStore, "pending_rollout_ids", lambda self: scans.append(self._root) or original(self))

    batch.flush(lambda: None)

    assert scans == [tmp_path]
    assert batch.retired == 3


def test_retirement_batch_does_not_scan_when_every_file_is_kept_anyway(tmp_path, monkeypatch):
    from nemo_gym.observability.model_calls import CaptureRetirementBatch, snapshot_model_call_capture

    store = CaptureStore(tmp_path)
    store.record("0-0", {"model_call_id": "c"})
    batch = CaptureRetirementBatch()
    batch.add("0-0", {}, snapshot_model_call_capture("0-0", [tmp_path]))
    monkeypatch.setattr(CaptureStore, "pending_rollout_ids", lambda self: pytest.fail("scanned for nothing"))

    batch.flush(lambda: None)

    assert dict(batch.kept) == {"no_capture_attachment": 1}


def test_retirement_batch_keeps_only_the_file_whose_rollout_has_a_pending_write(tmp_path):
    from nemo_gym.observability.model_calls import CaptureRetirementBatch, snapshot_model_call_capture

    store = CaptureStore(tmp_path)
    batch = CaptureRetirementBatch()
    for rollout_id in ("0-0", "1-0", "2-0"):
        store.record(rollout_id, {"model_call_id": rollout_id})
        batch.add(rollout_id, _COVERED_ROLLOUT, snapshot_model_call_capture(rollout_id, [tmp_path]))
    store.mark_pending("1-0", "late-call")

    batch.flush(lambda: None)

    assert [store.path_for(r).exists() for r in ("0-0", "1-0", "2-0")] == [False, True, False]
    assert (batch.retired, dict(batch.kept)) == (2, {"write_pending": 1})


def test_retirement_batch_is_due_when_full_or_old(monkeypatch):
    from nemo_gym.observability import model_calls
    from nemo_gym.observability.model_calls import CaptureRetirementBatch

    now = [1000.0]
    monkeypatch.setattr(model_calls.time, "monotonic", lambda: now[0])
    batch = CaptureRetirementBatch(max_size=3, max_age_s=5.0)

    assert batch.due() is False, "nothing is waiting"
    batch.add("0-0", {}, None)
    assert batch.due() is False
    now[0] += 5.0
    assert batch.due() is True, "old enough"
    batch.flush(lambda: None)
    assert batch.due() is False
    for rollout_id in ("1-0", "2-0", "3-0"):
        batch.add(rollout_id, {}, None)
    assert batch.due() is True, "full"
