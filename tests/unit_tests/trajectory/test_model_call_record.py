# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.trajectory.model_call_record import build_model_call_record


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


def test_as_arguments():
    from nemo_gym.trajectory.model_call_record import _as_arguments

    assert _as_arguments({"x": 1}) == {"x": 1}
    assert _as_arguments('{"y": 2}') == {"y": 2}
    assert _as_arguments("not json") == {"_raw": "not json"}
    assert _as_arguments(123) == {}


def test_cache_signal():
    from nemo_gym.trajectory.model_call_record import _cache_signal

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
    from nemo_gym.trajectory.model_call_record import extract_token_stats

    assert extract_token_stats(None)["tokens_total"] is None
    stats = extract_token_stats(
        {"prompt_tokens": 5, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 1}}
    )
    assert (stats["tokens_in"], stats["tokens_out"], stats["tokens_total"], stats["tokens_reasoning"]) == (5, 3, 8, 1)


def test_tool_calls_and_reasoning_chat_and_anthropic():
    from nemo_gym.trajectory.model_call_record import _tool_calls_and_reasoning

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
    from nemo_gym.trajectory.model_call_record import _tool_calls_and_reasoning

    # vLLM / newer servers emit `reasoning` (no `reasoning_content`).
    assert _tool_calls_and_reasoning({"choices": [{"message": {"content": "hi", "reasoning": "because"}}]}) == (
        [],
        "because",
    )
    # `reasoning_content` wins when both are present.
    both = {"choices": [{"message": {"reasoning_content": "rc", "reasoning": "r"}}]}
    assert _tool_calls_and_reasoning(both) == ([], "rc")


def test_tool_calls_and_reasoning_skips_non_dict_output_items():
    from nemo_gym.trajectory.model_call_record import _tool_calls_and_reasoning

    # A Responses `output` list may contain non-dict junk; it must be skipped, not crash.
    resp = {"output": [None, "junk", {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"}]}
    assert _tool_calls_and_reasoning(resp) == ([{"call_id": "c", "name": "f", "arguments": {}}], None)
    # Same guard on the Anthropic content blocks.
    anth = {"content": [None, "junk", {"type": "tool_use", "id": "t", "name": "g", "input": {"b": 2}}]}
    assert _tool_calls_and_reasoning(anth) == ([{"call_id": "t", "name": "g", "arguments": {"b": 2}}], None)


def test_tool_calls_and_reasoning_records_custom_tool_calls():
    from nemo_gym.trajectory.model_call_record import _tool_calls_and_reasoning

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


def test_aggregate_model_call_records_sums_and_counts():
    from nemo_gym.trajectory.model_call_record import ModelCallRecord, aggregate_model_call_records

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
    from nemo_gym.trajectory.model_call_record import extract_token_stats

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
    from nemo_gym.trajectory.model_call_record import extract_token_stats

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
    from nemo_gym.trajectory.model_call_record import extract_token_stats

    stats = extract_token_stats(usage)

    assert (stats["tokens_in"], stats["tokens_total"], stats["cache_creation_tokens"]) == (
        tokens_in,
        tokens_total,
        cache_creation,
    )


def test_extract_token_stats_openai_cached_not_double_counted():
    from nemo_gym.trajectory.model_call_record import extract_token_stats

    # OpenAI: input_tokens already includes cached tokens (cached_tokens is a subset) -> no fold.
    stats = extract_token_stats(
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "prompt_tokens_details": {"cached_tokens": 4}}
    )
    assert stats["tokens_in"] == 10
    assert stats["tokens_total"] == 15
    assert stats["cache_creation_tokens"] is None
