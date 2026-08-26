# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

from nemo_gym.atif_provider_reconciliation import (
    ProviderReconciliationError,
    ProviderReconciliationResult,
    ReconciledTokenStats,
    reconcile_provider_model_call,
)
from nemo_gym.relay_atif import AtifStep
from nemo_gym.rollout_observability import TrajectoryModelCall


def _step(
    *,
    message: Any = "answer",
    reasoning: str | None = "reasoning",
    tool_calls: list[dict[str, Any]] | None = None,
) -> AtifStep:
    return AtifStep.model_validate(
        {
            "step_id": 1,
            "source": "agent",
            "message": message,
            "reasoning_content": reasoning,
            "tool_calls": tool_calls,
        }
    )


def _tool_step() -> AtifStep:
    return _step(
        message="",
        tool_calls=[
            {
                "tool_call_id": "call-1",
                "function_name": "lookup",
                "arguments": {"query": "relay"},
            }
        ],
    )


def _response(dialect: str, *, tool_step: bool = False) -> dict[str, Any]:
    if dialect == "responses":
        output: list[dict[str, Any]] = [
            {
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": "reasoning"}],
            }
        ]
        if tool_step:
            output.append(
                {
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": '{"query":"relay"}',
                }
            )
        else:
            output.append(
                {
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                }
            )
        return {"status": "completed", "output": output}
    if dialect == "chat":
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None if tool_step else "answer",
            "reasoning_content": "reasoning",
        }
        if tool_step:
            message["tool_calls"] = [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query":"relay"}'},
                }
            ]
        return {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_step else "stop",
                    "message": message,
                }
            ]
        }
    assert dialect == "messages"
    content: list[dict[str, Any]] = [{"type": "thinking", "thinking": "reasoning"}]
    if tool_step:
        content.append(
            {
                "type": "tool_use",
                "id": "call-1",
                "name": "lookup",
                "input": {"query": "relay"},
            }
        )
    else:
        content.append({"type": "text", "text": "answer"})
    return {
        "type": "message",
        "role": "assistant",
        "stop_reason": "tool_use" if tool_step else "end_turn",
        "content": content,
    }


def _call(
    dialect: str | None,
    response: Any,
    *,
    finish_reason: str | None = None,
    response_status: str | None = None,
    token_stats: dict[str, int] | None = None,
) -> TrajectoryModelCall:
    return TrajectoryModelCall.model_validate(
        {
            "response": response,
            "response_metadata": {
                "dialect": dialect,
                "finish_reason": finish_reason,
                "response_status": response_status,
            },
            "token_stats": token_stats or {},
        }
    )


def _reconcile(
    dialect: str,
    response: dict[str, Any],
    *,
    step: AtifStep | None = None,
) -> ProviderReconciliationResult:
    reconciled_step = step or _step()
    has_tool_calls = bool(reconciled_step.tool_calls)
    finish_reason = {
        "responses": None,
        "chat": "tool_calls" if has_tool_calls else "stop",
        "messages": "tool_use" if has_tool_calls else "end_turn",
    }[dialect]
    return reconcile_provider_model_call(
        _call(
            dialect,
            response,
            finish_reason=finish_reason,
            response_status="completed" if dialect == "responses" else None,
        ),
        reconciled_step,
        path="turn",
    )


def _assert_rejected(
    dialect: str,
    response: dict[str, Any],
    message: str,
    *,
    step: AtifStep | None = None,
) -> None:
    with pytest.raises(ProviderReconciliationError, match=message):
        _reconcile(dialect, response, step=step)


def test_reconcile_accepts_missing_provider_response_and_returns_normalized_values() -> None:
    call = TrajectoryModelCall.model_validate(
        {
            "response": None,
            "response_metadata": {
                "dialect": "responses",
                "model": "model-a",
                "response_id": "response-a",
                "finish_reason": "stop",
            },
            "token_stats": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    )

    result = reconcile_provider_model_call(call, _step(), path="turn")

    assert result == ProviderReconciliationResult(
        finish_reason="stop",
        model_name="model-a",
        response_id="response-a",
        token_stats=ReconciledTokenStats(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


def test_reconcile_treats_chat_stream_null_model_as_missing_provider_evidence() -> None:
    response = _response("chat")
    response["model"] = None
    call = TrajectoryModelCall.model_validate(
        {
            "response": response,
            "response_metadata": {
                "dialect": "chat",
                "model": "requested-model",
                "response_id": "response-a",
                "finish_reason": "stop",
            },
            "token_stats": {},
        }
    )

    result = reconcile_provider_model_call(call, _step(), path="turn")

    assert result.model_name == "requested-model"


def test_reconcile_error_is_typed_and_keeps_the_caller_path() -> None:
    response = _response("responses")
    response["output_text"] = 7

    with pytest.raises(ProviderReconciliationError, match=r"turn\.response\.output_text: expected a string"):
        _reconcile("responses", response)


def test_reconcile_rejects_openai_total_that_disagrees_with_components() -> None:
    response = _response("responses")
    response["usage"] = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 6}

    _assert_rejected("responses", response, "total_tokens does not match")


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("call_id", None, "non-empty tool call ID"),
        ("call_id", " ", "non-empty tool call ID"),
        ("name", None, "non-empty tool name"),
        ("name", "", "non-empty tool name"),
        ("arguments", {"query": "relay"}, "JSON object string"),
        ("arguments", "not-json", "JSON object string"),
        ("arguments", "[]", "expected a JSON object"),
    ],
)
def test_reconcile_rejects_malformed_responses_function_calls(
    field_name: str,
    value: Any,
    message: str,
) -> None:
    response = _response("responses", tool_step=True)
    response["output"][1][field_name] = value

    _assert_rejected("responses", response, message, step=_tool_step())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda item: item.__setitem__("summary", {}), "summary: expected an array"),
        (lambda item: item.__setitem__("summary", ["bad"]), "expected a 'summary_text' object"),
        (
            lambda item: item.__setitem__("summary", [{"type": "summary_text", "text": 7}]),
            "summary\\[0\\]\\.text: expected a string",
        ),
        (lambda item: item.__setitem__("encrypted_content", "opaque"), "encrypted Responses reasoning"),
        (
            lambda item: item.__setitem__("content", [{"type": "reasoning_text", "text": "reasoning"}]),
            "contains both summary and content",
        ),
        (
            lambda item: item.__setitem__(
                "summary",
                [
                    {"type": "summary_text", "text": "reasoning"},
                    {"type": "summary_text", "text": "second"},
                ],
            ),
            "multiple Responses reasoning segments",
        ),
    ],
)
def test_reconcile_rejects_unrepresentable_responses_reasoning(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    response = _response("responses")
    mutate(response["output"][0])

    _assert_rejected("responses", response, message)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda response: response["output"].__setitem__(1, "bad"), "expected a Responses output object"),
        (lambda response: response["output"][1].__setitem__("role", "user"), "expected 'assistant'"),
        (lambda response: response["output"][1].__setitem__("content", {}), "expected an output content array"),
        (
            lambda response: response["output"][1]["content"].__setitem__(0, {"type": "image"}),
            "expected an 'output_text' object",
        ),
        (lambda response: response["output"][1]["content"][0].__setitem__("text", 7), "expected a string"),
        (lambda response: response["output"][1].__setitem__("type", "unknown"), "unsupported Responses output type"),
        (
            lambda response: response["output"].insert(
                1,
                {
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": "reasoning"}],
                },
            ),
            "multiple Responses reasoning items",
        ),
        (lambda response: response.__setitem__("output_text", "different"), "does not match the structured"),
    ],
)
def test_reconcile_rejects_malformed_responses_output(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    response = _response("responses")
    mutate(response)

    _assert_rejected("responses", response, message)


@pytest.mark.parametrize("text", (None, 7, True, []))
def test_reconcile_rejects_non_string_responses_output_text_item(text: Any) -> None:
    response = {"status": "completed", "output": [{"type": "output_text", "text": text}]}

    _assert_rejected("responses", response, "expected a string", step=_step(reasoning=None))


def test_reconcile_accepts_a_responses_output_text_item() -> None:
    response = {"status": "completed", "output": [{"type": "output_text", "text": "answer"}]}

    result = _reconcile("responses", response, step=_step(reasoning=None))

    assert result.finish_reason is None


def test_reconcile_accepts_bare_responses_output_text_for_one_scalar_message() -> None:
    response = {"status": "completed", "output_text": "answer"}

    result = _reconcile("responses", response, step=_step(reasoning=None))

    assert result.finish_reason is None


def test_reconcile_rejects_flattened_output_text_for_multipart_atif() -> None:
    response = {"status": "completed", "output_text": "firstsecond"}
    step = _step(
        message=[
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ],
        reasoning=None,
    )

    _assert_rejected("responses", response, "cannot prove multiple ATIF message-part boundaries", step=step)


def test_reconcile_accepts_a_chat_response_without_provider_output_evidence() -> None:
    result = _reconcile("chat", {})

    assert result.finish_reason == "stop"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda choice: choice.__setitem__("logprobs", []), "logprobs: expected an object"),
        (lambda choice: choice["message"].__setitem__("role", "user"), "expected 'assistant'"),
        (lambda choice: choice["message"].__setitem__("content", 7), "unsupported Chat content"),
        (
            lambda choice: choice["message"].__setitem__("content", [{"type": "image"}]),
            "expected a text object",
        ),
        (
            lambda choice: choice["message"].__setitem__("content", [{"type": "text", "text": 7}]),
            "expected a string",
        ),
        (lambda choice: choice["message"].__setitem__("reasoning_content", 7), "reasoning must be scalar text"),
        (
            lambda choice: choice["message"].__setitem__("reasoning", "different"),
            "reasoning aliases conflict",
        ),
        (lambda choice: choice["message"].__setitem__("function_call", {}), "legacy Chat function_call"),
        (lambda choice: choice["message"].__setitem__("tool_calls", {}), "tool_calls: expected an array"),
    ],
)
def test_reconcile_rejects_malformed_chat_output(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    response = _response("chat")
    mutate(response["choices"][0])

    _assert_rejected("chat", response, message)


def test_reconcile_accepts_chat_multipart_text_content() -> None:
    response = _response("chat")
    response["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second"},
    ]
    step = _step(
        message=[
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
    )

    _reconcile("chat", response, step=step)


@pytest.mark.parametrize(
    ("tool_call", "message"),
    [
        ("bad", "expected a function tool call object"),
        ({"type": "function"}, "expected a function tool call object"),
        (
            {
                "id": "call-1",
                "type": "custom",
                "function": {"name": "lookup", "arguments": '{"query":"relay"}'},
            },
            "expected 'function'",
        ),
    ],
)
def test_reconcile_rejects_malformed_chat_tool_calls(tool_call: Any, message: str) -> None:
    response = _response("chat", tool_step=True)
    response["choices"][0]["message"]["tool_calls"] = [tool_call]

    _assert_rejected("chat", response, message, step=_tool_step())


@pytest.mark.parametrize("declared_index", [True, 0.0, "0", 1])
def test_reconcile_rejects_inconsistent_chat_tool_call_index(declared_index: Any) -> None:
    response = _response("chat", tool_step=True)
    response["choices"][0]["message"]["tool_calls"][0]["index"] = declared_index

    _assert_rejected("chat", response, "expected 0 when present", step=_tool_step())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda response: response.__setitem__("type", "response"), "expected 'message'"),
        (lambda response: response.__setitem__("role", "user"), "expected 'assistant'"),
        (lambda response: response["content"].__setitem__(1, "bad"), "expected an Anthropic content block"),
        (lambda response: response["content"][1].__setitem__("text", 7), "expected a string"),
        (lambda response: response["content"][0].__setitem__("thinking", 7), "expected a string"),
        (lambda response: response["content"][0].__setitem__("type", "redacted_thinking"), "redacted Anthropic"),
        (lambda response: response["content"][1].__setitem__("type", "unknown"), "unsupported Anthropic"),
    ],
)
def test_reconcile_rejects_malformed_anthropic_output(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    response = _response("messages")
    mutate(response)

    _assert_rejected("messages", response, message)


def test_reconcile_rejects_anthropic_thinking_after_text() -> None:
    response = _response("messages")
    response["content"].append(response["content"].pop(0))

    _assert_rejected("messages", response, "thinking cannot follow")


def test_reconcile_rejects_anthropic_tool_input_that_is_not_an_object() -> None:
    response = _response("messages", tool_step=True)
    response["content"][1]["input"] = []

    _assert_rejected("messages", response, "expected a JSON object", step=_tool_step())


def test_reconcile_rejects_unknown_anthropic_tool_caller_shape() -> None:
    response = _response("messages", tool_step=True)
    response["content"][1]["caller"] = {"type": "direct", "name": "unexpected"}

    _assert_rejected("messages", response, "server-tool callers", step=_tool_step())


def test_reconcile_rejects_multiple_anthropic_thinking_blocks() -> None:
    response = _response("messages")
    response["content"].insert(1, copy.deepcopy(response["content"][0]))

    _assert_rejected("messages", response, "multiple Anthropic thinking blocks")
