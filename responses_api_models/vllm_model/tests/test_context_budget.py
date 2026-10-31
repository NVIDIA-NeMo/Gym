# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact per-request completion budgets at the model transport boundary."""

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from pytest import mark, raises

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
from responses_api_models.vllm_model.tests.test_app import _engine_bad_request_error


def _config(**overrides):
    return VLLMModelConfig(
        **{
            "host": "127.0.0.1",
            "port": 8081,
            "base_url": "http://model.test/v1",
            "api_key": "unused",
            "model": "served-model",
            "entrypoint": "app.py",
            "name": "policy_model",
            "return_token_id_information": False,
            "uses_reasoning_parser": False,
            **overrides,
        }
    )


def _model(*, enabled=True, count=900, window=1000, **overrides):
    model = VLLMModel(
        config=_config(**({"fit_completion_to_context": True} if enabled else {}), **overrides),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    client = MagicMock(spec=NeMoGymAsyncOpenAI)
    client.create_tokenize = AsyncMock(return_value={"count": count, "max_model_len": window, "tokens": []})
    client.create_chat_completion = AsyncMock(
        return_value={
            "id": "completion-1",
            "object": "chat.completion",
            "created": 1,
            "model": "served-model",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": count, "completion_tokens": 1, "total_tokens": count + 1},
        }
    )
    model._clients = [client]
    return model, client


def _body(**overrides):
    return NeMoGymChatCompletionCreateParamsNonStreaming(
        **{"messages": [{"role": "user", "content": "Solve this."}], "max_completion_tokens": 1000, **overrides}
    )


def _request():
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "headers": [],
            "query_string": b"",
            "server": ("gym.test", 80),
            "session": {SESSION_ID_KEY: "context-budget-test"},
        }
    )


@mark.parametrize(
    ("count", "window", "requested", "expected"),
    [(900, 1000, 1000, 100), (10, 1000, 1000, 990), (900, 1000, 30, 30), (999, 1000, 1000, 1)],
)
async def test_fit_uses_all_remaining_context_without_exceeding_request_cap(count, window, requested, expected):
    model, client = _model(count=count, window=window)
    body = _body(max_completion_tokens=requested)

    result = await model.chat_completions(_request(), body)

    assert result.choices[0].message.content == "ok"
    assert client.create_chat_completion.await_args.kwargs["max_completion_tokens"] == expected
    assert body.max_completion_tokens == requested
    client.create_tokenize.assert_awaited_once()
    client.create_chat_completion.assert_awaited_once()


async def test_fit_is_opt_in_and_does_not_add_tokenization_to_existing_requests():
    model, client = _model(enabled=False)

    await model.chat_completions(_request(), _body())

    assert client.create_chat_completion.await_args.kwargs["max_completion_tokens"] == 1000
    client.create_tokenize.assert_not_awaited()


@mark.parametrize("budget", [{"max_tokens": 30}, {"max_tokens": 30, "max_completion_tokens": 80}, {}])
async def test_fit_honors_chat_token_parameter_precedence_and_handles_unspecified_budget(budget):
    model, client = _model()
    body = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "x"}], **budget)

    await model.chat_completions(_request(), body)

    sent = client.create_chat_completion.await_args.kwargs
    expected = 80 if "max_completion_tokens" in budget else 30 if "max_tokens" in budget else 100
    assert sent.get("max_completion_tokens", sent.get("max_tokens")) == expected


async def test_fit_measures_preprocessed_history_tools_and_template_on_the_generation_backend():
    model, client = _model(
        uses_reasoning_parser=True,
        reasoning_field="reasoning",
        replace_developer_role_with_system=True,
        chat_template_kwargs={"enable_thinking": True},
        forward_request_chat_template_kwargs=True,
        sampling_overrides={"reasoning_effort": "high"},
    )
    alternate = MagicMock(spec=NeMoGymAsyncOpenAI)
    alternate.create_tokenize = AsyncMock(side_effect=AssertionError("must use selected generation backend"))
    model._clients = [alternate, client]
    model._resolve_client = MagicMock(return_value=client)
    tools = [{"type": "function", "function": {"name": "finish", "parameters": {"type": "object"}}}]
    body = _body(
        messages=[
            {"role": "developer", "content": "Solve using tools."},
            {"role": "assistant", "content": "<think>retained reasoning</think>Run the calculation."},
            {"role": "user", "content": "The calculation returned 3."},
        ],
        tools=deepcopy(tools),
        chat_template_kwargs={"mode": "research"},
    )

    await model.chat_completions(_request(), body)

    measured = client.create_tokenize.await_args.kwargs
    generated = client.create_chat_completion.await_args.kwargs
    assert measured == {
        "model": "served-model",
        "messages": [
            {"role": "system", "content": "Solve using tools."},
            {"role": "assistant", "content": "Run the calculation.", "reasoning": "retained reasoning"},
            {"role": "user", "content": "The calculation returned 3."},
        ],
        "tools": tools,
        "chat_template_kwargs": {"enable_thinking": True, "mode": "research", "reasoning_effort": "high"},
        "reasoning_effort": "high",
    }
    assert all(generated[key] == value for key, value in measured.items() if key != "chat_template_kwargs")
    assert generated["chat_template_kwargs"] == {"enable_thinking": True, "mode": "research"}
    model._resolve_client.assert_called_once()
    alternate.create_tokenize.assert_not_awaited()


@mark.parametrize(
    "measurement",
    [
        None,
        {},
        {"count": True, "max_model_len": 1000},
        {"count": -1, "max_model_len": 1000},
        {"count": "900", "max_model_len": 1000},
        {"count": 900, "max_model_len": False},
        {"count": 900, "max_model_len": 0},
        {"count": 900, "max_model_len": 1000.0},
    ],
)
async def test_fit_rejects_malformed_measurement_without_generation(measurement):
    model, client = _model()
    client.create_tokenize.return_value = measurement

    with raises(HTTPException) as caught:
        await model.chat_completions(_request(), _body())

    assert caught.value.status_code == 502
    client.create_chat_completion.assert_not_awaited()


@mark.parametrize("count", [1000, 1001])
async def test_fit_reports_exhausted_prompt_without_generation_or_a_minimum_output_floor(count):
    model, client = _model(count=count)

    with raises(HTTPException) as caught:
        await model.chat_completions(_request(), _body())

    assert caught.value.status_code == 400
    assert caught.value.detail["code"] == "context_length_exceeded"
    client.create_chat_completion.assert_not_awaited()


@mark.parametrize(("minimum", "accepted"), [(0, True), (100, True), (101, False)])
async def test_fit_never_silently_reduces_required_minimum_generation_tokens(minimum, accepted):
    model, client = _model(extra_body={"min_tokens": minimum})
    if accepted:
        await model.chat_completions(_request(), _body())
        sent = client.create_chat_completion.await_args.kwargs
        assert (sent["max_completion_tokens"], sent["min_tokens"]) == (100, minimum)
    else:
        with raises(HTTPException) as caught:
            await model.chat_completions(_request(), _body())
        assert caught.value.status_code == 400
        assert caught.value.detail["code"] == "invalid_output_token_parameter"
        client.create_chat_completion.assert_not_awaited()


@mark.parametrize("stage", ["tokenize", "generation"])
@mark.parametrize("message", ["maximum context length exceeded", "unsupported tool schema"])
async def test_fit_propagates_provider_rejection_once_even_when_legacy_propagation_is_disabled(stage, message):
    model, client = _model(propagate_context_overflow_errors=False)
    error = _engine_bad_request_error(message)
    failed_call = client.create_tokenize if stage == "tokenize" else client.create_chat_completion
    failed_call.side_effect = error

    with raises(type(error)) as caught:
        await model.chat_completions(_request(), _body())

    assert caught.value is error
    failed_call.assert_awaited_once()
    if stage == "tokenize":
        client.create_chat_completion.assert_not_awaited()


def test_fit_preserves_provider_context_error_through_the_public_route():
    model, client = _model(propagate_context_overflow_errors=False)
    client.create_chat_completion.side_effect = _engine_bad_request_error("maximum context length exceeded")
    app = model.setup_webserver()
    model.setup_exception_middleware(app)

    response = TestClient(app).post("/v1/chat/completions", json=_body().model_dump(exclude_unset=True))

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"
    client.create_chat_completion.assert_awaited_once()


@mark.parametrize("unsupported", [{"use_completions_api": True}, {"is_responses_native": True}])
def test_fit_rejects_unsupported_backends_at_configuration(unsupported):
    with raises(ValueError, match="fit_completion_to_context"):
        _config(fit_completion_to_context=True, **unsupported)


@mark.parametrize(
    ("options", "rendered"),
    [
        ({"reasoning_effort": "high"}, "high|True|none"),
        ({"reasoning_effort": "none"}, "none|False|none"),
        (
            {
                "reasoning_effort": "high",
                "documents": [{"text": "primary"}],
                "chat_template_kwargs": {
                    "reasoning_effort": "low",
                    "enable_thinking": False,
                    "documents": [{"text": "fallback"}],
                },
            },
            "high|False|primary",
        ),
        (
            {
                "reasoning_effort": None,
                "documents": None,
                "chat_template_kwargs": {"reasoning_effort": "low", "documents": [{"text": "fallback"}]},
            },
            "low|False|fallback",
        ),
        ({"documents": [], "chat_template_kwargs": {"documents": [{"text": "fallback"}]}}, "default|False|none"),
    ],
)
async def test_fit_counts_effective_template_options_instead_of_tokenize_ignored_top_level_fields(options, rendered):
    # The tokenizer's template renderer only reads these options from chat_template_kwargs.
    # Chat generation also interprets top-level effort/documents; the fixtures give its expected rendering.
    provider_options = {key: value for key, value in options.items() if key == "documents"}
    request_options = {key: value for key, value in options.items() if key != "documents"}
    model, client = _model(forward_request_chat_template_kwargs=True, extra_body=provider_options)
    observed = []

    async def tokenize(**payload):
        template = payload.get("chat_template_kwargs") or {}
        documents = template.get("documents") or []
        text = (
            f"{template.get('reasoning_effort', 'default')}|{template.get('enable_thinking', False)}|"
            f"{documents[0]['text'] if documents else 'none'}"
        )
        observed.append(text)
        return {"count": len(text), "max_model_len": 1000, "tokens": list(text.encode())}

    client.create_tokenize.side_effect = tokenize
    body = _body(**request_options)
    original = body.model_dump(exclude_unset=True)

    await model.chat_completions(_request(), body)

    assert observed == [rendered]
    assert client.create_chat_completion.await_args.kwargs["max_completion_tokens"] == 1000 - len(rendered)
    assert body.model_dump(exclude_unset=True) == original


@mark.parametrize(
    "unsupported",
    [
        {"truncate_prompt_tokens": -1},
        {"truncate_prompt_tokens": 10},
        {"response_format": {"type": "json_object"}},
        {"tools": [{"type": "function", "function": {"name": "finish"}}], "tool_choice": "none"},
        {"tools": [{"type": "function", "function": {"name": "finish"}}], "tool_choice": "required"},
        {
            "tools": [{"type": "function", "function": {"name": "finish"}}],
            "tool_choice": {"type": "function", "function": {"name": "finish"}},
        },
    ],
)
async def test_fit_rejects_prompt_changes_the_tokenizer_cannot_measure(unsupported):
    model, client = _model(extra_body=unsupported)

    with raises(HTTPException) as caught:
        await model.chat_completions(_request(), _body())

    assert caught.value.status_code == 400
    client.create_tokenize.assert_not_awaited()
    client.create_chat_completion.assert_not_awaited()
