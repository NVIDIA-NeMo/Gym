# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError

from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming
from responses_api_models.vllm_model.app import VLLMModel
from responses_api_models.vllm_model.streaming import streamed_chat_completion
from responses_api_models.vllm_model.tests.test_app import TestApp as _TestApp


def chunk(delta, finish=None, usage=None):
    return {
        "id": "synthetic",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "synthetic",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        "usage": usage,
    }


def client_for(events, done=True):
    data = (
        b": heartbeat\r\n\r\n"
        + b"".join(b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\r\n\r\n" for event in events)
        + (b"data: [DONE]\r\n\r\n" if done else b"")
    )

    async def blocks():
        # HTTP boundaries need not coincide with SSE lines or UTF-8 code points.
        for start in range(0, len(data), 7):
            yield data[start : start + 7]

    response = SimpleNamespace(content=SimpleNamespace(iter_any=blocks), release=MagicMock())
    client = SimpleNamespace(
        base_url="http://synthetic/v1", _request=AsyncMock(return_value=response), _raise_for_status=AsyncMock()
    )
    return client, response


def complete_events():
    return [
        chunk({"role": "assistant", "reasoning_content": "Plan é"}),
        chunk({"reasoning_content": " then act", "content": "hello "}),
        chunk(
            {
                "content": "world",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call-0",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"command":"printf '},
                    }
                ],
            }
        ),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'é"}'}}]}, "tool_calls"),
        {
            **chunk({}),
            "choices": [],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 12,
                "total_tokens": 17,
                "completion_tokens_details": {"reasoning_tokens": 10},
            },
        },
    ]


async def test_stream_preserves_reasoning_tools_usage_and_unicode():
    client, response = client_for(complete_events())
    result = await streamed_chat_completion(client, {"stream": True})
    choice = result["choices"][0]
    assert choice["message"]["reasoning_content"] == "Plan é then act"
    assert choice["message"]["content"] == "hello world"
    tool = choice["message"]["tool_calls"][0]
    assert tool["id"] == "call-0" and tool["function"]["arguments"] == '{"command":"printf é"}'
    assert "parsed_arguments" not in tool["function"]
    assert choice["finish_reason"] == "tool_calls"
    assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 10
    response.release.assert_called_once()


@pytest.mark.parametrize("case", ["disconnect", "missing_finish", "missing_usage", "provider_error"])
async def test_incomplete_stream_is_never_a_success(case):
    events = complete_events()
    if case == "missing_finish":
        events[3]["choices"][0]["finish_reason"] = None
    elif case == "missing_usage":
        events.pop()
    elif case == "provider_error":
        events = [{"error": {"message": "synthetic provider failure"}}]
    client, response = client_for(events, done=case != "disconnect")
    with pytest.raises((RuntimeError, ValueError)):
        await streamed_chat_completion(client, {})
    response.release.assert_called_once()


async def test_http_error_uses_existing_status_handler_and_releases():
    client, response = client_for([])
    error = ClientResponseError(MagicMock(), (), status=408, message="synthetic timeout")
    client._raise_for_status.side_effect = error
    with pytest.raises(ClientResponseError) as raised:
        await streamed_chat_completion(client, {})
    assert raised.value is error
    response.release.assert_called_once()


async def test_model_uses_stream_transport_without_changing_sampling(monkeypatch):
    server = _TestApp()._setup_server(monkeypatch)
    server.config.return_token_id_information = False
    server.config.uses_reasoning_parser = True
    server.config.stream_chat_completions = True
    server.config.sampling_overrides = {"max_tokens": 131072, "temperature": 1, "top_p": 0.95}
    client, _ = client_for(complete_events())
    monkeypatch.setattr(VLLMModel, "_resolve_client", lambda *_: client)
    body = NeMoGymChatCompletionCreateParamsNonStreaming(messages=[{"role": "user", "content": "synthetic"}])
    result = await server.chat_completions(MagicMock(), body)
    sent = client._request.call_args.kwargs["json"]
    assert sent["stream"] is True and sent["stream_options"]["include_usage"] is True
    assert sent["max_tokens"] == 131072 and sent["temperature"] == 1 and sent["top_p"] == 0.95
    assert result.usage.completion_tokens == 12
    assert result.choices[0].message.tool_calls[0].function.arguments == '{"command":"printf é"}'


@pytest.mark.parametrize("incompatible", ["return_token_id_information", "use_completions_api", "is_responses_native"])
def test_streaming_rejects_incompatible_protocols(monkeypatch, incompatible):
    server = _TestApp()._setup_server(monkeypatch)
    data = server.config.model_dump() | {"return_token_id_information": False, "stream_chat_completions": True}
    data[incompatible] = True
    with pytest.raises(ValueError, match="stream_chat_completions requires"):
        type(server.config).model_validate(data)
