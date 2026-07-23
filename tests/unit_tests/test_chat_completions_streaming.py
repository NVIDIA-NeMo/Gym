# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the streaming Chat Completions dialect on ``SimpleResponsesAPIModel``.

Every Gym model server's ``/v1/chat/completions`` accepts the wire dialect Chat-Completions
streaming harnesses (e.g. the OpenClaw agent PinchBench runs) speak: ``stream: true`` plus a
``stream_options`` block. The request is sanitized onto the strict params model and the complete
response is re-emitted as a synthesized ``chat.completion.chunk`` SSE stream. Non-streaming
requests keep the historical strict-validation behavior.
"""

import json
from time import time
from unittest.mock import MagicMock
from uuid import uuid4

import pydantic
import pytest
from fastapi import Body, Request
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    SimpleResponsesAPIModel,
    _parse_sse_events,
    _reconstruct_chat_sse,
)
from nemo_gym.chat_streaming import (
    sanitize_streaming_chat_body,
    synthesize_chat_completion_sse,
    validate_streaming_chat_params,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient


def _completion(
    *,
    content=None,
    tool_calls=None,
    reasoning=None,
    finish_reason="stop",
    usage=None,
    choices=None,
) -> NeMoGymChatCompletion:
    if choices is None:
        message = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        choices = [{"index": 0, "finish_reason": finish_reason, "message": message}]
    data = {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time()),
        "model": "downstream-model",
        "choices": choices,
    }
    if usage is not None:
        data["usage"] = usage
    return NeMoGymChatCompletion.model_validate(data)


_USAGE = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
_TOOL_CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"SF"}'}}


def _events(sse_text: str) -> list[dict]:
    """Parse the JSON ``data:`` payloads out of a chat SSE stream (excluding the ``[DONE]`` marker)."""
    events = []
    for block in sse_text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                payload = line[len("data: ") :]
                if payload == "[DONE]":
                    continue
                events.append(json.loads(payload))
    return events


class TestSanitizeStreamingChatBody:
    def test_drops_stream_and_stream_options(self) -> None:
        cleaned, include_usage = sanitize_streaming_chat_body(
            {"messages": [], "stream": True, "stream_options": {"include_usage": True}}
        )
        assert set(cleaned) == {"messages"}
        assert include_usage is True
        NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(cleaned)

    def test_include_usage_false_by_default(self) -> None:
        _, include_usage = sanitize_streaming_chat_body({"messages": [], "stream": True})
        assert include_usage is False

    def test_include_usage_false_when_flag_unset(self) -> None:
        _, include_usage = sanitize_streaming_chat_body(
            {"messages": [], "stream": True, "stream_options": {"include_usage": False}}
        )
        assert include_usage is False

    def test_drops_unknown_top_level_fields(self) -> None:
        cleaned, _ = sanitize_streaming_chat_body(
            {"messages": [], "stream": True, "client_bookkeeping": {"x": 1}, "temperature": 0.5}
        )
        assert set(cleaned) == {"messages", "temperature"}
        NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(cleaned)

    def test_keeps_known_sampling_and_tool_fields(self) -> None:
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 128,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
                }
            ],
        }
        cleaned, _ = sanitize_streaming_chat_body(body)
        params = NeMoGymChatCompletionCreateParamsNonStreaming.model_validate(cleaned)
        assert params.temperature == 0.7
        assert params.max_tokens == 128
        assert params.tools[0]["function"]["name"] == "get_weather"

    def test_does_not_mutate_caller_body(self) -> None:
        body = {"messages": [], "stream": True, "stream_options": {"include_usage": True}}
        sanitize_streaming_chat_body(body)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}


class TestValidateStreamingChatParams:
    def test_valid_body_passes(self) -> None:
        params = validate_streaming_chat_params({"messages": [{"role": "user", "content": "hi"}], "temperature": 1.0})
        assert params.temperature == 1.0

    def test_unfixable_errors_still_raise(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            validate_streaming_chat_params({"messages": [], "temperature": "not-a-number"})


class TestSynthesizeChatSSE:
    def test_text_event_sequence(self) -> None:
        completion = _completion(content="hello world", usage=_USAGE).model_dump(mode="json")
        text = "".join(synthesize_chat_completion_sse(completion))
        assert text.endswith("data: [DONE]\n\n")
        events = _events(text)
        # role delta first, terminal finish_reason last
        assert events[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert events[0]["object"] == "chat.completion.chunk"
        assert events[-1]["choices"][0]["finish_reason"] == "stop"
        assert events[-1]["choices"][0]["delta"] == {}
        # every chunk shares the completion identity
        assert {e["id"] for e in events} == {completion["id"]}

    def test_content_roundtrips_via_capture_reconstructor(self) -> None:
        completion = _completion(content="hello world", usage=_USAGE).model_dump(mode="json")
        text = "".join(synthesize_chat_completion_sse(completion))
        rebuilt = _reconstruct_chat_sse(_parse_sse_events(text.encode()))
        assert rebuilt["choices"][0]["message"]["content"] == "hello world"
        assert rebuilt["choices"][0]["finish_reason"] == "stop"

    def test_reasoning_delta_emitted(self) -> None:
        completion = _completion(content="answer", reasoning="let me think").model_dump(mode="json")
        text = "".join(synthesize_chat_completion_sse(completion))
        rebuilt = _reconstruct_chat_sse(_parse_sse_events(text.encode()))
        assert rebuilt["choices"][0]["message"]["reasoning_content"] == "let me think"
        assert rebuilt["choices"][0]["message"]["content"] == "answer"

    def test_tool_calls_roundtrip(self) -> None:
        completion = _completion(content=None, tool_calls=[_TOOL_CALL], finish_reason="tool_calls").model_dump(
            mode="json"
        )
        text = "".join(synthesize_chat_completion_sse(completion))
        rebuilt = _reconstruct_chat_sse(_parse_sse_events(text.encode()))
        tool_calls = rebuilt["choices"][0]["message"]["tool_calls"]
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "SF"}
        assert rebuilt["choices"][0]["finish_reason"] == "tool_calls"
        assert rebuilt["choices"][0]["message"]["content"] is None

    def test_no_content_chunk_when_content_empty(self) -> None:
        completion = _completion(content=None, tool_calls=[_TOOL_CALL], finish_reason="tool_calls").model_dump(
            mode="json"
        )
        events = _events("".join(synthesize_chat_completion_sse(completion)))
        assert all("content" not in e["choices"][0]["delta"] for e in events)

    def test_usage_chunk_emitted_only_when_requested(self) -> None:
        completion = _completion(content="hi", usage=_USAGE).model_dump(mode="json")

        without = _events("".join(synthesize_chat_completion_sse(completion, include_usage=False)))
        assert all(e.get("usage") is None for e in without)

        with_usage = _events("".join(synthesize_chat_completion_sse(completion, include_usage=True)))
        usage_chunks = [e for e in with_usage if e.get("usage") is not None]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["choices"] == []
        assert usage_chunks[0]["usage"]["total_tokens"] == 10

    def test_usage_chunk_skipped_when_usage_absent(self) -> None:
        completion = _completion(content="hi", usage=None).model_dump(mode="json")
        events = _events("".join(synthesize_chat_completion_sse(completion, include_usage=True)))
        assert all(e.get("usage") is None for e in events)

    def test_multiple_choices(self) -> None:
        choices = [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "a"}},
            {"index": 1, "finish_reason": "stop", "message": {"role": "assistant", "content": "b"}},
        ]
        completion = _completion(choices=choices).model_dump(mode="json")
        events = _events("".join(synthesize_chat_completion_sse(completion)))
        seen_indices = {e["choices"][0]["index"] for e in events}
        assert seen_indices == {0, 1}

    def test_empty_choices_still_terminates(self) -> None:
        completion = _completion(choices=[]).model_dump(mode="json")
        text = "".join(synthesize_chat_completion_sse(completion))
        assert text == "data: [DONE]\n\n"


class _EchoChatModel(SimpleResponsesAPIModel):
    """Fake model server capturing the params its chat_completions() receives and echoing input."""

    config: BaseResponsesAPIModelConfig
    last_params: object = None
    model_config = {"arbitrary_types_allowed": True}

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        object.__setattr__(self, "last_params", body)
        text = "hi"
        for message in body.messages:
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                text = message["content"]
        return _completion(content=text, usage=_USAGE)

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError


class _RequestAwareEchoChatModel(_EchoChatModel):
    saw_request: bool = False

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        object.__setattr__(self, "saw_request", isinstance(request, Request))
        return await super().chat_completions(body)


def _client(model_cls) -> tuple[TestClient, SimpleResponsesAPIModel]:
    server = model_cls(
        config=BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient, global_config_dict={}),
    )
    return TestClient(server.setup_webserver()), server


class TestChatDispatchRoute:
    def test_non_streaming_request_returns_plain_json(self) -> None:
        client, server = _client(_EchoChatModel)
        resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi there"}]})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["choices"][0]["message"]["content"] == "hi there"
        assert server.last_params.messages[0]["content"] == "hi there"

    def test_non_streaming_request_still_validates_strictly(self) -> None:
        client, _ = _client(_EchoChatModel)
        resp = client.post("/v1/chat/completions", json={"model": "x"})  # missing required messages
        assert resp.status_code == 422
        assert resp.json()["detail"][0]["loc"][0] == "body"

    def test_streaming_request_returns_synthesized_sse(self) -> None:
        client, server = _client(_EchoChatModel)
        resp = client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "hello"}]},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.text.endswith("data: [DONE]\n\n")
        # the server saw sanitized params (no stream flag reaches the strict model)
        assert server.last_params.stream is None
        rebuilt = _reconstruct_chat_sse(_parse_sse_events(resp.text.encode()))
        assert rebuilt["choices"][0]["message"]["content"] == "hello"

    def test_streaming_request_strips_bookkeeping_and_options(self) -> None:
        client, server = _client(_EchoChatModel)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "stream": True,
                "stream_options": {"include_usage": True},
                "client_bookkeeping": {"cli": "openclaw"},
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 200
        # include_usage propagated -> a usage chunk is present
        usage_chunks = [e for e in _events(resp.text) if e.get("usage") is not None]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["usage"]["total_tokens"] == 10

    def test_streaming_request_invalid_params_returns_422(self) -> None:
        client, _ = _client(_EchoChatModel)
        resp = client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}], "temperature": "not-a-number"},
        )
        assert resp.status_code == 422
        assert resp.json()["detail"][0]["loc"][0] == "body"

    def test_dispatch_handles_request_aware_signature(self) -> None:
        client, server = _client(_RequestAwareEchoChatModel)
        resp = client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert server.saw_request is True
