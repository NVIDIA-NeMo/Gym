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

import asyncio
import json
from time import time
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from aiohttp import ClientResponseError, ServerDisconnectedError
from fastapi import Body, FastAPI, Request
from fastapi.testclient import TestClient

from nemo_gym import base_responses_api_model
from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    SimpleResponsesAPIModel,
    _parse_sse_events,
    _reconstruct_chat_sse,
)
from nemo_gym.chat_streaming import (
    sanitize_streaming_chat_body,
    synthesize_chat_completion_sse,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    PermanentEndpointError,
)
from nemo_gym.server_utils import ServerClient


def _completion(
    *,
    content=None,
    tool_calls=None,
    reasoning=None,
    refusal=None,
    finish_reason="stop",
    usage=None,
    choices=None,
) -> NeMoGymChatCompletion:
    if choices is None:
        message = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        if refusal is not None:
            message["refusal"] = refusal
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

    @pytest.mark.parametrize("content", [None, "Additional information."])
    def test_refusal_delta_roundtrips(self, content) -> None:
        refusal = "I cannot help with that."
        completion = _completion(content=content, refusal=refusal).model_dump(mode="json")
        text = "".join(synthesize_chat_completion_sse(completion))
        events = _events(text)
        assert any(event["choices"][0]["delta"] == {"refusal": refusal} for event in events)
        assert events[-1]["choices"][0]["finish_reason"] == "stop"
        assert text.endswith("data: [DONE]\n\n")
        rebuilt = _reconstruct_chat_sse(_parse_sse_events(text.encode()))
        assert rebuilt["choices"][0]["message"]["refusal"] == refusal
        assert rebuilt["choices"][0]["message"]["content"] == content

    def test_reconstructor_joins_refusal_fragments(self) -> None:
        events = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"refusal": "I cannot "}}]},
            {"choices": [{"delta": {"refusal": "help with that."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        rebuilt = _reconstruct_chat_sse(events)
        assert rebuilt["choices"][0]["message"] == {
            "role": "assistant",
            "content": None,
            "refusal": "I cannot help with that.",
        }

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

    def test_non_streaming_request_does_not_forward_outer_tool_call_name(self) -> None:
        client, server = _client(_EchoChatModel)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "name": "get_weather",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            }
                        ],
                    }
                ]
            },
        )

        assert resp.status_code == 200
        forwarded = server.last_params.model_dump(exclude_unset=True)["messages"][0]["tool_calls"][0]
        assert "name" not in forwarded
        assert forwarded["function"]["name"] == "get_weather"

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

    @pytest.mark.parametrize("stream_value", ["false", "true", 1])
    def test_malformed_stream_value_stays_on_strict_path(self, stream_value) -> None:
        # Only a genuine boolean True streams; any other value is validated against the strict
        # model, where a non-``Literal[False]`` stream is rejected with a 422, as before.
        client, _ = _client(_EchoChatModel)
        resp = client.post(
            "/v1/chat/completions",
            json={"stream": stream_value, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 422
        assert resp.json()["detail"][0]["loc"][0] == "body"


def _legacy_app() -> TestClient:
    """A FastAPI app with the pre-PR direct typed-body binding, to compare 422 parity against."""
    app = FastAPI()

    async def chat_completions(body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()):
        return {"ok": True}

    app.post("/v1/chat/completions")(chat_completions)
    return TestClient(app)


_BAD_BODIES = [
    {"model": "x"},  # missing required messages
    {"messages": [], "temperature": "not-a-number"},  # wrong scalar type
    {"messages": "not-a-list"},  # messages wrong type
    {"messages": [{"role": "user"}]},  # user message missing required content
]


class Test422Parity:
    """The dispatch's 422 body must be byte-for-byte identical to the pre-PR typed-body binding."""

    @pytest.mark.parametrize("bad", _BAD_BODIES)
    def test_non_streaming_422_matches_legacy(self, bad) -> None:
        legacy, (client, _) = _legacy_app(), _client(_EchoChatModel)
        legacy_resp = legacy.post("/v1/chat/completions", json=bad)
        new_resp = client.post("/v1/chat/completions", json=bad)
        assert legacy_resp.status_code == 422 and new_resp.status_code == 422
        assert new_resp.json()["detail"] == legacy_resp.json()["detail"]

    @pytest.mark.parametrize("bad", _BAD_BODIES)
    def test_streaming_422_matches_legacy(self, bad) -> None:
        # The sanitized streaming body validates through the same path, so its 422 matches too
        # (stream:true is dropped by the sanitizer before validation).
        legacy, (client, _) = _legacy_app(), _client(_EchoChatModel)
        legacy_resp = legacy.post("/v1/chat/completions", json=bad)
        new_resp = client.post("/v1/chat/completions", json={**bad, "stream": True})
        assert legacy_resp.status_code == 422 and new_resp.status_code == 422
        assert new_resp.json()["detail"] == legacy_resp.json()["detail"]


class TestSynthesizeSystemFingerprint:
    def test_system_fingerprint_propagated_into_every_chunk(self) -> None:
        completion = _completion(content="hi", usage=_USAGE).model_dump(mode="json")
        completion["system_fingerprint"] = "fp_abc123"
        events = _events("".join(synthesize_chat_completion_sse(completion)))
        assert events
        assert all(event.get("system_fingerprint") == "fp_abc123" for event in events)


def _http_error(status: int) -> ClientResponseError:
    return ClientResponseError(
        request_info=MagicMock(real_url="http://router/v1/chat/completions"),
        history=(),
        status=status,
        message="upstream",
    )


class _SlowEchoChatModel(_EchoChatModel):
    """Echo model whose chat_completions() takes ``delay_s``; the first ``failures`` calls raise ``error``."""

    delay_s: float = 0.0
    failures: int = 0
    error: object = None
    calls: int = 0

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        object.__setattr__(self, "calls", self.calls + 1)
        await asyncio.sleep(self.delay_s)
        if self.calls <= self.failures:
            raise self.error or RuntimeError("upstream exploded")
        return await super().chat_completions(body)


_FAST_HEARTBEAT = {
    "model_server_sse_heartbeat_grace_seconds": 0.05,
    "model_server_sse_heartbeat_interval_seconds": 0.05,
    "model_server_sse_heartbeat_retry_backoff_seconds": 0.01,
}


def _slow_server(
    delay_s: float, failures: int = 0, error: BaseException | None = None, **config
) -> _SlowEchoChatModel:
    server = _SlowEchoChatModel(
        config=BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient, global_config_dict={**_FAST_HEARTBEAT, **config}),
    )
    object.__setattr__(server, "delay_s", delay_s)
    object.__setattr__(server, "failures", failures)
    object.__setattr__(server, "error", error)
    return server


def _slow_client(delay_s: float, fail: bool = False) -> TestClient:
    return TestClient(_slow_server(delay_s, failures=1 if fail else 0).setup_webserver())


class TestStreamingHeartbeat:
    """Long buffered generations keep the connection busy with SSE comment heartbeats."""

    def test_heartbeat_defaults(self) -> None:
        cfg = base_responses_api_model.ModelServerSSEHeartbeatConfig()
        assert cfg.model_server_sse_heartbeat_grace_seconds == 30.0
        assert cfg.model_server_sse_heartbeat_interval_seconds == 20.0
        assert cfg.model_server_sse_heartbeat_max_retries == 5
        assert cfg.model_server_sse_heartbeat_retry_backoff_seconds == 2.0

    @pytest.mark.parametrize(
        "error, retryable",
        [
            (_http_error(500), True),
            (_http_error(503), True),
            (_http_error(429), True),
            (_http_error(408), True),
            (_http_error(400), False),
            (_http_error(404), False),
            (ServerDisconnectedError(), True),
            (asyncio.TimeoutError(), True),
            (RuntimeError("bug"), False),
        ],
    )
    def test_retryable_after_commit(self, error, retryable) -> None:
        assert base_responses_api_model._retryable_after_commit(error) is retryable

    def test_permanent_endpoint_error_is_not_retried(self) -> None:
        error = PermanentEndpointError(request_info=MagicMock(), history=(), status=429, message="quota")
        assert base_responses_api_model._retryable_after_commit(error) is False

    def _post(self, client: TestClient):
        return client.post(
            "/v1/chat/completions", json={"stream": True, "messages": [{"role": "user", "content": "hello"}]}
        )

    def test_fast_call_has_no_heartbeat(self) -> None:
        resp = self._post(_slow_client(0.0))
        assert resp.status_code == 200
        assert ": keepalive" not in resp.text
        assert resp.text.endswith("data: [DONE]\n\n")

    def test_slow_call_heartbeats_then_replays_completion(self) -> None:
        resp = self._post(_slow_client(0.3))
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.text.startswith(": keepalive\n\n")
        assert resp.text.endswith("data: [DONE]\n\n")
        rebuilt = _reconstruct_chat_sse(_parse_sse_events(resp.text.encode()))
        assert rebuilt["choices"][0]["message"]["content"] == "hello"

    def test_slow_non_retryable_failure_becomes_terminal_error_event(self) -> None:
        server = _slow_server(0.3, failures=1)
        resp = self._post(TestClient(server.setup_webserver()))
        assert resp.status_code == 200
        assert resp.text.startswith(": keepalive\n\n")
        assert _events(resp.text)[-1]["error"]["message"] == "upstream exploded"
        assert server.calls == 1

    def test_slow_5xx_is_retried_then_replays_completion(self) -> None:
        # A late 500 used to end the agent session; the client would have retried an HTTP 500.
        server = _slow_server(0.3, failures=2, error=_http_error(500))
        resp = self._post(TestClient(server.setup_webserver()))
        assert resp.status_code == 200
        assert resp.text.startswith(": keepalive\n\n")
        assert resp.text.endswith("data: [DONE]\n\n")
        assert not any("error" in event for event in _events(resp.text))
        rebuilt = _reconstruct_chat_sse(_parse_sse_events(resp.text.encode()))
        assert rebuilt["choices"][0]["message"]["content"] == "hello"
        assert server.calls == 3

    def test_slow_4xx_is_not_retried(self) -> None:
        server = _slow_server(0.3, failures=5, error=_http_error(400))
        resp = self._post(TestClient(server.setup_webserver()))
        assert "error" in _events(resp.text)[-1]
        assert server.calls == 1

    def test_retries_are_bounded(self) -> None:
        server = _slow_server(0.2, failures=99, error=_http_error(500), model_server_sse_heartbeat_max_retries=2)
        resp = self._post(TestClient(server.setup_webserver()))
        assert "error" in _events(resp.text)[-1]
        assert server.calls == 3

    def test_no_server_side_retry_under_external_staging(self, monkeypatch) -> None:
        # Worker-owned capture poisons a failed call id, so the call must not be re-generated in place.
        monkeypatch.setattr(
            base_responses_api_model, "current_capture_context", lambda: MagicMock(external_staging=True)
        )
        server = _slow_server(0.3, failures=1, error=_http_error(500))
        resp = self._post(TestClient(server.setup_webserver()))
        assert "error" in _events(resp.text)[-1]
        assert server.calls == 1

    def test_fast_failure_still_raises(self) -> None:
        with pytest.raises(RuntimeError, match="upstream exploded"):
            self._post(_slow_client(0.0, fail=True))
