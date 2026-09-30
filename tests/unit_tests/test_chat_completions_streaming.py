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
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    ModelCallCaptureConfig,
    SimpleResponsesAPIModel,
    _parse_sse_events,
    _reconstruct_chat_sse,
    install_model_call_capture,
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


_ASGI_SCOPE = {
    "type": "http",
    "asgi": {"version": "3.0", "spec_version": "2.4"},
    "http_version": "1.1",
    "method": "POST",
    "scheme": "http",
    "path": "/v1/chat/completions",
    "raw_path": b"/v1/chat/completions",
    "query_string": b"",
    "headers": [],
    "server": ("test", 80),
    "client": ("test", 1),
}
_STREAM_BODY = {
    "stream": True,
    "stream_options": {"include_usage": True},
    "messages": [{"role": "user", "content": "hello"}],
}


async def _never_receive():
    await asyncio.Event().wait()


async def _dispatch_fake(invoke):
    return await SimpleResponsesAPIModel.chat_completions_dispatch(
        SimpleNamespace(_invoke_chat_completions=invoke),
        Request(_ASGI_SCOPE),
        _STREAM_BODY,
    )


async def test_delayed_chat_sends_headers_and_periodic_comments_before_completion(monkeypatch):
    import nemo_gym.chat_streaming as streaming

    monkeypatch.setattr(streaming, "CHAT_KEEPALIVE_SECONDS", 0.01)
    release = asyncio.Event()
    started = asyncio.Event()
    calls = 0
    expected = _completion(content="hello", reasoning="reason", tool_calls=[_TOOL_CALL], usage=_USAGE)

    async def invoke(request, params):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return expected

    response = await asyncio.wait_for(_dispatch_fake(invoke), timeout=1)
    sent = asyncio.Queue()
    task = asyncio.create_task(response(_ASGI_SCOPE, _never_receive, sent.put))
    try:
        first = await asyncio.wait_for(sent.get(), timeout=1)
        assert first["type"] == "http.response.start"
        assert first["status"] == 200
        await asyncio.wait_for(started.wait(), timeout=1)
        chunks = [(await asyncio.wait_for(sent.get(), timeout=1))["body"] for _ in range(3)]
        assert all(chunk == b": keepalive\n\n" for chunk in chunks)
        assert not release.is_set()
        release.set()
        await asyncio.wait_for(task, timeout=1)
        while not sent.empty():
            chunks.append((await sent.get()).get("body", b""))
        body = b"".join(chunks)
        rebuilt = _reconstruct_chat_sse(_parse_sse_events(body))
        assert rebuilt["choices"][0]["message"]["content"] == "hello"
        assert rebuilt["choices"][0]["message"]["tool_calls"][0]["function"] == _TOOL_CALL["function"]
        assert {key: rebuilt["usage"][key] for key in _USAGE} == _USAGE
        assert body.count(b"data: [DONE]") == 1
        assert calls == 1
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("spec", ["2.3", "2.4"])
async def test_chat_disconnect_cancels_pending_model_call(monkeypatch, spec):
    from starlette.requests import ClientDisconnect

    import nemo_gym.chat_streaming as streaming

    monkeypatch.setattr(streaming, "CHAT_KEEPALIVE_SECONDS", 0.01)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def invoke(request, params):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def receive():
        await started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if spec == "2.4" and started.is_set() and message["type"] == "http.response.body":
            raise OSError("client disconnected")

    response = await _dispatch_fake(invoke)
    scope = {**_ASGI_SCOPE, "asgi": {"version": "3.0", "spec_version": spec}}
    if spec == "2.4":
        with pytest.raises(ClientDisconnect):
            await asyncio.wait_for(response(scope, receive, send), timeout=1)
    else:
        await asyncio.wait_for(response(scope, receive, send), timeout=1)
    assert cancelled.is_set()


@pytest.mark.parametrize(
    "failure", [None, RuntimeError("backend failed"), TimeoutError(), HTTPException(400, detail={"error": "too long"})]
)
def test_chat_keepalive_preserves_capture_and_terminal_errors(tmp_path, failure):
    calls = []

    async def invoke(request, params):
        calls.append(params)
        await asyncio.sleep(0)
        if failure is not None:
            raise failure
        return _completion(content="hello", usage=_USAGE)

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def route(request: Request, body: dict):
        return await SimpleResponsesAPIModel.chat_completions_dispatch(
            SimpleNamespace(_invoke_chat_completions=invoke),
            request,
            body,
        )

    install_model_call_capture(
        app, ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=tmp_path)
    )
    with TestClient(app) as client:
        response = client.post("/ng-rollout/1-0/v1/chat/completions", json=_STREAM_BODY)
    assert response.status_code == 200
    assert response.text.startswith(": keepalive\n\n")
    records = [json.loads(line) for line in (tmp_path / "1-0.capture.jsonl").read_text().splitlines()]
    assert len(records) == len(calls) == 1
    record = records[0]
    if failure is None:
        assert record["error_category"] is None
        assert {key: record["response"]["usage"][key] for key in _USAGE} == _USAGE
        assert record["response"]["choices"][0]["message"]["content"] == "hello"
    else:
        assert record["error_category"] == "upstream_error"
        assert "[DONE]" not in response.text
        error = _events(response.text)[0]["error"]
        assert error["type"] == type(failure).__name__
        assert error["message"]
        assert error["code"] == (400 if isinstance(failure, HTTPException) else "internal_error")


def test_capture_ttft_ignores_split_sse_comments(tmp_path, monkeypatch):
    from starlette.responses import StreamingResponse

    import nemo_gym.base_responses_api_model as capture

    clock = [0.0]
    monkeypatch.setattr(capture.time, "perf_counter", lambda: clock[0])
    app = FastAPI()

    async def chunks():
        clock[0] = 0.2
        yield b": keep"
        clock[0] = 0.4
        yield b"alive\n\ndata: \n\n"
        clock[0] = 0.6
        yield b": another comment\n\n"
        clock[0] = 1.0
        for chunk in synthesize_chat_completion_sse(
            _completion(content="ok", usage=_USAGE).model_dump(), include_usage=True
        ):
            yield chunk[:2]
            yield chunk[2:]

    @app.post("/v1/chat/completions")
    async def route():
        return StreamingResponse(chunks(), media_type="text/event-stream")

    install_model_call_capture(
        app, ModelCallCaptureConfig(observability_enabled=True, model_call_capture_dir=tmp_path)
    )
    with TestClient(app) as client:
        response = client.post("/ng-rollout/1-0/v1/chat/completions", json=_STREAM_BODY)
    assert response.status_code == 200
    record = json.loads((tmp_path / "1-0.capture.jsonl").read_text())
    assert record["latency_ttft_ms"] == 1000.0
    assert record["error_category"] is None
    assert record["response"]["usage"]["total_tokens"] == 10
