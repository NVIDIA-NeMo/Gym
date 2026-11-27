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
"""Tests for the default ``/v1/messages`` route on ``SimpleResponsesAPIModel``.

Every Gym model server inherits an Anthropic Messages endpoint that maps Messages <-> Responses
around the server's own ``responses()``. Minimal servers cover both ``responses()`` signatures
and native-cause capture reconstruction.
"""

import asyncio
import json
from time import time
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from aiohttp import ClientResponseError
from fastapi import Body, HTTPException, Request
from fastapi.testclient import TestClient

from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.responses_api_types import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.models.anthropic_converter import ANTHROPIC_PING_SSE
from nemo_gym.models.base_responses_api_model import (
    PROPAGATED_REFUSAL_CODE_ATTRIBUTE,
    BaseResponsesAPIModelConfig,
    SimpleResponsesAPIModel,
)
from nemo_gym.models.model_call_capture import _reconstruct_anthropic_sse


def _build_response(text: str, model: str = "downstream-model") -> NeMoGymResponse:
    return NeMoGymResponse(
        id=f"resp_{uuid4().hex}",
        created_at=int(time()),
        model=model,
        object="response",
        output=[
            {
                "type": "message",
                "id": f"msg_{uuid4().hex}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


class _BodyOnlyModel(SimpleResponsesAPIModel):
    """A server whose responses() takes only `body` (like openai_model)."""

    config: BaseResponsesAPIModelConfig
    last_params: object = None
    model_config = {"arbitrary_types_allowed": True}

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        object.__setattr__(self, "last_params", body)
        return _build_response("hi from body-only")

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError


class _RequestAwareModel(SimpleResponsesAPIModel):
    """A server whose responses() also takes `request` (like vllm_model / azure)."""

    config: BaseResponsesAPIModelConfig
    saw_request: bool = False
    model_config = {"arbitrary_types_allowed": True}

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        object.__setattr__(self, "saw_request", isinstance(request, Request))
        return _build_response("hi from request-aware")

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError


def _config(**settings) -> BaseResponsesAPIModelConfig:
    return BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="", **settings)


def _client(model_cls, **settings) -> TestClient:
    server = model_cls(config=_config(**settings), server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    return TestClient(server.setup_webserver()), server


class TestDefaultMessagesRoute:
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("output_config", [None, {"effort": None}, {"effort": "max"}])
    def test_output_config_effort_remains_ignored(self, output_config: object, stream: bool) -> None:
        client, server = _client(_BodyOnlyModel)
        result = client.post(
            "/v1/messages",
            json={
                "model": "synthetic",
                "messages": [],
                "max_tokens": 8,
                "stream": stream,
                "output_config": output_config,
            },
        )
        assert result.status_code == 200
        assert "reasoning" not in server.last_params.model_dump(exclude_unset=True)

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize(
        ("output_config", "reasoning"),
        [(None, None), ({"effort": None}, None), ({"effort": "max"}, {"effort": "max"})],
    )
    def test_output_config_effort_is_forwarded_when_configured(
        self, output_config: object, reasoning: object, stream: bool
    ) -> None:
        client, server = _client(_BodyOnlyModel, forward_anthropic_reasoning_effort=True)
        result = client.post(
            "/v1/messages",
            json={
                "model": "synthetic",
                "messages": [],
                "max_tokens": 8,
                "stream": stream,
                "output_config": output_config,
            },
        )
        assert result.status_code == 200
        assert server.last_params.model_dump(exclude_unset=True).get("reasoning") == reasoning

    def test_nonleading_system_message_becomes_user_message_when_configured(self) -> None:
        body = {
            "model": "synthetic",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}, {"role": "system", "content": "the environment"}],
        }
        client, server = _client(_BodyOnlyModel, anthropic_nonleading_system_note="An operator wrote:")
        assert client.post("/v1/messages", json=body).status_code == 200
        assert [item.role for item in server.last_params.input] == ["user", "user"]
        assert [part["text"] for part in server.last_params.input[1].content] == [
            "An operator wrote:",
            "the environment",
        ]

        client, server = _client(_BodyOnlyModel)
        assert client.post("/v1/messages", json=body).status_code == 200
        assert [item.role for item in server.last_params.input] == ["user", "system"]

    def test_messages_route_registered_alongside_openai_routes(self) -> None:
        server = _BodyOnlyModel(config=_config(), server_client=MagicMock(spec=ServerClient, global_config_dict={}))
        paths = {route.path for route in server.setup_webserver().routes}
        assert {"/v1/messages", "/v1/responses", "/v1/chat/completions"} <= paths

    def test_body_only_responses_signature(self) -> None:
        client, server = _client(_BodyOnlyModel)
        resp = client.post(
            "/v1/messages",
            json={"model": "claude-x", "max_tokens": 32, "messages": [{"role": "user", "content": "hello"}]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["role"] == "assistant"
        assert data["content"] == [{"type": "text", "text": "hi from body-only"}]
        assert data["model"] == "claude-x"  # request model echoed back
        # the inbound Anthropic request was translated to Responses params before delegating
        assert server.last_params.input[0].content == "hello"
        assert server.last_params.max_output_tokens == 32

    def test_request_aware_responses_signature(self) -> None:
        client, server = _client(_RequestAwareModel)
        resp = client.post(
            "/v1/messages",
            json={"model": "claude-x", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["content"] == [{"type": "text", "text": "hi from request-aware"}]
        assert server.saw_request is True  # request was forwarded to responses()

    def test_streaming_returns_anthropic_sse(self) -> None:
        client, _ = _client(_BodyOnlyModel)
        resp = client.post(
            "/v1/messages",
            json={
                "model": "claude-x",
                "max_tokens": 8,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        assert "event: message_start" in body
        assert "event: content_block_delta" in body
        assert "event: message_stop" in body


_KEEPALIVE_S = 0.02
# Several keepalive intervals.
_SLOW_CALL_S = 0.15
_STREAM_BODY = {"model": "claude-x", "max_tokens": 8, "stream": True, "messages": [{"role": "user", "content": "hi"}]}


class _SlowModel(_BodyOnlyModel):
    """Answers, or raises ``failure``, after several keepalive intervals."""

    failure: Exception | None = None
    propagates_upstream_statuses: bool = True

    def _mints_response_ids(self) -> bool:
        return True

    def _should_propagate_upstream_http_error(self, error: ClientResponseError) -> bool:
        return self.propagates_upstream_statuses

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        await asyncio.sleep(_SLOW_CALL_S)
        if self.failure is not None:
            raise self.failure
        return await super().responses(body)


def _slow_client(failure: Exception | None = None, **settings) -> TestClient:
    """Return a test client of a ``_SlowModel`` server whose call raises ``failure`` when one is given."""
    client, server = _client(_SlowModel, **settings)
    object.__setattr__(server, "failure", failure)
    return client


def _blocked_backend(monkeypatch) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold the backend call of ``_SlowModel``; return the events of its start and of its cancellation."""
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def blocked(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(_SlowModel, "_invoke_responses", blocked)
    return started, cancelled


def _events(text: str) -> list[dict]:
    """Return the JSON payload of each ``data:`` line of the event stream ``text``."""
    return [json.loads(line[len("data: ") :]) for line in text.splitlines() if line.startswith("data: ")]


def _upstream_error(status: int, body: bytes, *, refusal_code: str | None = None) -> ClientResponseError:
    """Return the error that an upstream HTTP response with ``status`` and ``body`` raises.

    ``refusal_code`` marks it as a refusal that the model server propagates.
    """
    error = ClientResponseError(MagicMock(real_url="http://engine.test/v1/chat/completions"), (), status=status)
    error.response_content = body
    if refusal_code is not None:
        setattr(error, PROPAGATED_REFUSAL_CODE_ATTRIBUTE, refusal_code)
    return error


class TestMessagesStreamKeepalive:
    """``stream_keepalive_s`` holds a buffered Messages stream open while the backend computes."""

    def test_long_call_sends_pings_then_the_unchanged_message(self) -> None:
        buffered = _slow_client().post("/v1/messages", json=_STREAM_BODY)
        kept_alive = _slow_client(stream_keepalive_s=_KEEPALIVE_S).post("/v1/messages", json=_STREAM_BODY)

        assert kept_alive.status_code == 200
        assert kept_alive.headers["content-type"].startswith("text/event-stream")
        pings = kept_alive.text.count(ANTHROPIC_PING_SSE)
        assert pings >= 3, "the call outlasts several intervals"
        assert kept_alive.text.startswith(ANTHROPIC_PING_SSE * pings)
        message = kept_alive.text[len(ANTHROPIC_PING_SSE) * pings :]
        assert ANTHROPIC_PING_SSE not in buffered.text
        assert [event["type"] for event in _events(message)] == [event["type"] for event in _events(buffered.text)]
        assert _reconstruct_anthropic_sse(_events(kept_alive.text))["content"] == [
            {"type": "text", "text": "hi from body-only"}
        ]

    def test_fast_call_sends_no_ping(self) -> None:
        response = _slow_client(stream_keepalive_s=10 * _SLOW_CALL_S).post("/v1/messages", json=_STREAM_BODY)

        assert ANTHROPIC_PING_SSE not in response.text
        assert _events(response.text)[0]["type"] == "message_start"

    def test_fast_failure_keeps_its_http_status(self) -> None:
        client = _slow_client(
            HTTPException(status_code=429, detail="rate limited"), stream_keepalive_s=10 * _SLOW_CALL_S
        )

        assert client.post("/v1/messages", json=_STREAM_BODY).status_code == 429

    def test_request_that_is_not_streamed_gets_no_event(self) -> None:
        client = _slow_client(stream_keepalive_s=_KEEPALIVE_S)

        response = client.post("/v1/messages", json=_STREAM_BODY | {"stream": False})

        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["content"] == [{"type": "text", "text": "hi from body-only"}]

    @pytest.mark.parametrize(
        ("failure", "error"),
        [
            (RuntimeError("backend exploded late"), {"type": "api_error", "message": "backend exploded late"}),
            (
                HTTPException(status_code=429, detail="rate limited"),
                {"type": "rate_limit_error", "message": "HTTP 429: rate limited"},
            ),
            (
                _upstream_error(502, b"engine restarted"),
                {"type": "api_error", "message": "Upstream HTTP 502: engine restarted"},
            ),
            (
                _upstream_error(422, b"no such parameter"),
                {"type": "invalid_request_error", "message": "Upstream HTTP 422: no such parameter"},
            ),
            (
                _upstream_error(
                    400,
                    b'{"error": {"message": "This model\'s maximum context length is 8 tokens."}}',
                    refusal_code="context_length_exceeded",
                ),
                {"type": "invalid_request_error", "message": "This model's maximum context length is 8 tokens."},
            ),
        ],
        ids=["exception", "http exception", "upstream 502", "upstream 422", "propagated refusal"],
    )
    def test_failure_after_pings_ends_with_an_error_event(self, failure: Exception, error: dict) -> None:
        response = _slow_client(failure, stream_keepalive_s=_KEEPALIVE_S).post("/v1/messages", json=_STREAM_BODY)

        assert response.status_code == 200
        events = _events(response.text)
        assert {event["type"] for event in events[:-1]} == {"ping"}
        assert events[-1] == {"type": "error", "error": error}
        assert response.text.endswith(f"event: error\ndata: {json.dumps(events[-1])}\n\n")

    @pytest.mark.parametrize(
        ("failure", "error"),
        [
            (
                _upstream_error(502, b"engine restarted"),
                {"type": "api_error", "message": "HTTP 500: Model request failed"},
            ),
            (
                _upstream_error(
                    400,
                    b'{"error": {"message": "This model\'s maximum context length is 8 tokens."}}',
                    refusal_code="context_length_exceeded",
                ),
                {"type": "invalid_request_error", "message": "This model's maximum context length is 8 tokens."},
            ),
        ],
        ids=["upstream 502", "propagated refusal"],
    )
    def test_server_hiding_upstream_statuses_still_reports_a_refusal(self, failure: Exception, error: dict) -> None:
        client, server = _client(_SlowModel, stream_keepalive_s=_KEEPALIVE_S)
        object.__setattr__(server, "failure", failure)
        object.__setattr__(server, "propagates_upstream_statuses", False)

        response = client.post("/v1/messages", json=_STREAM_BODY)

        assert _events(response.text)[-1] == {"type": "error", "error": error}

    async def test_cancel_during_the_first_interval_cancels_the_backend(self, monkeypatch) -> None:
        _, server = _client(_SlowModel, stream_keepalive_s=30.0)
        started, cancelled = _blocked_backend(monkeypatch)
        call = asyncio.create_task(server.messages(MagicMock(), dict(_STREAM_BODY)))
        await asyncio.wait_for(started.wait(), 5)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert cancelled.is_set()

    async def test_closing_the_stream_after_pings_cancels_the_backend(self, monkeypatch) -> None:
        _, server = _client(_SlowModel, stream_keepalive_s=_KEEPALIVE_S)
        started, cancelled = _blocked_backend(monkeypatch)
        response = await server.messages(MagicMock(), dict(_STREAM_BODY))
        assert await anext(response.body_iterator) == ANTHROPIC_PING_SSE
        assert started.is_set()
        await response.body_iterator.aclose()
        assert cancelled.is_set()


def test_anthropic_sse_reconstruction_preserves_zero_native_causes() -> None:
    response = _reconstruct_anthropic_sse(
        [
            {
                "type": "message_start",
                "message": {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "model": "model",
                    "stop_reason": "end_turn",
                    "native_stop_reason": 0,
                    "native_finish_reason": 0,
                },
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "answer"}},
            {"type": "message_stop"},
        ]
    )
    assert response["stop_reason"] == "end_turn"
    assert response["content"] == [{"type": "text", "text": "answer"}]
    for key in ("native_stop_reason", "native_finish_reason"):
        assert response[key] == 0
        assert type(response[key]) is int
