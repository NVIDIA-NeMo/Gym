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
around the server's own ``responses()``. These tests use minimal fake servers to exercise the
default mapping for both ``responses()`` signatures (with and without a leading ``request``).
"""

from time import time
from unittest.mock import MagicMock
from uuid import uuid4

from fastapi import Body, Request
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient


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


def _config() -> BaseResponsesAPIModelConfig:
    return BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="")


def _client(model_cls) -> TestClient:
    server = model_cls(config=_config(), server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    return TestClient(server.setup_webserver()), server


class TestDefaultMessagesRoute:
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


def _build_thinking_only_response(model: str = "downstream-model") -> NeMoGymResponse:
    return NeMoGymResponse(
        id=f"resp_{uuid4().hex}",
        created_at=int(time()),
        model=model,
        object="response",
        output=[
            {
                "type": "reasoning",
                "id": f"rs_{uuid4().hex}",
                "summary": [{"type": "summary_text", "text": "still thinking..."}],
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


class _ThinkingOnlyModel(SimpleResponsesAPIModel):
    """A server whose model ends every generation with reasoning only (e.g. inside an unclosed <think>)."""

    config: BaseResponsesAPIModelConfig
    calls: int = 0
    finalized: int = 0
    model_config = {"arbitrary_types_allowed": True}

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        object.__setattr__(self, "calls", self.calls + 1)
        return _build_thinking_only_response()

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError

    async def _finalize_served_response(self, response) -> None:
        object.__setattr__(self, "finalized", self.finalized + 1)


class TestThinkingOnlyRetry:
    _BODY = {"model": "claude-x", "max_tokens": 32, "messages": [{"role": "user", "content": "fix the bug"}]}

    def setup_method(self) -> None:
        from nemo_gym import base_responses_api_model as module

        module._THINKING_ONLY_RETRIES.clear()

    def _server(self, model_cls=_ThinkingOnlyModel, **config_overrides):
        config = BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="", **config_overrides)
        server = model_cls(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
        return TestClient(server.setup_webserver()), server

    def test_disabled_by_default(self) -> None:
        client, _ = self._server()
        resp = client.post("/v1/messages", json=self._BODY)
        assert resp.status_code == 200
        assert resp.json()["content"][0]["type"] == "thinking"

    def test_thinking_only_returns_retryable_529(self) -> None:
        client, server = self._server(anthropic_thinking_only_max_retries=4)
        resp = client.post("/v1/messages", json=self._BODY)
        assert resp.status_code == 529
        assert resp.json() == {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "model returned a thinking-only turn; retry"},
        }
        assert resp.headers["x-should-retry"] == "true"
        # The sampled generation is still finalized for capture.
        assert server.finalized == 1

    def test_streaming_request_also_gets_529(self) -> None:
        client, _ = self._server(anthropic_thinking_only_max_retries=4)
        resp = client.post("/v1/messages", json={**self._BODY, "stream": True})
        assert resp.status_code == 529
        assert resp.json()["error"]["type"] == "overloaded_error"

    def test_visible_output_is_served_normally(self) -> None:
        client, _ = self._server(model_cls=_BodyOnlyModel, anthropic_thinking_only_max_retries=4)
        resp = client.post("/v1/messages", json=self._BODY)
        assert resp.status_code == 200
        assert resp.json()["content"] == [{"type": "text", "text": "hi from body-only"}]

    def test_falls_through_after_max_retries_for_same_request(self) -> None:
        client, server = self._server(anthropic_thinking_only_max_retries=2)
        statuses = [client.post("/v1/messages", json=self._BODY).status_code for _ in range(3)]
        assert statuses == [529, 529, 200]
        assert server.calls == 3
        # After a fall-through the tally for that body starts over.
        assert client.post("/v1/messages", json=self._BODY).status_code == 529

    def test_fall_through_returns_the_thinking_only_message(self) -> None:
        client, _ = self._server(anthropic_thinking_only_max_retries=1)
        assert client.post("/v1/messages", json=self._BODY).status_code == 529
        resp = client.post("/v1/messages", json=self._BODY)
        assert resp.status_code == 200
        assert resp.json()["content"] == [{"type": "thinking", "thinking": "still thinking...", "signature": ""}]
        assert resp.json()["stop_reason"] == "end_turn"

    def test_different_request_bodies_count_separately(self) -> None:
        client, _ = self._server(anthropic_thinking_only_max_retries=1)
        assert client.post("/v1/messages", json=self._BODY).status_code == 529
        other = {**self._BODY, "messages": [{"role": "user", "content": "something else"}]}
        assert client.post("/v1/messages", json=other).status_code == 529
        assert client.post("/v1/messages", json=self._BODY).status_code == 200
