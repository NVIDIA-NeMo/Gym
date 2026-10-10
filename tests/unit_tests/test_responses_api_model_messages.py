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

import json
from time import time
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import Body, Request
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    SimpleResponsesAPIModel,
    _soften_max_tokens_stop_reason,
)
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


class _TruncatedModel(SimpleResponsesAPIModel):
    """A server whose responses are cut off by a server-side max_output_tokens cap."""

    config: BaseResponsesAPIModelConfig
    model_config = {"arbitrary_types_allowed": True}

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        response = _build_response("partial answer").model_dump()
        response.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        return NeMoGymResponse.model_validate(response)

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        raise NotImplementedError


def _config(**kwargs) -> BaseResponsesAPIModelConfig:
    return BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="", **kwargs)


def _client(model_cls, **config_kwargs) -> TestClient:
    server = model_cls(
        config=_config(**config_kwargs), server_client=MagicMock(spec=ServerClient, global_config_dict={})
    )
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


class TestAnthropicMaxTokensAsEndTurn:
    _BODY = {"model": "claude-x", "max_tokens": 32000, "messages": [{"role": "user", "content": "hi"}]}

    def test_max_tokens_stop_reason_is_kept_by_default(self) -> None:
        client, _ = _client(_TruncatedModel)
        resp = client.post("/v1/messages", json=self._BODY)
        assert resp.status_code == 200
        assert resp.json()["stop_reason"] == "max_tokens"

    def test_flag_reports_truncated_turn_as_end_turn(self) -> None:
        client, _ = _client(_TruncatedModel, anthropic_max_tokens_as_end_turn=True)
        resp = client.post("/v1/messages", json=self._BODY)
        assert resp.status_code == 200
        assert resp.json()["stop_reason"] == "end_turn"
        assert resp.json()["content"] == [{"type": "text", "text": "partial answer"}]

    def test_flag_applies_to_streamed_responses(self) -> None:
        client, _ = _client(_TruncatedModel, anthropic_max_tokens_as_end_turn=True)
        resp = client.post("/v1/messages", json={**self._BODY, "stream": True})
        assert resp.status_code == 200
        stop_reasons = [
            json.loads(line[len("data:") :])["delta"]["stop_reason"]
            for line in resp.text.splitlines()
            if line.startswith("data:") and '"message_delta"' in line
        ]
        assert stop_reasons == ["end_turn"]

    def test_truncated_turn_with_tool_calls_becomes_tool_use(self) -> None:
        response = {
            "stop_reason": "max_tokens",
            "content": [{"type": "text", "text": "x"}, {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}],
        }
        _soften_max_tokens_stop_reason(response)
        assert response["stop_reason"] == "tool_use"

    @pytest.mark.parametrize("reason", ["end_turn", "tool_use", "refusal", "stop_sequence"])
    def test_other_stop_reasons_are_untouched(self, reason: str) -> None:
        response = {"stop_reason": reason, "content": []}
        _soften_max_tokens_stop_reason(response)
        assert response["stop_reason"] == reason
