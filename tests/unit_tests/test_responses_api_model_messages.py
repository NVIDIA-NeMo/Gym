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
from aiohttp import ClientResponseError
from fastapi import Body, Request
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    CaptureStore,
    SimpleResponsesAPIModel,
    _reconstruct_chat_sse,
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


def _build_chat_completion() -> NeMoGymChatCompletion:
    return NeMoGymChatCompletion.model_validate(
        {
            "id": "chatcmpl_test",
            "created": 123,
            "model": "downstream-model",
            "object": "chat.completion",
            "service_tier": "default",
            "system_fingerprint": "fp_test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "first",
                        "reasoning_content": "thinking",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": '{"id":1}'},
                            }
                        ],
                    },
                },
                {
                    "index": 1,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "second"},
                },
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
    )


class _BodyOnlyModel(SimpleResponsesAPIModel):
    """A server whose responses() takes only `body` (like openai_model)."""

    config: BaseResponsesAPIModelConfig
    last_params: object = None
    last_chat_params: object = None
    model_config = {"arbitrary_types_allowed": True}

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        object.__setattr__(self, "last_params", body)
        return _build_response("hi from body-only")

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        object.__setattr__(self, "last_chat_params", body)
        return _build_chat_completion()


class _RequestAwareModel(SimpleResponsesAPIModel):
    """A server whose responses() also takes `request` (like vllm_model / azure)."""

    config: BaseResponsesAPIModelConfig
    saw_request: bool = False
    saw_chat_request: bool = False
    last_chat_params: object = None
    model_config = {"arbitrary_types_allowed": True}

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        object.__setattr__(self, "saw_request", isinstance(request, Request))
        return _build_response("hi from request-aware")

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        object.__setattr__(self, "saw_chat_request", isinstance(request, Request))
        object.__setattr__(self, "last_chat_params", body)
        return _build_chat_completion()


class _FailingModel(_BodyOnlyModel):
    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        error = ClientResponseError(
            request_info=MagicMock(real_url="https://example.invalid/v1/chat/completions"),
            history=(),
            status=403,
            message="Forbidden",
        )
        error.response_content = b'{"error":{"message":"denied"}}'
        raise error


def _config() -> BaseResponsesAPIModelConfig:
    return BaseResponsesAPIModelConfig(host="0.0.0.0", port=8099, entrypoint="", name="")


def _client(model_cls) -> TestClient:
    server = model_cls(config=_config(), server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    return TestClient(server.setup_webserver()), server


def _sse_payloads(body: str) -> list[dict]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


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


@pytest.mark.parametrize("model_cls", [_BodyOnlyModel, _RequestAwareModel])
def test_chat_completions_streaming_wraps_existing_nonstreaming_handler(model_cls) -> None:
    client, server = _client(model_cls)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "requested-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.endswith("data: [DONE]\n\n")
    assert server.last_chat_params.stream is False
    assert server.last_chat_params.stream_options is None
    if model_cls is _RequestAwareModel:
        assert server.saw_chat_request is True

    payloads = _sse_payloads(response.text)
    reconstructed = NeMoGymChatCompletion.model_validate(_reconstruct_chat_sse(payloads))
    assert reconstructed.model_dump(mode="json") == _build_chat_completion().model_dump(mode="json")


def test_chat_completions_nonstreaming_is_unchanged() -> None:
    client, server = _client(_BodyOnlyModel)
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["object"] == "chat.completion"
    assert server.last_chat_params.stream is None


def test_streaming_chat_completion_capture_keeps_wire_request_and_reconstructs_response(tmp_path) -> None:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {
        "observability_enabled": True,
        "model_call_capture_dir": str(tmp_path),
    }
    server = _BodyOnlyModel(config=_config(), server_client=server_client)
    client = TestClient(server.setup_webserver())

    response = client.post(
        "/ng-rollout/rollout-1/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    exchange = CaptureStore(tmp_path).read("rollout-1")[0]
    assert exchange["status_code"] == 200
    assert exchange["error_category"] is None
    assert exchange["request"]["stream"] is True
    assert "stream_options" not in exchange["request"]
    reconstructed = NeMoGymChatCompletion.model_validate(exchange["response"])
    assert reconstructed.model_dump(mode="json") == _build_chat_completion().model_dump(mode="json")
    assert all("usage" not in payload for payload in _sse_payloads(response.text))
    assert exchange["response_raw"].endswith("data: [DONE]\n\n")


def test_capture_preserves_upstream_error_status_and_body(tmp_path) -> None:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {
        "observability_enabled": True,
        "model_call_capture_dir": str(tmp_path),
    }
    server = _FailingModel(config=_config(), server_client=server_client)
    client = TestClient(server.setup_webserver(), raise_server_exceptions=False)

    response = client.post(
        "/ng-rollout/rollout-1/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 500
    exchange = CaptureStore(tmp_path).read("rollout-1")[0]
    assert exchange["status_code"] == 403
    assert exchange["error_category"] == "auth"
    assert json.loads(exchange["response_raw"]) == {"error": {"message": "denied"}}
