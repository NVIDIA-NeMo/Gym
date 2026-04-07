# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.server_utils import ServerClient
from responses_api_models.litellm_model.app import (
    LiteLLMModelServer,
    LiteLLMModelServerConfig,
    _normalize_to_response,
)


# -- Fixtures / helpers -------------------------------------------------------

NATIVE_RESPONSE = {
    "id": "resp_abc123",
    "created_at": 1700000000.0,
    "model": "openai/openai/gpt-5.4",
    "object": "response",
    "output": [
        {
            "id": "msg_abc123",
            "content": [{"annotations": [], "text": "Hello!", "type": "output_text"}],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    ],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}

CHAT_COMPLETION_RESPONSE = {
    "id": "chatcmpl-xyz789",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {"content": "Hi from Opus!", "role": "assistant"},
        }
    ],
    "created": 1700000000,
    "model": "azure/anthropic/claude-opus-4-6",
    "object": "chat.completion",
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

HYBRID_RESPONSE = {
    "id": "chatcmpl-hybrid456",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "azure/anthropic/claude-opus-4-6",
    "output": [
        {
            "id": "msg_hybrid",
            "content": [{"annotations": [], "text": "Hybrid text!", "type": "output_text"}],
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    ],
    "usage": {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
}


def _make_server() -> LiteLLMModelServer:
    config = LiteLLMModelServerConfig(
        host="0.0.0.0",
        port=8081,
        openai_base_url="https://litellm.example.com/v1",
        openai_api_key="dummy_key",  # pragma: allowlist secret
        openai_model="dummy_model",
        entrypoint="",
        name="",
    )
    return LiteLLMModelServer(config=config, server_client=MagicMock(spec=ServerClient))


# -- Unit tests for _normalize_to_response ------------------------------------


class TestNormalizeToResponse:
    def test_native_response_passthrough(self) -> None:
        """Native response format passes through unchanged."""
        data = deepcopy(NATIVE_RESPONSE)
        result = _normalize_to_response(data)
        assert result["object"] == "response"
        assert result["output"][0]["content"][0]["text"] == "Hello!"

    def test_reasoning_effort_none_fix(self) -> None:
        """reasoning.effort = 'none' (string) is normalized to null."""
        data = deepcopy(NATIVE_RESPONSE)
        data["reasoning"] = {"effort": "none"}
        result = _normalize_to_response(data)
        assert result["reasoning"]["effort"] is None
        # Still a native response, not converted
        assert result["object"] == "response"

    def test_reasoning_effort_valid_preserved(self) -> None:
        """Valid reasoning.effort values are not touched."""
        data = deepcopy(NATIVE_RESPONSE)
        data["reasoning"] = {"effort": "high"}
        result = _normalize_to_response(data)
        assert result["reasoning"]["effort"] == "high"

    def test_chat_completion_normalization(self) -> None:
        """Standard chat.completion with choices[] is normalized."""
        data = deepcopy(CHAT_COMPLETION_RESPONSE)
        result = _normalize_to_response(data)
        assert result["object"] == "response"
        assert result["output"][0]["content"][0]["text"] == "Hi from Opus!"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_hybrid_format_normalization(self) -> None:
        """chat.completion with output[] (LiteLLM hybrid) is normalized."""
        data = deepcopy(HYBRID_RESPONSE)
        result = _normalize_to_response(data)
        assert result["object"] == "response"
        assert result["output"][0]["content"][0]["text"] == "Hybrid text!"
        assert result["usage"]["input_tokens"] == 8

    def test_chat_completion_empty_content(self) -> None:
        """chat.completion with empty content produces empty text."""
        data = deepcopy(CHAT_COMPLETION_RESPONSE)
        data["choices"][0]["message"]["content"] = None
        result = _normalize_to_response(data)
        assert result["object"] == "response"
        assert result["output"][0]["content"][0]["text"] == ""


# -- Integration tests for the server -----------------------------------------


class TestLiteLLMModelServer:
    async def test_responses_native_format(self) -> None:
        """Server handles native response format from LiteLLM (e.g. GPT-5.4)."""
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        mock_data = deepcopy(NATIVE_RESPONSE)

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(return_value=mock_data)

        resp = client.post("/v1/responses", json={"input": "hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "response"
        assert body["output"][0]["content"][0]["text"] == "Hello!"

    async def test_responses_chat_completion_format(self) -> None:
        """Server normalizes chat.completion format from LiteLLM (e.g. Opus via Azure)."""
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        mock_data = deepcopy(CHAT_COMPLETION_RESPONSE)

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(return_value=mock_data)

        resp = client.post("/v1/responses", json={"input": "hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "response"
        assert body["output"][0]["content"][0]["text"] == "Hi from Opus!"

    async def test_responses_reasoning_effort_fix(self) -> None:
        """Server fixes reasoning.effort='none' before validation."""
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        mock_data = deepcopy(NATIVE_RESPONSE)
        mock_data["reasoning"] = {"effort": "none"}

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(return_value=mock_data)

        resp = client.post("/v1/responses", json={"input": "hello"})
        assert resp.status_code == 200

    async def test_chat_completions_passthrough(self) -> None:
        """chat_completions() is inherited from SimpleModelServer and works unchanged."""
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        mock_chat_data = deepcopy(CHAT_COMPLETION_RESPONSE)

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(return_value=mock_chat_data)

        resp = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"

    async def test_model_override(self) -> None:
        """Config model name is always used regardless of request body."""
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        mock_data = deepcopy(NATIVE_RESPONSE)
        called_args = {}

        async def mock_create_response(**kwargs):
            nonlocal called_args
            called_args = kwargs
            return mock_data

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(side_effect=mock_create_response)

        client.post("/v1/responses", json={"input": "hello", "model": "wrong_model"})
        assert called_args.get("model") == "dummy_model"
