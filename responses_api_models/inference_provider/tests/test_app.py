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
import json
from unittest.mock import AsyncMock, MagicMock

from aiohttp.client_exceptions import ClientResponseError
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI
from nemo_gym.server_utils import ServerClient
from responses_api_models.inference_provider.app import (
    InferenceProvider,
    InferenceProviderConfig,
)


FIXED_TIME = 1691418000
FIXED_UUID = "abc123"


class FakeUUID:
    hex = FIXED_UUID


def _make_server(**overrides):
    defaults = dict(
        host="0.0.0.0",
        port=8081,
        base_url="https://api.example.com/v1",
        api_key="test-key",  # pragma: allowlist secret
        model="test-model",
        entrypoint="",
        name="",
    )
    defaults.update(overrides)
    config = InferenceProviderConfig(**defaults)
    return InferenceProvider(config=config, server_client=MagicMock(spec=ServerClient))


def _mock_chat_response(content="Hello!", finish_reason="stop", tool_calls=None, usage=None):
    response = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "finish_reason": finish_reason,
                "index": 0,
                "message": {
                    "content": content,
                    "role": "assistant",
                },
            }
        ],
        "created": FIXED_TIME,
        "model": "test-model",
        "object": "chat.completion",
    }
    if tool_calls:
        response["choices"][0]["message"]["tool_calls"] = tool_calls
    if usage:
        response["usage"] = usage
    return response


def _provider_error(status: int, message: str, response_body: str | None = None) -> ClientResponseError:
    error = ClientResponseError(MagicMock(), (), status=status, message="provider request failed", headers=None)
    error.message = message
    error.response_content = (response_body or message).encode("utf-8")
    return error


class TestSanity:
    async def test_server_instantiation(self) -> None:
        server = _make_server()
        assert server.config.model == "test-model"
        assert server.config.base_url == "https://api.example.com/v1"
        assert server.config.num_concurrent_requests == 1000
        assert server.config.uses_reasoning_parser is False

    async def test_server_with_custom_config(self) -> None:
        server = _make_server(
            num_concurrent_requests=500,
            uses_reasoning_parser=True,
            extra_body={"frequency_penalty": 0.5},
        )
        assert server.config.num_concurrent_requests == 500
        assert server.config.uses_reasoning_parser is True
        assert server.config.extra_body == {"frequency_penalty": 0.5}


class TestInferenceProvider:
    async def test_basic_chat_completion(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        mock_data = _mock_chat_response(content="Hello! How can I help?")

        async def mock_create_chat(**kwargs):
            return mock_data

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello! How can I help?"

    async def test_model_default_from_config(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server(model="my-configured-model")
        app = server.setup_webserver()
        client = TestClient(app)

        called_kwargs = {}

        async def mock_create_chat(**kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            return _mock_chat_response()

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert called_kwargs["model"] == "my-configured-model"

    async def test_request_model_is_overridden_by_config(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server(model="default-model")
        app = server.setup_webserver()
        client = TestClient(app)

        called_kwargs = {}

        async def mock_create_chat(**kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            return _mock_chat_response()

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "model": "override-model",
            },
        )
        assert called_kwargs["model"] == "default-model"

    async def test_extra_body_merged(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server(extra_body={"frequency_penalty": 0.5, "presence_penalty": 0.3})
        app = server.setup_webserver()
        client = TestClient(app)

        called_kwargs = {}

        async def mock_create_chat(**kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            return _mock_chat_response()

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert called_kwargs["frequency_penalty"] == 0.5
        assert called_kwargs["presence_penalty"] == 0.3

    async def test_extra_body_does_not_override_request_params(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server(extra_body={"temperature": 0.5})
        app = server.setup_webserver()
        client = TestClient(app)

        called_kwargs = {}

        async def mock_create_chat(**kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            return _mock_chat_response()

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.9,
            },
        )
        assert called_kwargs["temperature"] == 0.9

    async def test_reasoning_parser_strips_think_tags_from_input(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server(uses_reasoning_parser=True)
        app = server.setup_webserver()
        client = TestClient(app)

        called_kwargs = {}

        async def mock_create_chat(**kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            return _mock_chat_response(content="The answer is 42.")

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "What is 6*7?"},
                    {"role": "assistant", "content": "<think>Let me calculate...</think>The answer is 42."},
                ]
            },
        )
        # The think tags should be stripped from the assistant message in the request
        assistant_msg = called_kwargs["messages"][1]
        assert "<think>" not in assistant_msg["content"]
        assert assistant_msg["content"] == "The answer is 42."

    async def test_reasoning_parser_wraps_reasoning_content_in_response(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server(uses_reasoning_parser=True)
        app = server.setup_webserver()
        client = TestClient(app)

        mock_data = _mock_chat_response(content="The answer is 42.")
        mock_data["choices"][0]["message"]["reasoning_content"] = "Let me think about this..."

        async def mock_create_chat(**kwargs):
            return mock_data

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "What is 6*7?"}]},
        )
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        assert content == "<think>Let me think about this...</think>The answer is 42."
        assert "reasoning_content" not in data["choices"][0]["message"]


class TestProviderErrors:
    async def test_chat_completion_provider_auth_error_is_structured(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        auth_error_content = json.dumps({"error": {"message": "Invalid API key", "type": "authentication_error"}})
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(
            side_effect=_provider_error(401, "Authentication failed", auth_error_content)
        )

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "whoami"}]},
        )
        assert response.status_code == 401
        detail = response.json()["detail"]
        assert detail["provider_status"] == 401
        assert detail["category"] == "authentication"
        assert detail["retryable"] is False
        assert detail["provider_context"]["base_url"] == "https://api.example.com/v1"
        assert detail["model"] == "test-model"
        assert detail["message"] == "Invalid API key"

    async def test_chat_completion_provider_rate_limit_error_is_retryable(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        rate_error_content = json.dumps({"error": {"message": "Rate limit exceeded"}})
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(
            side_effect=_provider_error(429, "Rate limit exceeded", rate_error_content)
        )

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "too much"}]},
        )
        assert response.status_code == 429
        detail = response.json()["detail"]
        assert detail["provider_status"] == 429
        assert detail["category"] == "rate_limit"
        assert detail["retryable"] is True
        assert detail["message"] == "Rate limit exceeded"

    async def test_chat_completion_provider_request_error_is_structured(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        request_error_content = json.dumps({"error": {"message": "Missing required field: messages"}})
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(
            side_effect=_provider_error(400, "Bad request", request_error_content)
        )

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "bad payload"}]},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["provider_status"] == 400
        assert detail["category"] == "request_error"
        assert detail["retryable"] is False
        assert detail["message"] == "Missing required field: messages"

    async def test_chat_completion_provider_transient_error_is_retryable(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        transient_error_content = json.dumps({"error": {"message": "Upstream provider unavailable"}})
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(
            side_effect=_provider_error(503, "Service unavailable", transient_error_content)
        )

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "retry me"}]},
        )
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["provider_status"] == 503
        assert detail["category"] == "transient_upstream_failure"
        assert detail["retryable"] is True
        assert detail["message"] == "Upstream provider unavailable"

    async def test_chat_completion_provider_status_zero_falls_back_to_500(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(
            side_effect=_provider_error(0, "Connection closed by upstream", "Connection closed by upstream")
        )

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 500
        detail = response.json()["detail"]
        assert detail["provider_status"] == 500
        assert detail["category"] == "transient_upstream_failure"
        assert detail["retryable"] is True
        assert detail["message"] == "Connection closed by upstream"

    async def test_chat_completion_provider_plain_text_error_body_is_preserved(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        plain_text_error = "gateway timeout while reading upstream response"
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(
            side_effect=_provider_error(502, "Bad gateway", plain_text_error)
        )

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 502
        detail = response.json()["detail"]
        assert detail["provider_status"] == 502
        assert detail["category"] == "transient_upstream_failure"
        assert detail["retryable"] is True
        assert detail["message"] == plain_text_error

    async def test_chat_completion_provider_error_message_is_truncated_and_compacted(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        long_message = "  ".join(["very-long-provider-error-message"] * 20)
        error_content = json.dumps({"error": {"message": long_message}})
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(
            side_effect=_provider_error(418, "Provider said no", error_content)
        )

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 418
        detail = response.json()["detail"]
        assert detail["provider_status"] == 418
        assert detail["category"] == "provider_error"
        assert detail["retryable"] is False
        assert len(detail["message"]) == 200
        assert detail["message"].endswith("...")

    async def test_responses_provider_error_uses_same_structured_contract(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        model_error_content = json.dumps({"error": {"message": "Model test-model does not exist"}})
        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(
            side_effect=_provider_error(404, "Model not found", model_error_content)
        )

        response = client.post(
            "/v1/responses",
            json={"input": "whoami"},
        )
        assert response.status_code == 404
        detail = response.json()["detail"]
        assert detail["provider_status"] == 404
        assert detail["category"] == "model_not_found"
        assert detail["retryable"] is False
        assert detail["provider_context"]["base_url"] == "https://api.example.com/v1"
        assert detail["message"] == "Model test-model does not exist"


class TestResponses:
    async def test_basic_responses(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        monkeypatch.setattr("responses_api_models.inference_provider.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("responses_api_models.inference_provider.app.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        mock_data = _mock_chat_response(content="Hello from the model!")

        async def mock_create_chat(**kwargs):
            return mock_data

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        response = client.post(
            "/v1/responses",
            json={"input": "hello"},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["id"] == f"resp_{FIXED_UUID}"
        assert data["model"] == "test-model"
        assert data["object"] == "response"
        assert len(data["output"]) == 1
        assert data["output"][0]["type"] == "message"
        assert data["output"][0]["content"][0]["text"] == "Hello from the model!"

    async def test_responses_defaults_tool_choice_to_auto(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        monkeypatch.setattr("responses_api_models.inference_provider.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("responses_api_models.inference_provider.app.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        async def mock_create_chat(**kwargs):
            return _mock_chat_response()

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        response = client.post("/v1/responses", json={"input": "hello"})
        assert response.status_code == 200
        assert response.json()["tool_choice"] == "auto"

    async def test_responses_preserves_explicit_tool_choice(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        monkeypatch.setattr("responses_api_models.inference_provider.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("responses_api_models.inference_provider.app.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        async def mock_create_chat(**kwargs):
            return _mock_chat_response()

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        response = client.post("/v1/responses", json={"input": "hello", "tool_choice": "required"})
        assert response.status_code == 200
        assert response.json()["tool_choice"] == "required"

    async def test_responses_with_tool_calls(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        monkeypatch.setattr("responses_api_models.inference_provider.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("responses_api_models.inference_provider.app.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        mock_data = _mock_chat_response(
            content=None,
            finish_reason="tool_calls",
            tool_calls=[
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                    },
                }
            ],
        )

        async def mock_create_chat(**kwargs):
            return mock_data

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        response = client.post(
            "/v1/responses",
            json={
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "What's the weather?"}],
                        "type": "message",
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                        "description": "Get the weather for a location",
                        "strict": True,
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()

        function_calls = [o for o in data["output"] if o["type"] == "function_call"]
        assert len(function_calls) == 1
        assert function_calls[0]["name"] == "get_weather"
        assert function_calls[0]["arguments"] == '{"location": "San Francisco"}'
        assert function_calls[0]["call_id"] == "call_abc123"

    async def test_responses_with_reasoning(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server(uses_reasoning_parser=True)
        app = server.setup_webserver()
        client = TestClient(app)

        monkeypatch.setattr("responses_api_models.inference_provider.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("responses_api_models.inference_provider.app.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        mock_data = _mock_chat_response(content="<think>Let me reason about this...</think>The answer is 42.")

        async def mock_create_chat(**kwargs):
            return mock_data

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        response = client.post("/v1/responses", json={"input": "What is 6*7?"})
        assert response.status_code == 200
        data = response.json()

        reasoning_items = [o for o in data["output"] if o["type"] == "reasoning"]
        message_items = [o for o in data["output"] if o["type"] == "message"]

        assert len(reasoning_items) == 1
        assert reasoning_items[0]["summary"][0]["text"] == "Let me reason about this..."
        assert len(message_items) == 1
        assert message_items[0]["content"][0]["text"] == "The answer is 42."

    async def test_responses_without_reasoning_parser_keeps_think_tags_inline(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server(uses_reasoning_parser=False)
        app = server.setup_webserver()
        client = TestClient(app)

        monkeypatch.setattr("responses_api_models.inference_provider.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("responses_api_models.inference_provider.app.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        mock_data = _mock_chat_response(content="<think>Let me reason about this...</think>The answer is 42.")

        async def mock_create_chat(**kwargs):
            return mock_data

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        response = client.post("/v1/responses", json={"input": "What is 6*7?"})
        assert response.status_code == 200
        data = response.json()

        reasoning_items = [o for o in data["output"] if o["type"] == "reasoning"]
        message_items = [o for o in data["output"] if o["type"] == "message"]

        assert len(reasoning_items) == 0
        assert len(message_items) == 1
        assert message_items[0]["content"][0]["text"] == "<think>Let me reason about this...</think>The answer is 42."

    async def test_responses_with_usage(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        monkeypatch.setattr("responses_api_models.inference_provider.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("responses_api_models.inference_provider.app.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        mock_data = _mock_chat_response(content="Hi!")
        mock_data["usage"] = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

        async def mock_create_chat(**kwargs):
            return mock_data

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        response = client.post("/v1/responses", json={"input": "hello"})
        data = response.json()

        assert data["usage"]["input_tokens"] == 10
        assert data["usage"]["output_tokens"] == 5
        assert data["usage"]["total_tokens"] == 15

    async def test_responses_incomplete_max_tokens(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        monkeypatch.setattr("responses_api_models.inference_provider.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("responses_api_models.inference_provider.app.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        mock_data = _mock_chat_response(content="Truncated output...", finish_reason="length")

        async def mock_create_chat(**kwargs):
            return mock_data

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        response = client.post("/v1/responses", json={"input": "Write a long essay"})
        data = response.json()

        assert data["incomplete_details"] == {"reason": "max_output_tokens"}

    async def test_responses_incomplete_content_filter(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        monkeypatch.setattr("responses_api_models.inference_provider.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("responses_api_models.inference_provider.app.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        mock_data = _mock_chat_response(content="", finish_reason="content_filter")

        async def mock_create_chat(**kwargs):
            return mock_data

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        response = client.post("/v1/responses", json={"input": "test"})
        data = response.json()

        assert data["incomplete_details"] == {"reason": "content_filter"}

    async def test_responses_converts_string_input(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        called_kwargs = {}

        async def mock_create_chat(**kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            return _mock_chat_response()

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        monkeypatch.setattr("responses_api_models.inference_provider.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("responses_api_models.inference_provider.app.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        client.post("/v1/responses", json={"input": "hello world"})

        # String input should be converted to a user message
        assert called_kwargs["messages"][0]["role"] == "user"

    async def test_responses_multistep_conversation(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server()
        app = server.setup_webserver()
        client = TestClient(app)

        monkeypatch.setattr("responses_api_models.inference_provider.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("responses_api_models.inference_provider.app.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("nemo_gym.responses_converter.uuid4", lambda: FakeUUID())

        called_kwargs = {}

        async def mock_create_chat(**kwargs):
            nonlocal called_kwargs
            called_kwargs = kwargs
            return _mock_chat_response(content="The delivery date is tomorrow.")

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

        response = client.post(
            "/v1/responses",
            json={
                "input": [
                    {"role": "user", "content": [{"type": "input_text", "text": "When will my order arrive?"}]},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_delivery_date",
                        "arguments": '{"order_id": "123"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "2026-05-11",
                    },
                ],
            },
        )
        assert response.status_code == 200

        # Verify the messages were properly converted
        messages = called_kwargs["messages"]
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"][0]["function"]["name"] == "get_delivery_date"
        assert messages[2]["role"] == "tool"
        assert messages[2]["content"] == "2026-05-11"


class TestConcurrency:
    async def test_semaphore_limits_concurrency(self, monkeypatch: MonkeyPatch) -> None:
        server = _make_server(num_concurrent_requests=2)
        assert server._semaphore._value == 2
