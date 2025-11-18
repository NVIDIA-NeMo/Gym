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
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from openai import AsyncAzureOpenAI
from pytest import MonkeyPatch

from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_models.azure_openai_model.app import (
    AzureOpenAIModelServer,
    AzureOpenAIModelServerConfig,
)


# Used for mocking created_at timestamp generation
FIXED_TIME = 1691418000
FIXED_UUID = "123"


class FakeUUID:
    """Used for mocking UUIDs"""

    hex = FIXED_UUID


class TestApp:
    def _setup_server(self, monkeypatch=None):
        config = AzureOpenAIModelServerConfig(
            host="0.0.0.0",
            port=8081,
            openai_base_url="https://prod.api.nvidia.com/llm/v1/azure",
            openai_api_key="dummy_key",  # pragma: allowlist secret
            openai_model="dummy_model",
            default_query={"api-version": "dummy_version"},
            num_concurrent_requests=8,
            entrypoint="",
            name="",
        )
        return AzureOpenAIModelServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_sanity(self) -> None:
        self._setup_server()

    async def test_chat_completions(self, monkeypatch: MonkeyPatch) -> None:
        server = self._setup_server()
        app = server.setup_webserver()
        client = TestClient(app)

        mock_chat_data = {
            "id": "chatcmpl-BzRdCFjIEIp59xXLBNYjdPPrcpDaa",  # pragma: allowlist secret
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "Hello! How can I help you today?",
                        "role": "assistant",
                    },
                }
            ],
            "created": 1753983922,
            "model": "dummy_model",
            "object": "chat.completion",
        }

        called_args_chat = {}

        async def mock_create_chat(**kwargs):
            nonlocal called_args_chat
            called_args_chat = kwargs
            return mock_chat_data

        server._client = MagicMock(spec=AsyncAzureOpenAI)
        server._client.chat.completions.create = AsyncMock(side_effect=mock_create_chat)

        chat_no_model = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert chat_no_model.status_code == 200
        assert called_args_chat.get("model") == "dummy_model"

        chat_with_model = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "model": "override_model",
            },
        )
        assert chat_with_model.status_code == 200
        assert called_args_chat.get("model") == "dummy_model"

        server._client.chat.completions.create.assert_any_await(
            messages=[{"role": "user", "content": "hi"}],
            model="dummy_model",
        )

    async def test_responses(self, monkeypatch: MonkeyPatch) -> None:
        server = self._setup_server(monkeypatch)
        app = server.setup_webserver()
        client = TestClient(app)

        monkeypatch.setattr("responses_api_models.azure_openai_model.app.uuid4", lambda: FakeUUID())
        monkeypatch.setattr("responses_api_models.azure_openai_model.app.time", lambda: FIXED_TIME)
        monkeypatch.setattr("responses_api_models.vllm_model.app.uuid4", lambda: FakeUUID())

        mock_response_data = NeMoGymChatCompletion(
            id="chtcmpl-123",
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(role="assistant", content="Hello! How can I help you today?"),
                )
            ],
            created=FIXED_TIME,
            model="dummy_model",
            object="chat.completion",
        )

        # Expected response
        expected_response = NeMoGymResponse(
            id="resp_123",
            created_at=FIXED_TIME,
            model="dummy_model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id="msg_123",
                    content=[
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                        }
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                )
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input="hello")

        # Mock the Azure OpenAI client directly since responses() calls it directly
        server._client = MagicMock(spec=AsyncAzureOpenAI)
        server._client.chat.completions.create = AsyncMock(return_value=mock_response_data)

        response = client.post(
            "/v1/responses",
            json=responses_create_params.model_dump(exclude_unset=True, mode="json"),
        )
        assert response.status_code == 200
        assert expected_response.model_dump() == response.json()
