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
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import AsyncAzureOpenAI
from pytest import MonkeyPatch

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionMessage,
    NeMoGymChoice,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
)
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
    @pytest.mark.parametrize(
        ("provider_tier", "request_tier", "expected_tier"),
        [
            ("on-demand", None, "on-demand"),
            ("on-demand", "default", "on-demand"),
            (None, "priority", "priority"),
            (None, None, None),
        ],
    )
    async def test_responses_service_tier(
        self, provider_tier: str | None, request_tier: str | None, expected_tier: str | None
    ) -> None:
        server = self._setup_server()
        client = TestClient(server.setup_webserver())
        completion = NeMoGymChatCompletion(
            id="chatcmpl-tier",
            created=FIXED_TIME,
            model="dummy_model",
            object="chat.completion",
            service_tier=provider_tier,
            choices=[
                NeMoGymChoice(
                    index=0,
                    finish_reason="stop",
                    message=NeMoGymChatCompletionMessage(role="assistant", content="hi"),
                )
            ],
        )
        server._client = MagicMock(spec=AsyncAzureOpenAI)
        server._client.chat.completions.create = AsyncMock(return_value=completion)

        response = client.post("/v1/responses", json={"input": "hello", "service_tier": request_tier})

        assert response.status_code == 200
        assert response.json()["service_tier"] == expected_tier

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
        return AzureOpenAIModelServer(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))

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
        monkeypatch.setattr("nemo_gym.models.chat_completions_converter.uuid4", lambda: FakeUUID())

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


class TestQueuedRequests:
    """Requests that wait for the concurrency permit while a streaming keepalive interval elapses."""

    def _setup(self) -> tuple[AzureOpenAIModelServer, asyncio.Event, dict]:
        config = AzureOpenAIModelServerConfig(
            host="0.0.0.0",
            port=8081,
            openai_base_url="https://prod.api.nvidia.com/llm/v1/azure",
            openai_api_key="dummy_key",  # pragma: allowlist secret
            openai_model="dummy_model",
            default_query={"api-version": "dummy_version"},
            num_concurrent_requests=1,
            stream_keepalive_s=0.02,
            entrypoint="",
            name="",
        )
        server = AzureOpenAIModelServer(
            config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={})
        )
        release = asyncio.Event()
        calls = {"started": 0, "in_flight": 0, "max_in_flight": 0}

        async def create(**kwargs):
            calls["started"] += 1
            calls["in_flight"] += 1
            calls["max_in_flight"] = max(calls["max_in_flight"], calls["in_flight"])
            try:
                await release.wait()
                return NeMoGymChatCompletion(
                    id="chatcmpl-queued",
                    created=FIXED_TIME,
                    model="dummy_model",
                    object="chat.completion",
                    choices=[
                        NeMoGymChoice(
                            index=0,
                            finish_reason="stop",
                            message=NeMoGymChatCompletionMessage(role="assistant", content="hi"),
                        )
                    ],
                )
            finally:
                calls["in_flight"] -= 1

        server._client = MagicMock(spec=AsyncAzureOpenAI)
        server._client.chat.completions.create = AsyncMock(side_effect=create)
        return server, release, calls

    @staticmethod
    async def _wait_for_first_call(calls: dict) -> None:
        async with asyncio.timeout(5):
            while calls["started"] == 0:
                await asyncio.sleep(0.005)

    @pytest.mark.parametrize(
        "tool_choice",
        [None, {"type": "file_search"}, {"type": "allowed_tools", "mode": "auto", "tools": [{"type": "file_search"}]}],
        ids=["tools", "tool_choice", "allowed_tools"],
    )
    async def test_hosted_tool_request_is_refused_with_422_while_the_permit_is_held(
        self, tool_choice: dict | None
    ) -> None:
        server, release, calls = self._setup()
        body = {"input": "q", "stream": True, "tools": [{"type": "file_search", "vector_store_ids": ["vs_1"]}]}
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        transport = httpx.ASGITransport(app=server.setup_webserver())
        async with httpx.AsyncClient(transport=transport, base_url="http://azure") as client:
            holder = asyncio.create_task(client.post("/v1/responses", json={"input": "hello"}))
            await self._wait_for_first_call(calls)

            refused = await asyncio.wait_for(client.post("/v1/responses", json=body), timeout=5)

            assert refused.status_code == 422
            assert "file_search" in refused.json()["detail"]
            assert not holder.done(), "the refusal came before the held call was released"
            release.set()
            assert (await holder).status_code == 200
        assert calls["started"] == 1, "the refused request never reached the backend"

    async def test_function_tool_request_waits_with_keepalives_and_completes_after_release(self) -> None:
        server, release, calls = self._setup()
        body = {
            "input": "q",
            "stream": True,
            "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}],
        }
        transport = httpx.ASGITransport(app=server.setup_webserver())
        async with httpx.AsyncClient(transport=transport, base_url="http://azure") as client:
            holder = asyncio.create_task(client.post("/v1/responses", json={"input": "hello"}))
            await self._wait_for_first_call(calls)
            queued = asyncio.create_task(client.post("/v1/responses", json=body))
            await asyncio.sleep(0.2)
            assert not queued.done(), "the request waits for the permit"
            assert calls["started"] == 1
            release.set()
            first, streamed = await asyncio.wait_for(asyncio.gather(holder, queued), timeout=5)

        assert first.status_code == 200
        assert streamed.status_code == 200
        events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ")]
        assert events[0]["type"] == "response.created"
        assert events[1]["type"] == "response.in_progress", "keepalives opened the stream while it waited"
        assert events[-1]["type"] == "response.completed"
        assert calls == {"started": 2, "in_flight": 0, "max_in_flight": 1}
