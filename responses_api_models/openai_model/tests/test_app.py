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
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.server_utils import ServerClient
from responses_api_models.openai_model.app import (
    NeMoGymAsyncOpenAI,
    SimpleModelServer,
    SimpleModelServerConfig,
)


class TestApp:
    def _setup_server(self, max_concurrent_requests=None, drop_input_reasoning_items=False):
        config = SimpleModelServerConfig(
            host="0.0.0.0",
            port=8081,
            openai_base_url="https://api.openai.com/v1",
            openai_api_key="dummy_key",  # pragma: allowlist secret
            openai_model="dummy_model",
            entrypoint="",
            name="",
            max_concurrent_requests=max_concurrent_requests,
            drop_input_reasoning_items=drop_input_reasoning_items,
        )
        return SimpleModelServer(config=config, server_client=MagicMock(spec=ServerClient))

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

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_chat_completion = AsyncMock(side_effect=mock_create_chat)

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

        server._client.create_chat_completion.assert_any_await(
            messages=[{"role": "user", "content": "hi"}],
            model="dummy_model",
        )

    async def test_responses(self, monkeypatch: MonkeyPatch) -> None:
        server = self._setup_server()
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        called_args_response = {}

        async def mock_create_response(**kwargs):
            nonlocal called_args_response
            called_args_response = kwargs
            return mock_response_data

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(side_effect=mock_create_response)

        # No model provided should use the one from the config
        res_no_model = client.post("/v1/responses", json={"input": "hello"})
        assert res_no_model.status_code == 200
        assert called_args_response.get("model") == "dummy_model"

        # model provided should override config
        res_with_model = client.post("/v1/responses", json={"input": "hello", "model": "override_model"})
        assert res_with_model.status_code == 200
        assert called_args_response.get("model") == "dummy_model"

        server._client.create_response.assert_any_await(input="hello", model="dummy_model")

    async def test_responses_parses_hosted_mcp_call(self, monkeypatch: MonkeyPatch) -> None:
        """A server-side ``mcp_call`` output item must validate (200), not 500.

        NVIDIA-hosted gpt-oss surfaces its built-in python tool as an ``mcp_call``;
        before it was in the response schema this returned a 500 that aborted the
        whole rollout collection.
        """
        server = self._setup_server()
        client = TestClient(server.setup_webserver())

        mock_response_data = {
            "id": "resp_mcp",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "type": "mcp_call",
                    "id": "mcp_1",
                    "name": "python",
                    "server_label": "exec",
                    "arguments": '{"code": "print(42)"}',
                    "output": "42\n",
                    "status": "completed",
                },
                {
                    "id": "msg_1",
                    "content": [{"annotations": [], "text": "(Answer: 42)", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(return_value=mock_response_data)

        res = client.post("/v1/responses", json={"input": "compute qed"})
        assert res.status_code == 200
        assert res.json()["output"][0]["type"] == "mcp_call"

    async def test_drop_input_reasoning_items_strips_reasoning(self, monkeypatch: MonkeyPatch) -> None:
        server = self._setup_server(drop_input_reasoning_items=True)
        client = TestClient(server.setup_webserver())

        called_args = {}

        async def mock_create_response(**kwargs):
            nonlocal called_args
            called_args = kwargs
            return {
                "id": "resp_1",
                "created_at": 0.0,
                "model": "dummy_model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }

        server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
        server._client.create_response = AsyncMock(side_effect=mock_create_response)

        res = client.post(
            "/v1/responses",
            json={
                "input": [
                    {"type": "reasoning", "id": "r1", "summary": []},
                    {"role": "user", "content": "hi", "type": "message"},
                ]
            },
        )
        assert res.status_code == 200
        sent_types = [item.get("type") for item in called_args["input"]]
        assert "reasoning" not in sent_types
        assert "message" in sent_types

    def test_semaphore_disabled_by_default(self) -> None:
        server = self._setup_server()
        assert isinstance(server._semaphore, type(nullcontext()))

    @pytest.mark.asyncio
    async def test_semaphore_caps_concurrency(self) -> None:
        server = self._setup_server(max_concurrent_requests=2)
        assert isinstance(server._semaphore, asyncio.Semaphore)

        in_flight = 0
        peak = 0

        async def worker() -> None:
            nonlocal in_flight, peak
            async with server._semaphore:
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1

        await asyncio.gather(*(worker() for _ in range(8)))
        assert peak == 2
