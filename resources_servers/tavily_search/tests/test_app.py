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
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.tavily_search.app import (
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
    TavilySearchVerifyRequest,
)


class TestApp:
    @fixture
    def config(self) -> TavilySearchResourcesServerConfig:
        return TavilySearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            tavily_api_key="test_api_key",
            exclude_domains_file_path="tests/dummy_exclude_domains_file.json",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

    @fixture
    def server(self, config: TavilySearchResourcesServerConfig) -> TavilySearchResourcesServer:
        return TavilySearchResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )

    def _msg(self, text: str) -> NeMoGymResponseOutputMessage:
        """Helper to create a NeMoGymResponseOutputMessage."""
        return NeMoGymResponseOutputMessage(
            id="msg_id",
            content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        )

    def _create_judge_response(self, text: str) -> dict[str, Any]:
        """Helper to create a mock judge NeMoGymResponse dict."""
        return NeMoGymResponse(
            id="judge_resp",
            created_at=0.0,
            model="judge_model",
            object="response",
            output=[self._msg(text)],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump()

    def _create_model_response(self, text: str) -> NeMoGymResponse:
        """Helper to create a model NeMoGymResponse."""
        return NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="test_model",
            object="response",
            output=[self._msg(text)],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    def test_sanity(self, config: TavilySearchResourcesServerConfig) -> None:
        TavilySearchResourcesServer(
            config=config, server_client=MagicMock(spec=ServerClient)
        )

    def test_postprocess_search_results(self, server: TavilySearchResourcesServer) -> None:
        """Test that _postprocess_search_results correctly formats Tavily search results."""
        raw_results = {
            "results": [
                {
                    "url": "https://example.com/page1",
                    "title": "Example Page 1",
                    "content": "This is the content of page 1",
                    "score": 0.95,
                    "raw_content": "raw content",
                },
                {
                    "url": "https://example.com/page2",
                    "title": "Example Page 2",
                    "content": "This is the content of page 2",
                    "score": 0.85,
                },
            ]
        }

        formatted_results = server._postprocess_search_results(raw_results)

        assert len(formatted_results) == 2
        assert formatted_results[0] == {
            "url": "https://example.com/page1",
            "title": "Example Page 1",
            "content": "This is the content of page 1",
        }
        assert formatted_results[1] == {
            "url": "https://example.com/page2",
            "title": "Example Page 2",
            "content": "This is the content of page 2",
        }
        assert "score" not in formatted_results[0]
        assert "raw_content" not in formatted_results[0]

    async def test_web_search(self, server: TavilySearchResourcesServer) -> None:
        """Test the web_search endpoint with mocked Tavily client."""
        mock_tavily_response = {
            "results": [
                {
                    "url": "https://nvidia.com/docs",
                    "title": "NVIDIA Documentation",
                    "content": "Official NVIDIA documentation for developers.",
                    "score": 0.99,
                },
            ]
        }
        server._tavily = MagicMock()
        server._tavily.search.return_value = mock_tavily_response

        request = TavilySearchRequest(query="NVIDIA GPU programming")
        response = await server.web_search(request)

        server._tavily.search.assert_called_once_with(
            "NVIDIA GPU programming",
            num_results=server.NUM_RESULTS,
            exclude_domains=server._exclude_domains,
        )
        results = json.loads(response.results_string)
        assert len(results) == 1
        assert results[0]["url"] == "https://nvidia.com/docs"
        assert "score" not in results[0]

    async def test_verify_correct_answer(self, config: TavilySearchResourcesServerConfig) -> None:
        """Test verify endpoint when judge determines answer is correct."""
        server_client = MagicMock(spec=ServerClient)
        server = TavilySearchResourcesServer(config=config, server_client=server_client)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(return_value=self._create_judge_response("correct: yes"))
        server_client.post = AsyncMock(return_value=post_mock)

        req = TavilySearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_model_response("The capital of France is Paris."),
            ground_truth="Paris",
            question="What is the capital of France?",
        )

        res = await server.verify(req)

        assert res.reward == approx(1.0)
        assert res.extracted_final_answer == "yes"
        assert server_client.post.call_count == 1

    async def test_verify_incorrect_answer(self, config: TavilySearchResourcesServerConfig) -> None:
        """Test verify endpoint when judge determines answer is incorrect."""
        server_client = MagicMock(spec=ServerClient)
        server = TavilySearchResourcesServer(config=config, server_client=server_client)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(return_value=self._create_judge_response("correct: no"))
        server_client.post = AsyncMock(return_value=post_mock)

        req = TavilySearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_model_response("The capital of France is London."),
            ground_truth="Paris",
            question="What is the capital of France?",
        )

        res = await server.verify(req)

        assert res.reward == approx(0.0)
        assert res.extracted_final_answer == "no"
        assert server_client.post.call_count == 1
