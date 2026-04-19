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
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pytest import approx, fixture

from nemo_gym.server_utils import SESSION_ID_KEY


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(os.path.dirname(_TEST_DIR), "tests", "dummy_exclude_domains_file.json")

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    BrowseRequest,
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
            tavily_api_key="test_api_key",  # pragma: allowlist secret
            exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

    @fixture
    def server(self, config: TavilySearchResourcesServerConfig) -> TavilySearchResourcesServer:
        return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_dummy_request(self) -> MagicMock:
        request_mock = MagicMock()
        request_mock.session = {SESSION_ID_KEY: "test_session_id"}
        return request_mock

    def _msg(self, text: str) -> NeMoGymResponseOutputMessage:
        return NeMoGymResponseOutputMessage(
            id="msg_id",
            content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        )

    def _create_judge_response(self, text: str) -> dict[str, Any]:
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

    # ---- Sanity ----

    def test_sanity(self, config: TavilySearchResourcesServerConfig) -> None:
        TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    # ---- _parse_judge ----

    def test_parse_judge_correct(self, server: TavilySearchResourcesServer) -> None:
        text = "extracted_final_answer: Paris\nreasoning: Matches.\ncorrect: yes"
        is_correct, extracted, parsed_ok = server._parse_judge(text)
        assert parsed_ok is True
        assert is_correct is True
        assert extracted == "Paris"

    def test_parse_judge_incorrect(self, server: TavilySearchResourcesServer) -> None:
        text = "extracted_final_answer: London\nreasoning: Does not match.\ncorrect: no"
        is_correct, extracted, parsed_ok = server._parse_judge(text)
        assert parsed_ok is True
        assert is_correct is False
        assert extracted == "London"

    def test_parse_judge_no_match(self, server: TavilySearchResourcesServer) -> None:
        text = "The model gave some random output without the expected format."
        is_correct, extracted, parsed_ok = server._parse_judge(text)
        assert parsed_ok is False

    def test_parse_judge_uses_last_match(self, server: TavilySearchResourcesServer) -> None:
        """When multiple correct: yes/no appear, last one wins."""
        text = "correct: yes\nextracted_final_answer: Wrong\ncorrect: no\nextracted_final_answer: Right"
        is_correct, extracted, parsed_ok = server._parse_judge(text)
        assert parsed_ok is True
        assert is_correct is False
        assert extracted == "Right"

    # ---- _is_url_excluded ----

    def test_is_url_excluded(self, server: TavilySearchResourcesServer) -> None:
        assert server._is_url_excluded("https://blacklisteddomain.com/page") is True
        assert server._is_url_excluded("https://sub.blacklisteddomain.com/page") is True
        assert server._is_url_excluded("https://example.com/page") is False

    # ---- _postprocess_search_results ----

    def test_postprocess_search_results(self, server: TavilySearchResourcesServer) -> None:
        results = {
            "results": [
                {
                    "url": "https://example.com/page1",
                    "title": "Example Page 1",
                    "content": "Content of page 1",
                    "raw_content": "Raw content of page 1",
                    "score": 0.95,
                },
                {
                    "url": "https://example.com/page2",
                    "title": "Example Page 2",
                    "content": "Content of page 2",
                    "score": 0.85,
                },
            ]
        }
        formatted = server._postprocess_search_results("test query", results, max_length=50000)
        assert "[Search Query]: test query" in formatted
        assert "Example Page 1" in formatted
        assert "https://example.com/page1" in formatted
        assert "Raw content of page 1" in formatted
        assert "0.95" not in formatted

    def test_postprocess_search_results_truncates(self, server: TavilySearchResourcesServer) -> None:
        long_content = "x" * 10000
        results = {
            "results": [
                {"url": "https://a.com", "title": "A", "content": "short", "raw_content": long_content},
                {"url": "https://b.com", "title": "B", "content": "also short"},
            ]
        }
        formatted = server._postprocess_search_results("q", results, max_length=50000)
        assert "[truncated]" in formatted

    # ---- search ----

    async def test_search(self, server: TavilySearchResourcesServer) -> None:
        mock_tavily_response = {
            "results": [
                {
                    "url": "https://nvidia.com",
                    "title": "NVIDIA",
                    "content": "NVIDIA content",
                    "raw_content": "NVIDIA raw",
                    "score": 0.99,
                },
            ]
        }
        mock_client = MagicMock()
        mock_client.search = AsyncMock(return_value=mock_tavily_response)
        server._async_tavily_clients = [mock_client]

        request = TavilySearchRequest(queries=["NVIDIA GPU"])
        response = await server.search(self._create_dummy_request(), request)

        mock_client.search.assert_called_once()
        assert "NVIDIA" in response.results_string

    async def test_search_empty_queries(self, server: TavilySearchResourcesServer) -> None:
        request = TavilySearchRequest(queries=[])
        response = await server.search(self._create_dummy_request(), request)
        assert response.results_string == "Query is none or empty"

    async def test_search_none_queries(self, server: TavilySearchResourcesServer) -> None:
        request = TavilySearchRequest(queries=None)
        response = await server.search(self._create_dummy_request(), request)
        assert response.results_string == "Query is none or empty"

    # ---- browse ----

    async def test_browse_excluded_urls(self, server: TavilySearchResourcesServer) -> None:
        request = BrowseRequest(urls=["https://blacklisteddomain.com/page"])
        response = await server.browse(self._create_dummy_request(), request)
        assert "Error: no URLs provided." in response.results_string

    async def test_browse_extract_failure(self, server: TavilySearchResourcesServer) -> None:
        mock_client = MagicMock()
        mock_client.extract = AsyncMock(side_effect=Exception("API error"))
        server._async_tavily_clients = [mock_client]

        request = BrowseRequest(urls=["https://example.com"])
        response = await server.browse(self._create_dummy_request(), request)
        assert "Failed to extract content" in response.results_string

    # ---- verify ----

    async def test_verify_correct_answer(self, config: TavilySearchResourcesServerConfig) -> None:
        server_client = MagicMock(spec=ServerClient)
        server = TavilySearchResourcesServer(config=config, server_client=server_client)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(
            return_value=self._create_judge_response("extracted_final_answer: Paris\ncorrect: yes")
        )
        server_client.post = AsyncMock(return_value=post_mock)

        req = TavilySearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_model_response("The capital of France is Paris."),
            ground_truth="Paris",
            question="What is the capital of France?",
        )
        res = await server.verify(self._create_dummy_request(), req)

        assert res.reward == approx(1.0)
        assert res.extracted_final_answer == "Paris"

    async def test_verify_incorrect_answer(self, config: TavilySearchResourcesServerConfig) -> None:
        server_client = MagicMock(spec=ServerClient)
        server = TavilySearchResourcesServer(config=config, server_client=server_client)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(
            return_value=self._create_judge_response("extracted_final_answer: London\ncorrect: no")
        )
        server_client.post = AsyncMock(return_value=post_mock)

        req = TavilySearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_model_response("The capital of France is London."),
            ground_truth="Paris",
            question="What is the capital of France?",
        )
        res = await server.verify(self._create_dummy_request(), req)

        assert res.reward == approx(0.0)
        assert res.extracted_final_answer == "London"

    # ---- _verify_answer_with_regex ----

    def test_verify_answer_with_regex_correct(self, server: TavilySearchResourcesServer) -> None:
        result = server._verify_answer_with_regex("Paris", "Answer: Paris Confidence: 95%")
        assert result.reward == approx(1.0)
        assert result.extracted_final_answer == "Paris"

    def test_verify_answer_with_regex_incorrect(self, server: TavilySearchResourcesServer) -> None:
        result = server._verify_answer_with_regex("Paris", "Answer: London Confidence: 80%")
        assert result.reward == approx(0.0)
        assert result.extracted_final_answer == "London"

    def test_verify_answer_with_regex_no_answer(self, server: TavilySearchResourcesServer) -> None:
        result = server._verify_answer_with_regex("Paris", "I don't know.")
        assert result.reward == approx(0.0)
        assert result.extracted_final_answer == ""
