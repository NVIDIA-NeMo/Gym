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
from unittest.mock import AsyncMock, MagicMock, call

import orjson
from fastapi.testclient import TestClient
from pytest import approx, fixture, mark, raises

from nemo_gym.client import SESSION_ID_KEY
from resources_servers.tavily_search import app as tavily_search_app


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.verification import JudgeError
from resources_servers.tavily_search.app import (
    _JUDGE_VERDICT_RE,
    FindInPageRequest,
    ScrollPageRequest,
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
            exclude_domains_file_path=os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json"),
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

    @fixture
    def server(self, config: TavilySearchResourcesServerConfig) -> TavilySearchResourcesServer:
        return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_dummy_request(self) -> MagicMock:
        request_mock = MagicMock()
        request_mock.session = {SESSION_ID_KEY: "abcd"}
        return request_mock

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

    # ---- Sanity ----

    def test_sanity(self, config: TavilySearchResourcesServerConfig) -> None:
        TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    # ---- _postprocess_search_results ----

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

        # Returns a list of formatted strings
        assert isinstance(formatted_results, list)
        joined = "".join(formatted_results)
        assert "Search Results" in joined
        assert "[1] Example Page 1 (example.com)" in joined
        assert "[2] Example Page 2 (example.com)" in joined
        assert "URL: https://example.com/page1" in joined
        assert "URL: https://example.com/page2" in joined
        assert "This is the content of page 1" in joined
        assert "This is the content of page 2" in joined
        # score and raw_content should NOT appear
        assert "0.95" not in joined
        assert "raw content" not in joined

    def test_postprocess_search_results_with_answer(self, server: TavilySearchResourcesServer) -> None:
        """Aggregate answers cannot be attributed to allowed URLs and must be discarded."""
        raw_results = {
            "answer": "The capital of France is Paris.",
            "results": [
                {"url": "https://example.com", "title": "T", "content": "C"},
            ],
        }
        formatted = server._postprocess_search_results(raw_results)
        joined = "".join(formatted)
        assert "Search Answer" not in joined
        assert "The capital of France is Paris." not in joined
        assert "[1]" in joined

    # ---- web_search ----

    async def test_web_search(self, server: TavilySearchResourcesServer) -> None:
        """Test the web_search endpoint with mocked _tavily_post."""
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
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=mock_tavily_response)
        server._async_tavily_clients = [mock_backend]

        request = TavilySearchRequest(query="NVIDIA GPU programming")
        response = await server.web_search(self._create_dummy_request(), request)

        mock_backend.search.assert_called_once()
        actual_call_args = mock_backend.search.call_args
        expected_call_args = call(
            "NVIDIA GPU programming",
            max_results=10,
            exclude_domains=["blacklisteddomain.com"],
            search_depth="advanced",
            include_answer=False,
            include_raw_content=False,
        )
        assert expected_call_args == actual_call_args

        # Response is now a formatted string, not JSON list
        assert "NVIDIA Documentation" in response.results_string
        assert "nvidia.com" in response.results_string

    async def test_web_search_none_query(self, server: TavilySearchResourcesServer) -> None:
        """Test web_search with None query returns error message."""
        request = TavilySearchRequest(query=None)
        response = await server.web_search(self._create_dummy_request(), request)
        assert response.results_string == "Query is none"

    async def test_web_search_long_query(self, server: TavilySearchResourcesServer) -> None:
        """Test web_search with overly long query returns error message."""
        request = TavilySearchRequest(query="x" * 401)
        response = await server.web_search(self._create_dummy_request(), request)
        assert response.results_string == "Query is too long"

    # ---- find_in_page ----

    async def test_find_in_page_none_url(self, server: TavilySearchResourcesServer) -> None:
        """Test find_in_page with None URL returns error."""
        request = FindInPageRequest(url=None, query="test")
        response = await server.find_in_page(self._create_dummy_request(), request)
        assert response.results_string == "URL is none"

    async def test_find_in_page_none_query(self, server: TavilySearchResourcesServer) -> None:
        """Test find_in_page with None query returns error."""
        request = FindInPageRequest(url="https://example.com", query=None)
        response = await server.find_in_page(self._create_dummy_request(), request)
        assert response.results_string == "Query is none"

    async def test_find_in_page_excluded_domain(self, server: TavilySearchResourcesServer) -> None:
        """Test find_in_page with excluded domain returns error."""
        request = FindInPageRequest(url="https://blacklisteddomain.com/page", query="test")
        response = await server.find_in_page(self._create_dummy_request(), request)
        assert response.results_string == "URL is in excluded domains"

    # ---- scroll_page ----

    async def test_scroll_page_none_url(self, server: TavilySearchResourcesServer) -> None:
        """Test scroll_page with None URL."""
        request = ScrollPageRequest(url=None)
        response = await server.scroll_page(self._create_dummy_request(), request)
        assert response.results_string == "URL is none"
        assert response.total_words == 0

    async def test_scroll_page_excluded_domain(self, server: TavilySearchResourcesServer) -> None:
        """Test scroll_page with excluded domain."""
        request = ScrollPageRequest(url="https://blacklisteddomain.com/page")
        response = await server.scroll_page(self._create_dummy_request(), request)
        assert response.results_string == "URL is in excluded domains"
        assert response.total_words == 0

    # ---- Utility functions ----

    def test_extract_domain(self, server: TavilySearchResourcesServer) -> None:
        assert server._extract_domain("https://en.wikipedia.org/wiki/Python") == "en.wikipedia.org"
        assert server._extract_domain("http://example.com/path") == "example.com"

    def test_clean_text(self, server: TavilySearchResourcesServer) -> None:
        text = "Hello [edit] world\n[Jump to content]\nContent here\u200b"
        cleaned = server._clean_text(text)
        assert "[edit]" not in cleaned
        assert "[Jump to content]" not in cleaned
        assert "\u200b" not in cleaned
        assert "Hello" in cleaned
        assert "Content here" in cleaned

    def test_add_line_numbers(self, server: TavilySearchResourcesServer) -> None:
        text = "first\nsecond\nthird"
        result = server._add_line_numbers(text)
        assert result == "L0: first\nL1: second\nL2: third"

    def test_truncate_text_short(self, server: TavilySearchResourcesServer) -> None:
        text = "short text"
        result, was_truncated = server._truncate_text(text)
        assert result == "short text"
        assert was_truncated is False

    def test_truncate_text_long(self, server: TavilySearchResourcesServer) -> None:
        text = "\n".join([f"Line {i}" for i in range(500)])
        result, was_truncated = server._truncate_text(text, max_chars=100)
        assert was_truncated is True
        assert len(result) <= 100
        # Should snap to last newline boundary
        assert result.endswith(result.split("\n")[-1])

    def test_is_url_excluded(self, server: TavilySearchResourcesServer) -> None:
        assert server._is_url_excluded("https://blacklisteddomain.com/page") is True
        assert server._is_url_excluded("https://sub.blacklisteddomain.com/page") is True
        assert server._is_url_excluded("https://example.com/page") is False

    # ---- verify (kept from original) ----

    async def test_verify_correct_answer(self, config: TavilySearchResourcesServerConfig) -> None:
        """Test verify endpoint when judge determines answer is correct."""
        server_client = MagicMock(spec=ServerClient)
        server = TavilySearchResourcesServer(config=config, server_client=server_client)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(return_value=self._create_judge_response("correct: yes"))
        post_mock.read = AsyncMock(return_value=orjson.dumps(post_mock.json.return_value))
        server_client.post = AsyncMock(return_value=post_mock)

        req = TavilySearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_model_response("The capital of France is Paris."),
            ground_truth="Paris",
            question="What is the capital of France?",
        )

        res = await server.verify(self._create_dummy_request(), req)

        assert res.reward == approx(1.0)
        assert res.extracted_final_answer == "yes"
        assert server_client.post.call_count == 1

    async def test_verify_incorrect_answer(self, config: TavilySearchResourcesServerConfig) -> None:
        """Test verify endpoint when judge determines answer is incorrect."""
        server_client = MagicMock(spec=ServerClient)
        server = TavilySearchResourcesServer(config=config, server_client=server_client)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(return_value=self._create_judge_response("correct: no"))
        post_mock.read = AsyncMock(return_value=orjson.dumps(post_mock.json.return_value))
        server_client.post = AsyncMock(return_value=post_mock)

        req = TavilySearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_model_response("The capital of France is London."),
            ground_truth="Paris",
            question="What is the capital of France?",
        )

        res = await server.verify(self._create_dummy_request(), req)

        assert res.reward == approx(0.0)
        assert res.extracted_final_answer == "no"
        assert server_client.post.call_count == 1

    async def test_api_key_rotation_sanity(self, server: TavilySearchResourcesServer) -> None:
        """Test multiple calls to Tavily rotate through API keys"""
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

        mock_backend1 = MagicMock()
        mock_backend1.search = AsyncMock(return_value=mock_tavily_response)

        mock_backend2 = MagicMock()
        mock_backend2.search = AsyncMock(return_value=mock_tavily_response)

        mock_backend3 = MagicMock()
        mock_backend3.search = AsyncMock(return_value=mock_tavily_response)

        server._async_tavily_clients = [mock_backend1, mock_backend2, mock_backend3]

        request = TavilySearchRequest(query="NVIDIA GPU programming")

        await server.web_search(self._create_dummy_request(), request)
        assert mock_backend1.search.call_count == 1

        await server.web_search(self._create_dummy_request(), request)
        assert mock_backend2.search.call_count == 1

        await server.web_search(self._create_dummy_request(), request)
        assert mock_backend3.search.call_count == 1

        await server.web_search(self._create_dummy_request(), request)
        assert mock_backend1.search.call_count == 2

        await server.web_search(self._create_dummy_request(), request)
        assert mock_backend2.search.call_count == 2

    async def test_metrics(self, server: TavilySearchResourcesServer) -> None:
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

        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=mock_tavily_response)
        server._async_tavily_clients = [mock_backend]

        request = TavilySearchRequest(query="NVIDIA GPU programming")

        await server.web_search(self._create_dummy_request(), request)
        await server.web_search(self._create_dummy_request(), request)
        await server.web_search(self._create_dummy_request(), request)
        await server.web_search(self._create_dummy_request(), request)
        await server.web_search(self._create_dummy_request(), request)

        expected_metrics_length = 5
        actual_metrics_length = len(server._session_id_to_metrics["abcd"].async_tavily_calls)
        assert expected_metrics_length == actual_metrics_length

    def test_each_worker_dumps_its_own_session_metrics_on_exit(
        self, config: TavilySearchResourcesServerConfig, tmp_path, monkeypatch
    ) -> None:
        """With several workers, each holds only its own sessions, so a shared file would keep only the last dump."""
        monkeypatch.setattr(tavily_search_app, "__file__", str(tmp_path / "app.py"))
        monkeypatch.setattr(tavily_search_app, "is_nemo_gym_fastapi_worker", lambda: True)
        config.dump_session_id_to_metrics_on_exit = True
        server = TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        server._session_id_to_metrics["abcd"]

        with TestClient(server.setup_webserver()):
            pass

        assert not (tmp_path / "session_id_metrics.json").exists()
        assert list(
            orjson.loads((tmp_path / f"session_id_metrics.{os.getppid()}.{os.getpid()}.json").read_bytes())
        ) == ["abcd"]

    # ---- judge verdict parsing (markdown emphasis) ----

    @mark.parametrize(
        "grading_response, expected",
        [
            # Plain form the prompt asks for still parses.
            ("correct: yes", "yes"),
            ("correct: no", "no"),
            # Markdown-bold label and/or value (hosted claude-sonnet-4-5 judge emits these).
            ("**correct:** yes", "yes"),
            ("**correct:** no", "no"),
            ("**correct**: yes", "yes"),
            ("correct: **yes**", "yes"),
            ("*correct:* yes", "yes"),
            # Case-insensitive, like the sibling search-judge parsers.
            ("Correct: Yes", "yes"),
            # The task-2 rollout shape: bold labels throughout, verdict after the reasoning block.
            (
                "**extracted_final_answer:** China\n\n**reasoning:** matches the gold answer\n\n"
                "**correct:** yes\n\n**confidence:** 100",
                "yes",
            ),
        ],
    )
    def test_judge_verdict_regex_accepts_markdown(self, grading_response: str, expected: str) -> None:
        match = _JUDGE_VERDICT_RE.search(grading_response)
        assert match is not None
        assert match.group(1).lower() == expected

    @mark.parametrize(
        "grading_response",
        [
            # Near-miss: an extracted_final_answer value of "yes" with no `correct:` verdict must NOT
            # be read as the verdict (the label must be the `correct` field).
            "**extracted_final_answer:** yes",
            "extracted_final_answer: yes",
            # `correct` inside `incorrect` is not the verdict label.
            "incorrect: yes",
            # No verdict at all.
            "I could not determine whether the answer matches.",
            # Value must be exactly yes/no.
            "correct: maybe",
        ],
    )
    def test_judge_verdict_regex_rejects_non_verdicts(self, grading_response: str) -> None:
        assert _JUDGE_VERDICT_RE.search(grading_response) is None

    async def _verify_with_judge_text(
        self, config: TavilySearchResourcesServerConfig, judge_text: str
    ) -> TavilySearchResourcesServer:
        """Run verify() with a mocked judge that returns ``judge_text`` and return the response."""
        server_client = MagicMock(spec=ServerClient)
        server = TavilySearchResourcesServer(config=config, server_client=server_client)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(return_value=self._create_judge_response(judge_text))
        post_mock.read = AsyncMock(return_value=orjson.dumps(post_mock.json.return_value))
        server_client.post = AsyncMock(return_value=post_mock)

        req = TavilySearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_model_response("The answer is China."),
            ground_truth="People's Republic of China",
            question="What country has Beijing as its capital?",
        )
        return await server.verify(self._create_dummy_request(), req)

    async def test_verify_accepts_bold_correct_verdict(self, config: TavilySearchResourcesServerConfig) -> None:
        """Task-2 repro: bold-labelled judge verdict scores the correct answer reward 1, not a silent 0."""
        judge_text = (
            "**extracted_final_answer:** China\n\n**reasoning:** 'China' is the common name for the "
            "People's Republic of China\n\n**correct:** yes\n\n**confidence:** 100"
        )
        res = await self._verify_with_judge_text(config, judge_text)
        assert res.reward == approx(1.0)
        assert res.extracted_final_answer == "yes"

    async def test_verify_raises_judge_failure_on_unparseable_verdict(
        self, config: TavilySearchResourcesServerConfig
    ) -> None:
        """A judge reply with no parseable verdict is an unusable grade (judge_failed), not a wrong answer."""
        with raises(JudgeError):
            await self._verify_with_judge_text(config, "I am unable to produce a verdict for this response.")
