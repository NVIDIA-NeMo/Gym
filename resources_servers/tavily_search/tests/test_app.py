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


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))

from gpt_oss.tools.simple_browser.backend import BackendError
from gpt_oss.tools.simple_browser.page_contents import PageContents

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.tavily_search.app import (
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
            tavily_api_key="test_api_key",
            exclude_domains_file_path=os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json"),
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            max_retries=2,
            retry_delay_seconds=0,  # No delay in tests
        )

    @fixture
    def server(self, config: TavilySearchResourcesServerConfig) -> TavilySearchResourcesServer:
        return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

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

    def _make_page_contents(self, text: str, url: str = "", title: str = "") -> PageContents:
        """Helper to create a PageContents for mocking backend.fetch()."""
        return PageContents(url=url, text=text, title=title, urls={}, snippets=None)

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
        """Test that _postprocess_search_results returns just the answer when present."""
        raw_results = {
            "answer": "The capital of France is Paris.",
            "results": [
                {"url": "https://example.com", "title": "T", "content": "C"},
            ],
        }
        formatted = server._postprocess_search_results(raw_results)
        joined = "".join(formatted)
        assert "Search Answer" in joined
        assert "The capital of France is Paris." in joined
        # Individual results should NOT be shown
        assert "[1]" not in joined

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
        mock_backend._tavily_post = AsyncMock(return_value=mock_tavily_response)
        server._backend = mock_backend

        request = TavilySearchRequest(query="NVIDIA GPU programming")
        response = await server.web_search(request)

        mock_backend._tavily_post.assert_called_once()
        call_args = mock_backend._tavily_post.call_args
        payload = call_args[0][2]  # third positional arg is the payload dict
        assert payload["query"] == "NVIDIA GPU programming"
        assert payload["max_results"] == server.MAX_RESULTS
        assert "blacklisteddomain.com" in payload["exclude_domains"]

        # Response is now a formatted string, not JSON list
        assert "NVIDIA Documentation" in response.results_string
        assert "nvidia.com" in response.results_string

    async def test_web_search_none_query(self, server: TavilySearchResourcesServer) -> None:
        """Test web_search with None query returns error message."""
        request = TavilySearchRequest(query=None)
        response = await server.web_search(request)
        assert response.results_string == "Query is none"

    async def test_web_search_long_query(self, server: TavilySearchResourcesServer) -> None:
        """Test web_search with overly long query returns error message."""
        request = TavilySearchRequest(query="x" * 401)
        response = await server.web_search(request)
        assert response.results_string == "Query is too long"

    async def test_web_search_retry_on_backend_error(self, server: TavilySearchResourcesServer) -> None:
        """Test that web_search retries on BackendError."""
        good_response = {
            "results": [
                {"url": "https://example.com", "title": "Title", "content": "Content"},
            ]
        }
        mock_backend = MagicMock()
        mock_backend._tavily_post = AsyncMock(side_effect=[BackendError("rate limited"), good_response])
        server._backend = mock_backend

        request = TavilySearchRequest(query="test query")
        response = await server.web_search(request)

        assert mock_backend._tavily_post.call_count == 2
        assert "Title" in response.results_string

    async def test_web_search_max_retries_exceeded(self, server: TavilySearchResourcesServer) -> None:
        """Test that web_search returns empty after max retries."""
        mock_backend = MagicMock()
        mock_backend._tavily_post = AsyncMock(side_effect=BackendError("server error"))
        server._backend = mock_backend

        request = TavilySearchRequest(query="test query")
        response = await server.web_search(request)

        assert mock_backend._tavily_post.call_count == server.config.max_retries
        assert response.results_string == "[]"

    # ---- find_in_page ----

    async def test_find_in_page(self, server: TavilySearchResourcesServer) -> None:
        """Test find_in_page endpoint with mocked backend.fetch()."""
        mock_page = self._make_page_contents(
            text="This is the page content about Python programming.",
            url="https://example.com/python",
            title="Python Guide",
        )
        mock_backend = MagicMock()
        mock_backend.fetch = AsyncMock(return_value=mock_page)
        server._backend = mock_backend

        request = FindInPageRequest(url="https://example.com/python", query="Python")
        response = await server.find_in_page(request)

        mock_backend.fetch.assert_called_once()
        call_kwargs = mock_backend.fetch.call_args
        assert call_kwargs[1]["query"] == "Python"
        assert call_kwargs[1]["display_urls"] is False

        # Check header
        assert "Content from: example.com" in response.results_string
        assert "URL: https://example.com/python" in response.results_string
        assert 'Query: "Python"' in response.results_string
        assert "========" in response.results_string
        # Check line numbers
        assert "L0:" in response.results_string
        # Check content
        assert "Python programming" in response.results_string

    async def test_find_in_page_none_url(self, server: TavilySearchResourcesServer) -> None:
        """Test find_in_page with None URL returns error."""
        request = FindInPageRequest(url=None, query="test")
        response = await server.find_in_page(request)
        assert response.results_string == "URL is none"

    async def test_find_in_page_none_query(self, server: TavilySearchResourcesServer) -> None:
        """Test find_in_page with None query returns error."""
        request = FindInPageRequest(url="https://example.com", query=None)
        response = await server.find_in_page(request)
        assert response.results_string == "Query is none"

    async def test_find_in_page_excluded_domain(self, server: TavilySearchResourcesServer) -> None:
        """Test find_in_page with excluded domain returns error."""
        request = FindInPageRequest(url="https://blacklisteddomain.com/page", query="test")
        response = await server.find_in_page(request)
        assert response.results_string == "URL is in excluded domains"

    async def test_find_in_page_empty_content(self, server: TavilySearchResourcesServer) -> None:
        """Test find_in_page returns 'No content found.' for empty page."""
        mock_page = self._make_page_contents(text="   ", url="https://example.com")
        mock_backend = MagicMock()
        mock_backend.fetch = AsyncMock(return_value=mock_page)
        server._backend = mock_backend

        request = FindInPageRequest(url="https://example.com", query="test")
        response = await server.find_in_page(request)
        assert response.results_string == "No content found."

    async def test_find_in_page_truncation(self, server: TavilySearchResourcesServer) -> None:
        """Test find_in_page truncates long content."""
        # Create content longer than MAX_RESULT_CHARS (2000)
        long_text = "Line of text number {i}\n" * 300  # ~7200 chars
        mock_page = self._make_page_contents(text=long_text, url="https://example.com")
        mock_backend = MagicMock()
        mock_backend.fetch = AsyncMock(return_value=mock_page)
        server._backend = mock_backend

        request = FindInPageRequest(url="https://example.com", query="test")
        response = await server.find_in_page(request)
        assert "[...truncated, use scroll_page for full content]" in response.results_string

    async def test_find_in_page_retry_on_backend_error(self, server: TavilySearchResourcesServer) -> None:
        """Test find_in_page retries on BackendError."""
        mock_page = self._make_page_contents(text="Content here", url="https://example.com")
        mock_backend = MagicMock()
        mock_backend.fetch = AsyncMock(side_effect=[BackendError("server error"), mock_page])
        server._backend = mock_backend

        request = FindInPageRequest(url="https://example.com", query="test")
        response = await server.find_in_page(request)

        assert mock_backend.fetch.call_count == 2
        assert "Content here" in response.results_string

    # ---- scroll_page ----

    async def test_scroll_page(self, server: TavilySearchResourcesServer) -> None:
        """Test scroll_page endpoint with mocked backend.fetch()."""
        page_text = " ".join([f"word{i}" for i in range(100)])
        mock_page = self._make_page_contents(text=page_text, url="https://example.com/article")
        mock_backend = MagicMock()
        mock_backend.fetch = AsyncMock(return_value=mock_page)
        server._backend = mock_backend

        request = ScrollPageRequest(url="https://example.com/article", start_index=0, n=50)
        response = await server.scroll_page(request)

        mock_backend.fetch.assert_called_once()
        assert "Page content from: example.com" in response.results_string
        assert "URL: https://example.com/article" in response.results_string
        assert "Showing words [0-50] of 100" in response.results_string
        assert response.total_words == 100
        assert "word0" in response.results_string
        assert "word49" in response.results_string

    async def test_scroll_page_caching(self, server: TavilySearchResourcesServer) -> None:
        """Test that scroll_page caches page content."""
        page_text = " ".join([f"word{i}" for i in range(100)])
        mock_page = self._make_page_contents(text=page_text, url="https://example.com/cached")
        mock_backend = MagicMock()
        mock_backend.fetch = AsyncMock(return_value=mock_page)
        server._backend = mock_backend

        # First call — should fetch
        request = ScrollPageRequest(url="https://example.com/cached", start_index=0, n=50)
        await server.scroll_page(request)
        assert mock_backend.fetch.call_count == 1

        # Second call — should use cache, NOT call fetch again
        request2 = ScrollPageRequest(url="https://example.com/cached", start_index=50, n=50)
        response2 = await server.scroll_page(request2)
        assert mock_backend.fetch.call_count == 1  # Still 1 — cache hit

        assert "word50" in response2.results_string
        assert response2.total_words == 100

    async def test_scroll_page_none_url(self, server: TavilySearchResourcesServer) -> None:
        """Test scroll_page with None URL."""
        request = ScrollPageRequest(url=None)
        response = await server.scroll_page(request)
        assert response.results_string == "URL is none"
        assert response.total_words == 0

    async def test_scroll_page_excluded_domain(self, server: TavilySearchResourcesServer) -> None:
        """Test scroll_page with excluded domain."""
        request = ScrollPageRequest(url="https://blacklisteddomain.com/page")
        response = await server.scroll_page(request)
        assert response.results_string == "URL is in excluded domains"
        assert response.total_words == 0

    async def test_scroll_page_retry_on_backend_error(self, server: TavilySearchResourcesServer) -> None:
        """Test scroll_page retries on BackendError."""
        page_text = "some content words here"
        mock_page = self._make_page_contents(text=page_text, url="https://example.com")
        mock_backend = MagicMock()
        mock_backend.fetch = AsyncMock(side_effect=[BackendError("timeout"), mock_page])
        server._backend = mock_backend

        request = ScrollPageRequest(url="https://example.com", start_index=0, n=100)
        response = await server.scroll_page(request)

        assert mock_backend.fetch.call_count == 2
        assert response.total_words == 4

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
