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
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
from aiohttp import ClientResponseError
from pytest import approx, fixture, raises

from nemo_gym.server_utils import SESSION_ID_KEY


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))

from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.you_search.app import (
    FindInPageRequest,
    ScrollPageRequest,
    YouSearchRequest,
    YouSearchResourcesServer,
    YouSearchResourcesServerConfig,
    YouSearchSingleAPICallMetrics,
    YouSearchVerifyRequest,
)


def _web_result(url: str, title: str, **extra: Any) -> dict[str, Any]:
    """A /v1/search web result with the fields every mode returns."""
    return {
        "url": url,
        "title": title,
        "description": f"Description of {title}",
        "snippets": [f"Snippet about {title}"],
        **extra,
    }


class TestApp:
    @fixture
    def config(self) -> YouSearchResourcesServerConfig:
        return YouSearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            ydc_api_key="test_api_key",  # pragma: allowlist secret
            exclude_domains_file_path=os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json"),
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

    @fixture
    def server(self, config: YouSearchResourcesServerConfig) -> YouSearchResourcesServer:
        return YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

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

    def test_sanity(self, config: YouSearchResourcesServerConfig) -> None:
        YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    # ---- _search_payload: the one knob that separates the arms ----

    def test_search_payload_snippets_sends_no_extraction(self, server: YouSearchResourcesServer) -> None:
        payload = server._search_payload("nvidia gpus")
        assert payload["query"] == "nvidia gpus"
        assert payload["count"] == 10
        assert "extraction" not in payload
        assert "crawl_timeout" not in payload

    def test_search_payload_highlights(self, config: YouSearchResourcesServerConfig) -> None:
        config.search_mode = "highlights"
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        payload = server._search_payload("nvidia gpus")
        assert payload["extraction"] == {"extraction_mode": "highlights"}
        # highlights is not a crawl, so it must not carry a crawl budget
        assert "crawl_timeout" not in payload

    def test_search_payload_full_page(self, config: YouSearchResourcesServerConfig) -> None:
        config.search_mode = "full_page"
        config.crawl_timeout = 42
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        payload = server._search_payload("nvidia gpus")
        assert payload["extraction"] == {
            "extraction_mode": "full_page",
            "full_page": {"extraction_formats": ["markdown"]},
        }
        assert payload["crawl_timeout"] == 42

    def test_search_payload_eco_is_query_and_count_only(self, config: YouSearchResourcesServerConfig) -> None:
        """eco_search ignores extraction, crawl budget, and exclude_domains."""
        config.search_mode = "eco"
        config.num_results = 4
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        payload = server._search_payload("nvidia gpus")
        assert payload == {"query": "nvidia gpus", "count": 4}
        assert server._search_endpoint() == "/v1/eco_search"

    def test_search_endpoint_defaults_to_unified_search(self, server: YouSearchResourcesServer) -> None:
        assert server._search_endpoint() == "/v1/search"

    async def test_web_search_eco_uses_eco_endpoint(self, config: YouSearchResourcesServerConfig) -> None:
        """eco returns the same envelope, minus `contents` and the news section."""
        config.search_mode = "eco"
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        server._post = AsyncMock(
            return_value={"results": {"web": [_web_result("https://example.com/p", "Eco Result")]}}
        )

        response = await server.web_search(self._create_dummy_request(), YouSearchRequest(query="q"))

        endpoint, payload = server._post.call_args.args
        assert endpoint == "/v1/eco_search"
        assert "extraction" not in payload
        assert "exclude_domains" not in payload
        assert "Eco Result" in response.results_string
        assert "Snippet about Eco Result" in response.results_string

    async def test_eco_still_filters_excluded_domains_client_side(
        self, config: YouSearchResourcesServerConfig
    ) -> None:
        """The API ignores exclude_domains on eco, so the client-side pass is load-bearing."""
        config.search_mode = "eco"
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        server._post = AsyncMock(
            return_value={
                "results": {
                    "web": [
                        _web_result("https://blacklisteddomain.com/p", "Blocked"),
                        _web_result("https://example.com/p", "Allowed"),
                    ]
                }
            }
        )

        response = await server.web_search(self._create_dummy_request(), YouSearchRequest(query="q"))
        assert "Blocked" not in response.results_string
        assert "[1] Allowed (example.com)" in response.results_string

    def test_search_payload_sends_exclude_domains(self, server: YouSearchResourcesServer) -> None:
        assert server._search_payload("q")["exclude_domains"] == ["blacklisteddomain.com"]

    def test_search_payload_caps_exclude_domains_at_api_limit(self, server: YouSearchResourcesServer) -> None:
        """You.com rejects >500 domains, so the wire list is truncated."""
        server._exclude_domains = [f"domain{i}.com" for i in range(600)]
        assert len(server._search_payload("q")["exclude_domains"]) == 500

    def test_search_payload_omits_exclude_domains_when_unset(self, config: YouSearchResourcesServerConfig) -> None:
        """The opt-out registry is optional — most deployments do not have one."""
        config.exclude_domains_file_path = None
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        assert server._exclude_domains == []
        assert "exclude_domains" not in server._search_payload("q")

    def test_search_payload_respects_num_results(self, config: YouSearchResourcesServerConfig) -> None:
        config.num_results = 3
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        assert server._search_payload("q")["count"] == 3

    # ---- _postprocess_search_results ----

    def test_postprocess_search_results(self, server: YouSearchResourcesServer) -> None:
        raw_results = {
            "results": {
                "web": [
                    _web_result("https://example.com/page1", "Example Page 1"),
                    _web_result("https://example.com/page2", "Example Page 2"),
                ]
            }
        }

        formatted_results = server._postprocess_search_results(raw_results)

        assert isinstance(formatted_results, list)
        joined = "".join(formatted_results)
        assert "Search Results" in joined
        assert "[1] Example Page 1 (example.com)" in joined
        assert "[2] Example Page 2 (example.com)" in joined
        assert "URL: https://example.com/page1" in joined
        assert "Snippet about Example Page 1" in joined
        assert "Description of Example Page 2" in joined

    def test_postprocess_drops_description_duplicating_snippet(self, server: YouSearchResourcesServer) -> None:
        """description is usually a truncated copy of snippets; emitting both doubles tokens."""
        raw_results = {
            "results": {
                "web": [
                    {
                        "url": "https://example.com/p",
                        "title": "Dupe",
                        "snippets": ["The prize was awarded for explaining innovation-driven growth"],
                        "description": "The prize was awarded for explaining innovation...",
                    }
                ]
            }
        }
        joined = "".join(server._postprocess_search_results(raw_results))
        assert joined.count("The prize was awarded for explaining innovation") == 1
        assert "innovation-driven growth" in joined

    def test_postprocess_keeps_description_adding_information(self, server: YouSearchResourcesServer) -> None:
        raw_results = {
            "results": {
                "web": [
                    {
                        "url": "https://example.com/p",
                        "title": "Distinct",
                        "snippets": ["a passage about growth"],
                        "description": "an unrelated summary of the page",
                    }
                ]
            }
        }
        joined = "".join(server._postprocess_search_results(raw_results))
        assert "a passage about growth" in joined
        assert "an unrelated summary of the page" in joined

    def test_dedupe_texts_preserves_order(self, server: YouSearchResourcesServer) -> None:
        assert server._dedupe_texts(["short", "a much longer distinct string"]) == [
            "short",
            "a much longer distinct string",
        ]
        assert server._dedupe_texts(["", "  ", "only"]) == ["only"]

    def test_postprocess_prefers_highlights_over_snippets(self, server: YouSearchResourcesServer) -> None:
        raw_results = {
            "results": {
                "web": [
                    _web_result(
                        "https://example.com/p",
                        "Highlighted",
                        contents={"highlights": ["the relevant passage", "a second passage"]},
                    )
                ]
            }
        }
        joined = "".join(server._postprocess_search_results(raw_results))
        assert "the relevant passage" in joined
        assert "a second passage" in joined
        assert "Snippet about Highlighted" not in joined

    def test_postprocess_prefers_markdown_over_highlights(self, server: YouSearchResourcesServer) -> None:
        raw_results = {
            "results": {
                "web": [
                    _web_result(
                        "https://example.com/p",
                        "Crawled",
                        contents={"markdown": "# full page body", "highlights": ["a passage"]},
                    )
                ]
            }
        }
        joined = "".join(server._postprocess_search_results(raw_results))
        assert "# full page body" in joined
        assert "a passage" not in joined

    def test_postprocess_falls_back_when_extraction_empty(self, server: YouSearchResourcesServer) -> None:
        """A crawl that times out still yields a usable result, not a blank one."""
        raw_results = {
            "results": {"web": [_web_result("https://example.com/p", "Timed Out", contents={"markdown": None})]}
        }
        joined = "".join(server._postprocess_search_results(raw_results))
        assert "Snippet about Timed Out" in joined

    def test_postprocess_excludes_news_by_default(self, server: YouSearchResourcesServer) -> None:
        raw_results = {
            "results": {
                "web": [_web_result("https://example.com/w", "Web Item")],
                "news": [_web_result("https://news.example.com/n", "News Item")],
            }
        }
        joined = "".join(server._postprocess_search_results(raw_results))
        assert "Web Item" in joined
        assert "News Item" not in joined

    def test_postprocess_includes_news_first_when_enabled(self, config: YouSearchResourcesServerConfig) -> None:
        config.include_news = True
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        raw_results = {
            "results": {
                "web": [_web_result("https://example.com/w", "Web Item")],
                "news": [_web_result("https://news.example.com/n", "News Item")],
            }
        }
        joined = "".join(server._postprocess_search_results(raw_results))
        assert "[1] News Item" in joined
        assert "[2] Web Item" in joined

    def test_postprocess_filters_excluded_domains_from_results(self, server: YouSearchResourcesServer) -> None:
        """Client-side enforcement, since the wire list is capped at 500."""
        raw_results = {
            "results": {
                "web": [
                    _web_result("https://blacklisteddomain.com/p", "Blocked"),
                    _web_result("https://example.com/p", "Allowed"),
                ]
            }
        }
        joined = "".join(server._postprocess_search_results(raw_results))
        assert "Blocked" not in joined
        assert "[1] Allowed (example.com)" in joined

    def test_postprocess_empty_results(self, server: YouSearchResourcesServer) -> None:
        joined = "".join(server._postprocess_search_results({"results": {"web": []}}))
        assert "No results found." in joined

    # ---- web_search ----

    async def test_web_search(self, server: YouSearchResourcesServer) -> None:
        server._post = AsyncMock(
            return_value={"results": {"web": [_web_result("https://nvidia.com/docs", "NVIDIA Documentation")]}}
        )

        response = await server.web_search(
            self._create_dummy_request(), YouSearchRequest(query="NVIDIA GPU programming")
        )

        server._post.assert_called_once()
        endpoint, payload = server._post.call_args.args
        assert endpoint == "/v1/search"
        assert payload["query"] == "NVIDIA GPU programming"

        assert "NVIDIA Documentation" in response.results_string
        assert "nvidia.com" in response.results_string

    async def test_web_search_none_query(self, server: YouSearchResourcesServer) -> None:
        response = await server.web_search(self._create_dummy_request(), YouSearchRequest(query=None))
        assert response.results_string == "Query is none"

    async def test_web_search_long_query(self, server: YouSearchResourcesServer) -> None:
        response = await server.web_search(self._create_dummy_request(), YouSearchRequest(query="x" * 401))
        assert response.results_string == "Query is too long"

    # ---- find_in_page ----

    async def test_find_in_page(self, server: YouSearchResourcesServer) -> None:
        server._post = AsyncMock(return_value=[{"url": "https://example.com/p", "markdown": "line one\nline two"}])

        response = await server.find_in_page(
            self._create_dummy_request(), FindInPageRequest(url="https://example.com/p", query="one")
        )

        endpoint, payload = server._post.call_args.args
        assert endpoint == "/v1/contents"
        assert payload["urls"] == ["https://example.com/p"]
        assert payload["formats"] == ["markdown"]

        assert "Content from: example.com" in response.results_string
        assert 'Query: "one"' in response.results_string
        assert "L0: line one" in response.results_string
        assert "L1: line two" in response.results_string

    async def test_find_in_page_no_content(self, server: YouSearchResourcesServer) -> None:
        server._post = AsyncMock(return_value=[{"url": "https://example.com/p", "markdown": None}])
        response = await server.find_in_page(
            self._create_dummy_request(), FindInPageRequest(url="https://example.com/p", query="q")
        )
        assert response.results_string == "No content found."

    async def test_find_in_page_none_url(self, server: YouSearchResourcesServer) -> None:
        response = await server.find_in_page(self._create_dummy_request(), FindInPageRequest(url=None, query="test"))
        assert response.results_string == "URL is none"

    async def test_find_in_page_none_query(self, server: YouSearchResourcesServer) -> None:
        response = await server.find_in_page(
            self._create_dummy_request(), FindInPageRequest(url="https://example.com", query=None)
        )
        assert response.results_string == "Query is none"

    async def test_find_in_page_excluded_domain(self, server: YouSearchResourcesServer) -> None:
        response = await server.find_in_page(
            self._create_dummy_request(),
            FindInPageRequest(url="https://blacklisteddomain.com/page", query="test"),
        )
        assert response.results_string == "URL is in excluded domains"

    # ---- scroll_page ----

    async def test_scroll_page_slices_and_caches(self, server: YouSearchResourcesServer) -> None:
        server._post = AsyncMock(
            return_value=[{"url": "https://example.com/p", "markdown": " ".join(f"w{i}" for i in range(100))}]
        )
        dummy_request = self._create_dummy_request()

        response = await server.scroll_page(
            dummy_request, ScrollPageRequest(url="https://example.com/p", start_index=10, n=5)
        )
        assert response.total_words == 100
        assert "Showing words [10-15] of 100" in response.results_string
        assert "w10 w11 w12 w13 w14" in response.results_string
        assert "w15" not in response.results_string

        # A second scroll of the same page must not re-crawl it.
        await server.scroll_page(dummy_request, ScrollPageRequest(url="https://example.com/p", start_index=20, n=5))
        assert server._post.call_count == 1

    async def test_scroll_page_none_url(self, server: YouSearchResourcesServer) -> None:
        response = await server.scroll_page(self._create_dummy_request(), ScrollPageRequest(url=None))
        assert response.results_string == "URL is none"
        assert response.total_words == 0

    async def test_scroll_page_excluded_domain(self, server: YouSearchResourcesServer) -> None:
        response = await server.scroll_page(
            self._create_dummy_request(), ScrollPageRequest(url="https://blacklisteddomain.com/page")
        )
        assert response.results_string == "URL is in excluded domains"
        assert response.total_words == 0

    # ---- Utility functions ----

    def test_extract_domain(self, server: YouSearchResourcesServer) -> None:
        assert server._extract_domain("https://en.wikipedia.org/wiki/Python") == "en.wikipedia.org"
        assert server._extract_domain("http://example.com/path") == "example.com"

    def test_clean_text(self, server: YouSearchResourcesServer) -> None:
        text = "Hello [edit] world\n[Jump to content]\nContent here​"
        cleaned = server._clean_text(text)
        assert "[edit]" not in cleaned
        assert "[Jump to content]" not in cleaned
        assert "​" not in cleaned
        assert "Hello" in cleaned
        assert "Content here" in cleaned

    def test_add_line_numbers(self, server: YouSearchResourcesServer) -> None:
        text = "first\nsecond\nthird"
        result = server._add_line_numbers(text)
        assert result == "L0: first\nL1: second\nL2: third"

    def test_truncate_text_short(self, server: YouSearchResourcesServer) -> None:
        result, was_truncated = server._truncate_text("short text")
        assert result == "short text"
        assert was_truncated is False

    def test_truncate_text_long(self, server: YouSearchResourcesServer) -> None:
        text = "\n".join([f"Line {i}" for i in range(500)])
        result, was_truncated = server._truncate_text(text, max_chars=100)
        assert was_truncated is True
        assert len(result) <= 100
        assert result.endswith(result.split("\n")[-1])

    def test_truncate_text_honours_configured_cap(self, config: YouSearchResourcesServerConfig) -> None:
        """full_page runs raise this so the crawl isn't truncated away."""
        config.max_result_chars = 20
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        _, was_truncated = server._truncate_text("x" * 50)
        assert was_truncated is True

    def test_is_url_excluded(self, server: YouSearchResourcesServer) -> None:
        assert server._is_url_excluded("https://blacklisteddomain.com/page") is True
        assert server._is_url_excluded("https://sub.blacklisteddomain.com/page") is True
        assert server._is_url_excluded("https://example.com/page") is False

    # ---- verify ----

    async def test_verify_correct_answer(self, config: YouSearchResourcesServerConfig) -> None:
        server_client = MagicMock(spec=ServerClient)
        server = YouSearchResourcesServer(config=config, server_client=server_client)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(return_value=self._create_judge_response("correct: yes"))
        post_mock.read = AsyncMock(return_value=orjson.dumps(post_mock.json.return_value))
        server_client.post = AsyncMock(return_value=post_mock)

        req = YouSearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_model_response("The capital of France is Paris."),
            ground_truth="Paris",
            question="What is the capital of France?",
        )

        res = await server.verify(self._create_dummy_request(), req)

        assert res.reward == approx(1.0)
        assert res.extracted_final_answer == "yes"
        assert server_client.post.call_count == 1

    async def test_verify_incorrect_answer(self, config: YouSearchResourcesServerConfig) -> None:
        server_client = MagicMock(spec=ServerClient)
        server = YouSearchResourcesServer(config=config, server_client=server_client)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(return_value=self._create_judge_response("correct: no"))
        post_mock.read = AsyncMock(return_value=orjson.dumps(post_mock.json.return_value))
        server_client.post = AsyncMock(return_value=post_mock)

        req = YouSearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_model_response("The capital of France is London."),
            ground_truth="Paris",
            question="What is the capital of France?",
        )

        res = await server.verify(self._create_dummy_request(), req)

        assert res.reward == approx(0.0)
        assert res.extracted_final_answer == "no"
        assert server_client.post.call_count == 1

    # ---- key rotation and metrics ----

    async def test_api_key_rotation_sanity(self, config: YouSearchResourcesServerConfig) -> None:
        """Multiple calls rotate through the configured keys, round-robin."""
        config.ydc_api_key = ["key1", "key2", "key3"]  # pragma: allowlist secret
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        assert [server._select_api_key() for _ in range(5)] == ["key1", "key2", "key3", "key1", "key2"]

    async def test_metrics(self, server: YouSearchResourcesServer) -> None:
        server._post = AsyncMock(
            return_value={"results": {"web": [_web_result("https://nvidia.com/docs", "NVIDIA Documentation")]}}
        )

        request = YouSearchRequest(query="NVIDIA GPU programming")
        for _ in range(5):
            await server.web_search(self._create_dummy_request(), request)

        calls = server._session_id_to_metrics["abcd"].you_api_calls
        assert len(calls) == 5
        assert all(c.function == "search" and c.time_taken is not None for c in calls)

    # ---- _post: the You.com transport ----

    def _fake_response(self, status: int, payload: Any = None, body: bytes = b"boom") -> MagicMock:
        """Stand-in for an aiohttp ClientResponse, enough for _post and raise_for_status."""
        response = MagicMock()
        response.status = status
        response.ok = status < 400
        response.json = AsyncMock(return_value=payload)
        response.content.read = AsyncMock(return_value=body)

        def _raise() -> None:
            raise ClientResponseError(request_info=None, history=(), status=status)

        response.raise_for_status = _raise
        return response

    async def test_post_returns_parsed_json(self, server: YouSearchResourcesServer) -> None:
        with patch("resources_servers.you_search.app.request", AsyncMock()) as mock_request:
            mock_request.return_value = self._fake_response(200, {"results": {"web": []}})
            assert await server._post("/v1/search", {"query": "q"}) == {"results": {"web": []}}
            assert mock_request.call_args.kwargs["headers"] == {
                "X-API-Key": "test_api_key"
            }  # pragma: allowlist secret

    async def test_post_raises_on_non_retryable_status(self, server: YouSearchResourcesServer) -> None:
        """A bad API key must fail loudly, not come back as an empty result set.

        401 is not in RETRY_ERROR_CODES; without an explicit raise the error body was
        returned as if it were results and every task scored 0.0 with a clean log.
        """
        with patch("resources_servers.you_search.app.request", AsyncMock()) as mock_request:
            mock_request.return_value = self._fake_response(401, {"detail": "invalid api key"})
            with raises(ClientResponseError):
                await server._post("/v1/search", {"query": "q"})
            assert mock_request.call_count == 1  # not retried

    async def test_web_search_propagates_auth_failure(self, server: YouSearchResourcesServer) -> None:
        """End to end: the 401 surfaces rather than becoming 'No results found.'"""
        with patch("resources_servers.you_search.app.request", AsyncMock()) as mock_request:
            mock_request.return_value = self._fake_response(401, {"detail": "invalid api key"})
            with raises(ClientResponseError):
                await server.web_search(self._create_dummy_request(), YouSearchRequest(query="q"))

    async def test_post_retries_transient_failure_then_succeeds(self, server: YouSearchResourcesServer) -> None:
        ok = self._fake_response(200, {"results": {}})
        with patch("resources_servers.you_search.app.request", AsyncMock()) as mock_request:
            mock_request.side_effect = [self._fake_response(500), ok]
            assert await server._post("/v1/search", {"query": "q"}) == {"results": {}}
            assert mock_request.call_count == 2

    async def test_post_rotates_key_on_retry(self, config: YouSearchResourcesServerConfig) -> None:
        """A retry after a 429 must move to the next key, not re-hit the throttled one."""
        config.ydc_api_key = ["key1", "key2"]  # pragma: allowlist secret
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        with patch("resources_servers.you_search.app.request", AsyncMock()) as mock_request:
            mock_request.side_effect = [self._fake_response(429), self._fake_response(200, {"ok": True})]
            assert await server._post("/v1/search", {"query": "q"}) == {"ok": True}
            used = [c.kwargs["headers"]["X-API-Key"] for c in mock_request.call_args_list]
            assert used == ["key1", "key2"]  # pragma: allowlist secret

    async def test_post_rate_limit_extends_try_budget(self, server: YouSearchResourcesServer) -> None:
        """429 grants an extra try, so 3 rate limits still leave a fourth attempt."""
        with patch("resources_servers.you_search.app.request", AsyncMock()) as mock_request:
            mock_request.side_effect = [self._fake_response(429)] * 3 + [self._fake_response(200, {"ok": True})]
            assert await server._post("/v1/search", {"query": "q"}) == {"ok": True}
            assert mock_request.call_count == 4

    async def test_post_raises_after_exhausting_retries(self, server: YouSearchResourcesServer) -> None:
        with patch("resources_servers.you_search.app.request", AsyncMock()) as mock_request:
            mock_request.return_value = self._fake_response(500)
            with raises(ClientResponseError):
                await server._post("/v1/search", {"query": "q"})
            assert mock_request.call_count == 3

    # ---- metrics record failures too ----

    async def test_metrics_records_failed_call(self, server: YouSearchResourcesServer) -> None:
        """Timing only the successes would make the latency histogram survivor-biased."""
        server._post = AsyncMock(side_effect=RuntimeError("boom"))
        with raises(RuntimeError):
            await server.web_search(self._create_dummy_request(), YouSearchRequest(query="q"))

        calls = server._session_id_to_metrics["abcd"].you_api_calls
        assert [c.status for c in calls] == ["failure"]
        assert calls[0].time_taken is not None

    # ---- page cache ----

    async def test_fetch_page_does_not_cache_empty_content(self, server: YouSearchResourcesServer) -> None:
        """One crawl timeout must not disable the URL for the process lifetime."""
        server._post = AsyncMock(return_value=[{"url": "https://example.com/p", "markdown": None}])
        metrics = server._session_id_to_metrics["abcd"]

        assert await server._fetch_page("https://example.com/p", metrics) == ""
        assert "https://example.com/p" not in server._page_cache

        # A later retry succeeds and is cached.
        server._post = AsyncMock(return_value=[{"url": "https://example.com/p", "markdown": "recovered"}])
        assert await server._fetch_page("https://example.com/p", metrics) == "recovered"
        assert server._page_cache["https://example.com/p"] == "recovered"

    async def test_page_cache_evicts_least_recently_used(self, config: YouSearchResourcesServerConfig) -> None:
        config.page_cache_max_entries = 2
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        metrics = server._session_id_to_metrics["abcd"]

        for name in ("a", "b"):
            server._post = AsyncMock(return_value=[{"markdown": name}])
            await server._fetch_page(f"https://example.com/{name}", metrics)

        # Touch "a" so "b" becomes the least recently used entry.
        await server._fetch_page("https://example.com/a", metrics)

        server._post = AsyncMock(return_value=[{"markdown": "c"}])
        await server._fetch_page("https://example.com/c", metrics)

        assert set(server._page_cache) == {"https://example.com/a", "https://example.com/c"}

    # ---- domain matching ----

    def test_is_url_excluded_normalizes_case_www_and_scheme(self, server: YouSearchResourcesServer) -> None:
        """The registry is a legal opt-out list; near-miss spellings must not slip through."""
        server._exclude_domains = [server._normalize_domain("WWW.BlackListedDomain.COM ")]
        assert server._exclude_domains == ["blacklisteddomain.com"]
        assert server._is_url_excluded("https://BlackListedDomain.com/page") is True
        assert server._is_url_excluded("https://www.blacklisteddomain.com/page") is True
        # A scheme-less URL puts the host in `path`, so urlparse().hostname would be None.
        assert server._is_url_excluded("blacklisteddomain.com/page") is True
        assert server._is_url_excluded("https://example.com/page") is False
        assert server._is_url_excluded("") is False

    def test_extract_domain_handles_scheme_less_url(self, server: YouSearchResourcesServer) -> None:
        assert server._extract_domain("example.com/path") == "example.com"

    # ---- _clean_text must not eat content ----

    def test_clean_text_keeps_content_line_starting_with_link(self, server: YouSearchResourcesServer) -> None:
        """`[Read...` once matched any line prefix and `.*$` deleted the rest of the line."""
        text = "[Reading list](https://example.com/l) - the 2025 laureate was Mokyr"
        assert "2025 laureate was Mokyr" in server._clean_text(text)

    def test_clean_text_still_strips_standalone_nav_lines(self, server: YouSearchResourcesServer) -> None:
        text = "[Jump to content]\n[View history](/history)\nreal content"
        cleaned = server._clean_text(text)
        assert "Jump to content" not in cleaned
        assert "View history" not in cleaned
        assert "real content" in cleaned

    def test_clean_text_keeps_inline_wikipedia_citation(self, server: YouSearchResourcesServer) -> None:
        """Unanchored, the sidebar rule also stripped citations out of running prose."""
        text = "The prize went to [Mokyr](https://en.wikipedia.org/wiki/Joel_Mokyr) in 2025."
        cleaned = server._clean_text(text)
        assert "Mokyr" in cleaned
        assert "in 2025." in cleaned

    def test_clean_text_strips_language_sidebar_line(self, server: YouSearchResourcesServer) -> None:
        text = "* [Deutsch](https://de.wikipedia.org/wiki/X)\nbody text"
        cleaned = server._clean_text(text)
        assert "Deutsch" not in cleaned
        assert "body text" in cleaned

    # ---- find_in_page relevance ----

    async def test_find_in_page_windows_on_query(self, config: YouSearchResourcesServerConfig) -> None:
        """/v1/contents takes no query, so relevance selection happens client-side."""
        config.max_result_chars = 120
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        page = (
            ("navigation chrome and boilerplate\n" * 20)
            + "the 2025 economics laureate was Joel Mokyr\n"
            + ("trailing filler\n" * 20)
        )
        server._post = AsyncMock(return_value=[{"markdown": page}])

        response = await server.find_in_page(
            self._create_dummy_request(),
            FindInPageRequest(url="https://example.com/p", query="economics laureate"),
        )

        assert "Joel Mokyr" in response.results_string
        assert "Showing the best match for the query" in response.results_string
        assert "[...truncated, use scroll_page for full content]" in response.results_string

    async def test_find_in_page_falls_back_to_head_without_match(self, config: YouSearchResourcesServerConfig) -> None:
        config.max_result_chars = 60
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        server._post = AsyncMock(return_value=[{"markdown": "alpha bravo charlie\n" * 20}])

        response = await server.find_in_page(
            self._create_dummy_request(),
            FindInPageRequest(url="https://example.com/p", query="nothing matches here"),
        )
        assert "Showing the best match" not in response.results_string
        assert "L0: alpha bravo charlie" in response.results_string

    def test_query_window_weights_rare_terms_over_filler(self, config: YouSearchResourcesServerConfig) -> None:
        """Filler words must not outvote the discriminative ones.

        Scoring every term equally put a window full of "the/first/prize" ahead of the one
        actually naming the person asked about, on the real Nobel economics article.
        """
        config.max_result_chars = 100
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        filler = "the first prize win the first prize win\n"
        page = (filler * 12) + "Elinor Ostrom was the first woman to win the prize\n" + (filler * 12)

        window, start = server._query_window(page, "Elinor Ostrom first woman to win the prize")

        assert "Ostrom" in window
        assert start > 0

    def test_query_window_short_text_is_untouched(self, server: YouSearchResourcesServer) -> None:
        assert server._query_window("short page", "anything") == ("short page", 0)

    def test_query_window_without_usable_terms_returns_head(self, config: YouSearchResourcesServerConfig) -> None:
        """Stopword-length tokens carry no signal, so fall back to the head of the page."""
        config.max_result_chars = 40
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        window, start = server._query_window("x" * 200, "a of")
        assert start == 0
        assert len(window) <= 40

    # ---- scroll_page bounds ----

    async def test_scroll_page_clamps_negative_start_index(self, server: YouSearchResourcesServer) -> None:
        """Negative indices wrapped to the tail while the header described another span."""
        server._post = AsyncMock(return_value=[{"markdown": " ".join(f"w{i}" for i in range(100))}])

        response = await server.scroll_page(
            self._create_dummy_request(), ScrollPageRequest(url="https://example.com/p", start_index=-5, n=3)
        )
        assert "Showing words [0-3] of 100" in response.results_string
        assert "w0 w1 w2" in response.results_string

    async def test_scroll_page_start_index_past_end(self, server: YouSearchResourcesServer) -> None:
        server._post = AsyncMock(return_value=[{"markdown": " ".join(f"w{i}" for i in range(10))}])
        response = await server.scroll_page(
            self._create_dummy_request(), ScrollPageRequest(url="https://example.com/p", start_index=500, n=5)
        )
        assert response.total_words == 10
        assert "Showing words [10-10] of 10" in response.results_string

    # ---- regex verifier and session bookkeeping ----

    async def test_verify_with_regex_verifier(self, config: YouSearchResourcesServerConfig) -> None:
        config.use_judge = False
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        req = YouSearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_model_response("Answer: Paris\nConfidence: 90%"),
            ground_truth="Paris",
            question="What is the capital of France?",
        )
        res = await server.verify(self._create_dummy_request(), req)
        assert res.reward == approx(1.0)
        assert res.extracted_final_answer == "Paris"

    async def test_verify_with_regex_verifier_no_match(self, config: YouSearchResourcesServerConfig) -> None:
        config.use_judge = False
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        req = YouSearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_model_response("I could not find it."),
            ground_truth="Paris",
            question="What is the capital of France?",
        )
        res = await server.verify(self._create_dummy_request(), req)
        assert res.reward == approx(0.0)
        assert res.extracted_final_answer == ""

    async def test_verify_releases_session_metrics(self, config: YouSearchResourcesServerConfig) -> None:
        """Otherwise the map grows by one entry per rollout for the process lifetime."""
        config.use_judge = False
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        server._post = AsyncMock(return_value={"results": {"web": []}})
        dummy_request = self._create_dummy_request()

        await server.web_search(dummy_request, YouSearchRequest(query="q"))
        assert "abcd" in server._session_id_to_metrics

        req = YouSearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_model_response("Answer: Paris\nConfidence: 90%"),
            ground_truth="Paris",
            question="q",
        )
        res = await server.verify(dummy_request, req)

        assert len(res.metrics.you_api_calls) == 1  # still attached to the response
        assert "abcd" not in server._session_id_to_metrics

    async def test_verify_keeps_session_metrics_when_dumping(self, config: YouSearchResourcesServerConfig) -> None:
        config.use_judge = False
        config.dump_session_id_to_metrics_on_exit = True
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        req = YouSearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_model_response("Answer: Paris\nConfidence: 90%"),
            ground_truth="Paris",
            question="q",
        )
        await server.verify(self._create_dummy_request(), req)
        assert "abcd" in server._session_id_to_metrics

    # ---- debug logging, lifespan, judge failure ----

    async def test_debug_mode_exercises_logging_paths(self, config: YouSearchResourcesServerConfig) -> None:
        """debug=True is the documented way to inspect a run; it must not crash."""
        config.debug = True
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        dummy_request = self._create_dummy_request()

        with patch("resources_servers.you_search.app.request", AsyncMock()) as mock_request:
            mock_request.return_value = self._fake_response(
                200, {"results": {"web": [_web_result("https://example.com/p", "T")]}}
            )
            await server.web_search(dummy_request, YouSearchRequest(query="q"))

        server._post = AsyncMock(return_value=[{"markdown": "line one\nline two"}])
        await server.find_in_page(dummy_request, FindInPageRequest(url="https://example.com/p", query="one"))
        # Second fetch hits the cache, covering the cache-hit branch.
        await server.scroll_page(dummy_request, ScrollPageRequest(url="https://example.com/p", n=5))
        assert server._post.call_count == 1

    async def test_lifespan_dumps_session_metrics(self, config: YouSearchResourcesServerConfig, tmp_path) -> None:
        config.dump_session_id_to_metrics_on_exit = True
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        server._session_id_to_metrics["abcd"].you_api_calls.append(
            YouSearchSingleAPICallMetrics(function="search", status="success", start_time=0.0, end_time=1.0)
        )

        app = server.setup_webserver()
        out_file = tmp_path / "session_id_metrics.json"
        with patch("resources_servers.you_search.app.Path") as mock_path:
            mock_path.return_value.parent.__truediv__.return_value = out_file
            async with app.router.lifespan_context(app):
                pass

        dumped = orjson.loads(out_file.read_bytes())
        assert dumped["abcd"]["you_api_calls"][0]["time_taken"] == approx(1.0)

    async def test_verify_raises_judge_error(self, config: YouSearchResourcesServerConfig) -> None:
        """A judge that fails must surface, not be scored as a wrong answer."""
        server = YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        req = YouSearchVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=self._create_model_response("Paris"),
            ground_truth="Paris",
            question="capital of France?",
        )
        with patch(
            "resources_servers.you_search.app.call_judge", AsyncMock(side_effect=JudgeError("judge unreachable"))
        ):
            with raises(JudgeError):
                await server.verify(self._create_dummy_request(), req)
