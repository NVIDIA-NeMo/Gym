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

import orjson
from pytest import approx, fixture

from nemo_gym.server_utils import SESSION_ID_KEY


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))

from nemo_gym.config_types import ModelServerRef
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
            you_api_key="test_api_key",  # pragma: allowlist secret
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
        config.you_api_key = ["key1", "key2", "key3"]  # pragma: allowlist secret
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
