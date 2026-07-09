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
import os
from contextlib import contextmanager
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, call

from pytest import approx, fixture, importorskip

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
        mock_backend.search = AsyncMock(return_value=mock_tavily_response)
        server._async_tavily_clients = [mock_backend]

        request = TavilySearchRequest(query="NVIDIA GPU programming")
        response = await server.web_search("abcd", request)

        mock_backend.search.assert_called_once()
        actual_call_args = mock_backend.search.call_args
        expected_call_args = call(
            "NVIDIA GPU programming",
            max_results=10,
            exclude_domains=["blacklisteddomain.com"],
            search_depth="advanced",
        )
        assert expected_call_args == actual_call_args

        # Response is now a formatted string, not JSON list
        assert "NVIDIA Documentation" in response.results_string
        assert "nvidia.com" in response.results_string

    async def test_web_search_none_query(self, server: TavilySearchResourcesServer) -> None:
        """Test web_search with None query returns error message."""
        request = TavilySearchRequest(query=None)
        response = await server.web_search("abcd", request)
        assert response.results_string == "Query is none"

    async def test_web_search_long_query(self, server: TavilySearchResourcesServer) -> None:
        """Test web_search with overly long query returns error message."""
        request = TavilySearchRequest(query="x" * 401)
        response = await server.web_search("abcd", request)
        assert response.results_string == "Query is too long"

    # ---- find_in_page ----

    async def test_find_in_page_none_url(self, server: TavilySearchResourcesServer) -> None:
        """Test find_in_page with None URL returns error."""
        request = FindInPageRequest(url=None, query="test")
        response = await server.find_in_page("abcd", request)
        assert response.results_string == "URL is none"

    async def test_find_in_page_none_query(self, server: TavilySearchResourcesServer) -> None:
        """Test find_in_page with None query returns error."""
        request = FindInPageRequest(url="https://example.com", query=None)
        response = await server.find_in_page("abcd", request)
        assert response.results_string == "Query is none"

    async def test_find_in_page_excluded_domain(self, server: TavilySearchResourcesServer) -> None:
        """Test find_in_page with excluded domain returns error."""
        request = FindInPageRequest(url="https://blacklisteddomain.com/page", query="test")
        response = await server.find_in_page("abcd", request)
        assert response.results_string == "URL is in excluded domains"

    # ---- scroll_page ----

    async def test_scroll_page_none_url(self, server: TavilySearchResourcesServer) -> None:
        """Test scroll_page with None URL."""
        request = ScrollPageRequest(url=None)
        response = await server.scroll_page("abcd", request)
        assert response.results_string == "URL is none"
        assert response.total_words == 0

    async def test_scroll_page_excluded_domain(self, server: TavilySearchResourcesServer) -> None:
        """Test scroll_page with excluded domain."""
        request = ScrollPageRequest(url="https://blacklisteddomain.com/page")
        response = await server.scroll_page("abcd", request)
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

        await server.web_search("abcd", request)
        assert mock_backend1.search.call_count == 1

        await server.web_search("abcd", request)
        assert mock_backend2.search.call_count == 1

        await server.web_search("abcd", request)
        assert mock_backend3.search.call_count == 1

        await server.web_search("abcd", request)
        assert mock_backend1.search.call_count == 2

        await server.web_search("abcd", request)
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

        await server.web_search("abcd", request)
        await server.web_search("abcd", request)
        await server.web_search("abcd", request)
        await server.web_search("abcd", request)
        await server.web_search("abcd", request)

        expected_metrics_length = 5
        actual_metrics_length = len(server._session_id_to_metrics["abcd"].async_tavily_calls)
        assert expected_metrics_length == actual_metrics_length


# --------------------------------------------------------------------------------------------
# Dual-transport suite: HTTP wire replay (byte-equal), MCP round trip, transport parity.
# All Tavily backends are mocked — no test here ever hits the real Tavily API.
# --------------------------------------------------------------------------------------------

RPC_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
EXPECTED_TOOLS = {"web_search", "find_in_page", "scroll_page"}
SEARCH_ANSWER_STRING = "Search Answer\n==============\nParis.\n"


def _json_bytes(payload: Any) -> bytes:
    """Starlette JSONResponse byte encoding (compact separators, non-ASCII preserved)."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")).encode("utf-8")


def _make_wire_server() -> TavilySearchResourcesServer:
    config = TavilySearchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="tavily_search",
        tavily_api_key="test_api_key",  # pragma: allowlist secret
        exclude_domains_file_path=os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json"),
        judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
        judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )
    server = TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    backend = MagicMock()
    backend.search = AsyncMock(return_value={"answer": "Paris."})
    backend.extract = AsyncMock(return_value={"results": [{"raw_content": "alpha beta gamma delta epsilon zeta"}]})
    server._async_tavily_clients = [backend]
    return server


@contextmanager
def _wire_client():
    """In-memory TestClient over the full app (no sockets, no fixed ports)."""
    importorskip("mcp")
    from fastapi.testclient import TestClient

    server = _make_wire_server()
    app = server.setup_webserver()
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        yield server, app, client


def _rpc(client, method: str, params: Optional[dict] = None, token: Optional[str] = None, rpc_id: int = 1):
    headers = dict(RPC_HEADERS)
    if token is not None:
        headers[TOKEN_HEADER] = token
    payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if method.startswith("notifications/"):
        payload["params"] = params or {}
    else:
        payload["id"] = rpc_id
        payload["params"] = params or {}
    return client.post("/mcp", headers=headers, json=payload, follow_redirects=False)


def _list_tools(client, token: Optional[str] = None) -> dict[str, dict]:
    resp = _rpc(client, "tools/list", token=token, rpc_id=2)
    assert resp.status_code == 200, resp.text
    return {tool["name"]: tool for tool in resp.json()["result"]["tools"]}


def _mcp_call(client, name: str, arguments: dict, token: Optional[str] = None) -> dict:
    resp = _rpc(client, "tools/call", {"name": name, "arguments": arguments}, token=token, rpc_id=3)
    assert resp.status_code == 200, resp.text
    return resp.json()["result"]


def _seed(client) -> str:
    resp = client.post("/seed_session", json={})
    assert resp.status_code == 200, resp.text
    return resp.json()["mcp"]["headers"][TOKEN_HEADER]


class TestHTTPWireReplay:
    """Byte-equal replay of the pre-migration HTTP wire format (captured before the gym_tool move)."""

    def test_web_search_answer_bytes(self) -> None:
        with _wire_client() as (_server, _app, client):
            resp = client.post("/web_search", json={"query": "capital of France"})
            assert resp.status_code == 200
            assert resp.content == _json_bytes({"results_string": SEARCH_ANSWER_STRING})

    def test_web_search_query_none_soft_error_bytes(self) -> None:
        with _wire_client() as (_server, _app, client):
            resp = client.post("/web_search", json={})
            assert resp.status_code == 200
            assert resp.content == _json_bytes({"results_string": "Query is none"})

    def test_web_search_query_too_long_soft_error_bytes(self) -> None:
        with _wire_client() as (_server, _app, client):
            resp = client.post("/web_search", json={"query": "x" * 401})
            assert resp.status_code == 200
            assert resp.content == _json_bytes({"results_string": "Query is too long"})

    def test_web_search_bad_type_is_422_with_body_loc(self) -> None:
        with _wire_client() as (_server, _app, client):
            resp = client.post("/web_search", json={"query": 123})
            assert resp.status_code == 422
            assert resp.json()["detail"][0]["loc"] == ["body", "query"]

    def test_find_in_page_bytes(self) -> None:
        with _wire_client() as (_server, _app, client):
            resp = client.post("/find_in_page", json={"url": "https://example.com/p", "query": "beta"})
            assert resp.status_code == 200
            assert resp.content == _json_bytes(
                {
                    "results_string": "Content from: example.com\n"
                    "URL: https://example.com/p\n"
                    'Query: "beta"\n'
                    "========================================\n"
                    "L0: alpha beta gamma delta epsilon zeta"
                }
            )

    def test_find_in_page_excluded_domain_soft_error_bytes(self) -> None:
        with _wire_client() as (_server, _app, client):
            resp = client.post("/find_in_page", json={"url": "https://blacklisteddomain.com/p", "query": "q"})
            assert resp.status_code == 200
            assert resp.content == _json_bytes({"results_string": "URL is in excluded domains"})

    def test_scroll_page_bytes(self) -> None:
        with _wire_client() as (_server, _app, client):
            resp = client.post("/scroll_page", json={"url": "https://example.com/p", "start_index": 1, "n": 3})
            assert resp.status_code == 200
            assert resp.content == _json_bytes(
                {
                    "results_string": "Page content from: example.com\n"
                    "URL: https://example.com/p\n"
                    "Showing words [1-4] of 6\n"
                    "========================================\n"
                    "L0: beta gamma delta",
                    "total_words": 6,
                }
            )

    def test_scroll_page_url_none_soft_error_bytes(self) -> None:
        with _wire_client() as (_server, _app, client):
            resp = client.post("/scroll_page", json={})
            assert resp.status_code == 200
            assert resp.content == _json_bytes({"results_string": "URL is none", "total_words": 0})

    def test_metrics_recorded_under_the_cookie_session(self) -> None:
        with _wire_client() as (server, _app, client):
            _seed(client)
            client.post("/web_search", json={"query": "a"})
            client.post("/web_search", json={"query": "b"})
            (sid,) = server._session_id_to_metrics.keys()
            assert len(server._session_id_to_metrics[sid].async_tavily_calls) == 2

    def test_seed_session_now_carries_mcp_metadata(self) -> None:
        with _wire_client() as (_server, _app, client):
            body = client.post("/seed_session", json={}).json()
            assert body["mcp"]["url_path"] == "/mcp"
            assert TOKEN_HEADER in body["mcp"]["headers"]

    def test_unknown_tool_lists_available_tools(self) -> None:
        with _wire_client() as (_server, _app, client):
            resp = client.post("/no_such_tool", json={})
            assert resp.status_code == 404
            assert "Available tools" in resp.json()["error"]
            assert "web_search" in resp.json()["error"]


class TestMCPRoundTrip:
    def test_tools_list_names_schemas_and_call(self) -> None:
        with _wire_client() as (_server, _app, client):
            token = _seed(client)
            tools = _list_tools(client, token=token)
            assert set(tools) == EXPECTED_TOOLS
            # session_id is never model-visible in the advertised schemas.
            for tool in tools.values():
                assert "session_id" not in tool["inputSchema"].get("properties", {})
            assert set(tools["scroll_page"]["inputSchema"]["properties"]) == {"url", "start_index", "n"}

            result = _mcp_call(client, "web_search", {"query": "capital of France"}, token=token)
            assert result.get("isError") is not True
            assert result["structuredContent"] == {"results_string": SEARCH_ANSWER_STRING}

    def test_call_without_token_is_clean_tool_error(self) -> None:
        with _wire_client() as (_server, _app, client):
            result = _mcp_call(client, "web_search", {"query": "q"}, token=None)
            assert result["isError"] is True
            assert TOKEN_HEADER in result["content"][0]["text"]

    def test_http_cookie_and_mcp_token_share_one_metrics_session(self) -> None:
        with _wire_client() as (server, _app, client):
            token = _seed(client)
            client.post("/web_search", json={"query": "over http"})
            result = _mcp_call(client, "web_search", {"query": "over mcp"}, token=token)
            assert result.get("isError") is not True
            (sid,) = server._session_id_to_metrics.keys()
            assert len(server._session_id_to_metrics[sid].async_tavily_calls) == 2


class TestTransportParity:
    NON_TOOL_PATHS = {"/seed_session", "/verify", "/aggregate_metrics", "/mcp", "/{tool_name}"}

    def test_tool_sets_identical_across_transports(self) -> None:
        with _wire_client() as (_server, app, client):
            mcp_names = set(_list_tools(client))

            http_names = set()
            for route in app.router.routes:
                path = getattr(route, "path", None)
                methods = getattr(route, "methods", None) or set()
                if path and "POST" in methods and path not in self.NON_TOOL_PATHS:
                    http_names.add(path.lstrip("/"))

            assert mcp_names == http_names == EXPECTED_TOOLS
