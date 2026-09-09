# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Parallel Search and Extract provider."""

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import fixture

import resources_servers.browsecomp_advanced_harness.app as app_module
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    BrowseRequest,
    ParallelAIOHTTPClient,
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json")


def _http_response(status: int, body: dict):
    response = MagicMock()
    response.status = status
    response.content.read = AsyncMock(return_value=json.dumps(body).encode())
    response.json = AsyncMock(return_value=body)
    return response


@fixture(autouse=True)
def _clear_retry_counts():
    app_module._PROVIDER_RETRY_COUNTS.set(None)
    yield
    app_module._PROVIDER_RETRY_COUNTS.set(None)


class TestParallelProvider:
    @fixture
    def config(self) -> TavilySearchResourcesServerConfig:
        return TavilySearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            search_provider="parallel",
            parallel_api_key="test_parallel_key",  # pragma: allowlist secret
            parallel_search_mode="fast",
            exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
        )

    @fixture
    def server(self, config: TavilySearchResourcesServerConfig) -> TavilySearchResourcesServer:
        return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _req(self) -> MagicMock:
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "test_session_id"}
        return request

    def test_config_requires_parallel_key(self) -> None:
        with pytest.raises(ValueError, match="parallel_api_key"):
            TavilySearchResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                search_provider="parallel",
                exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            )

    def test_config_rejects_unknown_parallel_mode(self) -> None:
        with pytest.raises(ValueError, match="parallel_search_mode"):
            TavilySearchResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                search_provider="parallel",
                parallel_api_key="test_parallel_key",  # pragma: allowlist secret
                parallel_search_mode="unknown",
                exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            )

    def test_parallel_keys_round_robin(self, config: TavilySearchResourcesServerConfig) -> None:
        config.parallel_api_key = ["key_one", "key_two"]  # pragma: allowlist secret
        server = TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        assert server._select_parallel_client().headers["x-api-key"] == "key_one"
        assert server._select_parallel_client().headers["x-api-key"] == "key_two"
        assert server._select_parallel_client().headers["x-api-key"] == "key_one"

    async def test_search_request_sets_excerpt_cap(self, monkeypatch) -> None:
        client = ParallelAIOHTTPClient(headers={"x-api-key": "test"})  # pragma: allowlist secret
        fake_request = AsyncMock(return_value=_http_response(200, {"results": []}))
        monkeypatch.setattr(app_module, "request", fake_request)

        await client.search("who won", num_results=10, mode="fast", exclude_domains=["example.com"])

        request_body = json.loads(fake_request.await_args.kwargs["data"])
        assert fake_request.await_args.kwargs["url"] == "https://api.parallel.ai/v1/search"
        assert request_body == {
            "search_queries": ["who won"],
            "mode": "fast",
            "advanced_settings": {
                "max_results": 10,
                "excerpt_settings": {"max_chars_per_result": 2000},
                "source_policy": {"exclude_domains": ["example.com"]},
            },
        }

    async def test_extract_request_sets_full_content_cap(self, monkeypatch) -> None:
        client = ParallelAIOHTTPClient(headers={"x-api-key": "test"})  # pragma: allowlist secret
        fake_request = AsyncMock(return_value=_http_response(200, {"results": []}))
        monkeypatch.setattr(app_module, "request", fake_request)

        await client.extract(
            ["https://example.com"],
            max_characters=10000,
            objective="find the answer",
        )

        request_body = json.loads(fake_request.await_args.kwargs["data"])
        assert fake_request.await_args.kwargs["url"] == "https://api.parallel.ai/v1/extract"
        assert request_body == {
            "urls": ["https://example.com"],
            "advanced_settings": {"full_content": {"max_chars_per_result": 10000}},
            "objective": "find the answer",
        }

    async def test_search_formats_parallel_excerpts(self, server: TavilySearchResourcesServer) -> None:
        parallel = MagicMock()
        parallel.search = AsyncMock(
            return_value={
                "results": [
                    {
                        "title": "Result title",
                        "url": "https://example.com/result",
                        "excerpts": ["first", "second"],
                    }
                ]
            }
        )
        server._parallel_clients = [parallel]

        response = await server.search(self._req(), TavilySearchRequest(queries=["who won"]))

        parallel.search.assert_awaited_once_with(
            "who won",
            num_results=5,
            mode="fast",
            exclude_domains=["blacklisteddomain.com"],
        )
        assert "[Search Query]: who won" in response.results_string
        assert "[Title]: Result title" in response.results_string
        assert "[URL]: https://example.com/result" in response.results_string
        assert "[Snippet]: first ... second" in response.results_string

    async def test_search_failure_is_returned_to_model(self, server: TavilySearchResourcesServer) -> None:
        parallel = MagicMock()
        parallel.search = AsyncMock(side_effect=RuntimeError("search failed"))
        server._parallel_clients = [parallel]

        response = await server.search(self._req(), TavilySearchRequest(queries=["who won"]))

        assert response.results_string == "Search failed: search failed"
        [record] = server._session_id_to_metrics["test_session_id"].async_tavily_calls
        assert record.function == "search"
        assert record.provider == "parallel"
        assert record.status == "error"

    async def test_browse_uses_parallel_full_content(self, server: TavilySearchResourcesServer) -> None:
        parallel = MagicMock()
        parallel.extract = AsyncMock(
            return_value={"results": [{"url": "https://example.com", "full_content": "FULL PAGE"}]}
        )
        server._parallel_clients = [parallel]

        response = await server.browse(
            self._req(), BrowseRequest(urls=["https://example.com"], goal="find the answer")
        )

        parallel.extract.assert_awaited_once_with(
            urls=["https://example.com"],
            max_characters=30000,
            objective="find the answer",
        )
        assert "[URL]: https://example.com" in response.results_string
        assert "FULL PAGE" in response.results_string
        records = server._session_id_to_metrics["test_session_id"].async_tavily_calls
        assert len(records) == 1
        assert records[0].function == "browse"
        assert records[0].provider == "parallel"

    async def test_browse_failure_is_returned_to_model(self, server: TavilySearchResourcesServer) -> None:
        parallel = MagicMock()
        parallel.extract = AsyncMock(side_effect=RuntimeError("extract failed"))
        server._parallel_clients = [parallel]

        response = await server.browse(self._req(), BrowseRequest(urls=["https://example.com"]))

        assert response.results_string == "Failed to extract content: extract failed"
        [record] = server._session_id_to_metrics["test_session_id"].async_tavily_calls
        assert record.function == "browse"
        assert record.provider == "parallel"
        assert record.status == "error"

    async def test_parallel_429_is_counted(self, monkeypatch) -> None:
        client = ParallelAIOHTTPClient(headers={})
        fake_request = AsyncMock(side_effect=[_http_response(429, {}), _http_response(200, {"results": []})])
        monkeypatch.setattr(app_module, "request", fake_request)

        await client.search("q", num_results=5)

        assert app_module._PROVIDER_RETRY_COUNTS.get() == {
            "num_429_retries": 1,
            "num_other_retries": 0,
        }
