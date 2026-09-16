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
"""Tests for the You.com search provider in browsecomp_advanced_harness.

Mirrors test_exa_provider.py. You.com's /v1/search nests results under
results.web (unlike Tavily/Exa's flat "results" list), supports a "snippets"
arm as well as "eco" (a separate, lighter endpoint that ignores exclude_domains
server-side -- exclusions must still be enforced client-side). browse always
pulls full markdown via /v1/contents, which answers with a bare array.
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import fixture

import resources_servers.browsecomp_advanced_harness.app as app_module
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    BrowseCompResourcesServerConfig,
    BrowseRequest,
    SearchRequest,
    YouAIOHTTPClient,
    YouSearchResourcesServer,
)


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json")


class TestYouProvider:
    @fixture
    def config(self) -> BrowseCompResourcesServerConfig:
        return BrowseCompResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            search_provider="you",
            ydc_api_key="test_ydc_key",  # pragma: allowlist secret
            exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
        )

    @fixture
    def server(self, config: BrowseCompResourcesServerConfig) -> YouSearchResourcesServer:
        return YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _req(self) -> MagicMock:
        m = MagicMock()
        m.session = {SESSION_ID_KEY: "test_session_id"}
        return m

    def _you_server_per_session(self, ws_root: str, **overrides) -> YouSearchResourcesServer:
        config = BrowseCompResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            search_provider="you",
            ydc_api_key="test_ydc_key",  # pragma: allowlist secret
            exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            workspace="per_session",
            workspace_root=ws_root,
            **overrides,
        )
        return YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    # ---- client construction ----

    def test_you_client_sends_json_content_type(self, server: YouSearchResourcesServer) -> None:
        # Regression: without this header You.com's API can't parse the JSON body and
        # responds 422 "Input should be a valid dictionary or object to extract fields from".
        assert server._you_clients[0].headers.get("Content-Type") == "application/json"

    # ---- config validation ----

    def test_config_requires_you_key(self) -> None:
        with pytest.raises(ValueError):
            BrowseCompResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                search_provider="you",
                exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            )

    def test_config_rejects_unknown_provider(self) -> None:
        with pytest.raises(ValueError):
            BrowseCompResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                search_provider="bogus",
                exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            )

    # ---- search: snippets/highlights/full_page arms ----

    async def test_you_search_snippets(self, server: YouSearchResourcesServer) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={"results": {"web": [{"title": "T1", "url": "https://x.com", "snippets": ["foo", "bar"]}]}}
        )
        server._you_clients = [mock]

        resp = await server.search(self._req(), SearchRequest(queries=["who won"]))
        mock.search.assert_called_once()
        assert "[Search Query]: who won" in resp.results_string
        assert "[Title]: T1" in resp.results_string
        assert "[URL]: https://x.com" in resp.results_string
        assert "[Snippet]: foo ... bar" in resp.results_string

    async def test_you_search_falls_back_to_description(self, server: YouSearchResourcesServer) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={"results": {"web": [{"title": "T1", "url": "https://x.com", "description": "desc only"}]}}
        )
        server._you_clients = [mock]

        resp = await server.search(self._req(), SearchRequest(queries=["q"]))
        assert "[Snippet]: desc only" in resp.results_string

    async def test_you_search_prefers_markdown_over_snippets(self, server: YouSearchResourcesServer) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={
                "results": {
                    "web": [
                        {
                            "title": "T1",
                            "url": "https://x.com",
                            "snippets": ["ignored"],
                            "contents": {"markdown": "FULL PAGE"},
                        }
                    ]
                }
            }
        )
        server._you_clients = [mock]

        resp = await server.search(self._req(), SearchRequest(queries=["q"]))
        assert "[Snippet]: FULL PAGE" in resp.results_string
        assert "ignored" not in resp.results_string

    async def test_you_search_mode_selects_endpoint_and_extraction(self) -> None:
        highlights_server = self._config_server(you_search_mode="highlights")
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": {"web": []}})
        highlights_server._you_clients = [mock]

        await highlights_server.search(self._req(), SearchRequest(queries=["q"]))
        _, kwargs = mock.search.call_args
        assert kwargs.get("mode") == "highlights"

    def _config_server(self, **overrides) -> YouSearchResourcesServer:
        config = BrowseCompResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            search_provider="you",
            ydc_api_key="test_ydc_key",  # pragma: allowlist secret
            exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            **overrides,
        )
        return YouSearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_you_search_eco_payload_is_minimal(self) -> None:
        server = self._config_server(you_search_mode="eco")
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": {"web": []}})
        server._you_clients = [mock]

        await server.search(self._req(), SearchRequest(queries=["q"]))
        _, kwargs = mock.search.call_args
        assert kwargs.get("mode") == "eco"

    async def test_you_eco_still_filters_excluded_domains_client_side(self) -> None:
        server = self._config_server(you_search_mode="eco")
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={
                "results": {
                    "web": [
                        {"title": "Blocked", "url": "https://blacklisteddomain.com/page", "snippets": ["x"]},
                        {"title": "OK", "url": "https://x.com", "snippets": ["y"]},
                    ]
                }
            }
        )
        server._you_clients = [mock]

        resp = await server.search(self._req(), SearchRequest(queries=["q"]))
        assert "Blocked" not in resp.results_string
        assert "OK" in resp.results_string

    async def test_you_search_passes_exclude_domains(self, server: YouSearchResourcesServer) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": {"web": []}})
        server._you_clients = [mock]

        await server.search(self._req(), SearchRequest(queries=["q"]))
        _, kwargs = mock.search.call_args
        assert "blacklisteddomain.com" in (kwargs.get("exclude_domains") or [])

    async def test_you_search_budget_drops_oversized(self, server: YouSearchResourcesServer) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={"results": {"web": [{"title": "BIG", "url": "https://x.com", "snippets": ["z" * 500]}]}}
        )
        server._you_clients = [mock]

        resp = await server.search(self._req(), SearchRequest(queries=["q"], max_total_length=50))
        assert "BIG" not in resp.results_string
        assert "[Search Query]: q" in resp.results_string

    async def test_you_search_writes_pages_in_terminal_mode(self, tmp_path) -> None:
        server = self._you_server_per_session(str(tmp_path))
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={"results": {"web": [{"title": "T", "url": "https://x.com", "snippets": ["hl"]}]}}
        )
        server._you_clients = [mock]

        server._get_page_writer("test_session_id")  # create the workspace
        resp = await server.search(self._req(), SearchRequest(queries=["q"]))

        pages = list((Path(tmp_path) / "test_session_id" / "pages").iterdir())
        assert len(pages) == 1
        assert "[Saved to]:" in resp.results_string

    # ---- browse ----

    async def test_you_browse_inline(self, server: YouSearchResourcesServer) -> None:
        mock = MagicMock()
        mock.get_contents = AsyncMock(return_value=[{"url": "https://x.com", "markdown": "FULL BODY TEXT"}])
        server._you_clients = [mock]

        resp = await server.browse(self._req(), BrowseRequest(urls=["https://x.com"]))
        mock.get_contents.assert_called_once()
        assert "[URL]: https://x.com" in resp.results_string
        assert "FULL BODY TEXT" in resp.results_string

    async def test_you_browse_page_writer(self, tmp_path) -> None:
        server = self._you_server_per_session(str(tmp_path))
        mock = MagicMock()
        mock.get_contents = AsyncMock(return_value=[{"url": "https://x.com", "markdown": "PAGE CONTENT"}])
        server._you_clients = [mock]
        server._get_page_writer("test_session_id")

        resp = await server.browse(self._req(), BrowseRequest(urls=["https://x.com"]))
        assert "[Saved to]:" in resp.results_string
        browse_files = list((Path(tmp_path) / "test_session_id" / "pages").glob("*browse*"))
        assert len(browse_files) == 1
        assert "PAGE CONTENT" in browse_files[0].read_text()

    async def test_you_browse_handles_plain_string_list(self, server: YouSearchResourcesServer) -> None:
        # Observed live shape: /v1/contents answers with content strings, positional
        # with the requested urls, not {"url","markdown"} dicts.
        mock = MagicMock()
        mock.get_contents = AsyncMock(return_value=["PLAIN STRING BODY"])
        server._you_clients = [mock]

        resp = await server.browse(self._req(), BrowseRequest(urls=["https://x.com"]))
        assert "[URL]: https://x.com" in resp.results_string
        assert "PLAIN STRING BODY" in resp.results_string

    async def test_you_browse_failure(self, server: YouSearchResourcesServer) -> None:
        mock = MagicMock()
        mock.get_contents = AsyncMock(side_effect=Exception("you boom"))
        server._you_clients = [mock]

        resp = await server.browse(self._req(), BrowseRequest(urls=["https://x.com"]))
        assert "Failed to extract content" in resp.results_string

    # ---- invalid API key / rate limits ----

    @staticmethod
    def _http_response(status: int, body: dict):
        r = MagicMock()
        r.status = status
        r.content.read = AsyncMock(return_value=json.dumps(body).encode())
        r.json = AsyncMock(return_value=body)
        return r

    async def test_you_401_aborts_benchmark(self, monkeypatch) -> None:
        client = YouAIOHTTPClient(headers={}, base_url="https://ydc-index.io", debug=False)
        fake_request = AsyncMock(return_value=self._http_response(401, {"error": "invalid key"}))
        monkeypatch.setattr(app_module, "request", fake_request)
        monkeypatch.setattr(app_module.os, "_exit", MagicMock(side_effect=SystemExit(1)))

        with pytest.raises(SystemExit):
            await client.search("q", num_results=5, mode="snippets", crawl_timeout=10)
        app_module.os._exit.assert_called_once_with(1)
        fake_request.assert_awaited_once()  # no retries on auth failure

    async def test_you_429_rotates_key_then_succeeds(self, monkeypatch) -> None:
        client = YouAIOHTTPClient(headers={}, base_url="https://ydc-index.io", debug=False)
        responses = [self._http_response(429, {"error": "rate limited"}), self._http_response(200, {"results": {}})]
        fake_request = AsyncMock(side_effect=responses)
        monkeypatch.setattr(app_module, "request", fake_request)
        monkeypatch.setattr(app_module, "sleep", AsyncMock())

        result = await client.search("q", num_results=5, mode="snippets", crawl_timeout=10)
        assert result == {"results": {}}
        assert fake_request.await_count == 2

    # ---- dispatch ----

    def test_select_server_class_picks_you(self) -> None:
        config = self._config_server().config
        assert app_module._select_server_class(config) is YouSearchResourcesServer

    def test_select_server_class_picks_tavily_class_for_tavily_and_exa(self) -> None:
        tavily_config = BrowseCompResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            search_provider="tavily",
            tavily_api_key="test_tavily_key",  # pragma: allowlist secret
            exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
        )
        assert app_module._select_server_class(tavily_config) is app_module.TavilySearchResourcesServer

        exa_config = BrowseCompResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            search_provider="exa",
            exa_api_key="test_exa_key",  # pragma: allowlist secret
            exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
        )
        assert app_module._select_server_class(exa_config) is app_module.TavilySearchResourcesServer

    # ---- metering ----

    async def test_metering_one_record_per_query(self, server: YouSearchResourcesServer) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": {"web": []}})
        server._you_clients = [mock]

        await server.search(self._req(), SearchRequest(queries=["q1", "q2", "q3"]))
        recs = [c for c in server._session_id_to_metrics["test_session_id"].provider_calls if c.function == "search"]
        assert len(recs) == 3
        assert all(c.provider == "you" for c in recs)
        assert all(c.time_taken is not None for c in recs)

    async def test_metering_browse_record(self, server: YouSearchResourcesServer) -> None:
        mock = MagicMock()
        mock.get_contents = AsyncMock(return_value=[{"url": "https://x.com", "markdown": "t"}])
        server._you_clients = [mock]

        await server.browse(self._req(), BrowseRequest(urls=["https://x.com"]))
        recs = [c for c in server._session_id_to_metrics["test_session_id"].provider_calls if c.function == "browse"]
        assert len(recs) == 1
        assert recs[0].provider == "you"

    # ---- max_results config ----

    async def test_you_search_uses_configured_max_results(self) -> None:
        server = self._config_server(max_results=10)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": {"web": []}})
        server._you_clients = [mock]

        await server.search(self._req(), SearchRequest(queries=["q"]))
        _, kwargs = mock.search.call_args
        assert kwargs.get("num_results") == 10
