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
"""Tests for the Exa search provider in browsecomp_advanced_harness.

The Exa path mirrors the bc_frankie reference: search returns highlight snippets
INLINE (never written to pages/, even in terminal mode); browse fetches full text
via /contents and reuses the existing disk/inline formatting. exclude_domains are
honored. Per-call metering records provider + function for the cost/latency summary.
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
    BrowseRequest,
    ExaAIOHTTPClient,
    TavilySearchAIOHTTPClient,
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json")


class TestExaProvider:
    @fixture
    def config(self) -> TavilySearchResourcesServerConfig:
        return TavilySearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            search_provider="exa",
            exa_api_key="test_exa_key",  # pragma: allowlist secret
            exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
        )

    @fixture
    def server(self, config: TavilySearchResourcesServerConfig) -> TavilySearchResourcesServer:
        return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _req(self) -> MagicMock:
        m = MagicMock()
        m.session = {SESSION_ID_KEY: "test_session_id"}
        return m

    def _exa_server_per_session(self, ws_root: str) -> TavilySearchResourcesServer:
        config = TavilySearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            search_provider="exa",
            exa_api_key="test_exa_key",  # pragma: allowlist secret
            exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            workspace="per_session",
            workspace_root=ws_root,
        )
        return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    # ---- config validation ----

    def test_config_requires_exa_key(self) -> None:
        with pytest.raises(ValueError):
            TavilySearchResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                search_provider="exa",
                exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            )

    def test_config_requires_tavily_key(self) -> None:
        with pytest.raises(ValueError):
            TavilySearchResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                search_provider="tavily",
                exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            )

    # ---- search ----

    async def test_exa_search_highlights(self, server: TavilySearchResourcesServer) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={
                "results": [
                    {"title": "T1", "url": "https://x.com", "highlights": ["foo", "bar"]},
                ]
            }
        )
        server._exa_clients = [mock]

        resp = await server.search(self._req(), TavilySearchRequest(queries=["who won"]))
        mock.search.assert_called_once()
        assert "[Search Query]: who won" in resp.results_string
        assert "[Title]: T1" in resp.results_string
        assert "[URL]: https://x.com" in resp.results_string
        assert "[Snippet]: foo ... bar" in resp.results_string
        # highlights-only: never the inline full-content marker
        assert "[Content]" not in resp.results_string

    async def test_exa_search_passes_exclude_domains(self, server: TavilySearchResourcesServer) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": []})
        server._exa_clients = [mock]

        await server.search(self._req(), TavilySearchRequest(queries=["q"]))
        _, kwargs = mock.search.call_args
        assert "blacklisteddomain.com" in (kwargs.get("exclude_domains") or [])

    async def test_exa_search_budget_drops_oversized(self, server: TavilySearchResourcesServer) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={"results": [{"title": "BIG", "url": "https://x.com", "highlights": ["z" * 500]}]}
        )
        server._exa_clients = [mock]

        resp = await server.search(self._req(), TavilySearchRequest(queries=["q"], max_total_length=50))
        # entry exceeds the 50-char per-query budget -> dropped; only the query header remains
        assert "BIG" not in resp.results_string
        assert "[Search Query]: q" in resp.results_string

    async def test_exa_search_no_page_writes(self, tmp_path) -> None:
        server = self._exa_server_per_session(str(tmp_path))
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={"results": [{"title": "T", "url": "https://x.com", "highlights": ["hl"]}]}
        )
        server._exa_clients = [mock]

        server._get_page_writer("test_session_id")  # create the workspace
        await server.search(self._req(), TavilySearchRequest(queries=["q"]))

        pages = list((Path(tmp_path) / "test_session_id" / "pages").iterdir())
        assert pages == []  # exa search is highlights-only, never writes pages

    # ---- browse ----

    async def test_exa_browse_inline(self, server: TavilySearchResourcesServer) -> None:
        mock = MagicMock()
        mock.get_contents = AsyncMock(return_value={"results": [{"url": "https://x.com", "text": "FULL BODY TEXT"}]})
        server._exa_clients = [mock]

        resp = await server.browse(self._req(), BrowseRequest(urls=["https://x.com"]))
        mock.get_contents.assert_called_once()
        assert "[URL]: https://x.com" in resp.results_string
        assert "FULL BODY TEXT" in resp.results_string

    async def test_exa_browse_page_writer(self, tmp_path) -> None:
        server = self._exa_server_per_session(str(tmp_path))
        mock = MagicMock()
        mock.get_contents = AsyncMock(return_value={"results": [{"url": "https://x.com", "text": "PAGE CONTENT"}]})
        server._exa_clients = [mock]
        server._get_page_writer("test_session_id")

        resp = await server.browse(self._req(), BrowseRequest(urls=["https://x.com"]))
        assert "[Saved to]:" in resp.results_string
        assert "[Preview]:" in resp.results_string
        browse_files = list((Path(tmp_path) / "test_session_id" / "pages").glob("*browse*"))
        assert len(browse_files) == 1
        assert "PAGE CONTENT" in browse_files[0].read_text()

    async def test_exa_browse_excluded_urls(self, server: TavilySearchResourcesServer) -> None:
        resp = await server.browse(self._req(), BrowseRequest(urls=["https://blacklisteddomain.com/page"]))
        assert "Error: no URLs provided." in resp.results_string

    async def test_exa_browse_failure(self, server: TavilySearchResourcesServer) -> None:
        mock = MagicMock()
        mock.get_contents = AsyncMock(side_effect=Exception("exa boom"))
        server._exa_clients = [mock]

        resp = await server.browse(self._req(), BrowseRequest(urls=["https://x.com"]))
        assert "Failed to extract content" in resp.results_string

    # ---- invalid API key ----

    @staticmethod
    def _http_response(status: int, body: dict):
        r = MagicMock()
        r.status = status
        r.content.read = AsyncMock(return_value=json.dumps(body).encode())
        r.json = AsyncMock(return_value=body)
        return r

    async def test_exa_401_aborts_benchmark(self, monkeypatch) -> None:
        client = ExaAIOHTTPClient(headers={}, base_url="https://api.exa.ai", debug=False)
        fake_request = AsyncMock(return_value=self._http_response(401, {"error": "invalid key"}))
        monkeypatch.setattr(app_module, "request", fake_request)
        monkeypatch.setattr(app_module.os, "_exit", MagicMock(side_effect=SystemExit(1)))

        with pytest.raises(SystemExit):
            await client.search("q", num_results=5)
        app_module.os._exit.assert_called_once_with(1)
        fake_request.assert_awaited_once()  # no retries on auth failure

    async def test_tavily_403_aborts_benchmark(self, monkeypatch) -> None:
        client = TavilySearchAIOHTTPClient(headers={}, base_url="https://api.tavily.com", debug=False)
        fake_request = AsyncMock(return_value=self._http_response(403, {"error": "forbidden"}))
        monkeypatch.setattr(app_module, "request", fake_request)
        monkeypatch.setattr(app_module.os, "_exit", MagicMock(side_effect=SystemExit(1)))

        with pytest.raises(SystemExit):
            await client.post("/search", {"query": "q"}, 30)
        app_module.os._exit.assert_called_once_with(1)
        fake_request.assert_awaited_once()

    # ---- dispatch ----

    async def test_dispatch_routes_to_exa_not_tavily(self, server: TavilySearchResourcesServer) -> None:
        exa = MagicMock()
        exa.search = AsyncMock(return_value={"results": []})
        tavily = MagicMock()
        tavily.search = AsyncMock(return_value={"results": []})
        server._exa_clients = [exa]
        server._async_tavily_clients = [tavily]

        await server.search(self._req(), TavilySearchRequest(queries=["q"]))
        exa.search.assert_called_once()
        tavily.search.assert_not_called()

    # ---- metering ----

    async def test_metering_one_record_per_query(self, server: TavilySearchResourcesServer) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": []})
        server._exa_clients = [mock]

        await server.search(self._req(), TavilySearchRequest(queries=["q1", "q2", "q3"]))
        recs = [
            c for c in server._session_id_to_metrics["test_session_id"].async_tavily_calls if c.function == "search"
        ]
        assert len(recs) == 3
        assert all(c.provider == "exa" for c in recs)
        assert all(c.time_taken is not None for c in recs)

    async def test_metering_browse_record(self, server: TavilySearchResourcesServer) -> None:
        mock = MagicMock()
        mock.get_contents = AsyncMock(return_value={"results": [{"url": "https://x.com", "text": "t"}]})
        server._exa_clients = [mock]

        await server.browse(self._req(), BrowseRequest(urls=["https://x.com"]))
        recs = [
            c for c in server._session_id_to_metrics["test_session_id"].async_tavily_calls if c.function == "browse"
        ]
        assert len(recs) == 1
        assert recs[0].provider == "exa"

    # ---- max_results config ----

    def _server_with_max_results(self, provider: str, max_results: int) -> TavilySearchResourcesServer:
        kwargs = dict(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            search_provider=provider,
            exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            max_results=max_results,
        )
        if provider == "exa":
            kwargs["exa_api_key"] = "test_exa_key"  # pragma: allowlist secret
        else:
            kwargs["tavily_api_key"] = "test_tavily_key"  # pragma: allowlist secret
        config = TavilySearchResourcesServerConfig(**kwargs)
        return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_max_results_config_default_is_5(self, config: TavilySearchResourcesServerConfig) -> None:
        assert config.max_results == 5

    async def test_exa_search_uses_configured_max_results(self) -> None:
        server = self._server_with_max_results("exa", 10)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": []})
        server._exa_clients = [mock]

        await server.search(self._req(), TavilySearchRequest(queries=["q"]))
        _, kwargs = mock.search.call_args
        assert kwargs.get("num_results") == 10

    async def test_tavily_search_uses_configured_max_results(self) -> None:
        server = self._server_with_max_results("tavily", 10)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": []})
        server._async_tavily_clients = [mock]

        await server.search(self._req(), TavilySearchRequest(queries=["q"]))
        _, kwargs = mock.search.call_args
        assert kwargs.get("max_results") == 10
