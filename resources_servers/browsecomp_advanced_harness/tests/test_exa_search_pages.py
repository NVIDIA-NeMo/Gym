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
"""Tests for exa search writing results to the pages/ workspace.

The `search` tool description and the system prompt both tell the model that results
are saved to pages/<idx>_search_*.txt with a [Saved to] path readable via bash_command.
On the exa path that was false: highlights came back inline and no page was ever
written. Checked across 1,239 exa search outputs in a control run, ZERO contain
'[Saved to]'. The workspace affordance covered `browse` only, and bash_command was
called 3x less often than search with 4.0% of its calls hitting 'No such file'.

exa_search_writes_pages closes that gap: exa search asks for full text alongside
highlights, writes each result to pages/, and returns the same
title/url/snippet/[Saved to] shape the tavily disk path already returns.

It defaults to True so the prompt and the harness agree out of the box. It was
introduced defaulting to False, which left every exa run shipping a prompt the harness
did not honour, and a run had to know to opt in. MEASURED 2026-09-02 against the live
exa API: asking for text is NOT a dollar cost — costDollars is
{"total": 0.007, "search": {"neural": 0.007}} with and without text, since exa bills
per query, not per result. What it costs is response size and latency: ~243k characters
per 10-result query versus ~19k for highlights alone. Set it to False to get the
smaller response back.
"""

import os
from unittest.mock import AsyncMock, MagicMock

from pytest import fixture

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json")


def _result(title: str, text: str = "", highlight: str = "hl") -> dict:
    return {
        "title": title,
        "url": f"https://{title}.com",
        "highlights": [highlight],
        "text": text,
    }


class TestExaSearchPages:
    def _server(self, ws_root: str | None = None, **overrides) -> TavilySearchResourcesServer:
        extra = {}
        if ws_root is not None:
            extra = {"workspace": "per_session", "workspace_root": ws_root}
        config = TavilySearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            search_provider="exa",
            exa_api_key="test_exa_key",  # pragma: allowlist secret
            exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            **extra,
            **overrides,
        )
        return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    @fixture
    def req(self) -> MagicMock:
        m = MagicMock()
        m.session = {SESSION_ID_KEY: "test_session_id"}
        return m

    # ---- default: pages are written, so the prompt's promise holds ----

    async def test_default_writes_pages(self, req: MagicMock, tmp_path) -> None:
        """The prompt promises search results are saved to pages/.

        The default has to make that true. Otherwise every exa run ships a prompt
        that lies, and the workspace affordance covers `browse` only.
        """
        server = self._server(ws_root=str(tmp_path))
        assert server.config.exa_search_writes_pages is True
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": [_result("a", text="FULL TEXT")]})
        server._exa_clients = [mock]

        resp = await server.search(req, TavilySearchRequest(queries=["q"]))

        assert "[Saved to]" in resp.results_string
        assert list(tmp_path.rglob("*_search_*.txt"))

    async def test_default_requests_text_from_the_provider(self, req: MagicMock, tmp_path) -> None:
        """Full text is what gets written to pages/, so the default must ask for it."""
        server = self._server(ws_root=str(tmp_path))
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": [_result("a")]})
        server._exa_clients = [mock]

        await server.search(req, TavilySearchRequest(queries=["q"]))

        _, kwargs = mock.search.call_args
        assert kwargs.get("include_text")

    # ---- opt-out: highlights stay inline, nothing written ----

    async def test_opt_out_is_inline_and_writes_no_pages(self, req: MagicMock, tmp_path) -> None:
        server = self._server(ws_root=str(tmp_path), exa_search_writes_pages=False)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": [_result("a", text="FULL TEXT")]})
        server._exa_clients = [mock]

        resp = await server.search(req, TavilySearchRequest(queries=["q"]))

        assert "[Saved to]" not in resp.results_string
        assert not list(tmp_path.rglob("*_search_*.txt"))

    async def test_opt_out_does_not_request_text_from_the_provider(self, req: MagicMock, tmp_path) -> None:
        """Opting out keeps the ~13x smaller response for a run that does not want pages."""
        server = self._server(ws_root=str(tmp_path), exa_search_writes_pages=False)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": [_result("a")]})
        server._exa_clients = [mock]

        await server.search(req, TavilySearchRequest(queries=["q"]))

        _, kwargs = mock.search.call_args
        assert not kwargs.get("include_text")

    # ---- enabled: pages written, [Saved to] returned ----

    async def test_writes_a_page_per_result(self, req: MagicMock, tmp_path) -> None:
        server = self._server(ws_root=str(tmp_path), exa_search_writes_pages=True)
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={"results": [_result("a", text="ALPHA BODY"), _result("b", text="BETA BODY")]}
        )
        server._exa_clients = [mock]

        resp = await server.search(req, TavilySearchRequest(queries=["q"]))

        pages = sorted(p.name for p in tmp_path.rglob("*_search_*.txt"))
        assert len(pages) == 2
        assert resp.results_string.count("[Saved to]") == 2

    async def test_page_contains_the_full_text(self, req: MagicMock, tmp_path) -> None:
        server = self._server(ws_root=str(tmp_path), exa_search_writes_pages=True)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": [_result("a", text="NEEDLE IN THE BODY")]})
        server._exa_clients = [mock]

        await server.search(req, TavilySearchRequest(queries=["q"]))

        page = next(tmp_path.rglob("*_search_*.txt"))
        assert "NEEDLE IN THE BODY" in page.read_text()

    async def test_requests_text_from_the_provider_when_enabled(self, req: MagicMock, tmp_path) -> None:
        server = self._server(ws_root=str(tmp_path), exa_search_writes_pages=True)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": [_result("a", text="body")]})
        server._exa_clients = [mock]

        await server.search(req, TavilySearchRequest(queries=["q"]))

        _, kwargs = mock.search.call_args
        assert kwargs.get("include_text") is True

    async def test_snippet_still_returned_inline(self, req: MagicMock, tmp_path) -> None:
        """The model must keep the highlight snippet; the page is additional, not a swap."""
        server = self._server(ws_root=str(tmp_path), exa_search_writes_pages=True)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": [_result("a", text="body", highlight="SNIP")]})
        server._exa_clients = [mock]

        resp = await server.search(req, TavilySearchRequest(queries=["q"]))

        assert "[Snippet]: SNIP" in resp.results_string

    # ---- degenerate cases ----

    async def test_result_without_text_gets_no_saved_line(self, req: MagicMock, tmp_path) -> None:
        server = self._server(ws_root=str(tmp_path), exa_search_writes_pages=True)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": [_result("a", text="")]})
        server._exa_clients = [mock]

        resp = await server.search(req, TavilySearchRequest(queries=["q"]))

        assert "[Saved to]" not in resp.results_string
        assert not list(tmp_path.rglob("*_search_*.txt"))

    async def test_no_workspace_means_no_pages_even_when_enabled(self, req: MagicMock) -> None:
        """workspace='none' has no page writer; the flag must not crash, just stay inline."""
        server = self._server(exa_search_writes_pages=True)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": [_result("a", text="body")]})
        server._exa_clients = [mock]

        resp = await server.search(req, TavilySearchRequest(queries=["q"]))

        assert "[Saved to]" not in resp.results_string
        assert "[Title]: a" in resp.results_string

    async def test_oversized_text_is_capped_at_max_page_bytes(self, req: MagicMock, tmp_path) -> None:
        server = self._server(ws_root=str(tmp_path), exa_search_writes_pages=True, max_page_bytes=50)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": [_result("a", text="x" * 5000)]})
        server._exa_clients = [mock]

        await server.search(req, TavilySearchRequest(queries=["q"]))

        page = next(tmp_path.rglob("*_search_*.txt"))
        assert page.read_text().count("x") == 50
