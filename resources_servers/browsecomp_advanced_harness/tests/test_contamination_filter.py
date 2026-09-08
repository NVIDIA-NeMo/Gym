# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Benchmark-contamination guard: drop provider results that reference BrowseComp.

A search or browse result that quotes the benchmark itself (a dataset mirror, a
leaderboard, simple-evals) hands the model the answer. Training on those rollouts
teaches retrieval of the answer key rather than research, and it surfaces later as
an unearned score on the benchmark being measured.

PER-ITEM, not per-response: one poisoned hit out of five costs that hit, not the
whole tool call, on a 60-turn rollout. The whole output is withheld only when
EVERY item is contaminated.

FOUR EXITS. The returned string is not the only way a result reaches the model --
in terminal mode `_search_one_to_disk` and `browse` write each raw page to
`pages/*.txt`, which the model later reads with grep/cat through the bash tool. A
filter applied only to the returned string would leave the contamination on disk
and fully readable. The four paths a provider result can leave by are:

    _exa_search_one · _postprocess_search_results (inline tavily)
    _search_one_to_disk (incl. page writes) · browse (incl. page writes)

Every test below drives the real public entry point (`search` / `browse`) with a
mocked provider client, so a new exit added later that bypasses the guard fails
here rather than leaking silently.
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import fixture

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    CONTAMINATED_MESSAGE,
    CONTAMINATION_PATTERNS,
    BrowseRequest,
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
    _drop_contaminated,
    _is_contaminated,
)


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json")


def _results() -> list[dict]:
    """Three tavily-shaped results, the middle one contaminated."""
    return [
        {"title": "clean one", "url": "https://a.example/1", "content": "alpha", "raw_content": "alpha body"},
        {
            "title": "BrowseComp dataset",
            "url": "https://hf.co/datasets/openai/BrowseComp",
            "content": "leak",
            "raw_content": "leak body",
        },
        {"title": "clean two", "url": "https://b.example/2", "content": "beta", "raw_content": "beta body"},
    ]


def _dirty_only() -> list[dict]:
    return [r for r in _results() if "Browse" in r["title"]]


def _req() -> MagicMock:
    m = MagicMock()
    m.session = {SESSION_ID_KEY: "test_session_id"}
    return m


def _server(provider: str, workspace_root: str = None) -> TavilySearchResourcesServer:
    kwargs = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        search_provider=provider,
        exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
    )
    if provider == "exa":
        kwargs["exa_api_key"] = "test_exa_key"  # pragma: allowlist secret
    else:
        kwargs["tavily_api_key"] = "test_tavily_key"  # pragma: allowlist secret
    if workspace_root is not None:
        kwargs["workspace"] = "per_session"
        kwargs["workspace_root"] = workspace_root
    config = TavilySearchResourcesServerConfig(**kwargs)
    return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestPatterns:
    def test_the_three_agreed_patterns(self) -> None:
        assert set(CONTAMINATION_PATTERNS) == {"browsecomp", "browse_comp", "simple-eval"}

    @pytest.mark.parametrize(
        "text",
        [
            "BrowseComp",
            "openai/browsecomp",  # 'browsecomp' already covers the org-prefixed form
            "huggingface.co/datasets/foo/browse_comp",
            "openai/simple-evals",  # 'simple-eval' covers the plural
            "SIMPLE-EVAL",
        ],
    )
    def test_matches_are_case_insensitive_substrings(self, text: str) -> None:
        assert _is_contaminated(text)

    @pytest.mark.parametrize(
        "text",
        ["browse the comp", "comprehensive browsing", "simple evaluation", "", None],
    )
    def test_innocuous_text_is_not_flagged(self, text) -> None:
        assert not _is_contaminated(text)

    def test_scans_every_field_including_nested_highlights(self) -> None:
        # Exa results carry their text in a `highlights` LIST, so a field-by-field
        # check that only looked at title/url/content would miss them entirely.
        assert _is_contaminated(json.dumps({"title": "x", "highlights": ["see browsecomp"]}))


class TestDropContaminated:
    def test_drops_only_the_offending_item(self) -> None:
        kept, dropped = _drop_contaminated(_results())

        assert dropped == 1
        assert [r["title"] for r in kept] == ["clean one", "clean two"]

    def test_all_clean_is_a_passthrough(self) -> None:
        clean = [r for r in _results() if "Browse" not in r["title"]]

        kept, dropped = _drop_contaminated(clean)

        assert dropped == 0 and kept == clean

    def test_empty_list_is_safe(self) -> None:
        assert _drop_contaminated([]) == ([], 0)


class TestInlineTavilySearchFilters:
    """Exit 1: `_postprocess_search_results`, the inline (no workspace) tavily path."""

    @fixture
    def server(self) -> TavilySearchResourcesServer:
        return _server("tavily")

    async def test_contaminated_result_is_absent_from_the_output(self, server) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": _results()})
        server._async_tavily_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))

        assert "clean one" in resp.results_string and "clean two" in resp.results_string
        assert "BrowseComp" not in resp.results_string and "leak" not in resp.results_string

    async def test_all_contaminated_withholds_the_whole_output(self, server) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": _dirty_only()})
        server._async_tavily_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))

        assert resp.results_string == CONTAMINATED_MESSAGE


class TestExaSearchFilters:
    """Exit 2: `_exa_search_one`, highlights returned inline."""

    @fixture
    def server(self) -> TavilySearchResourcesServer:
        return _server("exa")

    async def test_contaminated_highlight_is_dropped(self, server) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={
                "results": [
                    {"title": "clean", "url": "https://a.example/1", "highlights": ["fine"]},
                    {"title": "x", "url": "https://b.example/2", "highlights": ["from the simple-evals repo"]},
                ]
            }
        )
        server._exa_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))

        assert "clean" in resp.results_string
        assert "simple-evals" not in resp.results_string

    async def test_all_contaminated_withholds_the_whole_output(self, server) -> None:
        mock = MagicMock()
        mock.search = AsyncMock(
            return_value={"results": [{"title": "x", "url": "https://b/2", "highlights": ["browsecomp mirror"]}]}
        )
        server._exa_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))

        assert resp.results_string == CONTAMINATED_MESSAGE


class TestDiskSearchNeverWritesAContaminatedPage:
    """Exit 3: `_search_one_to_disk`. The one that matters most -- in terminal mode
    the model reads pages/*.txt with the bash tool, so a contaminated page written
    to disk is readable no matter what the tool output said."""

    async def test_page_is_not_written_for_the_contaminated_result(self, tmp_path) -> None:
        server = _server("tavily", str(tmp_path))
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": _results()})
        server._async_tavily_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))

        pages = sorted((Path(tmp_path) / "test_session_id" / "pages").iterdir())
        assert len(pages) == 2, f"expected 2 page writes, got {len(pages)}"
        on_disk = "\n".join(p.read_text() for p in pages).lower()
        assert "browsecomp" not in on_disk and "leak body" not in on_disk
        assert "clean one" in resp.results_string and "BrowseComp" not in resp.results_string

    async def test_all_contaminated_writes_nothing_and_withholds(self, tmp_path) -> None:
        server = _server("tavily", str(tmp_path))
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": _dirty_only()})
        server._async_tavily_clients = [mock]

        resp = await server.search(_req(), TavilySearchRequest(queries=["q"]))

        assert list((Path(tmp_path) / "test_session_id" / "pages").iterdir()) == []
        assert resp.results_string == CONTAMINATED_MESSAGE


class TestBrowseFilters:
    """Exit 4: `browse` has its own write loop and does not route through any of
    the search paths, so it is filtered separately -- above its page writes."""

    async def test_inline_browse_drops_the_contaminated_page(self) -> None:
        server = _server("tavily")
        mock = MagicMock()
        mock.extract = AsyncMock(
            return_value={
                "results": [
                    {"url": "https://a.example/1", "raw_content": "clean body"},
                    {"url": "https://hf.co/datasets/openai/BrowseComp", "raw_content": "leak body"},
                ]
            }
        )
        server._async_tavily_clients = [mock]

        resp = await server.browse(
            _req(), BrowseRequest(urls=["https://a.example/1", "https://hf.co/datasets/openai/BrowseComp"])
        )

        assert "clean body" in resp.results_string
        assert "BrowseComp" not in resp.results_string and "leak body" not in resp.results_string

    async def test_disk_browse_never_writes_the_contaminated_page(self, tmp_path) -> None:
        server = _server("tavily", str(tmp_path))
        mock = MagicMock()
        mock.extract = AsyncMock(
            return_value={
                "results": [
                    {"url": "https://a.example/1", "raw_content": "clean body"},
                    {"url": "https://b.example/2", "raw_content": "answers from openai/simple-evals"},
                ]
            }
        )
        server._async_tavily_clients = [mock]

        await server.browse(_req(), BrowseRequest(urls=["https://a.example/1", "https://b.example/2"]))

        pages = sorted((Path(tmp_path) / "test_session_id" / "pages").iterdir())
        assert len(pages) == 1
        assert "simple-evals" not in pages[0].read_text()

    async def test_all_contaminated_withholds_the_whole_output(self) -> None:
        server = _server("exa")
        mock = MagicMock()
        mock.get_contents = AsyncMock(
            return_value={"results": [{"url": "https://b.example/2", "text": "a browsecomp mirror"}]}
        )
        server._exa_clients = [mock]

        resp = await server.browse(_req(), BrowseRequest(urls=["https://b.example/2"]))

        assert resp.results_string == CONTAMINATED_MESSAGE
