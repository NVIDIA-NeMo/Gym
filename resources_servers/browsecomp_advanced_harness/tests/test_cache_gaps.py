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
"""Cache gaps: contamination on hits, failure paths, concurrent writes, rollout context on browse/verify/run."""

import hashlib
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.requests import Request

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness import rollout_context as RC
from resources_servers.browsecomp_advanced_harness.app import (
    BrowseRequest,
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
    TavilySearchVerifyRequest,
)
from resources_servers.browsecomp_advanced_harness.search_cache import SearchCache, normalize_query
from resources_servers.browsecomp_advanced_harness.tests.test_resources_sessions import _server as session_server
from resources_servers.browsecomp_advanced_harness.tests.test_resources_sessions import _verify_body


_EXCLUDE = str(Path(__file__).with_name("dummy_exclude_domains_file.json"))
_NEEDS_PERMISSIONS = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions"
)
_CLEAN = {"title": "clean one", "url": "https://a.example/1", "content": "alpha", "raw_content": "alpha body"}
_DIRTY = {
    "title": "BrowseComp dataset",
    "url": "https://hf.co/datasets/openai/BrowseComp",
    "content": "leak",
    "raw_content": "leak body",
}


def _server(cache_dir, *, provider="tavily", ws_root=None, read_only=False) -> TavilySearchResourcesServer:
    kwargs = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        search_provider=provider,
        exclude_domains_file_path=_EXCLUDE,
        search_cache_dir=cache_dir,
        search_cache_read_only=read_only,
        workspace="per_session" if ws_root else "none",
        workspace_root=ws_root,
    )
    kwargs["exa_api_key" if provider == "exa" else "tavily_api_key"] = "test_key"  # pragma: allowlist secret
    return TavilySearchResourcesServer(
        config=TavilySearchResourcesServerConfig(**kwargs), server_client=MagicMock(spec=ServerClient)
    )


def _req(sid="test_session_id", rollout_id=None) -> Request:
    headers = [(b"x-rollout-id", rollout_id.encode())] if rollout_id else []
    return Request({"type": "http", "session": {SESSION_ID_KEY: sid}, "headers": headers})


def _entry_path(root, kind, norm) -> Path:
    h = hashlib.sha256(norm.encode()).hexdigest()
    return Path(root, kind, h[:2], h[2:4], f"{h}.json")


# ---- contamination filter on cache hits --------------------------------------


class TestCachedResultsStillPassTheContaminationFilter:
    """The cache stores raw provider results, so every hit must be filtered on read."""

    async def test_inline_tavily_hit(self, tmp_path) -> None:
        server = _server(str(tmp_path / "c"))
        server._search_cache.put_search("q", [_CLEAN, _DIRTY], provider="tavily")
        provider = MagicMock(search=AsyncMock())
        server._async_tavily_clients = [provider]

        out = (await server.search(_req(), TavilySearchRequest(queries=["q"]))).results_string

        provider.search.assert_not_awaited()
        assert "alpha body" in out
        assert "leak" not in out and "BrowseComp" not in out

    async def test_exa_hit_served_under_tavily_config(self, tmp_path) -> None:
        server = _server(str(tmp_path / "c"))
        dirty = {"title": "x", "url": "https://e.example/2", "highlights": ["from the browsecomp answer key"]}
        clean = {"title": "y", "url": "https://e.example/1", "highlights": ["clean highlight"]}
        server._search_cache.put_search("q", [clean, dirty], provider="exa")
        server._async_tavily_clients = [MagicMock(search=AsyncMock())]

        out = (await server.search(_req(), TavilySearchRequest(queries=["q"]))).results_string

        assert "clean highlight" in out
        assert "answer key" not in out and "e.example/2" not in out

    async def test_disk_hit_never_writes_the_contaminated_page(self, tmp_path) -> None:
        ws_root = tmp_path / "ws"
        server = _server(str(tmp_path / "c"), ws_root=str(ws_root))
        server._search_cache.put_search("q", [_CLEAN, _DIRTY], provider="tavily")
        server._async_tavily_clients = [MagicMock(search=AsyncMock())]
        server._get_page_writer("test_session_id")

        await server.search(_req(), TavilySearchRequest(queries=["q"]))

        pages = [p.read_text() for p in (ws_root / "test_session_id" / "pages").glob("*")]
        assert len(pages) == 1 and "alpha body" in pages[0]
        assert "leak" not in pages[0]

    async def test_browse_drops_a_cached_contaminated_page_beside_a_live_one(self, tmp_path) -> None:
        server = _server(str(tmp_path / "c"))
        server._search_cache.put_extract("https://a.example/dirty", "the browsecomp answer key", provider="tavily")
        provider = MagicMock(
            extract=AsyncMock(return_value={"results": [{"url": "https://a.example/live", "raw_content": "LIVE"}]})
        )
        server._async_tavily_clients = [provider]

        out = (
            await server.browse(_req(), BrowseRequest(urls=["https://a.example/dirty", "https://a.example/live"]))
        ).results_string

        assert provider.extract.await_args.kwargs["urls"] == ["https://a.example/live"]
        assert "LIVE" in out
        assert "answer key" not in out

    async def test_miss_and_hit_agree_and_the_entry_stays_raw(self, tmp_path) -> None:
        root = str(tmp_path / "c")
        miss_server = _server(root)
        miss_server._async_tavily_clients = [MagicMock(search=AsyncMock(return_value={"results": [_CLEAN, _DIRTY]}))]
        miss = (await miss_server.search(_req(), TavilySearchRequest(queries=["q"]))).results_string

        hit_server = _server(root)
        hit_server._async_tavily_clients = [MagicMock(search=AsyncMock())]
        hit = (await hit_server.search(_req(), TavilySearchRequest(queries=["q"]))).results_string

        assert hit == miss and "leak" not in hit
        # Stored raw, so a later, wider filter applies to entries written before it.
        assert SearchCache(root).get_search("q") == ("tavily", [_CLEAN, _DIRTY])


# ---- failure paths -----------------------------------------------------------


class TestCacheFailurePaths:
    async def test_corrupt_entry_falls_through_to_the_provider_and_is_rewritten(self, tmp_path) -> None:
        root = str(tmp_path / "c")
        server = _server(root)
        server._search_cache.put_search("q", [_CLEAN], provider="tavily")
        _entry_path(root, "search", normalize_query("q")).write_text("{truncated")
        provider = MagicMock(search=AsyncMock(return_value={"results": [_CLEAN]}))
        server._async_tavily_clients = [provider]

        first = (await server.search(_req(), TavilySearchRequest(queries=["q"]))).results_string
        second = (await server.search(_req(), TavilySearchRequest(queries=["q"]))).results_string

        provider.search.assert_awaited_once()
        assert first == second and "alpha body" in first
        assert SearchCache(root).get_search("q") == ("tavily", [_CLEAN])

    @_NEEDS_PERMISSIONS
    async def test_read_only_disk_miss_on_an_unwritable_root(self, tmp_path) -> None:
        """The motivating case: someone else's 0555 root, terminal mode, a query it does not hold."""
        root = tmp_path / "c"
        SearchCache(str(root)).put_extract("https://x.example/p", "BODY", provider="tavily")
        before = sorted(str(p) for p in root.rglob("*"))
        dirs = [root, *[p for p in root.rglob("*") if p.is_dir()]]
        ws_root = tmp_path / "ws"
        for d in dirs:
            d.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            server = _server(str(root), ws_root=str(ws_root), read_only=True)
            provider = MagicMock(search=AsyncMock(return_value={"results": [_CLEAN]}))
            server._async_tavily_clients = [provider]
            server._get_page_writer("test_session_id")

            out = (await server.search(_req(), TavilySearchRequest(queries=["new q"]))).results_string

            provider.search.assert_awaited_once()
            assert "Search failed" not in out and "[Saved to]:" in out
            pages = [p.read_text() for p in (ws_root / "test_session_id" / "pages").glob("*")]
            assert len(pages) == 1 and pages[0].endswith("alpha body")
            assert sorted(str(p) for p in root.rglob("*")) == before
        finally:
            for d in dirs:
                d.chmod(stat.S_IRWXU)

    async def test_read_only_browse_miss_is_served_and_not_cached(self, tmp_path) -> None:
        root = str(tmp_path / "c")
        SearchCache(root)  # an existing, empty root
        server = _server(root, read_only=True)
        provider = MagicMock(
            extract=AsyncMock(return_value={"results": [{"url": "https://x.example/p", "raw_content": "LIVE"}]})
        )
        server._async_tavily_clients = [provider]

        out = (await server.browse(_req(), BrowseRequest(urls=["https://x.example/p"]))).results_string

        assert "LIVE" in out
        assert SearchCache(root).get_extract("https://x.example/p") is None

    @_NEEDS_PERMISSIONS
    def test_writable_mode_on_an_unwritable_root_fails_at_startup(self, tmp_path) -> None:
        """A misconfigured shared root stops the server before any rollout, not mid-run."""
        root = tmp_path / "c"
        root.mkdir()
        root.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            with pytest.raises(PermissionError):
                _server(str(root))
        finally:
            root.chmod(stat.S_IRWXU)


# ---- concurrent writes -------------------------------------------------------


def test_concurrent_same_key_writes_never_expose_a_partial_entry(tmp_path) -> None:
    """Writes publish by rename, so once a reader sees the entry it never sees a torn (miss-looking) file."""
    cache = SearchCache(str(tmp_path))
    payloads = [[{"writer": i, "body": "x" * 200_000}] for i in range(8)]
    stop = threading.Event()
    reads: list[list] = [[] for _ in range(4)]

    def reader(out):
        while not stop.is_set():
            out.append(cache.get_search("same key"))

    readers = [threading.Thread(target=reader, args=(out,)) for out in reads]
    for t in readers:
        t.start()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda p: [cache.put_search("same key", p, provider="tavily") for _ in range(20)], payloads))
    stop.set()
    for t in readers:
        t.join()

    valid = [("tavily", p) for p in payloads]
    for out in reads:
        first_hit = next((i for i, r in enumerate(out) if r is not None), len(out))
        assert all(r in valid for r in out[first_hit:])
    assert cache.get_search("same key") in valid
    assert not list(tmp_path.rglob("*.tmp"))


# ---- rollout context and totals ----------------------------------------------


class TestRolloutContext:
    async def test_browse_restores_the_callers_rollout_context(self, tmp_path) -> None:
        server = _server(str(tmp_path / "c"))
        server._search_cache.put_extract("https://x.example/p", "BODY", provider="tavily")
        RC.set_rollout_id("caller")
        await server.browse(_req(rollout_id="episode"), BrowseRequest(urls=["https://x.example/p"]))
        assert RC.current_rollout_id() == "caller"

    async def test_verify_restores_the_callers_rollout_context(self, tmp_path) -> None:
        server = session_server(tmp_path / "w", workspace="none", search_cache_dir=str(tmp_path / "c"))
        RC.set_rollout_id("caller")
        await server.verify(
            _req(rollout_id="episode"), TavilySearchVerifyRequest.model_validate(_verify_body("Paris"))
        )
        assert RC.current_rollout_id() == "caller"

    async def test_run_session_verify_logs_and_releases_its_totals(self, tmp_path, caplog) -> None:
        """An agent's /run has no close: its verify is the release point."""
        caplog.set_level("INFO")
        server = session_server(tmp_path / "w", workspace="none", search_cache_dir=str(tmp_path / "c"))
        server._async_tavily_clients = [MagicMock(search=AsyncMock(return_value={"results": [_CLEAN]}))]
        request = _req(sid="run-session", rollout_id="run-rollout")

        await server.search(request, TavilySearchRequest(queries=["q"]))
        assert RC.peek_phase_totals("run-rollout")["cache_put_search_n"] == 1
        await server.verify(request, TavilySearchVerifyRequest.model_validate(_verify_body("Paris")))

        assert any("[browsecomp][phases] rid=run-rollout" in r.message for r in caplog.records)
        assert RC.peek_phase_totals("run-rollout") == {}
        assert server._session_rollout_ids == {}
