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
"""Tests for the shared flat-file search/browse cache.

Two layers:
1. SearchCache unit tests — key normalization, provider tagging (legacy entries
   without a provider field read as "tavily"), the prefer-exa write rule,
   corrupt-file-is-a-miss, disabled no-op.
2. Server wiring tests — a miss calls the provider once and stores the raw
   response; a hit calls the provider zero times and produces BYTE-IDENTICAL
   output (store-raw / format-on-read); hits are formatted per the STORED
   provider; browse fetches only the missed URLs; empties are never cached;
   terminal-mode hits still materialize pages/ into the session workspace.
"""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from pytest import fixture

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    BrowseRequest,
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)
from resources_servers.browsecomp_advanced_harness.search_cache import (
    SearchCache,
    normalize_query,
    normalize_url,
)


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json")


class TestSearchCacheUnit:
    def test_search_roundtrip_returns_provider_and_results(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path))
        results = [{"title": "T", "url": "https://x.com", "content": "c"}]
        cache.put_search("who won", results, provider="tavily")
        assert cache.get_search("who won") == ("tavily", results)

    def test_search_miss_returns_none(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path))
        assert cache.get_search("never seen") is None

    def test_legacy_entry_without_provider_reads_as_tavily(self, tmp_path) -> None:
        # Entries migrated from the reference harness have no "provider" field.
        cache = SearchCache(str(tmp_path))
        norm = normalize_query("Old Query")
        h = hashlib.sha256(norm.encode()).hexdigest()
        d = tmp_path / "search" / h[:2] / h[2:4]
        d.mkdir(parents=True)
        (d / f"{h}.json").write_text(json.dumps({"query": "Old Query", "norm": norm, "results": [{"t": 1}]}))
        assert cache.get_search("old query") == ("tavily", [{"t": 1}])

    def test_prefer_exa_write_rule(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path))
        cache.put_search("q", [{"src": "tavily1"}], provider="tavily")
        cache.put_search("q", [{"src": "exa1"}], provider="exa")  # exa overwrites tavily
        assert cache.get_search("q") == ("exa", [{"src": "exa1"}])
        cache.put_search("q", [{"src": "tavily2"}], provider="tavily")  # tavily must NOT clobber exa
        assert cache.get_search("q") == ("exa", [{"src": "exa1"}])
        cache.put_search("q", [{"src": "exa2"}], provider="exa")  # exa refresh is fine
        assert cache.get_search("q") == ("exa", [{"src": "exa2"}])

    def test_prefer_exa_write_rule_extract(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path))
        cache.put_extract("https://x.com/p", "exa body", provider="exa")
        cache.put_extract("https://x.com/p", "tavily body", provider="tavily")
        assert cache.get_extract("https://x.com/p") == ("exa", "exa body")

    def test_query_normalization_hit(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path))
        cache.put_search('  "Foo   BAR"  ', [{"r": 1}], provider="tavily")
        assert cache.get_search("foo bar") == ("tavily", [{"r": 1}])
        assert normalize_query('  "Foo   BAR"  ') == "foo bar"

    def test_url_normalization_hit(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path))
        cache.put_extract("https://x.com/page/", "body", provider="tavily")
        assert cache.get_extract("https://x.com/page#section") == ("tavily", "body")
        assert normalize_url("https://x.com/page/#frag") == "https://x.com/page"

    def test_corrupt_file_is_miss(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path))
        cache.put_search("q", [{"r": 1}], provider="tavily")
        norm = normalize_query("q")
        h = hashlib.sha256(norm.encode()).hexdigest()
        (tmp_path / "search" / h[:2] / h[2:4] / f"{h}.json").write_text("{not json")
        assert cache.get_search("q") is None

    def test_disabled_is_noop(self) -> None:
        cache = SearchCache(None)
        assert not cache.enabled
        cache.put_search("q", [{"r": 1}], provider="tavily")
        assert cache.get_search("q") is None
        cache.put_extract("u", "c", provider="tavily")
        assert cache.get_extract("u") is None

    def test_stats(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path))
        cache.get_search("q")  # miss
        cache.put_search("q", [{"r": 1}], provider="tavily")
        cache.get_search("q")  # hit
        cache.get_extract("u")  # miss
        s = cache.stats()
        assert s["search_hits"] == 1 and s["search_misses"] == 1 and s["browse_misses"] == 1
        assert "search 1/2" in cache.summary_str()


class TestSearchCacheFuzzy:
    """Fuzzy search matching (option 1: global norms index).

    Semantics pinned: SEARCH ONLY (browse stays exact); serve-only (a fuzzy hit
    never writes under the new key); token_sort_ratio >= threshold on the
    NORMALIZED query; fuzzy hits counted separately; index = base file + per-
    process append shards under <root>/index/, loaded at construction.
    """

    RESULTS = [{"title": "T", "url": "https://x.com", "raw_content": "BODY"}]

    def test_fuzzy_hit_on_word_reorder(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
        cache.put_search("countries elected 68th session CSW 2024", self.RESULTS, provider="tavily")
        hit = cache.get_search("68th session CSW 2024 countries elected")
        assert hit == ("tavily", self.RESULTS)
        assert cache.stats()["search_fuzzy_hits"] == 1
        assert cache.stats()["search_hits"] == 1  # fuzzy counts as a hit too

    def test_fuzzy_below_threshold_is_miss(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
        cache.put_search("obama birth year", self.RESULTS, provider="tavily")
        assert cache.get_search("completely different topic entirely") is None
        assert cache.stats()["search_fuzzy_hits"] == 0

    def test_fuzzy_disabled_by_default(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path))
        cache.put_search("countries elected 68th session CSW 2024", self.RESULTS, provider="tavily")
        assert cache.get_search("68th session CSW 2024 countries elected") is None

    def test_fuzzy_never_applies_to_browse(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
        cache.put_extract("https://x.com/page/one", "BODY", provider="tavily")
        assert cache.get_extract("https://x.com/page/one2") is None

    def test_fuzzy_is_serve_only(self, tmp_path) -> None:
        import hashlib as _h

        cache = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
        cache.put_search("countries elected 68th session CSW 2024", self.RESULTS, provider="tavily")
        q2 = "68th session CSW 2024 countries elected"
        assert cache.get_search(q2) is not None
        h = _h.sha256(normalize_query(q2).encode()).hexdigest()
        assert not (tmp_path / "search" / h[:2] / h[2:4] / f"{h}.json").exists()

    def test_exact_hit_preferred_over_fuzzy(self, tmp_path) -> None:
        cache = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
        cache.put_search("obama birth year", [{"k": "exact"}], provider="tavily")
        cache.put_search("obama year birth", [{"k": "other"}], provider="tavily")
        assert cache.get_search("obama birth year") == ("tavily", [{"k": "exact"}])
        assert cache.stats()["search_fuzzy_hits"] == 0

    def test_index_shard_persists_across_instances(self, tmp_path) -> None:
        # Writer process appends its puts to an index shard; a FRESH instance
        # (another process/run) loads it and can fuzzy-match those queries.
        w = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
        w.put_search("countries elected 68th session CSW 2024", self.RESULTS, provider="tavily")
        r = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
        assert r.get_search("68th session CSW 2024 countries elected") == ("tavily", self.RESULTS)

    def test_build_index_script(self, tmp_path) -> None:
        # Entries written WITHOUT fuzzy (no shards) become fuzzy-visible after
        # the offline index build.
        plain = SearchCache(str(tmp_path))
        plain.put_search("countries elected 68th session CSW 2024", self.RESULTS, provider="tavily")
        from resources_servers.browsecomp_advanced_harness.build_cache_index import build_index

        n = build_index(str(tmp_path))
        assert n == 1
        cache = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
        assert cache.get_search("68th session CSW 2024 countries elected") == ("tavily", self.RESULTS)


class TestSearchCacheWiring:
    def _server(
        self, cache_dir: str, provider: str = "tavily", workspace: str = "none", ws_root: str | None = None
    ) -> TavilySearchResourcesServer:
        kwargs = dict(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            search_provider=provider,
            exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            search_cache_dir=cache_dir,
            workspace=workspace,
            workspace_root=ws_root,
        )
        if provider == "exa":
            kwargs["exa_api_key"] = "test_exa_key"  # pragma: allowlist secret
        else:
            kwargs["tavily_api_key"] = "test_tavily_key"  # pragma: allowlist secret
        config = TavilySearchResourcesServerConfig(**kwargs)
        return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _req(self) -> MagicMock:
        m = MagicMock()
        m.session = {SESSION_ID_KEY: "test_session_id"}
        return m

    @fixture
    def cache_dir(self, tmp_path) -> str:
        return str(tmp_path / "cache")

    _TAVILY_SEARCH_RESULTS = {
        "results": [
            {"title": "T1", "url": "https://x.com/a", "content": "snippet", "raw_content": "RAW CONTENT ONE"},
        ]
    }

    # ---- config default ----

    def test_cache_disabled_by_default(self) -> None:
        server = self._server(cache_dir=None)
        assert not server._search_cache.enabled

    def test_relative_cache_dir_resolves_against_harness_dir(self, tmp_path, monkeypatch) -> None:
        import resources_servers.browsecomp_advanced_harness.app as app_module

        monkeypatch.chdir(tmp_path)  # cwd must not matter
        server = self._server(cache_dir="search_cache_test_relative")
        expected = Path(app_module.__file__).parent / "search_cache_test_relative"
        try:
            assert Path(server._search_cache.root) == expected
        finally:
            import shutil

            shutil.rmtree(expected, ignore_errors=True)

    # ---- tavily search ----

    async def test_tavily_search_miss_then_hit_byte_identical(self, cache_dir) -> None:
        server1 = self._server(cache_dir)
        mock1 = MagicMock()
        mock1.search = AsyncMock(return_value=self._TAVILY_SEARCH_RESULTS)
        server1._async_tavily_clients = [mock1]
        resp1 = await server1.search(self._req(), TavilySearchRequest(queries=["who won"]))
        mock1.search.assert_called_once()

        # fresh server instance sharing the cache dir = another process/run
        server2 = self._server(cache_dir)
        mock2 = MagicMock()
        mock2.search = AsyncMock(return_value=self._TAVILY_SEARCH_RESULTS)
        server2._async_tavily_clients = [mock2]
        resp2 = await server2.search(self._req(), TavilySearchRequest(queries=["who won"]))
        mock2.search.assert_not_called()
        assert resp2.results_string == resp1.results_string

    async def test_tavily_search_empty_results_not_cached(self, cache_dir) -> None:
        for _ in range(2):
            server = self._server(cache_dir)
            mock = MagicMock()
            mock.search = AsyncMock(return_value={"results": []})
            server._async_tavily_clients = [mock]
            await server.search(self._req(), TavilySearchRequest(queries=["nothing"]))
            mock.search.assert_called_once()  # second run must fetch again

    async def test_search_hit_records_cache_hit_metric(self, cache_dir) -> None:
        server = self._server(cache_dir)
        server._search_cache.put_search("q", self._TAVILY_SEARCH_RESULTS["results"], provider="tavily")
        server._async_tavily_clients = [MagicMock()]
        await server.search(self._req(), TavilySearchRequest(queries=["q"]))
        recs = server._session_id_to_metrics["test_session_id"].async_tavily_calls
        assert any(r.status == "cache_hit" and r.function == "search" for r in recs)

    # ---- cross-provider hits ----

    async def test_exa_entry_served_under_tavily_config(self, cache_dir) -> None:
        # seed via a real exa-mode miss
        exa_server = self._server(cache_dir, provider="exa")
        exa_mock = MagicMock()
        exa_mock.search = AsyncMock(
            return_value={"results": [{"title": "E", "url": "https://e.com", "highlights": ["h1", "h2"]}]}
        )
        exa_server._exa_clients = [exa_mock]
        exa_resp = await exa_server.search(self._req(), TavilySearchRequest(queries=["shared q"]))

        tavily_server = self._server(cache_dir, provider="tavily")
        tavily_mock = MagicMock()
        tavily_server._async_tavily_clients = [tavily_mock]
        resp = await tavily_server.search(self._req(), TavilySearchRequest(queries=["shared q"]))
        tavily_mock.search.assert_not_called()
        # formatted per the STORED provider: exa snippet style, byte-identical to the exa miss
        assert resp.results_string == exa_resp.results_string
        assert "[Snippet]: h1 ... h2" in resp.results_string

    async def test_tavily_entry_served_under_exa_config(self, cache_dir) -> None:
        server = self._server(cache_dir, provider="exa")
        server._search_cache.put_search("q", self._TAVILY_SEARCH_RESULTS["results"], provider="tavily")
        server._exa_clients = [MagicMock()]
        resp = await server.search(self._req(), TavilySearchRequest(queries=["q"]))
        assert "RAW CONTENT ONE" in resp.results_string  # tavily inline formatting

    # ---- terminal mode ----

    async def test_tavily_disk_hit_materializes_pages(self, tmp_path, cache_dir) -> None:
        ws_root = str(tmp_path / "ws")
        server = self._server(cache_dir, workspace="per_session", ws_root=ws_root)
        server._search_cache.put_search("q", self._TAVILY_SEARCH_RESULTS["results"], provider="tavily")
        server._async_tavily_clients = [MagicMock()]
        server._get_page_writer("test_session_id")

        resp = await server.search(self._req(), TavilySearchRequest(queries=["q"]))
        assert "[Saved to]:" in resp.results_string
        pages = list((Path(ws_root) / "test_session_id" / "pages").glob("*search*"))
        assert len(pages) == 1
        assert "RAW CONTENT ONE" in pages[0].read_text()

    # ---- browse ----

    async def test_browse_fetches_only_missed_urls(self, cache_dir) -> None:
        server = self._server(cache_dir)
        server._search_cache.put_extract("https://x.com/hit", "CACHED BODY", provider="tavily")
        mock = MagicMock()
        mock.extract = AsyncMock(
            return_value={"results": [{"url": "https://x.com/miss", "raw_content": "FETCHED BODY"}]}
        )
        server._async_tavily_clients = [mock]

        resp = await server.browse(self._req(), BrowseRequest(urls=["https://x.com/hit", "https://x.com/miss"]))
        mock.extract.assert_called_once()
        _, kwargs = mock.extract.call_args
        assert kwargs.get("urls") == ["https://x.com/miss"]
        assert "CACHED BODY" in resp.results_string
        assert "FETCHED BODY" in resp.results_string
        # the miss is now cached
        assert server._search_cache.get_extract("https://x.com/miss") == ("tavily", "FETCHED BODY")

    async def test_browse_all_hits_no_provider_call(self, cache_dir) -> None:
        server = self._server(cache_dir)
        server._search_cache.put_extract("https://x.com/p", "BODY", provider="tavily")
        server._async_tavily_clients = [MagicMock()]
        resp = await server.browse(self._req(), BrowseRequest(urls=["https://x.com/p"]))
        assert "BODY" in resp.results_string

    async def test_browse_miss_then_hit_byte_identical(self, cache_dir) -> None:
        payload = {"results": [{"url": "https://x.com/p", "raw_content": "PAGE BODY"}]}
        server1 = self._server(cache_dir)
        mock1 = MagicMock()
        mock1.extract = AsyncMock(return_value=payload)
        server1._async_tavily_clients = [mock1]
        resp1 = await server1.browse(self._req(), BrowseRequest(urls=["https://x.com/p"]))

        server2 = self._server(cache_dir)
        mock2 = MagicMock()
        mock2.extract = AsyncMock(return_value=payload)
        server2._async_tavily_clients = [mock2]
        resp2 = await server2.browse(self._req(), BrowseRequest(urls=["https://x.com/p"]))
        mock2.extract.assert_not_called()
        assert resp2.results_string == resp1.results_string

    async def test_browse_empty_content_not_cached(self, cache_dir) -> None:
        server = self._server(cache_dir)
        mock = MagicMock()
        mock.extract = AsyncMock(return_value={"results": [{"url": "https://x.com/p", "raw_content": ""}]})
        server._async_tavily_clients = [mock]
        await server.browse(self._req(), BrowseRequest(urls=["https://x.com/p"]))
        assert server._search_cache.get_extract("https://x.com/p") is None

    async def test_exa_browse_stores_with_exa_provider(self, cache_dir) -> None:
        server = self._server(cache_dir, provider="exa")
        mock = MagicMock()
        mock.get_contents = AsyncMock(return_value={"results": [{"url": "https://x.com/p", "text": "EXA TEXT"}]})
        server._exa_clients = [mock]
        await server.browse(self._req(), BrowseRequest(urls=["https://x.com/p"]))
        assert server._search_cache.get_extract("https://x.com/p") == ("exa", "EXA TEXT")

    async def test_browse_disk_hit_materializes_pages(self, tmp_path, cache_dir) -> None:
        ws_root = str(tmp_path / "ws")
        server = self._server(cache_dir, workspace="per_session", ws_root=ws_root)
        server._search_cache.put_extract("https://x.com/p", "PAGE BODY", provider="tavily")
        server._async_tavily_clients = [MagicMock()]
        server._get_page_writer("test_session_id")

        resp = await server.browse(self._req(), BrowseRequest(urls=["https://x.com/p"]))
        assert "[Saved to]:" in resp.results_string
        pages = list((Path(ws_root) / "test_session_id" / "pages").glob("*browse*"))
        assert len(pages) == 1
        assert "PAGE BODY" in pages[0].read_text()
