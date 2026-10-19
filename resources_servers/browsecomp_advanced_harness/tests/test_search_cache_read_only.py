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
"""Tests for read-only cache mode: serve from a cache root you cannot write.

The motivating case is a second person pointing a training run at someone
else's cache root. Every path under <root> is 0755/0644 and owned by the other
user, so reads work and writes do not. Without a read-only mode the write is
attempted anyway, and `_write` does not catch OSError while its call site in
`_search_one_to_disk` sits OUTSIDE the try/except -- so a search that actually
SUCCEEDED reaches the model as "Search failed: [Errno 13] Permission denied",
after the provider call was already paid for.

What read_only must guarantee:
  1. no writes at all -- no entry files, no directories, no index shard
  2. reads are untouched: exact hits, fuzzy hits, and provider tagging
  3. misses are still counted, so hit rate stays honest
  4. it holds against a genuinely unwritable root, not just a flag check
"""

import os
import stat
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import fixture

from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)
from resources_servers.browsecomp_advanced_harness.search_cache import SearchCache


_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DUMMY_EXCLUDE_DOMAINS_FILE = os.path.join(_TEST_DIR, "dummy_exclude_domains_file.json")

_RESULTS = [{"title": "T", "url": "https://x.com/a", "content": "snippet", "raw_content": "RAW ONE"}]


def _populated_root(tmp_path, fuzzy=None):
    """A cache root with one search + one browse entry, written normally."""
    root = str(tmp_path / "cache")
    writer = SearchCache(root, fuzzy_threshold=fuzzy)
    writer.put_search("who won the 1998 race", _RESULTS, provider="tavily")
    writer.put_extract("https://x.com/a", "PAGE BODY", provider="tavily")
    return root


class TestReadOnlyDoesNotWrite:
    def test_put_search_stores_nothing(self, tmp_path) -> None:
        root = _populated_root(tmp_path)
        cache = SearchCache(root, read_only=True)
        cache.put_search("a brand new query", _RESULTS, provider="tavily")
        assert cache.get_search("a brand new query") is None

    def test_put_extract_stores_nothing(self, tmp_path) -> None:
        root = _populated_root(tmp_path)
        cache = SearchCache(root, read_only=True)
        cache.put_extract("https://x.com/never", "BODY", provider="tavily")
        assert cache.get_extract("https://x.com/never") is None

    def test_no_directories_are_created(self, tmp_path) -> None:
        """A bare root must stay bare. makedirs is the first thing __init__ does,
        and on someone else's root it is also the first thing that fails."""
        root = str(tmp_path / "untouched")
        os.makedirs(root)
        SearchCache(root, read_only=True)
        assert os.listdir(root) == []

    def test_no_index_shard_is_written(self, tmp_path) -> None:
        """The shard is the sidecar build_cache_index.py merges. A process that
        caches nothing has nothing to hand it."""
        root = _populated_root(tmp_path, fuzzy=0.8)
        before = sorted(os.listdir(os.path.join(root, "index")))
        cache = SearchCache(root, fuzzy_threshold=0.8, read_only=True)
        assert cache._shard_path is None
        cache.put_search("another new query", _RESULTS, provider="tavily")
        assert sorted(os.listdir(os.path.join(root, "index"))) == before

    def test_unwritable_root_raises_nothing(self, tmp_path) -> None:
        """The real coworker case: 0555 root, different owner. The flag must make
        this inert rather than merely unlikely."""
        root = _populated_root(tmp_path)
        for d in (os.path.join(root, "search"), os.path.join(root, "browse"), root):
            os.chmod(d, stat.S_IRUSR | stat.S_IXUSR)  # r-x, no write
        try:
            cache = SearchCache(root, read_only=True)
            cache.put_search("new", _RESULTS, provider="tavily")
            cache.put_extract("https://x.com/new", "BODY", provider="tavily")
        finally:
            for d in (root, os.path.join(root, "search"), os.path.join(root, "browse")):
                os.chmod(d, stat.S_IRWXU)

    def test_writes_still_work_when_flag_is_off(self, tmp_path) -> None:
        """Guard against the flag defaulting the wrong way: our own runs write."""
        root = _populated_root(tmp_path)
        cache = SearchCache(root)
        cache.put_search("normal write", _RESULTS, provider="tavily")
        assert cache.get_search("normal write") == ("tavily", _RESULTS)


class TestReadOnlyStillServes:
    def test_exact_search_hit(self, tmp_path) -> None:
        cache = SearchCache(_populated_root(tmp_path), read_only=True)
        assert cache.get_search("who won the 1998 race") == ("tavily", _RESULTS)

    def test_exact_browse_hit(self, tmp_path) -> None:
        cache = SearchCache(_populated_root(tmp_path), read_only=True)
        assert cache.get_extract("https://x.com/a") == ("tavily", "PAGE BODY")

    def test_fuzzy_hit_is_served(self, tmp_path) -> None:
        """Fuzzy is serve-only even when writable, so read_only must not cost
        recall for norms already in the index."""
        cache = SearchCache(_populated_root(tmp_path, fuzzy=0.8), fuzzy_threshold=0.8, read_only=True)
        assert cache.get_search("who won 1998 race the") == ("tavily", _RESULTS)

    def test_misses_are_still_counted(self, tmp_path) -> None:
        """Hit rate must stay truthful: a read-only run reports real misses
        rather than looking like everything hit."""
        cache = SearchCache(_populated_root(tmp_path), read_only=True)
        cache.get_search("who won the 1998 race")
        cache.get_search("nothing like this is cached")
        assert cache.stats()["search_hits"] == 1
        assert cache.stats()["search_misses"] == 1

    def test_disabled_beats_read_only(self, tmp_path) -> None:
        """root=None still disables everything; read_only does not resurrect it."""
        cache = SearchCache(None, read_only=True)
        assert not cache.enabled
        assert cache.get_search("anything") is None


class TestReadOnlyServerWiring:
    def _server(self, cache_dir, read_only=False) -> TavilySearchResourcesServer:
        config = TavilySearchResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            search_provider="tavily",
            tavily_api_key="test_tavily_key",  # pragma: allowlist secret
            exclude_domains_file_path=_DUMMY_EXCLUDE_DOMAINS_FILE,
            search_cache_dir=cache_dir,
            search_cache_read_only=read_only,
            workspace="none",
        )
        return TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    @fixture
    def req(self) -> MagicMock:
        m = MagicMock()
        m.session = {SESSION_ID_KEY: "test_session_id"}
        return m

    def test_config_defaults_to_writable(self, tmp_path) -> None:
        server = self._server(str(tmp_path / "cache"))
        assert server._search_cache.read_only is False

    def test_config_flag_reaches_the_cache(self, tmp_path) -> None:
        server = self._server(_populated_root(tmp_path), read_only=True)
        assert server._search_cache.read_only is True

    @pytest.mark.asyncio
    async def test_live_search_result_is_not_cached(self, tmp_path, req) -> None:
        """A miss still calls the provider and still answers the model -- it just
        leaves no trace in someone else's cache root."""
        root = _populated_root(tmp_path)
        server = self._server(root, read_only=True)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": _RESULTS})
        server._async_tavily_clients = [mock]

        body = TavilySearchRequest(queries=["a query that is not cached"])
        response = await server.search(req, body)

        assert "T" in response.results_string
        mock.search.assert_awaited_once()
        assert SearchCache(root).get_search("a query that is not cached") is None

    @pytest.mark.asyncio
    async def test_cached_query_never_calls_the_provider(self, tmp_path, req) -> None:
        server = self._server(_populated_root(tmp_path), read_only=True)
        mock = MagicMock()
        mock.search = AsyncMock(return_value={"results": _RESULTS})
        server._async_tavily_clients = [mock]

        body = TavilySearchRequest(queries=["who won the 1998 race"])
        response = await server.search(req, body)

        assert "T" in response.results_string
        mock.search.assert_not_awaited()
