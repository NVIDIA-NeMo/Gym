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
"""Cache writes: a failed write must not fail the tool call, and concurrent writers keep the prefer-exa rule."""

import os
import stat
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.requests import Request

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    BrowseRequest,
    TavilySearchRequest,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)
from resources_servers.browsecomp_advanced_harness.search_cache import SearchCache


_EXCLUDE = str(Path(__file__).with_name("dummy_exclude_domains_file.json"))


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions")
async def test_unwritable_cache_entry_dirs_do_not_fail_search_or_browse(tmp_path) -> None:
    config = TavilySearchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        tavily_api_key="test_key",  # pragma: allowlist secret
        exclude_domains_file_path=_EXCLUDE,
        search_cache_dir=str(tmp_path / "c"),
    )
    server = TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    # Quota, ENOSPC or a permission change mid-run: the entry directories stop accepting writes.
    dirs = [tmp_path / "c" / "search", tmp_path / "c" / "browse"]
    for d in dirs:
        d.chmod(stat.S_IRUSR | stat.S_IXUSR)
    server._async_tavily_clients = [
        MagicMock(
            search=AsyncMock(
                return_value={
                    "results": [{"title": "T", "url": "https://a.example/1", "content": "s", "raw_content": "LIVE"}]
                }
            ),
            extract=AsyncMock(return_value={"results": [{"url": "https://a.example/1", "raw_content": "PAGE"}]}),
        )
    ]
    request = Request({"type": "http", "session": {SESSION_ID_KEY: "s"}, "headers": []})
    try:
        search = await server.search(request, TavilySearchRequest(queries=["q1", "q2"]))
        browse = await server.browse(request, BrowseRequest(urls=["https://a.example/1"]))
    finally:
        for d in dirs:
            d.chmod(stat.S_IRWXU)
    assert search.results_string.count("LIVE") == 2
    assert "PAGE" in browse.results_string


# ---- the prefer-exa rule under concurrent writers -----------------------------


def _cache(root: Path) -> SearchCache:
    return SearchCache(str(root))


def _stored_provider(cache, query: str) -> str:
    return cache.get_search_entry(query)["provider"]


def test_tavily_write_does_not_overwrite_an_exa_entry_published_after_its_check(tmp_path, monkeypatch) -> None:
    cache = _cache(tmp_path / "c")
    write = cache._write
    create_only_writes = []

    def exa_publishes_first(kind, norm_key, payload, *, create_only=False):
        # The tavily writer found the key empty; an exa writer publishes before the tavily entry lands.
        if create_only:
            create_only_writes.append(norm_key)
            cache.put_search("q", [{"url": "https://e.example", "title": "E"}], provider="exa")
        return write(kind, norm_key, payload, create_only=create_only)

    monkeypatch.setattr(cache, "_write", exa_publishes_first)
    cache.put_search("q", [{"url": "https://t.example", "title": "T"}], provider="tavily")
    assert create_only_writes == ["q"]
    assert _stored_provider(cache, "q") == "exa"
    assert cache.stats()["put_errors"] == 0
    assert not list(tmp_path.rglob("*.tmp"))


def test_tavily_write_creates_a_missing_entry_and_replaces_a_tavily_one(tmp_path) -> None:
    cache = _cache(tmp_path / "c")
    cache.put_search("q", [{"url": "https://t.example/1", "title": "T1"}], provider="tavily")
    cache.put_search("q", [{"url": "https://t.example/2", "title": "T2"}], provider="tavily")
    assert cache.get_search_entry("q")["results"][0]["title"] == "T2"
    assert not list(tmp_path.rglob("*.tmp"))


def test_tavily_write_replaces_a_malformed_entry(tmp_path) -> None:
    cache = _cache(tmp_path / "c")
    cache.put_search("q", [{"url": "https://t.example/1", "title": "T1"}], provider="tavily")
    (entry_path,) = (tmp_path / "c" / "search").rglob("*.json")
    entry_path.write_text("{not json")
    cache.put_search("q", [{"url": "https://t.example/2", "title": "T2"}], provider="tavily")
    assert cache.get_search_entry("q")["results"][0]["title"] == "T2"


def test_create_only_write_falls_back_to_replace_without_hard_links(tmp_path, monkeypatch) -> None:
    cache = _cache(tmp_path / "c")

    def no_hard_links(src, dst):
        raise PermissionError("hard links not supported")

    monkeypatch.setattr(os, "link", no_hard_links)
    cache.put_extract("https://t.example/p", "PAGE", provider="tavily")
    assert cache.get_extract("https://t.example/p") == ("tavily", "PAGE")
    assert cache.stats()["put_errors"] == 0
    assert not list(tmp_path.rglob("*.tmp"))
