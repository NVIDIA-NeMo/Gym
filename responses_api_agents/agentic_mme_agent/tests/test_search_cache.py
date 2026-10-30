# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from PIL import Image

from responses_api_agents.agentic_mme_agent import retrieval
from responses_api_agents.agentic_mme_agent.retrieval import Retrieval, RetrievalConfig
from responses_api_agents.agentic_mme_agent.search_cache import SearchCache, normalize_query, shared_cache
from responses_api_agents.agentic_mme_agent.sqlite_fuzzy_index import build_master_db, master_db_path
from responses_api_agents.agentic_mme_agent.tools import ImageWorkspace


def _seed(root, query, results):
    cache = SearchCache(str(root), fuzzy_threshold=0)
    cache.put_search(query, results, "serpapi", "us", "en")
    return cache


def test_exact_fuzzy_and_number_guard(tmp_path) -> None:
    _seed(tmp_path, "Eiffel Tower height in meters 2010", {"organic": [{"title": "2010"}]})
    build_master_db(master_db_path(str(tmp_path)), [normalize_query("Eiffel Tower height in meters 2010")])
    cache = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
    assert cache.get_search("eiffel tower height in meters 2010")[1] == "exact"
    assert cache.get_search("height in meters of the Eiffel Tower 2010")[1] == "fuzzy"
    # same words, different year: not the same question
    assert cache.get_search("Eiffel Tower height in meters 2020") == (None, "miss")


@pytest.mark.asyncio
async def test_miss_calls_provider_then_later_calls_hit_without_network(tmp_path, monkeypatch) -> None:
    shared_cache.cache_clear()
    calls = []

    async def fake_payload(cfg, method, url, **kwargs):
        calls.append(url)
        if url == "https://serpapi.com/image":
            return json.dumps({"image_id": "img-1"}).encode()
        if "serpapi" in url:
            return json.dumps({"visual_matches": [{"title": "M", "link": "https://m.example"}]}).encode()
        if "jina" in url:
            return b"page text " * 50
        return json.dumps({"results": [{"title": "T", "url": "https://t.example", "content": "c"}]}).encode()

    monkeypatch.setattr(retrieval, "http_payload", fake_payload)
    config = RetrievalConfig(
        mode="live",
        search_provider="tavily",
        tavily_api_key="t",
        lens_provider="serpapi",
        serpapi_api_key="s",
        allow_image_upload=True,
        cache_root=str(tmp_path),
    )
    workspace = ImageWorkspace()
    workspace.add(Image.new("RGB", (4, 4), "red"))
    first = Retrieval(config)
    assert (await first.call("google_search", {"query": "Q one"}, workspace))["cache"] == "miss"
    assert (await first.call("google_lens_search", {"image_index": 0}, workspace))["cache"] == "miss"
    page = await first.call("fetch_webpage", {"url": "https://a.example/p", "max_chars": 20}, workspace)
    assert page["cache"] == "miss" and "full_text" not in page and len(page["context"]) == 20
    n = len(calls)
    second = Retrieval(config)  # a later rollout in the same process
    assert (await second.call("google_search", {"query": "  q ONE "}, workspace))["cache"] == "exact"
    lens = await second.call("google_lens_search", {"image_index": 0}, workspace)
    assert lens["cache"] == "exact" and lens["raw"]["organic"][0]["title"] == "M"
    page = await second.call("fetch_webpage", {"url": "https://a.example/p/", "max_chars": 100}, workspace)
    assert page["cache"] == "exact" and len(page["context"]) == 100  # the whole page was cached
    assert len(calls) == n  # no provider call for any hit


def test_rebuild_index_makes_new_entries_fuzzy_matchable(tmp_path) -> None:
    from responses_api_agents.agentic_mme_agent.search_cache import rebuild_index

    _seed(tmp_path, "tallest building in Paris 2024", {"organic": []})
    assert SearchCache(str(tmp_path), fuzzy_threshold=0.8).get_search("in Paris tallest building 2024")[1] == "miss"
    assert rebuild_index(str(tmp_path)) == 1
    assert SearchCache(str(tmp_path), fuzzy_threshold=0.8).get_search("in Paris tallest building 2024")[1] == "fuzzy"


def test_entries_are_shared_readable_under_a_strict_umask(tmp_path) -> None:
    import os
    import stat

    old = os.umask(0o077)  # e.g. root in a container
    try:
        _seed(tmp_path, "who painted the ceiling", {"organic": []})
    finally:
        os.umask(old)
    files = list((tmp_path / "search").rglob("*.json"))
    assert len(files) == 1
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o664
    for directory in (files[0].parent, files[0].parent.parent, tmp_path / "search"):
        assert stat.S_IMODE(directory.stat().st_mode) & 0o775 == 0o775


def test_unwritable_cache_never_fails_the_caller(tmp_path) -> None:
    import os

    (tmp_path / "search").mkdir()
    os.chmod(tmp_path / "search", 0o500)
    try:
        _seed(tmp_path, "read only cache", {"organic": []})  # must not raise
        assert SearchCache(str(tmp_path), fuzzy_threshold=0).get_search("read only cache") == (None, "miss")
    finally:
        os.chmod(tmp_path / "search", 0o700)
