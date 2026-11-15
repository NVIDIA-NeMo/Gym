# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os

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
    assert cache.get_search("eiffel tower height in meters 2010", "serpapi")[1] == "exact"
    assert cache.get_search("height in meters of the Eiffel Tower 2010", "serpapi")[1] == "fuzzy"
    # same words, different year: not the same question
    assert cache.get_search("Eiffel Tower height in meters 2020", "serpapi") == (None, "miss", None)


def test_fuzzy_matching_is_off_by_default(tmp_path) -> None:
    _seed(tmp_path, "capital of austria", {"organic": [{"title": "Vienna"}]})
    build_master_db(master_db_path(str(tmp_path)), ["capital of austria"])
    assert SearchCache(str(tmp_path)).get_search("capital of australia", "serpapi") == (None, "miss", None)
    assert RetrievalConfig().cache_fuzzy_threshold == 0
    # Opting in, the same near miss becomes a hit for the other question.
    assert SearchCache(str(tmp_path), fuzzy_threshold=0.8).get_search("capital of australia", "serpapi")[1:] == (
        "fuzzy",
        "capital of austria",
    )


@pytest.mark.asyncio
async def test_fuzzy_hit_shows_the_model_which_query_it_answers(tmp_path, monkeypatch) -> None:
    shared_cache.cache_clear()
    _seed(tmp_path, "capital of austria", {"organic": [{"title": "Vienna", "link": "https://v.example"}]})
    build_master_db(master_db_path(str(tmp_path)), ["capital of austria"])

    async def no_network(*args, **kwargs):
        raise AssertionError("a cache hit must not call the provider")

    monkeypatch.setattr(retrieval, "http_payload", no_network)
    workspace = ImageWorkspace()
    workspace.add(Image.new("RGB", (4, 4), "red"))
    config = RetrievalConfig(
        mode="live",
        search_provider="serpapi",
        serpapi_api_key="s",
        allow_image_upload=True,
        cache_root=str(tmp_path),
        cache_fuzzy_threshold=0.8,
    )
    out = await Retrieval(config).call("google_search", {"query": "capital of australia"}, workspace)
    assert out["cache"] == "fuzzy" and out["matched_query"] == "capital of austria"
    exact = await Retrieval(config).call("google_search", {"query": "Capital of Austria"}, workspace)
    assert exact["cache"] == "exact" and "matched_query" not in exact


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
    assert (
        SearchCache(str(tmp_path), fuzzy_threshold=0.8).get_search("in Paris tallest building 2024", "serpapi")[1]
        == "miss"
    )
    assert rebuild_index(str(tmp_path)) == 1
    assert (
        SearchCache(str(tmp_path), fuzzy_threshold=0.8).get_search("in Paris tallest building 2024", "serpapi")[1]
        == "fuzzy"
    )


def test_this_process_entries_are_fuzzy_matchable_before_any_rebuild(tmp_path) -> None:
    # No master.db yet: the in-memory local tier alone must serve a near match, and the new norm
    # must reach this process's shard for the next rebuild.
    cache = SearchCache(str(tmp_path), fuzzy_threshold=0.8)
    cache.put_search(
        "tallest building in Paris 2024", {"organic": [{"title": "Tour Montparnasse"}]}, "exa", "us", "en"
    )
    assert not os.path.exists(master_db_path(str(tmp_path)))
    results, kind, matched = cache.get_search("in Paris tallest building 2024", "exa")
    assert kind == "fuzzy" and matched == "tallest building in Paris 2024"
    assert results["organic"][0]["title"] == "Tour Montparnasse"
    shards = list((tmp_path / "index").glob("search_norms_*.txt"))
    assert [s.read_text().splitlines() for s in shards] == [["tallest building in paris 2024"]]
    # A second put of the same norm is not written to the shard again.
    cache.put_search("Tallest building in Paris 2024", {"organic": []}, "exa", "us", "en")
    assert shards[0].read_text().splitlines() == ["tallest building in paris 2024"]


def test_rebuild_index_reads_norms_from_shards(tmp_path) -> None:
    from responses_api_agents.agentic_mme_agent.search_cache import rebuild_index
    from responses_api_agents.agentic_mme_agent.sqlite_fuzzy_index import SqliteFuzzyIndex

    _seed(tmp_path, "who painted the sistine chapel ceiling", {"organic": []})
    # A norm another process recorded in its shard, with no entry file of its own here.
    (tmp_path / "index").mkdir()
    (tmp_path / "index" / "search_norms_node7_123.txt").write_text("who designed the louvre pyramid\n\n")
    assert rebuild_index(str(tmp_path)) == 2
    index = SqliteFuzzyIndex(str(tmp_path), threshold=0.8)
    assert index.lookup("the louvre pyramid who designed") == "who designed the louvre pyramid"
    assert index.lookup("sistine chapel ceiling who painted the") == "who painted the sistine chapel ceiling"


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


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a read-only directory")
def test_unwritable_cache_never_fails_the_caller(tmp_path) -> None:
    import os

    (tmp_path / "search").mkdir()
    os.chmod(tmp_path / "search", 0o500)
    try:
        _seed(tmp_path, "read only cache", {"organic": []})  # must not raise
        assert SearchCache(str(tmp_path), fuzzy_threshold=0).get_search("read only cache", "serpapi") == (
            None,
            "miss",
            None,
        )
    finally:
        os.chmod(tmp_path / "search", 0o700)


def test_entries_are_scoped_to_provider_and_locale(tmp_path) -> None:
    cache = SearchCache(str(tmp_path))
    cache.put_search("capital of australia", {"organic": [{"title": "Tavily"}]}, "tavily", "us", "en")
    cache.put_search("capital of australia", {"organic": [{"title": "SerpAPI"}]}, "serpapi", "us", "en")
    assert cache.get_search("capital of australia", "tavily")[0]["organic"][0]["title"] == "Tavily"
    assert cache.get_search("capital of australia", "serpapi")[0]["organic"][0]["title"] == "SerpAPI"
    assert cache.get_search("capital of australia", "serper") == (None, "miss", None)
    assert cache.get_search("capital of australia", "serpapi", "us", "de") == (None, "miss", None)
    image = Image.new("RGB", (4, 4), "red")
    cache.put_lens(image, {"organic": []}, "serpapi")
    assert cache.get_lens(image, "serpapi") == {"organic": []} and cache.get_lens(image, "serper") is None
    cache.put_browse("https://a.example/p", "exa text", "exa")
    assert (
        cache.get_browse("https://a.example/p/", "exa") == "exa text"
        and cache.get_browse("https://a.example/p", "jina") is None
    )


def test_entries_written_before_scoped_keys_still_serve_their_own_provider(tmp_path) -> None:
    cache = SearchCache(str(tmp_path))
    # The old layout: keyed by the bare norm / pixels / URL, with the provider stored in the entry.
    cache._write(
        "search",
        "capital of australia",
        {"norm": "capital of australia", "provider": "serpapi", "results": {"organic": []}},
    )
    cache._write("browse", "https://a.example/p", {"provider": "jina", "content": "old page"})
    assert cache.get_search("capital of australia", "serpapi")[1] == "exact"  # gl/hl missing: us/en
    assert cache.get_search("capital of australia", "tavily")[1] == "miss"
    assert cache.get_search("capital of australia", "serpapi", "de", "de")[1] == "miss"
    assert (
        cache.get_browse("https://a.example/p", "jina") == "old page"
        and cache.get_browse("https://a.example/p", "exa") is None
    )
