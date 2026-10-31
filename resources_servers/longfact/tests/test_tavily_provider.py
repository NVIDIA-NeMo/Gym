# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest

from resources_servers.longfact.tavily_provider import (
    CACHE_ADAPTER_ID,
    CACHE_FORMAT_VERSION,
    CACHE_METADATA_KEY,
    CACHE_RESPONSE_KEY,
    TavilyProvider,
    TavilyProviderError,
)


class FakeTavilyClient:
    def __init__(self):
        self.search_calls = 0
        self.search_kwargs = []

    async def search(self, **kwargs):
        self.search_calls += 1
        self.search_kwargs.append(kwargs)
        return {
            "results": [
                {
                    "title": "Primary source",
                    "url": "https://example.test/source",
                    "content": "Relevant snippet",
                    "score": 0.9,
                }
            ]
        }


def _provider(
    tmp_path,
    client=None,
    *,
    cache_mode="read_write",
    max_concurrency=4,
    search_max_attempts=4,
):
    return TavilyProvider(
        api_key="unused",
        search_depth="advanced",
        max_results=5,
        timeout_seconds=5,
        max_concurrency=max_concurrency,
        search_max_attempts=search_max_attempts,
        cache_mode=cache_mode,
        cache_dir=str(tmp_path) if cache_mode != "live" else None,
        client=client,
    )


@pytest.mark.asyncio
async def test_search_is_normalized_and_cached(tmp_path):
    client = FakeTavilyClient()
    provider = _provider(tmp_path, client, max_concurrency=1)

    first = await provider.search("citation identity")
    second = await provider.search("citation identity")
    replay = _provider(tmp_path, cache_mode="replay", max_concurrency=1)

    assert client.search_calls == 1
    assert first == second
    assert await replay.search("citation identity") == first
    cache_path = next(tmp_path.glob("search-*.json"))
    cache_document = json.loads(cache_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "adapter": CACHE_ADAPTER_ID,
        "format_version": CACHE_FORMAT_VERSION,
        "operation": "search",
        "request_sha256": cache_path.stem.removeprefix("search-"),
        "response_sha256": provider._response_digest(cache_document[CACHE_RESPONSE_KEY]),
    }
    assert cache_document[CACHE_METADATA_KEY] == expected_metadata
    assert cache_document[CACHE_RESPONSE_KEY]["results"][0]["url"] == first[0].url
    assert provider.cache_stats == {
        "mode": "read_write",
        "lookups": 2,
        "completed_lookups": 2,
        "persistent_hits": 1,
        "cross_process_hits": 0,
        "inflight_hits": 0,
        "initial_misses": 1,
        "live_fetches": 1,
        "writes": 1,
        "corruptions": 0,
        "effective_hits": 1,
        "persistent_hit_rate": 0.5,
        "effective_hit_rate": 0.5,
    }


@pytest.mark.asyncio
async def test_search_passes_configured_timeout_to_tavily_sdk(tmp_path):
    client = FakeTavilyClient()
    provider = _provider(tmp_path, client, cache_mode="live")

    await provider.search("timeout identity")

    assert client.search_kwargs == [
        {
            "query": "timeout identity",
            "search_depth": "advanced",
            "max_results": 5,
            "include_answer": False,
            "include_raw_content": False,
            "topic": "general",
            "timeout": 5,
        }
    ]


@pytest.mark.parametrize(
    ("raw_results", "expected_error"),
    [
        (None, "results as NoneType, expected array"),
        (["not-an-object"], "result 0 was str, expected object"),
    ],
)
@pytest.mark.asyncio
async def test_malformed_success_payload_becomes_provider_error(
    tmp_path,
    raw_results,
    expected_error,
):
    class MalformedClient(FakeTavilyClient):
        async def search(self, **kwargs):
            self.search_calls += 1
            self.search_kwargs.append(kwargs)
            return {"results": raw_results}

    client = MalformedClient()
    provider = _provider(tmp_path, client, cache_mode="live")

    with pytest.raises(TavilyProviderError, match=expected_error):
        await provider.search("malformed response")

    # A structurally invalid successful response follows the released
    # searcher's failed-step path; it is not replayed as a transport retry.
    assert client.search_calls == 1


@pytest.mark.asyncio
async def test_identical_concurrent_requests_are_coalesced(tmp_path):
    class SlowClient(FakeTavilyClient):
        async def search(self, **kwargs):
            self.search_calls += 1
            await asyncio.sleep(0.02)
            return {
                "results": [
                    {
                        "title": "Primary source",
                        "url": "https://example.test/source",
                        "content": "Relevant snippet",
                        "score": 0.9,
                    }
                ]
            }

    client = SlowClient()
    provider = _provider(
        tmp_path,
        client,
        max_concurrency=8,
        search_max_attempts=1,
    )

    results = await asyncio.gather(*(provider.search("same request") for _ in range(12)))

    assert client.search_calls == 1
    assert all(result == results[0] for result in results)
    assert len(list(tmp_path.glob("search-*.json"))) == 1
    assert provider.cache_stats["inflight_hits"] == 11
    assert provider.cache_stats["live_fetches"] == 1
    assert provider.cache_stats["effective_hit_rate"] == pytest.approx(11 / 12)


@pytest.mark.asyncio
async def test_live_mode_does_not_coalesce_requests(tmp_path):
    class SlowClient(FakeTavilyClient):
        async def search(self, **kwargs):
            await asyncio.sleep(0)
            return await super().search(**kwargs)

    client = SlowClient()
    provider = _provider(tmp_path, client, cache_mode="live", max_concurrency=4)

    await asyncio.gather(*(provider.search("same request") for _ in range(4)))

    assert client.search_calls == 4


@pytest.mark.asyncio
async def test_failed_coalesced_request_is_retried_by_next_call(tmp_path):
    class FailingClient:
        def __init__(self):
            self.search_calls = 0

        async def search(self, **kwargs):
            self.search_calls += 1
            await asyncio.sleep(0.01)
            raise RuntimeError("temporary outage")

    client = FailingClient()
    provider = _provider(
        tmp_path,
        client,
        max_concurrency=8,
        search_max_attempts=1,
    )

    first = await asyncio.gather(
        *(provider.search("same request") for _ in range(5)),
        return_exceptions=True,
    )
    assert client.search_calls == 1
    assert all(isinstance(result, TavilyProviderError) for result in first)

    with pytest.raises(TavilyProviderError, match="temporary outage"):
        await provider.search("same request")
    assert client.search_calls == 2


@pytest.mark.asyncio
async def test_search_retries_upstream_number_of_attempts(monkeypatch, tmp_path):
    class TransientClient(FakeTavilyClient):
        async def search(self, **kwargs):
            self.search_calls += 1
            if self.search_calls < 4:
                raise RuntimeError("temporary outage")
            return {
                "results": [
                    {
                        "title": "Recovered",
                        "url": "https://example.test/recovered",
                        "content": "Evidence",
                        "score": 1.0,
                    }
                ]
            }

    async def no_wait(delay):
        del delay

    monkeypatch.setattr(
        "resources_servers.longfact.tavily_provider.asyncio.sleep",
        no_wait,
    )
    client = TransientClient()
    provider = _provider(tmp_path, client, cache_mode="live")

    result = await provider.search("retry this request")

    assert client.search_calls == 4
    assert result[0].title == "Recovered"


@pytest.mark.asyncio
async def test_live_search_uses_one_upstream_pre_provider_jitter(monkeypatch, tmp_path):
    events = []

    class OrderedClient(FakeTavilyClient):
        async def search(self, **kwargs):
            events.append(("provider", kwargs["query"]))
            return await super().search(**kwargs)

    async def record_wait(delay):
        events.append(("sleep", delay))

    def fixed_jitter(lower, upper):
        assert (lower, upper) == (0, 0.3)
        return 0.125

    monkeypatch.setattr(
        "resources_servers.longfact.tavily_provider.asyncio.sleep",
        record_wait,
    )
    monkeypatch.setattr(
        "resources_servers.longfact.tavily_provider.random.uniform",
        fixed_jitter,
    )
    provider = _provider(tmp_path, OrderedClient(), cache_mode="live")

    await provider.search("jittered request")

    assert events == [
        ("sleep", 0.125),
        ("provider", "jittered request"),
    ]


@pytest.mark.asyncio
async def test_file_lock_coalesces_provider_instances(tmp_path):
    class RaceState:
        def __init__(self):
            self.calls = 0

    class RacingClient:
        def __init__(self, state, label):
            self.state = state
            self.label = label

        async def search(self, **kwargs):
            self.state.calls += 1
            await asyncio.sleep(0.02)
            return {
                "results": [
                    {
                        "title": self.label,
                        "url": f"https://example.test/{self.label}",
                        "content": self.label,
                        "score": 1.0,
                    }
                ]
            }

    state = RaceState()
    first_provider = _provider(tmp_path, RacingClient(state, "first"))
    second_provider = _provider(tmp_path, RacingClient(state, "second"))

    first, second = await asyncio.gather(
        first_provider.search("racing request"),
        second_provider.search("racing request"),
    )
    replay = await _provider(tmp_path, cache_mode="replay").search("racing request")

    assert state.calls == 1
    assert first == second == replay
    assert len(list(tmp_path.glob("search-*.json"))) == 1
    assert first_provider.cache_stats["live_fetches"] + second_provider.cache_stats["live_fetches"] == 1
    assert first_provider.cache_stats["cross_process_hits"] + second_provider.cache_stats["cross_process_hits"] == 1


@pytest.mark.asyncio
async def test_replay_accepts_legacy_raw_response_cache(tmp_path):
    replay = _provider(tmp_path, cache_mode="replay")
    payload = {
        "query": "legacy request",
        "search_depth": "advanced",
        "max_results": 5,
        "include_answer": False,
        "include_raw_content": False,
        "topic": "general",
    }
    cache_path = replay._cache_path("search", payload)
    assert cache_path is not None
    raw_response = {
        "results": [
            {
                "title": "Legacy",
                "url": "https://example.test/legacy",
                "content": "Cached before envelopes",
                "score": 0.5,
            }
        ]
    }
    cache_path.write_text(json.dumps(raw_response), encoding="utf-8")

    result = await replay.search("legacy request")

    assert result[0].title == "Legacy"


@pytest.mark.asyncio
async def test_replay_rejects_cache_identity_mismatch(tmp_path):
    provider = _provider(tmp_path, FakeTavilyClient())
    await provider.search("identity request")
    cache_path = next(tmp_path.glob("search-*.json"))
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    document[CACHE_METADATA_KEY]["operation"] = "extract"
    cache_path.write_text(json.dumps(document), encoding="utf-8")

    replay = _provider(tmp_path, cache_mode="replay")
    with pytest.raises(TavilyProviderError, match="cache identity mismatch"):
        await replay.search("identity request")


@pytest.mark.asyncio
async def test_read_write_quarantines_and_repairs_invalid_json(tmp_path):
    provider = _provider(tmp_path, FakeTavilyClient())
    await provider.search("repair invalid json")
    cache_path = next(tmp_path.glob("search-*.json"))
    cache_path.write_text('{"truncated":', encoding="utf-8")

    repair_client = FakeTavilyClient()
    repair_provider = _provider(tmp_path, repair_client)
    repaired = await repair_provider.search("repair invalid json")

    assert repair_client.search_calls == 1
    assert repair_provider.cache_stats["corruptions"] == 1
    assert repaired[0].url == "https://example.test/source"
    assert json.loads(cache_path.read_text(encoding="utf-8"))[CACHE_METADATA_KEY]["format_version"] == 2
    assert len(list((tmp_path / ".corrupt").glob("search-*.corrupt"))) == 1


@pytest.mark.asyncio
async def test_read_write_quarantines_and_repairs_checksum_mismatch(tmp_path):
    provider = _provider(tmp_path, FakeTavilyClient())
    await provider.search("repair checksum")
    cache_path = next(tmp_path.glob("search-*.json"))
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    document[CACHE_RESPONSE_KEY]["results"][0]["content"] = "silently corrupted"
    cache_path.write_text(json.dumps(document), encoding="utf-8")

    repair_client = FakeTavilyClient()
    repaired = await _provider(tmp_path, repair_client).search("repair checksum")

    assert repair_client.search_calls == 1
    assert repaired[0].snippet == "Relevant snippet"
    assert len(list((tmp_path / ".corrupt").glob("search-*.corrupt"))) == 1


@pytest.mark.asyncio
async def test_replay_accepts_version_one_envelope(tmp_path):
    provider = _provider(tmp_path, FakeTavilyClient())
    await provider.search("version one")
    cache_path = next(tmp_path.glob("search-*.json"))
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    document[CACHE_METADATA_KEY]["format_version"] = 1
    document[CACHE_METADATA_KEY].pop("response_sha256")
    cache_path.write_text(json.dumps(document), encoding="utf-8")

    replay = await _provider(tmp_path, cache_mode="replay").search("version one")

    assert replay[0].title == "Primary source"


@pytest.mark.asyncio
async def test_read_write_refreshes_unchecksummed_version_one_entry(tmp_path):
    provider = _provider(tmp_path, FakeTavilyClient())
    await provider.search("refresh version one")
    cache_path = next(tmp_path.glob("search-*.json"))
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    document[CACHE_METADATA_KEY]["format_version"] = 1
    document[CACHE_METADATA_KEY].pop("response_sha256")
    cache_path.write_text(json.dumps(document), encoding="utf-8")

    refresh_client = FakeTavilyClient()
    await _provider(tmp_path, refresh_client).search("refresh version one")
    refreshed = json.loads(cache_path.read_text(encoding="utf-8"))

    assert refresh_client.search_calls == 1
    assert refreshed[CACHE_METADATA_KEY]["format_version"] == CACHE_FORMAT_VERSION
    assert "response_sha256" in refreshed[CACHE_METADATA_KEY]
    assert len(list((tmp_path / ".corrupt").glob("search-*.corrupt"))) == 1


def test_cache_lock_waiters_do_not_use_executor_threads(tmp_path, monkeypatch):
    from resources_servers.longfact import tavily_provider

    def no_threads(*_args, **_kwargs):
        raise AssertionError("cache locks must not run in executor threads")

    monkeypatch.setattr(tavily_provider.asyncio, "to_thread", no_threads)
    provider = TavilyProvider(
        api_key="unused",
        search_depth="basic",
        max_results=1,
        timeout_seconds=1.0,
        max_concurrency=1,
        cache_mode="read_write",
        cache_dir=str(tmp_path / "cache"),
        client=object(),
    )
    path = tmp_path / "cache" / "entry.json"
    order = []

    async def hold(name, seconds):
        async with provider._cache_lock(path):
            order.append(f"{name}-in")
            await asyncio.sleep(seconds)
            order.append(f"{name}-out")

    async def main():
        await asyncio.gather(hold("a", 0.05), hold("b", 0.0))

    asyncio.run(main())

    assert order in (["a-in", "a-out", "b-in", "b-out"], ["b-in", "b-out", "a-in", "a-out"])


def _cache_provider(cache_dir, mode="read_write"):
    return TavilyProvider(
        api_key="unused",
        search_depth="basic",
        max_results=1,
        timeout_seconds=1.0,
        max_concurrency=1,
        cache_mode=mode,
        cache_dir=str(cache_dir),
        client=object(),
    )


def test_cache_dir_writable_by_other_users_is_refused(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)

    with pytest.raises(TavilyProviderError, match="writable by other users"):
        _cache_provider(shared)


def test_symlinked_cache_dir_is_refused(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)

    with pytest.raises(TavilyProviderError, match="not a symlink"):
        _cache_provider(link, mode="replay")


def test_new_cache_dir_is_private_to_the_user(tmp_path):
    _cache_provider(tmp_path / "fresh")

    assert (tmp_path / "fresh").stat().st_mode & 0o777 == 0o700


def test_cache_dir_under_a_parent_others_can_write_is_refused(tmp_path):
    parent = tmp_path / "team"
    parent.mkdir()
    parent.chmod(0o777)
    cache = parent / "cache"
    cache.mkdir(mode=0o700)

    with pytest.raises(TavilyProviderError, match="parent other users can replace"):
        _cache_provider(cache)


def test_cache_dir_under_a_sticky_shared_parent_is_accepted(tmp_path):
    parent = tmp_path / "sticky"
    parent.mkdir()
    parent.chmod(0o1777)

    _cache_provider(parent / "cache")


def test_aliased_cache_dir_is_checked_and_used_at_its_resolved_path(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    (shared / "cache").mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(shared)

    with pytest.raises(TavilyProviderError, match="parent other users can replace"):
        _cache_provider(alias / "cache")

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    (tmp_path / "private_alias").symlink_to(private)
    provider = _cache_provider(tmp_path / "private_alias" / "cache")

    assert provider._cache_dir == (private / "cache").resolve()
