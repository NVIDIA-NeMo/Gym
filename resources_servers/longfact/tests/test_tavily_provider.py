# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import fcntl
import json
import os
import threading
from pathlib import Path

import pytest
from aiohttp import ClientTimeout

import resources_servers.tavily_search.app as tavily_search_module
from resources_servers.longfact.tavily_provider import (
    CACHE_ADAPTER_ID,
    CACHE_FORMAT_VERSION,
    CACHE_METADATA_KEY,
    CACHE_RESPONSE_KEY,
    TavilyCacheCorruptionError,
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
    timeout_seconds=5,
):
    return TavilyProvider(
        api_key="unused",
        search_depth="advanced",
        max_results=5,
        timeout_seconds=timeout_seconds,
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


@pytest.mark.parametrize("timeout_seconds", [90, 5])
@pytest.mark.asyncio
async def test_configured_timeout_reaches_the_constructed_http_transport(tmp_path, monkeypatch, timeout_seconds):
    requested_timeouts = []

    class Response:
        status = 200

        async def json(self):
            return {"results": []}

        def release(self):
            pass

    async def request(method, **kwargs):
        assert method == "POST"
        requested_timeouts.append(kwargs["timeout"])
        return Response()

    monkeypatch.setattr(tavily_search_module, "request", request)
    provider = _provider(tmp_path, cache_mode="live", timeout_seconds=timeout_seconds)

    assert await provider.search("transport timeout") == []

    assert len(requested_timeouts) == 1
    assert isinstance(requested_timeouts[0], ClientTimeout)
    assert requested_timeouts[0].total == timeout_seconds


@pytest.mark.asyncio
async def test_provider_operation_deadline_cancels_the_search(tmp_path):
    class BlockedClient:
        cancelled = False

        async def search(self, **_kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True

    client = BlockedClient()
    provider = _provider(tmp_path, client, cache_mode="live", timeout_seconds=0.01, search_max_attempts=1)

    with pytest.raises(TavilyProviderError, match="failed after 1 attempts") as failure:
        await asyncio.wait_for(provider.search("blocked operation"), timeout=5)

    assert isinstance(failure.value.__cause__, TimeoutError)
    assert client.cancelled


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        ({}, "missing results"),
        ({"error": "search unavailable"}, "missing results"),
        ({"results": None}, "results as NoneType, expected array"),
        ({"results": ["not-an-object"]}, "result 0 was str, expected object"),
    ],
)
@pytest.mark.parametrize("cache_mode", ["live", "read_write"])
@pytest.mark.asyncio
async def test_malformed_success_payload_becomes_provider_error(
    tmp_path,
    response,
    expected_error,
    cache_mode,
):
    class MalformedClient(FakeTavilyClient):
        async def search(self, **kwargs):
            self.search_calls += 1
            self.search_kwargs.append(kwargs)
            return response

    client = MalformedClient()
    provider = _provider(tmp_path, client, cache_mode=cache_mode)

    with pytest.raises(TavilyProviderError, match=expected_error):
        await provider.search("malformed response")

    # A structurally invalid successful response follows the released
    # searcher's failed-step path; it is not replayed as a transport retry.
    assert client.search_calls == 1
    assert not list(tmp_path.glob("search-*.json"))
    assert provider.cache_stats["writes"] == 0


@pytest.mark.asyncio
async def test_explicit_empty_results_are_cached_and_replayed(tmp_path):
    class EmptyClient(FakeTavilyClient):
        async def search(self, **kwargs):
            self.search_calls += 1
            return {"results": []}

    client = EmptyClient()
    provider = _provider(tmp_path, client)

    assert await provider.search("no matches") == []
    assert await _provider(tmp_path, cache_mode="replay").search("no matches") == []
    assert client.search_calls == 1
    assert provider.cache_stats["writes"] == 1


@pytest.mark.parametrize("response", [{}, {"error": "search unavailable"}])
@pytest.mark.asyncio
async def test_replay_rejects_legacy_cache_missing_results(tmp_path, response):
    provider = _provider(tmp_path, FakeTavilyClient())
    await provider.search("malformed cached response")
    cache_path = next(tmp_path.glob("search-*.json"))
    cache_path.write_text(json.dumps(response), encoding="utf-8")

    with pytest.raises(TavilyCacheCorruptionError, match="missing results"):
        await _provider(tmp_path, cache_mode="replay").search("malformed cached response")


@pytest.mark.parametrize("response", [{}, {"error": "search unavailable"}])
@pytest.mark.asyncio
async def test_read_write_preserves_and_refreshes_cache_missing_results(tmp_path, response):
    provider = _provider(tmp_path, FakeTavilyClient())
    await provider.search("malformed cached response")
    cache_path = next(tmp_path.glob("search-*.json"))
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    document[CACHE_RESPONSE_KEY] = response
    document[CACHE_METADATA_KEY]["response_sha256"] = provider._response_digest(response)
    cache_path.write_text(json.dumps(document), encoding="utf-8")
    client = FakeTavilyClient()
    refreshed = _provider(tmp_path, client)

    evidence = await refreshed.search("malformed cached response")

    assert client.search_calls == 1
    assert evidence[0].snippet == "Relevant snippet"
    assert refreshed.cache_stats["corruptions"] == 1
    quarantined = list((tmp_path / ".corrupt").glob("search-*.corrupt"))
    assert len(quarantined) == 1
    assert json.loads(quarantined[0].read_text(encoding="utf-8")) == document


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


@pytest.mark.parametrize("filesystem_operation", ["read", "publish", "quarantine"])
@pytest.mark.asyncio
async def test_slow_cache_filesystem_allows_event_loop_progress(tmp_path, monkeypatch, filesystem_operation):
    client = FakeTavilyClient()
    writer = _provider(tmp_path, client)
    query = "slow filesystem"
    await writer.search(query)
    cache_path = next(tmp_path.glob("search-*.json"))
    if filesystem_operation == "publish":
        cache_path.unlink()
    elif filesystem_operation == "quarantine":
        cache_path.write_text("invalid json", encoding="utf-8")
    provider = _provider(tmp_path, client, cache_mode="replay" if filesystem_operation == "read" else "read_write")
    io_started = threading.Event()
    loop_progress = threading.Event()
    progress_during_io = []

    def slow_filesystem():
        io_started.set()
        progress_during_io.append(loop_progress.wait(timeout=1))

    if filesystem_operation == "read":
        original_read = Path.read_text

        def read(path, *args, **kwargs):
            if path == cache_path:
                slow_filesystem()
            return original_read(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read)
    elif filesystem_operation == "publish":
        original_fsync = os.fsync

        def fsync(fileno):
            slow_filesystem()
            return original_fsync(fileno)

        monkeypatch.setattr(os, "fsync", fsync)
    else:
        original_replace = os.replace

        def replace(source, target):
            if source == cache_path:
                slow_filesystem()
            return original_replace(source, target)

        monkeypatch.setattr(os, "replace", replace)

    async def observe_progress():
        while not io_started.is_set():
            await asyncio.sleep(0.001)
        loop_progress.set()

    evidence, _ = await asyncio.wait_for(asyncio.gather(provider.search(query), observe_progress()), timeout=5)

    assert progress_during_io == [True]
    assert evidence[0].snippet == "Relevant snippet"
    assert client.search_calls == (1 if filesystem_operation == "read" else 2)
    assert provider.cache_stats["corruptions"] == (1 if filesystem_operation == "quarantine" else 0)
    assert await _provider(tmp_path, cache_mode="replay").search(query) == evidence


@pytest.mark.parametrize("publication_fails", [False, True])
@pytest.mark.asyncio
async def test_cancelled_waiter_keeps_cache_publication_locked_and_replayable(
    tmp_path, monkeypatch, publication_fails
):
    first_client = FakeTavilyClient()
    second_client = FakeTavilyClient()
    first_provider = _provider(tmp_path, first_client)
    second_provider = _provider(tmp_path, second_client)
    publication_started = asyncio.Event()
    release_publication = threading.Event()
    loop = asyncio.get_running_loop()
    original_fsync = os.fsync
    original_link = os.link
    publication_error = OSError("cache publication failed") if publication_fails else None
    link_calls = 0

    def fsync(fileno):
        loop.call_soon_threadsafe(publication_started.set)
        if not release_publication.wait(timeout=5):
            raise TimeoutError("test did not release cache publication")
        return original_fsync(fileno)

    def link(source, target):
        nonlocal link_calls
        link_calls += 1
        if link_calls == 1 and publication_error is not None:
            raise publication_error
        return original_link(source, target)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "link", link)
    waiter = asyncio.create_task(first_provider.search("cancelled cache waiter"))
    publication = None
    follower = None
    cancellation = None
    try:
        await asyncio.wait_for(publication_started.wait(), timeout=5)
        [publication] = first_provider._inflight.values()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        publication.cancel("first publication cancellation")
        await asyncio.sleep(0)
        publication.cancel("second publication cancellation")
        await asyncio.sleep(0)

        follower = asyncio.create_task(second_provider.search("cancelled cache waiter"))

        async def wait_for_cache_miss():
            while second_provider.cache_stats["initial_misses"] == 0:
                await asyncio.sleep(0.001)

        await asyncio.wait_for(wait_for_cache_miss(), timeout=5)
        assert not publication.done()
        lock_path = next((tmp_path / ".locks").glob("*.lock"))
        with lock_path.open("a+b") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not follower.done()
        assert second_client.search_calls == 0
    finally:
        release_publication.set()
        if publication is not None:
            try:
                await asyncio.wait_for(publication, timeout=5)
            except asyncio.CancelledError as exc:
                cancellation = exc
        if follower is not None:
            evidence = await asyncio.wait_for(follower, timeout=5)

    assert cancellation is not None
    assert cancellation.args == ("first publication cancellation",)
    assert cancellation.__cause__ is publication_error
    assert first_client.search_calls == 1
    assert second_client.search_calls == int(publication_fails)
    assert second_provider.cache_stats["writes"] == int(publication_fails)
    assert second_provider.cache_stats["cross_process_hits"] == int(not publication_fails)
    assert await _provider(tmp_path, cache_mode="replay").search("cancelled cache waiter") == evidence
    assert not list(tmp_path.glob("*.tmp"))


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


@pytest.mark.parametrize("nonfinite_literal", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("cache_mode", ["replay", "read_write"])
@pytest.mark.asyncio
async def test_nonfinite_cache_response_is_rejected_or_preserved_and_repaired(tmp_path, nonfinite_literal, cache_mode):
    provider = _provider(tmp_path, FakeTavilyClient())
    await provider.search("nonfinite cached response")
    cache_path = next(tmp_path.glob("search-*.json"))
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    document[CACHE_RESPONSE_KEY]["response_time"] = "NONFINITE_LITERAL"
    corrupt_bytes = json.dumps(document).replace('"NONFINITE_LITERAL"', nonfinite_literal).encode()
    cache_path.write_bytes(corrupt_bytes)
    client = FakeTavilyClient()
    reader = _provider(tmp_path, client, cache_mode=cache_mode)

    if cache_mode == "replay":
        with pytest.raises(TavilyCacheCorruptionError, match="Malformed Tavily cache response") as failure:
            await reader.search("nonfinite cached response")
        assert isinstance(failure.value.__cause__, ValueError)
        assert client.search_calls == 0
        assert cache_path.read_bytes() == corrupt_bytes
        assert not (tmp_path / ".corrupt").exists()
    else:
        repaired = await reader.search("nonfinite cached response")

        assert client.search_calls == 1
        assert repaired[0].snippet == "Relevant snippet"
        assert reader.cache_stats["corruptions"] == 1
        quarantined = list((tmp_path / ".corrupt").glob("search-*.corrupt"))
        assert len(quarantined) == 1
        assert quarantined[0].read_bytes() == corrupt_bytes
        assert await _provider(tmp_path, cache_mode="replay").search("nonfinite cached response") == repaired


@pytest.mark.parametrize("format_version", [[], {}, True, 2.0, "2", None])
@pytest.mark.parametrize("cache_mode", ["replay", "read_write"])
@pytest.mark.asyncio
async def test_malformed_cache_format_is_rejected_or_preserved_and_repaired(tmp_path, format_version, cache_mode):
    provider = _provider(tmp_path, FakeTavilyClient())
    await provider.search("repair malformed format")
    cache_path = next(tmp_path.glob("search-*.json"))
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    document[CACHE_METADATA_KEY]["format_version"] = format_version
    corrupt_bytes = json.dumps(document).encode()
    cache_path.write_bytes(corrupt_bytes)
    client = FakeTavilyClient()
    reader = _provider(tmp_path, client, cache_mode=cache_mode)

    if cache_mode == "replay":
        with pytest.raises(TavilyCacheCorruptionError, match="Unsupported Tavily cache format"):
            await reader.search("repair malformed format")
        assert client.search_calls == 0
        assert cache_path.read_bytes() == corrupt_bytes
        assert not (tmp_path / ".corrupt").exists()
    else:
        repaired = await reader.search("repair malformed format")

        assert client.search_calls == 1
        assert repaired[0].snippet == "Relevant snippet"
        assert reader.cache_stats["corruptions"] == 1
        quarantined = list((tmp_path / ".corrupt").glob("search-*.corrupt"))
        assert len(quarantined) == 1
        assert quarantined[0].read_bytes() == corrupt_bytes
        assert json.loads(cache_path.read_text(encoding="utf-8"))[CACHE_METADATA_KEY]["format_version"] == (
            CACHE_FORMAT_VERSION
        )
        assert await _provider(tmp_path, cache_mode="replay").search("repair malformed format") == repaired


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
