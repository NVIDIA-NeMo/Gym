# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact provider rate-limit counting.

True 429s must be counted exactly (bc_frankie parity: `status_code == 429`),
never conflated with the other retryable statuses (500/502/503/504/520) that
share the retry loop's RATE_LIMIT/RETRY policy buckets. Counts propagate from
the client retry loops (via a task-local ContextVar) into the per-call metrics
records and are summed into top-level ints on the verify response, where Gym's
aggregator reports them alongside reward.
"""

import json
from time import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import resources_servers.browsecomp_advanced_harness.app as app_module
from resources_servers.browsecomp_advanced_harness.app import (
    _PROVIDER_RETRY_COUNTS,
    ExaAIOHTTPClient,
    TavilySearchAIOHTTPClient,
    TavilySearchMetrics,
    TavilySearchResourcesServer,
    TavilySearchSingleAsyncTavilyMetrics,
    _sum_provider_retry_counts,
)


def _http_response(status: int, body: dict):
    r = MagicMock()
    r.status = status
    r.content.read = AsyncMock(return_value=json.dumps(body).encode())
    r.json = AsyncMock(return_value=body)
    return r


@pytest.fixture(autouse=True)
def _clear_retry_counts():
    _PROVIDER_RETRY_COUNTS.set(None)
    yield
    _PROVIDER_RETRY_COUNTS.set(None)


@pytest.mark.asyncio
class TestClientRetryCounting:
    async def test_tavily_429_counted_exactly(self, monkeypatch) -> None:
        client = TavilySearchAIOHTTPClient(headers={}, base_url="https://api.tavily.com", debug=False)
        fake_request = AsyncMock(
            side_effect=[_http_response(429, {}), _http_response(429, {}), _http_response(200, {"results": []})]
        )
        monkeypatch.setattr(app_module, "request", fake_request)

        resp = await client.post("/search", '{"query": "q"}', 30)

        assert resp.status_code == 200
        assert _PROVIDER_RETRY_COUNTS.get() == {"num_429_retries": 2, "num_other_retries": 0}

    async def test_tavily_502_not_counted_as_429(self, monkeypatch) -> None:
        client = TavilySearchAIOHTTPClient(headers={}, base_url="https://api.tavily.com", debug=False)
        fake_request = AsyncMock(side_effect=[_http_response(502, {}), _http_response(200, {"results": []})])
        monkeypatch.setattr(app_module, "request", fake_request)

        await client.post("/search", '{"query": "q"}', 30)

        assert _PROVIDER_RETRY_COUNTS.get() == {"num_429_retries": 0, "num_other_retries": 1}

    async def test_exa_429_counted_exactly(self, monkeypatch) -> None:
        client = ExaAIOHTTPClient(headers={}, base_url="https://api.exa.ai", debug=False)
        fake_request = AsyncMock(side_effect=[_http_response(429, {}), _http_response(200, {"results": []})])
        monkeypatch.setattr(app_module, "request", fake_request)

        await client.search("q", num_results=5)

        assert _PROVIDER_RETRY_COUNTS.get() == {"num_429_retries": 1, "num_other_retries": 0}

    async def test_no_retries_leaves_counts_unset(self, monkeypatch) -> None:
        client = TavilySearchAIOHTTPClient(headers={}, base_url="https://api.tavily.com", debug=False)
        fake_request = AsyncMock(return_value=_http_response(200, {"results": []}))
        monkeypatch.setattr(app_module, "request", fake_request)

        await client.post("/search", '{"query": "q"}', 30)

        assert _PROVIDER_RETRY_COUNTS.get() is None


class TestRecordCall:
    def test_record_call_moves_counts_into_record_and_resets(self) -> None:
        metrics = TavilySearchMetrics()
        _PROVIDER_RETRY_COUNTS.set({"num_429_retries": 3, "num_other_retries": 1})

        TavilySearchResourcesServer._record_call(MagicMock(), metrics, "search", "tavily", "success", time())

        rec = metrics.async_tavily_calls[-1]
        assert rec.num_429_retries == 3
        assert rec.num_other_retries == 1
        # Reset so the next call in this task starts from zero.
        assert _PROVIDER_RETRY_COUNTS.get() is None

    def test_record_call_defaults_to_zero(self) -> None:
        metrics = TavilySearchMetrics()
        TavilySearchResourcesServer._record_call(MagicMock(), metrics, "browse", "exa", "error", time())
        rec = metrics.async_tavily_calls[-1]
        assert rec.num_429_retries == 0
        assert rec.num_other_retries == 0


class TestVerifySummation:
    def test_sum_provider_retry_counts(self) -> None:
        metrics = TavilySearchMetrics(
            async_tavily_calls=[
                TavilySearchSingleAsyncTavilyMetrics(
                    function="search",
                    provider="tavily",
                    status="success",
                    start_time=0.0,
                    end_time=1.0,
                    num_429_retries=2,
                    num_other_retries=0,
                ),
                TavilySearchSingleAsyncTavilyMetrics(
                    function="browse",
                    provider="tavily",
                    status="error",
                    start_time=0.0,
                    end_time=1.0,
                    num_429_retries=1,
                    num_other_retries=4,
                ),
            ]
        )
        assert _sum_provider_retry_counts(metrics) == (3, 4)

    def test_old_records_without_fields_sum_to_zero(self) -> None:
        metrics = TavilySearchMetrics(
            async_tavily_calls=[
                TavilySearchSingleAsyncTavilyMetrics(
                    function="search", provider="tavily", status="success", start_time=0.0, end_time=1.0
                )
            ]
        )
        assert _sum_provider_retry_counts(metrics) == (0, 0)
