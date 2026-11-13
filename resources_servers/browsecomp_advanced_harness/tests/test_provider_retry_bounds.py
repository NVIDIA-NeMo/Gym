# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A search/browse provider call ends: rate limits are waited out only within a wall-clock budget.

Rate limits (429, 502, 503, 504, 520) do not spend the call's attempts, so a throttled provider is waited
out. That wait is bounded by `rate_limit_wait_seconds`; past it they count as attempts and the call fails.
Every attempt also carries a total timeout and a cap on connection-error retries.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientTimeout

import resources_servers.browsecomp_advanced_harness.app as app_module
from nemo_gym.client import ServerClient
from resources_servers.browsecomp_advanced_harness.app import (
    ExaAIOHTTPClient,
    TavilySearchAIOHTTPClient,
    TavilySearchResourcesServer,
    TavilySearchResourcesServerConfig,
)


class ProviderGaveUp(Exception):
    pass


def _http_response(status: int, body: dict | None = None):
    response = MagicMock()
    response.status = status
    response.content.read = AsyncMock(return_value=json.dumps(body or {}).encode())
    response.json = AsyncMock(return_value=body or {})
    return response


def _responses(status: int, count: int = 40) -> list:
    """A finite supply of one status.

    The mock runs out and raises if a call retries more than expected, so a regression to unbounded retries fails
    at once instead of hanging: the fake clock's sleep never yields to the event loop, so a timeout cannot
    interrupt it.
    """
    return [_http_response(status) for _ in range(count)]


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A fake clock that every provider wait advances, so no test really sleeps."""
    state = SimpleNamespace(now=1000.0, sleeps=[])

    async def advance(seconds: float) -> None:
        state.sleeps.append(seconds)
        state.now += seconds

    monkeypatch.setattr(app_module, "monotonic", lambda: state.now)
    monkeypatch.setattr(app_module, "sleep", advance)
    monkeypatch.setattr(app_module, "raise_for_status", AsyncMock(side_effect=ProviderGaveUp("gave up")))
    return state


def _tavily(**kwargs) -> TavilySearchAIOHTTPClient:
    return TavilySearchAIOHTTPClient(headers={}, base_url="https://api.tavily.com", debug=False, **kwargs)


def _exa(**kwargs) -> ExaAIOHTTPClient:
    return ExaAIOHTTPClient(headers={}, base_url="https://api.exa.ai", **kwargs)


async def _call_tavily(client: TavilySearchAIOHTTPClient):
    return await client.post("/search", '{"query": "q"}', 30)


async def _call_exa(client: ExaAIOHTTPClient):
    return await client.search("q", num_results=5)


CLIENTS = [(_tavily, _call_tavily), (_exa, _call_exa)]


@pytest.mark.asyncio
@pytest.mark.parametrize(("make_client", "call"), CLIENTS)
class TestProviderRetryBounds:
    @pytest.mark.parametrize("status", [429, 502, 503, 504, 520])
    async def test_a_rate_limit_that_never_clears_ends_the_call_after_the_wait_budget(
        self, monkeypatch, clock, make_client, call, status: int
    ) -> None:
        fake_request = AsyncMock(side_effect=_responses(status))
        monkeypatch.setattr(app_module, "request", fake_request)

        with pytest.raises(ProviderGaveUp):
            await call(make_client(rate_limit_wait_seconds=10.0))

        # 0.5 s per wait: the budget is spent after 20 waits, and the 3 attempts a call is owed follow it.
        assert fake_request.await_count == 20 + 3
        assert sum(clock.sleeps) <= 10.0 + 3 * 0.5

    async def test_rate_limits_within_the_wait_budget_do_not_spend_attempts(
        self, monkeypatch, clock, make_client, call
    ) -> None:
        fake_request = AsyncMock(side_effect=[_http_response(429)] * 8 + [_http_response(200, {"results": []})])
        monkeypatch.setattr(app_module, "request", fake_request)

        await call(make_client(rate_limit_wait_seconds=600.0))

        assert fake_request.await_count == 9

    async def test_a_zero_wait_budget_spends_attempts_from_the_first_rate_limit(
        self, monkeypatch, clock, make_client, call
    ) -> None:
        fake_request = AsyncMock(side_effect=_responses(429))
        monkeypatch.setattr(app_module, "request", fake_request)

        with pytest.raises(ProviderGaveUp):
            await call(make_client(rate_limit_wait_seconds=0.0))

        assert fake_request.await_count == 3

    async def test_a_server_error_is_still_three_attempts(self, monkeypatch, clock, make_client, call) -> None:
        fake_request = AsyncMock(side_effect=_responses(500))
        monkeypatch.setattr(app_module, "request", fake_request)

        with pytest.raises(ProviderGaveUp):
            await call(make_client())

        assert fake_request.await_count == 3

    async def test_every_attempt_has_a_total_timeout_and_a_connection_retry_cap(
        self, monkeypatch, clock, make_client, call
    ) -> None:
        fake_request = AsyncMock(return_value=_http_response(200, {"results": []}))
        monkeypatch.setattr(app_module, "request", fake_request)

        await call(make_client(request_timeout_seconds=45.0))

        kwargs = fake_request.await_args.kwargs
        assert kwargs["timeout"] == ClientTimeout(total=45.0)
        assert kwargs["_max_connection_retries"] == 3


_EXCLUDE_DOMAINS_FILE = str(Path(__file__).parent / "dummy_exclude_domains_file.json")


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ({}, (600.0, 120.0)),
        ({"provider_rate_limit_wait_seconds": 7.0, "provider_request_timeout_seconds": 9.0}, (7.0, 9.0)),
    ],
)
def test_the_configured_bounds_reach_every_client_the_server_builds(limits: dict, expected: tuple) -> None:
    """The server forwards its limits to each Tavily and Exa client (several keys make several clients)."""
    config = TavilySearchResourcesServerConfig(
        host="h",
        port=1,
        entrypoint="app.py",
        name="n",
        exclude_domains_file_path=_EXCLUDE_DOMAINS_FILE,
        tavily_api_key=["tavily-key-1", "tavily-key-2"],  # pragma: allowlist secret
        exa_api_key=["exa-key-1", "exa-key-2"],  # pragma: allowlist secret
        **limits,
    )

    server = TavilySearchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    clients = [tavily._client for tavily in server._async_tavily_clients] + list(server._exa_clients)
    assert len(clients) == 4
    assert {(client.rate_limit_wait_seconds, client.request_timeout_seconds) for client in clients} == {expected}
