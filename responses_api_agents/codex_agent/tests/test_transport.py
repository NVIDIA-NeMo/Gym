# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import socket
import time
from contextlib import asynccontextmanager

import pytest
from aiohttp import web

from nemo_gym.client import get_global_aiohttp_client, request
from responses_api_agents.codex_agent.transport import model_retry_proxy, retry_delay


@pytest.mark.parametrize(
    "statuses",
    [(429, 429, 200), (429, 429, 429), (408, 425, 200), (500, 503, 503), (400,), (401,), (404,), (409,), (422,)],
)
async def test_only_statuses_that_may_clear_retry_and_every_attempt_keeps_body_and_owner(
    statuses, monkeypatch
) -> None:
    monkeypatch.setattr("nemo_gym.client.http_client.get_global_config_dict", lambda **kwargs: {})
    monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
    seen = []

    async def upstream(incoming):
        seen.append((await incoming.read(), incoming.headers.getall("x-session-id")))
        return web.Response(status=statuses[len(seen) - 1], body=b"payload", headers={"Retry-After": "0"})

    app = web.Application()
    app.router.add_post("/responses", upstream)
    runner = web.AppRunner(app)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        await runner.setup()
        try:
            await web.SockSite(runner, sock).start()
            async with model_retry_proxy(f"http://127.0.0.1:{sock.getsockname()[1]}", "owner") as url:
                response = await request(
                    "POST", f"{url}/responses", data=b'{"input":[]}', headers={"X-Session-Id": "wrong"}
                )
                try:
                    assert response.status == statuses[-1]
                    assert await response.read() == b"payload"
                finally:
                    response.release()
            assert seen == [(b'{"input":[]}', ["owner"])] * len(statuses)
        finally:
            await runner.cleanup()
            await get_global_aiohttp_client().close()


@pytest.mark.parametrize(
    ("retry_after", "delays"),
    [
        (None, [1, 2]),
        ("0", [0, 0]),
        (" 2.5 ", [2.5, 2.5]),
        ("45", [10, 10]),
        ("-3", [0, 0]),
        ("nan", [1, 2]),
        ("inf", [1, 2]),
        ("soon", [1, 2]),
    ],
)
def test_retry_delay_is_bounded_and_invalid_values_back_off(retry_after, delays) -> None:
    assert [retry_delay(retry_after, attempt) for attempt in (0, 1)] == delays


async def test_refused_connection_is_retried_twice_then_answers_502(monkeypatch) -> None:
    monkeypatch.setattr("nemo_gym.client.http_client.get_global_config_dict", lambda **kwargs: {})
    monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
    waits = []

    def delay(retry_after, attempt):
        waits.append((retry_after, attempt))
        return 0

    monkeypatch.setattr("responses_api_agents.codex_agent.transport.retry_delay", delay)
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        port = unused.getsockname()[1]
    try:
        async with model_retry_proxy(f"http://127.0.0.1:{port}", "owner") as url:
            response = await asyncio.wait_for(request("POST", f"{url}/responses", data=b"{}"), 5)
            try:
                assert response.status == 502
            finally:
                response.release()
        assert waits == [(None, 0), (None, 1)]
    finally:
        await get_global_aiohttp_client().close()


@pytest.fixture
async def http_client(monkeypatch):
    monkeypatch.setattr("nemo_gym.client.http_client.get_global_config_dict", lambda **kwargs: {})
    monkeypatch.setattr("nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT", None)
    yield
    await get_global_aiohttp_client().close()


@asynccontextmanager
async def upstream_server(handler):
    app = web.Application()
    app.router.add_post("/responses", handler)
    runner = web.AppRunner(app, handler_cancellation=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        await runner.setup()
        try:
            await web.SockSite(runner, sock).start()
            yield f"http://127.0.0.1:{sock.getsockname()[1]}"
        finally:
            await runner.cleanup()


@pytest.mark.parametrize("status", [429, 503])
async def test_truncated_retryable_body_is_resent(http_client, status) -> None:
    attempts = []

    async def upstream(incoming):
        attempts.append(await incoming.read())
        if len(attempts) == 1:
            response = web.StreamResponse(status=status, headers={"Content-Length": "100", "Retry-After": "0"})
            await response.prepare(incoming)
            await response.write(b"partial")
            incoming.transport.close()
            return response
        return web.Response(body=b"done")

    async with upstream_server(upstream) as base, model_retry_proxy(base, "owner") as url:
        response = await asyncio.wait_for(request("POST", f"{url}/responses", data=b"{}"), 5)
        try:
            assert response.status == 200
            assert await response.read() == b"done"
            assert attempts == [b"{}", b"{}"]
        finally:
            response.release()


@pytest.mark.parametrize("drops", [1, 3])
async def test_dropped_connection_is_resent(http_client, monkeypatch, drops) -> None:
    monkeypatch.setattr("responses_api_agents.codex_agent.transport.retry_delay", lambda *_: 0)
    attempts = []

    async def upstream(incoming):
        attempts.append(await incoming.read())
        if len(attempts) <= drops:
            incoming.transport.close()
        return web.Response(body=b"done")

    async with upstream_server(upstream) as base, model_retry_proxy(base, "owner") as url:
        response = await asyncio.wait_for(request("POST", f"{url}/responses", data=b"{}"), 5)
        try:
            assert response.status == (200 if drops == 1 else 502)
            assert len(attempts) == (2 if drops == 1 else 3)
        finally:
            response.release()


async def test_connection_timeout_is_not_resent(http_client, monkeypatch) -> None:
    from aiohttp import ConnectionTimeoutError

    calls = []

    async def timed_out(*args, **kwargs):
        calls.append(kwargs)
        raise ConnectionTimeoutError("connection timed out")

    monkeypatch.setattr("responses_api_agents.codex_agent.transport.request", timed_out)
    async with model_retry_proxy("http://model", "owner") as url:
        response = await asyncio.wait_for(request("POST", f"{url}/responses", data=b"{}"), 5)
        try:
            assert response.status == 502
            assert len(calls) == 1
            assert calls[0]["_max_num_tries"] == 1
        finally:
            response.release()


async def test_retry_after_controls_the_actual_wait(http_client) -> None:
    timestamps = []

    async def upstream(incoming):
        timestamps.append(time.monotonic())
        return web.Response(status=503 if len(timestamps) == 1 else 200, headers={"Retry-After": "1.5"})

    async with upstream_server(upstream) as base, model_retry_proxy(base, "owner") as url:
        response = await asyncio.wait_for(request("POST", f"{url}/responses", data=b"{}"), 5)
        try:
            assert response.status == 200
            assert len(timestamps) == 2
            assert timestamps[1] - timestamps[0] >= 1.5
        finally:
            response.release()


async def test_caller_disconnect_stops_resends(http_client) -> None:
    attempts = []
    received = asyncio.Event()

    async def upstream(incoming):
        attempts.append(await incoming.read())
        received.set()
        return web.Response(status=503, headers={"Retry-After": "1.5"})

    async with upstream_server(upstream) as base:
        async with model_retry_proxy(base, "owner") as url:
            pending = asyncio.create_task(request("POST", f"{url}/responses", data=b"{}"))
            await asyncio.wait_for(received.wait(), 5)
            # Let the reply reach the forwarder and enter its Retry-After sleep.
            await asyncio.sleep(0.05)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        await asyncio.sleep(1.6)
        assert attempts == [b"{}"]
