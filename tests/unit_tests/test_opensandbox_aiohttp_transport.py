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

import asyncio
import time
from typing import AsyncIterator

import aiohttp
import httpx
import pytest
from aiohttp import web

from nemo_gym.sandbox.providers.opensandbox.aiohttp_transport import AiohttpTransport, _map_exception


pytestmark = pytest.mark.sandbox


@pytest.fixture
async def server_url() -> AsyncIterator[str]:
    """Local aiohttp server exercising the transport behaviors that matter for the SDK."""

    async def echo_json(request: web.Request) -> web.Response:
        return web.json_response({"echo": await request.json(), "method": request.method})

    async def health(request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def sse(request: web.Request) -> web.StreamResponse:
        # SSE-style chunked stream: ?chunks=N&delay=S between chunks.
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i in range(int(request.query.get("chunks", "4"))):
            await resp.write(f"data: {i}\n\n".encode())
            await asyncio.sleep(float(request.query.get("delay", "0")))
        await resp.write_eof()
        return resp

    async def stall(request: web.Request) -> web.StreamResponse:
        # Two quick chunks, then silence: a stalled-but-open stream.
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i in range(2):
            await resp.write(f"data: {i}\n\n".encode())
            await asyncio.sleep(0.02)
        try:
            await asyncio.sleep(30)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def abort(request: web.Request) -> web.StreamResponse:
        # Hard TCP close mid-body: what a proxy reaping a connection looks like.
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b"data: 0\n\n")
        await asyncio.sleep(0.02)
        request.transport.abort()
        return resp

    async def redirect(request: web.Request) -> web.Response:
        raise web.HTTPFound("/health")

    app = web.Application()
    app.router.add_post("/json", echo_json)
    app.router.add_get("/health", health)
    app.router.add_get("/sse", sse)
    app.router.add_get("/stall", stall)
    app.router.add_get("/abort", abort)
    app.router.add_get("/redirect", redirect)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    yield f"http://127.0.0.1:{port}"
    await runner.cleanup()


def _client(timeout: httpx.Timeout | float = 10.0, **transport_kwargs: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=AiohttpTransport(**transport_kwargs), timeout=timeout)


async def test_roundtrip_and_concurrency(server_url: str) -> None:
    async with _client() as client:
        response = await client.post(f"{server_url}/json", json={"a": [1, 2]})
        assert response.status_code == 200
        assert response.json() == {"echo": {"a": [1, 2]}, "method": "POST"}

        results = await asyncio.gather(*(client.get(f"{server_url}/health") for _ in range(30)))
        assert all(r.status_code == 200 for r in results)


async def test_concurrency_not_capped_by_keepalive_limit(server_url: str) -> None:
    # Regression: max_keepalive_connections (idle-connection cap in httpx)
    # must not become aiohttp's limit_per_host (active-connection cap). All
    # traffic targets one host, so that mis-mapping serializes requests in
    # waves of the keepalive count: 8 requests of ~0.4s each at a cap of 2
    # would need 4 waves (~1.6s) instead of one (~0.4s).
    limits = httpx.Limits(max_connections=50, max_keepalive_connections=2)
    async with _client(limits=limits) as client:
        start = time.monotonic()
        responses = await asyncio.gather(
            *(client.get(f"{server_url}/sse", params={"chunks": 1, "delay": 0.4}) for _ in range(8))
        )
        elapsed = time.monotonic() - start
    assert all(r.status_code == 200 for r in responses)
    assert elapsed < 1.2, f"requests were serialized by a per-host cap: {elapsed:.2f}s for 8 x 0.4s"


async def test_streaming_is_incremental(server_url: str) -> None:
    # Chunks written 0.3s apart must be observed as they arrive; a transport
    # that buffers the body would deliver them all at once (near-zero gaps).
    arrivals = []
    async with _client() as client:
        async with client.stream("GET", f"{server_url}/sse", params={"chunks": 4, "delay": 0.3}) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    arrivals.append(time.monotonic())
    assert len(arrivals) == 4
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    assert all(gap > 0.1 for gap in gaps), f"stream was buffered, inter-chunk gaps: {gaps}"


async def test_mid_stream_stall_raises_read_timeout(server_url: str) -> None:
    async with _client(timeout=httpx.Timeout(10.0, read=0.3)) as client:
        chunks = 0
        with pytest.raises(httpx.ReadTimeout):
            async with client.stream("GET", f"{server_url}/stall") as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        chunks += 1
        assert chunks == 2  # data flowed until the stall


async def test_read_none_stream_outlives_other_timeouts(server_url: str) -> None:
    # The SDK's SSE client disables the read timeout but keeps connect/write
    # bounds; a stream whose total duration exceeds those bounds must survive
    # (guards against mapping any timeout onto the whole stream).
    timeout = httpx.Timeout(connect=0.3, read=None, write=0.3, pool=None)
    async with _client(timeout=timeout) as client:
        chunks = 0
        async with client.stream("GET", f"{server_url}/sse", params={"chunks": 4, "delay": 0.25}) as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    chunks += 1
        assert chunks == 4


async def test_mid_stream_disconnect_raises_diagnostic_read_error(server_url: str) -> None:
    async with _client() as client:
        with pytest.raises(httpx.ReadError) as exc_info:
            async with client.stream("GET", f"{server_url}/abort") as response:
                async for _ in response.aiter_lines():
                    pass
    # The message must say what happened, not just aiohttp's framing-parser
    # text, and the original aiohttp exception must stay chained for debugging.
    assert "connection closed before the response body completed" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, aiohttp.ClientPayloadError)


async def test_abandoned_stream_closes_cleanly_and_pool_survives(server_url: str) -> None:
    async with _client() as client:
        async with client.stream("GET", f"{server_url}/sse", params={"chunks": 500}) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: 2"):
                    break  # abandon mid-body; aclose must not raise
        followup = await client.get(f"{server_url}/health")
        assert followup.status_code == 200


async def test_redirects_stay_with_httpx(server_url: str) -> None:
    async with _client() as client:
        response = await client.get(f"{server_url}/redirect")
        assert response.status_code == 302
        followed = await client.get(f"{server_url}/redirect", follow_redirects=True)
        assert followed.status_code == 200 and len(followed.history) == 1


async def test_connect_error_maps_and_retries_run(server_url: str) -> None:
    async with _client(retries=2) as client:
        with pytest.raises(httpx.ConnectError):
            await client.get("http://127.0.0.1:9/health")  # discard port: nothing listens


async def test_injected_session_is_shared_and_not_closed(server_url: str) -> None:
    session = aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar())
    try:
        async with _client(client=session) as client:
            response = await client.get(f"{server_url}/health")
            assert response.status_code == 200
        assert not session.closed  # externally-owned session survives aclose()
    finally:
        await session.close()


async def test_limits_map_to_connector() -> None:
    transport = AiohttpTransport(
        limits=httpx.Limits(max_connections=7, max_keepalive_connections=3, keepalive_expiry=2.5)
    )
    session = await transport._get_session()
    try:
        assert session.connector.limit == 7
        # max_keepalive_connections must NOT reach limit_per_host: httpx's
        # knob caps idle connections while aiohttp's caps active concurrent
        # connections per host, and all this transport's traffic targets one
        # host, so mapping it would cap concurrency at the keepalive count.
        assert session.connector.limit_per_host == 0
        assert session.connector._keepalive_timeout == 2.5
    finally:
        await transport.aclose()
    assert session.closed  # transport-owned session closes with the transport

    unlimited = AiohttpTransport(limits=httpx.Limits(max_connections=None, max_keepalive_connections=0))
    session = await unlimited._get_session()
    try:
        assert session.connector.limit == 0  # aiohttp spelling of "unlimited"
        assert session.connector.limit_per_host == 0
    finally:
        await unlimited.aclose()

    # force_close (disable_connection_pooling) reaches the connector; the
    # keepalive timeout must be omitted then (aiohttp rejects the combination).
    no_pooling = AiohttpTransport(limits=httpx.Limits(max_connections=7, keepalive_expiry=2.5), force_close=True)
    session = await no_pooling._get_session()
    try:
        assert session.connector.force_close is True
    finally:
        await no_pooling.aclose()


def test_exception_taxonomy() -> None:
    request = httpx.Request("GET", "http://example.invalid/")

    def mapped(exc: Exception, *, in_body: bool = True) -> httpx.RequestError | None:
        return _map_exception(exc, request, in_body=in_body)

    # Payload errors: read timeout when a timeout caused them, else a read
    # error whose message states the connection closed mid-response.
    payload = aiohttp.ClientPayloadError("Response payload is not completed")
    assert type(mapped(payload)) is httpx.ReadError
    assert "connection closed before the response body completed" in str(mapped(payload))
    timed_out = aiohttp.ClientPayloadError("Response payload is not completed")
    timed_out.__cause__ = asyncio.TimeoutError()
    assert type(mapped(timed_out)) is httpx.ReadTimeout

    assert type(mapped(aiohttp.SocketTimeoutError())) is httpx.ReadTimeout
    assert type(mapped(aiohttp.ConnectionTimeoutError())) is httpx.ConnectTimeout
    assert type(mapped(aiohttp.ServerTimeoutError())) is httpx.TimeoutException
    assert type(mapped(asyncio.TimeoutError(), in_body=True)) is httpx.ReadTimeout
    assert type(mapped(asyncio.TimeoutError(), in_body=False)) is httpx.ConnectTimeout
    assert type(mapped(aiohttp.ServerDisconnectedError())) is httpx.RemoteProtocolError
    # Bare connection errors are phase-dependent: connect failure before the
    # response starts, read failure once the body is streaming.
    assert type(mapped(aiohttp.ClientConnectionError(), in_body=False)) is httpx.ConnectError
    assert type(mapped(aiohttp.ClientConnectionError(), in_body=True)) is httpx.ReadError
    assert type(mapped(aiohttp.ClientError())) is httpx.TransportError
    assert mapped(ValueError("not an aiohttp error")) is None
