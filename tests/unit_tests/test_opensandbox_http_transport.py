# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenSandbox SDK requests share Gym's pool without owning its lifetime."""

import asyncio
import gzip
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import aiohttp
import httpx
import pytest
from aiohttp import web

from nemo_gym import server_utils
from nemo_gym.sandbox.providers import _http_transport
from nemo_gym.sandbox.providers._http_transport import GymAiohttpTransport
from nemo_gym.sandbox.providers.e2b import _sdk as e2b_sdk
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider


pytestmark = pytest.mark.sandbox


@pytest.fixture
async def shared_server(monkeypatch):
    calls = []
    replies = []
    entered = asyncio.Event()
    release = asyncio.Event()

    async def echo(request):
        calls.append((request.transport, request.headers, await request.read()))
        if replies:
            status, headers = replies.pop(0)
            return web.Response(status=status, headers=headers, body=b"rejected")
        if request.path == "/broken":
            response = web.StreamResponse(headers={"Content-Length": "100"})
            await response.prepare(request)
            await response.write(b"short")
            request.transport.close()
            return response
        if request.path == "/hold":
            entered.set()
            await release.wait()
        return web.Response(body=gzip.compress(b"response body"), headers={"Content-Encoding": "gzip"})

    app = web.Application()
    app.router.add_route("*", "/{path}", echo)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=1, limit_per_host=1)) as client:
        monkeypatch.setattr(server_utils, "_GLOBAL_AIOHTTP_CLIENT", client)
        try:
            yield SimpleNamespace(
                url=f"http://127.0.0.1:{port}",
                client=client,
                calls=calls,
                replies=replies,
                entered=entered,
                release=release,
            )
        finally:
            release.set()
            await runner.cleanup()


async def test_providers_reuse_global_connections_and_do_not_close_them(shared_server):
    first = OpenSandboxProvider()
    second = OpenSandboxProvider()

    async def body():
        yield b"chunk one"
        yield b"chunk two"

    async with httpx.AsyncClient(transport=first._get_transport()) as client:
        async with client.stream(
            "POST", shared_server.url + "/echo", content=body(), headers={"OPEN-SANDBOX-API-KEY": "first"}
        ) as response:
            # Raw streaming must stay compressed: HTTPX owns decompression.
            raw = b"".join([part async for part in response.aiter_raw()])
            assert gzip.decompress(raw) == b"response body"
    await first.aclose()
    assert not shared_server.client.closed

    async with httpx.AsyncClient(transport=second._get_transport()) as client:
        response = await client.get(shared_server.url + "/echo", headers={"OPEN-SANDBOX-API-KEY": "second"})
        assert response.content == b"response body"
    await second.aclose()
    assert not shared_server.client.closed
    assert shared_server.calls[0][0] is shared_server.calls[1][0]
    assert shared_server.calls[0][1]["OPEN-SANDBOX-API-KEY"] == "first"
    assert shared_server.calls[1][1]["OPEN-SANDBOX-API-KEY"] == "second"
    assert shared_server.calls[0][2] == b"chunk onechunk two"


async def test_global_connection_limit_applies_across_providers(shared_server):
    first = OpenSandboxProvider()
    second = OpenSandboxProvider()
    async with (
        httpx.AsyncClient(transport=first._get_transport()) as one,
        httpx.AsyncClient(transport=second._get_transport()) as two,
    ):
        pending = asyncio.create_task(one.get(shared_server.url + "/hold"))
        try:
            await asyncio.wait_for(shared_server.entered.wait(), timeout=5)
            with pytest.raises(httpx.TimeoutException):
                await two.get(shared_server.url + "/echo", timeout=httpx.Timeout(5, pool=0.05))
            assert len(shared_server.calls) == 1
        finally:
            shared_server.release.set()
            await pending
        assert (await two.get(shared_server.url + "/echo")).status_code == 200
    await first.aclose()
    await second.aclose()


@pytest.mark.parametrize("verify", [False, True])
async def test_tls_is_set_per_request_and_timeouts_are_not_retried(monkeypatch, verify):
    provider = OpenSandboxProvider(connection={"tls_verify": verify})
    request = AsyncMock(side_effect=aiohttp.SocketTimeoutError("read timed out"))
    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: SimpleNamespace(request=request))
    async with httpx.AsyncClient(transport=provider._get_transport()) as client:
        with pytest.raises(httpx.ReadTimeout, match="read timed out"):
            await client.post("https://sandbox.example/command", content=b"run once", timeout=7)
    request.assert_awaited_once()
    kwargs = request.call_args.kwargs
    assert kwargs["ssl"] is verify
    assert kwargs["timeout"].sock_connect == 7
    assert kwargs["timeout"].sock_read == 7
    assert kwargs["timeout"].connect == 7
    assert kwargs["data"] == b"run once"
    await provider.aclose()


async def test_e2b_and_opensandbox_share_connections(shared_server):
    e2b = pytest.importorskip("e2b")
    from e2b.api import client_async

    e2b_sdk.require_e2b_sdk("Testing shared sandbox HTTP")
    provider = OpenSandboxProvider(connection={"tls_verify": True})
    async with httpx.AsyncClient(transport=client_async.get_transport(e2b.ConnectionConfig())) as client:
        assert (await client.get(shared_server.url + "/echo")).content == b"response body"
    assert not shared_server.client.closed
    async with httpx.AsyncClient(transport=provider._get_transport()) as client:
        assert (await client.get(shared_server.url + "/echo")).content == b"response body"
    await provider.aclose()
    assert shared_server.calls[0][0] is shared_server.calls[1][0]
    assert not shared_server.client.closed


@pytest.mark.parametrize(
    "error,expected",
    [
        (aiohttp.SocketTimeoutError("timeout"), httpx.ReadTimeout),
        (aiohttp.ConnectionTimeoutError("timeout"), httpx.ConnectTimeout),
        (TimeoutError("timeout"), httpx.TimeoutException),
        (aiohttp.ClientConnectionError("connection"), httpx.ConnectError),
        (aiohttp.ClientPayloadError("payload"), httpx.ReadError),
        (aiohttp.ServerDisconnectedError("disconnected"), httpx.ReadError),
        (aiohttp.InvalidURL("bad-url"), httpx.UnsupportedProtocol),
        (aiohttp.NonHttpUrlClientError("ftp://example"), httpx.UnsupportedProtocol),
        (
            aiohttp.ClientHttpProxyError(SimpleNamespace(real_url="http://proxy.example"), (), status=407),
            httpx.ProxyError,
        ),
        (RuntimeError("unexpected"), RuntimeError),
    ],
)
async def test_transport_maps_errors_without_retrying(monkeypatch, error, expected):
    request = AsyncMock(side_effect=error)
    monkeypatch.setattr(server_utils, "request", request)
    async with httpx.AsyncClient(transport=GymAiohttpTransport()) as client:
        with pytest.raises(expected):
            await client.post("https://sandbox.example/command", content=b"run once")
    request.assert_awaited_once()
    assert request.call_args.kwargs["_max_connection_retries"] == 1


async def test_transport_preserves_proxy_auth_and_headers(monkeypatch):
    proxy = httpx.Proxy("http://proxy.example:8080", auth=("user", "password"), headers={"X-Proxy": "value"})
    request = AsyncMock(side_effect=aiohttp.ConnectionTimeoutError("timeout"))
    monkeypatch.setattr(server_utils, "request", request)
    async with httpx.AsyncClient(transport=GymAiohttpTransport(proxy=proxy)) as client:
        with pytest.raises(httpx.ConnectTimeout):
            await client.get("https://sandbox.example/command")
    kwargs = request.call_args.kwargs
    assert kwargs["proxy"] == "http://proxy.example:8080"
    assert kwargs["proxy_auth"] == aiohttp.BasicAuth("user", "password")
    assert kwargs["proxy_headers"]["X-Proxy"] == "value"


async def test_truncated_stream_raises_read_error_and_releases_connection(shared_server):
    async with httpx.AsyncClient(transport=GymAiohttpTransport()) as client:
        with pytest.raises(httpx.ReadError):
            await client.get(shared_server.url + "/broken")
        response = await client.get(shared_server.url + "/echo", timeout=1)
        assert response.content == b"response body"
    assert not shared_server.client.closed


@pytest.mark.parametrize(
    "retry_after,expected_delay",
    [
        ("27", 27.0),
        (" 120 ", 120.0),
        ("0", 0.0),
        ("Wed, 23 Sep 2026 18:42:09 GMT", 27.0),
        ("Wed, 23 Sep 2026 18:41:00 GMT", 0.0),
        (None, 1.0),
        ("invalid", 1.0),
        ("", 1.0),
        ("-1", 1.0),
        ("NaN", 1.0),
        ("inf", 1.0),
    ],
)
async def test_rate_limit_retries_preserve_body_and_honor_retry_after(
    shared_server, monkeypatch, retry_after, expected_delay
):
    monkeypatch.setattr(_http_transport.time, "time", lambda: 1790188902.0)
    sleep = AsyncMock()
    monkeypatch.setattr(_http_transport, "asyncio", SimpleNamespace(sleep=sleep))
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    shared_server.replies.append((429, headers))
    async with httpx.AsyncClient(transport=GymAiohttpTransport()) as client:
        response = await client.post(shared_server.url + "/sandboxes", content=b"create sandbox", timeout=1)
    assert response.status_code == 200
    assert response.content == b"response body"
    sleep.assert_awaited_once_with(expected_delay)
    assert [body for _, _, body in shared_server.calls] == [b"create sandbox", b"create sandbox"]
    assert not shared_server.client.closed


async def test_rate_limit_retries_are_bounded_and_preserve_final_response(shared_server, monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(_http_transport, "asyncio", SimpleNamespace(sleep=sleep))
    shared_server.replies.extend([(429, {})] * 4)
    async with httpx.AsyncClient(transport=GymAiohttpTransport()) as client:
        response = await client.post(shared_server.url + "/sandboxes", json={"image": "test"}, timeout=1)
        assert response.status_code == 429
        assert response.content == b"rejected"
        assert len(shared_server.calls) == 4
        assert sleep.await_args_list == [call(1.0), call(2.0), call(4.0)]
        assert (await client.get(shared_server.url + "/echo", timeout=1)).status_code == 200


@pytest.mark.parametrize("status", [400, 401, 403, 500, 503])
async def test_other_http_errors_are_not_retried(shared_server, monkeypatch, status):
    sleep = AsyncMock()
    monkeypatch.setattr(_http_transport, "asyncio", SimpleNamespace(sleep=sleep))
    shared_server.replies.append((status, {"Retry-After": "27"}))
    async with httpx.AsyncClient(transport=GymAiohttpTransport()) as client:
        response = await client.post(shared_server.url + "/command", content=b"run once")
    assert response.status_code == status
    assert response.content == b"rejected"
    assert len(shared_server.calls) == 1
    sleep.assert_not_awaited()


async def test_rate_limited_upload_stream_is_not_replayed(shared_server, monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(_http_transport, "asyncio", SimpleNamespace(sleep=sleep))
    shared_server.replies.append((429, {"Retry-After": "27"}))

    async def body():
        yield b"streamed upload"

    async with httpx.AsyncClient(transport=GymAiohttpTransport()) as client:
        response = await client.post(shared_server.url + "/upload", content=body())
    assert response.status_code == 429
    assert response.content == b"rejected"
    assert [body for _, _, body in shared_server.calls] == [b"streamed upload"]
    sleep.assert_not_awaited()


async def test_cancelling_rate_limit_sleep_closes_response_without_retrying(monkeypatch):
    response = Mock(status=429, headers={"Retry-After": "27"})
    send = AsyncMock(return_value=response)
    monkeypatch.setattr(server_utils, "request", send)

    async def cancel_sleep(delay):
        response.close.assert_called_once_with()
        raise asyncio.CancelledError

    monkeypatch.setattr(_http_transport, "asyncio", SimpleNamespace(sleep=cancel_sleep))
    async with httpx.AsyncClient(transport=GymAiohttpTransport()) as client:
        with pytest.raises(asyncio.CancelledError):
            await client.post("https://sandbox.example/sandboxes", content=b"create sandbox")
    send.assert_awaited_once()
