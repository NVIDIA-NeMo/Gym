# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gzip

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from nemo_gym.models.policy_admission import create_app


@pytest.mark.asyncio
async def test_admission_holds_permits_through_streaming_and_bounds_waiters():
    released = asyncio.Event()
    calls = []

    async def generate(request):
        calls.append(await request.read())
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: first\n\n")
        await released.wait()
        await response.write(b"data: last\n\n")
        return response

    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/chat/completions", generate)
    async with TestServer(upstream_app) as upstream:
        app = create_app(upstream=str(upstream.make_url("/")), max_inflight=1, max_queued=1)
        async with TestServer(app, handler_cancellation=True) as proxy, aiohttp.ClientSession() as client:
            url = proxy.make_url("/v1/chat/completions")
            first = await client.post(url, data=b"one")
            assert await first.content.readline() == b"data: first\n"
            second = asyncio.create_task(client.post(url, data=b"two"))
            try:
                for _ in range(100):
                    async with client.get(proxy.make_url("/health")) as health:
                        if (await health.json())["outstanding"] == 2:
                            break
                    await asyncio.sleep(0.01)
                else:
                    pytest.fail("Second request was not queued")
                async with client.post(url, data=b"three") as rejected:
                    assert rejected.status == 503
                assert calls == [b"one"] and not second.done()
                released.set()
                await first.read()
                response = await asyncio.wait_for(second, 5)
                assert b"data: last" in await response.read()
                assert calls == [b"one", b"two"]
            finally:
                released.set()
                first.close()
                second.cancel()
                await asyncio.gather(second, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_stream_releases_its_permit():
    release = asyncio.Event()
    calls = 0

    async def generate(request):
        nonlocal calls
        calls += 1
        response = web.StreamResponse()
        await response.prepare(request)
        await response.write(b"started\n")
        if calls == 1:
            await release.wait()
        return response

    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/responses", generate)
    async with TestServer(upstream_app, handler_cancellation=True) as upstream:
        async with (
            TestServer(
                create_app(upstream=str(upstream.make_url("/")), max_inflight=1, max_queued=1),
                handler_cancellation=True,
            ) as proxy,
            aiohttp.ClientSession() as client,
        ):
            first = await client.post(proxy.make_url("/v1/responses"))
            await first.content.readline()
            first.close()
            try:
                second = await asyncio.wait_for(client.post(proxy.make_url("/v1/responses")), 5)
                assert second.status == 200 and await second.read() == b"started\n"
                assert calls == 2
            finally:
                release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 429, 500])
async def test_preserves_wire_body_headers_and_error_without_retry(status):
    calls = []
    body = gzip.compress(b'{"messages": [], "temperature": 0.7}')

    async def generate(request):
        calls.append((await request.read(), request.headers.get("Authorization"), request.query_string))
        return web.Response(status=status, body=body, headers={"Content-Encoding": "gzip", "X-Result": "kept"})

    upstream_app = web.Application(handler_args={"auto_decompress": False})
    upstream_app.router.add_post("/v1/chat/completions", generate)
    async with TestServer(upstream_app) as upstream:
        async with (
            TestServer(create_app(upstream=str(upstream.make_url("/")), max_inflight=1, max_queued=0)) as proxy,
            aiohttp.ClientSession(auto_decompress=False) as client,
        ):
            async with client.post(
                proxy.make_url("/v1/chat/completions?tag=a%2Fb"),
                data=body,
                headers={"Authorization": "Bearer test-value", "Content-Encoding": "gzip"},
            ) as result:
                assert result.status == status
                assert result.headers["X-Result"] == "kept"
                assert result.headers["Content-Encoding"] == "gzip"
                assert await result.read() == body
            assert calls == [(body, "Bearer test-value", "tag=a/b")]
