# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the proxy against local servers without external service access."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from responses_api_agents.opencode_multimodal_agent.network_runner import origin, proxy


@pytest.mark.parametrize("url", ["ftp://example.org", "https://user:secret@example.org", "http:///path"])
def test_reject_invalid_model_origin(url):
    with pytest.raises(ValueError):
        origin(url)


def test_model_origin_keeps_exact_port():
    assert origin("https://EXAMPLE.org/v1") == ("example.org", 443)
    assert origin("http://localhost:8123/v1") == ("localhost", 8123)


@asynccontextmanager
async def serve(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    async with server:
        yield server.sockets[0].getsockname()[1]


@pytest.mark.asyncio
async def test_proxy_denies_other_origin_without_connecting():
    contacted = asyncio.Event()

    async def target(reader, writer):
        contacted.set()
        writer.close()

    async with serve(target) as port:
        async with serve(lambda r, w: proxy(r, w, {("example.org", 443)})) as proxy_port:
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
            writer.write(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())
            await writer.drain()
            response = await asyncio.wait_for(reader.read(), 5)
            writer.close()
            await writer.wait_closed()
    assert response.startswith(b"HTTP/1.1 403")
    assert not contacted.is_set()


@pytest.mark.asyncio
async def test_http_proxy_rewrites_host_and_removes_proxy_credentials():
    received = asyncio.Future()

    async def target(reader, writer):
        received.set_result(await reader.readuntil(b"\r\n\r\n"))
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")
        await writer.drain()
        writer.close()

    async with serve(target) as port:
        async with serve(lambda r, w: proxy(r, w, {("127.0.0.1", port)})) as proxy_port:
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
            writer.write(
                f"GET http://127.0.0.1:{port}/v1?test=1 HTTP/1.1\r\n"
                "Host: forbidden.example\r\nProxy-Authorization: test-placeholder\r\n\r\n".encode()
            )
            await writer.drain()
            response = await asyncio.wait_for(reader.read(), 5)
            writer.close()
            await writer.wait_closed()
    request = received.result()
    assert request.startswith(b"GET /v1?test=1 HTTP/1.1\r\n")
    assert f"Host: 127.0.0.1:{port}\r\n".encode() in request
    assert b"forbidden.example" not in request and b"Proxy-Authorization" not in request
    assert response.endswith(b"OK")


@pytest.mark.asyncio
async def test_allowed_connect_tunnels_bidirectional_bytes():
    async def target(reader, writer):
        data = await reader.readexactly(4)
        writer.write(data.upper())
        await writer.drain()
        writer.close()

    async with serve(target) as port:
        async with serve(lambda r, w: proxy(r, w, {("127.0.0.1", port)})) as proxy_port:
            reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
            writer.write(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())
            await writer.drain()
            assert (await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)).startswith(b"HTTP/1.1 200")
            writer.write(b"ping")
            await writer.drain()
            assert await asyncio.wait_for(reader.readexactly(4), 5) == b"PING"
            writer.close()
            await writer.wait_closed()
