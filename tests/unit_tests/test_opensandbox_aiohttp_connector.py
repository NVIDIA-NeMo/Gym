# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regression tests for the uvloop descriptor-poisoning fix in the OpenSandbox client path.

The misuse pattern: a connect through ``aiohttp.TCPConnector`` is cancelled while
``loop.create_connection(sock=...)`` waits for ``connection_made``. Under uvloop that
closes the descriptor through libuv but leaves the Python socket holding its number;
closing that socket later kills whichever connection reused the number, after which
every new connect that lands on it fails with ``File descriptor N is used by transport``.
"""

from __future__ import annotations

import asyncio
import errno
import os
import socket
from typing import Any

import pytest


aiohttp = pytest.importorskip("aiohttp")
uvloop = pytest.importorskip("uvloop", reason="the failure mode is specific to uvloop")

from nemo_gym.sandbox.providers.opensandbox.aiohttp_connector import (  # noqa: E402
    FdSafeTCPConnector,
    loop_detaches_sockets_late,
)


def _fd_is_open(fd: int) -> bool:
    try:
        os.fstat(fd)
    except OSError as e:
        assert e.errno == errno.EBADF
        return False
    return True


async def _silent_server() -> tuple[asyncio.AbstractServer, int]:
    """A TCP server that accepts and never answers, so requests only end by cancellation."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.read()
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _cancel_during_create_connection(connector: aiohttp.TCPConnector, port: int) -> None:
    """Send one request through ``connector`` and cancel it exactly while
    ``loop.create_connection(sock=...)`` awaits ``connection_made``.

    uvloop calls the protocol factory before ``tr._open`` and schedules
    ``connection_made`` with call_soon afterwards, so a cancel scheduled from
    the factory runs first and lands on the ``await waiter``."""
    loop = asyncio.get_running_loop()
    real_factory = connector._factory

    def factory() -> Any:
        task = asyncio.current_task()
        assert task is not None
        loop.call_soon(task.cancel)
        return real_factory()

    connector._factory = factory
    async with aiohttp.ClientSession(connector=connector) as session:
        request = asyncio.ensure_future(session.get(f"http://127.0.0.1:{port}/"))
        with pytest.raises(asyncio.CancelledError):
            await request
        await asyncio.sleep(0)


def _recording_factory(created: list[socket.socket]):
    def make(addr_info: Any) -> socket.socket:
        family, type_, proto, _, _ = addr_info
        sock = socket.socket(family=family, type=type_, proto=proto)
        created.append(sock)
        return sock

    return make


def test_loop_detection() -> None:
    async def probe() -> bool:
        return loop_detaches_sockets_late()

    assert uvloop.run(probe()) is True
    assert asyncio.run(probe()) is False
    assert loop_detaches_sockets_late() is False  # no running loop


def test_stock_connector_leaves_fd_double_owned_under_uvloop() -> None:
    """Documents the root cause with aiohttp's own connector: after the cancel the
    socket still reports its descriptor although libuv already closed it, and
    closing the socket destroys the next connection that reused the number."""

    async def main() -> None:
        server, port = await _silent_server()
        created: list[socket.socket] = []
        connector = aiohttp.TCPConnector(socket_factory=_recording_factory(created))
        try:
            await _cancel_during_create_connection(connector, port)
            assert len(created) == 1
            (orphan,) = created
            stale_fd = orphan.fileno()
            assert stale_fd >= 0, "the Python socket was not detached"
            assert not _fd_is_open(stale_fd), "libuv should already have closed the descriptor"

            # The kernel hands the lowest free number to the next socket ...
            victim = socket.socket()
            try:
                assert victim.fileno() == stale_fd, "expected descriptor reuse"
                # ... and the dropped socket's close() takes it away from the victim.
                orphan.close()
                assert not _fd_is_open(victim.fileno())
            finally:
                victim.detach()
        finally:
            server.close()
            await server.wait_closed()

    uvloop.run(main())


def test_fd_safe_connector_detaches_cancelled_connect_under_uvloop() -> None:
    """With the fix the orphan is detached, so nothing can close a reused descriptor."""

    async def main() -> None:
        server, port = await _silent_server()
        created: list[socket.socket] = []
        connector = FdSafeTCPConnector(socket_factory=_recording_factory(created))
        try:
            await _cancel_during_create_connection(connector, port)
            assert len(created) == 1
            (orphan,) = created
            assert orphan.fileno() == -1, "the cancelled connect's socket must be detached"
            victim = socket.socket()
            try:
                orphan.close()  # no-op on a detached socket
                assert _fd_is_open(victim.fileno())
            finally:
                victim.close()
        finally:
            server.close()
            await server.wait_closed()

    uvloop.run(main())


def test_fd_safe_connector_keeps_socket_on_stdlib_loop() -> None:
    """The selector loop's transport closes the socket object itself; detaching
    there would leak the descriptor, so the connector must not touch it."""

    async def main() -> None:
        server, port = await _silent_server()
        created: list[socket.socket] = []
        connector = FdSafeTCPConnector(socket_factory=_recording_factory(created))
        try:
            await _cancel_during_create_connection(connector, port)
            assert len(created) == 1
            (sock,) = created
            # Either already closed by the stdlib transport or still owned by it; never detached.
            assert sock.fileno() == -1 or _fd_is_open(sock.fileno())
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(main())


def test_fd_safe_connector_completes_normal_requests() -> None:
    async def main() -> None:
        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        created: list[socket.socket] = []
        try:
            async with aiohttp.ClientSession(
                connector=FdSafeTCPConnector(socket_factory=_recording_factory(created))
            ) as session:
                async with session.get(f"http://127.0.0.1:{port}/") as resp:
                    assert resp.status == 200
                    assert await resp.read() == b"ok"
            assert len(created) == 1
        finally:
            server.close()
            await server.wait_closed()

    uvloop.run(main())


@pytest.mark.asyncio
async def test_provider_pty_client_uses_fd_safe_connector() -> None:
    from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider

    for tls_verify, expected_ssl in ((False, False), (True, True)):
        session = opensandbox_provider.OpenSandboxProvider(connection={"tls_verify": tls_verify})._pty_http_client()
        try:
            assert isinstance(session.connector, FdSafeTCPConnector)
            assert session.connector._ssl is expected_ssl
        finally:
            await session.close()
