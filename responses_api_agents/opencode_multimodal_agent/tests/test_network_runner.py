# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the proxy against local servers without external service access."""

import asyncio
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

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


@pytest.mark.skipif(sys.platform != "linux", reason="observes Linux process state")
@pytest.mark.asyncio
async def test_cancelled_wrapper_stops_descendants(tmp_path):
    from responses_api_agents.opencode_multimodal_agent.network_runner import run_child

    pidfile = tmp_path / "descendant.pid"
    code = (
        "import subprocess,time;from pathlib import Path;"
        "p=subprocess.Popen(['sleep','30']);"
        f"Path({str(pidfile)!r}).write_text(str(p.pid));time.sleep(30)"
    )
    task = asyncio.create_task(run_child([sys.executable, "-c", code]))
    child = None
    try:
        for _ in range(100):
            if pidfile.is_file():
                child = int(pidfile.read_text())
                break
            await asyncio.sleep(0.02)
        assert child is not None, "probe child did not start"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        for _ in range(100):
            try:
                state = Path(f"/proc/{child}/stat").read_text().rsplit(")", 1)[1].split()[0]
                if state == "Z":
                    break
            except FileNotFoundError:
                break
            await asyncio.sleep(0.02)
        else:
            pytest.fail("cancelled wrapper left its descendant running")
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if child is not None:
            try:
                os.kill(child, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux user and network namespaces")
@pytest.mark.asyncio
async def test_real_network_wrapper_blocks_direct_host_connection():
    async def target(reader, writer):
        writer.close()
        await writer.wait_closed()

    async with serve(target) as port:
        # Prove the target is reachable outside the wrapper before testing denial.
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.close()
        await writer.wait_closed()
        code = (
            "import socket;"
            f"s=socket.socket();s.settimeout(1);result=s.connect_ex(('127.0.0.1',{port}));"
            "print(result);raise SystemExit(0 if result else 7)"
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).parents[1] / "network_runner.py"),
            "--mode",
            "model_only",
            "--model-url",
            f"http://127.0.0.1:{port}/v1",
            "--",
            sys.executable,
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
    if process.returncode and b"unshare failed: Operation not permitted" in stderr:
        pytest.skip("host disables unprivileged user/network namespaces")
    assert process.returncode == 0, stderr.decode(errors="replace")
    assert int(stdout.strip()) != 0
