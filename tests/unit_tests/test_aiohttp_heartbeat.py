# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import time

from aiohttp import ClientSession, ClientTimeout

from nemo_gym.aiohttp_heartbeat import CRLF, HeartbeatTCPConnector
from nemo_gym.server_utils import GlobalAIOHTTPAsyncClientConfig


async def _slow_echo_server(stats: dict):
    """Minimal HTTP/1.1 server: skips+counts empty lines before a request-line (RFC 9112 §2.2),
    delays by the JSON 'delay' field, and keeps the connection open for the next request."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        stats["conns"] += 1
        try:
            while True:
                line = await reader.readline()
                while line in (b"\r\n", b"\n"):
                    stats["stray_crlf"] += 1
                    line = await reader.readline()
                if not line:
                    return
                head = line
                while True:
                    nxt = await reader.readline()
                    head += nxt
                    if nxt in (b"\r\n", b""):
                        break
                cl = [h for h in head.split(b"\r\n") if h.lower().startswith(b"content-length:")]
                body = await reader.readexactly(int(cl[0].split(b":")[1])) if cl else b""
                req = json.loads(body or b"{}")
                stats["requests"] += 1
                await asyncio.sleep(req.get("delay", 0))
                payload = json.dumps({"echo": req.get("message"), "n": stats["requests"]}).encode()
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(payload)
                )
                writer.write(payload)
                await writer.drain()
        finally:
            writer.close()

    return await asyncio.start_server(handle, "127.0.0.1", 0)


async def test_heartbeat_keeps_writing_while_request_in_flight_and_connection_stays_reusable():
    stats = {"stray_crlf": 0, "requests": 0, "conns": 0}
    server = await _slow_echo_server(stats)
    port = server.sockets[0].getsockname()[1]
    beats = []
    connector = HeartbeatTCPConnector(heartbeat=0.2, on_heartbeat=beats.append, limit=4, keepalive_timeout=30)
    async with server, ClientSession(connector=connector, timeout=ClientTimeout()) as session:
        t0 = time.monotonic()
        async with session.post(f"http://127.0.0.1:{port}/", json={"message": "slow", "delay": 1.0}) as r1:
            assert await r1.json() == {"echo": "slow", "n": 1}
        assert 1.0 <= time.monotonic() - t0 < 2.5

        # Heartbeats were written while the response was pending, and the server ignored them.
        assert connector.heartbeats_sent >= 3
        assert stats["stray_crlf"] >= 3
        assert beats and all(isinstance(b, int) and b >= 1 for b in beats)

        # Same connection is still good for the next request.
        async with session.post(f"http://127.0.0.1:{port}/", json={"message": "fast", "delay": 0}) as r2:
            assert await r2.json() == {"echo": "fast", "n": 2}
        assert stats["conns"] == 1

        # No request in flight -> no heartbeats.
        sent_before_idle = connector.heartbeats_sent
        await asyncio.sleep(0.5)
        assert connector.heartbeats_sent == sent_before_idle


async def test_heartbeat_disabled_by_default_behaves_like_plain_connector():
    stats = {"stray_crlf": 0, "requests": 0, "conns": 0}
    server = await _slow_echo_server(stats)
    port = server.sockets[0].getsockname()[1]
    connector = HeartbeatTCPConnector(heartbeat=GlobalAIOHTTPAsyncClientConfig().global_aiohttp_crlf_heartbeat_seconds)
    async with server, ClientSession(connector=connector, timeout=ClientTimeout()) as session:
        async with session.post(f"http://127.0.0.1:{port}/", json={"message": "x", "delay": 0.3}) as r:
            assert r.status == 200
    assert connector.heartbeats_sent == 0
    assert stats["stray_crlf"] == 0
    assert connector._hb_task is None


def test_config_default_is_off():
    assert GlobalAIOHTTPAsyncClientConfig().global_aiohttp_crlf_heartbeat_seconds == 0
    assert (
        GlobalAIOHTTPAsyncClientConfig.model_validate(
            {"global_aiohttp_crlf_heartbeat_seconds": 60}
        ).global_aiohttp_crlf_heartbeat_seconds
        == 60
    )


def test_negative_heartbeat_rejected():
    try:
        HeartbeatTCPConnector(heartbeat=-1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_crlf_constant():
    assert CRLF == b"\r\n"
