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
import json
import struct
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxPtyError, SandboxPtySpec
from nemo_gym.sandbox.providers.opensandbox.pty import (
    OpenSandboxPtySession,
    _effective_command,
    open_pty_session,
)


pytestmark = pytest.mark.sandbox


def _text(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(payload))


def _binary(data: bytes) -> SimpleNamespace:
    return SimpleNamespace(type=aiohttp.WSMsgType.BINARY, data=data)


CONNECTED = _text({"type": "connected", "session_id": "s-1", "mode": "pty"})


class FakeWs:
    """Scripted WebSocket: yields queued messages, then parks until closed."""

    def __init__(self, messages: list[SimpleNamespace], close_code: int | None = 1000) -> None:
        self._messages = list(messages)
        self._drained = asyncio.Event()
        self.sent: list[bytes | str] = []
        self.closed = False
        self.close_code = close_code

    def __aiter__(self) -> "FakeWs":
        return self

    async def __anext__(self) -> SimpleNamespace:
        if self._messages:
            return self._messages.pop(0)
        self._drained.set()
        while not self.closed:
            await asyncio.sleep(0.001)
        raise StopAsyncIteration

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True


class FakeResponse:
    def __init__(self, status: int, payload: dict[str, Any] | None = None) -> None:
        self.status = status
        self._payload = payload or {}

    async def json(self) -> dict[str, Any]:
        return self._payload

    async def text(self) -> str:
        return json.dumps(self._payload)

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def __await__(self) -> Any:
        # aiohttp request methods are both awaitable and async context managers.
        async def _resolve() -> "FakeResponse":
            return self

        return _resolve().__await__()


class FakeHttpClient:
    def __init__(self, ws: FakeWs | None = None, post_status: int = 201, ws_error: Exception | None = None) -> None:
        self._ws = ws
        self._post_status = post_status
        self._ws_error = ws_error
        self.post_calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []
        self.delete_calls: list[tuple[str, dict[str, str]]] = []
        self.ws_calls: list[tuple[str, dict[str, str]]] = []
        self.closed = False

    def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str], timeout: Any = None) -> FakeResponse:
        self.post_calls.append((url, json, headers))
        return FakeResponse(self._post_status, {"session_id": "s-1"})

    def delete(self, url: str, *, headers: dict[str, str], timeout: Any = None) -> FakeResponse:
        self.delete_calls.append((url, headers))
        return FakeResponse(200)

    async def ws_connect(self, url: str, *, headers: dict[str, str]) -> FakeWs:
        self.ws_calls.append((url, headers))
        if self._ws_error is not None:
            raise self._ws_error
        assert self._ws is not None
        return self._ws

    async def close(self) -> None:
        self.closed = True


async def _session_over(
    messages: list[SimpleNamespace], *, close_code: int | None = 1000
) -> tuple[OpenSandboxPtySession, FakeWs, FakeHttpClient]:
    ws = FakeWs(messages, close_code=close_code)
    client = FakeHttpClient(ws=ws)
    session = OpenSandboxPtySession(
        client=client,  # type: ignore[arg-type]
        ws=ws,  # type: ignore[arg-type]
        session_id="s-1",
        session_url="http://server/v1/sandboxes/sb-1/proxy/44772/pty/s-1",
        headers={"OPEN-SANDBOX-API-KEY": "k"},
        request_timeout_s=5.0,
    )
    return session, ws, client


async def test_frame_decode_and_exit() -> None:
    replay = b"\x03" + struct.pack(">Q", 0) + b"replay"
    session, ws, _ = await _session_over(
        [
            CONNECTED,
            _binary(b"\x01hello"),
            _binary(replay),
            _binary(b"\x02err"),
            _text({"type": "exit", "exit_code": 0}),
        ]
    )
    assert await session.read() == b"hello"
    assert await session.read() == b"replay"
    assert await session.read_stderr() == b"err"
    ws.closed = True
    assert await session.read() == b""
    assert await session.read() == b""
    assert await session.read_stderr() == b""
    assert await session.wait_exit() == 0
    assert session.mode == "pty"
    await session.close()


async def test_replay_may_precede_connected() -> None:
    # Live-observed proxy behavior: replay frames can arrive before the
    # JSON connected frame.
    replay = b"\x03" + struct.pack(">Q", 0) + b"early"
    session, ws, _ = await _session_over([_binary(replay), CONNECTED])
    await session._wait_connected(1.0)
    assert await session.read() == b"early"
    await session.close()


async def test_frame_encode() -> None:
    session, ws, _ = await _session_over([CONNECTED])
    await session.write(b"ls\n")
    await session.resize(40, 120)
    await session.send_signal("SIGINT")
    assert ws.sent[0] == b"\x00ls\n"
    assert json.loads(ws.sent[1]) == {"type": "resize", "cols": 120, "rows": 40}
    assert json.loads(ws.sent[2]) == {"type": "signal", "signal": "SIGINT"}
    await session.close()


@pytest.mark.parametrize(
    ("close_code", "match"),
    [(4001, "taken over"), (1008, "already has an attached client"), (1006, "close code 1006")],
)
async def test_abnormal_close_raises(close_code: int, match: str) -> None:
    session, ws, _ = await _session_over([CONNECTED], close_code=close_code)
    ws.closed = True
    with pytest.raises(SandboxPtyError, match=match):
        await session.read()
    with pytest.raises(SandboxPtyError, match=match):
        await session.wait_exit()
    await session.close()


async def test_error_frame_is_fatal() -> None:
    session, ws, _ = await _session_over(
        [CONNECTED, _text({"type": "error", "code": "STDIN_WRITE_FAILED", "error": "boom"})]
    )
    ws.closed = True
    with pytest.raises(SandboxPtyError, match="STDIN_WRITE_FAILED"):
        await session.read()
    await session.close()


async def test_read_and_wait_exit_timeouts() -> None:
    session, ws, _ = await _session_over([CONNECTED])
    with pytest.raises(TimeoutError):
        await session.read(timeout_s=0.01)
    with pytest.raises(TimeoutError):
        await session.wait_exit(timeout_s=0.01)
    # The shared exit future must survive a timed-out waiter (shield).
    assert not session._exit.cancelled()
    await session.close()


async def test_close_is_idempotent_and_tears_down() -> None:
    session, ws, client = await _session_over([CONNECTED])
    await session.close()
    await session.close()
    assert ws.closed
    assert client.closed
    assert client.delete_calls == [
        ("http://server/v1/sandboxes/sb-1/proxy/44772/pty/s-1", {"OPEN-SANDBOX-API-KEY": "k"})
    ]
    with pytest.raises(SandboxPtyError, match="closed"):
        await session.write(b"x")
    with pytest.raises(SandboxPtyError, match="closed before process exit"):
        await session.wait_exit()


async def test_aiter_yields_until_eof() -> None:
    session, ws, _ = await _session_over(
        [CONNECTED, _binary(b"\x01a"), _binary(b"\x01b"), _text({"type": "exit", "exit_code": 3})]
    )
    ws.closed = True
    chunks = [chunk async for chunk in session]
    assert chunks == [b"a", b"b"]
    assert await session.wait_exit() == 3
    await session.close()


async def test_open_pty_session_wiring_and_resize() -> None:
    ws = FakeWs([CONNECTED])
    client = FakeHttpClient(ws=ws)
    spec = SandboxPtySpec(cwd="/tmp", rows=50, cols=200)
    session = await open_pty_session(
        client=client,  # type: ignore[arg-type]
        base_url="http://server/v1/sandboxes/sb-1/proxy/44772",
        headers={"OPEN-SANDBOX-API-KEY": "k", "X-EXECD-ACCESS-TOKEN": "tok"},
        spec=spec,
        request_timeout_s=5.0,
    )
    url, body, headers = client.post_calls[0]
    assert url == "http://server/v1/sandboxes/sb-1/proxy/44772/pty"
    assert body == {"cwd": "/tmp"}
    assert headers["X-EXECD-ACCESS-TOKEN"] == "tok"
    ws_url, ws_headers = client.ws_calls[0]
    assert ws_url == "ws://server/v1/sandboxes/sb-1/proxy/44772/pty/s-1/ws"
    assert ws_headers == headers
    assert json.loads(ws.sent[0]) == {"type": "resize", "cols": 200, "rows": 50}
    await session.close()


async def test_open_pty_session_https_becomes_wss_and_default_size_skips_resize() -> None:
    ws = FakeWs([CONNECTED])
    client = FakeHttpClient(ws=ws)
    session = await open_pty_session(
        client=client,  # type: ignore[arg-type]
        base_url="https://server/v1/sandboxes/sb-1/proxy/44772",
        headers={},
        spec=SandboxPtySpec(),
        request_timeout_s=5.0,
    )
    assert client.ws_calls[0][0].startswith("wss://")
    assert client.post_calls[0][1] == {}
    assert ws.sent == []
    await session.close()


@pytest.mark.parametrize(
    ("post_status", "match"),
    [(404, "execd >= 1.0.10"), (500, "HTTP 500")],
)
async def test_open_pty_session_create_failure(post_status: int, match: str) -> None:
    client = FakeHttpClient(post_status=post_status)
    with pytest.raises(SandboxPtyError, match=match):
        await open_pty_session(
            client=client,  # type: ignore[arg-type]
            base_url="http://server/base",
            headers={},
            spec=SandboxPtySpec(),
            request_timeout_s=5.0,
        )
    assert client.closed


async def test_open_pty_session_ws_failure_deletes_session() -> None:
    client = FakeHttpClient(ws_error=RuntimeError("upgrade refused"))
    with pytest.raises(SandboxPtyError, match="upgrade refused"):
        await open_pty_session(
            client=client,  # type: ignore[arg-type]
            base_url="http://server/base",
            headers={},
            spec=SandboxPtySpec(),
            request_timeout_s=5.0,
        )
    assert client.delete_calls[0][0] == "http://server/base/pty/s-1"
    assert client.closed


def test_effective_command_rewrites() -> None:
    assert _effective_command(SandboxPtySpec()) is None
    assert _effective_command(SandboxPtySpec(command="htop")) == "htop"
    env_only = _effective_command(SandboxPtySpec(env={"A": "b c"}))
    assert env_only == "env A='b c' sh -c 'exec \"$(command -v bash || echo sh)\"'"
    assert _effective_command(SandboxPtySpec(command="id", env={"A": "1"})) == "env A=1 sh -c id"
    assert _effective_command(SandboxPtySpec(user="worker")) == "su -s /bin/sh worker"
    assert _effective_command(SandboxPtySpec(command="id", user="worker")) == "su -s /bin/sh -c id worker"
    assert _effective_command(SandboxPtySpec(user="root")) is None
    with pytest.raises(ValueError, match="user name"):
        _effective_command(SandboxPtySpec(user=0))


async def test_provider_create_pty_resolves_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tenacity", reason="tenacity optional sandbox dependency is not installed")
    pytest.importorskip("opensandbox", reason="opensandbox SDK is not installed")
    from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider

    class FakeRaw:
        async def get_endpoint(self, port: int) -> SimpleNamespace:
            assert port == 44772
            return SimpleNamespace(
                endpoint="server/v1/sandboxes/sb-1/proxy/44772",
                headers={"X-EXECD-ACCESS-TOKEN": "tok"},
            )

    provider = OpenSandboxProvider(connection={"domain": "server", "api_key": "k", "protocol": "http"})
    ws = FakeWs([CONNECTED])
    client = FakeHttpClient(ws=ws)
    monkeypatch.setattr(provider, "_pty_http_client", lambda: client)

    handle = SandboxHandle(sandbox_id="sb-1", provider_name="opensandbox", raw=FakeRaw())
    session = await provider.create_pty(handle, SandboxPtySpec(cwd="/w"))
    url, body, headers = client.post_calls[0]
    assert url == "http://server/v1/sandboxes/sb-1/proxy/44772/pty"
    assert body == {"cwd": "/w"}
    assert headers == {"X-EXECD-ACCESS-TOKEN": "tok", "OPEN-SANDBOX-API-KEY": "k"}
    await session.close()


async def test_open_pty_session_invalid_spec_closes_client() -> None:
    client = FakeHttpClient()
    with pytest.raises(ValueError, match="user name"):
        await open_pty_session(
            client=client,  # type: ignore[arg-type]
            base_url="http://server/base",
            headers={},
            spec=SandboxPtySpec(user=0),
            request_timeout_s=5.0,
        )
    assert client.closed
    assert client.post_calls == []


async def test_pipe_mode_url_and_no_initial_resize() -> None:
    ws = FakeWs([_text({"type": "connected", "session_id": "s-1", "mode": "pipe"})])
    client = FakeHttpClient(ws=ws)
    session = await open_pty_session(
        client=client,  # type: ignore[arg-type]
        base_url="http://server/base",
        headers={},
        spec=SandboxPtySpec(command="make", rows=50, cols=200, pty=False),
        request_timeout_s=5.0,
    )
    assert client.ws_calls[0][0] == "ws://server/base/pty/s-1/ws?pty=0"
    assert ws.sent == []
    assert session.mode == "pipe"
    await session.close()


async def test_pipe_mode_splits_streams() -> None:
    session, ws, _ = await _session_over(
        [
            _text({"type": "connected", "session_id": "s-1", "mode": "pipe"}),
            _binary(b"\x01out"),
            _binary(b"\x02err"),
            _text({"type": "exit", "exit_code": 5}),
        ]
    )
    assert await session.read() == b"out"
    assert await session.read_stderr() == b"err"
    ws.closed = True
    assert await session.read() == b""
    assert await session.read_stderr() == b""
    assert await session.wait_exit() == 5
    await session.close()


async def test_facade_passes_pipe_mode() -> None:
    from nemo_gym.sandbox import AsyncSandbox as _AS
    from nemo_gym.sandbox.providers.base import SandboxSpec as _Spec

    class Recorder:
        name = "rec"

        def __init__(self) -> None:
            self.specs: list[SandboxPtySpec] = []

        async def create(self, spec: Any) -> SandboxHandle:
            return SandboxHandle(sandbox_id="s", provider_name="rec", raw=None)

        async def create_pty(self, handle: SandboxHandle, spec: SandboxPtySpec) -> object:
            self.specs.append(spec)
            return object()

        async def exec(self, *a: Any, **k: Any) -> None: ...
        async def upload_file(self, *a: Any) -> None: ...
        async def download_file(self, *a: Any) -> None: ...
        async def status(self, *a: Any) -> None: ...
        async def close(self, *a: Any) -> None: ...
        async def aclose(self) -> None: ...

    provider = Recorder()
    sandbox = _AS(provider)
    await sandbox.start(_Spec(image="i"))
    await sandbox.pty.create("make", pty=False)
    assert provider.specs[0].pty is False
    await sandbox.stop()


@pytest.mark.parametrize(
    ("takeover", "since", "expected_query"),
    [
        (True, None, "?takeover=1"),
        (False, None, ""),
        (True, 0, "?takeover=1&since=0"),
        (False, 4096, "?since=4096"),
    ],
)
async def test_attach_pty_session_query(takeover: bool, since: int | None, expected_query: str) -> None:
    from nemo_gym.sandbox.providers.opensandbox.pty import attach_pty_session

    ws = FakeWs([CONNECTED])
    client = FakeHttpClient(ws=ws)
    session = await attach_pty_session(
        client=client,  # type: ignore[arg-type]
        base_url="http://server/base",
        headers={"OPEN-SANDBOX-API-KEY": "k"},
        session_id="s-1",
        takeover=takeover,
        since=since,
        request_timeout_s=5.0,
    )
    assert client.ws_calls[0][0] == f"ws://server/base/pty/s-1/ws{expected_query}"
    assert client.post_calls == [], "attach must not create a new session"
    assert session.session_id == "s-1"
    await session.close()


async def test_attach_pty_session_failure_closes_client() -> None:
    from nemo_gym.sandbox.providers.opensandbox.pty import attach_pty_session

    client = FakeHttpClient(ws_error=RuntimeError("gone"))
    with pytest.raises(SandboxPtyError, match="Failed to attach to PTY session s-9"):
        await attach_pty_session(
            client=client,  # type: ignore[arg-type]
            base_url="http://server/base",
            headers={},
            session_id="s-9",
            request_timeout_s=5.0,
        )
    assert client.closed
    assert client.delete_calls == [], "attach must not delete a session it did not create"


async def test_evicted_session_reports_takeover() -> None:
    session, ws, _ = await _session_over([CONNECTED], close_code=4001)
    ws.closed = True
    with pytest.raises(SandboxPtyError, match="taken over"):
        await session.read()
    await session.close()


async def test_provider_attach_pty_reuses_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tenacity", reason="tenacity optional sandbox dependency is not installed")
    pytest.importorskip("opensandbox", reason="opensandbox SDK is not installed")
    from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider

    class FakeRaw:
        async def get_endpoint(self, port: int) -> SimpleNamespace:
            return SimpleNamespace(endpoint="server/v1/sandboxes/sb-1/proxy/44772", headers={})

    provider = OpenSandboxProvider(connection={"domain": "server", "api_key": "k", "protocol": "https"})
    client = FakeHttpClient(ws=FakeWs([CONNECTED]))
    monkeypatch.setattr(provider, "_pty_http_client", lambda: client)
    handle = SandboxHandle(sandbox_id="sb-1", provider_name="opensandbox", raw=FakeRaw())
    session = await provider.attach_pty(handle, "s-7", takeover=True, since=10)
    assert client.ws_calls[0][0] == "wss://server/v1/sandboxes/sb-1/proxy/44772/pty/s-7/ws?takeover=1&since=10"
    await session.close()


async def test_attach_rejected_before_connected_raises_and_cleans_up() -> None:
    # A second attach without takeover is closed 1008 before any connected frame.
    ws = FakeWs([], close_code=1008)
    ws.closed = True
    client = FakeHttpClient(ws=ws)
    with pytest.raises(SandboxPtyError, match="already has an attached client"):
        await open_pty_session(
            client=client,  # type: ignore[arg-type]
            base_url="http://server/base",
            headers={},
            spec=SandboxPtySpec(),
            request_timeout_s=5.0,
        )
    assert client.closed
    assert client.delete_calls[0][0] == "http://server/base/pty/s-1"


async def test_connected_timeout_closes_session_and_client() -> None:
    # Server accepts the socket but never sends `connected`.
    ws = FakeWs([])
    client = FakeHttpClient(ws=ws)
    with pytest.raises(TimeoutError):
        await open_pty_session(
            client=client,  # type: ignore[arg-type]
            base_url="http://server/base",
            headers={},
            spec=SandboxPtySpec(),
            request_timeout_s=0.05,
        )
    assert client.closed
    assert ws.closed
    assert client.delete_calls[0][0] == "http://server/base/pty/s-1"


async def test_empty_and_short_frames_do_not_signal_eof() -> None:
    replay_no_payload = b"\x03" + struct.pack(">Q", 0)
    session, ws, _ = await _session_over(
        [
            CONNECTED,
            _binary(b""),  # empty frame
            _binary(b"\x01"),  # bare channel byte, no payload
            _binary(replay_no_payload),  # replay header with no payload
            _binary(b"\x02"),  # bare stderr channel
            _binary(b"\x07nope"),  # unknown channel
            _binary(b"\x01real"),
            _text({"type": "pong"}),
            _text({"type": "exit", "exit_code": 0}),
        ]
    )
    ws.closed = True
    assert [chunk async for chunk in session] == [b"real"], "empty payloads must not end iteration"
    assert await session.wait_exit() == 0
    await session.close()


async def test_malformed_text_frame_surfaces_as_error() -> None:
    session, ws, _ = await _session_over([CONNECTED, SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="{not json")])
    ws.closed = True
    with pytest.raises(SandboxPtyError, match="malformed frame"):
        await session.read()
    await session.close()


async def test_send_after_close_raises_for_every_sender() -> None:
    session, ws, _ = await _session_over([CONNECTED])
    await session.close()
    for send in (
        lambda: session.write(b"x"),
        lambda: session.resize(10, 10),
        lambda: session.send_signal("SIGINT"),
    ):
        with pytest.raises(SandboxPtyError, match="closed"):
            await send()


async def test_send_failure_becomes_sandbox_pty_error() -> None:
    session, ws, _ = await _session_over([CONNECTED])

    async def boom(_: Any) -> None:
        raise ConnectionResetError("peer gone")

    ws.send_bytes = boom  # type: ignore[assignment]
    with pytest.raises(SandboxPtyError, match="connection lost"):
        await session.write(b"x")
    await session.close()


async def test_provider_aclose_closes_live_pty_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("tenacity", reason="tenacity optional sandbox dependency is not installed")
    pytest.importorskip("opensandbox", reason="opensandbox SDK is not installed")
    from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider

    class FakeRaw:
        async def get_endpoint(self, port: int) -> SimpleNamespace:
            return SimpleNamespace(endpoint="server/base", headers={})

    provider = OpenSandboxProvider(connection={"domain": "server", "protocol": "http"})
    ws = FakeWs([CONNECTED])
    client = FakeHttpClient(ws=ws)
    monkeypatch.setattr(provider, "_pty_http_client", lambda: client)
    handle = SandboxHandle(sandbox_id="sb-1", provider_name="opensandbox", raw=FakeRaw())
    session = await provider.create_pty(handle, SandboxPtySpec())
    await provider.aclose()
    assert client.closed, "aclose must close PTY-owned aiohttp clients"
    assert ws.closed
    await session.close()
