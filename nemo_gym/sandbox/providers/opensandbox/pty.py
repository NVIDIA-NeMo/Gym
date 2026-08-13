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

"""execd PTY sessions over the OpenSandbox server proxy.

Speaks execd's documented PTY wire protocol directly: the
released OpenSandbox SDKs expose no PTY API. Sessions live at
``{base}/pty[/{session_id}[/ws]]`` where ``base`` is the sandbox's execd
endpoint as resolved by the SDK (through the server proxy when
``use_server_proxy`` is set).

execd sends the exit frame and closes the socket concurrently with its output
pumps, so a clean EOF means the socket drained, not that every byte the process
wrote was delivered. Callers that need the tail should have the command emit a
sentinel and read until it appears.
"""

import asyncio
import json
import shlex
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode

import aiohttp

from nemo_gym.sandbox.providers.base import SandboxPtyError, SandboxPtySpec


CHAN_STDIN = 0
CHAN_STDOUT = 1
CHAN_STDERR = 2
CHAN_REPLAY = 3
# 0x03 replay frames carry a big-endian uint64 absolute byte offset after the
# channel byte.
REPLAY_HEADER_BYTES = 9
WS_CLOSE_TAKEN_OVER = 4001
WS_CLOSE_POLICY_VIOLATION = 1008

# Mirrors execd's shell pick (bash when available, else sh) for env-only specs.
_DEFAULT_SHELL_SNIPPET = 'exec "$(command -v bash || echo sh)"'
# execd drops unrecognized signal names without reporting an error.
SUPPORTED_SIGNALS = frozenset({"SIGINT", "SIGTERM", "SIGKILL", "SIGQUIT", "SIGHUP"})


def _effective_command(spec: SandboxPtySpec) -> str | None:
    """Rewrite env/user into the command: execd's PTY create accepts only
    ``cwd`` and ``command``."""
    if isinstance(spec.user, bool) or isinstance(spec.user, int):
        raise ValueError("OpenSandbox PTY sessions require a user name, not a uid")

    command = spec.command
    if spec.env:
        assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in spec.env.items())
        command = f"env {assignments} sh -c {shlex.quote(command or _DEFAULT_SHELL_SNIPPET)}"
    if spec.user is not None and spec.user != "root":
        if command is None:
            command = f"su -s /bin/sh {shlex.quote(spec.user)}"
        else:
            command = f"su -s /bin/sh -c {shlex.quote(command)} {shlex.quote(spec.user)}"
    return command


class OpenSandboxPtySession:
    """One live PTY WebSocket. Created via :func:`open_pty_session`."""

    def __init__(
        self,
        *,
        client: aiohttp.ClientSession,
        ws: aiohttp.ClientWebSocketResponse,
        session_id: str,
        session_url: str,
        headers: dict[str, str],
        request_timeout_s: float | None,
        owned: bool = True,
    ) -> None:
        self._client = client
        self._ws = ws
        self.session_id = session_id
        self._session_url = session_url
        self._headers = headers
        self._request_timeout_s = request_timeout_s
        # Attached sessions belong to whoever created them: closing one detaches
        # rather than ending it.
        self._owned = owned
        self.mode: str | None = None
        self.replay_offset: int | None = None
        self._output: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._stderr: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._exit: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self._connected: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._error: SandboxPtyError | None = None
        self._closed = False
        self._pump_task = asyncio.create_task(self._pump())

    @property
    def closed(self) -> bool:
        """True once the session can no longer run commands: after ``close()``,
        or once the connection pump has ended (process exit, takeover eviction,
        or connection loss). Resources are released by ``close()``."""
        return self._closed or self._pump_task.done()

    async def _pump(self) -> None:
        """Sole reader of the WebSocket; fans frames out to queue/future/event."""
        try:
            async for message in self._ws:
                if message.type == aiohttp.WSMsgType.BINARY:
                    data = message.data
                    if not data:
                        continue
                    channel = data[0]
                    # Empty payloads are dropped: b"" is the drained-EOF signal.
                    if channel == CHAN_STDOUT and len(data) > 1:
                        await self._output.put(data[1:])
                    elif channel == CHAN_STDERR and len(data) > 1:
                        await self._stderr.put(data[1:])
                    elif channel == CHAN_REPLAY and len(data) > REPLAY_HEADER_BYTES:
                        # Replay is one merged stream regardless of mode. The
                        # server clamps a `since` older than its 1 MiB buffer,
                        # so a higher offset here means output was evicted.
                        self.replay_offset = int.from_bytes(data[1:REPLAY_HEADER_BYTES], "big")
                        await self._output.put(data[REPLAY_HEADER_BYTES:])
                elif message.type == aiohttp.WSMsgType.TEXT:
                    try:
                        frame = json.loads(message.data)
                        frame_type = frame.get("type")
                        if frame_type == "connected":
                            self.mode = frame.get("mode")
                            if not self._connected.done():
                                self._connected.set_result(None)
                        elif frame_type == "exit":
                            if not self._exit.done():
                                self._exit.set_result(int(frame.get("exit_code", -1)))
                        elif frame_type == "error":
                            self._error = SandboxPtyError(
                                f"PTY session failed: {frame.get('code')}: {frame.get('error')}"
                            )
                    except (ValueError, TypeError, AttributeError) as e:
                        if self._error is None:
                            self._error = SandboxPtyError(f"PTY session sent a malformed frame: {e!r}")
                        break
                elif message.type == aiohttp.WSMsgType.ERROR:
                    break
        finally:
            if not self._exit.done():
                self._exit.set_exception(self._close_error())
                self._exit.exception()  # retrieved; silences never-retrieved warnings
            # A pump that ends before `connected` arrived means the session
            # never became usable; fail the waiter rather than letting it
            # proceed with mode=None.
            if not self._connected.done():
                self._connected.set_exception(self._close_error())
                self._connected.exception()  # retrieved; silences never-retrieved warnings
            await self._output.put(None)
            await self._stderr.put(None)

    def _close_error(self) -> SandboxPtyError:
        if self._error is not None:
            return self._error
        code = self._ws.close_code
        if code == WS_CLOSE_TAKEN_OVER:
            return SandboxPtyError("PTY session was taken over by another client")
        if code == WS_CLOSE_POLICY_VIOLATION:
            return SandboxPtyError("PTY session already has an attached client")
        return SandboxPtyError(f"PTY connection closed unexpectedly (close code {code})")

    async def _wait_connected(self, timeout_s: float | None) -> None:
        # shield: the future is shared; a timed-out waiter must not cancel it.
        await asyncio.wait_for(asyncio.shield(self._connected), timeout=timeout_s)

    async def _read_stream(self, queue: asyncio.Queue[bytes | None], timeout_s: float | None) -> bytes:
        chunk = await asyncio.wait_for(queue.get(), timeout=timeout_s)
        if chunk is None:
            # Keep the EOF observable by subsequent reads and iterators.
            await queue.put(None)
            if self._exit.done() and self._exit.exception() is None:
                return b""
            raise self._exit.exception() if self._exit.done() else self._close_error()
        return chunk

    async def read(self, *, timeout_s: float | None = None) -> bytes:
        return await self._read_stream(self._output, timeout_s)

    async def read_stderr(self, *, timeout_s: float | None = None) -> bytes:
        return await self._read_stream(self._stderr, timeout_s)

    def __aiter__(self) -> AsyncIterator[bytes]:
        async def _iterate() -> AsyncIterator[bytes]:
            while chunk := await self.read():
                yield chunk

        return _iterate()

    async def _send(self, frame: bytes | str) -> None:
        if self._closed or self._ws.closed:
            raise SandboxPtyError("PTY session is closed")
        try:
            if isinstance(frame, bytes):
                await self._ws.send_bytes(frame)
            else:
                await self._ws.send_str(frame)
        except (aiohttp.ClientError, ConnectionResetError) as e:
            raise SandboxPtyError(f"PTY connection lost: {e}") from e

    async def write(self, data: bytes) -> None:
        await self._send(bytes([CHAN_STDIN]) + data)

    async def resize(self, rows: int, cols: int) -> None:
        await self._send(json.dumps({"type": "resize", "cols": cols, "rows": rows}))

    async def send_signal(self, signal: str) -> None:
        if signal not in SUPPORTED_SIGNALS:
            raise ValueError(f"execd PTY supports only {sorted(SUPPORTED_SIGNALS)}, got {signal!r}")
        await self._send(json.dumps({"type": "signal", "signal": signal}))

    async def wait_exit(self, *, timeout_s: float | None = None) -> int:
        # shield: the future is shared; a timed-out waiter must not cancel it.
        return await asyncio.wait_for(asyncio.shield(self._exit), timeout=timeout_s)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Settle the exit future before the pump can observe our own close.
        if not self._exit.done():
            self._exit.set_exception(SandboxPtyError("PTY session closed before process exit"))
            self._exit.exception()  # retrieved; silences never-retrieved warnings
        self._pump_task.cancel()
        # Let the pump's finally run before tearing the socket down.
        await asyncio.gather(self._pump_task, return_exceptions=True)
        try:
            await self._ws.close()
            if self._owned:
                try:
                    await self._client.delete(
                        self._session_url,
                        headers=self._headers,
                        timeout=aiohttp.ClientTimeout(total=self._request_timeout_s),
                    )
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    pass
        finally:
            await self._client.close()

    async def __aenter__(self) -> "OpenSandboxPtySession":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


async def open_pty_session(
    *,
    client: aiohttp.ClientSession,
    base_url: str,
    headers: dict[str, str],
    spec: SandboxPtySpec,
    request_timeout_s: float | None,
) -> OpenSandboxPtySession:
    """Create an execd PTY session and attach its WebSocket.

    Owns ``client``: it is closed on failure and by ``session.close()``.
    """
    body: dict[str, str] = {}
    if spec.cwd is not None:
        body["cwd"] = spec.cwd
    try:
        command = _effective_command(spec)
    except BaseException:
        await client.close()
        raise
    if command is not None:
        body["command"] = command

    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    try:
        async with client.post(f"{base_url}/pty", json=body, headers=headers, timeout=timeout) as response:
            if response.status == 404:
                raise SandboxPtyError(
                    "PTY create returned 404: execd inside this sandbox image predates "
                    "PTY support (execd >= 1.0.10 required)"
                )
            if response.status not in (200, 201):
                raise SandboxPtyError(f"PTY create failed with HTTP {response.status}: {await response.text()}")
            session_id = (await response.json())["session_id"]

        try:
            ws = await _connect_ws(
                client=client,
                base_url=base_url,
                headers=headers,
                session_id=session_id,
                query={} if spec.pty else {"pty": "0"},
                request_timeout_s=request_timeout_s,
            )
        except BaseException:
            try:
                await client.delete(f"{base_url}/pty/{session_id}", headers=headers, timeout=timeout)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            raise
    except SandboxPtyError:
        await client.close()
        raise
    except Exception as e:
        await client.close()
        raise SandboxPtyError(f"Failed to open PTY session: {e}") from e
    except BaseException:
        await client.close()
        raise

    session = await _start_session(
        client=client,
        ws=ws,
        base_url=base_url,
        session_id=session_id,
        headers=headers,
        request_timeout_s=request_timeout_s,
    )
    # execd hardcodes 80x24 at spawn; size is only settable post-attach.
    if spec.pty and (spec.rows, spec.cols) != (24, 80):
        try:
            await session.resize(spec.rows, spec.cols)
        except BaseException:
            await session.close()
            raise
    return session


async def attach_pty_session(
    *,
    client: aiohttp.ClientSession,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    takeover: bool = True,
    since: int | None = None,
    request_timeout_s: float | None,
) -> OpenSandboxPtySession:
    """Attach to an existing execd PTY session. Owns ``client`` as above."""
    query: dict[str, str] = {}
    if takeover:
        query["takeover"] = "1"
    if since is not None:
        query["since"] = str(since)
    try:
        ws = await _connect_ws(
            client=client,
            base_url=base_url,
            headers=headers,
            session_id=session_id,
            query=query,
            request_timeout_s=request_timeout_s,
        )
    except SandboxPtyError:
        await client.close()
        raise
    except aiohttp.WSServerHandshakeError as e:
        await client.close()
        if e.status == 409:
            raise SandboxPtyError(
                f"PTY session {session_id} already has an attached client (pass takeover=True to evict)"
            ) from e
        raise SandboxPtyError(f"Failed to attach to PTY session {session_id}: {e}") from e
    except Exception as e:
        await client.close()
        raise SandboxPtyError(f"Failed to attach to PTY session {session_id}: {e}") from e
    except BaseException:
        await client.close()
        raise
    return await _start_session(
        client=client,
        ws=ws,
        base_url=base_url,
        session_id=session_id,
        headers=headers,
        request_timeout_s=request_timeout_s,
        owned=False,
    )


async def _connect_ws(
    *,
    client: aiohttp.ClientSession,
    base_url: str,
    headers: dict[str, str],
    session_id: str,
    query: dict[str, str],
    request_timeout_s: float | None,
) -> aiohttp.ClientWebSocketResponse:
    ws_url = f"ws{base_url.removeprefix('http')}/pty/{session_id}/ws"
    if query:
        ws_url += "?" + urlencode(query)
    return await asyncio.wait_for(client.ws_connect(ws_url, headers=headers), timeout=request_timeout_s)


async def _start_session(
    *,
    client: aiohttp.ClientSession,
    ws: aiohttp.ClientWebSocketResponse,
    base_url: str,
    session_id: str,
    headers: dict[str, str],
    request_timeout_s: float | None,
    owned: bool = True,
) -> OpenSandboxPtySession:
    session = OpenSandboxPtySession(
        client=client,
        ws=ws,
        session_id=session_id,
        session_url=f"{base_url}/pty/{session_id}",
        headers=headers,
        request_timeout_s=request_timeout_s,
        owned=owned,
    )
    try:
        await session._wait_connected(request_timeout_s)
    except BaseException:
        await session.close()
        raise
    return session
