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

"""Interactive sessions over the E2B SDK's ``sandbox.pty``."""

import asyncio
import contextlib
import shlex
import uuid
from typing import Any

from nemo_gym.sandbox.providers.base import SandboxPtyError, SandboxPtySpec


_EOF = object()
_KILL_SIGNALS = {"SIGTERM", "SIGKILL"}


class E2BPtySession:
    """One client-owned E2B PTY process.

    E2B always starts ``/bin/bash -i -l`` for a PTY, so the image needs bash and
    a ``SandboxPtySpec.command`` is run by typing ``exec bash -c <command>`` into
    that shell (it replaces the shell, so its exit code is the session's).
    """

    mode = "pty"

    def __init__(self, sandbox: Any, spec: SandboxPtySpec, *, pty_size: Any) -> None:
        self.session_id = uuid.uuid4().hex
        self._sandbox = sandbox
        self._spec = spec
        self._pty_size = pty_size
        self._rows = spec.rows
        self._cols = spec.cols
        self._output: asyncio.Queue[bytes | object] = asyncio.Queue()
        self._exit: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self._closed = False
        self._handle: Any | None = None
        self._waiter: asyncio.Task[None] | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    async def start(self) -> "E2BPtySession":
        spec = self._spec
        envs = {str(k): str(v) for k, v in (spec.env or {}).items()}
        kwargs: dict[str, Any] = {}
        if spec.user is not None:
            kwargs["user"] = str(spec.user)
        # timeout=0 lifts the SDK's default 60 s bound on the running stream.
        self._handle = await self._sandbox.pty.create(
            self._pty_size(rows=self._rows, cols=self._cols),
            self._on_data,
            cwd=spec.cwd,
            envs=envs or None,
            timeout=0,
            **kwargs,
        )
        self._waiter = asyncio.create_task(self._wait_for_exit())
        if spec.command is not None:
            # One quoted unit: a bare `exec a; b` would replace the shell with
            # `a` and never run `b`, losing the command's own exit status.
            await self.write(f"exec bash -c {shlex.quote(spec.command)}\n".encode())
        return self

    def _on_data(self, data: bytes) -> None:
        self._output.put_nowait(bytes(data))

    async def _wait_for_exit(self) -> None:
        try:
            try:
                result = await self._handle.wait()
                code = result.exit_code
            except Exception as exc:  # noqa: BLE001 - CommandExitException carries the exit code
                code = getattr(exc, "exit_code", None)
                if code is None:
                    if not self._exit.done():
                        self._exit.set_exception(SandboxPtyError(f"E2B PTY stream failed: {exc}"))
                    return
            if not self._exit.done():
                self._exit.set_result(code)
        finally:
            self._output.put_nowait(_EOF)

    async def read(self, *, timeout_s: float | None = None) -> bytes:
        if timeout_s is None:
            value = await self._output.get()
        else:
            async with asyncio.timeout(timeout_s):
                value = await self._output.get()
        if value is _EOF:
            # Keep the sentinel so later reads and other iterators also see EOF.
            self._output.put_nowait(_EOF)
            if self._exit.done() and self._exit.exception() is not None:
                raise self._exit.exception()  # type: ignore[misc]
            return b""
        return value  # type: ignore[return-value]

    async def read_stderr(self, *, timeout_s: float | None = None) -> bytes:
        # PTY mode merges stderr into the terminal stream: block until the
        # process exits, then report EOF.
        with contextlib.suppress(SandboxPtyError):
            await self.wait_exit(timeout_s=timeout_s)
        return b""

    def __aiter__(self) -> "E2BPtySession":
        return self

    async def __anext__(self) -> bytes:
        chunk = await self.read()
        if not chunk:
            raise StopAsyncIteration
        return chunk

    async def write(self, data: bytes) -> None:
        if self._closed:
            raise SandboxPtyError("E2B PTY session is closed")
        await self._sandbox.pty.send_stdin(self._handle.pid, data)

    async def resize(self, rows: int, cols: int) -> None:
        if self._closed:
            raise SandboxPtyError("E2B PTY session is closed")
        if (rows, cols) == (self._rows, self._cols):
            return
        await self._sandbox.pty.resize(self._handle.pid, self._pty_size(rows=rows, cols=cols))
        self._rows, self._cols = rows, cols

    async def send_signal(self, signal: str) -> None:
        name = signal.upper()
        if name == "SIGINT":
            await self.write(b"\x03")
        elif name in _KILL_SIGNALS:
            # E2B's PTY kill is SIGKILL; there is no gentler signal to send.
            await self._sandbox.pty.kill(self._handle.pid)
        else:
            raise NotImplementedError(f"E2B PTY cannot send {signal}")

    async def wait_exit(self, *, timeout_s: float | None = None) -> int:
        if timeout_s is None:
            return await asyncio.shield(self._exit)
        async with asyncio.timeout(timeout_s):
            return await asyncio.shield(self._exit)

    async def run_detached(self, command: str, *, poll_interval_s: float = 15.0) -> tuple[bytes, int | None]:
        del command, poll_interval_s
        raise NotImplementedError("E2B PTY does not support detached PTY execution")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._handle is not None and not self._exit.done():
            with contextlib.suppress(Exception):
                await self._sandbox.pty.kill(self._handle.pid)
        if self._waiter is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(self._waiter, timeout=10)
        if not self._exit.done():
            self._exit.cancel()

    async def __aenter__(self) -> "E2BPtySession":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()


def validate_pty_spec(spec: SandboxPtySpec) -> None:
    """Reject spec fields the E2B SDK cannot honor, rather than dropping them."""
    if not spec.pty:
        raise ValueError("The e2b provider only supports PTY mode; pipe mode (pty=False) is not available")
    if spec.command is not None and not spec.command.strip():
        raise ValueError("SandboxPtySpec.command must not be blank")
