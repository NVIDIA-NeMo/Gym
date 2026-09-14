# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Interactive sessions over MFN ExecStream."""

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from typing import Any

from nemo_gym.sandbox.providers.base import SandboxPtyError, SandboxPtySpec
from nemo_gym.sandbox.providers.mfn.protos import mfn_sandbox_pb2 as pb


_EOF = object()


class MFNPtySession:
    """One client-owned MFN duplex exec stream."""

    def __init__(self, stub: Any, sandbox_id: str, spec: SandboxPtySpec, *, shell: str) -> None:
        self.session_id = uuid.uuid4().hex
        self.mode = "pty" if spec.pty else "pipe"
        self._stub = stub
        self._sandbox_id = sandbox_id
        self._spec = spec
        self._shell = shell
        self._requests: asyncio.Queue[Any] = asyncio.Queue()
        self._stdout: asyncio.Queue[bytes | object] = asyncio.Queue()
        self._stderr: asyncio.Queue[bytes | object] = asyncio.Queue()
        self._exit: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self._closed = False
        self._call: Any | None = None
        self._reader: asyncio.Task[None] | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    def _command(self) -> list[str]:
        command = self._spec.command
        if command is None:
            return [self._shell]
        if self._spec.user not in (None, "root", 0):
            import shlex

            command = f"su -s {shlex.quote(self._shell)} -c {shlex.quote(command)} {shlex.quote(str(self._spec.user))}"
        return [self._shell, "-c", command]

    async def start(self) -> "MFNPtySession":
        request = pb.ExecRequest(
            sandbox_id=self._sandbox_id,
            command=self._command(),
            cwd=self._spec.cwd or "",
            env={str(key): str(value) for key, value in (self._spec.env or {}).items()},
            unbuffered=True,
        )
        if self._spec.pty:
            request.pty.CopyFrom(pb.Pty(rows=self._spec.rows, cols=self._spec.cols))
        await self._requests.put(pb.ExecStreamRequest(request=request))
        self._call = self._stub.ExecStream(self._request_iterator())
        self._reader = asyncio.create_task(self._read_responses())
        return self

    async def _request_iterator(self) -> AsyncIterator[Any]:
        while True:
            item = await self._requests.get()
            if item is _EOF:
                return
            yield item

    async def _read_responses(self) -> None:
        error: BaseException | None = None
        try:
            async for response in self._call:
                if response.HasField("output"):
                    queue = self._stdout
                    if not self._spec.pty and response.output.stream == pb.ExecOutput.STDERR:
                        queue = self._stderr
                    await queue.put(bytes(response.output.data))
                elif response.HasField("complete"):
                    complete = response.complete
                    if complete.error:
                        await (self._stdout if self._spec.pty else self._stderr).put(complete.error.encode())
                    if complete.termination_detail:
                        await (self._stdout if self._spec.pty else self._stderr).put(
                            complete.termination_detail.encode()
                        )
                    if not self._exit.done():
                        self._exit.set_result(complete.exit_code)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            error = SandboxPtyError(f"MFN PTY stream failed: {exc}")
            if not self._exit.done():
                self._exit.set_exception(error)
        finally:
            if not self._exit.done() and error is None:
                self._exit.set_exception(SandboxPtyError("MFN PTY stream ended without an exit result"))
            await self._stdout.put(_EOF)
            await self._stderr.put(_EOF)

    async def _read(self, queue: asyncio.Queue[bytes | object], timeout_s: float | None) -> bytes:
        if timeout_s is None:
            value = await queue.get()
        else:
            async with asyncio.timeout(timeout_s):
                value = await queue.get()
        return b"" if value is _EOF else value  # type: ignore[return-value]

    async def read(self, *, timeout_s: float | None = None) -> bytes:
        return await self._read(self._stdout, timeout_s)

    async def read_stderr(self, *, timeout_s: float | None = None) -> bytes:
        return await self._read(self._stderr, timeout_s)

    def __aiter__(self) -> "MFNPtySession":
        return self

    async def __anext__(self) -> bytes:
        chunk = await self.read()
        if not chunk:
            raise StopAsyncIteration
        return chunk

    async def write(self, data: bytes) -> None:
        if self._closed:
            raise SandboxPtyError("MFN PTY session is closed")
        await self._requests.put(pb.ExecStreamRequest(stdin=pb.SendStdInRequest(data=data)))

    async def resize(self, rows: int, cols: int) -> None:
        if (rows, cols) == (self._spec.rows, self._spec.cols):
            return
        raise NotImplementedError("MFN PTY dimensions are fixed when ExecStream starts")

    async def send_signal(self, signal: str) -> None:
        if signal.upper() == "SIGINT" and self._spec.pty:
            await self.write(b"\x03")
            return
        raise NotImplementedError(f"MFN ExecStream cannot send {signal}")

    async def wait_exit(self, *, timeout_s: float | None = None) -> int:
        if timeout_s is None:
            return await asyncio.shield(self._exit)
        async with asyncio.timeout(timeout_s):
            return await asyncio.shield(self._exit)

    async def run_detached(self, command: str, *, poll_interval_s: float = 15.0) -> tuple[bytes, int | None]:
        del command, poll_interval_s
        raise NotImplementedError("MFN ExecStream does not support detached PTY execution")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._requests.put(_EOF)
        if self._call is not None and not self._exit.done():
            cancel = getattr(self._call, "cancel", None)
            if cancel is not None:
                cancel()
        if self._reader is not None:
            with contextlib.suppress(asyncio.CancelledError, SandboxPtyError):
                await self._reader
        if not self._exit.done():
            self._exit.cancel()

    async def __aenter__(self) -> "MFNPtySession":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
