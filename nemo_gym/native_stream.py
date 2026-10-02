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
"""Incremental native pipe observation without taking ownership of execution."""

import asyncio
import json
import os
import signal
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from importlib import import_module
from inspect import isawaitable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.global_config import ATTEMPT_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.rollout_correlation import trajectory_identity


PIPE_CHUNK_BYTES = 16 * 1024
NativeChannel = Literal["stdout", "stderr"]


_SCOPE_HEADER = "x-nemo-gym-native-scope"
_MAX_SCOPE_BYTES = 8192
_NATIVE_SCOPE: ContextVar["NativeInvocationScope | None"] = ContextVar("native_invocation_scope", default=None)


class NativeInvocationScope(BaseModel):
    """Collector-owned identity bound before agent requests are multiplexed."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    task_id: str
    rollout_id: str
    agent_name: str
    task_index: int = Field(ge=0)
    rollout_index: int = Field(ge=0)
    attempt_index: int = Field(default=0, ge=0)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "NativeInvocationScope":
        task_id, rollout_id = trajectory_identity(row)
        return cls(
            task_id=task_id,
            rollout_id=rollout_id,
            agent_name=row["agent_ref"]["name"],
            task_index=row[TASK_INDEX_KEY_NAME],
            rollout_index=row[ROLLOUT_INDEX_KEY_NAME],
            attempt_index=row.get(ATTEMPT_INDEX_KEY_NAME, 0),
        )

    def headers(self) -> dict[str, str]:
        value = json.dumps(self.model_dump(), ensure_ascii=True, separators=(",", ":"))
        if len(value) > _MAX_SCOPE_BYTES:
            raise ValueError("native observation scope exceeds header limit")
        return {_SCOPE_HEADER: value}


def native_stream_headers() -> dict[str, str]:
    """Forward observation scope only on an agent's explicit internal self-call."""
    scope = _NATIVE_SCOPE.get()
    return scope.headers() if scope is not None else {}


class NativeScopeMiddleware:
    """Request-local scope; malformed metadata is rejected before execution.

    This is correlation within Gym's trusted service network, not authentication.
    Context is always reset, including cancellation and failed requests.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        values = [value for key, value in scope.get("headers", []) if key.lower() == _SCOPE_HEADER.encode()]
        try:
            if len(values) > 1 or (values and len(values[0]) > _MAX_SCOPE_BYTES):
                raise ValueError("invalid observation scope header")
            metadata = NativeInvocationScope.model_validate_json(values[0]) if values else None
        except ValueError:
            await send({"type": "http.response.start", "status": 400, "headers": []})
            await send({"type": "http.response.body", "body": b"invalid native observation scope"})
            return
        token = _NATIVE_SCOPE.set(metadata)
        try:
            await self.app(scope, receive, send)
        finally:
            _NATIVE_SCOPE.reset(token)


class NativeStreamConfig(BaseModel):
    """Trusted installed observer factory; native execution remains owned by Gym.

    The factory is ``module:attribute`` and receives keyword arguments
    ``options``, ``agent_name``, ``rollout_id`` and ``scope``. It must return a fresh
    NativeStreamObserver synchronously, without blocking the event loop.
    Like a component entrypoint, this is trusted executable configuration.
    """

    model_config = ConfigDict(extra="forbid")
    factory: str = Field(pattern=r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")
    options: dict[str, str] = Field(default_factory=dict)

    def create(self, *, agent_name: str, rollout_id: str | None) -> "NativeStreamObserver":
        module, attribute = self.factory.split(":")
        factory = getattr(import_module(module), attribute)
        scope = _NATIVE_SCOPE.get()
        if scope is not None and (scope.rollout_id != rollout_id or scope.agent_name != agent_name):
            raise ValueError("native observation scope does not match invocation")
        observer = factory(options=dict(self.options), agent_name=agent_name, rollout_id=rollout_id, scope=scope)
        if not isinstance(observer, NativeStreamObserver):
            raise TypeError("native stream factory must return NativeStreamObserver")
        return observer


@dataclass
class NativeStreamObserver:
    """A per-invocation, nonblocking consumer of exact native pipe bytes.

    An empty chunk marks EOF for that channel. Chunks are not text or JSON
    boundaries. Consumers must bound their own queues and parsers. A failing
    consumer is disabled and recorded here; it cannot stop native pipe drainage.
    Owners must retain ``failed`` as an observation gap, never a successful capture.
    Start/consume callbacks may return awaitables with a three-second cooperative
    deadline. Synchronous callbacks must not block; awaitables must honor cancellation.
    """

    consume: Callable[[NativeChannel, bytes], Awaitable[None] | None]
    failed: bool = False
    on_start: Callable[[str, str | None], Awaitable[None] | None] | None = None
    on_finish: Callable[[int | None, bool], Awaitable[None]] | None = None
    _closing: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    async def start(self, instruction: str, system_prompt: str | None) -> None:
        """Observe resolved runner input once, before spawning the native process."""
        if self.on_start is not None and not self.failed:
            try:
                async with asyncio.timeout(3):
                    result = self.on_start(instruction, system_prompt)
                    if isawaitable(result):
                        await result
            except asyncio.CancelledError:
                self.failed = True
                raise
            except Exception:
                self.failed = True

    async def close(self, *, returncode: int | None, incomplete: bool = False) -> None:
        """Finalize once within three seconds, preserving caller cancellation.

        The callback receives process exit and observation completeness, not
        task correctness. A failed content callback still receives finalization.
        The callback must cooperate with cancellation and must not block.
        """
        self.failed |= incomplete
        if self._closing is None:
            self._closing = asyncio.create_task(self._finish(returncode))
        cancelled = False
        while not self._closing.done():
            try:
                await asyncio.shield(self._closing)
            except asyncio.CancelledError:
                cancelled = True
        self._closing.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _finish(self, returncode: int | None) -> None:
        try:
            if self.on_finish is not None:
                async with asyncio.timeout(3):
                    await self.on_finish(returncode, self.failed)
        except (Exception, asyncio.CancelledError):
            self.failed = True

    async def emit(self, channel: NativeChannel, chunk: bytes) -> None:
        if self.failed:
            return
        try:
            result = self.consume(channel, chunk)
            if isawaitable(result):
                async with asyncio.timeout(3):
                    await result
        except asyncio.CancelledError:
            self.failed = True
            raise
        except Exception:
            self.failed = True


async def read_native_pipe(
    stream: asyncio.StreamReader, channel: NativeChannel, observer: NativeStreamObserver | None = None
) -> bytes:
    """Read one native pipe with optional bounded, lossless observation."""
    chunks: list[bytes] = []
    try:
        while chunk := await stream.read(PIPE_CHUNK_BYTES):
            if observer is not None:
                await observer.emit(channel, chunk)
            chunks.append(chunk)
        if observer is not None:
            await observer.emit(channel, b"")
        return b"".join(chunks)
    except BaseException:
        if observer is not None:
            observer.failed = True
        raise


async def communicate_native(
    process: asyncio.subprocess.Process,
    observer: NativeStreamObserver,
) -> tuple[bytes, bytes]:
    """Drain both native pipes, observing bounded chunks before process exit.

    Like ``Process.communicate()``, this retains complete stdout/stderr for
    existing native parsers. Only observation chunks are bounded; transcript
    retention is not. The caller owns process timeout, termination and reaping.
    Cancellation cancels and joins both readers before returning to that owner.
    stdin must already be closed or unused.
    """
    if process.stdout is None or process.stderr is None:
        raise ValueError("native observation requires both output pipes")

    readers = [
        asyncio.create_task(read_native_pipe(process.stdout, "stdout", observer)),
        asyncio.create_task(read_native_pipe(process.stderr, "stderr", observer)),
    ]
    try:
        stdout, stderr = await asyncio.gather(*readers)
        await process.wait()
        return stdout, stderr
    except BaseException:
        observer.failed = True
        raise
    finally:
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)


def kill_native_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill the native subprocess and every child in its process group.

    Killing only the direct child leaves the npm shim's vendored-binary child alive, holding the
    stdout pipe open — the post-kill ``communicate()`` would then block until the orphan exits.
    """
    # start_new_session=True makes the original PID the group ID. Looking it
    # up after the leader exits loses the group while inherited pipes stay open.
    try:
        pid = proc.pid
    except AttributeError:  # Lightweight process doubles used by embedders/tests.
        proc.kill()
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
