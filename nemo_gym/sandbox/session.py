# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-server lifecycle for one supervised harness activation."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import cast

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.process_supervisor import CleanupReceipt, exec_timeout
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.supervisor_client import (
    remove_session_directory,
    stop_and_confirm_cleanup,
    supervised_launch_command,
)


LOG = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class SandboxCommand:
    """Harness argv and the interpreter/path used to launch its supervisor."""

    argv: list[str]
    python: str = "python3"
    supervisor_path: str | None = None


@dataclass(kw_only=True)
class SandboxSession[Artifacts]:
    """Execute, stop, capture, then release an owned or borrowed sandbox.

    Runtime installation belongs in the adapter's seed hook. ``prepare`` stages
    activation input and returns a command; ``collect`` copies harness artifacts
    to the agent server after cleanup is confirmed. It may run a bounded snapshot
    command, but must not restart the harness. Parsing responses stays in the adapter.

    Each stop, collection and release phase is bounded by ``close_timeout`` (or
    the timeout passed to ``close``). Failed cleanup/release retains the handle
    for retry. Failed capture is recorded separately and does not prevent release.
    HTTP request binding and activation retries belong to the agent session layer.
    """

    sandbox: AsyncSandbox
    directory: str
    workdir: str | None
    harness: str
    owns_sandbox: bool = False
    launch_started: bool = field(default=False, init=False)
    cleanup: CleanupReceipt | None = field(default=None, init=False)
    artifacts: Artifacts | None = field(default=None, init=False)
    capture_error: Exception | None = field(default=None, init=False)
    sandbox_stopped: bool = field(default=False, init=False)
    closing: bool = field(default=False, init=False)
    closed: bool = field(default=False, init=False)
    _prepare_task: asyncio.Future[SandboxCommand] | None = field(default=None, init=False)
    _exec_task: asyncio.Task[SandboxExecResult] | None = field(default=None, init=False)
    _finalize_task: asyncio.Task[None] | None = field(default=None, init=False)
    _close_task: asyncio.Task[None] | None = field(default=None, init=False)
    _collect: Callable[[], Awaitable[Artifacts]] | None = field(default=None, init=False)
    _capture_attempted: bool = field(default=False, init=False)

    async def execute(
        self,
        *,
        prepare: Callable[[], Awaitable[SandboxCommand]],
        collect: Callable[[], Awaitable[Artifacts]],
        timeout: float,
        close_timeout: float,
    ) -> Artifacts:
        """Run once, preserving captured artifacts even when execution raises.

        Cancellation stops the remote harness before releasing provider exec.
        Callers that merely stop waiting (such as disconnected HTTP requests)
        must shield their shared activation task.
        """
        if self.closing or self._prepare_task is not None:
            raise RuntimeError(f"{self.harness} sandbox session is closing or already activated")
        # Register both hooks before yielding, so close can fence preparation too.
        self._collect = collect
        self._prepare_task = asyncio.ensure_future(prepare())
        try:
            command = await asyncio.shield(self._prepare_task)
            if self.closing or self._finalize_task is not None:
                raise RuntimeError(f"{self.harness} sandbox session closed before launch")
            cleanup_timeout = close_timeout / 3
            launch = supervised_launch_command(
                directory=self.directory,
                command=command.argv,
                timeout=timeout,
                cleanup_timeout=cleanup_timeout,
                python=command.python,
                supervisor_path=command.supervisor_path,
            )
            self.launch_started = True
            self._exec_task = asyncio.create_task(
                self.sandbox.exec(
                    launch,
                    cwd=self.workdir,
                    timeout_s=exec_timeout(timeout=timeout, cleanup_timeout=cleanup_timeout),
                )
            )
            result = await asyncio.shield(self._exec_task)
            if result.error_type == "timeout":
                raise TimeoutError(f"{self.harness} sandbox supervisor exceeded its execution deadline")
            if result.error_type:
                raise RuntimeError(f"{self.harness} sandbox execution failed: {result.error_type}: {result.stderr}")
            # A worker's nonzero exit can still carry useful, adapter-specific output.
        except BaseException:
            try:
                await self._finish(timeout=close_timeout)
            except Exception:
                LOG.exception("%s cleanup remains unconfirmed; close must retry", self.harness)
            raise
        await self._finish(timeout=close_timeout)
        if self.capture_error is not None:
            raise self.capture_error
        # Collection either returned artifacts (including a valid None) or recorded an error.
        return cast(Artifacts, self.artifacts)

    async def stop_runner(self, *, timeout: float) -> None:
        """Fence a delayed launch or confirm descendant cleanup before transport cancellation."""
        if self.sandbox_stopped or not self.launch_started or self.cleanup is not None:
            return
        try:
            async with asyncio.timeout(timeout):
                self.cleanup = await stop_and_confirm_cleanup(
                    self.sandbox,
                    directory=self.directory,
                    workdir=self.workdir,
                    timeout=timeout,
                    harness=self.harness,
                )
        except TimeoutError as error:
            raise RuntimeError(f"{self.harness} launch outcome is unknown; cleanup deadline exceeded") from error

    async def _finish(self, *, timeout: float) -> None:
        if self._finalize_task is None or (
            self._finalize_task.done()
            and (self._finalize_task.cancelled() or self._finalize_task.exception() is not None)
        ):
            self._finalize_task = asyncio.create_task(self._finalize(timeout=timeout))
        await asyncio.shield(self._finalize_task)

    async def _cancel_and_wait[T](self, task: asyncio.Future[T], *, timeout: float) -> None:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
        except Exception:
            if not task.done():
                raise
            # Execution/preparation errors are propagated by execute, independently of cleanup.

    async def _finalize(self, *, timeout: float) -> None:
        if self._prepare_task is not None:
            await self._cancel_and_wait(self._prepare_task, timeout=timeout)
        await self.stop_runner(timeout=timeout)
        if self.launch_started and not self._capture_attempted and self._collect is not None:
            if self.sandbox_stopped:
                self.capture_error = RuntimeError(f"{self.harness} sandbox stopped before artifact capture")
            else:
                try:
                    async with asyncio.timeout(timeout):
                        self.artifacts = await self._collect()
                    self.capture_error = None
                except Exception as error:
                    self.capture_error = error
                    LOG.exception("%s artifact capture failed after confirmed cleanup", self.harness)
            self._capture_attempted = True
        await self._release_exec(timeout=timeout)

    async def _release_exec(self, *, timeout: float) -> None:
        if (self.cleanup is not None or self.sandbox_stopped) and self._exec_task is not None:
            await self._cancel_and_wait(self._exec_task, timeout=timeout)

    async def close(self, *, timeout: float) -> None:
        """Join finalization before release; concurrent/retried closes never duplicate capture."""
        self.closing = True
        if self.closed:
            return
        if self._close_task is None or (
            self._close_task.done() and (self._close_task.cancelled() or self._close_task.exception() is not None)
        ):
            self._close_task = asyncio.create_task(self._close(timeout=timeout))
        await asyncio.shield(self._close_task)

    async def _close(self, *, timeout: float) -> None:
        try:
            await self._finish(timeout=timeout)
        except Exception:
            if not self.owns_sandbox:
                raise
            # Container teardown remains the authority if graceful cleanup is unavailable.
            LOG.exception("%s graceful cleanup failed; stopping owned sandbox", self.harness)
            if not self._capture_attempted:
                self.capture_error = RuntimeError(f"{self.harness} artifact capture unavailable during forced stop")
        if self.owns_sandbox:
            if not self.sandbox_stopped:
                async with asyncio.timeout(timeout):
                    await self.sandbox.stop()
                self.sandbox_stopped = True
            await self._release_exec(timeout=timeout)
        else:
            async with asyncio.timeout(timeout):
                await remove_session_directory(
                    self.sandbox,
                    directory=self.directory,
                    workdir=self.workdir,
                    timeout=timeout,
                    harness=self.harness,
                )
                await self.sandbox.disconnect()
        self.closed = True
