# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, reusable JSON-lines subprocesses for the two chemistry runtimes."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ScoringWorkerError(RuntimeError):
    """A scorer exited or broke its JSON-lines protocol."""

    def __init__(self, message: str, *, code: str = "upstream_error") -> None:
        super().__init__(message)
        self.code = code


@dataclass(eq=False)
class _Slot:
    key: tuple[tuple[str, ...], Path | None] | None = None
    busy: bool = False
    owner: asyncio.Task | None = None
    process: asyncio.subprocess.Process | None = None
    stderr_task: asyncio.Task[None] | None = None
    stderr_tail: bytes = b""

    async def drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while chunk := await self.process.stderr.read(8192):
            self.stderr_tail = (self.stderr_tail + chunk)[-2000:]

    async def stop(self) -> None:
        if self.process is not None:
            if self.process.returncode is None:
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
            # Drain buffered stdout after killing; Process.wait() alone can hang
            # when a verbose/broken worker filled the reader buffer.
            if self.process.stdout is not None:
                while await self.process.stdout.read(65536):
                    pass
            await self.process.wait()
        if self.stderr_task is not None:
            await self.stderr_task
        self.process = None
        self.stderr_task = None
        self.key = None


class ScoringPool:
    """Keep at most max_workers processes, preferring idle workers with the same runtime."""

    def __init__(self, max_workers: int) -> None:
        self._slots = [_Slot() for _ in range(max_workers)]
        self._condition = asyncio.Condition()
        self._closed = False

    async def score(
        self, command: list[str], payload: dict[str, Any], *, cwd: Path | None, timeout: float
    ) -> dict[str, Any]:
        """Score one answer; discard a worker on timeout, cancellation, or protocol failure."""
        key = (tuple(command), cwd)
        async with self._condition:
            await self._condition.wait_for(lambda: self._closed or any(not slot.busy for slot in self._slots))
            if self._closed:
                raise ScoringWorkerError("ChemCoTBench scoring pool is closed")
            available = [slot for slot in self._slots if not slot.busy]
            slot = next((slot for slot in available if slot.key == key), None)
            if slot is None:
                slot = next((slot for slot in available if slot.process is None), available[0])
            slot.busy = True
            slot.owner = asyncio.current_task()
        try:
            # Queueing does not consume the per-answer scoring timeout.
            async with asyncio.timeout(timeout):
                if slot.key != key or slot.process is None or slot.process.returncode is not None:
                    await slot.stop()
                    slot.stderr_tail = b""
                    slot.process = await asyncio.create_subprocess_exec(
                        *command,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=cwd,
                        # Optimization diagnostics can exceed asyncio's default 64 KiB line limit.
                        limit=16 * 1024 * 1024,
                    )
                    slot.key = key
                    slot.stderr_task = asyncio.create_task(slot.drain_stderr())
                process = slot.process
                assert process.stdin is not None and process.stdout is not None
                process.stdin.write(json.dumps(payload).encode() + b"\n")
                await process.stdin.drain()
                line = await process.stdout.readline()
                if not line:
                    await slot.stop()
                    raise ScoringWorkerError(slot.stderr_tail.decode(errors="replace") or "Scoring worker exited")
                try:
                    result = json.loads(line)
                    if not isinstance(result, dict):
                        raise ValueError("Expected an object")
                except (ValueError, UnicodeError) as error:
                    raise ScoringWorkerError("Invalid scoring worker JSON", code="invalid_result") from error
                return result
        except BaseException:
            # A cancelled request must not leave a late response for the next request.
            await slot.stop()
            raise
        finally:
            async with self._condition:
                slot.busy = False
                slot.owner = None
                self._condition.notify_all()

    async def close(self) -> None:
        """Reject queued requests and reap all workers, including in-flight ones."""
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
        active = [slot.owner for slot in self._slots if slot.owner is not None]
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)
        await asyncio.gather(*(slot.stop() for slot in self._slots))
