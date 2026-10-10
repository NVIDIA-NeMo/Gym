# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bound local transfer I/O without blocking the resource server's event loop."""

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context


logger = logging.getLogger(__name__)


async def _join_worker[T](future: asyncio.Future[T]) -> T:
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        # Python cannot stop a running thread. Keep the caller's files and pool
        # slot alive until it finishes, even if shutdown cancels us again.
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        try:
            future.result()
        except Exception:
            logger.warning("Archive worker failed while its caller was being cancelled", exc_info=True)
        raise


class ArchiveWorkers:
    """One bounded transfer-I/O pool shared by all sessions of a resources server."""

    def __init__(self, max_workers: int) -> None:
        if max_workers < 1:
            raise ValueError("Archive concurrency must be positive")
        self._slots = asyncio.Semaphore(max_workers)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tb4-archive")
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    async def run[T](self, operation: Callable[[], T]) -> T:
        """Wait asynchronously for capacity, then run and join one local operation."""
        if self._closing:
            raise RuntimeError("Archive workers are shutting down")
        async with self._slots:
            if self._closing:
                raise RuntimeError("Archive workers are shutting down")
            future = asyncio.get_running_loop().run_in_executor(self._executor, copy_context().run, operation)
            return await _join_worker(future)

    async def aclose(self) -> None:
        """Drain submitted work and join threads without blocking the event loop."""
        self._closing = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(asyncio.to_thread(self._executor.shutdown, wait=True))
        await _join_worker(self._close_task)

    async def __aenter__(self) -> "ArchiveWorkers":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()


async def run_local[T](operation: Callable[[], T], workers: ArchiveWorkers | None) -> T:
    """Use the server's pool, or a scoped single worker for standalone transfers."""
    if workers is not None:
        return await workers.run(operation)
    async with ArchiveWorkers(1) as standalone:
        return await standalone.run(operation)
