# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution strategies for synchronous web-environment backends."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Protocol


class WebOperationRunner(Protocol):
    """Run synchronous backend operations without prescribing a browser stack."""

    async def run(self, operation: Callable[..., Any], *args: Any) -> Any: ...

    async def close(self) -> None: ...


class DirectWebOperationRunner:
    """Run inexpensive or natively asynchronous-safe backend calls inline."""

    async def run(self, operation: Callable[..., Any], *args: Any) -> Any:
        return operation(*args)

    async def close(self) -> None:
        return None


class ThreadAffineWebOperationRunner:
    """Run every operation on one dedicated thread.

    Playwright's synchronous API is greenlet- and thread-affine. The dedicated
    thread keeps those calls off the FastAPI event loop. It is not the browser
    isolation boundary: headed desktop-control runtimes still scale as one
    resources-server process or container per DISPLAY.
    """

    def __init__(
        self,
        *,
        thread_name_prefix: str = "web-runtime",
        finalizer: Callable[[], Any] | None = None,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name_prefix)
        self._finalizer = finalizer
        self._closed = False

    async def run(self, operation: Callable[..., Any], *args: Any) -> Any:
        if self._closed:
            raise RuntimeError("web operation runner has already stopped")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(operation, *args))

    async def close(self) -> None:
        if self._closed:
            return
        try:
            if self._finalizer is not None:
                await self.run(self._finalizer)
        finally:
            # ``shutdown(wait=True)`` is synchronous and can otherwise pin the
            # FastAPI event loop behind a slow Playwright/browser teardown.
            # Run the join on asyncio's shared worker pool; the browser calls
            # themselves remain confined to this runner's single worker.
            await asyncio.to_thread(
                self._executor.shutdown,
                wait=True,
                cancel_futures=True,
            )
            self._finalizer = None
            self._closed = True
