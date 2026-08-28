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

    BrowserGym's synchronous Playwright instance is greenlet- and
    thread-affine.  Native visual runtimes can reuse this policy when their
    browser controller has the same constraint, without coupling the common
    session lifecycle to BrowserGym.
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
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._finalizer = None
            self._closed = True
