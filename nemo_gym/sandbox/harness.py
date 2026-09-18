# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution contracts and thread bridge for resource-owned agent harnesses."""

import asyncio
from concurrent.futures import Future
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field


class HarnessOutcome(BaseModel):
    reason: str
    exit_code: int | None = None
    detail: str | None = None
    artifacts: list[str] = Field(default_factory=list)


class HarnessContext(BaseModel):
    session_id: str
    instruction: str
    user: str | int | None = None
    workdir: str | None = None
    setup_timeout_sec: float = Field(default=360, gt=0)
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    skills_dir: str | None = None


class WorkerBridge:
    """Join a synchronous agent loop and the asynchronous operations it started."""

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.pending = set()
        self.tasks = set()
        self.closed = False
        self.lock = Lock()

    def start(self, factory, future):
        with self.lock:
            if self.closed:
                future.cancel()
                return
            try:
                task = self.loop.create_task(factory())
            except Exception as exc:
                future.set_exception(exc)
                return
            self.tasks.add(task)

        def completed(task):
            self.tasks.discard(task)
            if future.cancelled():
                return
            if task.cancelled():
                future.cancel()
            elif error := task.exception():
                future.set_exception(error)
            else:
                future.set_result(task.result())

        task.add_done_callback(completed)

    def call(self, factory):
        with self.lock:
            if self.closed:
                raise RuntimeError("Episode is closed")
            future = Future()
            self.pending.add(future)
            self.loop.call_soon_threadsafe(self.start, factory, future)
        try:
            return future.result()
        finally:
            with self.lock:
                self.pending.discard(future)

    def close(self):
        with self.lock:
            self.closed = True
            for future in self.pending:
                future.cancel()
        for task in list(self.tasks):
            task.cancel()

    async def aclose(self):
        self.close()
        await asyncio.gather(*list(self.tasks), return_exceptions=True)
