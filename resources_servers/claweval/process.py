# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, serialized communication with a task-owned native Python process."""

import asyncio
import json
import os
import signal
from pathlib import Path
from typing import IO


class TaskSetupRejectedError(ValueError):
    """A deterministic task/configuration refusal from the native worker."""


class TaskProcess:
    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.log: IO[bytes] | None = None
        self.broken = False

    async def start(self, python: str, output: Path) -> None:
        output.mkdir(parents=True, exist_ok=True)
        self.log = (output / "worker.log").open("wb")
        self.process = await asyncio.create_subprocess_exec(
            python,
            "-u",
            str(Path(__file__).with_name("worker.py")),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=self.log,
            start_new_session=True,
            limit=64 * 1024 * 1024,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

    async def call(self, operation: str, payload: dict, *, timeout: float) -> object:
        async with self.lock:
            process = self.process
            if self.broken or process is None or process.returncode is not None:
                raise RuntimeError("ClawEval task worker is unavailable; inspect worker.log")
            assert process.stdin is not None and process.stdout is not None
            try:
                async with asyncio.timeout(timeout):
                    process.stdin.write((json.dumps({"operation": operation, **payload}) + "\n").encode())
                    await process.stdin.drain()
                    line = await process.stdout.readline()
                    if not line:
                        raise RuntimeError("ClawEval task worker exited; inspect worker.log")
                    reply = json.loads(line)
            except BaseException:
                # A cancelled read must never leave a stale response for the next request.
                self.broken = True
                raise
            if "error" in reply:
                if operation == "seed" and reply.get("error_type") in {
                    "ValueError",
                    "FileNotFoundError",
                    "ValidationError",
                }:
                    raise TaskSetupRejectedError(reply["error"])
                raise RuntimeError(reply["error"])
            return reply["result"]

    async def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.returncode is None:
            if not self.broken:
                try:
                    await self.call("close", {}, timeout=90)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5)
                    except TimeoutError:
                        pass
                except (RuntimeError, TimeoutError, BrokenPipeError, ValueError):
                    self.broken = True
            if process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                await asyncio.wait_for(process.wait(), timeout=60)
            except TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.wait_for(process.wait(), timeout=5)
        if self.log is not None:
            self.log.close()
        self.process = None
