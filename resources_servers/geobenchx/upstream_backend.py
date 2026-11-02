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

"""Process pool that runs the real upstream GeoBenchX tools on the local data snapshot."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


WORKER_SCRIPT = Path(__file__).with_name("upstream_worker.py")
# Large rasters and dataframe descriptions can produce long JSON lines.
STREAM_LIMIT = 64 * 1024 * 1024


def _error(message: str) -> dict[str, Any]:
    return {"ok": False, "result": f"Error: {message}"}


class _WorkerProcess:
    def __init__(
        self,
        python: str,
        upstream_src: str,
        env: dict[str, str],
        max_sessions: int,
        session_idle_ttl_s: float,
        startup_timeout_s: float,
    ) -> None:
        self._cmd = [
            python,
            str(WORKER_SCRIPT),
            "--upstream-src",
            upstream_src,
            "--max-sessions",
            str(max_sessions),
            "--idle-ttl-s",
            str(session_idle_ttl_s),
        ]
        self._env = env
        self._startup_timeout_s = startup_timeout_s
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._next_id = 0
        self._closed = False

    async def _ensure_started(self) -> asyncio.subprocess.Process:
        if self._proc is not None and self._proc.returncode is None:
            return self._proc
        if self._closed:
            raise RuntimeError("GeoBenchX upstream worker pool is closed")
        self._proc = await asyncio.create_subprocess_exec(
            *self._cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=sys.stderr,
            env=self._env,
            limit=STREAM_LIMIT,
        )
        try:
            # Importing the geospatial stack is slow, so startup has its own bound.
            async with asyncio.timeout(self._startup_timeout_s):
                ready = json.loads(await self._proc.stdout.readline() or b"{}")
        except (TimeoutError, ValueError):
            ready = {}
        if not isinstance(ready, dict) or not ready.get("ready"):
            await self._kill()
            raise RuntimeError("GeoBenchX upstream worker failed to start")
        return self._proc

    async def start(self) -> None:
        async with self._lock:
            await self._ensure_started()

    async def request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        # A cancelled caller abandons the reply but the transaction runs to completion
        # (bounded by the timeouts), so the worker and its other sessions stay usable.
        return await asyncio.shield(self._transaction(payload, timeout))

    async def _transaction(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        async with self._lock:
            try:
                proc = await self._ensure_started()
            except Exception:
                return _error("tool worker failed to start; session state was lost.")
            self._next_id += 1
            request_id = self._next_id
            try:
                async with asyncio.timeout(timeout):
                    proc.stdin.write((json.dumps({**payload, "id": request_id}) + "\n").encode())
                    await proc.stdin.drain()
                    line = await proc.stdout.readline()
            except TimeoutError:
                await self._kill()
                return _error(f"tool timed out after {timeout:.0f}s; session state was lost.")
            except (ConnectionError, OSError):
                line = b""
            try:
                response = json.loads(line) if line else None
            except ValueError:
                response = None
            if not isinstance(response, dict) or response.pop("id", None) != request_id:
                await self._kill()
                return _error("tool worker crashed; session state was lost.")
            return response

    async def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()

    async def close(self) -> None:
        # Not under the lock: a hung tool call must not delay shutdown. The pending
        # transaction sees end-of-stream and returns an error.
        self._closed = True
        await self._kill()


class UpstreamToolPool:
    """Routes each session to a fixed worker process so its dataframes stay in one place."""

    def __init__(
        self,
        python: str,
        upstream_src: str,
        stat_data_path: str,
        geo_data_path: str,
        scratch_path: str,
        num_workers: int,
        tool_timeout_s: float,
        max_sessions_per_worker: int,
        startup_timeout_s: float = 300.0,
        session_idle_ttl_s: float = 3600.0,
    ) -> None:
        for label, path in (("stat_data_path", stat_data_path), ("geo_data_path", geo_data_path)):
            if not Path(path).is_dir():
                raise FileNotFoundError(f"GeoBenchX {label} does not exist: {path}")
        if not (Path(upstream_src) / "geobenchx" / "tools.py").is_file():
            raise FileNotFoundError(f"GeoBenchX upstream_src has no geobenchx/tools.py: {upstream_src}")
        Path(scratch_path).mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "STATDATAPATH": stat_data_path,
            "GEODATAPATH": geo_data_path,
            "SCRATCHPATH": scratch_path,
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(Path(scratch_path) / ".mpl"),
        }
        self._timeout = tool_timeout_s
        self._workers = [
            _WorkerProcess(python, upstream_src, env, max_sessions_per_worker, session_idle_ttl_s, startup_timeout_s)
            for _ in range(num_workers)
        ]

    def _worker(self, session_id: str) -> _WorkerProcess:
        digest = hashlib.sha256(session_id.encode()).digest()
        return self._workers[int.from_bytes(digest[:8], "big") % len(self._workers)]

    async def start(self) -> None:
        """Start every worker, so a broken tool environment fails at boot rather than mid-rollout."""
        await asyncio.gather(*(w.start() for w in self._workers))

    async def reset(self, session_id: str) -> dict[str, Any]:
        return await self._worker(session_id).request({"op": "reset", "session": session_id}, self._timeout)

    async def release(self, session_id: str) -> dict[str, Any]:
        return await self._worker(session_id).request({"op": "release", "session": session_id}, self._timeout)

    async def call(self, session_id: str, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._worker(session_id).request(
            {"op": "call", "session": session_id, "tool": tool, "arguments": arguments}, self._timeout
        )

    async def close(self) -> None:
        await asyncio.gather(*(w.close() for w in self._workers))
