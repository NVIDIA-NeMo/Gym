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
"""A pool of ``appworld serve environment`` subprocesses, one per live episode.

**Why a process pool and not in-process AppWorld objects.** ``AppWorld.__init__``
calls ``AppWorld.close_all()``, which stops every live time-freezer, clears the
process-global DB cache and closes every open API collection. Each task also
freezes wall-clock time process-wide (freezegun) to that task's simulated
datetime. So a process hosts exactly **one** live AppWorld environment, and a
second ``seed_session`` in the same process would silently destroy the first.
Upstream encodes the same rule: in-process mode is ``parallelizable_across="all"``
(i.e. across *processes*) while remote mode is ``parallelizable_across="batch"``
— one server per concurrent task.

Each worker is therefore a separate ``appworld serve environment --port P``
process, leased exclusively for the lifetime of one episode and returned to the
pool on ``/close``. Running the environment out-of-process has two more
benefits: the agent's arbitrary Python never executes inside the gym server, and
AppWorld's global time freeze cannot leak into gym's own timeouts or logging.

All HTTP goes through ``nemo_gym.server_utils.request`` (aiohttp), per the repo
rule against httpx in async paths.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from nemo_gym.server_utils import get_response_json, raise_for_status, request


logger = logging.getLogger(__name__)

# Grace period between SIGTERM and SIGKILL when shutting a worker down.
TERMINATE_TIMEOUT_SECS = 10.0

_PR_SET_PDEATHSIG = 1


class AppWorldWorkerError(RuntimeError):
    """A worker process failed to start, died, or returned an error."""


def _die_with_parent() -> None:  # pragma: no cover — runs post-fork in the child
    """Ask the kernel to SIGTERM this child if the gym server ever dies.

    Workers are put in their own session (so a Ctrl-C on the gym CLI's process
    group doesn't tear them down mid-request), which also means they would
    survive a server crash and linger holding ports. PR_SET_PDEATHSIG closes
    that gap.
    """
    if sys.platform != "linux":
        return
    import ctypes  # noqa: PLC0415 — child-only, keep it off the import path

    ctypes.CDLL("libc.so.6", use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)


def _free_port(preferred: int) -> int:
    """``preferred`` if bindable, else an ephemeral port chosen by the OS."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred))
        except OSError:
            sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class AppWorldWorker:
    """One ``appworld serve environment`` process and the HTTP calls against it."""

    def __init__(
        self,
        index: int,
        port: int,
        root: str,
        executable: str,
        log_fpath: Path,
        request_timeout_secs: float,
    ) -> None:
        self.index = index
        self.port = port
        self.root = root
        # ``appworld`` from the isolated venv built by setup_appworld.py — never
        # importable from this process's venv (conflicting pins).
        self.executable = executable
        self.log_fpath = log_fpath
        self.request_timeout_secs = request_timeout_secs
        self.process: Optional[subprocess.Popen[bytes]] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def spawn(self) -> None:
        env = os.environ.copy()
        env["APPWORLD_ROOT"] = self.root
        self.log_fpath.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered append so a crashed worker's traceback survives for triage.
        log_handle = open(self.log_fpath, "ab", buffering=0)  # noqa: SIM115 — closed with the process
        self.process = subprocess.Popen(
            [
                self.executable,
                "serve",
                "environment",
                "--port",
                str(self.port),
                "--root",
                self.root,
                "--no-show-usage",
            ],
            cwd=self.root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            preexec_fn=_die_with_parent,  # noqa: PLW1509 — see the docstring
        )
        self._log_handle = log_handle
        logger.info("appworld worker %d starting on port %d (log: %s)", self.index, self.port, self.log_fpath)

    async def _accepting_connections(self) -> bool:
        """Whether anything is listening yet — a plain TCP connect, no HTTP.

        Deliberately not ``request()``: gym's aiohttp helper retries
        ``ClientOSError`` (which connection-refused raises) in an *unbounded*
        loop, so probing a not-yet-listening port through it never returns and
        the readiness deadline below would never be evaluated.
        """
        try:
            _, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", self.port), timeout=2)
        except (OSError, asyncio.TimeoutError):
            return False
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        return True

    async def wait_until_ready(self, timeout_secs: float, poll_interval_secs: float = 0.25) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_secs
        while True:
            if not self.alive:
                raise AppWorldWorkerError(
                    f"appworld worker {self.index} exited during startup "
                    f"(exit code {self.process.returncode if self.process else 'n/a'}); see {self.log_fpath}"
                )
            if await self._accepting_connections():
                response = await request(method="GET", url=f"{self.url}/", timeout=10)
                await response.read()
                if response.ok:
                    logger.info("appworld worker %d ready on port %d", self.index, self.port)
                    return
            if asyncio.get_running_loop().time() >= deadline:
                raise AppWorldWorkerError(
                    f"appworld worker {self.index} did not become ready within {timeout_secs}s; see {self.log_fpath}"
                )
            await asyncio.sleep(poll_interval_secs)

    async def call(self, url_path: str, payload: Dict[str, Any]) -> Any:
        """POST to the worker and return the unwrapped ``output`` field.

        Every AppWorld environment-server route answers ``{"output": ...}``.

        The whole call is bounded by ``request_timeout_secs``: ``request()``
        retries connection errors forever, so a worker that dies mid-episode
        would otherwise hang the rollout instead of failing it.
        """
        if not self.alive:
            raise AppWorldWorkerError(f"appworld worker {self.index} is not running; see {self.log_fpath}")
        try:
            response = await asyncio.wait_for(
                request(
                    method="POST",
                    url=f"{self.url}{url_path}",
                    json=payload,
                    timeout=self.request_timeout_secs,
                ),
                timeout=self.request_timeout_secs,
            )
        except asyncio.TimeoutError as exc:
            raise AppWorldWorkerError(
                f"appworld worker {self.index} did not answer {url_path} within "
                f"{self.request_timeout_secs}s; see {self.log_fpath}"
            ) from exc
        await raise_for_status(response)
        body = await get_response_json(response)
        return body["output"] if isinstance(body, dict) and "output" in body else body

    async def terminate(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=TERMINATE_TIMEOUT_SECS)
            except asyncio.TimeoutError:
                process.kill()
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(process.wait)
        with contextlib.suppress(Exception):
            self._log_handle.close()


class AppWorldWorkerPool:
    """Lazily-started, fixed-size pool handing out exclusive worker leases."""

    def __init__(
        self,
        num_workers: int,
        port_start: int,
        root: str,
        executable: str,
        startup_timeout_secs: float,
        request_timeout_secs: float,
        log_dir: Optional[str] = None,
    ) -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        self.num_workers = num_workers
        self.port_start = port_start
        self.root = root
        self.executable = executable
        self.startup_timeout_secs = startup_timeout_secs
        self.request_timeout_secs = request_timeout_secs
        self.log_dir = Path(log_dir) if log_dir else Path(root) / ".nemo_gym_worker_logs"
        self.workers: List[AppWorldWorker] = []
        self._free: Optional[asyncio.Queue[AppWorldWorker]] = None
        self._start_lock = asyncio.Lock()

    async def start(self) -> None:
        """Spawn every worker and wait for all of them to answer. Idempotent."""
        async with self._start_lock:
            if self._free is not None:
                return
            workers = [
                AppWorldWorker(
                    index=index,
                    port=_free_port(self.port_start + index),
                    root=self.root,
                    executable=self.executable,
                    log_fpath=self.log_dir / f"worker_{index}.log",
                    request_timeout_secs=self.request_timeout_secs,
                )
                for index in range(self.num_workers)
            ]
            for worker in workers:
                worker.spawn()
            try:
                await asyncio.gather(*(w.wait_until_ready(self.startup_timeout_secs) for w in workers))
            except Exception:
                await asyncio.gather(*(w.terminate() for w in workers), return_exceptions=True)
                raise
            self.workers = workers
            free: asyncio.Queue[AppWorldWorker] = asyncio.Queue()
            for worker in workers:
                free.put_nowait(worker)
            self._free = free
            logger.info("appworld worker pool ready: %d workers, root=%s", self.num_workers, self.root)

    async def acquire(self) -> AppWorldWorker:
        """Lease a worker, blocking until one frees up. Respawns dead workers."""
        if self._free is None:
            await self.start()
        assert self._free is not None
        worker = await self._free.get()
        if not worker.alive:
            logger.warning("appworld worker %d died; respawning", worker.index)
            await self._respawn(worker)
        return worker

    async def release(self, worker: AppWorldWorker) -> None:
        assert self._free is not None, "release() before start()"
        self._free.put_nowait(worker)

    async def _respawn(self, worker: AppWorldWorker) -> None:
        await worker.terminate()
        worker.port = _free_port(worker.port)
        worker.spawn()
        await worker.wait_until_ready(self.startup_timeout_secs)

    async def stop(self) -> None:
        await asyncio.gather(*(worker.terminate() for worker in self.workers), return_exceptions=True)
        self.workers = []
        self._free = None
