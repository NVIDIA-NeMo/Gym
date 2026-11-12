# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lease browser instances from a pool of local processes, one X display each.

Each slot is an Xvfb display plus a ``nemo_gym.web.browser_instance`` server
started with that display, so one resources server can hold as many headed
PyAutoGUI sessions as it has slots. A slot serves one session at a time; after
release it is reused only if its instance is idle and healthy, and is
restarted otherwise. Instances exit on their own when this process dies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path

from nemo_gym.web.browser_instance import DIRECT, TOKEN_ENV, TRANSPORT
from nemo_gym.web.browser_session import BrowserSessionError, BrowserSessionHandle, BrowserSessionSpec


LOG = logging.getLogger("nemo_gym.web.browser_instance_pool")


class _LaunchRollbackError(BrowserSessionError):
    """A partial instance could not be confirmed stopped after launch failed."""


@dataclass
class _Slot:
    index: int
    display: str
    port: int
    token: str = ""
    xvfb: subprocess.Popen[bytes] | None = None
    server: subprocess.Popen[bytes] | None = None
    tmpdir: Path | None = None
    leases: int = 0
    restarts: int = field(default=0)


class LocalBrowserInstancePoolProvider:
    """Serve leases from ``size`` local browser instances.

    The resources server's ``max_sessions`` should equal ``size``; the session
    manager rejects seeds beyond it before asking for a lease.
    """

    name = "local_instance_pool"

    def __init__(
        self,
        *,
        size: int,
        port_base: int,
        display_base: int = 100,
        host: str = "127.0.0.1",
        log_dir: str = "cache/visual-browser/instances",
        python: str | None = None,
        screen: str = "1920x1080x24",
        startup_timeout_seconds: float = 120.0,
        request_timeout_seconds: float = 600.0,
    ) -> None:
        if size < 1:
            raise ValueError("local_instance_pool size must be at least 1")
        self._slots = [_Slot(index=i, display=f":{display_base + i}", port=port_base + i) for i in range(size)]
        self._host = host
        self._log_dir = Path(log_dir).expanduser().resolve()
        self._python = python or sys.executable
        self._screen = screen
        self._startup_timeout = startup_timeout_seconds
        self._request_timeout = request_timeout_seconds
        self._idle: asyncio.Queue[_Slot] | None = None
        self._leased: dict[int, _Slot] = {}
        self._lifecycle_lock = asyncio.Lock()
        self._startup_cleanup: asyncio.Task[None] | None = None
        self._recycling: dict[tuple[int, int], asyncio.Task[None]] = {}
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False

    async def start(self) -> None:
        closing = self._close_task
        if closing is not None:
            await asyncio.shield(closing)
        async with self._lifecycle_lock:
            if closing is not None and self._close_task is closing:
                self._close_task = None
                self._closing = False
            if self._closing:
                raise BrowserSessionError("local_instance_pool is closing")
            if self._startup_cleanup is not None:
                await asyncio.shield(self._startup_cleanup)
                self._startup_cleanup = None
            if self._idle is not None:
                return
            for tool in ("Xvfb", "xdpyinfo"):
                if shutil.which(tool) is None:
                    raise BrowserSessionError(f"{tool} is required for local browser instances")
            self._log_dir.mkdir(parents=True, exist_ok=True)
            launches = asyncio.gather(
                *(asyncio.to_thread(self._launch, slot) for slot in self._slots), return_exceptions=True
            )
            try:
                results = await asyncio.shield(launches)
                for result in results:
                    if isinstance(result, BaseException):
                        raise result
            except BaseException:
                # Threads may still be launching after an RPC is cancelled.
                # Own rollback independently, and make retry/close wait for it.
                self._startup_cleanup = asyncio.create_task(self._rollback_start(launches))
                await asyncio.shield(self._startup_cleanup)
                raise
            self._idle = asyncio.Queue()
            for slot in self._slots:
                self._idle.put_nowait(slot)
            LOG.info("event=browser_instance_pool_ready size=%d", len(self._slots))

    async def _rollback_start(self, launches: asyncio.Future) -> None:
        await launches
        results = await asyncio.gather(
            *(asyncio.to_thread(self._stop, slot) for slot in self._slots), return_exceptions=True
        )
        self._idle = None
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def acquire(self, spec: BrowserSessionSpec) -> BrowserSessionHandle:
        idle = self._idle
        if idle is None or self._closing:
            raise BrowserSessionError("local_instance_pool has not been started")
        try:
            slot = await idle.get()
        except asyncio.QueueShutDown as exc:
            raise BrowserSessionError("local_instance_pool is closing") from exc
        if self._closing or self._idle is not idle:
            raise BrowserSessionError("local_instance_pool is closing")
        slot.leases += 1
        self._leased[slot.index] = slot
        LOG.info(
            "event=browser_instance_leased slot=%d display=%s session=%s lease_count=%d",
            slot.index,
            slot.display,
            spec.metadata.get("rollout_session_id", ""),
            slot.leases,
        )
        return BrowserSessionHandle(
            session_id=spec.metadata.get("rollout_session_id"),
            provider_name=self.name,
            transport=TRANSPORT,
            endpoint=f"http://{self._host}:{slot.port}",
            metadata={
                "slot": slot.index,
                "lease_generation": slot.leases,
                "display": slot.display,
                "token": slot.token,
                "request_timeout_seconds": self._request_timeout,
            },
            owner_pid=os.getpid(),
        )

    async def release(self, handle: BrowserSessionHandle) -> None:
        slot = self._leased.get(int(handle.metadata["slot"]))
        if slot is None or handle.metadata.get("lease_generation") != slot.leases:
            return
        key = slot.index, slot.leases
        recycling = self._recycling.get(key)
        if recycling is None:
            if self._closing:
                assert self._close_task is not None
                await asyncio.shield(self._close_task)
                return
            assert self._idle is not None
            # RPC cancellation must not discard a thread that is still checking,
            # stopping or replacing the instance. Retries join this same owner.
            recycling = asyncio.create_task(self._recycle(slot, key, self._idle))
            self._recycling[key] = recycling
        await asyncio.shield(recycling)

    async def _recycle(self, slot: _Slot, key: tuple[int, int], idle: asyncio.Queue[_Slot]) -> None:
        try:
            healthy = await asyncio.to_thread(self._idle_and_healthy, slot)
            if not healthy:
                LOG.warning("event=browser_instance_restart slot=%d display=%s", slot.index, slot.display)
                await asyncio.to_thread(self._stop, slot)
                slot.restarts += 1
                try:
                    await asyncio.to_thread(self._launch, slot)
                except _LaunchRollbackError:
                    raise
                except Exception:
                    # _launch has rolled back. Retire this slot, rather than
                    # lending a failed replacement or starting it twice.
                    LOG.exception("event=browser_instance_restart_failed slot=%d", slot.index)
                    self._leased.pop(slot.index)
                    self._recycling.pop(key)
                    return
        except Exception:
            # Retain the failed task and its lease until close can stop the
            # instance. A duplicate release must not report uncertain cleanup
            # as success merely because the first waiter stopped awaiting it.
            LOG.exception("event=browser_instance_recycling_failed slot=%d", slot.index)
            raise
        self._leased.pop(slot.index)
        self._recycling.pop(key)
        if not self._closing and self._idle is idle:
            idle.put_nowait(slot)

    async def close(self) -> None:
        if self._close_task is not None and self._close_task.done() and self._close_task.exception() is not None:
            self._close_task = None
        if self._close_task is None:
            self._closing = True
            if self._idle is not None:
                self._idle.shutdown(immediate=True)
            self._close_task = asyncio.create_task(self._close())
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        async with self._lifecycle_lock:
            if self._idle is not None:
                self._idle.shutdown(immediate=True)
            if self._startup_cleanup is not None:
                await asyncio.shield(self._startup_cleanup)
                self._startup_cleanup = None
            if self._recycling:
                await asyncio.gather(*tuple(self._recycling.values()), return_exceptions=True)
            results = await asyncio.gather(
                *(asyncio.to_thread(self._stop, slot) for slot in self._slots), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            self._idle = None
            self._leased.clear()
            self._recycling.clear()

    def _launch(self, slot: _Slot) -> None:
        try:
            self._launch_slot(slot)
        except BaseException:
            try:
                self._stop(slot)
            except Exception as exc:
                raise _LaunchRollbackError(f"browser instance rollback failed for slot {slot.index}") from exc
            raise

    def _launch_slot(self, slot: _Slot) -> None:
        slot.token = secrets.token_hex(16)
        slot.tmpdir = Path(tempfile.mkdtemp(prefix=f"browser-instance-{slot.index}-"))
        with (self._log_dir / f"slot-{slot.index:03d}.log").open("ab") as log:
            slot.xvfb = subprocess.Popen(
                ["Xvfb", slot.display, "-screen", "0", self._screen, "-ac", "-nolisten", "tcp"],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            env = dict(os.environ)
            env.update(
                {
                    "DISPLAY": slot.display,
                    "XDG_SESSION_TYPE": "x11",
                    "TMPDIR": str(slot.tmpdir),
                    TOKEN_ENV: slot.token,
                }
            )
            env.pop("WAYLAND_DISPLAY", None)
            # Xvfb runs with -ac, but python-xlib (PyAutoGUI) still opens an
            # Xauthority file; give each display its own empty one.
            xauthority = slot.tmpdir / "Xauthority"
            xauthority.touch(mode=0o600)
            env["XAUTHORITY"] = str(xauthority)
            deadline = time.monotonic() + self._startup_timeout
            while subprocess.run(["xdpyinfo", "-display", slot.display], env=env, capture_output=True).returncode:
                if slot.xvfb.poll() is not None or time.monotonic() > deadline:
                    raise BrowserSessionError(f"Xvfb {slot.display} did not start (slot {slot.index})")
                time.sleep(0.2)
            slot.server = subprocess.Popen(
                [
                    self._python,
                    "-m",
                    "nemo_gym.web.browser_instance",
                    "--host",
                    self._host,
                    "--port",
                    str(slot.port),
                    "--parent-pid",
                    str(os.getpid()),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while not self._idle_and_healthy(slot):
                if slot.server.poll() is not None or time.monotonic() > deadline:
                    raise BrowserSessionError(
                        f"browser instance on port {slot.port} did not start (slot {slot.index})"
                    )
                time.sleep(0.2)
            LOG.info("event=browser_instance_started slot=%d display=%s port=%d", slot.index, slot.display, slot.port)

    def _idle_and_healthy(self, slot: _Slot) -> bool:
        if slot.server is None or slot.server.poll() is not None or slot.xvfb is None or slot.xvfb.poll() is not None:
            return False
        try:
            with DIRECT.open(f"http://{self._host}:{slot.port}/healthz", timeout=10) as response:
                health = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
            return False
        return bool(health.get("ok")) and not health.get("busy")

    @staticmethod
    def _stop(slot: _Slot) -> None:
        for process, sig in ((slot.server, signal.SIGTERM), (slot.xvfb, signal.SIGTERM)):
            if process is None or process.poll() is not None:
                continue
            try:
                os.killpg(process.pid, sig)
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
        slot.server = slot.xvfb = None
        if slot.tmpdir is not None:
            shutil.rmtree(slot.tmpdir, ignore_errors=True)
            slot.tmpdir = None
