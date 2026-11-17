# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from dataclasses import dataclass, field

import pytest

from nemo_gym.web.browser_instance_pool import LocalBrowserInstancePoolProvider
from nemo_gym.web.browser_session import BrowserSessionError, BrowserSessionSpec
from tests.unit_tests.web.test_session_manager import _manager


@dataclass
class _Processes:
    """Control only blocking instance-process boundaries; the pool stays real."""

    launches: int = 0
    stops: int = 0
    healthy: bool = True
    live: bool = False
    blocked: str | None = None
    failure: str | None = None
    entered: threading.Event = field(default_factory=threading.Event)
    proceed: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)

    def boundary(self, name: str) -> None:
        if self.blocked == name:
            self.entered.set()
            try:
                if not self.proceed.wait(3):
                    raise TimeoutError(f"test did not unblock {name}")
            finally:
                self.blocked = None
                self.finished.set()
        if self.failure == name:
            raise RuntimeError(f"{name} failed")

    def launch(self, slot) -> None:
        self.launches += 1
        self.boundary("launch")
        self.live = self.healthy = True

    def stop(self, slot) -> None:
        self.stops += 1
        self.boundary("stop")
        self.live = False

    def health(self, slot) -> bool:
        self.boundary("health")
        return self.healthy


@pytest.fixture
async def pool(monkeypatch, tmp_path):
    processes = _Processes()
    provider = LocalBrowserInstancePoolProvider(size=1, port_base=42000, log_dir=str(tmp_path / "pool"))
    monkeypatch.setattr("shutil.which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(provider, "_launch", processes.launch)
    monkeypatch.setattr(provider, "_stop", processes.stop)
    monkeypatch.setattr(provider, "_idle_and_healthy", processes.health)
    await provider.start()
    try:
        yield provider, processes
    finally:
        processes.proceed.set()
        processes.failure = None
        if processes.entered.is_set():
            assert await asyncio.to_thread(processes.finished.wait, 3)
        await asyncio.wait_for(provider.close(), 3)


async def _turn() -> None:
    # Give an already-ready release/acquire coroutine its first event-loop turn.
    await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["health", "stop", "launch"])
@pytest.mark.parametrize("cancellation", ["direct", "manager_deadline"])
async def test_recycling_survives_waiter_cancellation(pool, tmp_path, stage, cancellation):
    provider, processes = pool
    handle = await provider.acquire(BrowserSessionSpec())
    processes.healthy = False
    processes.blocked = stage
    manager, _ = _manager(
        tmp_path / "manager", browser_session_provider=provider, browser_release_timeout_seconds=0.02
    )
    first = asyncio.create_task(
        provider.release(handle) if cancellation == "direct" else manager._release_browser_lease(handle)
    )
    retries = []
    try:
        assert await asyncio.to_thread(processes.entered.wait, 2)
        if cancellation == "direct":
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            assert await first is False
            assert manager._browser_leases_released == 0
        # The first manager deadline must not make a cleanup retry report success
        # while the real instance operation is still blocked.
        manager.config.browser_release_timeout_seconds = 2
        for _ in range(2):
            retries.append(asyncio.create_task(manager._release_browser_lease(handle)))
        done, _ = await asyncio.wait(retries, timeout=0.02)
        assert not done
        processes.proceed.set()
        assert await asyncio.wait_for(asyncio.gather(*retries), 2) == [True, True]
        assert processes.launches == 2 and processes.stops == 1
        replacement = await asyncio.wait_for(provider.acquire(BrowserSessionSpec()), 1)
        waiting = asyncio.create_task(provider.acquire(BrowserSessionSpec()))
        try:
            await _turn()
            assert not waiting.done()
            await provider.release(replacement)
            await asyncio.wait_for(waiting, 1)
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
    finally:
        processes.proceed.set()
        await asyncio.gather(first, *retries, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", [False, True])
async def test_old_lease_cannot_release_a_reused_slot(pool, busy):
    provider, processes = pool
    old = await provider.acquire(BrowserSessionSpec())
    await provider.release(old)
    current = await provider.acquire(BrowserSessionSpec())
    processes.healthy = not busy
    counts = processes.launches, processes.stops
    await provider.release(old)
    assert (processes.launches, processes.stops) == counts
    waiting = asyncio.create_task(provider.acquire(BrowserSessionSpec()))
    try:
        await _turn()
        assert not waiting.done()
        assert old.metadata["lease_generation"] != current.metadata["lease_generation"]
        await provider.release(current)
        next_handle = await asyncio.wait_for(waiting, 1)
        assert next_handle.metadata["lease_generation"] > current.metadata["lease_generation"]
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_waits_for_recycling_even_if_close_waiter_is_cancelled(pool):
    provider, processes = pool
    handle = await provider.acquire(BrowserSessionSpec())
    processes.healthy = False
    processes.blocked = "launch"
    releasing = asyncio.create_task(provider.release(handle))
    closing = None
    try:
        assert await asyncio.to_thread(processes.entered.wait, 2)
        closing = asyncio.create_task(provider.close())
        await _turn()
        assert not closing.done()
        assert processes.stops == 1  # only the recycling stop, not shutdown yet
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        processes.proceed.set()
        await asyncio.wait_for(releasing, 2)
        await asyncio.wait_for(provider.close(), 2)
        assert processes.launches == 2 and processes.stops == 2
        assert not processes.live
        with pytest.raises(BrowserSessionError):
            await provider.acquire(BrowserSessionSpec())
    finally:
        processes.proceed.set()
        await asyncio.gather(releasing, *(tuple() if closing is None else (closing,)), return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_replacement_is_retired_and_duplicate_release_does_not_retry(pool):
    provider, processes = pool
    handle = await provider.acquire(BrowserSessionSpec())
    processes.healthy = False
    processes.failure = "launch"
    await provider.release(handle)
    await provider.release(handle)
    assert processes.launches == 2 and processes.stops == 1
    assert not processes.live
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(provider.acquire(BrowserSessionSpec()), 0.02)


@pytest.mark.asyncio
async def test_failed_launch_rollback_retains_uncertain_cleanup(pool, monkeypatch):
    provider, processes = pool
    handle = await provider.acquire(BrowserSessionSpec())
    processes.healthy = False

    def partial_launch(slot):
        processes.launches += 1
        processes.live = True
        processes.failure = "stop"
        raise RuntimeError("replacement launch failed")

    # Keep the real launch wrapper: its rollback also fails, leaving an owned
    # partial process. This differs from a safely rolled-back failed launch.
    monkeypatch.setattr(provider, "_launch", LocalBrowserInstancePoolProvider._launch.__get__(provider))
    monkeypatch.setattr(provider, "_launch_slot", partial_launch)
    for _ in range(2):
        with pytest.raises(BrowserSessionError, match="rollback"):
            await provider.release(handle)
    assert processes.live
    assert processes.launches == 2 and processes.stops == 2
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(provider.acquire(BrowserSessionSpec()), 0.02)
    processes.failure = None
    await provider.close()
    assert not processes.live


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["health", "stop"])
async def test_unfinished_cleanup_failure_cannot_become_a_successful_duplicate(pool, stage):
    provider, processes = pool
    handle = await provider.acquire(BrowserSessionSpec())
    processes.healthy = False
    processes.failure = stage
    for _ in range(2):
        with pytest.raises(RuntimeError, match=f"{stage} failed"):
            await provider.release(handle)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(provider.acquire(BrowserSessionSpec()), 0.02)


@pytest.mark.asyncio
async def test_close_wakes_an_acquisition_waiting_for_a_recycling_slot(pool):
    provider, processes = pool
    handle = await provider.acquire(BrowserSessionSpec())
    processes.healthy = False
    processes.blocked = "launch"
    releasing = asyncio.create_task(provider.release(handle))
    acquiring = asyncio.create_task(provider.acquire(BrowserSessionSpec()))
    closing = None
    try:
        assert await asyncio.to_thread(processes.entered.wait, 2)
        closing = asyncio.create_task(provider.close())
        with pytest.raises(BrowserSessionError):
            await asyncio.wait_for(acquiring, 1)
        processes.proceed.set()
        await asyncio.wait_for(asyncio.gather(releasing, closing), 2)
    finally:
        processes.proceed.set()
        await asyncio.gather(
            releasing, acquiring, *(tuple() if closing is None else (closing,)), return_exceptions=True
        )
