# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Force lock contention at cohort transitions instead of relying on timing luck."""

import asyncio
import gc
import inspect

import pytest
from fastapi import HTTPException

import resources_servers.genrm_compare.app as genrm
from resources_servers.genrm_compare.tests.test_cohort_lifecycle import member


class ObservedLock(asyncio.Lock):
    def __init__(self):
        super().__init__()
        self.arrivals = asyncio.Queue()

    async def acquire(self):
        if self.locked():
            self.arrivals.put_nowait(asyncio.current_task())
        return await super().acquire()


@pytest.mark.parametrize("cancel_replacement", [False, True])
async def test_attempt_retires_old_group_when_last_member_wins_lock(server, cancel_replacement):
    started = asyncio.Event()
    release = asyncio.Event()

    async def compare(*args, **kwargs):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # Even an already-produced backend reply must not publish old rewards.
            return [99.0, 99.0], {}, [], []
        return [3.0, 3.0], {}, [], []

    server._run_compare = compare
    first = asyncio.create_task(server.verify(member(0)))
    await asyncio.sleep(0)
    old = next(iter(server._verify_cohorts.values()))
    old.lock = lock = ObservedLock()
    tasks = [first]
    await lock.acquire()
    try:
        last = asyncio.create_task(server.verify(member(1)))
        tasks.append(last)
        assert await asyncio.wait_for(lock.arrivals.get(), 1) is last
        replacement = asyncio.create_task(server.verify(member(0, attempt=1)))
        tasks.append(replacement)
        assert await asyncio.wait_for(lock.arrivals.get(), 1) is replacement
        if cancel_replacement:
            replacement.cancel()
            await asyncio.gather(replacement, return_exceptions=True)
            # A request cancelled before retirement must not commit its attempt.
            assert server._latest_group_attempts["group"].latest_attempt == 0
        lock.release()
        await asyncio.wait_for(started.wait(), 1)
        if cancel_replacement:
            replacement = asyncio.create_task(server.verify(member(0, attempt=1)))
            tasks.append(replacement)
        results = await asyncio.wait_for(asyncio.gather(first, last, return_exceptions=True), 1)
        assert all(isinstance(r, HTTPException) and r.status_code == 503 for r in results)
        assert old.phase == "failed" and not old.rewards
        release.set()
        second = await server.verify(member(1, attempt=1))
        assert (await replacement).reward == second.reward == 3.0
        with pytest.raises(HTTPException) as error:
            await server.verify(member(0))
        assert error.value.status_code == 409
    finally:
        if lock.locked():
            lock.release()
        release.set()
        await server.aclose()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("disconnect", [False, True])
async def test_failed_evaluation_start_releases_peers_even_with_disconnect(server, monkeypatch, disconnect):
    original_start = asyncio.create_task
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled = []
    loop.set_exception_handler(lambda loop, context: unhandled.append(context))
    rejected = []
    waiters = []
    holder_release = asyncio.Event()

    first = original_start(server.verify(member(0)))
    await asyncio.sleep(0)
    cohort = next(iter(server._verify_cohorts.values()))
    cohort.lock = lock = ObservedLock()
    await lock.acquire()
    last = original_start(server.verify(member(1)))
    assert await asyncio.wait_for(lock.arrivals.get(), 1) is last

    async def hold_lock():
        async with lock:
            await holder_release.wait()

    holder = original_start(hold_lock())
    assert await asyncio.wait_for(lock.arrivals.get(), 1) is holder

    def start(coro, **kwargs):
        if kwargs.get("name", "").startswith("genrm-cohort-evaluation"):
            rejected.append(coro)
            waiters.extend(w for m in cohort.members.values() for w in m.waiters)
            if disconnect:
                loop.call_soon(asyncio.current_task().cancel)
            raise RuntimeError("evaluation task factory failed")
        return original_start(coro, **kwargs)

    monkeypatch.setattr(genrm.asyncio, "create_task", start)
    try:
        lock.release()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        holder_release.set()
        await holder
        results = await asyncio.wait_for(asyncio.gather(first, last, return_exceptions=True), 0.5)
        assert isinstance(results[0], HTTPException) and results[0].status_code == 503
        assert "evaluation task factory failed" in results[0].detail
        assert isinstance(results[1], (HTTPException, asyncio.CancelledError))
        assert cohort.phase == "failed" and not cohort.rewards
        assert cohort.collection_timeout_task is None and cohort.evaluation_task is None
        assert all(w.done() for w in waiters)
        assert all(not m.waiters for m in cohort.members.values())
        assert len(rejected) == 1 and inspect.getcoroutinestate(rejected[0]) == inspect.CORO_CLOSED
        waiters.clear()
        results.clear()
        del first, last
        gc.collect()
        await asyncio.sleep(0)
        assert not unhandled
    finally:
        holder_release.set()
        await holder
        await server.aclose()
        loop.set_exception_handler(previous_handler)


async def test_failed_collection_start_fails_registered_member(server, monkeypatch):
    original_start = asyncio.create_task
    rejected = []

    def start(coro, **kwargs):
        if kwargs.get("name", "").startswith("genrm-cohort-collection"):
            rejected.append(coro)
            raise RuntimeError("collection task factory failed")
        return original_start(coro, **kwargs)

    monkeypatch.setattr(genrm.asyncio, "create_task", start)
    with pytest.raises(HTTPException) as error:
        await server.verify(member(0))
    assert error.value.status_code == 503 and "collection task factory failed" in error.value.detail
    cohort = next(iter(server._verify_cohorts.values()))
    assert cohort.phase == "failed" and all(not m.waiters for m in cohort.members.values())
    assert not server._cohort_tasks
    assert len(rejected) == 1 and inspect.getcoroutinestate(rejected[0]) == inspect.CORO_CLOSED
