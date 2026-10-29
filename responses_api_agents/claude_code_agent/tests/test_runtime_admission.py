# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio

import pytest

from responses_api_agents.claude_code_agent import runtime


async def test_more_seeds_than_slots_finish_without_spending_install_deadline_in_queue(monkeypatch):
    monkeypatch.setattr(runtime, "_RUNTIME_SETUP_SLOTS", asyncio.Semaphore(4))
    active = maximum = 0
    entered = []

    async def seed(i):
        nonlocal active, maximum
        async with runtime.runtime_setup_admission(timeout=2):
            async with asyncio.timeout(0.1):
                active += 1
                maximum = max(maximum, active)
                entered.append(i)
                await asyncio.sleep(0.02)
                active -= 1

    await asyncio.gather(*(seed(i) for i in range(96)))
    assert maximum == 4 and active == 0
    assert entered == list(range(96))
    assert runtime._RUNTIME_SETUP_SLOTS._value == 4


async def test_expired_queue_does_not_release_unowned_slot(monkeypatch):
    slots = asyncio.Semaphore(1)
    monkeypatch.setattr(runtime, "_RUNTIME_SETUP_SLOTS", slots)
    async with runtime.runtime_setup_admission(timeout=1):
        with pytest.raises(TimeoutError):
            async with runtime.runtime_setup_admission(timeout=0.01):
                pytest.fail("Queue must not admit another setup")
        assert slots._value == 0
    assert slots._value == 1


async def test_cancelled_install_releases_capacity(monkeypatch):
    slots = asyncio.Semaphore(1)
    monkeypatch.setattr(runtime, "_RUNTIME_SETUP_SLOTS", slots)
    entered = asyncio.Event()

    async def setup():
        async with runtime.runtime_setup_admission(timeout=1):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(setup())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with runtime.runtime_setup_admission(timeout=0.01):
        assert slots._value == 0
    assert slots._value == 1
