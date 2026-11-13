# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


@pytest.mark.parametrize("value", [-1, 0.5, True, "2"])
def test_create_cap_requires_a_nonnegative_integer(value) -> None:
    with pytest.raises(ValueError, match="max_in_flight"):
        opensandbox_provider.OpenSandboxCreateConfig(max_in_flight=value)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_create_spacing_requires_a_finite_nonnegative_interval(value) -> None:
    with pytest.raises(ValueError, match="min_interval_s"):
        opensandbox_provider.OpenSandboxCreateConfig(min_interval_s=value)


async def test_create_spacing_applies_before_the_retry_budget(monkeypatch) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(create={"min_interval_s": 3})
    now = 10.0
    starts = []
    waits = []

    async def sleep(delay):
        nonlocal now
        waits.append(delay)
        now += delay

    async def create(spec):
        starts.append(now)
        return spec.image

    monkeypatch.setattr(opensandbox_provider.asyncio, "get_running_loop", lambda: SimpleNamespace(time=lambda: now))
    monkeypatch.setattr(opensandbox_provider.asyncio, "sleep", sleep)
    monkeypatch.setattr(provider, "_create_with_retries", create)
    assert [await provider.create(SandboxSpec(image="image")) for _ in range(3)] == ["image"] * 3
    assert starts == [10, 13, 16]
    assert waits == [3, 3]


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_create_cap_holds_waiters_and_releases_after_every_outcome(monkeypatch, outcome) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(create={"max_in_flight": 2})
    entered = [asyncio.Event() for _ in range(3)]
    release = asyncio.Event()
    calls = 0

    async def create(spec):
        nonlocal calls
        index = calls
        calls += 1
        entered[index].set()
        if index < 2:
            await release.wait()
        if outcome == "failure" and index == 0:
            raise RuntimeError("create failed")
        return spec.image

    monkeypatch.setattr(provider, "_create_with_retries", create)
    tasks = [asyncio.create_task(provider.create(SandboxSpec(image="image"))) for _ in range(3)]
    try:
        await asyncio.wait_for(asyncio.gather(entered[0].wait(), entered[1].wait()), 2)
        await asyncio.sleep(0)
        assert calls == 2
        assert not entered[2].is_set()
        if outcome == "cancel":
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            await asyncio.wait_for(entered[2].wait(), 2)
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert calls == 3
        assert results[1:] == ["image", "image"]
        if outcome == "failure":
            assert isinstance(results[0], RuntimeError)
        elif outcome == "success":
            assert results[0] == "image"
    finally:
        release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_create_limits_are_disabled_by_default(monkeypatch) -> None:
    provider = opensandbox_provider.OpenSandboxProvider()
    create = AsyncMock(return_value="created")
    monkeypatch.setattr(provider, "_create_with_retries", create)
    assert await provider.create(SandboxSpec(image="image")) == "created"
    assert create.await_count == 1
    assert provider._create_slots is None
    assert provider._last_create_start is None
