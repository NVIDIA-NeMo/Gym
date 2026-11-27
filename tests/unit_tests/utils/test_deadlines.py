# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from time import monotonic

import pytest

from nemo_gym.utils.deadlines import monotonic_timeout
from tests.helpers.event_loop_clock import run_on_clock_ahead_loop


def test_future_deadline_does_not_fire_on_an_ahead_clock():
    async def main():
        async with monotonic_timeout(monotonic() + 120) as timer:
            await asyncio.sleep(0.01)
        return timer.expired()

    assert run_on_clock_ahead_loop(main()) is False


def test_past_deadline_fires_on_an_ahead_clock():
    async def main():
        async with monotonic_timeout(monotonic() - 1):
            await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        run_on_clock_ahead_loop(main())


def test_near_deadline_fires_after_its_remaining_time():
    async def main():
        started = monotonic()
        with pytest.raises(TimeoutError):
            async with monotonic_timeout(started + 0.05):
                await asyncio.sleep(10)
        return monotonic() - started

    assert 0.04 <= run_on_clock_ahead_loop(main()) < 5


def test_none_never_expires():
    async def main():
        async with monotonic_timeout(None) as timer:
            await asyncio.sleep(0.01)
        return timer.when()

    assert run_on_clock_ahead_loop(main()) is None
