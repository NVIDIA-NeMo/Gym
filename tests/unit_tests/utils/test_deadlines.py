# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from time import monotonic

import pytest

from nemo_gym.utils.deadlines import monotonic_timeout


class AheadClockLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock runs far ahead of ``time.monotonic()``, as uvloop's does on macOS."""

    def time(self) -> float:
        return super().time() + 1_000_000


def run_ahead(coroutine):
    loop = AheadClockLoop()
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()


def test_timeout_at_with_a_monotonic_deadline_fires_at_once_on_an_ahead_clock():
    async def main():
        async with asyncio.timeout_at(monotonic() + 120):
            await asyncio.sleep(0.01)

    with pytest.raises(TimeoutError):
        run_ahead(main())


def test_future_deadline_does_not_fire_on_an_ahead_clock():
    async def main():
        async with monotonic_timeout(monotonic() + 120) as timer:
            await asyncio.sleep(0.01)
        return timer.expired()

    assert run_ahead(main()) is False


def test_past_deadline_fires_on_an_ahead_clock():
    async def main():
        async with monotonic_timeout(monotonic() - 1):
            await asyncio.sleep(10)

    with pytest.raises(TimeoutError):
        run_ahead(main())


def test_near_deadline_fires_after_its_remaining_time():
    async def main():
        started = monotonic()
        with pytest.raises(TimeoutError):
            async with monotonic_timeout(started + 0.05):
                await asyncio.sleep(10)
        return monotonic() - started

    assert 0.04 <= run_ahead(main()) < 5


def test_none_never_expires():
    async def main():
        async with monotonic_timeout(None) as timer:
            await asyncio.sleep(0.01)
        return timer.when()

    assert run_ahead(main()) is None
