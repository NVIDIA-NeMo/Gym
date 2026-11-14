# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


@pytest.fixture(autouse=True)
def poll_waits_timed_by_sleep(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., Any]]:
    """Time the OpenSandbox command poll's wait with ``asyncio.sleep``, which these tests replace to control time.

    The provider waits for a wakeup under ``asyncio.timeout``, which runs on the event loop's own timer and so
    ignores a replaced ``asyncio.sleep``. Here the provider's own wait still runs, with an interval that never
    ends, and the end of ``asyncio.sleep(interval_s)`` wakes it instead. Yields the provider's unchanged wait.
    """
    wait_command_poll = opensandbox_provider.OpenSandboxProvider._wait_command_poll

    async def wait(self: Any, poll: Any, *, interval_s: float) -> None:
        async def interval() -> None:
            await asyncio.sleep(interval_s)
            poll.wakeup.set()

        # Not started when a wakeup was delivered before the wait: the provider then returns without waiting.
        timer = asyncio.ensure_future(interval())
        try:
            await wait_command_poll(self, poll, interval_s=float("inf"))
        finally:
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)

    monkeypatch.setattr(opensandbox_provider.OpenSandboxProvider, "_wait_command_poll", wait)
    yield wait_command_poll
