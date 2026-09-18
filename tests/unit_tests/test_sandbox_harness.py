# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from nemo_gym.sandbox.harness import WorkerBridge


pytestmark = pytest.mark.sandbox


async def test_factory_error_reaches_worker_without_waiting_for_episode_timeout():
    """A synchronous adapter error must reach the worker instead of leaving it blocked."""
    bridge = WorkerBridge()

    def invalid_action():
        raise KeyError("command")

    worker = asyncio.create_task(asyncio.to_thread(bridge.call, invalid_action))
    try:
        with pytest.raises(KeyError, match="command"):
            await asyncio.wait_for(asyncio.shield(worker), timeout=1)
    finally:
        await bridge.aclose()
        await asyncio.gather(worker, return_exceptions=True)
