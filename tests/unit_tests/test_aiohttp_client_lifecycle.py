# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import queue
import threading

import pytest

import nemo_gym.server_utils
from nemo_gym.server_utils import close_global_aiohttp_client, global_aiohttp_client_exit


class _LoopBoundClient:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.closed = False
        self.close_calls = 0
        self.close_loop: asyncio.AbstractEventLoop | None = None

    async def close(self) -> None:
        self.close_calls += 1
        self.close_loop = asyncio.get_running_loop()
        self.closed = True


def _set_global_client(client: _LoopBoundClient) -> None:
    nemo_gym.server_utils._GLOBAL_AIOHTTP_CLIENT = client


def test_exit_closes_on_stopped_owner_loop_clears_and_is_idempotent() -> None:
    owner_loop = asyncio.new_event_loop()
    client = _LoopBoundClient(owner_loop)
    _set_global_client(client)
    try:
        global_aiohttp_client_exit()
        global_aiohttp_client_exit()

        assert client.closed
        assert client.close_calls == 1
        assert client.close_loop is owner_loop
        assert not nemo_gym.server_utils.is_global_aiohttp_client_setup()
    finally:
        owner_loop.close()


def test_exit_schedules_close_on_running_owner_loop_in_another_thread() -> None:
    ready: queue.Queue[tuple[asyncio.AbstractEventLoop, _LoopBoundClient]] = queue.Queue()

    def run_owner_loop() -> None:
        owner_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(owner_loop)
        ready.put((owner_loop, _LoopBoundClient(owner_loop)))
        owner_loop.run_forever()
        owner_loop.close()

    owner_thread = threading.Thread(target=run_owner_loop)
    owner_thread.start()
    owner_loop, client = ready.get(timeout=2)
    _set_global_client(client)
    try:
        global_aiohttp_client_exit()

        assert client.closed
        assert client.close_calls == 1
        assert client.close_loop is owner_loop
        assert not nemo_gym.server_utils.is_global_aiohttp_client_setup()
    finally:
        owner_loop.call_soon_threadsafe(owner_loop.stop)
        owner_thread.join(timeout=2)
    assert not owner_thread.is_alive()


def test_exit_with_closed_owner_loop_clears_without_foreign_loop_close() -> None:
    owner_loop = asyncio.new_event_loop()
    client = _LoopBoundClient(owner_loop)
    owner_loop.close()
    _set_global_client(client)

    global_aiohttp_client_exit()

    assert not client.closed
    assert client.close_calls == 0
    assert not nemo_gym.server_utils.is_global_aiohttp_client_setup()


def test_explicit_async_close_clears_and_is_idempotent() -> None:
    async def scenario() -> None:
        owner_loop = asyncio.get_running_loop()
        client = _LoopBoundClient(owner_loop)
        _set_global_client(client)

        await close_global_aiohttp_client()
        await close_global_aiohttp_client()

        assert client.closed
        assert client.close_calls == 1
        assert client.close_loop is owner_loop
        assert not nemo_gym.server_utils.is_global_aiohttp_client_setup()

    asyncio.run(scenario())


def test_explicit_async_close_rejects_a_foreign_loop() -> None:
    owner_loop = asyncio.new_event_loop()
    client = _LoopBoundClient(owner_loop)
    _set_global_client(client)
    try:
        with pytest.raises(RuntimeError, match="owning event loop"):
            asyncio.run(close_global_aiohttp_client())

        assert not client.closed
        assert nemo_gym.server_utils.is_global_aiohttp_client_setup()
        owner_loop.run_until_complete(close_global_aiohttp_client())
    finally:
        if not client.closed:
            owner_loop.run_until_complete(client.close())
        nemo_gym.server_utils._GLOBAL_AIOHTTP_CLIENT = None
        owner_loop.close()
