# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the Gym HTTP pool alive across native Jupyter VLM query loops."""

import asyncio
import atexit
from threading import Lock, Thread

from nemo_gym import server_utils


_kernel_loop: asyncio.AbstractEventLoop | None = None
_kernel_thread: Thread | None = None
_kernel_loop_lock = Lock()


def _kernel_transport_loop() -> asyncio.AbstractEventLoop:
    """Keep Gym's process-wide aiohttp pool on one loop inside each kernel.

    Native VLMModule invokes asyncio.run in a worker thread for every query.
    Those caller loops close after each query; pooled HTTP connections must
    instead remain on this kernel-owned loop across calls and cells.
    """
    global _kernel_loop, _kernel_thread
    with _kernel_loop_lock:
        if _kernel_loop is None:
            _kernel_loop = asyncio.new_event_loop()
            _kernel_thread = Thread(target=_kernel_loop.run_forever, daemon=True, name="spatialclaw-gym-http")
            _kernel_thread.start()
        return _kernel_loop


def _shutdown_kernel_transport() -> None:
    global _kernel_loop, _kernel_thread
    loop, thread = _kernel_loop, _kernel_thread
    if loop is None or thread is None:
        return

    async def close_session() -> None:
        if server_utils.is_global_aiohttp_client_setup():
            await server_utils.get_global_aiohttp_client().close()

    try:
        asyncio.run_coroutine_threadsafe(close_session(), loop).result(timeout=5)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        if not thread.is_alive():
            loop.close()
        _kernel_loop = None
        _kernel_thread = None


atexit.register(_shutdown_kernel_transport)
