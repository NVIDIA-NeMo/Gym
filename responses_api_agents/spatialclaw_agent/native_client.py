# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use Gym's HTTP transport beneath the original SpatialClaw LLM client."""

import asyncio
import atexit
from threading import Lock, Thread
from types import SimpleNamespace
from typing import Any

from openai.types.chat import ChatCompletion

from nemo_gym import server_utils
from nemo_gym.openai_utils import NeMoGymAsyncOpenAI


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


class GymChatClient:
    """The non-streaming Chat Completions interface consumed by SpatialClaw."""

    def __init__(self, *, base_url: str, api_key: str, kernel_transport: bool = False) -> None:
        self._kernel_transport = kernel_transport
        self._client = NeMoGymAsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            max_connection_retries=0,
            max_http_attempts=1,
        )
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs: Any) -> ChatCompletion:
        """Preserve SDK extra-body semantics while using Gym's aiohttp transport."""
        extra_body = kwargs.pop("extra_body", None) or {}
        if kwargs.get("stream"):
            raise ValueError("SpatialClaw's Gym transport requires non-streaming completions")
        payload = kwargs | extra_body
        if self._kernel_transport:
            future = asyncio.run_coroutine_threadsafe(
                self._client.create_chat_completion(**payload), _kernel_transport_loop()
            )
            result = await asyncio.wrap_future(future)
        else:
            result = await self._client.create_chat_completion(**payload)
        return ChatCompletion.model_validate(result)

    async def close(self) -> None:
        """Gym owns the shared aiohttp session and closes it at server shutdown."""


def create_native_client(config: Any) -> Any:
    """Retain upstream prompting, response parsing, retries, and usage accounting.

    The local subclass is cloudpickle-compatible: SpatialClaw injects its LLM
    client into each Jupyter kernel for the original VLM query tools.
    """
    from spatial_agent.llm.client import LLMClient

    class GymLLMClient(LLMClient):
        def __setstate__(self, state: dict[str, Any]) -> None:
            super().__setstate__(state)
            self._gym_kernel_transport = True

        def _get_client(self, endpoint: str) -> GymChatClient:
            client = self._client_pool.get(endpoint)
            if client is None:
                client = GymChatClient(
                    base_url=endpoint,
                    api_key=self._api_key,
                    kernel_transport=getattr(self, "_gym_kernel_transport", False),
                )
                self._client_pool[endpoint] = client
            return client

    return GymLLMClient(config)
