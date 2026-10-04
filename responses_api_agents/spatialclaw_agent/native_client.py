# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use Gym's HTTP transport beneath the original SpatialClaw LLM client."""

from types import SimpleNamespace
from typing import Any

from openai.types.chat import ChatCompletion

from nemo_gym.openai_utils import NeMoGymAsyncOpenAI


class GymChatClient:
    """The non-streaming Chat Completions interface consumed by SpatialClaw."""

    def __init__(self, *, base_url: str, api_key: str) -> None:
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
        def _get_client(self, endpoint: str) -> GymChatClient:
            client = self._client_pool.get(endpoint)
            if client is None:
                client = GymChatClient(base_url=endpoint, api_key=self._api_key)
                self._client_pool[endpoint] = client
            return client

    return GymLLMClient(config)
