# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym-native model client used by synthetic tool-use generation servers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status
from resources_servers.synthetic_tool_use.common.models import ModelRoleConfig


@dataclass(frozen=True)
class GeneratedText:
    text: str
    raw_response: dict[str, Any]
    provider_attempts: int


class AsyncTextGenerator(Protocol):
    async def generate(self, messages: list[dict[str, str]]) -> GeneratedText: ...


class ProviderGenerationError(RuntimeError):
    pass


def _is_retryable_provider_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status", getattr(exc, "status_code", None))
    if status_code in {408, 409, 425, 429} or (isinstance(status_code, int) and status_code >= 500):
        return True
    return isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)) or exc.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "ClientConnectionError",
        "ClientOSError",
        "RateLimitError",
        "ServerDisconnectedError",
    }


def _response_text(response: NeMoGymChatCompletion) -> str:
    if not response.choices:
        raise ValueError("model returned no choices")
    text = response.choices[0].message.content
    if text is None:
        raise ValueError("model returned no text content")
    return text


class GymModelGenerator:
    """Adapt a configured Gym model server to the stage generator protocol."""

    def __init__(
        self,
        *,
        server_client: ServerClient,
        model_server: ModelServerRef,
        role: ModelRoleConfig,
    ) -> None:
        self.server_client = server_client
        self.model_server = model_server
        self.role = role
        self._semaphore = asyncio.Semaphore(role.concurrency)

    async def generate(self, messages: list[dict[str, str]]) -> GeneratedText:
        delay = self.role.retry_initial_backoff_seconds
        for attempt in range(1, self.role.provider_attempts + 1):
            try:
                request = NeMoGymChatCompletionCreateParamsNonStreaming(messages=messages, **self.role.sampling)
                async with self._semaphore:
                    http_response = await self.server_client.post(
                        server_name=self.model_server.name,
                        url_path="/v1/chat/completions",
                        json=request,
                    )
                    await raise_for_status(http_response)
                    response = NeMoGymChatCompletion.model_validate(await get_response_json(http_response))
                return GeneratedText(
                    text=_response_text(response),
                    raw_response=response.model_dump(mode="json"),
                    provider_attempts=attempt,
                )
            except Exception as exc:
                if attempt >= self.role.provider_attempts or not _is_retryable_provider_error(exc):
                    raise ProviderGenerationError(str(exc)) from exc
                if delay:
                    await asyncio.sleep(delay)
                delay = min(
                    max(delay * 2, self.role.retry_initial_backoff_seconds), self.role.retry_max_backoff_seconds
                )
        raise AssertionError("provider retry loop exhausted without returning or raising")
