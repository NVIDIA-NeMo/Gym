# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible model clients used by offline seed generation."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Protocol

from openai import AsyncOpenAI

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
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 425, 429} or (isinstance(status_code, int) and status_code >= 500):
        return True
    return isinstance(exc, (TimeoutError, ConnectionError, asyncio.TimeoutError)) or exc.__class__.__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }


class OpenAIChatGenerator:
    def __init__(self, config: ModelRoleConfig) -> None:
        self.config = config
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(f"missing API key environment variable: {config.api_key_env}")
        self.client = AsyncOpenAI(api_key=api_key, base_url=config.base_url, timeout=config.timeout_seconds)
        self._semaphore = asyncio.Semaphore(config.concurrency)

    async def close(self) -> None:
        await self.client.close()

    async def generate(self, messages: list[dict[str, str]]) -> GeneratedText:
        delay = self.config.retry_initial_backoff_seconds
        for attempt in range(1, self.config.provider_attempts + 1):
            try:
                async with self._semaphore:
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        **self.config.sampling,
                    )
                text = response.choices[0].message.content
                if text is None:
                    raise ValueError("model returned no text content")
                return GeneratedText(
                    text=text,
                    raw_response=response.model_dump(mode="json"),
                    provider_attempts=attempt,
                )
            except Exception as exc:
                if attempt >= self.config.provider_attempts or not _is_retryable_provider_error(exc):
                    raise ProviderGenerationError(str(exc)) from exc
                if delay:
                    await asyncio.sleep(delay)
                delay = min(
                    max(delay * 2, self.config.retry_initial_backoff_seconds), self.config.retry_max_backoff_seconds
                )
        raise AssertionError("provider retry loop exhausted without returning or raising")
