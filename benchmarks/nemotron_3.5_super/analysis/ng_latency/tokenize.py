# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Count tokens of extracted text with the live engine's ``/tokenize`` endpoint (post-run only)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Optional

import aiohttp


TokenCounter = Callable[[list[str]], list[Optional[int]]]


def tokenize_url_from_policy_url(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return f"{base}/tokenize"


async def _count(session: aiohttp.ClientSession, url: str, text: str, semaphore: asyncio.Semaphore) -> Optional[int]:
    if text == "":
        return 0
    async with semaphore:
        try:
            async with session.post(url, json={"prompt": text, "add_special_tokens": False}) as response:
                if response.status != 200:
                    return None
                payload = await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError):
            return None
    count = payload.get("count") if isinstance(payload, dict) else None
    return int(count) if isinstance(count, int) else None


async def count_tokens_async(tokenize_url: str, texts: list[str], *, concurrency: int = 16) -> list[Optional[int]]:
    semaphore = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        return list(await asyncio.gather(*(_count(session, tokenize_url, text, semaphore) for text in texts)))


def count_tokens(tokenize_url: str, texts: list[str], *, concurrency: int = 16) -> list[Optional[int]]:
    """Synchronous entry point for the post-run CLI (never used inside a running server)."""
    return asyncio.run(count_tokens_async(tokenize_url, texts, concurrency=concurrency))
