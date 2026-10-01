# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded policy-request retries shared by the native and Harbor LAB loops."""

import ast
import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

import aiohttp
from tenacity import AsyncRetrying, before_sleep_log, retry_if_exception, stop_after_attempt, wait_exponential


LOG = logging.getLogger(__name__)


def _http_status(exc: aiohttp.ClientResponseError) -> int:
    # Gym's exception middleware wraps upstream HTTP errors in a JSON string
    # with status 500. Preserve numeric provider codes when they are available.
    if exc.status == 500:
        try:
            payload = json.loads(getattr(exc, "response_content", b""))
            if isinstance(payload, str) and payload.startswith("Hit an exception in "):
                payload = json.loads(ast.literal_eval(payload.split(" calling an inner server: ", 1)[1]))
            error = payload.get("error") if isinstance(payload, dict) else None
            code = int(error["code"]) if isinstance(error, dict) else 0
            if 400 <= code < 600:
                return code
        except (ValueError, TypeError, KeyError, IndexError, SyntaxError):
            pass
    return exc.status


def _retryable(exc: BaseException, *, retry_404: bool) -> bool:
    if isinstance(exc, aiohttp.ClientResponseError):
        status = _http_status(exc)
        return status in {408, 429} or 500 <= status < 600 or (retry_404 and status == 404)
    return isinstance(exc, (TimeoutError, ConnectionError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError))


async def retry_model_request[T](
    call: Callable[[], Awaitable[T]], *, timeout_seconds: float | None, retry_404: bool = False
) -> T:
    """Make at most three attempts at a model request, never replaying tools.

    Each attempt retains its own request timeout. Use Harbor's Tenacity pattern
    with exponential backoff (1 then 2 seconds); cancellation propagates.
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_if_exception(lambda exc: _retryable(exc, retry_404=retry_404)),
        before_sleep=before_sleep_log(LOG, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            return await asyncio.wait_for(call(), timeout=timeout_seconds)
    raise AssertionError("Retry loop exited without a result or exception")  # pragma: no cover
