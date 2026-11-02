# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Count HTTP retries per request and per rollout attempt, across Gym server hops.

A retry is counted when another transport attempt of the same request starts, never where the failure
is caught: a failure that exhausts the retry budget raises without starting an attempt, so it adds
nothing. ``nemo_gym.server_utils`` calls :func:`record_http_retry_started` from that one place.

Each incoming Gym request and each dispatched rollout attempt opens a scope. Retries started inside the
scope, including in concurrent child tasks, add to it. Every Gym response reports its scope's total in
``x-ng-http-retry-count``, and a caller adds the value from each completed internal response to its own
scope, so the rollout driver sees the retries of the whole call tree below one ``/run``.

The total is a lower bound. Retries made after the response headers are sent (streaming or background
work), and retries inside a transport attempt that failed before returning headers, cannot reach the
caller. A server without this header reports nothing. The caller still counts its own retries.
"""

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional

from starlette.types import ASGIApp, Message, Receive, Scope, Send


logger = logging.getLogger(__name__)

HTTP_RETRY_COUNT_HEADER = "x-ng-http-retry-count"


@dataclass
class HttpRetryCount:
    """Retries started in one scope, plus those its completed downstream Gym calls reported."""

    total: int = 0


@contextmanager
def http_retry_scope() -> Iterator[HttpRetryCount]:
    """Count the retries of one incoming request or one rollout attempt, including its child tasks."""
    count = HttpRetryCount()
    token = _CURRENT.set(count)
    try:
        yield count
    finally:
        _CURRENT.reset(token)


def record_http_retry_started() -> None:
    """Count one retry: call when another transport attempt of a request starts, not when one fails."""
    count = _CURRENT.get()
    if count is not None:
        count.total += 1


_MAX_COUNT_DIGITS = 18


def record_downstream_http_retries(response: object) -> None:
    """Add the retries a completed Gym response reports to the current scope.

    Call only for responses from Gym servers: an external endpoint must not add arbitrary numbers to a
    rollout's count. A missing or malformed header adds nothing, and so does a response object without
    headers: counting must never fail a request that succeeded.
    """
    count = _CURRENT.get()
    if count is None:
        return
    headers = getattr(response, "headers", None)
    raw = headers.get(HTTP_RETRY_COUNT_HEADER) if isinstance(headers, Mapping) else None
    if raw is None:
        return
    # Only a plain nonnegative decimal of bounded length counts; signs, spaces, non-ASCII digits and values too long
    # to be a retry count are rejected, so the conversion below cannot fail.
    if not (isinstance(raw, str) and raw.isascii() and raw.isdigit() and len(raw) <= _MAX_COUNT_DIGITS):
        logger.debug("Ignoring malformed %s header value %r", HTTP_RETRY_COUNT_HEADER, raw)
        return
    count.total += int(raw)


class HttpRetryCountMiddleware:
    """Open a retry scope per HTTP request and report its total in the response headers.

    The header replaces any value the application set, such as one copied from a proxied downstream
    response: that response's count already reached this scope when it was received.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        with http_retry_scope() as count:

            async def send_with_retry_count(message: Message) -> None:
                if message["type"] == "http.response.start":
                    message = {**message, "headers": _with_retry_count(message.get("headers", []), count.total)}
                await send(message)

            await self.app(scope, receive, send_with_retry_count)


_CURRENT: ContextVar[Optional[HttpRetryCount]] = ContextVar("nemo_gym_http_retry_count", default=None)


def _with_retry_count(headers: list[tuple[bytes, bytes]], total: int) -> list[tuple[bytes, bytes]]:
    name = HTTP_RETRY_COUNT_HEADER.encode()
    return [(key, value) for key, value in headers if key.lower() != name] + [(name, str(total).encode())]
