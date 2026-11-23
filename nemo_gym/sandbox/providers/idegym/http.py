# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The HTTP transport the IdeGYM client sends its requests through.

The transport stays on httpx-aiohttp's own ``AiohttpTransport`` rather than
:class:`nemo_gym.sandbox.providers._http_transport.GymAiohttpTransport`: the SDK runs
on a private loop thread, and Gym's global aiohttp client is re-created per running
loop, so sharing it would tear it away from the main loop.
"""

import asyncio
import logging

import aiohttp
import httpx

from nemo_gym.sandbox.providers._http_transport import _retry_after_seconds
from nemo_gym.sandbox.providers.idegym.config import IdeGymConnectionConfig, TransportBackend


LOGGER = logging.getLogger(__name__)

_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD"})
_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})


def build_transport(connection: IdeGymConnectionConfig) -> httpx.AsyncBaseTransport:
    """Build the pool-bounded transport for the configured backend, with the provider's retries."""
    # Retries live in RetryingTransport alone: httpx-aiohttp ignores its `retries`
    # argument, and two layers retrying would multiply the attempts.
    inner: httpx.AsyncBaseTransport | None = None
    if connection.transport_backend == TransportBackend.AIOHTTP:
        try:
            from httpx_aiohttp import AiohttpTransport
        except ImportError:
            LOGGER.warning(
                "connection.transport_backend=aiohttp requested but httpx-aiohttp is not installed; "
                "falling back to the httpx transport"
            )
        else:
            # httpx-aiohttp passes None on as "keep aiohttp's default cap of 100", so no cap
            # is spelled 0, as aiohttp spells it. aiohttp has no keepalive-count knob, so
            # max_keepalive_connections has nothing to apply to on this backend.
            max_connections = 0 if connection.max_connections is None else connection.max_connections
            limits = httpx.Limits(max_connections=max_connections, keepalive_expiry=connection.keepalive_expiry_s)
            inner = AiohttpTransport(limits=limits)
    if inner is None:
        limits = httpx.Limits(
            max_connections=connection.max_connections,
            max_keepalive_connections=connection.max_keepalive_connections,
            keepalive_expiry=connection.keepalive_expiry_s,
        )
        inner = httpx.AsyncHTTPTransport(limits=limits, retries=0)
    return RetryingTransport(
        inner,
        connect_retries=connection.connect_retries,
        poll_retries=connection.poll_retries,
        retry_delay_s=connection.retry_delay_s,
    )


def _never_sent(exc: httpx.TransportError) -> bool:
    if not isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return False
    cause = exc.__cause__
    # httpx-aiohttp maps every aiohttp.ClientConnectionError to ConnectTimeout, including
    # a server that disconnects after reading the request, so its cause decides.
    if isinstance(cause, aiohttp.ClientError):
        return isinstance(cause, (aiohttp.ClientConnectorError, aiohttp.ConnectionTimeoutError))
    return True


def _replayable(request: httpx.Request) -> bool:
    try:
        request.content
    except httpx.RequestNotRead:
        return False
    return True


class RetryingTransport(httpx.AsyncBaseTransport):
    """Retry the requests that are safe to send again.

    Any method is retried when its connection never opened, up to ``connect_retries``.
    GET and HEAD are also retried on any other transport error and on a 429/502/503/504,
    up to ``poll_retries``: the SDK sends each operation-status poll once, so one flaky
    poll would otherwise abort a long exec or start. A POST, PUT or DELETE that reached
    the server is never sent again, since that could run a command twice or start a
    second server.
    """

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport,
        *,
        connect_retries: int,
        poll_retries: int,
        retry_delay_s: float,
    ) -> None:
        self._transport = transport
        self._connect_retries = connect_retries
        self._poll_retries = poll_retries
        self._retry_delay_s = retry_delay_s

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        idempotent = request.method in _IDEMPOTENT_METHODS
        # A streamed body has been consumed by the first attempt and cannot be sent again.
        replayable = _replayable(request)
        connect_used = 0
        poll_used = 0
        while True:
            attempt = connect_used + poll_used
            try:
                response = await self._transport.handle_async_request(request)
            except httpx.TransportError as exc:
                if replayable and _never_sent(exc) and connect_used < self._connect_retries:
                    connect_used += 1
                elif replayable and idempotent and poll_used < self._poll_retries:
                    poll_used += 1
                else:
                    raise
                delay = self._retry_delay_s * 2**attempt
                reason = type(exc).__name__
            else:
                if response.status_code not in _RETRYABLE_STATUSES or not idempotent:
                    return response
                if poll_used >= self._poll_retries:
                    return response
                poll_used += 1
                try:
                    delay = _retry_after_seconds(
                        response.headers.get("Retry-After"), fallback=self._retry_delay_s * 2**attempt
                    )
                finally:
                    # Free the pool slot before sleeping, including on cancellation.
                    await response.aclose()
                reason = f"HTTP {response.status_code}"
            LOGGER.warning(
                "Retrying IdeGYM request; method=%s path=%s reason=%s retry=%s sleep_s=%s",
                request.method,
                request.url.path,
                reason,
                attempt + 1,
                delay,
            )
            await asyncio.sleep(delay)

    async def aclose(self) -> None:
        await self._transport.aclose()
