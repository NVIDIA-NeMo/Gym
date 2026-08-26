# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-house aiohttp-backed ``httpx.AsyncBaseTransport`` for the OpenSandbox SDK.

The OpenSandbox SDK is httpx-based with a pluggable transport, and its hottest
path is a long-lived SSE stream per executed command, so transport behavior on
streams is what matters most here. Compared to the ``httpx-aiohttp`` bridge it
replaces, this transport:

- Maps errors by phase with the real cause preserved. Mid-stream peer/proxy
  disconnects raise ``httpx.ReadError`` with a message that says the connection
  closed mid-response (aiohttp's own message for these is a framing-parser
  error, "Not enough data to satisfy transfer length header", which reads like
  a server bug); payload errors caused by a timeout raise ``httpx.ReadTimeout``;
  connection errors raise ``ConnectError`` before the response starts and
  ``ReadError`` after. The original aiohttp exception is always chained.
- Cannot mask an in-flight exception from its cleanup path. Closing an
  unfinished response hard-closes the connection instead of draining it, so
  the ``ReadTimeout``/``CancelledError`` that interrupted the stream is what
  callers see. Fully-consumed responses still release their connection for
  keep-alive reuse.
- Applies connect retries, which the bridge accepts but drops.
  ``max_keepalive_connections`` has no aiohttp equivalent and is deliberately
  not mapped: aiohttp's ``limit_per_host`` caps active concurrent connections
  per host (not idle ones), which would throttle all traffic to this
  transport's single target host. Idle connections are bounded by
  ``keepalive_expiry`` -> ``keepalive_timeout``, the only honest keepalive
  mapping (see the comment in ``_build_session``).

httpx client semantics are preserved: bodies stream incrementally, the
``read`` timeout applies per read operation (``read=None`` keeps an SSE stream
open indefinitely), redirects, cookies, and content-encoding stay owned by the
httpx client, never the transport.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable

import aiohttp
import httpx


def _chain_has_timeout(exc: BaseException, depth: int = 5) -> bool:
    """Return whether the exception's cause/context chain contains a timeout."""
    current: BaseException | None = exc
    for _ in range(depth):
        if current is None:
            return False
        if isinstance(current, (asyncio.TimeoutError, TimeoutError)):
            return True
        current = current.__cause__ or current.__context__
    return False


def _map_exception(exc: Exception, request: httpx.Request, *, in_body: bool) -> httpx.RequestError | None:
    """Translate an aiohttp exception into the equivalent httpx error, or None to re-raise as-is.

    ``in_body`` selects the taxonomy for ambiguous classes: the same connection
    error is a connect failure before the response starts but a read failure
    once the body is streaming.
    """
    detail = f"{type(exc).__name__}: {exc}"
    if isinstance(exc, aiohttp.ClientPayloadError):
        # Raised by aiohttp's payload parser when the body ends before the
        # declared length, i.e. the peer or a proxy dropped the connection
        # mid-response (or a read timed out mid-body, hence the chain check).
        if _chain_has_timeout(exc):
            return httpx.ReadTimeout(detail, request=request)
        return httpx.ReadError(
            f"connection closed before the response body completed "
            f"(peer or proxy dropped the connection mid-response); {detail}",
            request=request,
        )
    if isinstance(exc, aiohttp.ServerTimeoutError):
        if isinstance(exc, getattr(aiohttp, "SocketTimeoutError", ())):
            return httpx.ReadTimeout(detail, request=request)
        if isinstance(exc, getattr(aiohttp, "ConnectionTimeoutError", ())):
            return httpx.ConnectTimeout(detail, request=request)
        return httpx.TimeoutException(detail, request=request)
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        timeout_cls = httpx.ReadTimeout if in_body else httpx.ConnectTimeout
        return timeout_cls(detail, request=request)
    if isinstance(exc, aiohttp.ServerDisconnectedError):
        return httpx.RemoteProtocolError(
            f"server disconnected without completing the response ({detail})", request=request
        )
    if isinstance(exc, aiohttp.ClientProxyConnectionError):
        return httpx.ProxyError(detail, request=request)
    if isinstance(exc, aiohttp.ClientConnectorError):
        return httpx.ConnectError(detail, request=request)
    if isinstance(exc, (aiohttp.NonHttpUrlClientError, aiohttp.InvalidUrlClientError)):
        return httpx.UnsupportedProtocol(detail, request=request)
    if isinstance(exc, aiohttp.ClientConnectionError):
        connection_cls = httpx.ReadError if in_body else httpx.ConnectError
        return connection_cls(detail, request=request)
    if isinstance(exc, aiohttp.ClientError):
        return httpx.TransportError(detail, request=request)
    return None


class _AiohttpByteStream(httpx.AsyncByteStream):
    """Incremental byte stream over an aiohttp response body."""

    CHUNK_SIZE = 64 * 1024

    def __init__(self, response: aiohttp.ClientResponse, request: httpx.Request) -> None:
        self._response = response
        self._request = request

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._response.content.iter_chunked(self.CHUNK_SIZE):
                yield chunk
        except Exception as exc:
            mapped = _map_exception(exc, self._request, in_body=True)
            if mapped is not None:
                raise mapped from exc
            raise

    async def aclose(self) -> None:
        # A fully-read response gives its connection back for keep-alive
        # reuse; an abandoned or errored stream hard-closes it. Draining an
        # unfinished body here (``release()``) can itself raise a payload
        # error during cleanup and mask the exception that interrupted the
        # stream, so it must only happen at EOF.
        if self._response.content.is_eof():
            self._response.release()
        else:
            self._response.close()


class AiohttpTransport(httpx.AsyncBaseTransport):
    """httpx async transport that sends requests through a shared aiohttp ClientSession.

    Args:
        limits: httpx pool limits. ``max_connections`` caps in-flight
            connections (None = unlimited) and ``keepalive_expiry`` is the
            idle socket lifetime. ``max_keepalive_connections`` is ignored:
            aiohttp has no idle-pool-size knob (see ``_build_session``).
        force_close: Close the connection after every request instead of
            pooling it (``limits.keepalive_expiry`` is then ignored: aiohttp
            rejects a keepalive timeout on a force-closed connector).
        verify: Disable TLS certificate verification when False.
        retries: Retry a failed TCP connect this many times. Connect failures
            happen before anything is sent, so retrying cannot duplicate a
            request.
        client: Optional externally-owned ``aiohttp.ClientSession`` (or a
            zero-argument factory called lazily on the running loop). When
            given, ``limits`` and ``verify`` are ignored: the session's own
            connector governs pooling.
        close_client: Whether ``aclose()`` closes the session. Defaults to
            closing only a session this transport created.
    """

    def __init__(
        self,
        *,
        limits: httpx.Limits | None = None,
        force_close: bool = False,
        verify: bool = True,
        retries: int = 0,
        client: aiohttp.ClientSession | Callable[[], aiohttp.ClientSession] | None = None,
        close_client: bool | None = None,
    ) -> None:
        self._limits = limits or httpx.Limits(max_connections=100, max_keepalive_connections=20)
        self._force_close = force_close
        self._verify = verify
        self._retries = retries
        self._client_arg = client
        self._session: aiohttp.ClientSession | None = client if isinstance(client, aiohttp.ClientSession) else None
        self._close_client = (
            close_client if close_client is not None else not isinstance(client, aiohttp.ClientSession)
        )
        # Created lazily with the session: __init__ may run outside any event loop.
        self._session_lock: asyncio.Lock | None = None

    def _build_session(self) -> aiohttp.ClientSession:
        if callable(self._client_arg):
            return self._client_arg()
        # A force-closed connector rejects any keepalive timeout, so the
        # keepalive mapping only applies to pooling connectors.
        keepalive_kwargs = {} if self._force_close else {"keepalive_timeout": self._limits.keepalive_expiry}
        connector = aiohttp.TCPConnector(
            # aiohttp: limit=0 means unlimited; httpx: None means unlimited.
            limit=self._limits.max_connections or 0,
            # limits.max_keepalive_connections is deliberately NOT mapped.
            # httpx's knob caps IDLE reusable connections; aiohttp's
            # limit_per_host caps ACTIVE concurrent connections per host, and
            # this transport's traffic targets a single host, so mapping the
            # stock value (20) would throttle all concurrency to 20. aiohttp
            # has no idle-pool-size knob; idle connections are bounded by
            # keepalive_timeout expiry instead.
            limit_per_host=0,
            force_close=self._force_close,
            ssl=None if self._verify else False,
            **keepalive_kwargs,
        )
        return aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(),  # timeouts come per request from httpx
            cookie_jar=aiohttp.DummyCookieJar(),  # httpx owns cookies
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is not None:
            return self._session
        if self._session_lock is None:
            self._session_lock = asyncio.Lock()
        async with self._session_lock:
            if self._session is None:
                self._session = self._build_session()
        return self._session

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        session = await self._get_session()
        timeout = request.extensions.get("timeout", {})

        try:
            data: bytes | httpx.AsyncByteStream = request.content
        except httpx.RequestNotRead:
            data = request.stream  # aiohttp handles chunked framing itself
            request.headers.pop("transfer-encoding", None)

        for attempt in range(self._retries + 1):
            try:
                response = await session.request(
                    method=request.method,
                    url=str(request.url),
                    headers=request.headers.multi_items(),
                    data=data,
                    allow_redirects=False,  # httpx owns redirects
                    auto_decompress=False,  # httpx owns content-encoding
                    compress=False,
                    timeout=aiohttp.ClientTimeout(
                        sock_connect=timeout.get("connect"),
                        sock_read=timeout.get("read"),
                        connect=timeout.get("pool"),
                    ),
                )
                break
            except aiohttp.ClientConnectorError as exc:
                if attempt >= self._retries:
                    raise httpx.ConnectError(f"{type(exc).__name__}: {exc}", request=request) from exc
            except Exception as exc:
                mapped = _map_exception(exc, request, in_body=False)
                if mapped is not None:
                    raise mapped from exc
                raise

        return httpx.Response(
            status_code=response.status,
            headers=response.headers.items(),
            stream=_AiohttpByteStream(response, request),
            extensions={"http_version": b"HTTP/1.1"},
        )

    async def aclose(self) -> None:
        if self._close_client and self._session is not None:
            await self._session.close()
