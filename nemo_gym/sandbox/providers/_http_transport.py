# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt sandbox SDK HTTPX requests to Gym's shared aiohttp client."""

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager

import aiohttp
import httpx

from nemo_gym import server_utils


@contextmanager
def _map_aiohttp_exceptions() -> Iterator[None]:
    # SDK retry policies expect HTTPX errors, including while consuming streams.
    # More specific subclasses must precede their base classes.
    try:
        yield
    except aiohttp.SocketTimeoutError as exc:
        raise httpx.ReadTimeout(str(exc)) from exc
    except aiohttp.ConnectionTimeoutError as exc:
        raise httpx.ConnectTimeout(str(exc)) from exc
    except TimeoutError as exc:
        raise httpx.TimeoutException(str(exc)) from exc
    except (aiohttp.ClientProxyConnectionError, aiohttp.ClientHttpProxyError) as exc:
        raise httpx.ProxyError(str(exc)) from exc
    except (aiohttp.ClientPayloadError, aiohttp.ServerDisconnectedError) as exc:
        raise httpx.ReadError(str(exc)) from exc
    except aiohttp.ClientConnectionError as exc:
        raise httpx.ConnectError(str(exc)) from exc
    except (aiohttp.InvalidURL, aiohttp.NonHttpUrlClientError) as exc:
        raise httpx.UnsupportedProtocol(str(exc)) from exc


class _AiohttpResponseStream(httpx.AsyncByteStream):
    def __init__(self, response: aiohttp.ClientResponse) -> None:
        self._response = response

    async def __aiter__(self) -> AsyncIterator[bytes]:
        with _map_aiohttp_exceptions():
            async for chunk in self._response.content.iter_chunked(16 * 1024):
                yield chunk

    async def aclose(self) -> None:
        with _map_aiohttp_exceptions():
            self._response.release()
            await self._response.wait_for_close()


class GymAiohttpTransport(httpx.AsyncBaseTransport):
    """Use Gym's connection pool without transferring session ownership to an SDK."""

    def __init__(self, *, verify: bool = True, proxy: httpx.Proxy | None = None) -> None:
        self.verify = verify
        self.proxy = proxy

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            data = request.content or None
        except httpx.RequestNotRead:
            data = request.stream
            # aiohttp frames streamed bodies itself.
            request.headers.pop("transfer-encoding", None)

        timeout = request.extensions.get("timeout", {})
        with _map_aiohttp_exceptions():
            response = await server_utils.request(
                method=request.method,
                url=str(request.url),
                # Provider/SDK retries own the operation semantics. In particular,
                # a failed command submission must not be replayed by this adapter.
                _max_connection_retries=1,
                headers=request.headers.multi_items(),
                data=data,
                allow_redirects=False,
                auto_decompress=False,
                compress=False,
                # aiohttp includes ssl in its pool key. A fresh SSLContext per
                # adapter would prevent connection reuse between providers.
                ssl=self.verify,
                server_hostname=request.extensions.get("sni_hostname"),
                proxy=str(self.proxy.url) if self.proxy else None,
                proxy_auth=aiohttp.BasicAuth(*self.proxy.auth) if self.proxy and self.proxy.auth else None,
                proxy_headers=self.proxy.headers if self.proxy else None,
                timeout=aiohttp.ClientTimeout(
                    total=None,
                    connect=timeout.get("pool"),
                    sock_connect=timeout.get("connect"),
                    sock_read=timeout.get("read"),
                ),
            )

        try:
            extensions = {"http_version": f"HTTP/{response.version.major}.{response.version.minor}".encode()}
            if response.reason:
                extensions["reason_phrase"] = response.reason.encode()
            return httpx.Response(
                status_code=response.status,
                headers=response.raw_headers,
                stream=_AiohttpResponseStream(response),
                request=request,
                extensions=extensions,
            )
        except BaseException:
            response.close()
            raise

    async def aclose(self) -> None:
        # Gym owns the session. Closing one sandbox must not close other sandboxes'
        # connections or the HTTP clients used by model/resources servers.
        pass
