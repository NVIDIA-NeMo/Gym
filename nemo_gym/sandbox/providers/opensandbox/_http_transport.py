# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt the OpenSandbox SDK's HTTPX requests to Gym's shared aiohttp client."""

import httpx
from aiohttp import ClientTimeout
from httpx_aiohttp.transport import AiohttpResponseStream, map_aiohttp_exceptions

from nemo_gym import server_utils


class GymAiohttpTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, verify: bool) -> None:
        self.verify = verify

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            data = request.content or None
        except httpx.RequestNotRead:
            data = request.stream
            # aiohttp frames streamed bodies itself.
            request.headers.pop("transfer-encoding", None)

        timeout = request.extensions.get("timeout", {})
        with map_aiohttp_exceptions():
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
                timeout=ClientTimeout(
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
                stream=AiohttpResponseStream(response),
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
