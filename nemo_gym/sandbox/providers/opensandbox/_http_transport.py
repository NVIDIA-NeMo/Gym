# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve ended-sandbox responses that SDK 0.1.15 otherwise discards."""

import httpx


class OpenSandboxHttpTransport(httpx.AsyncBaseTransport):
    """Retain bounded command-status HTTP 410 bodies for the SDK error converter."""

    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self._transport = transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._transport.handle_async_request(request)
        # SDK 0.1.15's generated status endpoint has no 410 schema and discards
        # its body. Its HTTPStatusError converter does retain that same JSON.
        # Scope this bridge to status reads; no command is submitted or replayed.
        _, separator, execution_id = request.url.path.rpartition("/command/status/")
        if (
            response.status_code != 410
            or request.method != "GET"
            or not separator
            or not execution_id
            or "/" in execution_id
        ):
            return response
        body = bytearray()
        try:
            async for chunk in response.aiter_bytes():
                body.extend(chunk[: 65537 - len(body)])
                if len(body) > 65536:
                    break
        finally:
            await response.aclose()
        # Oversized bodies supply no termination evidence. The observed HTTP
        # status survives, but a truncated JSON document is never interpreted.
        content = bytes(body) if len(body) <= 65536 else b""
        headers = dict(response.headers)
        headers.pop("content-encoding", None)
        headers.pop("content-length", None)
        observed = httpx.Response(410, headers=headers, content=content, request=request)
        raise httpx.HTTPStatusError("OpenSandbox command status returned HTTP 410", request=request, response=observed)

    async def aclose(self) -> None:
        await self._transport.aclose()
