# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP tool grants, reached with the MCP SDK over Gym's shared aiohttp client.

The SDK speaks streamable HTTP through an ``httpx.AsyncClient``. Its transport here hands every request to
``nemo_gym.client.request``, so no httpx connection pool runs in the Agent Server. Like CrewAI's own MCP tools, each
listing and each call opens a fresh MCP session: concurrent calls share no connection state, and close has no
connection to tear down. Gym's Resources Servers serve MCP statelessly, so nothing is lost between calls.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import httpx
from mcp import ClientSession, McpError
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, Tool

from nemo_gym.client import request as http_request
from nemo_gym.server_runtime import MCPToolAccess


class _ResponseBody(httpx.AsyncByteStream):
    def __init__(self, response: aiohttp.ClientResponse) -> None:
        self._response = response

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._response.content.iter_chunked(16 * 1024):
            yield chunk

    async def aclose(self) -> None:
        self._response.release()


class GymHTTPTransport(httpx.AsyncBaseTransport):
    """Send an httpx client's requests through Gym's shared aiohttp client."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        timeout = request.extensions.get("timeout", {})
        response = await http_request(
            method=request.method,
            url=str(request.url),
            headers=request.headers.multi_items(),
            data=await request.aread() or None,
            allow_redirects=False,
            # httpx decodes the body itself from the headers passed through below.
            auto_decompress=False,
            # MCP requests are not idempotent: a dropped connection fails the call instead of sending it again.
            _resend_after_disconnect=False,
            timeout=aiohttp.ClientTimeout(
                total=None, sock_connect=timeout.get("connect"), sock_read=timeout.get("read")
            ),
        )
        return httpx.Response(
            status_code=response.status,
            headers=response.raw_headers,
            stream=_ResponseBody(response),
            request=request,
        )


@asynccontextmanager
async def mcp_session(access: MCPToolAccess, *, timeout_seconds: float) -> AsyncIterator[ClientSession]:
    """An initialized MCP session with the grant's endpoint and headers, closed on exit."""
    async with httpx.AsyncClient(
        transport=GymHTTPTransport(),
        headers=access.connection.headers,
        timeout=httpx.Timeout(timeout_seconds),
    ) as client:
        async with streamable_http_client(str(access.connection.url), http_client=client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def list_tools(access: MCPToolAccess, *, timeout_seconds: float) -> list[Tool]:
    async with mcp_session(access, timeout_seconds=timeout_seconds) as session:
        tools: list[Tool] = []
        cursor = None
        while True:
            page = await session.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.nextCursor
            if cursor is None:
                return tools


async def call_tool(
    access: MCPToolAccess, name: str, arguments: dict[str, Any], *, timeout_seconds: float
) -> CallToolResult:
    """Call one tool. A JSON-RPC error from the server, such as invalid arguments, raises ``McpError``."""
    async with mcp_session(access, timeout_seconds=timeout_seconds) as session:
        try:
            return await session.call_tool(name, arguments)
        except McpError as error:
            # Raised inside the session, it would leave the SDK's task groups wrapped in an ExceptionGroup.
            rejected = error
    raise rejected
