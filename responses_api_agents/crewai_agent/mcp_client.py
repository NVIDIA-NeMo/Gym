# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP tool grants, reached from the task sandbox with the MCP SDK over streamable HTTP.

Runs in the sandbox runner, which has no Gym. A grant is the ``name``, ``url`` and ``headers`` of an
``MCPToolAccess``. Like CrewAI's own MCP tools, each listing and each call opens a fresh MCP session: concurrent calls
share no connection state, and stopping the crew leaves no connection to tear down. Gym's Resources Servers serve MCP
statelessly, so nothing is lost between calls.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from mcp import ClientSession, McpError
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, Tool


@asynccontextmanager
async def mcp_session(grant: dict[str, Any], *, timeout_seconds: float) -> AsyncIterator[ClientSession]:
    """An initialized MCP session with the grant's endpoint and headers, closed on exit."""
    async with httpx.AsyncClient(headers=grant["headers"], timeout=httpx.Timeout(timeout_seconds)) as client:
        async with streamable_http_client(grant["url"], http_client=client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def list_tools(grant: dict[str, Any], *, timeout_seconds: float) -> list[Tool]:
    async with mcp_session(grant, timeout_seconds=timeout_seconds) as session:
        tools: list[Tool] = []
        cursor = None
        while True:
            page = await session.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.nextCursor
            if cursor is None:
                return tools


async def call_tool(
    grant: dict[str, Any], name: str, arguments: dict[str, Any], *, timeout_seconds: float
) -> CallToolResult:
    """Call one tool. A JSON-RPC error from the server, such as invalid arguments, raises ``McpError``."""
    async with mcp_session(grant, timeout_seconds=timeout_seconds) as session:
        try:
            return await session.call_tool(name, arguments)
        except McpError as error:
            # Raised inside the session, it would leave the SDK's task groups wrapped in an ExceptionGroup.
            rejected = error
    raise rejected
