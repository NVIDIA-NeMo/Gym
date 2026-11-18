# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import TypeAdapter, ValidationError

from nemo_gym.server_runtime.tool_access import (
    DirectHTTPToolAccess,
    MCPSSEConnection,
    MCPStdioConnection,
    MCPStreamableHTTPConnection,
    MCPToolAccess,
    ToolAccess,
    mcp_grants_over,
    mcp_server_urls,
    unusable_required_grants,
)


def test_tool_access_supports_named_direct_http_and_mcp_servers() -> None:
    accesses = TypeAdapter(list[ToolAccess]).validate_python(
        [
            {
                "kind": "direct_http",
                "name": "typed-tools",
                "required": True,
                "base_url": "http://resources:8000",
                "cookies": {"session": "value"},
            },
            {
                "kind": "mcp",
                "name": "task-tools",
                "required": True,
                "connection": {
                    "transport": "streamable_http",
                    "url": "http://resources:8000/mcp",
                    "headers": {"Authorization": "Bearer scoped"},
                },
            },
        ]
    )

    assert isinstance(accesses[0], DirectHTTPToolAccess)
    assert isinstance(accesses[1], MCPToolAccess)
    assert isinstance(accesses[1].connection, MCPStreamableHTTPConnection)


def test_mcp_http_connection_requires_an_absolute_url() -> None:
    with pytest.raises(ValidationError, match="valid URL"):
        MCPStreamableHTTPConnection(url="/mcp")


def test_tool_access_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MCPStreamableHTTPConnection(url="http://resources:8000/mcp", unknown=True)


def test_mcp_tool_access_reads_each_transport_by_its_tag() -> None:
    accesses = TypeAdapter(list[MCPToolAccess]).validate_python(
        [
            # A connection without a transport is a streamable HTTP one, as before the other transports.
            {"name": "http", "required": True, "connection": {"url": "http://tools:8000/mcp"}},
            {"name": "sse", "required": True, "connection": {"transport": "sse", "url": "http://tools:9000/sse"}},
            {
                "name": "stdio",
                "required": False,
                "connection": {"transport": "stdio", "command": "python3", "args": ["server.py"], "env": {"A": "1"}},
            },
        ]
    )

    assert [type(access.connection) for access in accesses] == [
        MCPStreamableHTTPConnection,
        MCPSSEConnection,
        MCPStdioConnection,
    ]
    stdio = accesses[2].connection
    assert (stdio.command, stdio.args, stdio.env) == ("python3", ["server.py"], {"A": "1"})
    # A grant survives the seed request's JSON round trip with its transport.
    assert [MCPToolAccess.model_validate(access.model_dump(mode="json")) for access in accesses] == accesses


@pytest.mark.parametrize(
    "connection",
    [
        {"transport": "websocket", "url": "ws://tools:8000"},
        {"transport": "stdio", "command": ""},
        {"transport": "stdio", "command": "python3", "url": "http://tools:8000/mcp"},
        {"transport": "sse", "url": "/sse"},
    ],
)
def test_mcp_connection_rejects_an_unknown_transport_or_another_transports_fields(connection) -> None:
    with pytest.raises(ValidationError):
        MCPToolAccess(name="tools", required=True, connection=connection)


def grant(name: str, connection: dict, *, required: bool = True) -> MCPToolAccess:
    return MCPToolAccess(name=name, required=required, connection=connection)


HTTP = {"url": "http://tools:8000/mcp"}
SSE = {"transport": "sse", "url": "http://memory:9000/sse"}
STDIO = {"transport": "stdio", "command": "python3"}


def test_a_harness_takes_the_mcp_grants_over_its_transports_and_refuses_the_required_others() -> None:
    accesses = [
        DirectHTTPToolAccess(name="typed", required=True, base_url="http://resources:8000"),
        DirectHTTPToolAccess(name="typed-optional", required=False, base_url="http://resources:8000"),
        grant("http", HTTP),
        grant("sse", SSE),
        grant("stdio", STDIO, required=False),
        grant("stdio-required", STDIO),
    ]

    assert [access.name for access in mcp_grants_over(accesses, {"streamable_http", "stdio"})] == [
        "http",
        "stdio",
        "stdio-required",
    ]
    assert unusable_required_grants(accesses, {"streamable_http", "stdio"}) == ["sse (sse)", "typed"]
    assert unusable_required_grants(accesses, {"streamable_http"}) == [
        "sse (sse)",
        "stdio-required (stdio)",
        "typed",
    ]


def test_only_network_mcp_servers_have_urls_to_allow() -> None:
    grants = [grant("http", HTTP), grant("stdio", STDIO), grant("sse", SSE)]

    assert mcp_server_urls(grants) == ["http://tools:8000/mcp", "http://memory:9000/sse"]
