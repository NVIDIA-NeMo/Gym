# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from jsonschema import Draft202012Validator
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


# MCP grant connections as JSON, each with whether a grant carrying it is valid. Parsing and the exported JSON
# schema must agree on every one, so a client that checks a seed body against the schema sends what an agent takes.
MCP_CONNECTIONS = [
    # A connection without a transport is a streamable HTTP one, as before the other transports existed.
    pytest.param({"url": "http://tools:8000/mcp"}, True, id="untagged-http"),
    pytest.param(
        {"transport": "streamable_http", "url": "http://tools:8000/mcp", "headers": {"A": "1"}},
        True,
        id="streamable_http",
    ),
    pytest.param({"transport": "sse", "url": "http://tools:9000/sse"}, True, id="sse"),
    pytest.param(
        {"transport": "stdio", "command": "python3", "args": ["server.py"], "env": {"A": "1"}}, True, id="stdio"
    ),
    pytest.param({"command": "python3"}, False, id="untagged-stdio"),
    # Harbor's spellings of streamable HTTP are not grant transports: the Harbor server maps them to it.
    pytest.param({"transport": "streamable-http", "url": "http://tools:8000/mcp"}, False, id="harbor-streamable-http"),
    pytest.param({"transport": "http", "url": "http://tools:8000/mcp"}, False, id="harbor-http"),
    pytest.param({"transport": "websocket", "url": "ws://tools:8000"}, False, id="unknown-transport"),
    pytest.param({"transport": None, "url": "http://tools:8000/mcp"}, False, id="null-transport"),
    pytest.param({"url": "http://tools:8000/mcp", "command": "python3"}, False, id="untagged-url-and-command"),
    pytest.param(
        {"transport": "stdio", "command": "python3", "url": "http://tools:8000/mcp"}, False, id="stdio-with-url"
    ),
    pytest.param(
        {"transport": "sse", "url": "http://tools:9000/sse", "command": "python3"}, False, id="sse-with-command"
    ),
    pytest.param(
        {"transport": "streamable_http", "url": "http://tools:8000/mcp", "args": []}, False, id="http-with-args"
    ),
]


def _parses(grant: dict) -> bool:
    try:
        MCPToolAccess.model_validate(grant)
    except ValidationError:
        return False
    return True


@pytest.mark.parametrize(("connection", "valid"), MCP_CONNECTIONS)
def test_the_grant_json_schema_accepts_exactly_the_connections_parsing_accepts(connection: dict, valid: bool) -> None:
    grant = {"kind": "mcp", "name": "tools", "required": True, "connection": connection}
    schema = MCPToolAccess.model_json_schema()
    Draft202012Validator.check_schema(schema)

    assert _parses(grant) is valid
    assert Draft202012Validator(schema).is_valid(grant) is valid


def test_a_grant_built_with_default_transports_reaches_json_with_its_transport() -> None:
    # The Harbor server builds its grants this way; a JSON body must name every transport but streamable HTTP.
    grants = [
        MCPToolAccess(name="http", required=True, connection=MCPStreamableHTTPConnection(url="http://tools:8000/mcp")),
        MCPToolAccess(name="sse", required=True, connection=MCPSSEConnection(url="http://tools:9000/sse")),
        MCPToolAccess(name="stdio", required=True, connection=MCPStdioConnection(command="python3")),
    ]
    dumped = [access.model_dump(mode="json") for access in grants]

    assert [access["connection"]["transport"] for access in dumped] == ["streamable_http", "sse", "stdio"]
    for mode in ("validation", "serialization"):
        validator = Draft202012Validator(MCPToolAccess.model_json_schema(mode=mode))
        assert all(validator.is_valid(access) for access in dumped)
    assert [MCPToolAccess.model_validate(access) for access in dumped] == grants


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
