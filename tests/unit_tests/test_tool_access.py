# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import TypeAdapter, ValidationError

from nemo_gym.tool_access import (
    DirectHTTPToolAccess,
    MCPStdioConnection,
    MCPStreamableHTTPConnection,
    MCPToolAccess,
    ToolAccess,
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
            {
                "kind": "mcp",
                "name": "memory",
                "required": False,
                "connection": {
                    "transport": "stdio",
                    "command": "memory-server",
                    "args": ["--stdio"],
                    "env": {"SCOPE": "rollout"},
                },
            },
        ]
    )

    assert isinstance(accesses[0], DirectHTTPToolAccess)
    assert isinstance(accesses[1], MCPToolAccess)
    assert isinstance(accesses[1].connection, MCPStreamableHTTPConnection)
    assert isinstance(accesses[2].connection, MCPStdioConnection)


def test_mcp_http_connection_requires_an_absolute_url() -> None:
    with pytest.raises(ValidationError, match="valid URL"):
        MCPStreamableHTTPConnection(url="/mcp")


def test_tool_access_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MCPStdioConnection(command="server", unknown=True)
