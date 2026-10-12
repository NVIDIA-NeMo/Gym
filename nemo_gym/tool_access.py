# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-visible tool access contracts."""

from typing import Annotated, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field


class DirectHTTPToolAccess(BaseModel):
    """Connect a trusted Python agent to typed tool routes."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["direct_http"] = "direct_http"
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    required: bool
    base_url: AnyHttpUrl
    cookies: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)


class MCPStreamableHTTPConnection(BaseModel):
    """Connect to an MCP server over an absolute HTTP endpoint."""

    model_config = ConfigDict(extra="forbid")

    transport: Literal["streamable_http"] = "streamable_http"
    url: AnyHttpUrl
    headers: dict[str, str] = Field(default_factory=dict)


class MCPToolAccess(BaseModel):
    """Describe one named MCP server that an agent adapter must configure.

    Names are session-wide identifiers. Agent session requests reject duplicate names,
    and adapters must also reject collisions with their static tool configuration.
    Failure to establish a required server fails session seeding; an optional server may
    be skipped. On agent-session close, adapters disconnect clients without stopping the
    remote server.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["mcp"] = "mcp"
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    required: bool
    connection: MCPStreamableHTTPConnection


ToolAccess = Annotated[
    DirectHTTPToolAccess | MCPToolAccess,
    Field(discriminator="kind"),
]
