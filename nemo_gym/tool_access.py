# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-visible tool access contracts."""

from typing import Annotated, Any, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator


class DirectHTTPToolAccess(BaseModel):
    """Connect a trusted Python agent to typed tool routes."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["direct_http"] = "direct_http"
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    required: bool
    base_url: AnyHttpUrl
    cookies: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    batch_path: str | None = Field(default=None, pattern=r"^/")
    tool_call_context: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_execution_mode(self) -> "DirectHTTPToolAccess":
        if self.batch_path is not None and self.tool_call_context is not None:
            raise ValueError("batch_path and tool_call_context are mutually exclusive")
        return self


class ContextualToolCallRequest(BaseModel):
    """Normal tool arguments plus hidden model-round context for Resources."""

    model_config = ConfigDict(extra="forbid")

    arguments: dict[str, Any]
    tool_call_context: dict[str, Any]
    tool_call_id: str = Field(min_length=1)
    round_id: str = Field(min_length=1)
    assistant_response: dict[str, Any]


class ContextualToolCallResponse(BaseModel):
    """Separate the model-visible output from hidden Resources context."""

    model_config = ConfigDict(extra="forbid")

    output: str | None
    tool_call_context: dict[str, Any]
    limit_reached: bool = False


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
