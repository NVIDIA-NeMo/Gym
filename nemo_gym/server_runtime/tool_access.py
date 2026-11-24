# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-visible tool access contracts."""

from collections.abc import Collection, Iterable
from typing import Annotated, Any, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Discriminator, Field, Tag


# Direct HTTP has no field for a tool's error result, which MCP reports as `isError`, so a tool route that answers a
# call with an error result sets this header to "true".
TOOL_ERROR_HEADER = "X-NeMo-Gym-Tool-Error"

# The ID the model gave a tool call (its `call_id`), which an agent may send with a direct HTTP call, so the server can
# join its record to the agent's trajectory.
TOOL_CALL_ID_HEADER = "X-NeMo-Gym-Tool-Call-Id"

# A tool route sets this header to "true" when its reply asks the agent to end its activation: ``simple_agent`` finishes
# the tool calls of the current model response and returns, without calling the model again. Other agents ignore the
# header and call the model again. MCP has no field for it.
END_ACTIVATION_HEADER = "X-NeMo-Gym-End-Activation"


class DirectHTTPToolAccess(BaseModel):
    """Connect a trusted Python agent to typed tool routes.

    A route answers a call whose result is an error with ``X-NeMo-Gym-Tool-Error: true`` (``TOOL_ERROR_HEADER``),
    even when its status is 200. An agent may send the model's ``call_id`` in ``X-NeMo-Gym-Tool-Call-Id``
    (``TOOL_CALL_ID_HEADER``). A reply with ``X-NeMo-Gym-End-Activation: true`` (``END_ACTIVATION_HEADER``) asks
    the agent to end its activation: ``simple_agent`` runs the rest of the model response's tool calls and
    returns without another model call. Other agents ignore the header and call the model again.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["direct_http"] = "direct_http"
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    required: bool
    base_url: AnyHttpUrl
    cookies: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)


def _transport_required_in_json(schema: dict[str, Any]) -> None:
    # Python callers may leave `transport` to its default, but a JSON connection without one is a streamable HTTP
    # one (`_mcp_transport`), so the exported schema requires the tag of every other transport. Otherwise an untagged
    # HTTP connection would match two branches of the connection's `oneOf`, and an untagged stdio one would pass the
    # schema while parsing rejects it.
    schema["properties"]["transport"].pop("default")
    schema["required"] = ["transport", *schema["required"]]


class MCPStreamableHTTPConnection(BaseModel):
    """Connect to an MCP server over an absolute HTTP endpoint."""

    model_config = ConfigDict(extra="forbid")

    transport: Literal["streamable_http"] = "streamable_http"
    url: AnyHttpUrl
    headers: dict[str, str] = Field(default_factory=dict)


class MCPSSEConnection(BaseModel):
    """Connect to an MCP server over the HTTP+SSE transport: an event stream at ``url`` that names the endpoint the
    client posts its messages to."""

    model_config = ConfigDict(extra="forbid", json_schema_extra=_transport_required_in_json)

    transport: Literal["sse"] = "sse"
    url: AnyHttpUrl
    headers: dict[str, str] = Field(default_factory=dict)


class MCPStdioConnection(BaseModel):
    """Start an MCP server as a process that speaks MCP on its standard input and output.

    The harness's MCP client starts ``command`` with ``args`` where the harness runs (inside the task sandbox for a
    sandboxed harness, as the user the harness runs as), with ``env`` added to the environment the client gives
    its servers, and stops it with the harness.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra=_transport_required_in_json)

    transport: Literal["stdio"] = "stdio"
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)


def _mcp_transport(connection: Any) -> str:
    # A connection without a transport is a streamable HTTP one, as before the other transports existed.
    if isinstance(connection, dict):
        return connection.get("transport", "streamable_http")
    return getattr(connection, "transport", "streamable_http")


MCPConnection = Annotated[
    Annotated[MCPStreamableHTTPConnection, Tag("streamable_http")]
    | Annotated[MCPSSEConnection, Tag("sse")]
    | Annotated[MCPStdioConnection, Tag("stdio")],
    Discriminator(_mcp_transport),
]
MCPTransport = Literal["streamable_http", "sse", "stdio"]


class MCPToolAccess(BaseModel):
    """Describe one named MCP server that an agent adapter must configure.

    Names are session-wide identifiers. Agent session requests reject duplicate names,
    and adapters must also reject collisions with their static tool configuration.
    Failure to establish a required server fails session seeding; an optional server may
    be skipped. An adapter whose client cannot use a grant's transport refuses the seed when
    the grant is required and leaves the grant out otherwise.

    A streamable HTTP or SSE server runs elsewhere: on agent-session close, adapters disconnect
    clients without stopping it. A stdio server is a process the harness's MCP client starts
    for the session and that ends with the harness; adapters stop nothing remote.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["mcp"] = "mcp"
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    required: bool
    connection: MCPConnection


ToolAccess = Annotated[
    DirectHTTPToolAccess | MCPToolAccess,
    Field(discriminator="kind"),
]


def mcp_grants_over(accesses: Iterable[ToolAccess], transports: Collection[MCPTransport]) -> list[MCPToolAccess]:
    """The MCP grants in ``accesses`` whose connection uses one of ``transports``, in order.

    A harness whose MCP client reaches servers only over ``transports`` configures these and leaves the other MCP
    grants out; its seed refuses the required ones (:func:`unusable_required_grants`).
    """
    return [
        access
        for access in accesses
        if isinstance(access, MCPToolAccess) and access.connection.transport in transports
    ]


def unusable_required_grants(accesses: Iterable[ToolAccess], transports: Collection[MCPTransport]) -> list[str]:
    """The required grants in ``accesses`` that a harness whose MCP client reaches servers only over ``transports``
    cannot use, sorted: a direct HTTP grant as its name, an MCP grant over another transport as ``name (transport)``.
    """
    unusable = []
    for access in accesses:
        if not access.required:
            continue
        if not isinstance(access, MCPToolAccess):
            unusable.append(access.name)
        elif access.connection.transport not in transports:
            unusable.append(f"{access.name} ({access.connection.transport})")
    return sorted(unusable)


def mcp_server_urls(grants: Iterable[MCPToolAccess]) -> list[str]:
    """The URLs of the servers ``grants`` reach over the network, in order: a stdio server runs beside the harness
    and has none. A harness that restricts its sandbox's network allows these hosts."""
    return [str(grant.connection.url) for grant in grants if not isinstance(grant.connection, MCPStdioConnection)]
