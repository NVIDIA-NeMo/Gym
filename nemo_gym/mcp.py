# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared client-side helpers for rollout-scoped Gym MCP servers."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

from nemo_gym.base_resources_server import (
    NEMO_GYM_MCP_METADATA_KEY,
    MCPToolCallProvenance,
)
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse


@dataclass(frozen=True, slots=True)
class RolloutMCPServer:
    """Validated Gym MCP metadata resolved to an absolute resources-server URL."""

    server_name: str
    url: str
    transport: str
    headers: dict[str, str]
    tool_names: Optional[tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class AgentExecutionResult:
    """An agent response and its authoritative structured MCP provenance."""

    response: NeMoGymResponse
    mcp_tool_call_provenance: Optional[dict[str, MCPToolCallProvenance]] = None


def resources_server_base_url(server_client: Any, resources_server_name: str) -> str:
    """Resolve a configured resources-server instance to its base URL."""
    config = get_first_server_config_dict(server_client.global_config_dict, resources_server_name)
    return server_client._build_server_base_url(config)


def parse_rollout_mcp_server(
    seed_response_json: Mapping[str, Any],
    *,
    resources_server_name: str,
    resources_server_base_url: str | Callable[[], str],
    logger: logging.Logger | None = None,
) -> RolloutMCPServer | None:
    """Parse Gym ``/seed_session`` MCP metadata without assuming a harness config shape."""
    if NEMO_GYM_MCP_METADATA_KEY not in seed_response_json:
        return None

    metadata = seed_response_json[NEMO_GYM_MCP_METADATA_KEY]
    if not isinstance(metadata, Mapping):
        raise ValueError("MCP seed metadata must be an object")

    raw_server_name = metadata.get("server_name")
    server_name = resources_server_name if raw_server_name is None else raw_server_name
    if not isinstance(server_name, str) or not server_name:
        raise ValueError("MCP seed metadata server_name must be a non-empty string")

    raw_url_path = metadata.get("url_path")
    url_path = "/mcp" if raw_url_path is None else raw_url_path
    if not isinstance(url_path, str):
        raise ValueError("MCP seed metadata url_path must be a string")

    raw_transport = metadata.get("transport")
    transport = "http" if raw_transport is None else raw_transport
    if not isinstance(transport, str):
        raise ValueError("MCP seed metadata transport must be a string")
    transport = transport.replace("_", "-")

    raw_headers = metadata.get("headers")
    if raw_headers is not None and not isinstance(raw_headers, Mapping):
        raise ValueError("MCP seed metadata headers must be an object")
    headers: dict[str, str] = {}
    if isinstance(raw_headers, Mapping):
        for key, value in raw_headers.items():
            if not isinstance(key, str) or not isinstance(value, (str, int, float, bool)):
                raise ValueError("MCP seed metadata headers must contain scalar values")
            headers[key] = str(value)
    if not headers and logger is not None:
        logger.warning(
            "MCP seed metadata for %r has no headers; the tool endpoint will be called without a "
            "session token and may reject the calls.",
            server_name,
        )

    raw_tool_names = metadata.get("tool_names")
    tool_names: Optional[tuple[str, ...]]
    if raw_tool_names is None:
        tool_names = None
    elif not isinstance(raw_tool_names, list):
        raise ValueError("MCP seed metadata tool_names must be an array")
    else:
        tool_names = tuple(name for name in raw_tool_names if isinstance(name, str) and name)

    base_url = resources_server_base_url() if callable(resources_server_base_url) else resources_server_base_url
    return RolloutMCPServer(
        server_name=server_name,
        url=f"{base_url.rstrip('/')}/{url_path.lstrip('/')}",
        transport=transport,
        headers=headers,
        tool_names=tool_names,
    )


def build_mcp_tool_aliases(
    server: RolloutMCPServer,
    *,
    wire_name: Callable[[str, str], str],
) -> Optional[dict[str, MCPToolCallProvenance]]:
    """Map harness wire names to canonical identities, omitting ambiguous aliases."""
    if server.tool_names is None:
        return None

    aliases: dict[str, MCPToolCallProvenance] = {}
    ambiguous_aliases: set[str] = set()
    for tool_name in server.tool_names:
        alias = wire_name(server.server_name, tool_name)
        identity = MCPToolCallProvenance(server_name=server.server_name, tool_name=tool_name)
        previous = aliases.get(alias)
        if previous is not None and previous != identity:
            aliases.pop(alias)
            ambiguous_aliases.add(alias)
        elif alias not in ambiguous_aliases:
            aliases[alias] = identity
    return aliases


def provenance_from_response_aliases(
    response: NeMoGymResponse,
    aliases: Mapping[str, MCPToolCallProvenance],
) -> dict[str, MCPToolCallProvenance]:
    """Join emitted Responses function calls to canonical identities by wire name."""
    provenance: dict[str, MCPToolCallProvenance] = {}
    for item in response.output:
        if getattr(item, "type", None) != "function_call":
            continue
        call_id = getattr(item, "call_id", None)
        name = getattr(item, "name", None)
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str):
            continue
        identity = aliases.get(name)
        if identity is not None:
            provenance[call_id] = identity
    return provenance
