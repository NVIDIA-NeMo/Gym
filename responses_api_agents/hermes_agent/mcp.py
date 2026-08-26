# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map Gym MCP tools to Hermes wire names while preserving canonical identity for verification."""

from __future__ import annotations

import re
from typing import Any, Optional

from nemo_gym.base_resources_server import NEMO_GYM_MCP_METADATA_KEY
from nemo_gym.openai_utils import NeMoGymResponse


def _sanitize_name_component(value: str) -> str:
    """Mirror Hermes 0.20.5's provider-safe MCP name normalization."""
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def hermes_mcp_tool_aliases(seed_response_json: dict[str, Any]) -> Optional[dict[str, dict[str, str]]]:
    """Build the exact Hermes alias map for the rollout-specific Gym MCP server."""
    metadata = seed_response_json.get(NEMO_GYM_MCP_METADATA_KEY)
    if not isinstance(metadata, dict):
        return None

    server_name = metadata.get("server_name")
    tool_names = metadata.get("tool_names")
    if not isinstance(server_name, str) or not server_name or not isinstance(tool_names, list):
        return None

    aliases: dict[str, dict[str, str]] = {}
    ambiguous_aliases: set[str] = set()
    for tool_name in tool_names:
        if not isinstance(tool_name, str) or not tool_name:
            continue
        alias = f"mcp__{_sanitize_name_component(server_name)}__{_sanitize_name_component(tool_name)}"
        identity = {"server_name": server_name, "tool_name": tool_name}
        previous = aliases.get(alias)
        if previous is not None and previous != identity:
            aliases.pop(alias)
            ambiguous_aliases.add(alias)
        elif alias not in ambiguous_aliases:
            aliases[alias] = identity
    return aliases


def response_mcp_tool_call_provenance(
    response: NeMoGymResponse,
    aliases: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Attach canonical identity only to emitted calls matching an advertised Hermes alias."""
    provenance: dict[str, dict[str, str]] = {}
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
