# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Describe Hermes MCP names without changing the model-authored response."""

import json
from collections.abc import Mapping


HERMES_MCP_SERVERS_METADATA_KEY = "hermes_mcp_servers"
_MAX_METADATA_BYTES = 512


def hermes_mcp_metadata(server_names: list[str]) -> dict[str, str]:
    """Encode the complete granted MCP names for verifier-only normalization.

    Hermes replaces ``-`` and ``.`` with ``_`` in server names. Refuse grants
    whose resulting names collide rather than guessing which server ran a tool.
    """
    if type(server_names) is not list or any(type(name) is not str or not name for name in server_names):
        raise ValueError("Hermes MCP server names must be a list of nonempty strings")
    sanitized = [name.replace("-", "_").replace(".", "_") for name in server_names]
    if len(set(sanitized)) != len(server_names):
        raise ValueError("Hermes MCP server names must be unique after replacing '-' and '.' with '_'")
    encoded = json.dumps(server_names, separators=(",", ":"), ensure_ascii=True)
    if len(encoded.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValueError("Hermes MCP server metadata exceeds 512 bytes")
    return {HERMES_MCP_SERVERS_METADATA_KEY: encoded} if server_names else {}


def normalize_hermes_mcp_tool_name(name: str, metadata: Mapping[str, str] | None) -> str:
    """Return Gym's verifier spelling only with explicit, valid Hermes grants.

    Match the complete grant list longest-first so ``store_b`` does not become
    tool ``b_append`` on ``store``. Unknown names and existing Gym names stay as
    emitted. Callers must apply this only to a copied verifier input.
    """
    if metadata is None or HERMES_MCP_SERVERS_METADATA_KEY not in metadata:
        return name
    encoded = metadata[HERMES_MCP_SERVERS_METADATA_KEY]
    if type(encoded) is not str or len(encoded.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise ValueError("Hermes MCP server metadata must be a JSON string of at most 512 bytes")
    try:
        server_names = json.loads(encoded)
    except (ValueError, TypeError) as error:
        raise ValueError("Hermes MCP server metadata must contain a JSON list of server names") from error
    hermes_mcp_metadata(server_names)
    for server in sorted(server_names, key=len, reverse=True):
        prefix = "mcp_" + server.replace("-", "_").replace(".", "_") + "_"
        if name.startswith(prefix):
            return f"mcp__{server}__{name[len(prefix) :]}"
    return name
