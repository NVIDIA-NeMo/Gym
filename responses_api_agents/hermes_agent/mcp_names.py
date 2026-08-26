# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes-specific MCP wire-name translation."""

from __future__ import annotations

import re


def _sanitize_name_component(value: str) -> str:
    """Mirror Hermes 0.20.5's provider-safe MCP name normalization."""
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def hermes_mcp_wire_name(server_name: str, tool_name: str) -> str:
    """Mirror Hermes 0.20.5's flattened provider-safe MCP tool name."""
    return f"mcp__{_sanitize_name_component(server_name)}__{_sanitize_name_component(tool_name)}"
