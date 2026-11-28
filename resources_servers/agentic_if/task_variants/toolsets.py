# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Built-in tool selections; never executable config."""

from resources_servers.agentic_if.task_variants.catalog import BUILTIN_TOOLS


def configurable_tools(tool_names: dict[str, str] | None) -> frozenset[str]:
    """Omission retains the built-in tools; a mapping selects exactly its installed IDs."""
    builtin = BUILTIN_TOOLS
    if tool_names is None:
        return builtin
    if not set(tool_names) <= builtin:
        raise ValueError(f"unsupported tool for opencode: {sorted(set(tool_names) - builtin)}")
    return frozenset(tool_names)


def renamed_tools(tool_names: dict[str, str] | None) -> dict[str, str]:
    """The selected built-in tools the model sees under another name (their aliases), in the row's order."""
    return {tool: name for tool, name in (tool_names or {}).items() if name != tool}
