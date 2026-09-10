# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Type definitions for Salesforce tools."""

from typing import Any, Callable, TypedDict


class ToolMetadata(TypedDict):
    """Metadata for a Salesforce tool mapping it to the implementation."""

    selected_api: str
    action: str
    type: str
    action_id: str


# Registry to store tool metadata separately (avoids adding attributes to functions)
TOOL_METADATA: dict[Callable[..., Any], ToolMetadata] = {}


def register_metadata(func: Callable[..., Any], metadata: ToolMetadata) -> None:
    """Register metadata for a tool function."""
    TOOL_METADATA[func] = metadata
