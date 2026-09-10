# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Meta-tools for tool discovery and execution.

Provides search_tools (BM25 search) and execute_tool (dynamic dispatch)
over the AutomationBench tool registry.
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from automationbench.schema import WorldState
from automationbench.tool_wrapper import _create_tool_wrapper
from automationbench.utils.bm25 import BM25Scorer


class ToolRegistry:
    """Registry of AutomationBench tools with BM25 search and dynamic execution.

    Builds an in-memory index of tool names, descriptions, and parameter
    schemas at init time. Supports BM25 keyword search.
    """

    def __init__(self, tools: list[Callable]) -> None:
        self._tool_map: dict[str, Callable] = {}
        self._tool_info: list[dict[str, Any]] = []
        self._searchable_texts: list[str] = []

        for func in tools:
            name = getattr(func, "__name__", repr(func))
            full_desc = self._get_full_description(func)
            params = self._get_parameter_schema(func)

            self._tool_map[name] = func
            self._tool_info.append(
                {
                    "name": name,
                    "description": full_desc,
                    "parameters": params,
                }
            )
            # Parameter vocabulary makes searches work even when a tool name is terse.
            param_parts: list[str] = []
            for param_name, param_info in params.get("properties", {}).items():
                param_parts.append(param_name)
                if isinstance(param_info, dict) and param_info.get("description"):
                    param_parts.append(param_info["description"])
            params_text = " ".join(param_parts)
            searchable = f"{name}: {full_desc}"
            if params_text:
                searchable += f" {params_text}"
            self._searchable_texts.append(searchable)

        self._bm25 = BM25Scorer(self._searchable_texts)

    @staticmethod
    def _get_full_description(func: Callable) -> str:
        """Extract full docstring with 'world' parameter documentation removed."""
        doc = inspect.getdoc(func)
        if not doc:
            return getattr(func, "__name__", repr(func))
        lines = doc.split("\n")
        filtered: list[str] = []
        skip_world = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("world:"):
                skip_world = True
                continue
            if skip_world:
                # Continuation lines of world param are indented or empty
                if stripped and not stripped.endswith(":") and not line.startswith(" " * 4 + " "):
                    skip_world = False
                else:
                    continue
            filtered.append(line)
        return "\n".join(filtered)

    @staticmethod
    def _get_parameter_schema(func: Callable) -> dict[str, Any]:
        """Build JSON schema for tool parameters, excluding 'world'."""
        from agents.function_schema import function_schema

        wrapper = _create_tool_wrapper(func, args_to_skip=["world"])
        schema_obj = function_schema(wrapper)
        params = schema_obj.params_json_schema
        params.pop("$defs", None)
        return params

    def bm25(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """BM25 ranking over tool names and descriptions."""
        return [self._tool_info[i] for i in self._bm25.top_k(query, k=top_k)]

    def execute(self, tool_name: str, arguments: str, **injected: Any) -> str:
        """Execute a tool by name with JSON arguments string."""
        func = self._tool_map.get(tool_name)
        if func is None:
            raise ValueError(
                f"Unknown tool: {tool_name}. Use search_tools to discover available tools."
            )
        parsed_args = json.loads(arguments)
        merged = {**parsed_args, **injected}
        result = func(**merged)
        if isinstance(result, str):
            return result
        return json.dumps(result)


# ---------------------------------------------------------------------------
# Module-level singleton (lazily initialized)
# ---------------------------------------------------------------------------

_registry: ToolRegistry | None = None


def _get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        from automationbench.tools import ALL_TOOLS

        _registry = ToolRegistry(tools=ALL_TOOLS)
    return _registry


# ---------------------------------------------------------------------------
# Standalone tool functions
# ---------------------------------------------------------------------------


def search_tools(query: str, top_k: int = 5) -> str:
    """Find available tools by name or description.

    Tool names follow the pattern {service}_{action} (e.g., salesforce_query,
    gmail_send_email, slack_send_channel_message).

    Uses BM25 keyword-based relevance search. Works with service names,
    action words, or multi-word queries.
    Examples: "salesforce", "send email", "update deal", "slack channel"

    Args:
        query: Search query — service names, keywords, or a description.
        top_k: Maximum number of results to return (default: 20).

    Returns:
        JSON string with a list of matching tools, each containing name,
        description, and parameter schema.
    """
    registry = _get_registry()
    results = registry.bm25(query, top_k=top_k)
    return json.dumps(results, indent=2)


def make_search_tools(default_top_k: int = 20, max_top_k: int | None = None) -> Callable:
    """Return a search_tools function with a custom default and optional hard cap on top_k."""
    cap = max_top_k
    default = default_top_k

    def _search_tools(query: str, top_k: int = default) -> str:
        """Find available tools by name or description.

        Tool names follow the pattern {service}_{action} (e.g., salesforce_query,
        gmail_send_email, slack_send_channel_message).

        Uses BM25 keyword-based relevance search. Works with service names,
        action words, or multi-word queries.
        Examples: "salesforce", "send email", "update deal", "slack channel"

        Args:
            query: Search query — service names, keywords, or a description.
            top_k: Maximum number of results to return (default: 20).

        Returns:
            JSON string with a list of matching tools, each containing name,
            description, and parameter schema.
        """
        registry = _get_registry()
        actual_k = min(top_k, cap) if cap is not None else top_k
        results = registry.bm25(query, top_k=actual_k)
        return json.dumps(results, indent=2)

    _search_tools.__name__ = "search_tools"
    return _search_tools


def execute_tool(world: WorldState, tool_name: str, arguments: str) -> str:
    """Execute a discovered tool by name with the given arguments.

    Use search_tools first to find the right tool and its parameter schema,
    then call this with the tool name and a JSON string of arguments.

    Args:
        world: The current world state (injected automatically).
        tool_name: The exact tool name from search results.
        arguments: JSON string of arguments matching the tool's parameter schema.

    Returns:
        The tool's return value (JSON string).
    """
    registry = _get_registry()
    return registry.execute(tool_name, arguments, world=world)
