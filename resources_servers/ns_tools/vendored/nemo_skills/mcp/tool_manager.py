# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module-based tool runtime: Tool interface and ToolManager.

This provides a lightweight plugin system for tools that can be supplied as
Python modules/classes, without requiring a YAML configuration. Each tool
module exposes a concrete class implementing the Tool interface (no "Provider"
suffix per convention). The manager imports and instantiates tools, applies
per-tool overrides, exposes OpenAI-aligned tool schemas, and routes calls.
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List

from .utils import locate


class FatalToolError(Exception):
    """Exception for fatal tool errors that should stop the entire process.

    Use this for unrecoverable errors like authentication failures where
    continuing would be pointless (e.g., invalid API keys).
    """

    pass


class Tool(ABC):
    """Abstract base for module-based tools.

    Conventions:
    - list_tools() returns a list of dicts with keys: name, description, input_schema.
    - execute() is invoked with the bare tool name (no provider prefix).
    """

    @abstractmethod
    def default_config(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def configure(self, overrides: Dict[str, Any] | None = None, context: Dict[str, Any] | None = None) -> None:
        pass

    @abstractmethod
    async def list_tools(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def execute(
        self, tool_name: str, arguments: Dict[str, Any], extra_args: Dict[str, Any] | None = None
    ) -> Any:
        pass

    async def shutdown(self) -> None:  # Optional hook
        return None

    def post_configure(self) -> None:
        return None


class ToolManager:
    """Registry/Router for module-based tools.

    - Loads tool classes from module specs using nemo_skills.mcp.utils.locate.
    - Applies per-tool overrides based on the tool class name.
    - Exposes merged tool list with qualified names: "{ToolClassName}.{tool_name}".
    - Routes tool executions by splitting the qualified name.
    """

    def __init__(
        self,
        module_specs: Iterable[str],
        overrides: Dict[str, Dict[str, Any]] | None = None,
        context: Dict[str, Any] | None = None,
    ) -> None:
        self._tools: Dict[str, Tool] = {}
        self._qualified_tool_map: Dict[str, str] = {}  # qualified name -> tool_class_name
        self._tool_list_cache: List[Dict[str, Any]] | None = None
        self._raw_to_qualified_map: Dict[str, str] = {}  # raw name -> qualified name
        self._list_lock = asyncio.Lock()

        overrides = overrides or {}
        context = context or {}

        for spec in module_specs or []:
            tool_cls_or_obj = locate(spec)
            tool_cls = tool_cls_or_obj
            if not inspect.isclass(tool_cls_or_obj):
                # Allow passing an already-instantiated object
                tool_cls = tool_cls_or_obj.__class__
            tool: Tool = tool_cls() if inspect.isclass(tool_cls_or_obj) else tool_cls_or_obj

            provider_key = tool.__class__.__name__
            if provider_key in self._tools:
                raise ValueError(f"Duplicate tool class registered: '{provider_key}'")

            tool.configure((overrides.get(provider_key) if overrides else None), context)
            tool.post_configure()
            self._tools[provider_key] = tool

    async def shutdown(self) -> None:
        for tool in self._tools.values():
            try:
                await tool.shutdown()
            except Exception:
                # Best effort; do not propagate shutdown issues
                pass

    async def list_all_tools(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        async with self._list_lock:
            if use_cache and self._tool_list_cache is not None:
                return self._tool_list_cache

            merged: List[Dict[str, Any]] = []
            local_qualified_map: Dict[str, str] = {}

            async def load_provider(provider_id: str, tool: Tool) -> List[Dict[str, Any]]:
                entries = await tool.list_tools()
                out: List[Dict[str, Any]] = []
                local_seen: set[str] = set()

                for entry in entries or []:
                    raw_name = entry.get("name")
                    if not raw_name or raw_name in local_seen:
                        continue
                    local_seen.add(raw_name)
                    qualified_name = f"{provider_id}.{raw_name}"
                    if qualified_name in local_qualified_map:
                        raise ValueError(f"Duplicate qualified tool name: '{qualified_name}'")
                    if raw_name in self._raw_to_qualified_map:
                        raise ValueError(f"Duplicate raw tool name across providers: '{raw_name}'")
                    self._raw_to_qualified_map[raw_name] = qualified_name
                    local_qualified_map[qualified_name] = provider_id
                    # dropping title (recursively) as it's meant to be shown to users, not model
                    entry.get("input_schema", {}).pop("title", None)
                    for parameter in entry.get("input_schema", {}).get("properties", {}).values():
                        parameter.pop("title", None)

                    out.append({**entry, "name": raw_name, "server": provider_id})
                return out

            tasks = [load_provider(pid, tool) for pid, tool in self._tools.items()]
            results = await asyncio.gather(*tasks)
            for lst in results:
                merged.extend(lst)

            self._qualified_tool_map = local_qualified_map
            self._tool_list_cache = merged
            return merged

    def _resolve(self, qualified_name: str) -> tuple[Tool, str]:
        if "." not in qualified_name:
            raise ValueError(f"Tool name must be qualified as 'ToolClassName.tool'. Received: '{qualified_name}'")
        provider_id, bare_name = qualified_name.split(".", 1)
        tool = self._tools.get(provider_id)
        if tool is None:
            raise ValueError(f"No tool registered for class '{provider_id}'")
        return tool, bare_name

    async def execute_tool(self, raw_name: str, args: Dict[str, Any], extra_args: Dict[str, Any] | None = None):
        tool, bare = self._resolve(self._raw_to_qualified_map[raw_name])
        return await tool.execute(bare, args, extra_args=extra_args)
