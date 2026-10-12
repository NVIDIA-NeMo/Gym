# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""MCPToolset whose discovery and invocation are scoped to the active Gym rollout."""

import copy
from collections.abc import Iterator
from typing import Any

from haystack.core.serialization import allow_deserialization_module
from haystack.tools import Tool
from haystack_integrations.tools.mcp.mcp_tool import (
    AsyncExecutor,
    MCPServerInfo,
    _extract_first_text_element,
    _MCPClientSessionManager,
)
from haystack_integrations.tools.mcp.mcp_toolset import MCPToolset

from nemo_gym.base_resources_server import NEMO_GYM_MCP_SESSION_TOKEN_HEADER
from responses_api_agents.haystack_agent import chat_generator


allow_deserialization_module("responses_api_agents.haystack_agent.mcp_toolset")


class ContextAwareMCPToolset(MCPToolset):
    """Spawn ordinary, populated MCP toolsets using each rollout's credentials."""

    def __init__(
        self,
        server_info: MCPServerInfo,
        tool_names: list[str] | None = None,
        connection_timeout: float = 30.0,
        invocation_timeout: float = 30.0,
        eager_connect: bool = False,
        inputs_from_state: dict[str, dict[str, str]] | None = None,
        outputs_to_state: dict[str, dict[str, dict[str, Any]]] | None = None,
        outputs_to_string: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        # Pipeline deserialization precedes the rollout token. Keep the configuration,
        # but connect only when a rollout copy is warmed up.
        super().__init__(
            server_info,
            tool_names,
            connection_timeout,
            invocation_timeout,
            False,
            inputs_from_state,
            outputs_to_state,
            outputs_to_string,
        )
        self.eager_connect = eager_connect

    def spawn(self, *, allow_filtering: bool | None = None) -> "ContextAwareMCPToolset":
        spawned = copy.copy(self)
        state = chat_generator._current_run_state.get()
        if state is None or getattr(self, "_rollout_state", None) is not state:
            spawned.tools = []
            spawned._warmup_called = False
        spawned._selected_tool_names = None
        if allow_filtering is not None:
            spawned._allow_filtering = allow_filtering
        spawned.warm_up()
        return spawned

    def _connect_and_load_tools(self) -> list[Tool]:
        tools = self.tools_for_current_rollout(getattr(self, "_allow_filtering", False))
        self._rollout_state = chat_generator._current_run_state.get()
        return tools

    def _worker_for_current_rollout(self) -> _MCPClientSessionManager:
        state = chat_generator._current_run_state.get()
        if state is None or not state.mcp_headers.get(NEMO_GYM_MCP_SESSION_TOKEN_HEADER):
            raise RuntimeError(
                "MCP tool calls require X-NeMo-Gym-Session-Token from /seed_session. "
                "Call this agent through /run, or provide that header to /v1/responses."
            )

        key = id(self)
        worker = state.mcp_workers.get(key)
        if worker is None:
            with state.mcp_lock:
                worker = state.mcp_workers.get(key)
                if worker is None:
                    server_info = copy.copy(self.server_info)
                    server_info.headers = {**(self.server_info.headers or {}), **state.mcp_headers}
                    worker = _MCPClientSessionManager(server_info.create_client(), timeout=self.connection_timeout)
                    state.mcp_workers[key] = worker
        return worker

    def tools_for_current_rollout(self, allow_filtering: bool = False) -> list[Tool]:
        """Discover authenticated schemas without changing the shared toolset."""
        worker = self._worker_for_current_rollout()
        tool_infos = worker.tools()

        available_names = {tool.name for tool in tool_infos}
        if self.tool_names is not None:
            excluded_names = available_names - set(self.tool_names)
            if excluded_names and not allow_filtering:
                raise ValueError(
                    f"MCP tool_names excludes tools available to this rollout: {sorted(excluded_names)}. "
                    "Set tool_names to null to include all tools, or set allow_mcp_tool_filtering: true "
                    "on the Haystack agent to intentionally select a subset."
                )

        def invoke(tool_name: str, outputs_to_state: dict[str, Any] | None, **kwargs: Any) -> Any:
            result = AsyncExecutor.get_instance().run(
                worker._client.call_tool(tool_name, kwargs), timeout=self.invocation_timeout
            )
            return _extract_first_text_element(result) if outputs_to_state else result

        def create_invoke(tool_name: str, outputs_to_state: dict[str, Any] | None):
            def invoke_tool(**kwargs: Any) -> Any:
                return invoke(tool_name, outputs_to_state, **kwargs)

            return invoke_tool

        tools = []
        for tool_info in tool_infos:
            if self.tool_names is not None and tool_info.name not in self.tool_names:
                continue
            outputs_to_state = self.outputs_to_state.get(tool_info.name)
            tools.append(
                Tool(
                    name=tool_info.name,
                    description=tool_info.description or "",
                    parameters=tool_info.inputSchema,
                    function=create_invoke(tool_info.name, outputs_to_state),
                    inputs_from_state=self.inputs_from_state.get(tool_info.name),
                    outputs_to_state=outputs_to_state,
                    outputs_to_string=self.outputs_to_string.get(tool_info.name),
                )
            )
        self._validate_state_configs({tool.name for tool in tools})
        return tools


def configure_mcp_url(tools: Any, mcp_url: str) -> int:
    """Point all context-aware MCP toolsets in an Agent's tool collection at Gym."""
    toolsets = list(context_aware_mcp_toolsets(tools))
    for toolset in toolsets:
        toolset.server_info.url = mcp_url
    return len(toolsets)


def has_context_aware_mcp_toolset(tools: Any) -> bool:
    return next(context_aware_mcp_toolsets(tools), None) is not None


def spawn_mcp_toolsets(tools: Any, allow_filtering: bool) -> Any:
    """Replace MCP templates with rollout copies, preserving local toolsets and wrappers."""
    if isinstance(tools, ContextAwareMCPToolset):
        return tools.spawn(allow_filtering=allow_filtering)
    if isinstance(tools, (list, tuple, set)):
        return [spawn_mcp_toolsets(tool, allow_filtering) for tool in tools]
    if has_context_aware_mcp_toolset(tools):
        tools = copy.copy(tools)
        for attribute in ("toolsets", "_toolsets"):
            nested = getattr(tools, attribute, None)
            if isinstance(nested, (list, tuple, set)):
                setattr(tools, attribute, spawn_mcp_toolsets(nested, allow_filtering))
    return tools


def context_aware_mcp_toolsets(tools: Any) -> Iterator[ContextAwareMCPToolset]:
    """Visit configured MCP toolsets without discovering or flattening their tools."""
    if isinstance(tools, ContextAwareMCPToolset):
        yield tools
    elif isinstance(tools, (list, tuple, set)):
        for tool in tools:
            yield from context_aware_mcp_toolsets(tool)
    else:
        for attribute in ("toolsets", "_toolsets"):
            nested = getattr(tools, attribute, None)
            if isinstance(nested, (list, tuple, set)):
                yield from context_aware_mcp_toolsets(nested)


def close_rollout_mcp_sessions(state: chat_generator._GenRunState) -> None:
    """Release every token-authenticated session created during one rollout."""
    for worker in state.mcp_workers.values():
        worker.stop()
    state.mcp_workers.clear()
