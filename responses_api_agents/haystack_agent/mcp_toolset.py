# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""MCPToolset whose invocation client is scoped to the active Gym rollout."""

import copy
from typing import Any

from haystack.core.serialization import allow_deserialization_module
from haystack.tools import Tool
from haystack_integrations.tools.mcp.mcp_tool import (
    AsyncExecutor,
    MCPClient,
    MCPToolNotFoundError,
    _extract_first_text_element,
    _MCPClientSessionManager,
)
from haystack_integrations.tools.mcp.mcp_toolset import MCPToolset

from nemo_gym.base_resources_server import NEMO_GYM_MCP_SESSION_TOKEN_HEADER
from responses_api_agents.haystack_agent import chat_generator


allow_deserialization_module("responses_api_agents.haystack_agent.mcp_toolset")


class ContextAwareMCPToolset(MCPToolset):
    """Discover stable schemas once, but authenticate MCP calls with the rollout token."""

    def _client_for_current_rollout(self) -> MCPClient:
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
                    server_info.headers = dict(state.mcp_headers)
                    worker = _MCPClientSessionManager(server_info.create_client(), timeout=self.connection_timeout)
                    state.mcp_workers[key] = worker
        return worker._client

    def _connect_and_load_tools(self) -> list[Tool]:
        """Fetch schemas without a rollout token; tool calls connect lazily with one."""
        worker = _MCPClientSessionManager(self.server_info.create_client(), timeout=self.connection_timeout)
        try:
            tool_infos = worker.tools()
        finally:
            worker.stop()

        available_names = {tool.name for tool in tool_infos}
        if self.tool_names:
            missing_names = set(self.tool_names) - available_names
            if missing_names:
                raise MCPToolNotFoundError(
                    message=(
                        f"The following tools were not found: {', '.join(missing_names)}. "
                        f"Available tools: {', '.join(available_names)}"
                    ),
                    tool_name=next(iter(missing_names)),
                    available_tools=list(available_names),
                )

        def invoke(tool_name: str, outputs_to_state: dict[str, Any] | None, **kwargs: Any) -> Any:
            result = AsyncExecutor.get_instance().run(
                self._client_for_current_rollout().call_tool(tool_name, kwargs), timeout=self.invocation_timeout
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
    if isinstance(tools, ContextAwareMCPToolset):
        tools.server_info.url = mcp_url
        return 1
    if isinstance(tools, (list, tuple, set)):
        return sum(configure_mcp_url(tool, mcp_url) for tool in tools)
    for attribute in ("toolsets", "_toolsets"):
        nested = getattr(tools, attribute, None)
        if isinstance(nested, (list, tuple, set)):
            return sum(configure_mcp_url(tool, mcp_url) for tool in nested)
    return 0


def has_context_aware_mcp_toolset(tools: Any) -> bool:
    if isinstance(tools, ContextAwareMCPToolset):
        return True
    if isinstance(tools, (list, tuple, set)):
        return any(has_context_aware_mcp_toolset(tool) for tool in tools)
    return any(
        isinstance(nested, (list, tuple, set)) and any(has_context_aware_mcp_toolset(tool) for tool in nested)
        for nested in (getattr(tools, attribute, None) for attribute in ("toolsets", "_toolsets"))
    )


def close_rollout_mcp_sessions(state: chat_generator._GenRunState) -> None:
    """Release every token-authenticated session created during one rollout."""
    for worker in state.mcp_workers.values():
        worker.stop()
    state.mcp_workers.clear()
