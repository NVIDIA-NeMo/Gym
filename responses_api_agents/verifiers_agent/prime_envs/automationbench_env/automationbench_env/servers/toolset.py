"""Per-rollout tools over one live AutomationBench world."""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING, Annotated, Any, Callable

import verifiers.v1 as vf
from automationbench import rubric
from automationbench.schema.world import WorldState
from automationbench.tools.api.contract import APIContract
from automationbench.tools.api.encode import base64_encode
from mcp.types import CallToolResult
from pydantic import Field

from automationbench_env.common import (
    TOOL_PREFIX,
    AutomationBenchData,
    AutomationBenchState,
    AutomationBenchToolsetConfig,
    compute_allowed_services,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

SearchQuery = Annotated[str, Field(min_length=1)]
SearchLimit = Annotated[int, Field(ge=1, le=20)]


class AutomationBenchToolset(vf.Toolset[AutomationBenchToolsetConfig, AutomationBenchState]):
    TOOL_PREFIX = TOOL_PREFIX

    async def setup(self) -> None:
        self._lock = asyncio.Lock()

    async def setup_task(self, task: AutomationBenchData) -> None:
        self._task = task
        self._world = WorldState(**task.initial_state)
        self._world.meta.allowed_services = compute_allowed_services(
            task.initial_state, task.assertions, task.zapier_tools
        )
        self._api = APIContract(self._world)

    def register(self, mcp: FastMCP) -> None:
        mcp.add_tool(self._make_api_search())
        mcp.add_tool(base64_encode)
        mcp.add_tool(
            self._serialized(self._with_state(self._make_api_fetch())),
            name="api_fetch",
            structured_output=False,
        )
        # FastMCP 1.x derives schemas only from Python annotations and has no public
        # custom-schema argument. The stored schema is model-facing; call validation
        # still goes through the callable and APIContract below.
        mcp._tool_manager.get_tool("api_fetch").parameters = self._api.fetch_schema()

    def _make_api_search(self) -> Callable:
        def call(query: SearchQuery, top_k: SearchLimit = 5) -> dict[str, Any]:
            return self._api.search(query, top_k)

        call.__name__ = "api_search"
        call.__doc__ = f"""Search the APIs connected to this workspace.

Returns canonical endpoint IDs and JSON schemas for path, query, and body objects as structured
JSON. The api_fetch tool definition contains the same schemas as concrete endpoint branches. Every result
has a simulator implementation and belongs to one of these connected services:
{", ".join(self._api.connected_services)}. Google Sheets searches also return the accessible
spreadsheet and worksheet IDs and titles.
"""
        return call

    def _make_api_fetch(self) -> Callable:
        def call(
            endpoint: str,
            path: dict[str, Any] | None = None,
            params: dict[str, Any] | None = None,
            body: dict[str, Any] | None = None,
        ) -> CallToolResult:
            result = self._api.execute(endpoint, path, params, body)
            rubric_state = {
                "info": {"assertions": self._task.assertions},
                "world": self._world,
                "initial_state": self._task.initial_state,
            }
            self.state.partial_credit = rubric.partial_credit(rubric_state)
            return result

        call.__name__ = "api_fetch"
        call.__doc__ = """Execute an endpoint returned by api_search.

Args:
    endpoint: Canonical endpoint ID returned by api_search.
    path: Values matching that endpoint branch's path JSON schema.
    params: Values matching that endpoint branch's query JSON schema.
    body: Values matching that endpoint branch's body JSON schema.

The endpoint fixes its service, method, and internal route. Unknown endpoints, unavailable
services, invalid fields, and simulator application failures are returned as MCP tool errors.
"""
        return call

    def _serialized(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            async with self._lock:
                return await fn(*args, **kwargs)

        return wrapper


if __name__ == "__main__":
    AutomationBenchToolset.run()
