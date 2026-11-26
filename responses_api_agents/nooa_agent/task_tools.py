# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Harbor task's own MCP servers and skills directory, on the NOOA agent that runs in the task sandbox.

Each MCP server becomes a tool object on the agent, the way NOOA exposes any tool, and the skills directory is
named in the prompt. The controller reads the declarations at seed (``task_context.py``); this module is the
worker's side and imports nothing the sandbox runtime lacks.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from typing import Any, Literal

from nooa.mcp import MCPBaseClient, MCPManager, MCPTool, create_mcp_client
from pydantic import BaseModel, ConfigDict, Field


LOGGER = logging.getLogger(__name__)

# The context block that names the task's skills directory to the model.
TASK_SKILLS_CONTEXT = "task_skills"
_CLOSE_TIMEOUT_SECONDS = 10


class TaskMCPServer(BaseModel):
    """One MCP server a task declares, as the worker in the sandbox connects to it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    # The agent attribute the server's tools are on: the name as a Python identifier.
    attribute: str
    transport: Literal["stdio", "sse", "streamable-http"]
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)


class HeldMCPClient:
    """One MCP session, opened by the first tool call and kept until the run ends.

    NOOA's clients open a session for every tool call. A server that keeps state per session, such as a browser's
    element references, loses it between two calls, so the task's tools call through this client instead. A
    session that ended is opened again by the next call.
    """

    def __init__(self, client: MCPBaseClient, server_name: str) -> None:
        self._client = client
        self.server_name = server_name
        self._lock = asyncio.Lock()
        self._holder: asyncio.Task[None] | None = None
        self._session: asyncio.Future[Any] | None = None
        self._release: asyncio.Event | None = None

    async def _hold(self, session: asyncio.Future[Any], release: asyncio.Event) -> None:
        # The transport's task group belongs to this task, so it is entered and left here, not in a tool call.
        try:
            async with self._client.connect_to_server() as connected:
                session.set_result(connected)
                await release.wait()
        except asyncio.CancelledError:
            if not session.done():
                session.cancel()
            raise
        except Exception as error:
            if session.done():
                LOGGER.warning("The session with the task's MCP server %r ended: %r", self.server_name, error)
            else:
                session.set_exception(error)

    @asynccontextmanager
    async def connect_to_server(self) -> AsyncGenerator[Any, None]:
        """Yield the held session, as an ``MCPBaseClient`` yields a new one."""
        async with self._lock:
            if self._holder is None or self._holder.done():
                self._session = asyncio.get_running_loop().create_future()
                self._release = asyncio.Event()
                self._holder = asyncio.create_task(self._hold(self._session, self._release))
            session = self._session
        yield await session

    async def close(self) -> None:
        holder, release = self._holder, self._release
        if holder is None or release is None:
            return
        release.set()
        _, pending = await asyncio.wait({holder}, timeout=_CLOSE_TIMEOUT_SECONDS)
        if pending:
            holder.cancel()
            await asyncio.wait({holder}, timeout=_CLOSE_TIMEOUT_SECONDS)
        if holder.done() and not holder.cancelled() and (error := holder.exception()) is not None:
            LOGGER.warning("Closing the task's MCP server %r failed: %r", self.server_name, error)


async def open_task_tools(servers: list[TaskMCPServer]) -> tuple[dict[str, MCPTool], list[HeldMCPClient]]:
    """One NOOA tool object per task MCP server, by agent attribute, and the clients that hold their sessions.

    Every server is connected to once to list its tools. The task declares them for its tools, so a server that
    does not answer fails the run as a dependency outage rather than leave the model without them.
    """
    tools: dict[str, MCPTool] = {}
    held: list[HeldMCPClient] = []
    for server in servers:
        try:
            if server.transport == "stdio":
                listed = await MCPManager.create_stdio_server(server.name, command=server.command, args=server.args)
            else:
                listed = await MCPManager.create_url_server(server.name, server.url, transport=server.transport)
        except Exception as error:
            raise ConnectionError(f"The task's MCP server {server.name!r} did not connect: {error}") from error
        client = create_mcp_client(
            transport=server.transport, url=server.url, command=server.command, args=server.args or None
        )
        # The class NOOA generated from the server's tool list, on a client that keeps its session.
        held.append(HeldMCPClient(client, server.name))
        tools[server.attribute] = type(listed)(held[-1], server.name)
    return tools, held


async def close_task_tools(held: list[HeldMCPClient]) -> None:
    """End the held sessions. A session that does not close is logged: the run's result is already decided."""
    for client in held:
        try:
            await client.close()
        except Exception:
            LOGGER.warning("Could not close the task's MCP server %r", client.server_name, exc_info=True)


def agent_class_with_task_tools(agent_class: type[Any], tools: Mapping[str, MCPTool]) -> type[Any]:
    """A per-rollout agent subclass that carries the task's MCP tools as attributes.

    The tools sit on the class, so an agent the model delegates to has them too. The class is made for one
    rollout, so the tool objects are not shared between rollouts.
    """
    if not tools:
        return agent_class
    namespace: dict[str, Any] = {
        "__module__": agent_class.__module__,
        "__doc__": agent_class.__doc__,
        "__annotations__": {attribute: type(tool) for attribute, tool in tools.items()},
        **tools,
    }
    metaclass = type(agent_class)
    execution = getattr(agent_class, "_execution_config", None)
    if execution is None:
        return metaclass(f"{agent_class.__name__}WithTaskTools", (agent_class,), namespace)
    return metaclass(f"{agent_class.__name__}WithTaskTools", (agent_class,), namespace, execution=execution)


def name_task_skills(agent: Any, skills_dir: str | None) -> None:
    """Name the task's skills directory in the agent's prompt; NOOA has no loader for ``SKILL.md`` folders."""
    if skills_dir is not None:
        agent.context[TASK_SKILLS_CONTEXT] = f"Task skills are in {skills_dir}. Read the relevant SKILL.md files."
