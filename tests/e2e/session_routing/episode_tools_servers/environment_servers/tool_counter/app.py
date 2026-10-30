# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ``BaseEnvironmentServerWithTools`` whose episodes call their own tools through the server's public port.

Only ``test_episode_tools_routing_e2e.py`` runs it.
Its directory is on that test's PYTHONPATH, under the ``environment_servers`` namespace,
so uvicorn can import it as ``environment_servers.tool_counter.app`` in each worker.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Literal, Optional

import aiohttp
import httpx
from fastapi import Body
from pydantic import BaseModel

from nemo_gym.base_environment_server import CleanupContext
from nemo_gym.base_environment_server_with_tools import BaseEnvironmentServerWithTools
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.episode_tools import TypedToolset, episode_tool
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.tool_access import DirectHTTPToolAccess, MCPToolAccess


class CounterTask(BaseModel):
    calls: int = 0
    # Where to write the episode's grant and worker, for a test that needs them while the episode runs.
    grant_file: Optional[str] = None
    # How long the episode keeps serving its tools after its calls.
    hold_seconds: float = 0
    # How the episode calls its tools: as simple_agent does, or as an MCP-only harness such as Codex does.
    transport: Literal["direct_http", "mcp"] = "direct_http"


class CounterRequest(BaseEpisodeRequest[CounterTask]):
    pass


class CounterResult(BaseModel):
    owner_pid: int
    statuses: list[int]
    # The worker that ran each call, as the tool reported it; None for a call that did not run.
    ran_on: list[Optional[int]]
    total: int
    # The tools the episode listed over MCP; empty over direct HTTP.
    listed: list[str] = []


class CounterResponse(BaseEpisodeResponse[CounterResult]):
    pass


class Amount(BaseModel):
    amount: int


class Counter:
    def __init__(self) -> None:
        self.total = 0

    @episode_tool
    async def add(self, arguments: Amount) -> dict[str, int]:
        """Add an amount to the episode's total, and report the worker that ran the call."""
        self.total += arguments.amount
        return {"total": self.total, "pid": os.getpid()}


class ToolCounterServer(BaseEnvironmentServerWithTools[CounterRequest, CounterResponse]):
    request_model = CounterRequest
    response_model = CounterResponse

    async def run_episode(self, request: CounterRequest, cleanup: CleanupContext) -> CounterResponse:
        task = request.task.task_input
        counter = Counter()
        served = await self.serve_tools(request, TypedToolset(counter), cleanup)
        grant = next(access for access in served.tool_accesses if isinstance(access, DirectHTTPToolAccess))
        base_url = str(grant.base_url).rstrip("/")
        headers = dict(grant.headers)
        if task.grant_file is not None:
            Path(task.grant_file).write_text(
                json.dumps({"headers": headers, "base_url": base_url, "pid": os.getpid()})
            )
        statuses: list[int] = []
        ran_on: list[Optional[int]] = []
        listed: list[str] = []
        if task.transport == "mcp":
            mcp_grant = next(access for access in served.tool_accesses if isinstance(access, MCPToolAccess))
            listed = await _call_over_mcp(mcp_grant, task.calls, statuses, ran_on)
        else:
            # Not Gym's shared client: a new connection for each call, so the kernel hands each one to any worker.
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True)) as client:
                for _ in range(task.calls):
                    async with client.post(f"{base_url}/add", json={"amount": 1}, headers=headers) as reply:
                        statuses.append(reply.status)
                        ran_on.append((await reply.json()).get("pid"))
        await asyncio.sleep(task.hold_seconds)
        return CounterResponse(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=CounterResult(
                owner_pid=os.getpid(), statuses=statuses, ran_on=ran_on, total=counter.total, listed=listed
            ),
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return AggregateMetrics()


def _new_connection_per_request(
    headers: Optional[dict[str, str]] = None,
    timeout: Optional[httpx.Timeout] = None,
    auth: Optional[httpx.Auth] = None,
) -> httpx.AsyncClient:
    """The MCP SDK's client without keep-alive, so the kernel hands each MCP request to any worker."""
    return httpx.AsyncClient(
        headers=headers, timeout=timeout, auth=auth, limits=httpx.Limits(max_keepalive_connections=0)
    )


async def _call_over_mcp(
    grant: MCPToolAccess, calls: int, statuses: list[int], ran_on: list[Optional[int]]
) -> list[str]:
    """List the episode's tools and call ``add`` over MCP, as an MCP-only harness does; return the listed names."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(
        str(grant.connection.url),
        headers=dict(grant.connection.headers),
        httpx_client_factory=_new_connection_per_request,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = [tool.name for tool in (await session.list_tools()).tools]
            for _ in range(calls):
                result = await session.call_tool("add", {"amount": 1})
                statuses.append(500 if result.isError else 200)
                ran_on.append(None if result.isError else json.loads(result.content[0].text).get("pid"))
    return listed


if __name__ == "__main__":
    ToolCounterServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ToolCounterServer.run_webserver()  # noqa: F401
