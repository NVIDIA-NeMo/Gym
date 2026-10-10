# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpointing a resources server whose tools an agent harness calls over MCP, not with the session cookie."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator
from unittest.mock import MagicMock

import httpx
from omegaconf import DictConfig

from nemo_gym.base_resources_server import NEMO_GYM_MCP_SESSION_TOKEN_HEADER, BaseResourcesServerConfig
from nemo_gym.mcp_auto_exposure import maybe_auto_expose
from nemo_gym.server_utils import ServerClient
from tests.unit_tests.test_checkpoint_resources import AUTH, SEED, CounterServer, control, signal_waits, until


RPC_HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


@asynccontextmanager
async def mcp_server() -> AsyncIterator[tuple[CounterServer, httpx.AsyncClient]]:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="counter", expose_tools_over_mcp=True)
    server = CounterServer(config=config, server_client=server_client, counters={})
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://r") as client:
            yield server, client


async def seed(client: httpx.AsyncClient, rollout_id: str = "r") -> str:
    """Seed a session the way an agent harness does, and return only its MCP token: the harness keeps no cookie."""
    response = await client.post(f"/ng-rollout/{rollout_id}/seed_session", json=SEED)
    client.cookies.clear()
    return response.json()["mcp"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER]


async def increment(client: httpx.AsyncClient, token: str) -> httpx.Response:
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "increment", "arguments": {}}}
    headers = {**RPC_HEADERS, NEMO_GYM_MCP_SESSION_TOKEN_HEADER: token}
    return await client.post("/mcp", headers=headers, json=body)


def count(response: httpx.Response) -> int:
    result = response.json()["result"]
    assert result.get("isError") is not True, result
    return json.loads(result["content"][0]["text"])["count"]


async def test_a_tool_call_during_a_checkpoint_waits_for_resume_instead_of_failing(tmp_path: Path) -> None:
    async with mcp_server() as (server, client):
        token = await seed(client)
        assert count(await increment(client, token)) == 1
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        waiting_for_resume = signal_waits(server._checkpoint)
        waiting = asyncio.create_task(increment(client, token))
        await asyncio.wait_for(waiting_for_resume.wait(), 5)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        assert not waiting.done()
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)

        # The checkpoint holds the state from before the waiting call, which ran only after resume.
        assert commit.json()["manifest"]["record_count"] == 1
        assert count(await waiting) == 2


async def test_a_tool_call_in_flight_holds_up_prepare() -> None:
    async with mcp_server() as (server, client):
        token = await seed(client)
        server.gate = asyncio.Event()
        call = asyncio.create_task(increment(client, token))
        await until(lambda: server._checkpoint.inflight == 1)
        missed = await client.post("/ng-control/v1/checkpoint/prepare", json=control(timeout=0.2), headers=AUTH)
        server.gate.set()
        assert count(await call) == 1
        ready = await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)

    assert missed.json()["report"]["counts"] == {"inflight": 1, "sessions": 1}
    assert ready.json()["phase"] == "prepared"


async def test_a_retire_waits_for_a_tool_call_in_flight_over_mcp() -> None:
    async with mcp_server() as (server, client):
        token = await seed(client)
        server.gate = asyncio.Event()
        call = asyncio.create_task(increment(client, token))
        await until(lambda: server._checkpoint.inflight == 1)
        retire = asyncio.create_task(
            client.post(
                "/ng-control/v1/checkpoint/retire",
                json=control("retire", episode_ids=[{"rollout_id": "r"}]),
                headers=AUTH,
            )
        )
        # The retire refuses the session, then waits for the call in flight before freeing it.
        await until(lambda: server._checkpoint.status_extra()["retired_sessions"] == 1)
        waited = not retire.done() and server._checkpoint.readiness().counts["sessions"] == 1
        server.gate.set()
        retired = await retire
        await call

    assert waited and retired.status_code == 200
    assert server.counters == {}


async def test_the_harness_keeps_its_token_across_a_restore(tmp_path: Path) -> None:
    async with mcp_server() as (_, client):
        token = await seed(client)
        await increment(client, token)
        await increment(client, token)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)

    async with mcp_server() as (server, fresh):
        await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        waiting_for_resume = signal_waits(server._checkpoint)
        waiting = asyncio.create_task(increment(fresh, token))
        await asyncio.wait_for(waiting_for_resume.wait(), 5)
        assert not waiting.done()
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        assert count(await waiting) == 3


async def test_checkpoint_control_routes_are_not_tools() -> None:
    async with mcp_server() as (_, client):
        token = await seed(client)
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        listed = await client.post(
            "/mcp", headers={**RPC_HEADERS, NEMO_GYM_MCP_SESSION_TOKEN_HEADER: token}, json=body
        )

    assert [tool["name"] for tool in listed.json()["result"]["tools"]] == ["increment"]


async def test_a_late_tool_call_over_mcp_for_a_retired_attempt_is_refused_until_forget() -> None:
    async with mcp_server() as (server, client):
        token = await seed(client)
        await client.post(
            "/ng-control/v1/checkpoint/retire", json=control("retire", episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )
        # The harness sent this call before its agent was retired; it names only its session.
        late = await increment(client, token)
        await client.post("/ng-control/v1/checkpoint/forget", json=control("forget", rollout_ids=["r"]), headers=AUTH)

    assert late.status_code == 409 and late.json()["error"]["code"] == "stale_attempt"
    # Nothing was recreated for the retired session.
    assert server.counters == {}
