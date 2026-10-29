# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpointing a resources server whose tools an agent harness calls over MCP, not with the session cookie."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional
from unittest.mock import MagicMock

import httpx
from omegaconf import DictConfig

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId
from nemo_gym.server_runtime import NEMO_GYM_MCP_SESSION_TOKEN_HEADER, BaseResourcesServerConfig
from nemo_gym.server_runtime.base_resources_server import mcp_session_token_serializer
from nemo_gym.server_runtime.mcp_auto_exposure import maybe_auto_expose
from tests.unit_tests.server_runtime.test_checkpoint_resources import (
    AUTH,
    SEED,
    CounterServer,
    control,
    signal_waits,
    until,
)


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
    token, _ = await seed_session(client, rollout_id)
    return token


async def seed_session(
    client: httpx.AsyncClient, capture_key: str = "r", cookies: Optional[dict[str, str]] = None
) -> tuple[str, dict[str, str]]:
    """Seed as an environment server does, and return the MCP token it hands the harness and the session cookie it keeps.

    A continuation seeds again under its new attempt with the cookie, which claims the restored session.
    """
    client.cookies.clear()
    client.cookies.update(cookies or {})
    response = await client.post(f"/ng-rollout/{capture_key}/seed_session", json=SEED)
    assert response.status_code == 200, response.text
    client.cookies.clear()
    return response.json()["mcp"]["headers"][NEMO_GYM_MCP_SESSION_TOKEN_HEADER], dict(response.cookies)


async def checkpoint(client: httpx.AsyncClient, checkpoint_dir: Path, checkpoint_id: str = "c1") -> None:
    prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(checkpoint_id), headers=AUTH)
    assert prepared.json()["phase"] == "prepared", prepared.text
    committed = await client.post(
        "/ng-control/v1/checkpoint/commit",
        json=control(checkpoint_id, checkpoint_dir=str(checkpoint_dir)),
        headers=AUTH,
    )
    assert committed.status_code == 200, committed.text


async def restore(client: httpx.AsyncClient, checkpoint_dir: Path, episode: dict, *, resume: bool = True) -> None:
    restored = await client.post(
        "/ng-control/v1/checkpoint/restore",
        json=control("r1", checkpoint_dir=str(checkpoint_dir), episode_ids=[episode]),
        headers=AUTH,
    )
    assert restored.status_code == 200, restored.text
    if resume:
        await client.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)


def refused(response: httpx.Response) -> bool:
    return response.status_code == 409 and response.json()["error"]["code"] == "checkpoint_stale_attempt"


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


async def test_a_restore_refuses_the_replaced_attempt_s_token_and_a_seed_of_the_replacement_gets_one_that_works(
    tmp_path: Path,
) -> None:
    async with mcp_server() as (_, client):
        token, cookies = await seed_session(client)
        await increment(client, token)
        await increment(client, token)
        await checkpoint(client, tmp_path)

    async with mcp_server() as (server, fresh):
        await restore(fresh, tmp_path, {"rollout_id": "r"}, resume=False)
        # A harness of the replaced attempt that outlived the crash calls before resume: it waits, then is refused.
        waiting_for_resume = signal_waits(server._checkpoint)
        stale = asyncio.create_task(increment(fresh, token))
        await asyncio.wait_for(waiting_for_resume.wait(), 5)
        waited = not stale.done()
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        stale_reply = await stale
        after_stale = dict(server.counters)
        # The replacement seeds under its own attempt with the session's cookie, which claims the restored session.
        new_token, _ = await seed_session(fresh, "r-a1", cookies)
        continued = count(await increment(fresh, new_token))

    assert waited and refused(stale_reply)
    assert list(after_stale.values()) == [2]
    assert new_token != token
    # The new token continues the restored session.
    assert continued == 3 and list(server.counters.values()) == [3]


async def test_a_stale_harness_cannot_create_state_for_a_session_the_fresh_process_never_saw() -> None:
    async with mcp_server() as (_, client):
        token = await seed(client)
        await increment(client, token)

    # Gym restarted without restoring the rollout: the controller retires the crashed attempt and starts it over.
    async with mcp_server() as (server, fresh):
        await fresh.post(
            "/ng-control/v1/checkpoint/retire", json=control("retire", episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )
        late = await increment(fresh, token)

    assert refused(late)
    # Nothing was created for a session no seed bound to an episode, which nothing would ever free.
    assert server.counters == {}


async def test_a_token_without_an_attempt_is_refused_when_checkpointing_is_on() -> None:
    async with mcp_server() as (server, client):
        await seed(client)
        [session_id] = server.counters
        # As minted by a process without checkpointing, such as one before the restart.
        signer = mcp_session_token_serializer(server.get_session_middleware_key())
        unattributed = await increment(client, signer.dumps({"sid": session_id, "tools": None}))

    assert refused(unattributed)
    assert server.counters == {session_id: 0}


async def test_a_second_restore_refuses_the_first_replacement_s_token(tmp_path: Path) -> None:
    async with mcp_server() as (_, client):
        _, cookies = await seed_session(client)
        await checkpoint(client, tmp_path / "first")

    async with mcp_server() as (_, first):
        await restore(first, tmp_path / "first", {"rollout_id": "r"})
        replacement_token, cookies = await seed_session(first, "r-a1", cookies)
        await increment(first, replacement_token)
        await checkpoint(first, tmp_path / "second", "c2")

    async with mcp_server() as (server, second):
        await restore(second, tmp_path / "second", {"rollout_id": "r", "attempt": 1})
        stale = await increment(second, replacement_token)
        token, _ = await seed_session(second, "r-a2", cookies)
        continued = count(await increment(second, token))

    assert refused(stale)
    assert continued == 2 and list(server.counters.values()) == [2]


async def test_a_refused_stale_call_does_not_claim_a_restored_session(tmp_path: Path) -> None:
    async with mcp_server() as (_, client):
        token = await seed(client)
        await checkpoint(client, tmp_path)

    async with mcp_server() as (server, fresh):
        await restore(fresh, tmp_path, {"rollout_id": "r"})
        stale = await increment(fresh, token)
        # Still unclaimed, so a commit that no longer continues the episode deletes the session.
        unclaimed = await server._checkpoint.unclaimed()

    assert refused(stale)
    assert unclaimed == [EpisodeId(rollout_id="r", attempt=1)]


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
        # The harness sent this call before its agent was retired; its token names the retired attempt.
        late = await increment(client, token)
        await client.post("/ng-control/v1/checkpoint/forget", json=control("forget", rollout_ids=["r"]), headers=AUTH)

    assert late.status_code == 409 and late.json()["error"]["code"] == "checkpoint_stale_attempt"
    # Nothing was recreated for the retired session.
    assert server.counters == {}
