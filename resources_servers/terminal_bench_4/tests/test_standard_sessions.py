# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.terminal_bench_4.app import TerminalBench4ResourcesServer
from resources_servers.terminal_bench_4.models import SandboxedVerifyResponse, SeedSessionResponse
from resources_servers.terminal_bench_4.tests.test_terminal_bench_science import server  # noqa: F401


def seed(server):
    name, task = next(iter(server._tasks.items()))
    return ResourcesSeedSessionRequest(
        resources_session_id="standard-session",
        episode_id={"rollout_id": "test", "attempt": 2},
        task_id={"taskset": "tb4-test", "task_id": name},
        task_data={"task_name": name, "task_ref": task["ref"], "dataset_ref": server._manifest["ref"]},
    )


async def prepared(self, session):
    session.phase = "ready"
    session.seed_response = SeedSessionResponse(
        session_id=session.session_id,
        instruction="Solve the task from its pinned package",
        sandbox_access={"connection": {"provider_config_ref": "cpu", "descriptor": {}}, "workdir": "/app"},
    )


@pytest.mark.parametrize(
    "status,metadata,reason,started",
    [
        ("completed", {}, "completed", True),
        ("incomplete", {}, "timeout", True),
        ("completed", {"miniswe_termination": "nonzero_exit", "miniswe_agent_started": "true"}, "nonzero_exit", True),
        ("failed", {}, "infrastructure_error", False),
    ],
)
async def test_standard_http_seed_verify_and_close(server, monkeypatch, status, metadata, reason, started):
    async def prepare(session):
        await prepared(server, session)

    preparation = AsyncMock(side_effect=prepare)
    monkeypatch.setattr(server, "_prepare_session", preparation)

    async def finalize(session):
        assert session.verify_body.termination.reason == reason
        assert session.verify_body.agent_started is started
        assert session.verify_body.responses_create_params.input == "Solve the task from its pinned package"
        session.verified_response = SandboxedVerifyResponse(
            **session.verify_body.model_dump(),
            reward=0,
            evaluation_completed=started,
            infrastructure_error=None if started else "agent failed",
        )

    monkeypatch.setattr(server, "_finalize_session", finalize)
    body = seed(server)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.setup_webserver()), base_url="http://test"
    ) as client:
        first = await client.post("/seed_session", json=body.model_dump(mode="json"))
        assert first.status_code == 200, first.text
        assert first.json()["resources_session_id"] == body.resources_session_id
        assert (await client.post("/seed_session", json=body.model_dump(mode="json"))).json() == first.json()
        assert preparation.await_count == 1
        verified = await client.post(
            "/verify",
            json={
                "responses_create_params": {"input": []},
                "response": {
                    "id": "r",
                    "object": "response",
                    "created_at": 0,
                    "model": "m",
                    "output": [],
                    "status": status,
                    "metadata": metadata,
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                },
            },
        )
        assert verified.status_code == 200, verified.text
        assert verified.json()["mask_sample"] is (not started)
        closed = await client.post(
            "/close_session",
            json={"resources_session_id": body.resources_session_id, "episode_id": body.episode_id.model_dump()},
        )
        assert closed.status_code == 200, closed.text
        assert closed.json() == {"resources_session_id": body.resources_session_id}
        assert (await client.post("/seed_session", json=body.model_dump(mode="json"))).status_code == 410


async def test_unknown_close_fences_seed_and_rebinding_fails(server, monkeypatch):
    monkeypatch.setattr(TerminalBench4ResourcesServer, "_prepare_session", prepared)
    body = seed(server)
    request = SimpleNamespace(session={SESSION_ID_KEY: "owner"})
    await server.seed_session(request, body)
    other = body.model_copy(update={"episode_id": body.episode_id.model_copy(update={"attempt": 3})})
    with pytest.raises(HTTPException, match="bound"):
        await server.seed_session(request, other)
    unknown = body.model_copy(update={"resources_session_id": "unknown"})
    await server.close_resources_session(
        request, ResourcesCloseSessionRequest(resources_session_id="unknown", episode_id=body.episode_id)
    )
    with pytest.raises(HTTPException) as error:
        await server.seed_session(request, unknown)
    assert error.value.status_code == 410


async def test_close_waits_for_preparation_and_rejects_late_seed(server, monkeypatch):
    ready, proceed = asyncio.Event(), asyncio.Event()

    async def prepare(self, session):
        ready.set()
        await proceed.wait()
        await prepared(self, session)

    monkeypatch.setattr(TerminalBench4ResourcesServer, "_prepare_session", prepare)
    body = seed(server)
    request = SimpleNamespace(session={SESSION_ID_KEY: "owner"})
    opening = asyncio.create_task(server.seed_session(request, body))
    await ready.wait()
    closing = asyncio.create_task(
        server.close_resources_session(
            request,
            ResourcesCloseSessionRequest(resources_session_id=body.resources_session_id, episode_id=body.episode_id),
        )
    )
    await asyncio.sleep(0)
    assert not closing.done()
    proceed.set()
    with pytest.raises(HTTPException) as error:
        await opening
    assert error.value.status_code == 410
    await closing
    assert next(iter(server._sessions.values())).phase == "closed"


async def test_standard_seed_configuration_failure_is_terminal(server, monkeypatch):
    async def prepare(session):
        session.phase = "closed"
        session.result["exception_info"] = {"exception_type": "ExecLimitConfigurationError"}
        session.seed_response = SeedSessionResponse(
            session_id=session.session_id,
            termination={"reason": "infrastructure_error", "detail": "exec limit is too short"},
        )

    monkeypatch.setattr(server, "_prepare_session", prepare)
    request = SimpleNamespace(session={SESSION_ID_KEY: "owner"})
    with pytest.raises(HTTPException) as error:
        await server.seed_session(request, seed(server))
    assert error.value.status_code == 422
