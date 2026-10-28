# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.terminal_bench_4.app import TerminalBench4ResourcesServer
from resources_servers.terminal_bench_4.models import SeedSessionResponse
from resources_servers.terminal_bench_4.tests.test_terminal_bench_science import (  # noqa: F401
    preparation,
    seed_body,
    server,
)


def body():
    task = preparation.load_manifest()["tasks"][0]
    return seed_body("terminal-bench-science/" + task["name"], task["ref"])


def request():
    return SimpleNamespace(session={SESSION_ID_KEY: "caller"})


async def test_close_before_seed_fences_late_allocation(server):
    assert await server.close_resources_session(request(), body()) == {"closed": True}
    with pytest.raises(HTTPException) as exc:
        await server.seed_session(request(), body())
    assert exc.value.status_code == 410
    assert server._sessions == {}


async def test_close_waits_for_racing_seed_then_stops_without_grading(server, monkeypatch):
    ready, proceed = asyncio.Event(), asyncio.Event()
    environment = SimpleNamespace(closed=False)

    async def stop():
        environment.closed = True

    environment.stop = AsyncMock(side_effect=stop)
    environment.resource_identities = lambda: []
    environment.session_id = "test"
    environment.cleanup_errors = []
    environment.resources = []

    async def prepare(self, session):
        session.environment = environment
        ready.set()
        await proceed.wait()
        session.seed_response = SeedSessionResponse(session_id=session.session_id)
        session.phase = "ready"

    monkeypatch.setattr(TerminalBench4ResourcesServer, "_prepare_session", prepare)
    task = asyncio.create_task(server.seed_session(request(), body()))
    await ready.wait()
    closing = asyncio.create_task(server.close_resources_session(request(), body()))
    await asyncio.sleep(0)
    assert not closing.done()
    proceed.set()
    await task
    assert await closing == {"closed": True}
    assert environment.closed
    environment.stop.assert_awaited_once()
    assert await server.close_resources_session(request(), body()) == {"closed": True}
    environment.stop.assert_awaited_once()
    session = next(iter(server._sessions.values()))
    assert session.phase == "closed" and session.verify_body is None


async def test_close_reports_stop_failure(server, monkeypatch):
    async def prepare(self, session):
        environment = SimpleNamespace(closed=False, session_id="test", cleanup_errors=[], resources=[])
        environment.stop = AsyncMock(side_effect=RuntimeError("provider unavailable"))
        environment.resource_identities = lambda: []
        session.environment = environment
        session.seed_response = SeedSessionResponse(session_id=session.session_id)
        session.phase = "ready"

    monkeypatch.setattr(TerminalBench4ResourcesServer, "_prepare_session", prepare)
    await server.seed_session(request(), body())
    with pytest.raises(HTTPException) as exc:
        await server.close_resources_session(request(), body())
    assert exc.value.status_code == 503
