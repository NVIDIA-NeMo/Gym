# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI, Request
from omegaconf import DictConfig
from pydantic import JsonValue

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient


AUTH = {"authorization": "Bearer t"}


class CounterServer(SimpleResourcesServer):
    ray_enabled = False
    checkpoint_mode: ClassVar[str] = "exported"
    counters: dict[str, int] = {}
    gate: Any = None

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/increment")(self.increment)
        return app

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        self.counters.setdefault(request.session[SESSION_ID_KEY], 0)
        return BaseSeedSessionResponse()

    async def increment(self, request: Request) -> dict:
        if self.gate is not None:
            await self.gate.wait()
        session_id = request.session[SESSION_ID_KEY]
        self.counters[session_id] = self.counters.get(session_id, 0) + 1
        return {"count": self.counters[session_id]}

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)

    async def export_session_states(self, session_ids: list[str]) -> dict[str, JsonValue]:
        return {
            session_id: {"count": self.counters[session_id]}
            for session_id in session_ids
            if session_id in self.counters
        }

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        if any(not isinstance(state.get("count"), int) for state in states.values()):
            raise ValueError("invalid counter state")
        self.counters.update({session_id: state["count"] for session_id, state in states.items()})

    async def retire_session_state(self, session_id: str) -> None:
        self.counters.pop(session_id, None)


class RestartOnlyServer(CounterServer):
    checkpoint_mode: ClassVar[str] = "restart_only"


def make_server(server_type: type[CounterServer] = CounterServer) -> tuple[CounterServer, httpx.AsyncClient]:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="counter")
    server = server_type(config=config, server_client=server_client, counters={})
    app = server.setup_webserver()
    return server, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://r")


def control(checkpoint_id: str = "c1", *, timeout: float = 5, **extra) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + timeout, **extra}


SEED = {"responses_create_params": {"input": "hi"}}
# Every restore below continues the episodes its source checkpoint exported.
SCOPE = [{"rollout_id": "r"}, {"rollout_id": "r", "attempt": 1}]


async def test_seeded_session_state_is_committed_and_restored(tmp_path: Path) -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        await client.post("/increment")
        await client.post("/increment")
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        cookies = dict(client.cookies)

    [session_id] = server.counters
    assert commit.json()["manifest"]["record_count"] == 1

    restored, fresh_client = make_server()
    async with fresh_client:
        fresh_client.cookies.update(cookies)
        await fresh_client.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=SCOPE),
            headers=AUTH,
        )
        parked = await fresh_client.post("/increment")
        await fresh_client.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        after = (await fresh_client.post("/increment")).json()

    assert parked.status_code == 409 and parked.json()["error"]["code"] == "resources_admission_closed"
    assert after == {"count": 3}
    assert restored.counters == {session_id: 3}


async def test_in_flight_request_blocks_prepare_and_verify_ends_the_session() -> None:
    server, client = make_server()
    server.gate = asyncio.Event()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        pending = asyncio.create_task(client.post("/increment"))
        await asyncio.sleep(0.05)
        missed = (
            await client.post("/ng-control/v1/checkpoint/prepare", json=control(timeout=0.1), headers=AUTH)
        ).json()
        server.gate.set()
        await pending
        prepared = (await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)).json()
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        await client.post("/verify", json={"responses_create_params": {"input": "hi"}, "response": _response()})
        status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()

    assert missed["report"]["counts"] == {"inflight": 1, "sessions": 1}
    assert prepared["phase"] == "prepared"
    assert status["report"]["counts"]["sessions"] == 0


async def test_restart_only_sessions_block_until_their_rollout_is_retired() -> None:
    server, client = make_server(RestartOnlyServer)
    async with client:
        await client.post("/ng-rollout/r-a2/seed_session", json=SEED)
        blocked = (
            await client.post("/ng-control/v1/checkpoint/prepare", json=control(timeout=0.1), headers=AUTH)
        ).json()
        await client.post(
            "/ng-control/v1/checkpoint/retire",
            json=control(episode_ids=[{"rollout_id": "r", "attempt": 2}]),
            headers=AUTH,
        )
        prepared = (await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)).json()
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        stale = await client.post("/increment")

    assert blocked["report"]["blockers"] == ["r-a2"]
    assert prepared["phase"] == "prepared"
    # restart_only servers implement no hooks, so a retired session is only fenced, not exported or cleared.
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "stale_attempt"
    assert list(server.counters.values()) == [0]


async def test_invalid_restored_state_installs_nothing(tmp_path: Path) -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        server.counters[next(iter(server.counters))] = "corrupt"
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)

    restored, fresh_client = make_server()
    async with fresh_client:
        with pytest.raises(ValueError, match="invalid counter state"):
            await fresh_client.post(
                "/ng-control/v1/checkpoint/restore",
                json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=SCOPE),
                headers=AUTH,
            )
        status = (await fresh_client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()

    assert restored.counters == {}
    assert status["phase"] == "idle" and status["report"]["counts"]["sessions"] == 0


def _response() -> dict:
    return {
        "id": "resp",
        "created_at": 0.0,
        "model": "m",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


class StatelessServer(CounterServer):
    checkpoint_mode: ClassVar[str] = "stateless"

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        raise NotImplementedError


async def test_stateless_server_restores_without_hooks(tmp_path: Path) -> None:
    server, client = make_server(StatelessServer)
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)

    _, fresh_client = make_server(StatelessServer)
    async with fresh_client:
        restored = await fresh_client.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=SCOPE),
            headers=AUTH,
        )

    assert restored.json()["phase"] == "restored"


class ResetStepServer(CounterServer):
    """A Gymnasium-style protocol: /reset starts a session and a terminal /step ends it."""

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/reset")(self.reset)
        app.post("/step")(self.step)
        return app

    async def reset(self, request: Request) -> dict:
        self.counters[request.session[SESSION_ID_KEY]] = 0
        self.checkpoint_session_started(request)
        return {}

    async def step(self, request: Request, body: dict) -> dict:
        session_id = request.session[SESSION_ID_KEY]
        self.counters[session_id] += 1
        terminated = bool(body.get("terminal"))
        if terminated:
            self.counters.pop(session_id)
            self.checkpoint_session_ended(request)
        return {"terminated": terminated}


async def test_protocols_report_lifecycle_explicitly_when_no_route_reveals_it(tmp_path: Path) -> None:
    server, client = make_server(ResetStepServer)
    async with client:
        await client.post("/ng-rollout/live/reset")
        await client.post("/ng-rollout/live/step", json={})
        client.cookies.clear()
        await client.post("/ng-rollout/done/reset")
        await client.post("/ng-rollout/done/step", json={"terminal": True})
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit",
            json=control(checkpoint_dir=str(tmp_path), episode_ids=None),
            headers=AUTH,
        )

    # Only the live episode is exported; the terminated one ended through the explicit hook.
    assert commit.json()["episode_ids"] == ["live"]


@pytest.mark.parametrize("server_type", [CounterServer, RestartOnlyServer])
async def test_a_seed_arriving_after_admission_closed_is_admitted_but_not_checkpointed(
    tmp_path: Path, server_type: type[CounterServer]
) -> None:
    server, client = make_server(server_type)
    async with client:
        prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        # The environment server's seed step replays after a crash, so the checkpoint did not wait for it.
        seed = await client.post("/ng-rollout/r/seed_session", json=SEED)
        tool = await client.post("/increment")
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        # After resume the session is an ordinary live session again.
        await client.post("/increment")
        later = await client.post("/ng-control/v1/checkpoint/prepare", json=control("c2", timeout=0.2), headers=AUTH)

    assert prepared.json()["phase"] == "prepared"
    assert seed.status_code == 200
    assert tool.json()["error"]["code"] == "resources_admission_closed"
    assert commit.json()["manifest"]["record_count"] == 0
    if server_type is RestartOnlyServer:
        assert later.json()["report"]["blockers"] == ["r"]
    else:
        assert later.json()["phase"] == "prepared"


class ReplayVerifyServer(CounterServer):
    checkpoint_verify: ClassVar[str] = "replay"


@pytest.mark.parametrize("server_type, expected", [(CounterServer, "wait"), (ReplayVerifyServer, "replay")])
async def test_the_seed_reply_and_status_report_the_verify_mode(
    server_type: type[CounterServer], expected: str
) -> None:
    _, client = make_server(server_type)
    async with client:
        seed = await client.post("/ng-rollout/r/seed_session", json=SEED)
        status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()

    assert seed.headers["x-ng-checkpoint-verify"] == expected
    assert status["verify"] == expected


async def test_the_seed_reply_reports_nothing_when_checkpointing_is_off() -> None:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({})
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="counter")
    app = ReplayVerifyServer(config=config, server_client=server_client, counters={}).setup_webserver()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://r") as client:
        seed = await client.post("/seed_session", json=SEED)

    assert seed.status_code == 200
    assert "x-ng-checkpoint-verify" not in seed.headers


async def test_a_session_the_server_already_dropped_is_not_exported_and_does_not_fail_the_commit(
    tmp_path: Path,
) -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        # For example, a verifier that raised after cleaning up its state.
        server.counters.clear()
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )

    assert commit.status_code == 200
    assert commit.json()["manifest"]["record_count"] == 0
    assert server._checkpoint.readiness().counts["sessions"] == 0


async def test_a_close_during_a_checkpoint_waits_for_resume_and_then_ends_the_session() -> None:
    server, client = make_server()
    close = {"resources_session_id": "s", "episode_id": {"rollout_id": "r"}}
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        # An episode a controller retired runs its final cleanup while the checkpoint is open.
        closing = asyncio.create_task(client.post("/ng-rollout/r/close_session", json=close))
        await asyncio.sleep(0.1)
        waiting = not closing.done()
        # A waiting close is not in flight, so it does not hold up the checkpoint.
        prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        closed = await closing

    assert waiting and prepared.json()["phase"] == "prepared"
    assert closed.status_code == 200
    assert server._checkpoint.readiness().counts["sessions"] == 0


async def test_a_retired_session_can_still_be_closed_to_release_it() -> None:
    server, client = make_server(RestartOnlyServer)
    close = {"resources_session_id": "s", "episode_id": {"rollout_id": "r"}}
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post(
            "/ng-control/v1/checkpoint/retire", json=control(episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )
        tool = await client.post("/increment")
        # The retired episode's final cleanup: a restart_only server releases its state only through close.
        closed = await client.post("/ng-rollout/r/close_session", json=close)

    assert tool.status_code == 409 and tool.json()["error"]["code"] == "stale_attempt"
    assert closed.status_code == 200


async def test_a_commit_that_no_longer_continues_a_restored_session_releases_it(tmp_path: Path) -> None:
    _, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/increment")
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path / "first")), headers=AUTH
        )

    restored, fresh = make_server()
    async with fresh:
        await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path / "first"), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        restored_counters = dict(restored.counters)
        # The controller continues nothing from the next checkpoint: the restored session is never used.
        await fresh.post("/ng-control/v1/checkpoint/prepare", json=control("c2"), headers=AUTH)
        await fresh.post(
            "/ng-control/v1/checkpoint/commit",
            json=control("c2", checkpoint_dir=str(tmp_path / "second"), episode_ids=[]),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("c2"), headers=AUTH)

    assert list(restored_counters.values()) == [1]
    assert restored.counters == {}
    assert restored._checkpoint.readiness().counts["sessions"] == 0
