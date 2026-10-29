# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request
from omegaconf import DictConfig
from pydantic import JsonValue

from nemo_gym._checkpoint.control import CheckpointRequest
from nemo_gym._checkpoint.steps import CHECKPOINT_CLOSE_FOLLOWS_HEADER
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.core import EpisodeId
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient


AUTH = {"authorization": "Bearer t"}


class CounterServer(SimpleResourcesServer):
    ray_enabled = False
    checkpoint_mode: ClassVar[str] = "exported"
    counters: dict[str, int] = {}
    # The sessions each /close_session handler run closed, by cookie session ID.
    closed: list[str] = []
    gate: Any = None

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/increment")(self.increment)
        return app

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        self.counters.setdefault(request.session[SESSION_ID_KEY], 0)
        return BaseSeedSessionResponse()

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        self.closed.append(request.session[SESSION_ID_KEY])
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

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


VERIFY = {
    "responses_create_params": {"input": "hi"},
    "response": {
        "id": "resp",
        "created_at": 0.0,
        "model": "m",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    },
}
CLOSE_FOLLOWS = {CHECKPOINT_CLOSE_FOLLOWS_HEADER: "1"}


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


async def until(condition: Callable[[], bool], timeout: float = 5) -> None:
    """Yield to the event loop until ``condition`` holds, so a test never guesses how long work takes."""
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


def signal_waits(participant: Any) -> asyncio.Event:
    """Return an event set once a request starts waiting for admission to reopen."""
    waiting = asyncio.Event()
    wait_open = participant.wait_open

    async def signalling_wait_open() -> None:
        waiting.set()
        await wait_open()

    participant.wait_open = signalling_wait_open
    return waiting


SEED = {"responses_create_params": {"input": "hi"}}
# Every restore below continues the episodes its source checkpoint exported.
SCOPE = [{"rollout_id": "r"}, {"rollout_id": "r", "attempt": 1}]


async def test_seeded_session_state_is_committed_and_restored(tmp_path: Path) -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        await client.post("/ng-rollout/r-a1/increment")
        await client.post("/ng-rollout/r-a1/increment")
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
        # A request before resume waits for it instead of being refused.
        waiting_for_resume = signal_waits(restored._checkpoint)
        parked = asyncio.create_task(fresh_client.post("/ng-rollout/r-a2/increment"))
        await asyncio.wait_for(waiting_for_resume.wait(), 5)
        waited = not parked.done()
        await fresh_client.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        resumed = (await parked).json()
        after = (await fresh_client.post("/ng-rollout/r-a2/increment")).json()

    assert waited and resumed == {"count": 3}
    assert after == {"count": 4}
    assert restored.counters == {session_id: 4}


async def test_in_flight_request_blocks_prepare_and_verify_ends_the_session() -> None:
    server, client = make_server()
    server.gate = asyncio.Event()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        pending = asyncio.create_task(client.post("/increment"))
        await until(lambda: server._checkpoint.inflight == 1)
        missed = (
            await client.post("/ng-control/v1/checkpoint/prepare", json=control(timeout=0.1), headers=AUTH)
        ).json()
        server.gate.set()
        await pending
        prepared = (await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)).json()
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        await client.post(
            "/ng-rollout/r/verify", json={"responses_create_params": {"input": "hi"}, "response": _response()}
        )
        status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()

    assert missed["report"]["counts"] == {"inflight": 1, "sessions": 1}
    assert prepared["phase"] == "prepared"
    assert status["report"]["counts"]["sessions"] == 0


async def test_a_restart_only_session_is_a_restart_that_keeps_running_through_a_checkpoint(tmp_path: Path) -> None:
    server, client = make_server(RestartOnlyServer)
    async with client:
        seeded = await client.post("/ng-rollout/r-a2/seed_session", json=SEED)
        prepared = (await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)).json()
        # Nothing of this server is in the checkpoint, so its session's rollout keeps running through it.
        during = await client.post("/increment")
        refused = await client.post(
            "/ng-control/v1/checkpoint/commit",
            json=control(checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r", "attempt": 2}]),
            headers=AUTH,
        )
        committed = await client.post(
            "/ng-control/v1/checkpoint/commit",
            json=control(checkpoint_dir=str(tmp_path), episode_ids=[]),
            headers=AUTH,
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)

    # The seed reply tells the episode that seeded it to start over after a crash.
    assert seeded.headers["x-ng-checkpoint-restart"] == "1"
    assert prepared["phase"] == "prepared"
    assert prepared["report"]["restarts"] == ["r-a2"] and prepared["report"]["blockers"] == []
    assert during.status_code == 200
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "checkpoint_restart_in_scope"
    assert committed.status_code == 200 and committed.json()["manifest"]["record_count"] == 0


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

    checkpoint_session_starts: ClassVar[tuple[str, ...]] = ("/seed_session", "/reset")
    reset_gate: Any = None

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/reset")(self.reset)
        app.post("/step")(self.step)
        return app

    async def reset(self, request: Request) -> dict:
        if self.reset_gate is not None:
            await self.reset_gate.wait()
        self.counters[request.session[SESSION_ID_KEY]] = 0
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


class StatelessResetStepServer(ResetStepServer):
    checkpoint_mode: ClassVar[str] = "stateless"


class RestartOnlyResetStepServer(ResetStepServer):
    checkpoint_mode: ClassVar[str] = "restart_only"


@pytest.mark.parametrize("server_type", [ResetStepServer, StatelessResetStepServer, RestartOnlyResetStepServer])
async def test_a_declared_session_start_during_a_commit_s_export_waits_for_resume(
    tmp_path: Path, server_type: type[ResetStepServer]
) -> None:
    server, client = make_server(server_type)
    participant = server._checkpoint
    exporting, release_export = asyncio.Event(), asyncio.Event()
    export = participant.export

    async def held_export(episode_ids: Any) -> Any:
        exporting.set()
        await release_export.wait()
        return await export(episode_ids)

    participant.export = held_export
    async with client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        # The commit has checked its scope, which names r, and is exporting.
        commit = asyncio.create_task(
            client.post(
                "/ng-control/v1/checkpoint/commit",
                json=control(checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
                headers=AUTH,
            )
        )
        await asyncio.wait_for(exporting.wait(), 5)
        waiting_for_resume = signal_waits(participant)
        reset = asyncio.create_task(client.post("/ng-rollout/r/reset"))
        await until(lambda: waiting_for_resume.is_set() or reset.done())
        release_export.set()
        committed = await commit
        started_before_resume = reset.done() or participant.readiness().counts["sessions"] != 0
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        started = await reset

    assert not started_before_resume
    assert committed.status_code == 200 and committed.json()["episode_ids"] == []
    assert started.status_code == 200
    assert list(participant._sessions.values()) == [EpisodeId(rollout_id="r")]


async def test_a_restart_only_declared_session_start_in_flight_is_reported_as_a_restart() -> None:
    server, client = make_server(RestartOnlyResetStepServer)
    server.reset_gate = asyncio.Event()
    async with client:
        reset = asyncio.create_task(client.post("/ng-rollout/r/reset"))
        await until(lambda: server._checkpoint.inflight == 1)
        prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        server.reset_gate.set()
        started = await reset
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)

    assert prepared.json()["phase"] == "prepared" and prepared.json()["report"]["restarts"] == ["r"]
    # Every session start tells its caller what the seed reply does.
    assert started.headers["x-ng-checkpoint-restart"] == "1"


@pytest.mark.parametrize("path", ["/seed_session", "/reset"])
async def test_a_session_start_without_a_rollout_prefix_is_refused_before_its_handler_runs(path: str) -> None:
    server, client = make_server(ResetStepServer)
    async with client:
        refused = await client.post(path, json=SEED)
        status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()

    assert refused.status_code == 400 and refused.json()["error"]["code"] == "checkpoint_rollout_id_required"
    assert server.counters == {}
    assert status["report"]["counts"] == {"inflight": 0, "sessions": 0}


def _session_end_body(path: str, rollout_id: str) -> dict:
    if path == "/close_session":
        return {"resources_session_id": "s", "episode_id": {"rollout_id": rollout_id}}
    return {"responses_create_params": {"input": "hi"}, "response": _response()}


@pytest.mark.parametrize("path", ["/close_session", "/verify"])
async def test_a_session_end_without_a_rollout_prefix_is_refused_before_its_handler_runs(path: str) -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        refused = await client.post(path, json=_session_end_body(path, "r"))

    # Without the prefix, nothing could tell whether the request comes from the attempt that owns the session.
    assert refused.status_code == 400 and refused.json()["error"]["code"] == "checkpoint_rollout_id_required"
    assert server._checkpoint.readiness().counts["sessions"] == 1


class OwnAppServer(CounterServer):
    """A server whose setup_webserver builds its own app, as Gymnasium servers did."""

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/seed_session")(self.seed_session)
        return app


def test_a_resources_server_whose_app_left_out_the_checkpoint_participant_refuses_to_start() -> None:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="counter")
    server = OwnAppServer(config=config, server_client=server_client, counters={})
    app = server.setup_webserver()

    with pytest.raises(ValueError, match="OwnAppServer.setup_webserver did not install the checkpoint participant"):
        server.setup_checkpoint_participant(app)


def test_a_resources_server_whose_app_installs_the_participant_passes_the_startup_check() -> None:
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="counter")
    enabled = MagicMock(spec=ServerClient)
    enabled.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    disabled = MagicMock(spec=ServerClient)
    disabled.global_config_dict = DictConfig({})
    shared = CounterServer(config=config, server_client=enabled, counters={})
    own_app_without_checkpointing = OwnAppServer(config=config, server_client=disabled, counters={})

    shared.setup_checkpoint_participant(shared.setup_webserver())
    own_app_without_checkpointing.setup_checkpoint_participant(own_app_without_checkpointing.setup_webserver())

    assert shared._checkpoint is not None
    assert own_app_without_checkpointing._checkpoint is None


@pytest.mark.parametrize("server_type", [CounterServer, StatelessServer])
async def test_a_seed_arriving_during_an_open_checkpoint_waits_for_resume(
    tmp_path: Path, server_type: type[CounterServer]
) -> None:
    server, client = make_server(server_type)
    async with client:
        prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        # The environment server seeds inside a replay step, so the checkpoint did not wait for it;
        # the seed waits instead, and the environment's boundary stays before it.
        waiting_for_resume = signal_waits(server._checkpoint)
        seed = asyncio.create_task(client.post("/ng-rollout/r/seed_session", json=SEED))
        await asyncio.wait_for(waiting_for_resume.wait(), 5)
        waiting = not seed.done()
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        seeded = await seed
        later = await client.post("/ng-control/v1/checkpoint/prepare", json=control("c2", timeout=0.2), headers=AUTH)

    assert prepared.json()["phase"] == "prepared" and waiting
    assert commit.json()["manifest"]["record_count"] == 0
    assert seeded.status_code == 200
    # After resume the session is an ordinary live session.
    assert later.json()["phase"] == "prepared"


async def test_a_seed_in_flight_when_admission_closes_holds_up_prepare_and_is_exported(tmp_path: Path) -> None:
    class SlowSeedServer(CounterServer):
        release: Any = None

        async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
            await self.release.wait()
            return await super().seed_session(request, body)

    server, client = make_server(SlowSeedServer)
    server.release = asyncio.Event()
    async with client:
        seed = asyncio.create_task(client.post("/ng-rollout/r/seed_session", json=SEED))
        await until(lambda: server._checkpoint.inflight == 1)
        missed = await client.post("/ng-control/v1/checkpoint/prepare", json=control(timeout=0.1), headers=AUTH)
        server.release.set()
        await seed
        prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)

    assert missed.json()["phase"] == "preparing"
    assert prepared.json()["phase"] == "prepared"
    # The environment may record its boundary after the seed before the commit, so the session is exported.
    assert commit.json()["manifest"]["record_count"] == 1


class ReplayVerifyServer(CounterServer):
    checkpoint_verify: ClassVar[str] = "replay"


@pytest.mark.parametrize("server_type, expected", [(CounterServer, "wait"), (ReplayVerifyServer, "replay")])
async def test_the_seed_reply_and_status_report_the_verify_step_type(
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
        waiting_for_resume = signal_waits(server._checkpoint)
        closing = asyncio.create_task(client.post("/ng-rollout/r/close_session", json=close))
        await asyncio.wait_for(waiting_for_resume.wait(), 5)
        waiting = not closing.done()
        # A waiting close is not in flight, so it does not hold up the checkpoint.
        prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        closed = await closing

    assert waiting and prepared.json()["phase"] == "prepared"
    assert closed.status_code == 200
    assert server._checkpoint.readiness().counts["sessions"] == 0


async def test_a_retire_waits_for_the_sessions_request_in_flight() -> None:
    server, client = make_server()
    server.gate = asyncio.Event()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        tool = asyncio.create_task(client.post("/increment"))
        await until(lambda: server._checkpoint.inflight == 1)
        retire = asyncio.create_task(
            client.post(
                "/ng-control/v1/checkpoint/retire",
                json=control("retire", episode_ids=[{"rollout_id": "r"}]),
                headers=AUTH,
            )
        )
        # The retire refuses the session, then waits for the request in flight before freeing it.
        await until(lambda: server._checkpoint.status_extra()["retired_sessions"] == 1)
        waited = not retire.done() and server._checkpoint.readiness().counts["sessions"] == 1
        # A new request for the session while it stops is refused.
        refused = await client.post("/increment")
        server.gate.set()
        retired, finished = await retire, await tool

    assert waited and retired.status_code == 200 and finished.status_code == 200
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert server.counters == {} and server._checkpoint.readiness().counts["sessions"] == 0


async def test_a_commit_that_no_longer_continues_a_restored_session_retires_it(tmp_path: Path) -> None:
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
        committed = await fresh.post(
            "/ng-control/v1/checkpoint/commit",
            json=control("c2", checkpoint_dir=str(tmp_path / "second"), episode_ids=[]),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("c2"), headers=AUTH)

    assert list(restored_counters.values()) == [1]
    # Outside the scope, so not exported.
    assert committed.json()["episode_ids"] == []
    assert restored.counters == {}
    assert restored._checkpoint.readiness().counts["sessions"] == 0


async def test_late_requests_of_a_retired_attempt_are_refused_until_forget_but_a_close_is_not() -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        retired = await client.post(
            "/ng-control/v1/checkpoint/retire", json=control("retire", episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )
        # A tool call its caller sent before its own retire, carrying only the session cookie.
        late_tool = await client.post("/increment")
        # A seed for the retired attempt would create session state nothing frees.
        late_seed = await client.post("/ng-rollout/r/seed_session", json=SEED)
        close = await client.post(
            "/ng-rollout/r/close_session", json={"resources_session_id": "s", "episode_id": {"rollout_id": "r"}}
        )
        # The next attempt seeds its own session and is not affected.
        client.cookies.clear()
        next_seed = await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()
        forgotten = await client.post(
            "/ng-control/v1/checkpoint/forget", json=control("forget", rollout_ids=["r"]), headers=AUTH
        )
        client.cookies.clear()
        after_forget = await client.post("/ng-rollout/r/seed_session", json=SEED)

    assert retired.status_code == 200
    assert late_tool.status_code == 409 and late_tool.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert late_seed.status_code == 409 and late_seed.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert next_seed.status_code == 200 and close.status_code == 200
    assert status["retired_sessions"] == 1 and status["retired_rollouts"] == 1
    assert forgotten.status_code == 200 and after_forget.status_code == 200
    assert server._checkpoint.status_extra()["retired_sessions"] == 0


class SlowRetireServer(CounterServer):
    release: Any = None
    entered: Any = None

    async def retire_session_state(self, session_id: str) -> None:
        self.entered.set()
        await self.release.wait()
        await super().retire_session_state(session_id)


async def retire_cut_short_in_its_hook(
    client: httpx.AsyncClient, server: SlowRetireServer, retire: dict
) -> httpx.Response:
    """Retire with a deadline that passes inside the held hook, retrying if a stall spent it before the hook.

    A retry is idempotent: it marks the attempt again and goes straight back to the hook.
    """
    for _ in range(20):
        cut_short = await client.post(
            "/ng-control/v1/checkpoint/retire", json={**retire, "deadline_ts": time.time() + 0.1}, headers=AUTH
        )
        if server.entered.is_set():
            return cut_short
    pytest.fail("the retire never reached its hook")


async def test_a_session_whose_retire_was_cut_short_stays_until_a_retry_frees_it_and_blocks_meanwhile(
    tmp_path: Path,
) -> None:
    server, client = make_server(SlowRetireServer)
    server.release, server.entered = asyncio.Event(), asyncio.Event()
    retire = control("retire", episode_ids=[{"rollout_id": "r"}])
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        cut_short = await retire_cut_short_in_its_hook(client, server, retire)
        # The session is still being freed: the next checkpoint waits for it and does not export it.
        prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(timeout=0.1), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        server.release.set()
        retried = await client.post("/ng-control/v1/checkpoint/retire", json=retire, headers=AUTH)

    assert cut_short.json()["error"]["code"] == "checkpoint_deadline_exceeded"
    assert prepared.json()["phase"] == "preparing" and prepared.json()["report"]["blockers"] == ["r"]
    assert retried.status_code == 200 and server.counters == {}
    assert server._checkpoint.readiness().counts["sessions"] == 0


async def test_a_restart_only_server_s_seeds_wait_out_a_checkpoint_so_its_restarts_do_not_change(
    tmp_path: Path,
) -> None:
    server, client = make_server(RestartOnlyServer)
    async with client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        waiting_for_resume = signal_waits(server._checkpoint)
        seed = asyncio.create_task(client.post("/ng-rollout/r/seed_session", json=SEED))
        await asyncio.wait_for(waiting_for_resume.wait(), 5)
        # A seed asks to wait whether or not a checkpoint is open; give one that did not wait time to finish.
        await asyncio.sleep(0.05)
        waiting = not seed.done()
        restarts_during = server._checkpoint.readiness().restarts
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        seeded = await seed

    assert waiting and restarts_during == []
    assert seeded.status_code == 200


async def test_a_restored_session_the_export_drops_does_not_break_the_commit(tmp_path: Path) -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)

    restored, fresh = make_server()
    commit = control("c2", checkpoint_dir=str(tmp_path / "second"), episode_ids=[{"rollout_id": "r", "attempt": 1}])
    async with fresh:
        await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        # The server drops the restored session before anything used it, so its export leaves it out.
        restored.counters.clear()
        await fresh.post("/ng-control/v1/checkpoint/prepare", json=control("c2"), headers=AUTH)
        committed = await fresh.post("/ng-control/v1/checkpoint/commit", json=commit, headers=AUTH)

    assert committed.status_code == 200, committed.text
    assert committed.json()["episode_ids"] == []


class GatedSeedServer(CounterServer):
    seed_gate: Any = None
    seeding: Any = None

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        self.seeding.set()
        await self.seed_gate.wait()
        return await super().seed_session(request, body)


async def test_a_retire_waits_for_a_seed_of_its_attempt_and_frees_the_session_it_creates() -> None:
    server, client = make_server(GatedSeedServer)
    server.seed_gate, server.seeding = asyncio.Event(), asyncio.Event()
    async with client:
        seed = asyncio.create_task(client.post("/ng-rollout/r/seed_session", json=SEED))
        await server.seeding.wait()
        retire = asyncio.create_task(
            client.post(
                "/ng-control/v1/checkpoint/retire", json=control(episode_ids=[{"rollout_id": "r"}]), headers=AUTH
            )
        )
        await asyncio.sleep(0.05)
        waited = not retire.done()
        server.seed_gate.set()
        await seed
        retired = await retire

    assert waited
    assert retired.status_code == 200
    # Nothing of the retired attempt is left: the session the seed created was freed.
    assert server._checkpoint.readiness().counts["sessions"] == 0 and server.counters == {}


async def retire_cut_short_during_a_seed(client: httpx.AsyncClient, server: GatedSeedServer, retire: dict) -> Any:
    """Start a seed of the retired attempt, then a retire whose deadline passes while it waits for that seed."""
    server.seed_gate, server.seeding = asyncio.Event(), asyncio.Event()
    seed = asyncio.create_task(client.post("/ng-rollout/r/seed_session", json=SEED))
    await server.seeding.wait()
    cut_short = await client.post(
        "/ng-control/v1/checkpoint/retire", json={**retire, "deadline_ts": time.time() + 0.1}, headers=AUTH
    )
    return seed, cut_short


async def test_a_seed_that_finishes_after_its_retire_was_cut_short_registers_its_session_as_retired() -> None:
    server, client = make_server(GatedSeedServer)
    retire = control("retire", episode_ids=[{"rollout_id": "r"}])
    forget = control("forget", rollout_ids=["r"])
    async with client:
        seed, cut_short = await retire_cut_short_during_a_seed(client, server, retire)
        server.seed_gate.set()
        seeded = await seed
        # A tool call without a rollout prefix names only its session.
        late = await client.post("/increment")
        exported = await server._checkpoint.export(None)
        refused = await client.post("/ng-control/v1/checkpoint/forget", json=forget, headers=AUTH)
        retried = await client.post("/ng-control/v1/checkpoint/retire", json=retire, headers=AUTH)
        forgotten = await client.post("/ng-control/v1/checkpoint/forget", json=forget, headers=AUTH)

    assert cut_short.json()["error"]["code"] == "checkpoint_deadline_exceeded" and seeded.status_code == 200
    assert late.status_code == 409 and late.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert exported == []
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "checkpoint_retire_incomplete"
    # The retried retire frees the session the seed created.
    assert retried.status_code == 200 and server.counters == {}
    assert forgotten.status_code == 200 and server._checkpoint.readiness().counts["sessions"] == 0


class LateFailingSeedServer(CounterServer):
    """Seeds normally until a test installs ``seed_gate``; from then on each seed waits for it and then fails."""

    seed_gate: Any = None
    seeding: Any = None

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        if self.seed_gate is None:
            return await super().seed_session(request, body)
        self.seeding.set()
        await self.seed_gate.wait()
        raise HTTPException(status_code=500, detail="the seed failed")


async def test_a_retire_cut_short_while_a_seed_is_in_flight_still_marks_the_attempt_s_existing_session() -> None:
    server, client = make_server(LateFailingSeedServer)
    retire = control("retire", episode_ids=[{"rollout_id": "r"}])
    forget = control("forget", rollout_ids=["r"])
    async with client, httpx.AsyncClient(transport=client._transport, base_url="http://r") as other:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        server.seed_gate, server.seeding = asyncio.Event(), asyncio.Event()
        # A second seed of the same attempt, on its own session, is still running when the retire arrives.
        seed = asyncio.create_task(other.post("/ng-rollout/r/seed_session", json=SEED))
        await server.seeding.wait()
        cut_short = await client.post(
            "/ng-control/v1/checkpoint/retire", json={**retire, "deadline_ts": time.time() + 0.1}, headers=AUTH
        )
        server.seed_gate.set()
        failed = await seed
        # A tool call without a rollout prefix names only its session.
        late = await client.post("/increment")
        ready = server._checkpoint.ready()
        refused = await client.post("/ng-control/v1/checkpoint/forget", json=forget, headers=AUTH)
        retried = await client.post("/ng-control/v1/checkpoint/retire", json=retire, headers=AUTH)
        forgotten = await client.post("/ng-control/v1/checkpoint/forget", json=forget, headers=AUTH)

    assert cut_short.json()["error"]["code"] == "checkpoint_deadline_exceeded" and failed.status_code == 500
    # The session the attempt had before the retire is marked although the retire never got past the seed.
    assert late.status_code == 409 and late.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert not ready
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "checkpoint_retire_incomplete"
    assert retried.status_code == 200 and server.counters == {}
    assert forgotten.status_code == 200 and server._checkpoint.readiness().counts["sessions"] == 0


async def test_a_retire_that_arrives_after_its_deadline_still_marks_the_attempt_s_sessions() -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        # A retire that arrives after its deadline only marks the attempt: it stops and frees nothing.
        late_retire = await client.post(
            "/ng-control/v1/checkpoint/retire",
            json=control("retire", timeout=-1, episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        late = await client.post("/increment")
        refused = await client.post(
            "/ng-control/v1/checkpoint/forget", json=control("forget", rollout_ids=["r"]), headers=AUTH
        )

    assert late_retire.json()["error"]["code"] == "checkpoint_deadline_exceeded"
    assert late.status_code == 409 and late.json()["error"]["code"] == "checkpoint_stale_attempt"
    # Forgetting would lower the mark and leave the attempt's session live.
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "checkpoint_retire_incomplete"


async def test_a_session_is_open_from_its_seed_until_a_close_or_a_verify_ends_it() -> None:
    server, client = make_server()
    participant = server.checkpoint_participant
    close = {"resources_session_id": "s", "episode_id": {"rollout_id": "closed"}}
    verify = {"responses_create_params": {"input": "hi"}, "response": _response()}
    async with client, httpx.AsyncClient(transport=client._transport, base_url="http://r") as other:
        await client.post("/ng-rollout/closed/seed_session", json=SEED)
        await other.post("/ng-rollout/verified/seed_session", json=SEED)
        closed, verified = server.counters
        seeded = participant.is_open(closed) and participant.is_open(verified)
        await client.post("/ng-rollout/closed/close_session", json=close)
        await other.post("/ng-rollout/verified/verify", json=verify)

    assert seeded
    assert not participant.is_open(closed) and not participant.is_open(verified)
    assert not participant.is_open("never-seeded")


async def test_a_session_is_not_open_once_a_retire_marks_it_even_before_it_is_freed() -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        [session_id] = server.counters
        # A retire that arrives after its deadline only marks the attempt, and leaves its session to a retried retire.
        await client.post(
            "/ng-control/v1/checkpoint/retire",
            json=control("retire", timeout=-1, episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )

    assert server.counters == {session_id: 0}
    assert not server.checkpoint_participant.is_open(session_id)


async def test_a_restored_session_is_open_before_any_request_uses_it(tmp_path: Path) -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)
    [session_id] = server.counters

    restored, fresh = make_server()
    async with fresh:
        await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )

    assert restored.checkpoint_participant.is_open(session_id)


async def test_forget_is_refused_while_a_seed_of_a_retired_attempt_is_in_flight() -> None:
    server, client = make_server(GatedSeedServer)
    retire = control("retire", episode_ids=[{"rollout_id": "r"}])
    forget = control("forget", rollout_ids=["r"])
    async with client:
        seed, _ = await retire_cut_short_during_a_seed(client, server, retire)
        refused = await client.post("/ng-control/v1/checkpoint/forget", json=forget, headers=AUTH)
        server.seed_gate.set()
        await seed
        late = await client.post("/increment")

    # Forgetting now would let the seed bind its session as live once it finishes.
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "checkpoint_retire_incomplete"
    assert late.status_code == 409 and late.json()["error"]["code"] == "checkpoint_stale_attempt"


async def test_forget_is_refused_while_a_retire_has_not_freed_its_sessions() -> None:
    server, client = make_server(SlowRetireServer)
    server.release, server.entered = asyncio.Event(), asyncio.Event()
    retire = control("retire", episode_ids=[{"rollout_id": "r"}])
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await retire_cut_short_in_its_hook(client, server, retire)
        refused = await client.post("/ng-control/v1/checkpoint/forget", json=control(rollout_ids=["r"]), headers=AUTH)
        # Still refused: forgetting would have made the half-retired session live again.
        late = await client.post("/increment")
        server.release.set()
        await client.post("/ng-control/v1/checkpoint/retire", json=retire, headers=AUTH)
        forgotten = await client.post(
            "/ng-control/v1/checkpoint/forget", json=control(rollout_ids=["r"]), headers=AUTH
        )

    assert refused.status_code == 409 and refused.json()["error"]["code"] == "checkpoint_retire_incomplete"
    assert late.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert forgotten.status_code == 200


async def test_a_request_cancelled_while_notifying_is_no_longer_counted() -> None:
    from nemo_gym._checkpoint.resources import ResourcesCheckpointMiddleware

    server, _ = make_server()
    participant = server._checkpoint

    async def app(scope: dict, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    async def cancelled() -> None:
        raise asyncio.CancelledError

    participant.notify = cancelled
    middleware = ResourcesCheckpointMiddleware(app, participant=participant)
    scope = {"type": "http", "method": "POST", "path": "/increment", "headers": [], "session": {SESSION_ID_KEY: "s"}}

    async def send(message: dict) -> None:
        pass

    with pytest.raises(asyncio.CancelledError):
        await middleware(scope, None, send)

    # Otherwise every later prepare would wait for a request that ended long ago.
    assert participant.inflight == 0 and participant.ready()


async def test_a_stateless_server_drains_and_refuses_a_retired_session() -> None:
    server, client = make_server(StatelessServer)
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post(
            "/ng-control/v1/checkpoint/retire", json=control(episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )
        # A tool call without a rollout prefix names only its session.
        late = await client.post("/increment")

    assert late.status_code == 409 and late.json()["error"]["code"] == "checkpoint_stale_attempt"


async def test_deleting_an_unclaimed_restored_session_refuses_nothing_later(tmp_path: Path) -> None:
    _, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)

    restored, fresh = make_server()
    async with fresh:
        await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        await fresh.post("/ng-control/v1/checkpoint/prepare", json=control("c2"), headers=AUTH)
        await fresh.post(
            "/ng-control/v1/checkpoint/commit",
            json=control("c2", checkpoint_dir=str(tmp_path / "second"), episode_ids=[]),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("c2"), headers=AUTH)
        status = (await fresh.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()
        # The controller may still start the rollout over as the attempt the restore installed.
        fresh.cookies.clear()
        reseeded = await fresh.post("/ng-rollout/r-a1/seed_session", json=SEED)

    # Only the attempt the restore replaced is refused, until the controller forgets the rollout.
    assert status["retired_sessions"] == 0 and status["retired_rollouts"] == 1
    assert reseeded.status_code == 200


def test_resources_admission_refusals_are_the_shared_admission_closed_error() -> None:
    from nemo_gym._checkpoint.errors import AdmissionClosedError

    server, _ = make_server()
    server._checkpoint.accepting = False
    with pytest.raises(AdmissionClosedError):
        server._checkpoint.admit("s", "/increment")


def test_mcp_session_tokens_are_decoded_with_one_serializer_per_server(monkeypatch: pytest.MonkeyPatch) -> None:
    import itsdangerous

    from nemo_gym._checkpoint.resources import MCPTokenClaims
    from nemo_gym.base_resources_server import mcp_session_token_serializer

    server, _ = make_server()
    signer = mcp_session_token_serializer(server.get_session_middleware_key())
    token = signer.dumps({"sid": "s", "episode": "r-a1"})
    built: list[str] = []
    original = itsdangerous.URLSafeSerializer.__init__

    def counting(self: Any, *args: Any, **kwargs: Any) -> None:
        built.append("serializer")
        original(self, *args, **kwargs)

    monkeypatch.setattr(itsdangerous.URLSafeSerializer, "__init__", counting)

    def scope(value: bytes) -> dict:
        return {"headers": [(b"x-nemo-gym-session-token", value)]}

    claims = [server._mcp_token_claims(scope(token.encode())) for _ in range(3)]
    unsigned = server._mcp_token_claims(scope(b"not-a-token"))
    undecodable = server._mcp_token_claims(scope(b"\xff\xfe"))
    not_a_dict = server._mcp_token_claims(scope(signer.dumps([1]).encode()))
    without_attempt = server._mcp_token_claims(scope(signer.dumps({"sid": "s", "tools": None}).encode()))
    malformed_attempt = server._mcp_token_claims(scope(signer.dumps({"sid": "s", "episode": "-a1"}).encode()))

    assert claims == [MCPTokenClaims("s", EpisodeId(rollout_id="r", attempt=1))] * 3
    assert unsigned is None and undecodable is None and not_a_dict is None
    # A token that names no attempt, or one whose attempt does not decode, still names its session.
    assert without_attempt == malformed_attempt == MCPTokenClaims("s", None)
    # Only this server's own signer is built while counting.
    assert built.count("serializer") == 1


def test_an_exported_server_without_its_hooks_fails_at_startup() -> None:
    class Unfinished(SimpleResourcesServer):
        ray_enabled = False
        checkpoint_mode: ClassVar[str] = "exported"

        async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
            return BaseVerifyResponse(**body.model_dump(), reward=1.0)

    class FromAMixin(CounterServer):
        """Hooks inherited from an intermediate base class count."""

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="counter")
    with pytest.raises(ValueError, match="does not implement"):
        Unfinished(config=config, server_client=server_client).setup_webserver()
    FromAMixin(config=config, server_client=server_client, counters={}).setup_webserver()


async def test_retiring_one_rollout_does_not_scan_every_session() -> None:
    server, _ = make_server()
    participant = server._checkpoint
    for index in range(1000):
        participant.seeded(f"s{index}", EpisodeId(rollout_id=f"r{index}"))

    class Unscannable(dict):
        def items(self):  # noqa: ANN202
            raise AssertionError("retire scanned every session")

        def __iter__(self):  # noqa: ANN204
            raise AssertionError("retire scanned every session")

    participant._sessions = Unscannable(participant._sessions)
    await participant.retire(EpisodeId(rollout_id="r7"))

    assert "s7" not in participant._sessions and len(participant._sessions) == 999


async def test_deleting_one_rollout_s_restored_sessions_does_not_scan_every_restored_session() -> None:
    server, _ = make_server()
    participant = server._checkpoint
    for index in range(1000):
        participant.seeded(f"s{index}", EpisodeId(rollout_id=f"r{index}", attempt=1))
    participant._unclaimed.update(participant._sessions)

    class Unscannable(set):
        def __iter__(self):  # noqa: ANN204
            raise AssertionError("deleting a restored episode scanned every restored session")

    participant._unclaimed = Unscannable(participant._unclaimed)
    await participant.delete_restored(EpisodeId(rollout_id="r7", attempt=1))

    assert "s7" not in participant._sessions and len(participant._sessions) == 999
    assert "s7" not in participant._unclaimed and "s8" in participant._unclaimed


def test_ready_does_not_list_blockers() -> None:
    server, _ = make_server()
    participant = server._checkpoint
    participant.seeded("s", EpisodeId(rollout_id="r"))

    def listing() -> None:
        raise AssertionError("ready() built the full readiness report")

    participant.readiness = listing
    assert participant.ready()
    participant.inflight = 1
    assert not participant.ready()


class ReplacingSeedServer(CounterServer):
    """Seeds under the caller's resources session ID, as interactive_browser and swebench_pro do."""

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        request.session[SESSION_ID_KEY] = "caller-session"
        return await super().seed_session(request, body)


async def test_a_session_whose_id_the_seed_replaces_is_tracked_under_its_new_id() -> None:
    server, client = make_server(ReplacingSeedServer)
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        tracked = server._checkpoint.readiness().counts["sessions"]
        await client.post(
            "/ng-rollout/r/close_session",
            json={"resources_session_id": "caller-session", "episode_id": {"rollout_id": "r"}},
        )

    assert tracked == 1
    # The close names the new ID, so nothing of the finished session is left behind.
    assert server._checkpoint.readiness().counts["sessions"] == 0


async def test_a_seed_waiting_out_a_restore_does_not_hold_up_a_retire_of_its_attempt(tmp_path: Path) -> None:
    _, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)

    restored, fresh = make_server()
    async with fresh:
        await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        # A retried seed of the restored attempt arrives while admission is still closed, and waits for resume.
        waiting_for_resume = signal_waits(restored._checkpoint)
        seed = asyncio.create_task(fresh.post("/ng-rollout/r-a1/seed_session", json=SEED))
        await asyncio.wait_for(waiting_for_resume.wait(), 5)
        retired = await asyncio.wait_for(
            fresh.post(
                "/ng-control/v1/checkpoint/retire",
                json=control("r1", episode_ids=[{"rollout_id": "r", "attempt": 1}], timeout=2),
                headers=AUTH,
            ),
            timeout=3,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        seeded = await seed

    assert retired.status_code == 200, retired.text
    assert seeded.status_code == 409 and seeded.json()["error"]["code"] == "checkpoint_stale_attempt"


class FixedIdSeedServer(CounterServer):
    """Seeds under a caller-assigned session ID, so a replayed seed names the session it created before."""

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        request.session[SESSION_ID_KEY] = "caller-session"
        return await super().seed_session(request, body)


async def test_a_seed_of_a_restored_session_s_id_claims_it(tmp_path: Path) -> None:
    _, client = make_server(FixedIdSeedServer)
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)

    restored, fresh = make_server(FixedIdSeedServer)
    async with fresh:
        await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        pending_before = await restored._checkpoint.unclaimed()
        # The environment continues from before its seed, so it seeds the same session again.
        await fresh.post("/ng-rollout/r-a1/seed_session", json=SEED)
        pending_after = await restored._checkpoint.unclaimed()

    assert pending_before == [EpisodeId(rollout_id="r", attempt=1)]
    # Claimed, so a commit never deletes it as unused restored state.
    assert pending_after == []


async def test_after_a_restore_a_late_request_of_the_replaced_attempt_is_refused(tmp_path: Path) -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)
        cookies = dict(client.cookies)

    restored, fresh_client = make_server()
    # This server holds no record of "unrecorded", which the restore continues all the same.
    scope = [{"rollout_id": "r"}, {"rollout_id": "unrecorded"}]
    async with fresh_client:
        fresh_client.cookies.update(cookies)
        await fresh_client.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=scope),
            headers=AUTH,
        )
        await fresh_client.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        # A harness of the crashed attempt that outlived the crash uses the session restored for its replacement.
        late = await fresh_client.post("/ng-rollout/r/increment")
        late_seed = await fresh_client.post("/ng-rollout/unrecorded/seed_session", json=SEED)
        replacement = await fresh_client.post("/ng-rollout/r-a1/increment")

    for refused in (late, late_seed):
        assert refused.status_code == 409 and refused.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert replacement.json() == {"count": 1}


async def test_after_a_restore_a_close_from_the_replaced_attempt_is_refused_and_keeps_the_session(
    tmp_path: Path,
) -> None:
    _, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/increment")
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path / "first")), headers=AUTH
        )
        cookies = dict(client.cookies)

    restored, fresh = make_server()
    stale = {"resources_session_id": "s", "episode_id": {"rollout_id": "r"}}
    own = {"resources_session_id": "s", "episode_id": {"rollout_id": "r", "attempt": 1}}
    commit = control("c2", checkpoint_dir=str(tmp_path / "second"), episode_ids=[{"rollout_id": "r", "attempt": 1}])
    async with fresh:
        fresh.cookies.update(cookies)
        await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path / "first"), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        # A harness of the crashed attempt that outlived the crash ends its episode on the restored session.
        stale_close = await fresh.post("/ng-rollout/r/close_session", json=stale)
        await fresh.post("/ng-control/v1/checkpoint/prepare", json=control("c2"), headers=AUTH)
        committed = await fresh.post("/ng-control/v1/checkpoint/commit", json=commit, headers=AUTH)
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("c2"), headers=AUTH)
        # The replacement owns the session and still closes it.
        own_close = await fresh.post("/ng-rollout/r-a1/close_session", json=own)

    assert stale_close.status_code == 409 and stale_close.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert committed.json()["manifest"]["record_count"] == 1
    assert own_close.status_code == 200 and restored._checkpoint.readiness().counts["sessions"] == 0


@pytest.mark.parametrize("path", ["/close_session", "/verify"])
async def test_a_request_on_a_session_from_another_rollout_is_refused(path: str) -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        refused = await client.post(f"/ng-rollout/q{path}", json=_session_end_body(path, "q"))

    assert refused.status_code == 409 and refused.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert server._checkpoint.readiness().counts["sessions"] == 1


async def test_a_restart_only_seed_in_flight_when_admission_closes_is_reported_as_a_restart(tmp_path: Path) -> None:
    class SlowSeedServer(RestartOnlyServer):
        release: Any = None

        async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
            await self.release.wait()
            return await super().seed_session(request, body)

    server, client = make_server(SlowSeedServer)
    server.release = asyncio.Event()
    async with client:
        seed = asyncio.create_task(client.post("/ng-rollout/r/seed_session", json=SEED))
        await until(lambda: server._checkpoint.inflight == 1)
        # The server never holds up prepare, but its seed reply will say restart, so prepare says so already.
        prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        server.release.set()
        seeded = await seed
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit",
            json=control(checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "s"}]),
            headers=AUTH,
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)

    assert prepared.json()["phase"] == "prepared" and prepared.json()["report"]["restarts"] == ["r"]
    assert seeded.headers["x-ng-checkpoint-restart"] == "1"
    assert commit.status_code == 200, commit.text


async def test_a_restart_only_seed_waits_again_when_a_new_checkpoint_closes_before_it_resumes() -> None:
    server, client = make_server(RestartOnlyServer)
    participant = server._checkpoint
    waits: list[None] = []
    wait_open = participant._open.wait

    async def counting_wait() -> bool:
        waits.append(None)
        return await wait_open()

    participant._open.wait = counting_wait
    async with client:
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        seed = asyncio.create_task(client.post("/ng-rollout/r/seed_session", json=SEED))
        await until(lambda: len(waits) == 1)
        # The checkpoint resumes and a new one closes admission before the waiting seed runs again.
        await participant.open_admission()
        await participant.close_admission(CheckpointRequest(**control("c2")))
        await until(lambda: len(waits) == 2 or seed.done())
        admitted_during_the_second = seed.done() or participant.readiness().restarts != []
        await participant.open_admission()
        seeded = await seed

    assert not admitted_during_the_second
    assert seeded.status_code == 200


@pytest.mark.parametrize("path", ["/ng-rollout/r/seed_session", "/close_session", "/verify"])
async def test_a_session_change_survives_a_disconnect_once_the_reply_has_started(path: str) -> None:
    from nemo_gym._checkpoint.resources import ResourcesCheckpointMiddleware
    from nemo_gym.core import rollout_context

    server, _ = make_server()
    participant = server._checkpoint
    participant.seeded("s", EpisodeId(rollout_id="q"))

    async def app(scope: dict, receive: Any, send: Any) -> None:
        if path.endswith("/seed_session"):
            # The handler replaces the session's ID, as a seed with the caller's resources session ID does.
            scope["session"][SESSION_ID_KEY] = "seeded"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    async def disconnected(message: dict) -> None:
        # The client went away: the reply cannot be sent, and the request is cancelled.
        raise asyncio.CancelledError

    middleware = ResourcesCheckpointMiddleware(app, participant=participant)
    scope = {
        "type": "http",
        "method": "POST",
        "path": path.removeprefix("/ng-rollout/r"),
        "headers": [],
        "session": {SESSION_ID_KEY: "s"},
    }
    # A seed starts a session for its own rollout; a session end comes from the rollout that owns "s".
    with rollout_context("r" if path.endswith("/seed_session") else "q"):
        with pytest.raises(asyncio.CancelledError):
            await middleware(scope, None, disconnected)

    tracked = set(participant._sessions)
    assert tracked == ({"s", "seeded"} if path.endswith("/seed_session") else set())


@pytest.mark.parametrize("name, value", [("checkpoint_mode", "exportd"), ("checkpoint_verify", "replayed")])
def test_an_unknown_checkpoint_declaration_fails_at_startup(name: str, value: str) -> None:
    server_type = type("Typo", (CounterServer,), {name: value, "__annotations__": {name: ClassVar[str]}})
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="counter")

    with pytest.raises(ValueError, match=f"{name}={value!r}"):
        server_type(config=config, server_client=server_client, counters={}).setup_webserver()


@pytest.mark.parametrize("server_type", [RestartOnlyServer, StatelessServer])
async def test_a_retire_frees_the_sessions_of_a_server_that_does_not_export_them(
    server_type: type[CounterServer],
) -> None:
    server, client = make_server(server_type)
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/increment")
        retired = await client.post(
            "/ng-control/v1/checkpoint/retire", json=control(episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )
        # A request the retired attempt had already sent arrives late.
        late = await client.post("/increment")

    assert retired.status_code == 200
    assert late.json()["error"]["code"] == "checkpoint_stale_attempt"
    # Nothing else would free it: the late request that would have ended the session is refused.
    assert server.counters == {}


async def test_live_sessions_outside_the_commit_scope_are_not_exported(tmp_path: Path) -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        client.cookies.clear()
        await client.post("/ng-rollout/q/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        # Another participant reported q as a restart, so the coordinator commits r alone.
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit",
            json=control(checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)

    assert commit.json()["episode_ids"] == ["r"]


class GatedVerifyServer(ReplayVerifyServer):
    verify_gate: Any = None
    verifying: Any = None

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        self.verifying.set()
        await self.verify_gate.wait()
        return await super().verify(body)


async def test_a_replayable_verify_that_ends_between_environment_and_resources_prepare_is_exported_and_replays(
    tmp_path: Path,
) -> None:
    server, client = make_server(GatedVerifyServer)
    server.verify_gate, server.verifying = asyncio.Event(), asyncio.Event()
    close = {"resources_session_id": "s", "episode_id": {"rollout_id": "r", "attempt": 1}}
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        cookies = dict(client.cookies)
        verify = asyncio.create_task(client.post("/ng-rollout/r/verify", json=VERIFY, headers=CLOSE_FOLLOWS))
        await server.verifying.wait()
        # Coordination prepares the environment first.
        # It records the episode's boundary before this verification, which a restore runs again,
        # and it does not reach this server, whose admission is still open.
        server.verify_gate.set()
        verified = await verify
        prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        committed = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)

    restored, fresh = make_server(GatedVerifyServer)
    restored.verify_gate, restored.verifying = asyncio.Event(), asyncio.Event()
    restored.verify_gate.set()
    async with fresh:
        fresh.cookies.update(cookies)
        await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        # The replacement continues from the boundary before the verification and runs it again.
        replayed = await fresh.post("/ng-rollout/r-a1/verify", json=VERIFY, headers=CLOSE_FOLLOWS)
        closed = await fresh.post("/ng-rollout/r-a1/close_session", json=close)

    assert verified.status_code == 200 and prepared.json()["phase"] == "prepared"
    assert committed.json()["episode_ids"] == ["r"] and committed.json()["manifest"]["record_count"] == 1
    assert replayed.status_code == 200 and replayed.json()["reward"] == 1.0
    assert closed.status_code == 200 and restored.closed == list(restored.counters)
    assert restored.checkpoint_participant.readiness().counts["sessions"] == 0


async def test_a_replayable_verify_whose_caller_never_closes_ends_the_session_even_during_a_checkpoint(
    tmp_path: Path,
) -> None:
    server, client = make_server(ReplayVerifyServer)
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        # A caller that never closes, such as a restart-only agent's own /run, which no checkpoint continues.
        verified = await client.post("/ng-rollout/r/verify", json=VERIFY)
        during = server._checkpoint.readiness().counts["sessions"]
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)

    assert verified.status_code == 200
    # Nothing else would end it, so it ends now rather than being tracked forever.
    assert during == 0


async def test_after_a_restore_only_the_replacement_s_prefixed_calls_reach_the_session(tmp_path: Path) -> None:
    _, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-rollout/r/increment")
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)
        cookies = dict(client.cookies)

    restored, fresh = make_server()
    async with fresh:
        fresh.cookies.update(cookies)
        await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        # A harness of the replaced attempt, with and without its rollout prefix.
        stale = await fresh.post("/ng-rollout/r/increment")
        cookie_only = await fresh.post("/increment")
        unclaimed = await restored.checkpoint_participant.unclaimed()
        counts_after_refusals = dict(restored.counters)
        replacement = await fresh.post("/ng-rollout/r-a1/increment")

    [session_id] = restored.counters
    assert stale.status_code == 409 and stale.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert cookie_only.status_code == 400 and cookie_only.json()["error"]["code"] == "checkpoint_rollout_id_required"
    # Neither refused call changed the restored state or claimed the session,
    # so a commit that drops the episode would still delete it.
    assert counts_after_refusals == {session_id: 1} and unclaimed == [EpisodeId(rollout_id="r", attempt=1)]
    assert replacement.json() == {"count": 2}
    assert await restored.checkpoint_participant.unclaimed() == []


async def test_stale_and_cookie_only_calls_stay_refused_across_a_second_restore(tmp_path: Path) -> None:
    _, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path / "first")), headers=AUTH
        )
        cookies = dict(client.cookies)

    _, first = make_server()
    async with first:
        first.cookies.update(cookies)
        await first.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path / "first"), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        await first.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        await first.post("/ng-rollout/r-a1/increment")
        await first.post("/ng-control/v1/checkpoint/prepare", json=control("c2"), headers=AUTH)
        await first.post(
            "/ng-control/v1/checkpoint/commit",
            json=control(
                "c2", checkpoint_dir=str(tmp_path / "second"), episode_ids=[{"rollout_id": "r", "attempt": 1}]
            ),
            headers=AUTH,
        )

    restored, second = make_server()
    async with second:
        second.cookies.update(cookies)
        await second.post(
            "/ng-control/v1/checkpoint/restore",
            json=control(
                "r2", checkpoint_dir=str(tmp_path / "second"), episode_ids=[{"rollout_id": "r", "attempt": 1}]
            ),
            headers=AUTH,
        )
        await second.post("/ng-control/v1/checkpoint/resume", json=control("r2"), headers=AUTH)
        refused = [
            await second.post("/ng-rollout/r/increment"),
            await second.post("/ng-rollout/r-a1/increment"),
            await second.post("/increment"),
        ]
        unclaimed = await restored.checkpoint_participant.unclaimed()
        replacement = await second.post("/ng-rollout/r-a2/increment")

    assert [response.json()["error"]["code"] for response in refused] == [
        "checkpoint_stale_attempt",
        "checkpoint_stale_attempt",
        "checkpoint_rollout_id_required",
    ]
    assert unclaimed == [EpisodeId(rollout_id="r", attempt=2)]
    assert replacement.json() == {"count": 2}


async def test_a_replayable_verify_whose_caller_closes_the_session_afterwards_leaves_it_open_until_the_close() -> None:
    server, client = make_server(ReplayVerifyServer)
    participant = server.checkpoint_participant
    close = {"resources_session_id": "s", "episode_id": {"rollout_id": "r"}}
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        [session_id] = server.counters
        verified = await client.post("/ng-rollout/r/verify", json=VERIFY, headers=CLOSE_FOLLOWS)
        after_verify = participant.is_open(session_id)
        closed = await client.post("/ng-rollout/r/close_session", json=close)

    assert verified.status_code == 200 and closed.status_code == 200
    assert after_verify and server.closed == [session_id]
    assert not participant.is_open(session_id)


async def test_a_waited_for_verify_ends_the_session_whether_or_not_its_caller_closes_afterwards() -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        [session_id] = server.counters
        verified = await client.post("/ng-rollout/r/verify", json=VERIFY, headers=CLOSE_FOLLOWS)

    # A checkpoint waits for this verification, so no boundary before it can outlive it.
    assert verified.status_code == 200
    assert not server.checkpoint_participant.is_open(session_id)


async def test_a_request_in_flight_for_a_restart_an_earlier_stage_reported_does_not_hold_up_prepare(
    tmp_path: Path,
) -> None:
    """The environment server reported r as a restart, and r's tool call is still running here.

    The commit leaves r out, so prepare does not wait for the call, and r's session is not exported.
    The prepare request names r only for this checkpoint: once it resumes, the next checkpoint waits for r again.
    """
    server, client = make_server()
    server.gate = asyncio.Event()
    async with client:
        await client.post("/ng-rollout/q/seed_session", json=SEED)
        client.cookies.clear()
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        tool_call = asyncio.create_task(client.post("/ng-rollout/r/increment"))
        await until(lambda: server._checkpoint.inflight == 1)
        prepared = (
            await client.post("/ng-control/v1/checkpoint/prepare", json=control(restarts=["r"]), headers=AUTH)
        ).json()
        # Without a scope, so only the restart decides that r's session stays out.
        commit = (
            await client.post(
                "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
            )
        ).json()
        running_through_the_commit = not tool_call.done()
        server.gate.set()
        await tool_call
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        server.gate = asyncio.Event()
        second_call = asyncio.create_task(client.post("/ng-rollout/r/increment"))
        await until(lambda: server._checkpoint.inflight == 1)
        missed = (
            await client.post("/ng-control/v1/checkpoint/prepare", json=control("c2", timeout=0.2), headers=AUTH)
        ).json()
        server.gate.set()
        await second_call
        await client.post("/ng-control/v1/checkpoint/resume", json=control("c2"), headers=AUTH)

    assert prepared["phase"] == "prepared"
    assert commit["episode_ids"] == ["q"]
    assert running_through_the_commit
    assert missed["phase"] == "preparing" and missed["report"]["counts"]["inflight"] == 1
