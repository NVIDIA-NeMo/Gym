# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A resources server whose sessions own sandboxes, checkpointed through the real resources participant."""

import time
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI, Request
from omegaconf import DictConfig
from pydantic import JsonValue, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.sandbox.checkpoint import SandboxCheckpointError, SandboxSessionCheckpointer
from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from tests.unit_tests.test_sandbox_checkpoint import FakeSnapshotProvider


AUTH = {"authorization": "Bearer t"}
SEED = {"responses_create_params": {"input": "hi"}}
# Every restore below continues the episode its source checkpoint exported.
SCOPE = [{"rollout_id": "r"}, {"rollout_id": "r", "attempt": 1}]


class NotesServer(SimpleResourcesServer):
    """Each session appends lines to a file inside its own sandbox."""

    ray_enabled = False
    checkpoint_mode: ClassVar[str] = "exported"
    provider: Any = None
    _sandboxes: SandboxSessionCheckpointer = PrivateAttr(default=None)

    def setup_webserver(self) -> FastAPI:
        self._sandboxes = SandboxSessionCheckpointer(self.provider, parallelism=4)
        app = super().setup_webserver()
        app.post("/append")(self.append)
        return app

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        await self._sandboxes.create(request.session[SESSION_ID_KEY], SandboxSpec(image="notes:1"))
        return BaseSeedSessionResponse()

    async def append(self, request: Request, body: dict) -> dict:
        sandbox = self._sandboxes.get(request.session[SESSION_ID_KEY])
        await sandbox.exec(f"echo {body['line']} >> notes")
        return {"sandbox_id": sandbox.handle.sandbox_id}

    async def verify(self, request: Request, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # The episode is over: its sandbox and the snapshots of it go.
        await self._sandboxes.stop(request.session[SESSION_ID_KEY], forget_snapshots=True)
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)

    async def export_session_states(self, session_ids: list[str]) -> dict[str, JsonValue]:
        return await self._sandboxes.export(session_ids)

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        await self._sandboxes.restore(states)

    async def retire_session_state(self, session_id: str) -> None:
        await self._sandboxes.stop(session_id)

    async def park_session_states(self, session_ids: list[str]) -> None:
        await self._sandboxes.park(session_ids)

    async def current_sandbox_access(self, session_id: str) -> SandboxAccess | None:
        if session_id not in self._sandboxes:
            return None
        return await self._sandboxes.access(session_id, provider_config_ref="sandbox", workdir="/work")


def make_server(provider: FakeSnapshotProvider) -> tuple[NotesServer, httpx.AsyncClient]:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="notes")
    server = NotesServer(config=config, server_client=server_client, provider=provider)
    app = server.setup_webserver()
    return server, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://r")


def control(checkpoint_id: str = "c1", *, timeout: float = 5, **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + timeout, **extra}


VERIFY = {
    "responses_create_params": {"input": "hi"},
    "response": {
        "output": [],
        "id": "",
        "created_at": 0,
        "model": "",
        "object": "response",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    },
}


async def test_the_snapshot_is_committed_and_a_fresh_server_forks_it(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    _, client = make_server(provider)
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        await client.post("/append", json={"line": "one"})
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        # The rollout continues after the checkpoint in the same, never paused, sandbox.
        after = await client.post("/append", json={"line": "two"})
        cookies = dict(client.cookies)

    assert commit.json()["manifest"]["record_count"] == 1
    assert ("snapshot", "sb-1") in provider.calls and provider.boxes["sb-1"]["state"] == "running"
    assert after.json()["sandbox_id"] == "sb-1"
    assert provider.files("sb-1") == ["echo one >> notes", "echo two >> notes"]
    [snapshot] = provider.snapshots.values()
    assert snapshot["fs"] == ["echo one >> notes"]

    # Every Gym process dies. A fresh server restores the checkpoint and the replacement attempt replays "two".
    _, fresh = make_server(provider)
    async with fresh:
        fresh.cookies.update(cookies)
        restore = await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=SCOPE),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        replayed = await fresh.post("/append", json={"line": "two"})

    assert restore.status_code == 200, restore.text
    forked = replayed.json()["sandbox_id"]
    assert forked != "sb-1"
    # The fork holds the checkpoint's filesystem, not the drifted one, so the replayed call lands once.
    assert provider.files(forked) == ["echo one >> notes", "echo two >> notes"]
    assert provider.boxes["sb-1"]["state"] == "stopped"


async def test_a_commit_that_no_longer_continues_a_restored_session_stops_its_sandbox(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    _, client = make_server(provider)
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/append", json={"line": "one"})
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path / "first")), headers=AUTH
        )

    restored, fresh = make_server(provider)
    async with fresh:
        await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path / "first"), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        [forked] = restored._sandboxes.session_ids
        fork_id = restored._sandboxes.get(forked).handle.sandbox_id
        assert fork_id != "sb-1" and provider.boxes["sb-1"]["state"] == "stopped"
        # The controller continues nothing from the next checkpoint: the restored session is never used.
        await fresh.post("/ng-control/v1/checkpoint/prepare", json=control("c2"), headers=AUTH)
        committed = await fresh.post(
            "/ng-control/v1/checkpoint/commit",
            json=control("c2", checkpoint_dir=str(tmp_path / "second"), episode_ids=[]),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("c2"), headers=AUTH)

    assert committed.json()["episode_ids"] == []
    assert restored._sandboxes.session_ids == []
    assert provider.boxes[fork_id]["state"] == "stopped"
    assert len(provider.snapshots) == 1, "a retire keeps the snapshot: the first checkpoint still names it"


async def test_retire_stops_the_sandbox_and_refuses_the_attempts_later_calls() -> None:
    provider = FakeSnapshotProvider()
    server, client = make_server(provider)
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/append", json={"line": "one"})
        retired = await client.post(
            "/ng-control/v1/checkpoint/retire",
            json=control("retire", episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        late = await client.post("/append", json={"line": "late"})

    assert retired.status_code == 200, retired.text
    assert provider.boxes["sb-1"]["state"] == "stopped" and server._sandboxes.session_ids == []
    assert late.status_code == 409 and late.json()["error"]["code"] == "stale_attempt"


async def test_a_snapshot_failure_fails_the_commit_and_the_episode_continues_after_resume(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    _, client = make_server(provider)
    provider.fail[("snapshot", "sb-1")] = RuntimeError("registry push failed")
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/append", json={"line": "one"})
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        # A hook error surfaces as a server error on the commit, as the stack's own hook tests expect; the ASGI
        # transport re-raises it here.
        with pytest.raises(SandboxCheckpointError, match="could not be snapshotted"):
            await client.post(
                "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
            )
        status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        after = await client.post("/append", json={"line": "two"})

    assert status["phase"] == "prepared", "the controller resumes and checkpoints again later"
    assert not list(tmp_path.rglob("manifest*")), "a failed export publishes nothing"
    assert after.status_code == 200 and provider.files("sb-1") == ["echo one >> notes", "echo two >> notes"]
    assert provider.boxes["sb-1"]["state"] == "running"


async def test_a_commit_the_controller_stops_after_kills_the_exported_sandboxes(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server, client = make_server(provider)
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/append", json={"line": "one"})
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path), stop=True), headers=AUTH
        )

    assert commit.status_code == 200, commit.text
    assert commit.json()["manifest"]["record_count"] == 1
    # The checkpoint is durable, the stopped Gym holds no compute, and the snapshot restores the session.
    assert provider.boxes["sb-1"]["state"] == "stopped" and server._sandboxes.session_ids == []
    [snapshot] = provider.snapshots.values()
    assert snapshot["fs"] == ["echo one >> notes"]
    _, fresh = make_server(provider)
    async with fresh:
        restore = await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=SCOPE),
            headers=AUTH,
        )
    assert restore.status_code == 200, restore.text


async def test_a_park_failure_does_not_fail_the_stop_commit(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    _, client = make_server(provider)
    provider.fail[("close", "sb-1")] = RuntimeError("backend busy")
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path), stop=True), headers=AUTH
        )

    assert commit.status_code == 200, commit.text
    assert commit.json()["phase"] == "committed"
    assert provider.boxes["sb-1"]["state"] == "running", "runs out its TTL; the checkpoint is durable regardless"


async def test_verify_ends_the_episode_and_deletes_its_snapshots(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    _, client = make_server(provider)
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        assert len(provider.snapshots) == 1
        verified = await client.post("/verify", json=VERIFY)

    assert verified.status_code == 200, verified.text
    assert provider.boxes["sb-1"]["state"] == "stopped" and provider.snapshots == {}


async def test_sandbox_access_follows_the_sandbox_across_a_restore(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    _, client = make_server(provider)
    async with client:
        unknown = await client.post("/sandbox_access")
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        before = (await client.post("/sandbox_access")).json()
        await client.post("/append", json={"line": "one"})
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        await client.post("/append", json={"line": "two"})  # the live sandbox moves on
        cookies = dict(client.cookies)

    _, fresh = make_server(provider)
    async with fresh:
        fresh.cookies.update(cookies)
        await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=SCOPE),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        after = (await fresh.post("/sandbox_access")).json()

    assert unknown.status_code == 404
    assert before["connection"]["descriptor"]["sandbox_id"] == "sb-1" and before["workdir"] == "/work"
    # The restore forked the sandbox, so a borrower that asks again is pointed at the fork.
    assert after["connection"]["descriptor"]["sandbox_id"] != "sb-1"
    assert provider.boxes[after["connection"]["descriptor"]["sandbox_id"]]["state"] == "running"
