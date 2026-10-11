# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partial-rollout checkpoints of the litmus agent's code-execution sessions.

Each session's cell history and sandbox are exported at commit and rebuilt in a fresh server after a crash. The
provider fakes a snapshot as a copy of the sandbox's files, so a restore forks a sandbox that still holds the
replay driver, and the restored cell history replays into it.
"""

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from omegaconf import DictConfig

from nemo_gym.sandbox import SandboxHandle, SandboxStatus, list_providers, register_provider
from nemo_gym.server_utils import ServerClient
from resources_servers.litmus_agent.app import LitmusAgentConfig, LitmusAgentResourcesServer
from resources_servers.litmus_agent.tests.test_app import _LocalFakeProvider


class _SnapshottingFakeProvider(_LocalFakeProvider):
    """The local fake, with explicit snapshots of the sandbox's files and connect by id, as OpenSandbox has."""

    name = "litmus_fake_snapshots"
    state: dict[str, str] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    files_by_sandbox: dict[str, dict[str, str]] = {}
    counter = 0

    async def create(self, spec) -> SandboxHandle:
        handle = await super().create(spec)
        cls = _SnapshottingFakeProvider
        cls.counter += 1
        sandbox_id = f"{self.name}-{cls.counter}"
        handle.sandbox_id = sandbox_id
        snapshot_id = spec.provider_options.get("snapshot_id")
        self._files[sandbox_id] = dict(cls.snapshots[snapshot_id]["files"]) if snapshot_id else {}
        cls.files_by_sandbox[sandbox_id] = self._files[sandbox_id]
        cls.state[sandbox_id] = "running"
        return handle

    async def exec(self, handle, command, **kwargs):
        if _SnapshottingFakeProvider.state.get(handle.sandbox_id) != "running":
            raise RuntimeError(f"sandbox {handle.sandbox_id} is not running")
        self._files.setdefault(handle.sandbox_id, _SnapshottingFakeProvider.files_by_sandbox[handle.sandbox_id])
        return await super().exec(handle, command, **kwargs)

    async def status(self, handle) -> SandboxStatus:
        state = _SnapshottingFakeProvider.state.get(handle.sandbox_id, "stopped")
        return SandboxStatus.RUNNING if state == "running" else SandboxStatus.STOPPED

    async def close(self, handle) -> None:
        await super().close(handle)
        _SnapshottingFakeProvider.state[handle.sandbox_id] = "stopped"

    async def snapshot(self, handle, *, name=None) -> str:
        cls = _SnapshottingFakeProvider
        if cls.state.get(handle.sandbox_id) != "running":
            raise RuntimeError(f"sandbox {handle.sandbox_id} is not running")
        cls.counter += 1
        snapshot_id = f"snap-{cls.counter}"
        cls.snapshots[snapshot_id] = {
            "sandbox_id": handle.sandbox_id,
            "files": dict(cls.files_by_sandbox[handle.sandbox_id]),
        }
        return snapshot_id

    async def delete_snapshot(self, snapshot_id: str) -> None:
        _SnapshottingFakeProvider.snapshots.pop(snapshot_id, None)

    async def serialize_handle(self, handle, *, scope=None) -> dict:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor) -> SandboxHandle:
        sandbox_id = str(descriptor["sandbox_id"])
        if _SnapshottingFakeProvider.state.get(sandbox_id, "stopped") == "stopped":
            raise RuntimeError(f"sandbox {sandbox_id} is gone")
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)


if _SnapshottingFakeProvider.name not in list_providers():
    register_provider(_SnapshottingFakeProvider.name, _SnapshottingFakeProvider)


AUTH = {"authorization": "Bearer t"}
SEED = {"responses_create_params": {"input": "hi"}}
SCOPE = [{"rollout_id": "r"}, {"rollout_id": "r", "attempt": 1}]


@pytest.fixture(autouse=True)
def reset_backend() -> None:
    cls = _SnapshottingFakeProvider
    cls.state.clear()
    cls.snapshots.clear()
    cls.files_by_sandbox.clear()
    cls.counter = 0


def make_server() -> tuple[LitmusAgentResourcesServer, httpx.AsyncClient]:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = LitmusAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="litmus_agent",
        sandbox_provider={_SnapshottingFakeProvider.name: {}},
        sandbox_spec={"image": "fake:latest"},
    )
    server = LitmusAgentResourcesServer(config=config, server_client=server_client)
    app = server.setup_webserver()
    return server, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://r")


def control(checkpoint_id: str = "c1", *, timeout: float = 5, **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + timeout, **extra}


async def test_cell_history_and_sandbox_are_restored_in_a_fresh_server(tmp_path: Path) -> None:
    server, client = make_server()
    tool = f"/{server.config.code_exec_tool_name}"
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        first = await client.post(tool, json={"code": "x = 21"})
        status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()
        prepare = await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        at_commit = dict(_SnapshottingFakeProvider.state)
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        # The live rollout continues past the checkpoint before the crash.
        drift = await client.post(tool, json={"code": "x = 1000"})
        cookies = dict(client.cookies)

    assert first.status_code == 200 and drift.status_code == 200
    assert status["report"]["counts"] == {"inflight": 0, "sessions": 1}, status
    assert prepare.status_code == 200 and prepare.json()["report"]["ready"], (status.get("mode"), status, prepare.text)
    assert commit.status_code == 200, (prepare.text, commit.text)
    assert commit.json()["manifest"]["record_count"] == 1
    [sandbox_id] = at_commit
    assert at_commit[sandbox_id] == "running" and len(_SnapshottingFakeProvider.snapshots) == 1

    restored, fresh = make_server()
    async with fresh:
        fresh.cookies.update(cookies)
        restore = await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=SCOPE),
            headers=AUTH,
        )
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        after = await fresh.post(tool, json={"code": "print(x * 2)"})

    assert restore.status_code == 200, restore.text
    # The restored history is the checkpoint's (x = 21), not the drifted one, and it replays in a fork of the
    # snapshot that still holds the replay driver.
    assert after.text.strip() == "42"
    [session] = restored._sessions.values()
    assert session.cells == ["x = 21", "print(x * 2)"]
    assert session.sandbox.handle.sandbox_id != sandbox_id
    assert _SnapshottingFakeProvider.state[sandbox_id] == "stopped", "the superseded sandbox is stopped"
    assert len(_SnapshottingFakeProvider.snapshots) == 1, "the checkpoint still names the snapshot"


async def test_a_session_without_a_sandbox_yet_checkpoints_as_empty(tmp_path: Path) -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        prepare = await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
    assert commit.status_code == 200, (prepare.text, commit.text)
    assert commit.json()["manifest"]["record_count"] == 1
    assert server._sessions == {} and _SnapshottingFakeProvider.state == {}


async def test_retire_stops_the_sessions_sandbox() -> None:
    server, client = make_server()
    tool = f"/{server.config.code_exec_tool_name}"
    async with client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post(tool, json={"code": "x = 1"})
        status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()
        retired = await client.post(
            "/ng-control/v1/checkpoint/retire",
            json=control("retire", episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        late = await client.post(tool, json={"code": "print(x)"})

    assert retired.status_code == 200, retired.text
    assert status["report"]["counts"] == {"inflight": 0, "sessions": 1}, status
    assert server._sessions == {}
    assert list(_SnapshottingFakeProvider.state.values()) == ["stopped"]
    assert late.status_code == 409 and late.json()["error"]["code"] == "stale_attempt"
