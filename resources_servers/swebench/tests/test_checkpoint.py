# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SWE-bench server's sandboxes through the real resources checkpoint participant.

The fake provider behaves like OpenSandbox with a durable snapshot store: a snapshot is an explicit copy of a
running sandbox that lives until deleted, and a sandbox can be created from it. A checkpoint snapshots the
agent's sandbox and leaves it running; a restore forks the snapshot under a new id.
"""

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from omegaconf import DictConfig
from pytest import MonkeyPatch

from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.checkpoint import SandboxCheckpointError, SandboxCheckpointState
from nemo_gym.server_utils import ServerClient
from resources_servers.swebench.app import SwebenchResourcesServer, SwebenchResourcesServerConfig


AUTH = {"authorization": "Bearer t"}
SCOPE = [{"rollout_id": "r"}, {"rollout_id": "r", "attempt": 1}]
INSTANCE = {
    "repo": "astropy/astropy",
    "instance_id": "astropy__astropy-12907",
    "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",  # pragma: allowlist secret
    "patch": "diff --git a/x b/x",
    "test_patch": "diff --git a/t b/t",
    "problem_statement": "Something is wrong.",
    "hints_text": "",
    "created_at": "2022-03-03T15:14:54Z",
    "version": "4.3",
    "FAIL_TO_PASS": "[]",
    "PASS_TO_PASS": "[]",
    "environment_setup_commit": "298ccb478e6bf092953bca67a3d29dc6c35f6752",  # pragma: allowlist secret
    "difficulty": "<15 min fix",
    "subset": "verified",
    "split": "test",
}
SEED = {**INSTANCE, "responses_create_params": {"input": []}}
VERIFY = {
    **INSTANCE,
    "responses_create_params": {"input": []},
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


class SnapshotProvider:
    """Explicit snapshots copy the sandbox's command history; create forks one; connect is by id."""

    name = "fake-opensandbox"

    def __init__(self) -> None:
        self.boxes: dict[str, dict[str, Any]] = {}
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self.fail_snapshot = False
        self._counter = 0

    def _box(self, sandbox_id: str) -> dict[str, Any]:
        box = self.boxes.get(sandbox_id)
        if box is None or box["state"] == "stopped":
            raise RuntimeError(f"sandbox {sandbox_id} is gone")
        return box

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self._counter += 1
        sandbox_id = f"sb-{self._counter}"
        self.calls.append(("create", sandbox_id))
        snapshot_id = spec.provider_options.get("snapshot_id")
        if (spec.image is None) == (snapshot_id is None):
            raise ValueError("exactly one of image or snapshot_id must be specified")
        commands = list(self.snapshots[snapshot_id]["commands"]) if snapshot_id else []
        self.boxes[sandbox_id] = {"state": "running", "commands": commands, "from_snapshot": snapshot_id}
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        box = self._box(handle.sandbox_id)
        box["commands"].append(command)
        if command == "pwd":
            return SandboxExecResult(stdout="/testbed\n", stderr=None, return_code=0)
        if "git --no-pager diff" in command:
            return SandboxExecResult(stdout="diff --git a/f b/f\n+fixed\n", stderr=None, return_code=0)
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        self._box(handle.sandbox_id)["commands"].append(f"upload {target_path}")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        pass

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        box = self.boxes.get(handle.sandbox_id)
        if box is None or box["state"] == "stopped":
            return SandboxStatus.STOPPED
        return SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        self.calls.append(("close", handle.sandbox_id))
        if handle.sandbox_id in self.boxes:
            self.boxes[handle.sandbox_id]["state"] = "stopped"

    async def aclose(self) -> None:
        pass

    async def snapshot(self, handle: SandboxHandle, *, name: str | None = None) -> str:
        self.calls.append(("snapshot", handle.sandbox_id))
        if self.fail_snapshot:
            raise RuntimeError("registry push failed")
        box = self._box(handle.sandbox_id)
        self._counter += 1
        snapshot_id = f"snap-{self._counter}"
        self.snapshots[snapshot_id] = {"sandbox_id": handle.sandbox_id, "commands": list(box["commands"])}
        return snapshot_id

    async def delete_snapshot(self, snapshot_id: str) -> None:
        self.calls.append(("delete_snapshot", snapshot_id))
        self.snapshots.pop(snapshot_id, None)

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        sandbox_id = str(descriptor["sandbox_id"])
        self._box(sandbox_id)
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)


def ops(provider: SnapshotProvider, op: str) -> list[str]:
    return [sandbox_id for name, sandbox_id in provider.calls if name == op]


def control(checkpoint_id: str = "c1", *, timeout: float = 5, **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + timeout, **extra}


@pytest.fixture
def provider(monkeypatch: MonkeyPatch) -> SnapshotProvider:
    provider = SnapshotProvider()
    monkeypatch.setattr("resources_servers.swebench.app.get_global_config_dict", lambda: {})
    monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_config", lambda *_: provider)
    monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr("resources_servers.swebench.app.patch_swebench_multilingual_sandbox", AsyncMock())
    monkeypatch.setattr(
        "resources_servers.swebench.app.run_instance", AsyncMock(return_value={"resolved": True, "completed": True})
    )
    return provider


def make_server(*, parallelism: int = 4, on_stop: str = "kill") -> tuple[SwebenchResourcesServer, httpx.AsyncClient]:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = SwebenchResourcesServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="swebench",
        sandbox_provider="sandbox",
        sandbox_config={"ttl_s": 18000, "resources": {"cpu": 2}},
        sandbox_checkpoint={"parallelism": parallelism, "on_stop": on_stop},
        apply_anti_cheating=False,
        clear_swebench_debug_logs=True,
    )
    server = SwebenchResourcesServer(config=config, server_client=server_client)
    app = server.setup_webserver()
    return server, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://r", timeout=30)


async def test_a_commit_snapshots_the_agents_sandbox_and_leaves_it_running(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    server, client = make_server()
    async with client:
        seeded = await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        assert seeded.status_code == 200, seeded.text
        assert seeded.json()["sandbox_handle"] == "sb-1"
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        at_commit = provider.boxes["sb-1"]["state"]
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        verified = await client.post("/verify", json=VERIFY)

    assert commit.status_code == 200, commit.text
    assert commit.json()["manifest"]["record_count"] == 1
    assert at_commit == "running" and ops(provider, "snapshot") == ["sb-1"]
    [record_path] = (tmp_path / "gym").rglob("records-*.jsonl")
    [record] = [json.loads(line) for line in record_path.read_text().splitlines()]
    state = SandboxCheckpointState.model_validate(record["state"]["sandbox"])
    assert state.descriptor["sandbox_id"] == "sb-1" and state.snapshot_id == "snap-2"
    assert record["state"]["workdir"] == "/testbed"
    # Verify extracted the patch from the agent's sandbox, graded in a fresh one, stopped the agent's, and deleted
    # the snapshots of it: the episode is over.
    assert verified.status_code == 200, verified.text
    assert verified.json()["model_patch"] == "diff --git a/f b/f\n+fixed\n" and verified.json()["reward"] == 1
    assert len(ops(provider, "create")) == 2, "the only new sandbox is the verifier's"
    assert provider.boxes["sb-1"]["state"] == "stopped" and provider.snapshots == {}
    assert "sb-1" not in server._checkpointer().session_ids


async def test_a_fresh_server_forks_the_snapshot_and_the_replacement_reseeds_into_the_fork(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    _, client = make_server()
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        # The episode continues in the live sandbox, which moves past the checkpoint before the crash.
        cookies = dict(client.cookies)
    provider.boxes["sb-1"]["commands"].append("echo drift > file")

    fresh_server, fresh = make_server()
    async with fresh:
        fresh.cookies.update(cookies)
        restore = await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=SCOPE),
            headers=AUTH,
        )
        assert restore.status_code == 200, restore.text
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        # The replacement attempt re-seeds the session: it gets the fork, and no other sandbox is created.
        reseeded = await fresh.post("/ng-rollout/r-a2/seed_session", json=SEED)
        access = await fresh.post("/sandbox_access")
        verified = await fresh.post("/verify", json=VERIFY)

    assert reseeded.status_code == 200, reseeded.text
    fork = reseeded.json()["sandbox_handle"]
    assert fork == "sb-3" and provider.boxes[fork]["from_snapshot"] == "snap-2"
    assert "echo drift > file" not in provider.boxes[fork]["commands"], "the fork holds the checkpoint, not the drift"
    assert access.json()["connection"]["descriptor"]["sandbox_id"] == fork and access.json()["workdir"] == "/testbed"
    assert provider.boxes["sb-1"]["state"] == "stopped", "the superseded sandbox is stopped"
    assert verified.status_code == 200, verified.text
    assert verified.json()["model_patch"].endswith("+fixed\n")
    assert len(ops(provider, "create")) == 3, "the original, the fork, and the verifier's"
    assert provider.boxes[fork]["state"] == "stopped"
    assert fresh_server._checkpointer().session_ids == []


async def test_a_snapshot_failure_fails_the_commit_without_publishing(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    _, client = make_server()
    provider.fail_snapshot = True
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        with pytest.raises(SandboxCheckpointError, match="could not be snapshotted"):
            await client.post(
                "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
            )
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        provider.fail_snapshot = False
        verified = await client.post("/verify", json=VERIFY)

    assert not list(tmp_path.rglob("manifest*"))
    assert verified.status_code == 200, "the episode continues in its sandbox"


async def test_a_commit_the_controller_stops_after_kills_the_sandbox(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    server, client = make_server()
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path), stop=True), headers=AUTH
        )

    assert commit.status_code == 200, commit.text
    assert provider.boxes["sb-1"]["state"] == "stopped" and server._checkpointer().session_ids == []
    assert len(provider.snapshots) == 1, "the checkpoint restores the session from its snapshot"


async def test_on_stop_none_leaves_the_sandbox_running(provider: SnapshotProvider, tmp_path: Path) -> None:
    _, client = make_server(on_stop="none")
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path), stop=True), headers=AUTH
        )

    assert commit.status_code == 200, commit.text
    assert provider.boxes["sb-1"]["state"] == "running"


async def test_retiring_the_attempt_stops_its_sandbox_and_keeps_its_snapshots(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    server, client = make_server()
    async with client:
        # Attempt 0 of rollout r; a retire names the attempt and everything before it.
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        retired = await client.post(
            "/ng-control/v1/checkpoint/retire", json=control("retire", episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )

    assert retired.status_code == 200, retired.text
    assert provider.boxes["sb-1"]["state"] == "stopped"
    assert server._checkpointer().session_ids == [] and server._workdirs == {}
    # The checkpoint still names the snapshot; the sweep reclaims it once no retained checkpoint does.
    assert len(provider.snapshots) == 1 and ops(provider, "delete_snapshot") == []


async def test_sandbox_access_names_the_sessions_sandbox_and_checkout(provider: SnapshotProvider) -> None:
    _, client = make_server()
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        access = await client.post("/sandbox_access")

    assert access.status_code == 200, access.text
    assert access.json() == {
        # The descriptor is the sandbox's own serialization: the provider's id plus the spec's workdir.
        "connection": {
            "kind": "direct",
            "provider_config_ref": "sandbox",
            "descriptor": {"sandbox_id": "sb-1", "workdir": None},
        },
        "workdir": "/testbed",
    }
