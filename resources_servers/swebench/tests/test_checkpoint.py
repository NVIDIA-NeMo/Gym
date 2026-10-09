# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SWE-bench server's sandboxes through the real resources checkpoint participant.

The fake provider behaves like OpenSandbox on Kubernetes: a pause freezes the sandbox and leaves a snapshot,
the resume consumes that snapshot, and the snapshot listing exists but is never authoritative. A checkpoint
can therefore restore a session's sandbox only while it is still paused.
"""

import asyncio
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


class ConsumingSnapshotProvider:
    """Pause freezes and snapshots; resume thaws and consumes the snapshot; connect is by id."""

    name = "fake-opensandbox"
    snapshot_survives_resume = False

    def __init__(self) -> None:
        self.boxes: dict[str, dict[str, Any]] = {}
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
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
        self.boxes[sandbox_id] = {"state": "running", "commands": [], "image": spec.image}
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        box = self._box(handle.sandbox_id)
        if box["state"] != "running":
            raise RuntimeError(f"sandbox {handle.sandbox_id} is {box['state']}")
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
        return SandboxStatus.PAUSED if box["state"] == "paused" else SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        self.calls.append(("close", handle.sandbox_id))
        if handle.sandbox_id in self.boxes:
            self.boxes[handle.sandbox_id]["state"] = "stopped"

    async def aclose(self) -> None:
        pass

    async def pause(self, handle: SandboxHandle) -> None:
        self.calls.append(("pause", handle.sandbox_id))
        box = self._box(handle.sandbox_id)
        box["state"] = "paused"
        self._counter += 1
        self.snapshots[f"snap-{self._counter}"] = {"sandbox_id": handle.sandbox_id, "order": self._counter}

    async def resume(self, handle: SandboxHandle) -> None:
        self.calls.append(("resume", handle.sandbox_id))
        box = self._box(handle.sandbox_id)
        if box["state"] != "paused":
            raise RuntimeError(f"sandbox {handle.sandbox_id} is not paused")
        box["state"] = "running"
        for snapshot_id in [s for s, snap in self.snapshots.items() if snap["sandbox_id"] == handle.sandbox_id]:
            del self.snapshots[snapshot_id]

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        sandbox_id = str(descriptor["sandbox_id"])
        self._box(sandbox_id)
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def latest_snapshot_id(self, handle: SandboxHandle) -> str | None:
        mine = [(s["order"], sid) for sid, s in self.snapshots.items() if s["sandbox_id"] == handle.sandbox_id]
        return max(mine)[1] if mine else None


def ops(provider: ConsumingSnapshotProvider, op: str) -> list[str]:
    return [sandbox_id for name, sandbox_id in provider.calls if name == op]


def control(checkpoint_id: str = "c1", *, timeout: float = 5, **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + timeout, **extra}


@pytest.fixture
def provider(monkeypatch: MonkeyPatch) -> ConsumingSnapshotProvider:
    provider = ConsumingSnapshotProvider()
    monkeypatch.setattr("resources_servers.swebench.app.get_global_config_dict", lambda: {})
    monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_config", lambda *_: provider)
    monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr("resources_servers.swebench.app.patch_swebench_multilingual_sandbox", AsyncMock())
    monkeypatch.setattr(
        "resources_servers.swebench.app.run_instance", AsyncMock(return_value={"resolved": True, "completed": True})
    )
    return provider


def make_server(*, parallelism: int = 4) -> tuple[SwebenchResourcesServer, httpx.AsyncClient]:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = SwebenchResourcesServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="swebench",
        sandbox_provider="sandbox",
        sandbox_config={"ttl_s": 18000, "resources": {"cpu": 2}},
        sandbox_checkpoint={"parallelism": parallelism},
        apply_anti_cheating=False,
        clear_swebench_debug_logs=True,
    )
    server = SwebenchResourcesServer(config=config, server_client=server_client)
    app = server.setup_webserver()
    return server, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://r", timeout=30)


async def test_a_commit_pauses_the_agents_sandbox_and_the_resume_thaws_it_in_place(
    provider: ConsumingSnapshotProvider, tmp_path: Path
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
        paused = provider.boxes["sb-1"]["state"]
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        # Eager resume runs in the background once admission reopens.
        for _ in range(50):
            if provider.boxes["sb-1"]["state"] == "running":
                break
            await asyncio.sleep(0.01)
        thawed = provider.boxes["sb-1"]["state"]
        verified = await client.post("/verify", json=VERIFY)

    assert commit.status_code == 200, commit.text
    assert commit.json()["manifest"]["record_count"] == 1
    assert paused == "paused"
    [record_path] = (tmp_path / "gym").rglob("records-*.jsonl")
    [record] = [json.loads(line) for line in record_path.read_text().splitlines()]
    state = SandboxCheckpointState.model_validate(record["state"]["sandbox"])
    assert state.restore_point == "paused" and state.descriptor["sandbox_id"] == "sb-1"
    assert state.expires_at is not None and state.expires_at - state.paused_at <= 18000
    assert record["state"]["workdir"] == "/testbed"
    assert thawed == "running", "the resume phase thawed the sandbox before its next use"
    # Verify extracted the patch from the agent's sandbox, graded in a fresh one, and stopped the agent's.
    assert verified.status_code == 200, verified.text
    assert verified.json()["model_patch"] == "diff --git a/f b/f\n+fixed\n" and verified.json()["reward"] == 1
    assert len(ops(provider, "create")) == 2, "the only new sandbox is the verifier's"
    assert provider.boxes["sb-1"]["state"] == "stopped"
    assert "sb-1" not in server._checkpointer().session_ids


async def test_a_fresh_server_restores_the_paused_sandbox_and_the_replacement_reseeds_into_it(
    provider: ConsumingSnapshotProvider, tmp_path: Path
) -> None:
    _, client = make_server()
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)
        cookies = dict(client.cookies)
    assert provider.boxes["sb-1"]["state"] == "paused"

    # Every Gym process dies before anything resumes the sandbox. A fresh server restores the checkpoint.
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
        # The replacement attempt re-seeds the session: it gets the same sandbox, and no new one is created.
        reseeded = await fresh.post("/ng-rollout/r-a2/seed_session", json=SEED)
        verified = await fresh.post("/verify", json=VERIFY)

    assert reseeded.status_code == 200, reseeded.text
    assert reseeded.json()["sandbox_handle"] == "sb-1"
    assert ops(provider, "resume") == ["sb-1"] and provider.boxes["sb-1"]["commands"].count("pwd") == 1
    assert verified.status_code == 200, verified.text
    assert verified.json()["model_patch"].endswith("+fixed\n")
    assert len(ops(provider, "create")) == 2, "the only new sandbox is the verifier's"
    assert provider.boxes["sb-1"]["state"] == "stopped"
    assert fresh_server._checkpointer().session_ids == []


async def test_a_restore_after_the_sandbox_resumed_is_a_typed_failure(
    provider: ConsumingSnapshotProvider, tmp_path: Path
) -> None:
    _, client = make_server()
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        # The run continued: the eager resume thawed the sandbox and consumed its snapshot.
        for _ in range(50):
            if provider.boxes["sb-1"]["state"] == "running":
                break
            await asyncio.sleep(0.01)
        cookies = dict(client.cookies)
    assert provider.boxes["sb-1"]["state"] == "running" and provider.snapshots == {}

    _, fresh = make_server()
    async with fresh:
        fresh.cookies.update(cookies)
        # The hook's typed error is the restore's failure: the controller falls back to restarting the rollout.
        with pytest.raises(SandboxCheckpointError, match="'.*' .*resumed after the checkpoint") as error:
            await fresh.post(
                "/ng-control/v1/checkpoint/restore",
                json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=SCOPE),
                headers=AUTH,
            )

    assert "keeps no durable snapshot" in str(error.value)
    assert ops(provider, "create") == ["sb-1"], "no fork was attempted"
    assert provider.boxes["sb-1"]["state"] == "running", "the live sandbox is left alone"


async def test_retiring_the_attempt_stops_its_sandbox(provider: ConsumingSnapshotProvider) -> None:
    server, client = make_server()
    async with client:
        # Attempt 0 of rollout r; a retire names the attempt and everything before it.
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        retired = await client.post(
            "/ng-control/v1/checkpoint/retire", json=control("retire", episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )

    assert retired.status_code == 200, retired.text
    assert provider.boxes["sb-1"]["state"] == "stopped"
    assert server._checkpointer().session_ids == [] and server._workdirs == {}


async def test_sandbox_access_names_the_sessions_sandbox_and_checkout(provider: ConsumingSnapshotProvider) -> None:
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
