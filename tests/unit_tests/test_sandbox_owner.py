# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``SandboxOwnerResourcesServer`` through the real resources checkpoint participant.

A small owner seeds a session's sandbox with a setup that returns per-session data, and verify reads it back.
``SnapshotProvider`` behaves like OpenSandbox with a durable snapshot store; ``PlainProvider`` is a provider that
cannot snapshot, as docker, apptainer and enroot are.
"""

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Optional
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import Request
from omegaconf import DictConfig

from nemo_gym.base_resources_server import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.owner import SandboxOwnerConfig, SandboxOwnerResourcesServer, SessionData
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient


AUTH = {"authorization": "Bearer t"}
SCOPE = [{"rollout_id": "r"}, {"rollout_id": "r", "attempt": 1}]
VERIFY = {
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

# Each server's app, built once per test: a test may open several clients, one per session.
_APPS: dict[int, Any] = {}


class PlainProvider:
    """A provider without snapshots; connect is by id."""

    name = "fake-plain"

    def __init__(self) -> None:
        self.boxes: dict[str, dict[str, Any]] = {}
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
        self.boxes[sandbox_id] = {"state": "running", "commands": [], "from_snapshot": None}
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        self._box(handle.sandbox_id)["commands"].append(command)
        if command == "pwd":
            return SandboxExecResult(stdout="/repo\n", stderr=None, return_code=0)
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        pass

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        pass

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        box = self.boxes.get(handle.sandbox_id)
        return SandboxStatus.STOPPED if box is None or box["state"] == "stopped" else SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        self.calls.append(("close", handle.sandbox_id))
        if handle.sandbox_id in self.boxes:
            self.boxes[handle.sandbox_id]["state"] = "stopped"

    async def aclose(self) -> None:
        pass

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        sandbox_id = str(descriptor["sandbox_id"])
        self._box(sandbox_id)
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)


class SnapshotProvider(PlainProvider):
    """Explicit snapshots copy the sandbox's command history; create forks one."""

    name = "fake-opensandbox"

    def __init__(self) -> None:
        super().__init__()
        self.snapshots: dict[str, dict[str, Any]] = {}

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        handle = await super().create(spec)
        snapshot_id = spec.provider_options.get("snapshot_id")
        if snapshot_id is not None:
            box = self.boxes[handle.sandbox_id]
            box["commands"] = list(self.snapshots[snapshot_id]["commands"])
            box["from_snapshot"] = snapshot_id
        return handle

    async def snapshot(self, handle: SandboxHandle, *, name: str | None = None) -> str:
        self.calls.append(("snapshot", handle.sandbox_id))
        box = self._box(handle.sandbox_id)
        self._counter += 1
        snapshot_id = f"snap-{self._counter}"
        self.snapshots[snapshot_id] = {"sandbox_id": handle.sandbox_id, "commands": list(box["commands"])}
        return snapshot_id

    async def wait_snapshot_ready(self, snapshot_id: str) -> None:
        pass

    async def delete_snapshot(self, snapshot_id: str) -> None:
        self.calls.append(("delete_snapshot", snapshot_id))
        self.snapshots.pop(snapshot_id, None)


class OwnerConfig(SandboxOwnerConfig):
    lends: bool = True


class OwnerVerifyRequest(BaseVerifyRequest):
    pass


class OwnerSeedResponse(BaseSeedSessionResponse):
    sandbox_handle: str


class Owner(SandboxOwnerResourcesServer):
    """Seeds a session with a setup that returns ``{"base_commit": <n>}``; verify reports the data it finds."""

    config: OwnerConfig
    setups: ClassVar[int] = 0
    setup_error: ClassVar[Optional[BaseException]] = None
    setup_result: ClassVar[Any] = "default"

    def lends_sandboxes(self) -> bool:
        return self.config.lends

    async def _setup(self, sandbox: AsyncSandbox) -> Optional[SessionData]:
        type(self).setups += 1
        await sandbox.exec("git rev-parse HEAD")
        if self.setup_error is not None:
            raise self.setup_error
        if self.setup_result != "default":
            return self.setup_result
        return {"base_commit": f"commit-{type(self).setups}"}

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> OwnerSeedResponse:
        session_id = request.session[SESSION_ID_KEY]
        sandbox = self.session_sandbox(session_id)
        if sandbox is None:
            sandbox = await self.start_session_sandbox(session_id, SandboxSpec(image="img:1"), setup=self._setup)
        return OwnerSeedResponse(sandbox_handle=sandbox.handle.sandbox_id)

    async def verify(self, request: Request, body: OwnerVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        self.require_session_sandbox(session_id)
        data = self.session_data(session_id)
        await self.end_session_sandbox(session_id)
        return BaseVerifyResponse(**body.model_dump(), reward=float(bool(data.get("base_commit"))))


@pytest.fixture(autouse=True)
def fresh_owner_class() -> None:
    Owner.setups = 0
    Owner.setup_error = None
    Owner.setup_result = "default"
    _APPS.clear()


def use_provider(monkeypatch: pytest.MonkeyPatch, provider: PlainProvider) -> PlainProvider:
    monkeypatch.setattr("nemo_gym.sandbox.owner.get_global_config_dict", lambda: {})
    monkeypatch.setattr("nemo_gym.sandbox.owner.resolve_provider_config", lambda *_: provider)
    return provider


@pytest.fixture
def snapshots(monkeypatch: pytest.MonkeyPatch) -> SnapshotProvider:
    return use_provider(monkeypatch, SnapshotProvider())


@pytest.fixture
def plain(monkeypatch: pytest.MonkeyPatch) -> PlainProvider:
    return use_provider(monkeypatch, PlainProvider())


def make_owner(
    state_dir: Optional[Path], *, checkpointing: bool = True, num_workers: Optional[int] = None, lends: bool = True
) -> Owner:
    block: dict[str, Any] = {"enabled": checkpointing, "control_auth_token": "t"}
    if state_dir is not None:
        block["state_dir"] = str(state_dir)
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": block})
    config = OwnerConfig(
        host="",
        port=0,
        entrypoint="",
        name="owner",
        sandbox_provider="sandbox",
        sandbox_checkpoint={"parallelism": 4},
        num_workers=num_workers,
        lends=lends,
    )
    return Owner(config=config, server_client=server_client)


def client_for(server: Owner) -> httpx.AsyncClient:
    """A client of the server's app; one per session, since the session is the client's cookie."""
    if id(server) not in _APPS:
        _APPS[id(server)] = server.setup_webserver()
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=_APPS[id(server)]), base_url="http://r", timeout=30)


def control(checkpoint_id: str = "c1", **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5, **extra}


async def commit(client: httpx.AsyncClient, checkpoint_dir: Path, checkpoint_id: str = "c1") -> httpx.Response:
    prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(checkpoint_id), headers=AUTH)
    assert prepared.status_code == 200, prepared.text
    committed = await client.post(
        "/ng-control/v1/checkpoint/commit",
        json=control(checkpoint_id, checkpoint_dir=str(checkpoint_dir)),
        headers=AUTH,
    )
    await client.post("/ng-control/v1/checkpoint/resume", json=control(checkpoint_id), headers=AUTH)
    return committed


def resources_records(checkpoint_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for path in (checkpoint_dir / "gym").rglob("records-*.jsonl")
        for line in path.read_text().splitlines()
    ]


async def restore(client: httpx.AsyncClient, checkpoint_dir: Path) -> httpx.Response:
    restored = await client.post(
        "/ng-control/v1/checkpoint/restore",
        json=control("r1", checkpoint_dir=str(checkpoint_dir), episode_ids=SCOPE),
        headers=AUTH,
    )
    await client.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
    return restored


def ops(provider: PlainProvider, op: str) -> list[str]:
    return [sandbox_id for name, sandbox_id in provider.calls if name == op]


# -- exported state and the per-session data -----------------------------------------------------------------


async def test_the_sandbox_its_workdir_and_the_setup_data_survive_a_restore(
    snapshots: SnapshotProvider, tmp_path: Path
) -> None:
    server = make_owner(tmp_path / "state")
    async with client_for(server) as client:
        seeded = await client.post("/ng-rollout/r-a1/seed_session", json={})
        cookies = dict(client.cookies)
        committed = await commit(client, tmp_path / "ckpt")

    assert seeded.status_code == 200 and committed.status_code == 200, (seeded.text, committed.text)
    [record] = resources_records(tmp_path / "ckpt")
    assert record["state"]["workdir"] == "/repo" and record["state"]["data"] == {"base_commit": "commit-1"}

    fresh = make_owner(tmp_path / "state")
    async with client_for(fresh) as client:
        client.cookies.update(cookies)
        restored = await restore(client, tmp_path / "ckpt")
        reseeded = await client.post("/ng-rollout/r-a2/seed_session", json={})
        access = await client.post("/ng-rollout/r-a2/sandbox_access")
        session_id = next(iter(fresh._sandboxes.session_ids))
        data_before_verify = fresh.session_data(session_id)
        verified = await client.post("/ng-rollout/r-a2/verify", json=VERIFY)

    assert restored.status_code == 200, restored.text
    fork = reseeded.json()["sandbox_handle"]
    assert snapshots.boxes[fork]["from_snapshot"] == "snap-2"
    assert Owner.setups == 1, "the re-seed reused the restored sandbox and did not run the setup again"
    assert access.json()["workdir"] == "/repo"
    assert data_before_verify == {"base_commit": "commit-1"}
    assert verified.status_code == 200 and verified.json()["reward"] == 1.0, verified.text
    assert fresh.session_data(session_id) == {}, "end_session_sandbox drops the data"


async def test_a_setup_that_returns_nothing_writes_the_state_without_data_and_restores_it_as_empty(
    snapshots: SnapshotProvider, tmp_path: Path
) -> None:
    Owner.setup_result = None
    server = make_owner(tmp_path / "state")
    async with client_for(server) as client:
        await client.post("/ng-rollout/r-a1/seed_session", json={})
        cookies = dict(client.cookies)
        await commit(client, tmp_path / "ckpt")

    [record] = resources_records(tmp_path / "ckpt")
    assert set(record["state"]) == {"sandbox", "workdir"}, "the SWE-bench shape: no data key"

    fresh = make_owner(tmp_path / "state")
    async with client_for(fresh) as client:
        client.cookies.update(cookies)
        assert (await restore(client, tmp_path / "ckpt")).status_code == 200
    [session_id] = fresh._sandboxes.session_ids
    assert fresh.session_data(session_id) == {}


async def test_a_restore_with_one_malformed_state_installs_nothing(
    snapshots: SnapshotProvider, tmp_path: Path
) -> None:
    server = make_owner(tmp_path / "state")
    sandbox_state = {
        "provider_name": "fake-opensandbox",
        "descriptor": {"sandbox_id": "sb-9"},
        "snapshot_id": "snap-9",
        "spec": {"image": None},
        "snapshot_at": 0.0,
    }
    snapshots.snapshots["snap-9"] = {"sandbox_id": "sb-9", "commands": []}

    with pytest.raises(ValueError, match="invalid data"):
        await server.restore_session_states(
            {
                "good": {"sandbox": sandbox_state, "workdir": "/repo", "data": {"base_commit": "abc"}},
                "bad": {"sandbox": sandbox_state, "workdir": "/repo", "data": ["not", "an", "object"]},
            }
        )

    assert ops(snapshots, "create") == [], "nothing was forked"
    assert server.session_sandbox("good") is None and server.session_data("good") == {}


# -- the sandbox's lifetime ----------------------------------------------------------------------------------


async def test_a_failed_setup_stops_the_sandbox_and_tracks_nothing(
    snapshots: SnapshotProvider, tmp_path: Path
) -> None:
    Owner.setup_error = RuntimeError("anti-cheat failed")
    server = make_owner(tmp_path / "state")
    async with client_for(server) as client:
        with pytest.raises(RuntimeError, match="anti-cheat failed"):
            await client.post("/ng-rollout/r-a1/seed_session", json={})
        Owner.setup_error = None
        committed = await commit(client, tmp_path / "ckpt")

    assert snapshots.boxes["sb-1"]["state"] == "stopped"
    assert server._sandboxes.session_ids == [] and server._workdirs == {} and server._session_data == {}
    assert committed.status_code == 200 and ops(snapshots, "snapshot") == [], "nothing to export"


async def test_a_setup_that_returns_something_other_than_a_json_object_fails_the_seed(
    snapshots: SnapshotProvider, tmp_path: Path
) -> None:
    Owner.setup_result = {"started": object()}
    server = make_owner(tmp_path / "state")
    async with client_for(server) as client:
        with pytest.raises(ValueError):
            await client.post("/ng-rollout/r-a1/seed_session", json={})

    assert snapshots.boxes["sb-1"]["state"] == "stopped" and server._sandboxes.session_ids == []


async def test_retire_and_end_keep_the_snapshots_and_release_deletes_them(
    snapshots: SnapshotProvider, tmp_path: Path
) -> None:
    server = make_owner(tmp_path / "state")
    async with client_for(server) as client, client_for(server) as other:
        await client.post("/ng-rollout/r/seed_session", json={})
        await other.post("/ng-rollout/s/seed_session", json={})
        await commit(client, tmp_path / "ckpt")
        retired = await client.post(
            "/ng-control/v1/checkpoint/retire", json=control("retire", episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )
        verified = await other.post("/ng-rollout/s/verify", json=VERIFY)
        kept = dict(snapshots.snapshots)
        released = await client.post(
            "/ng-control/v1/checkpoint/release", json=control("c9", kept_checkpoint_ids=[]), headers=AUTH
        )
        await server._sandboxes.settle()

    assert retired.status_code == 200 and verified.status_code == 200, (retired.text, verified.text)
    assert {snapshots.boxes[box]["state"] for box in ("sb-1", "sb-2")} == {"stopped"}
    assert len(kept) == 2, "neither the retire nor the end of the episode deleted a snapshot"
    assert released.status_code == 200 and snapshots.snapshots == {}
    assert server._session_data == {} and server._workdirs == {}


async def test_a_verify_for_a_session_without_a_sandbox_answers_409(
    snapshots: SnapshotProvider, tmp_path: Path
) -> None:
    server = make_owner(tmp_path / "state")
    async with client_for(server) as client:
        await client.post("/ng-rollout/r-a1/seed_session", json={})
        first = await client.post("/ng-rollout/r-a1/verify", json=VERIFY)
        again = await client.post("/ng-rollout/r-a1/verify", json=VERIFY)

    assert first.status_code == 200
    assert again.status_code == 409 and "has no sandbox" in again.json()["detail"]


async def test_an_empty_close_ends_the_cookie_session_and_succeeds_when_already_closed(
    snapshots: SnapshotProvider, tmp_path: Path
) -> None:
    server = make_owner(tmp_path / "state")
    async with client_for(server) as client:
        await client.post("/ng-rollout/r-a1/seed_session", json={})
        closed = await client.post("/ng-rollout/r-a1/close_session", json={})
        again = await client.post("/ng-rollout/r-a1/close_session", json={})
        access = await client.post("/ng-rollout/r-a1/sandbox_access")
        committed = await commit(client, tmp_path / "ckpt")

    assert closed.status_code == 200 and closed.json() == {"closed": True}, closed.text
    assert again.status_code == 200, again.text
    assert snapshots.boxes["sb-1"]["state"] == "stopped" and server._sandboxes.session_ids == []
    assert access.status_code == 404, "the agent finds no sandbox to continue in"
    assert committed.status_code == 200 and ops(snapshots, "snapshot") == [], "a closed session is not exported"


async def test_a_typed_close_gets_the_base_default(snapshots: SnapshotProvider) -> None:
    server = make_owner(None, checkpointing=False)
    async with client_for(server) as client:
        closed = await client.post(
            "/close_session", json={"resources_session_id": "x", "episode_id": {"rollout_id": "r"}}
        )
        malformed = await client.post("/close_session", json={"resources_session_id": "x"})

    assert closed.status_code == 200 and closed.json() == {"resources_session_id": "x"}, closed.text
    assert malformed.status_code == 422


# -- which mode the server runs in ---------------------------------------------------------------------------


async def test_a_provider_that_cannot_snapshot_runs_restart_only(
    plain: PlainProvider, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="nemo_gym.sandbox.owner"):
        server = make_owner(None)
    async with client_for(server) as client:
        seeded = await client.post("/ng-rollout/r-a1/seed_session", json={})
        committed = await commit(client, tmp_path / "ckpt")
        verified = await client.post("/ng-rollout/r-a1/verify", json=VERIFY)

    assert server.checkpoint_mode_in_effect() == "restart_only"
    assert seeded.headers.get("x-ng-checkpoint-restart") == "1", "the agent marks the rollout a restart"
    assert committed.status_code == 200, "a commit with the session in flight succeeds; nothing is exported"
    assert resources_records(tmp_path / "ckpt") == []
    assert verified.status_code == 200 and verified.json()["reward"] == 1.0
    warnings = [r.getMessage() for r in caplog.records if "cannot snapshot" in r.getMessage()]
    assert warnings == [
        "owner: sandbox provider 'fake-plain' cannot snapshot; its rollouts restart after a crash instead of continuing"
    ]


def test_a_provider_that_can_snapshot_runs_exported_and_needs_a_state_dir(
    snapshots: SnapshotProvider, tmp_path: Path
) -> None:
    assert make_owner(tmp_path / "state").checkpoint_mode_in_effect() == "exported"
    with pytest.raises(ValueError, match="checkpoint.state_dir is required"):
        make_owner(None)


def test_an_exported_server_with_several_workers_refuses_to_start(snapshots: SnapshotProvider, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="num_workers=2 is not supported yet"):
        make_owner(tmp_path / "state", num_workers=2)
    with pytest.raises(ValueError, match="num_workers=8 is not supported yet"):
        make_owner(tmp_path / "state", num_workers=8)


def test_a_restart_only_server_with_several_workers_starts(plain: PlainProvider) -> None:
    assert make_owner(None, num_workers=2).checkpoint_mode_in_effect() == "restart_only"


def test_an_instance_that_lends_no_sandbox_runs_restart_only_without_a_provider_or_state_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_provider(*_: Any) -> Any:
        raise AssertionError("an instance that lends no sandbox builds no provider at startup")

    monkeypatch.setattr("nemo_gym.sandbox.owner.resolve_provider_config", no_provider)

    server = make_owner(None, num_workers=8, lends=False)

    assert server.checkpoint_mode_in_effect() == "restart_only"
    assert server._state_dir is None and server._sandboxes is None and server._provider is None


def test_without_checkpointing_nothing_is_checked_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nemo_gym.sandbox.owner.resolve_provider_config", lambda *_: pytest.fail("no provider"))

    server = make_owner(None, checkpointing=False, num_workers=8)

    assert server._provider is None and server._state_dir is None
