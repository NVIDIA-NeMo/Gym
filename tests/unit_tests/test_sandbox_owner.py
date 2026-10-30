# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandbox owner base for resources servers, against an in-process provider with durable snapshots."""

import asyncio
import time
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import Request
from pydantic import BaseModel

from nemo_gym._checkpoint.steps import EpisodeSteps
from nemo_gym.base_resources_server import (
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox_owner import SandboxOwnerConfig, SandboxOwnerResourcesServer, SessionSetup
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.testing.checkpoint_conformance import (
    TOKEN,
    FakeSandboxProvider,
    FakeSnapshotProvider,
    check_sandbox_owner,
    checkpointing_config,
)


AUTH = {"Authorization": f"Bearer {TOKEN}"}


class _Task(BaseModel):
    note: str


class NotesOwner(SandboxOwnerResourcesServer):
    """A minimal owner: its task setup writes a note in the main sandbox; verify captures the note."""

    config: SandboxOwnerConfig
    checkpoint_verify: ClassVar[str] = "replay"
    provider: Any = None
    sidecar: bool = False
    fail_setup: bool = False

    def shared_provider(self) -> Any:
        return self.provider

    async def _set_up(self, session_id: str, body: ResourcesSeedSessionRequest) -> None:
        try:
            task = _Task.model_validate(body.task_data)
        except ValueError as error:
            raise ResourcesSeedRejectedError(str(error)) from error

        async def setup(sandbox: AsyncSandbox) -> SessionSetup:
            if self.fail_setup:
                raise RuntimeError("setup failed")
            await sandbox.exec(f"echo {task.note} > note")
            return SessionSetup(workdir="/testbed", data={"note": task.note})

        await self.start_session_sandbox(session_id, SandboxSpec(image="notes", ttl_s=60), setup)
        if self.sidecar:
            await self.start_session_sandbox(
                session_id,
                SandboxSpec(image="db"),
                lambda sandbox: _setup_result("/db"),
                name="db",
            )

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> Any:
        return await self.seed_task_sandbox_session(request, body, self._set_up)

    async def verify(self, request: Request, body: BaseVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        note = await self.capture_once(
            session_id, "captured", lambda: _captured(self, session_id), release_sandbox_after=True
        )
        return BaseVerifyResponse(**body.model_dump(), reward=float(note == "hello"))


async def _setup_result(workdir: str) -> SessionSetup:
    return SessionSetup(workdir=workdir)


async def _captured(server: NotesOwner, session_id: str) -> str:
    await server.require_session_sandbox(session_id).exec("cat note")
    return server.session_data(session_id)["note"]


def make_owner(provider: Any, global_config: dict, **fields: Any) -> NotesOwner:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = global_config
    server = NotesOwner(
        config=SandboxOwnerConfig(host="", port=0, entrypoint="", name="notes", sandbox_provider="sandbox"),
        server_client=client,
    )
    server.provider = provider
    for name, value in fields.items():
        setattr(server, name, value)
    return server


def _seed(attempt: int = 0, session_id: str = "rs-1", note: str = "hello") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="r", attempt=attempt),
        task_id=TaskId(taskset="notes", task_id="t"),
        task_data={"note": note},
    )


def _http(server: NotesOwner) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.setup_webserver(), raise_app_exceptions=False),
        base_url="http://owner",
    )


def _control(checkpoint_id: str, **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 30, **extra}


async def _post_seed(http: httpx.AsyncClient, seed: ResourcesSeedSessionRequest) -> httpx.Response:
    return await http.post(
        f"/ng-rollout/{seed.episode_id.capture_key}/seed_session", json=seed.model_dump(mode="json")
    )


async def _capture(server: NotesOwner, session_id: str) -> None:
    await server.capture_once(
        session_id, "captured", lambda: _captured(server, session_id), release_sandbox_after=True
    )


async def test_the_notes_owner_conforms(tmp_path: Path) -> None:
    await check_sandbox_owner(make_owner, _seed(), tmp_path / "state", capture=_capture)


async def test_a_seed_whose_setup_fails_stops_its_sandbox_and_tracks_nothing(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_owner(provider, checkpointing_config(tmp_path), fail_setup=True)
    async with _http(server) as http:
        failed = await _post_seed(http, _seed())
        server.fail_setup = False
        retried = await _post_seed(http, _seed())

    assert failed.status_code == 500
    assert retried.status_code == 200, retried.text
    # Only the retry's sandbox is left: the failed seed stopped its own.
    assert len(provider.live()) == 1


async def test_a_seed_with_invalid_task_data_is_refused_without_a_sandbox(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_owner(provider, checkpointing_config(tmp_path))
    bad = _seed().model_copy(update={"task_data": {"other": 1}})
    async with _http(server) as http:
        refused = await _post_seed(http, bad)

    assert refused.status_code == 422
    assert provider.boxes == {}


async def test_every_sandbox_of_a_session_is_exported_and_forked_together(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    server = make_owner(provider, config, sidecar=True)
    async with _http(server) as http:
        assert (await _post_seed(http, _seed())).status_code == 200
        for operation, body in (
            ("prepare", _control("c1")),
            ("commit", _control("c1", checkpoint_dir=str(tmp_path / "ckpt"))),
            ("resume", _control("c1")),
        ):
            assert (
                await http.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=AUTH)
            ).status_code == 200
    assert len(provider.snapshots) == 2

    restored = make_owner(provider, config, sidecar=True)
    async with _http(restored) as http:
        for operation, body in (
            ("restore", _control("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}])),
            ("resume", _control("r1")),
        ):
            assert (
                await http.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=AUTH)
            ).status_code == 200
        assert (await _post_seed(http, _seed(attempt=1))).status_code == 200

    assert set(restored.session_sandboxes("rs-1")) == {"main", "db"}
    assert {
        provider.boxes[sandbox.sandbox_id]["from_snapshot"] for sandbox in restored.session_sandboxes("rs-1").values()
    } == set(provider.snapshots)


async def test_a_commit_during_a_capture_waits_for_it_and_exports_the_captured_data_alone(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_owner(provider, checkpointing_config(tmp_path / "state"))
    capturing, finish = asyncio.Event(), asyncio.Event()

    async def slow_capture() -> str:
        capturing.set()
        await finish.wait()
        return "patch"

    async with _http(server) as http:
        assert (await _post_seed(http, _seed())).status_code == 200
        capture = asyncio.create_task(server.capture_once("rs-1", "patch", slow_capture, release_sandbox_after=True))
        await capturing.wait()
        export = asyncio.create_task(server.export_session_states(["rs-1"], checkpoint_id="c1"))
        await asyncio.sleep(0.05)
        waited = not export.done()
        finish.set()
        await capture
        states = await export

    assert waited, "the export waits for the capture holding the session's lock"
    assert states["rs-1"]["sandboxes"] == {}
    assert states["rs-1"]["data"] == {"note": "hello", "patch": "patch"}
    assert provider.snapshots == {}


async def test_a_failed_capture_stores_nothing_and_keeps_the_sandbox(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_owner(provider, checkpointing_config(tmp_path))

    async def failing() -> str:
        raise RuntimeError("git failed")

    async with _http(server) as http:
        assert (await _post_seed(http, _seed())).status_code == 200
        with pytest.raises(RuntimeError):
            await server.capture_once("rs-1", "patch", failing, release_sandbox_after=True)

    assert "patch" not in server.session_data("rs-1")
    assert server.session_sandbox("rs-1") is not None and len(provider.live()) == 1


async def test_an_owner_whose_fork_cannot_continue_after_the_agent_closed_says_so_on_its_seed(tmp_path: Path) -> None:
    class InPlaceGrader(NotesOwner):
        restore_policy: ClassVar[str] = "before_agent_close"

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = checkpointing_config(tmp_path)
    server = InPlaceGrader(
        config=SandboxOwnerConfig(host="", port=0, entrypoint="", name="notes", sandbox_provider="sandbox"),
        server_client=client,
    )
    server.provider = FakeSnapshotProvider()
    async with _http(server) as http:
        seeded = await _post_seed(http, _seed())

    assert seeded.headers["x-ng-checkpoint-restore-policy"] == "before_agent_close"


async def test_a_session_nobody_closed_is_released_after_its_lifetime(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_owner(provider, checkpointing_config(tmp_path))
    async with _http(server) as http:
        assert (await _post_seed(http, _seed(session_id="abandoned"))).status_code == 200
        server._sessions["abandoned"].lifetime_s = 0.0
        assert (await _post_seed(http, _seed(session_id="next"))).status_code == 200
        for _ in range(100):
            if server.session_sandbox("abandoned") is None:
                break
            await asyncio.sleep(0.01)
        late = await _post_seed(http, _seed(session_id="abandoned"))

    assert server.session_expired("abandoned")
    assert server.session_sandbox("abandoned") is None
    assert late.status_code == 409
    assert len(provider.live()) == 1


async def test_a_provider_that_cannot_snapshot_needs_no_state_dir(tmp_path: Path) -> None:
    server = make_owner(FakeSandboxProvider(), {"checkpoint": {"enabled": True, "control_auth_token": TOKEN}})
    async with _http(server) as http:
        seeded = await _post_seed(http, _seed())

    assert seeded.status_code == 200 and seeded.headers["x-ng-checkpoint-restart"] == "1"


def test_a_provider_that_snapshots_needs_a_state_dir() -> None:
    server = make_owner(FakeSnapshotProvider(), {"checkpoint": {"enabled": True, "control_auth_token": TOKEN}})
    with pytest.raises(ValueError, match="state_dir"):
        server.setup_webserver()


async def test_without_checkpointing_the_owner_still_seeds_closes_and_shuts_down(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_owner(provider, {})
    async with _http(server) as http:
        assert (await _post_seed(http, _seed(session_id="a"))).status_code == 200
        assert (await _post_seed(http, _seed(session_id="b"))).status_code == 200
        closed = await http.post(
            "/close_session", json={"resources_session_id": "a", "episode_id": {"rollout_id": "r", "attempt": 0}}
        )
        assert closed.status_code == 200
    assert len(provider.live()) == 1
    await server.shutdown()
    assert provider.live() == []


async def test_a_blocking_episode_becomes_a_restart_at_once_while_a_checkpoint_is_open() -> None:
    notified: list[bool] = []

    async def notify() -> None:
        notified.append(True)

    steps = EpisodeSteps(notify)
    steps.begin("r")
    steps.close()
    # The episode is between boundaries, so the open checkpoint waits for it: marking it cannot wait for resume.
    await asyncio.wait_for(steps.mark_restart("r"), 1)

    assert steps.restarts() == ["r"] and steps.blocker_count() == 0


async def test_a_parked_episode_becomes_a_restart_only_after_resume() -> None:
    async def notify() -> None:
        pass

    steps = EpisodeSteps(notify)
    steps.begin("r")
    steps.close()
    parked = asyncio.create_task(steps.boundary("r", {"next": "verify"}))
    await asyncio.sleep(0.01)
    marking = asyncio.create_task(steps.mark_restart("r"))
    await asyncio.sleep(0.01)
    during = steps.restarts()
    steps.open()
    await asyncio.wait_for(asyncio.gather(parked, marking), 1)

    assert during == [] and steps.restarts() == ["r"]
