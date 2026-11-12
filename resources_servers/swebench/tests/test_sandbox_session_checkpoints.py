# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoints of episode sessions over a task sandbox, against an in-process provider with durable snapshots."""

import asyncio
import time
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import Request
from pydantic import BaseModel, JsonValue

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    FakeSandboxProvider,
    FakeSnapshotProvider,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    check_sandbox_session_checkpoints,
    checkpointing_config,
    lost_session_verification,
)
from resources_servers.swebench.sandbox_session_checkpoints import (
    CheckpointedSandboxSessionResourcesServer,
    SandboxCheckpointFields,
)


class _Config(SandboxCheckpointFields, BaseResourcesServerConfig):
    sandbox_provider: str = "sandbox"
    sandbox_config: dict[str, Any] = {}


class _Task(BaseModel):
    note: str


class _VerifyRequest(_Task, BaseVerifyRequest):
    pass


class NotesServer(CheckpointedSandboxSessionResourcesServer):
    """A minimal server: its seed writes a note in the task sandbox, and verify captures the note and grades it."""

    config: _Config
    checkpoint_verify: ClassVar[str] = "replay"
    task_model: ClassVar[type[BaseModel]] = _Task
    provider: Any = None
    fail_setup: bool = False

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        # The task state a restore needs: the note the seed wrote, as SWE-bench keeps the commit it started from.
        self._notes: dict[str, str] = {}

    def shared_provider(self) -> Any:
        return self.provider

    async def _start_task_sandbox(self, session_id: str, task: _Task) -> str:
        spec = SandboxSpec(image="notes", ttl_s=60)

        async def setup(sandbox: AsyncSandbox) -> None:
            if self.fail_setup:
                raise RuntimeError("setup failed")
            await sandbox.exec(f"echo {task.note} > note")

        await self.new_task_sandbox(session_id, spec).start_with_setup(spec, setup)
        self._notes[session_id] = task.note
        return "/testbed"

    def _forget_task_sandbox_state(self, session_id: str) -> None:
        self._notes.pop(session_id, None)

    def _export_task_sandbox_state(self, session_id: str) -> dict[str, JsonValue]:
        return {"note": self._notes[session_id]} if session_id in self._notes else {}

    def _restore_task_sandbox_state(self, session_id: str, state: dict[str, JsonValue]) -> None:
        if "note" in state:
            self._notes[session_id] = str(state["note"])

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        return await self.seed_task_sandbox_session(request, body, _Task)

    async def verify(self, request: Request, body: _VerifyRequest) -> BaseVerifyResponse:
        return await self.verify_task_sandbox_session(
            request,
            body,
            verify=lambda: self._grade(request.session[SESSION_ID_KEY], body),
            lost_response=lambda reason, failure_kind: BaseVerifyResponse.model_validate(
                body.model_dump() | lost_session_verification(reason, failure_kind=failure_kind)
            ),
        )

    async def _grade(self, session_id: str, body: _VerifyRequest) -> BaseVerifyResponse:
        note = await self.capture_task_sandbox_patch(session_id, lambda: self._read_note(session_id))
        await self.stop_task_sandbox_after_capture(session_id)
        return BaseVerifyResponse.model_validate(body.model_dump() | {"reward": float(note == "hello")})

    async def _read_note(self, session_id: str) -> str:
        await self._session_id_to_sandbox[session_id].exec("cat note")
        return self._notes[session_id]


def make_server(provider: Any, global_config: dict, **fields: Any) -> NotesServer:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = global_config
    server = NotesServer(config=_Config(host="", port=0, entrypoint="", name="notes"), server_client=client)
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


def _http(server: NotesServer) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.setup_webserver(), raise_app_exceptions=False),
        base_url="http://resources",
    )


def _auth(global_config: dict) -> dict:
    return {"Authorization": f"Bearer {global_config['checkpoint']['control_auth_token']}"}


def _control(checkpoint_id: str, **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 30, **extra}


async def _post_seed(http: httpx.AsyncClient, seed: ResourcesSeedSessionRequest) -> httpx.Response:
    return await http.post(
        f"/ng-rollout/{seed.episode_id.capture_key}/seed_session", json=seed.model_dump(mode="json")
    )


async def _checkpoint(http: httpx.AsyncClient, config: dict, checkpoint_dir: Path) -> None:
    for operation, body in (
        ("prepare", _control("c1")),
        ("commit", _control("c1", checkpoint_dir=str(checkpoint_dir))),
        ("resume", _control("c1")),
    ):
        reply = await http.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=_auth(config))
        assert reply.status_code == 200, reply.text


async def _restore(http: httpx.AsyncClient, config: dict, checkpoint_dir: Path) -> None:
    for operation, body in (
        ("restore", _control("r1", checkpoint_dir=str(checkpoint_dir), episode_ids=[{"rollout_id": "r"}])),
        ("resume", _control("r1")),
    ):
        reply = await http.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=_auth(config))
        assert reply.status_code == 200, reply.text


async def test_the_notes_server_conforms(tmp_path: Path) -> None:
    await check_sandbox_session_checkpoints(make_server, _seed(), tmp_path / "state")


async def test_a_restored_session_continues_with_its_task_state_and_grades_the_note_it_wrote(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    async with _http(make_server(provider, config)) as http:
        assert (await _post_seed(http, _seed())).status_code == 200
        await _checkpoint(http, config, tmp_path / "ckpt")

    restored = make_server(provider, config)
    async with _http(restored) as http:
        await _restore(http, config, tmp_path / "ckpt")
        reseeded = await _post_seed(http, _seed(attempt=1))
        verified = await http.post(
            "/ng-rollout/r-a1/verify",
            json={"note": "hello", "responses_create_params": {"input": "x"}, "response": _response()},
            cookies=reseeded.cookies,
        )

    assert reseeded.status_code == 200, reseeded.text
    # The fork carries the seed's note: the replacement attempt's seed did not set the task up again.
    [fork] = [box for box in provider.boxes.values() if box["from_snapshot"] is not None]
    assert fork["commands"] == ["echo hello > note", "cat note"]
    assert provider.created_from_image() == 1
    assert verified.status_code == 200, verified.text
    assert verified.json()["reward"] == 1.0


async def test_a_seed_whose_setup_fails_stops_its_sandbox_and_tracks_nothing(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path), fail_setup=True)
    async with _http(server) as http:
        failed = await _post_seed(http, _seed())
        server.fail_setup = False
        retried = await _post_seed(http, _seed())

    assert failed.status_code == 500
    assert retried.status_code == 200, retried.text
    # Only the retry's sandbox is left, and only it is in the checkpointer: the failed seed stopped and forgot its own.
    assert len(provider.live()) == 1
    assert server._checkpointer().keys == ["rs-1"]


async def test_a_commit_waits_for_a_task_sandbox_being_stopped_and_then_exports_the_session_without_it(
    tmp_path: Path,
) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    stopping, finish = asyncio.Event(), asyncio.Event()
    close = provider.close

    async def slow_close(handle: Any) -> None:
        stopping.set()
        await finish.wait()
        await close(handle)

    async with _http(server) as http:
        assert (await _post_seed(http, _seed())).status_code == 200
        provider.close = slow_close
        stop = asyncio.create_task(server.stop_task_sandbox_after_capture("rs-1"))
        await stopping.wait()
        export = asyncio.create_task(server.export_session_states(["rs-1"], checkpoint_id="c1"))
        await asyncio.sleep(0.05)
        waited = not export.done()
        finish.set()
        await stop
        states = await export

    assert waited, "the export waits for the stop holding the session's export lock"
    assert states["rs-1"]["sandbox"] is None
    assert states["rs-1"]["seed"]["task_data"] == {"note": "hello"}
    assert provider.snapshots == {}


async def test_a_capture_a_checkpoint_cannot_store_fails_the_export_loudly(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    async with _http(server) as http:
        assert (await _post_seed(http, _seed())).status_code == 200

        async def unstorable() -> object:
            return object()

        await server.capture_task_sandbox_patch("rs-1", unstorable)
        with pytest.raises(TypeError, match="captured a patch a checkpoint cannot store"):
            await server.export_session_states(["rs-1"], checkpoint_id="c1")


async def test_a_server_whose_fork_cannot_continue_after_the_agent_closed_says_so_on_its_seed(tmp_path: Path) -> None:
    class InPlaceGrader(NotesServer):
        restore_policy: ClassVar[str] = "before_agent_close"

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = checkpointing_config(tmp_path)
    server = InPlaceGrader(config=_Config(host="", port=0, entrypoint="", name="notes"), server_client=client)
    server.provider = FakeSnapshotProvider()
    async with _http(server) as http:
        seeded = await _post_seed(http, _seed())

    assert seeded.headers["x-ng-checkpoint-restore-policy"] == "before_agent_close"


async def test_a_provider_that_cannot_snapshot_needs_no_state_dir(tmp_path: Path) -> None:
    config = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    async with _http(make_server(FakeSandboxProvider(), config)) as http:
        seeded = await _post_seed(http, _seed())

    assert seeded.status_code == 200 and seeded.headers["x-ng-checkpoint-restart"] == "1"


def test_a_provider_that_snapshots_needs_a_state_dir() -> None:
    server = make_server(FakeSnapshotProvider(), {"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    with pytest.raises(ValueError, match="state_dir"):
        server.setup_webserver()


async def test_without_checkpointing_the_server_still_seeds_closes_and_shuts_down(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, {})
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
    assert server._checkpointer().keys == []


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
