# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoints of episode sessions over a task sandbox, against an in-process provider with durable snapshots."""

import asyncio
import logging
import time
from pathlib import Path
from time import monotonic
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import anyio
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
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SandboxCheckpointError,
    lost_session_verification,
    seed_restarts,
)
from resources_servers.swebench.sandbox_session_checkpoints import (
    CheckpointedSandboxSessionResourcesServer,
    SandboxCheckpointFields,
)
from resources_servers.swebench.testing.checkpoint_conformance import (
    FakeSandboxProvider,
    FakeSnapshotProvider,
    check_sandbox_session_checkpoints,
    checkpointing_config,
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


class AgentRunSeeds(NotesServer):
    """Also takes the task itself as a seed, which is what an agent's ``/run`` sends the SWE servers."""

    async def seed_session(
        self, request: Request, body: ResourcesSeedSessionRequest | _Task
    ) -> ResourcesSeedSessionResponse:
        if isinstance(body, ResourcesSeedSessionRequest):
            return await super().seed_session(request, body)
        session_id = request.session[SESSION_ID_KEY]
        async with self._locked_task_session(session_id) as record:
            await self._stop_task_sandbox(session_id)
            record.workdir = await self._start_task_sandbox(session_id, body)
            return ResourcesSeedSessionResponse(resources_session_id=session_id)


class GradesAfterCapture(NotesServer):
    """Holds a verify once it captured the note and stopped the task sandbox, as a grading that takes long does."""

    grading: Any = None
    finish: Any = None

    async def stop_task_sandbox_after_capture(self, session_id: str) -> None:
        await super().stop_task_sandbox_after_capture(session_id)
        self.grading.set()
        await self.finish.wait()


def make_server(
    provider: Any, global_config: dict, *, server_type: type[NotesServer] = NotesServer, **fields: Any
) -> NotesServer:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = global_config
    server = server_type(config=_Config(host="", port=0, entrypoint="", name="notes"), server_client=client)
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


# Attempt 7 too: the kit must derive the other episodes it seeds from the seed it is given.
@pytest.mark.parametrize("attempt", [0, 7])
async def test_the_notes_server_conforms(tmp_path: Path, attempt: int) -> None:
    await check_sandbox_session_checkpoints(
        make_server, _seed(attempt=attempt), tmp_path / "state", verify_body=_verify_body()
    )


class ForgetsNoTaskState(NotesServer):
    def _forget_task_sandbox_state(self, session_id: str) -> None:
        pass


class RestoresNoTaskState(NotesServer):
    def _restore_task_sandbox_state(self, session_id: str, state: dict[str, JsonValue]) -> None:
        pass


class RestoresTaskStateWhereVerifyDoesNotReadIt(NotesServer):
    """Exports, restores and forgets its task state, but its verify reads only what a seed recorded."""

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._restored_notes: dict[str, str] = {}

    def _forget_task_sandbox_state(self, session_id: str) -> None:
        super()._forget_task_sandbox_state(session_id)
        self._restored_notes.pop(session_id, None)

    def _export_task_sandbox_state(self, session_id: str) -> dict[str, JsonValue]:
        note = self._notes.get(session_id, self._restored_notes.get(session_id))
        return {} if note is None else {"note": note}

    def _restore_task_sandbox_state(self, session_id: str, state: dict[str, JsonValue]) -> None:
        self._restored_notes[session_id] = str(state["note"])


@pytest.mark.parametrize(
    "server_type, failure",
    [
        (ForgetsNoTaskState, "a close forgets the session's task state"),
        (RestoresNoTaskState, "a restore installs the task state the commit exported"),
        # Only the server's own /verify, on a checkpoint taken before it captured the patch, finds this one.
        (RestoresTaskStateWhereVerifyDoesNotReadIt, "/verify after a restore replies as it does without the crash"),
    ],
)
async def test_the_kit_refuses_a_server_that_loses_or_keeps_task_state_it_should_not(
    tmp_path: Path, server_type: type[NotesServer], failure: str
) -> None:
    def make(provider: Any, global_config: dict) -> NotesServer:
        return make_server(provider, global_config, server_type=server_type)

    with pytest.raises(AssertionError, match=failure):
        await check_sandbox_session_checkpoints(make, _seed(), tmp_path / "state", verify_body=_verify_body())


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
    assert server._checkpointer().keys == ["rs-1/task"]


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
        async with asyncio.timeout(10):
            await stopping.wait()
        export = asyncio.create_task(server.export_session_states(["rs-1"], checkpoint_id="c1"))
        await asyncio.sleep(0.05)
        waited = not export.done()
        finish.set()
        await stop
        states = await export

    assert waited, "the export waits for the stop holding the session's export lock"
    assert states["rs-1"]["sandboxes"] == {"task": None}
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

    # Refused before the sandbox was snapshotted: every later commit would fail the same way.
    assert not [call for call in provider.calls if call[0] == "snapshot"]


async def test_a_provider_that_cannot_snapshot_needs_no_state_dir(tmp_path: Path) -> None:
    config = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    async with _http(make_server(FakeSandboxProvider(), config)) as http:
        seeded = await _post_seed(http, _seed())

    assert seeded.status_code == 200 and seed_restarts(seeded.headers)


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


def _verify_body() -> dict:
    return {"note": "hello", "responses_create_params": {"input": "x"}, "response": _response()}


def _close_path(seed: ResourcesSeedSessionRequest) -> tuple[str, dict]:
    body = {"resources_session_id": seed.resources_session_id, "episode_id": seed.episode_id.model_dump(mode="json")}
    return f"/ng-rollout/{seed.episode_id.capture_key}/close_session", body


def _running_forks(provider: FakeSnapshotProvider) -> list[str]:
    return [
        sandbox_id
        for sandbox_id, box in provider.boxes.items()
        if box["from_snapshot"] is not None and box["state"] == "running"
    ]


def _started(server: NotesServer) -> NotesServer:
    """``server`` with its app set up: only then does it know whether it exports sessions, and so its state_dir."""
    server.setup_webserver()
    return server


async def _seeded(server: NotesServer, session_ids: list[str]) -> NotesServer:
    if server._checkpoint is None:
        _started(server)
    for session_id in session_ids:
        body = _seed(session_id=session_id)
        await server.seed_task_sandbox_session(_request(session_id), body, _Task)
    return server


async def _exported(server: NotesServer, session_ids: list[str]) -> dict[str, JsonValue]:
    await _seeded(server, session_ids)
    return await server.export_session_states(session_ids, checkpoint_id="c1")


def _snapshotted(provider: FakeSnapshotProvider) -> list[str]:
    """The sandbox of every snapshot call, in order."""
    return [sandbox_id for call, sandbox_id in provider.calls if call == "snapshot"]


def _request(session_id: str) -> Any:
    request = MagicMock()
    request.session = {SESSION_ID_KEY: session_id}
    return request


# --- stops and the export lock ------------------------------------------------------------------------------------


async def test_a_release_ends_the_wait_of_a_stop_for_a_claimed_task_sandbox(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    await _exported(server, ["rs-1"])
    # An older consumer claims the sandbox for verify and takes it out of the registry until it releases it.
    server._claim_task_sandbox("rs-1")
    sandbox = server._session_id_to_sandbox.pop("rs-1")
    stopping = asyncio.create_task(server._stop_task_sandbox("rs-1"))
    await asyncio.sleep(0.05)
    assert not stopping.done(), "the stop waits for the claim to be released"

    # The stop must not hold the export lock while it waits: the release takes it.
    async with asyncio.timeout(2):
        await server._release_task_sandbox("rs-1", sandbox)
        await stopping

    assert provider.live() == []
    assert server._checkpointer().keys == []


async def test_a_commit_waits_for_a_release_that_is_stopping_the_task_sandbox(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    await _exported(server, ["rs-1"])
    server._claim_task_sandbox("rs-1")
    sandbox = server._session_id_to_sandbox.pop("rs-1")
    stopping, finish = asyncio.Event(), asyncio.Event()
    close = provider.close

    async def slow_close(handle: Any) -> None:
        stopping.set()
        await finish.wait()
        await close(handle)

    provider.close = slow_close
    release = asyncio.create_task(server._release_task_sandbox("rs-1", sandbox))
    async with asyncio.timeout(10):
        await stopping.wait()
    export = asyncio.create_task(server.export_session_states(["rs-1"], checkpoint_id="c2"))
    await asyncio.sleep(0.05)
    waited = not export.done()
    finish.set()
    await release
    states = await export

    assert waited, "the export waits for the release holding the session's export lock"
    assert states["rs-1"]["sandboxes"] == {"task": None}
    assert _snapshotted(provider) == ["sb-1"], "the sandbox being stopped was not snapshotted again"


async def test_a_commit_after_a_release_exports_the_session_without_a_sandbox(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    await _exported(server, ["rs-1"])
    server._claim_task_sandbox("rs-1")

    await server._release_task_sandbox("rs-1", server._session_id_to_sandbox.pop("rs-1"))
    states = await server.export_session_states(["rs-1"], checkpoint_id="c2")

    # The release untracked the sandbox: the checkpointer no longer holds what the server stopped.
    assert provider.live() == [] and server._checkpointer().keys == []
    assert states["rs-1"]["sandboxes"] == {"task": None} and _snapshotted(provider) == ["sb-1"]


async def test_a_close_waits_for_a_commit_snapshotting_its_task_sandbox(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    snapshotting, finish = asyncio.Event(), asyncio.Event()
    snapshot = provider.snapshot

    async def slow_snapshot(handle: Any, **kwargs: Any) -> str:
        snapshotting.set()
        await finish.wait()
        return await snapshot(handle, **kwargs)

    async with _http(server) as http:
        assert (await _post_seed(http, _seed())).status_code == 200
        provider.snapshot = slow_snapshot
        export = asyncio.create_task(server.export_session_states(["rs-1"], checkpoint_id="c1"))
        async with asyncio.timeout(10):
            await snapshotting.wait()
        path, body = _close_path(_seed())
        close = asyncio.create_task(http.post(path, json=body))
        await asyncio.sleep(0.05)
        waited = not close.done() and len(provider.live()) == 1
        finish.set()
        states = await export
        closed = await close

    assert waited, "the close does not stop a sandbox a commit is snapshotting"
    assert states["rs-1"]["sandboxes"]["task"] is not None, "the commit exported the session with its snapshot"
    assert closed.status_code == 200, closed.text
    assert provider.live() == [] and server._checkpointer().keys == []


async def test_a_close_drops_the_session_from_the_checkpointer_without_waiting_for_a_turn(tmp_path: Path) -> None:
    """A client that disconnects cancels a close at whatever it waits for, once the sandbox has stopped."""
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    server.config.sandbox_checkpoint.parallelism = 1
    async with _http(server) as http:
        assert (await _post_seed(http, _seed())).status_code == 200
        checkpointer = server._checkpointer()
        path, body = _close_path(_seed())
        async with checkpointer._semaphore:  # every turn is taken, as while a commit submits its snapshots
            closed = await asyncio.wait_for(http.post(path, json=body), timeout=10)

    assert closed.status_code == 200, closed.text
    assert provider.live() == [] and checkpointer.keys == []


async def test_a_capture_waits_for_a_commit_snapshotting_its_task_sandbox(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    snapshotting, finish = asyncio.Event(), asyncio.Event()
    snapshot = provider.snapshot

    async def slow_snapshot(handle: Any, **kwargs: Any) -> str:
        snapshotting.set()
        await finish.wait()
        return await snapshot(handle, **kwargs)

    await _seeded(server, ["rs-1"])
    provider.snapshot = slow_snapshot
    export = asyncio.create_task(server.export_session_states(["rs-1"], checkpoint_id="c1"))
    async with asyncio.timeout(10):
        await snapshotting.wait()
    # A replayable verify is admitted during a commit; its capture must not run in the sandbox being snapshotted,
    # nor leave the commit with a capture it read before and a "captured" flag it read after.
    capture = asyncio.create_task(server.capture_task_sandbox_patch("rs-1", lambda: server._read_note("rs-1")))
    await asyncio.sleep(0.05)
    waited = not capture.done()
    finish.set()
    states = await export

    assert waited, "the capture waits for the commit holding the session's export lock"
    assert states["rs-1"]["sandboxes"]["task"] is not None
    assert (states["rs-1"]["capture_ready"], states["rs-1"]["capture"]) == (False, None)
    assert await capture == "hello"
    await server.shutdown()


async def test_an_expired_session_stops_its_task_sandbox_and_is_no_longer_exported(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    await _exported(server, ["rs-1"])
    server._open_task_sessions["rs-1"] = 0.0

    await server._close_abandoned_task_sessions(monotonic())

    assert provider.live() == [] and server._checkpointer().keys == []
    assert await server.export_session_states(["rs-1"], checkpoint_id="c2") == {}


async def test_retiring_a_session_stops_its_task_sandbox_and_keeps_its_snapshots(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    await _exported(server, ["rs-1"])
    snapshots = dict(provider.snapshots)

    await server.retire_session_state("rs-1")
    # A retire that was cut short runs again for the same session.
    await server.retire_session_state("rs-1")

    assert provider.live() == [] and server._checkpointer().keys == []
    assert "rs-1" not in server._task_sessions and "rs-1" not in server._notes
    assert len(snapshots) == 1 and provider.snapshots == snapshots, "a kept checkpoint may still restore the attempt"


async def test_registering_a_second_task_sandbox_for_a_session_is_refused_and_keeps_the_first(tmp_path: Path) -> None:
    server = make_server(FakeSnapshotProvider(), checkpointing_config(tmp_path / "state"))
    spec = SandboxSpec(image="notes", ttl_s=60)
    first = server.new_task_sandbox("rs-1", spec)

    with pytest.raises(ValueError, match="already has a sandbox"):
        server.new_task_sandbox("rs-1", spec)

    assert server._session_id_to_sandbox["rs-1"] is first


# --- commit --------------------------------------------------------------------------------------------------------


async def test_a_commit_fails_for_a_task_sandbox_that_was_not_started_with_new_task_sandbox(tmp_path: Path) -> None:
    class DirectRegistration(NotesServer):
        async def _start_task_sandbox(self, session_id: str, task: _Task) -> str:
            spec = SandboxSpec(image="notes", ttl_s=60)
            sandbox = AsyncSandbox(self.shared_provider(), spec, owns_provider=False)
            self._session_id_to_sandbox[session_id] = sandbox
            await sandbox.start()
            return "/testbed"

    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"), server_type=DirectRegistration)

    # Exporting it without a snapshot would restore a session that has no sandbox.
    with pytest.raises(RuntimeError, match="without new_task_sandbox"):
        await _exported(server, ["rs-1"])
    assert provider.snapshots == {}
    await server.shutdown()


async def test_a_snapshot_that_fails_once_is_taken_again(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    server.config.sandbox_checkpoint.snapshot_retry_delay_s = 0
    provider.fail_snapshots = 1

    states = await _exported(server, ["rs-1"])

    assert [call for call in provider.calls if call[0] == "snapshot"] == [("snapshot", "sb-1"), ("snapshot", "sb-1")]
    assert states["rs-1"]["sandboxes"]["task"]["snapshot_id"] in provider.snapshots


async def test_a_new_process_deletes_the_snapshots_a_crashed_one_took(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    await _exported(make_server(provider, config), ["rs-1"])
    assert len(provider.snapshots) == 1

    # It has created no sandbox: the index under state_dir names what the crashed process took.
    replacement = _started(make_server(provider, config))
    await replacement.delete_external_state(frozenset())
    await replacement._checkpointer().settle()

    assert provider.snapshots == {}


# --- restore -------------------------------------------------------------------------------------------------------


async def test_a_session_checkpointed_after_its_capture_is_restored_without_a_sandbox(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    server = make_server(provider, config)
    await _exported(server, ["rs-1"])
    captured = await server.capture_task_sandbox_patch("rs-1", lambda: server._read_note("rs-1"))
    await server.stop_task_sandbox_after_capture("rs-1")
    states = await server.export_session_states(["rs-1"], checkpoint_id="c2")

    restored = make_server(provider, config)
    await restored.restore_session_states(states)
    again = await restored.capture_task_sandbox_patch("rs-1", AsyncMock(side_effect=AssertionError("captured again")))

    assert states["rs-1"]["sandboxes"] == {"task": None} and states["rs-1"]["capture"] == captured == "hello"
    assert again == "hello", "the restored session grades the capture the checkpoint saved"
    assert _running_forks(provider) == [] and "rs-1" not in restored._session_id_to_sandbox


async def test_a_restore_that_fails_after_forking_stops_the_forks_and_can_be_retried(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    states = await _exported(make_server(provider, config), ["a", "b"])
    restored = make_server(provider, config)
    response = restored._task_sandbox_response

    async def failing(session_id: str, record: Any) -> ResourcesSeedSessionResponse:
        if session_id == "b":
            raise RuntimeError("provider unavailable")
        return await response(session_id, record)

    object.__setattr__(restored, "_task_sandbox_response", failing)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        await restored.restore_session_states(states)

    # All or none: "a" was installed before "b" failed, and both forks were running.
    assert _running_forks(provider) == []
    assert restored._task_sessions == {} and restored._session_id_to_sandbox == {} and restored._notes == {}
    assert restored._open_task_sessions == {} and restored._checkpointer().keys == []

    object.__setattr__(restored, "_task_sandbox_response", response)
    await restored.restore_session_states(states)
    assert len(_running_forks(provider)) == 2 and sorted(restored._task_sessions) == ["a", "b"]
    await restored.shutdown()


async def test_a_restore_cancelled_while_installing_stops_the_forks_and_installs_nothing(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    states = await _exported(make_server(provider, config), ["a", "b"])
    restored = make_server(provider, config)
    response = restored._task_sandbox_response
    installing = asyncio.Event()

    async def hanging(session_id: str, record: Any) -> ResourcesSeedSessionResponse:
        if session_id == "b":
            installing.set()
            await asyncio.Event().wait()
        return await response(session_id, record)

    object.__setattr__(restored, "_task_sandbox_response", hanging)
    restoring = asyncio.create_task(restored.restore_session_states(states))
    async with asyncio.timeout(10):
        await installing.wait()
    # A restore past its deadline is cancelled: "a" is installed and both forks are running.
    restoring.cancel()
    with pytest.raises(asyncio.CancelledError):
        await restoring

    assert _running_forks(provider) == []
    assert restored._task_sessions == {} and restored._session_id_to_sandbox == {} and restored._notes == {}
    assert restored._open_task_sessions == {} and restored._checkpointer().keys == []


async def test_a_failed_restore_does_not_wait_for_a_fork_whose_stop_hangs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    states = await _exported(make_server(provider, config), ["a", "b"])
    restored = make_server(provider, config)
    object.__setattr__(restored, "_task_sandbox_response", AsyncMock(side_effect=RuntimeError("install failed")))
    monkeypatch.setattr("nemo_gym.server_runtime._checkpoint.sandbox.STOP_TIMEOUT_S", 0.05)
    close = provider.close

    async def hung(handle: Any) -> None:
        # Only the forks: the restore also stops the sandboxes it replaced, in the background.
        if provider.boxes[handle.sandbox_id]["from_snapshot"] is None:
            return await close(handle)
        await asyncio.Event().wait()

    provider.close = hung
    # The rollback is shielded from cancellation, so a stop without a ceiling would hold the restore forever.
    async with asyncio.timeout(10):
        with pytest.raises(RuntimeError, match="install failed"):
            await restored.restore_session_states(states)

    # The sessions are free at once; the forks that did not stop are left to the checkpointer, so shutdown stops them.
    assert restored._task_sessions == {} and restored._session_id_to_sandbox == {}
    assert restored._checkpointer().keys == [] and len(_running_forks(provider)) == 2
    provider.close = close
    await restored.shutdown()
    assert _running_forks(provider) == []


async def test_shutdown_stops_the_forks_a_failed_restore_could_not_stop(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    states = await _exported(make_server(provider, config), ["a", "b"])
    restored = make_server(provider, config)
    object.__setattr__(restored, "_task_sandbox_response", AsyncMock(side_effect=RuntimeError("install failed")))
    close = provider.close
    provider.close = AsyncMock(side_effect=RuntimeError("provider unavailable"))

    with pytest.raises(RuntimeError, match="install failed"):
        await restored.restore_session_states(states)
    await restored._checkpointer().settle()
    # Neither session holds its fork any more: the checkpointer keeps both to stop them again.
    assert len(_running_forks(provider)) == 2 and restored._task_sessions == {}
    assert restored._session_id_to_sandbox == {} and restored._checkpointer().keys == []

    provider.close = close
    await restored.shutdown()
    assert _running_forks(provider) == []


async def test_a_restore_can_be_retried_while_the_forks_of_a_failed_one_cannot_be_stopped(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    states = await _exported(make_server(provider, config), ["a", "b"])
    restored = make_server(provider, config)
    response = restored._task_sandbox_response
    object.__setattr__(restored, "_task_sandbox_response", AsyncMock(side_effect=RuntimeError("install failed")))
    close = provider.close

    async def unavailable(handle: Any) -> None:
        # Only the forks: the restore also stops the sandboxes it replaced, in the background.
        if provider.boxes[handle.sandbox_id]["from_snapshot"] is None:
            return await close(handle)
        raise RuntimeError("provider unavailable")

    provider.close = unavailable
    with pytest.raises(RuntimeError, match="install failed"):
        await restored.restore_session_states(states)
    assert len(_running_forks(provider)) == 2

    # The install works again, the provider still does not: the retry is not refused for the forks left running.
    object.__setattr__(restored, "_task_sandbox_response", response)
    await restored.restore_session_states(states)

    assert sorted(restored._task_sessions) == ["a", "b"] and restored._checkpointer().keys == ["a/task", "b/task"]
    assert len(_running_forks(provider)) == 4, "the retry forked again beside the two forks that did not stop"
    forks = {sandbox.sandbox_id for sandbox in restored._session_id_to_sandbox.values()}
    assert len(forks) == 2 and forks <= set(_running_forks(provider))

    provider.close = close
    await restored.shutdown()
    assert _running_forks(provider) == []


@pytest.mark.parametrize(
    "change, error",
    [
        ({"workdir": None}, "invalid notes session state"),
        ({"unknown": 1}, "invalid notes session state"),
        ({"seed": {"task_data": {}}}, "invalid notes session state"),
        # A later format is named by its version, whatever else it changed.
        ({"version": 2, "unknown": 1}, "has version 2; this server reads version 1"),
        ({"sandboxes": {"sidecar": None}}, r"saves sandboxes \['sidecar'\], and this server restores only 'task'"),
    ],
)
async def test_a_restore_of_an_invalid_session_state_forks_nothing(tmp_path: Path, change: dict, error: str) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    states = await _exported(make_server(provider, config), ["rs-1"])
    state = dict(states["rs-1"])
    for name, value in change.items():
        state[name] = state[name] | value if isinstance(value, dict) else value
    restored = make_server(provider, config)

    with pytest.raises(ValueError, match=error):
        await restored.restore_session_states({"rs-1": state})

    assert _running_forks(provider) == [] and restored._task_sessions == {}


async def test_a_restore_of_a_session_the_worker_already_holds_forks_nothing(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    states = await _exported(server, ["rs-1"])

    with pytest.raises(ValueError, match="already held by this worker"):
        await server.restore_session_states(states)

    assert _running_forks(provider) == [] and len(provider.live()) == 1


# --- startup and shutdown ------------------------------------------------------------------------------------------


def test_a_server_that_exports_sessions_without_a_task_model_fails_at_startup(tmp_path: Path) -> None:
    class NoTaskModel(NotesServer):
        task_model: ClassVar[Any] = None

    server = make_server(FakeSnapshotProvider(), checkpointing_config(tmp_path), server_type=NoTaskModel)
    # Otherwise the first restore after a crash is what finds out.
    with pytest.raises(TypeError, match="task_model must be the pydantic model"):
        server.setup_webserver()


def test_a_provider_that_cannot_snapshot_is_warned_about_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server = make_server(FakeSandboxProvider(), checkpointing_config(tmp_path))

    with caplog.at_level(logging.WARNING):
        modes = [server.checkpoint_mode_in_effect() for _ in range(3)]

    assert modes == ["restart_only"] * 3
    assert [record.getMessage() for record in caplog.records if "cannot snapshot" in record.getMessage()] == [
        "notes: sandbox provider 'fake' cannot snapshot; its episodes restart after a crash instead of continuing"
    ]


async def test_the_shared_provider_is_built_once_from_the_config_and_closed_at_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ConfiguredProvider(NotesServer):
        shared_provider = CheckpointedSandboxSessionResourcesServer.shared_provider

    module = "resources_servers.swebench.sandbox_session_checkpoints"
    provider = FakeSnapshotProvider()
    provider.aclose = AsyncMock()
    create = MagicMock(return_value=provider)
    monkeypatch.setattr(f"{module}.get_global_config_dict", lambda: {"sandbox": {"fake": {}}})
    monkeypatch.setattr(f"{module}.create_provider", create)
    server = make_server(None, checkpointing_config(tmp_path / "state"), server_type=ConfiguredProvider)

    await _exported(server, ["rs-1"])
    assert server.shared_provider() is provider
    create.assert_called_once_with({"fake": {}})

    await server.shutdown()
    assert provider.live() == []
    provider.aclose.assert_awaited_once_with()


# --- the saved format ------------------------------------------------------------------------------------------------


async def test_a_commit_saves_the_task_sandbox_by_name_under_a_format_version(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    server = make_server(provider, config)

    states = await _exported(server, ["rs-1"])

    assert states["rs-1"]["version"] == 1 and list(states["rs-1"]["sandboxes"]) == ["task"]
    assert states["rs-1"]["sandboxes"]["task"]["snapshot_id"] in provider.snapshots
    # The checkpointer tracks the sandbox by session and name, so a session can hold more than one.
    assert server._checkpointer().keys == ["rs-1/task"]

    restored = make_server(provider, config)
    await restored.restore_session_states(states)
    fork = restored._session_id_to_sandbox["rs-1"]
    assert restored._checkpointer().get("rs-1/task") is fork and [fork.sandbox_id] == _running_forks(provider)

    await restored.retire_session_state("rs-1")
    assert restored._checkpointer().keys == [] and _running_forks(provider) == []


# --- a seed that is not an episode seed ------------------------------------------------------------------------------


async def test_a_session_seeded_without_an_episode_seed_is_a_restart_that_no_commit_snapshots(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    server = make_server(provider, config, server_type=AgentRunSeeds)
    async with _http(server) as http:
        seeded = await http.post("/ng-rollout/r/seed_session", json={"note": "hello"})
        prepared = await http.post("/ng-control/v1/checkpoint/prepare", json=_control("c1"), headers=_auth(config))
        # The episode is a restart, so the controller leaves it out of the commit's scope.
        commit = _control("c1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[])
        committed = await http.post("/ng-control/v1/checkpoint/commit", json=commit, headers=_auth(config))
        await http.post("/ng-control/v1/checkpoint/resume", json=_control("c1"), headers=_auth(config))
        [session_id] = server._session_id_to_sandbox
        running = len(provider.live())
        closed = await http.post(
            "/ng-rollout/r/close_session",
            json={"resources_session_id": session_id, "episode_id": {"rollout_id": "r", "attempt": 0}},
        )

    # No restore could install the session, which has no episode seed, so its reply says the episode starts over.
    assert seeded.status_code == 200, seeded.text
    assert seed_restarts(seeded.headers)
    assert prepared.json()["report"]["restarts"] == ["r"]
    assert committed.status_code == 200 and committed.json()["manifest"]["record_count"] == 0
    assert provider.snapshots == {} and _snapshotted(provider) == []
    # The session still owns its sandbox: a close stops it.
    assert running == 1 and closed.status_code == 200 and provider.live() == []
    assert server._checkpointer().keys == []


@pytest.mark.parametrize("checkpointing", [True, False])
async def test_shutdown_stops_the_task_sandbox_of_a_session_seeded_without_an_episode_seed(
    tmp_path: Path, checkpointing: bool
) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state") if checkpointing else {}
    server = make_server(provider, config, server_type=AgentRunSeeds)
    async with _http(server) as http:
        seeded = await http.post("/ng-rollout/r/seed_session", json={"note": "hello"})

    assert seeded.status_code == 200, seeded.text
    # Only a server that exports sessions has anything to say about this one.
    assert seed_restarts(seeded.headers) is checkpointing
    assert len(provider.live()) == 1
    await server.shutdown()
    assert provider.live() == []


async def test_an_episode_seed_on_the_same_server_is_still_exported(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    server = make_server(provider, config, server_type=AgentRunSeeds)
    async with _http(server) as http:
        seeded = await _post_seed(http, _seed())
        await _checkpoint(http, config, tmp_path / "ckpt")

    assert seeded.status_code == 200 and not seed_restarts(seeded.headers)
    assert server._checkpointer().keys == ["rs-1/task"] and len(provider.snapshots) == 1
    await server.shutdown()


# --- what a commit checks before it snapshots, and how long it holds each session ------------------------------------


async def test_a_session_past_its_deadline_that_no_request_swept_is_left_out_of_a_commit(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    server.config.sandbox_checkpoint.snapshot_retry_delay_s = 0
    await _seeded(server, ["a", "b"])
    # Past its lifetime the provider ended the sandbox of "a", and no request has run the sweep since.
    server._open_task_sessions["a"] = monotonic() - 1
    provider.boxes[server._session_id_to_sandbox["a"].sandbox_id]["state"] = "stopped"

    states = await server.export_session_states(["a", "b"], checkpoint_id="c1")

    # Snapshotting "a" would fail the commit for "b" too.
    assert sorted(states) == ["b"] and _snapshotted(provider) == [server._session_id_to_sandbox["b"].sandbox_id]
    assert not server._task_sessions["a"].expired, "the sweep still closes the session when a request arrives"
    await server.shutdown()


async def test_a_session_still_being_graded_past_its_deadline_is_checkpointed_with_its_capture(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    grading, finish = asyncio.Event(), asyncio.Event()
    server = make_server(provider, config, server_type=GradesAfterCapture, grading=grading, finish=finish)
    async with _http(server) as http:
        seeded = await _post_seed(http, _seed())
        verify = asyncio.create_task(http.post("/ng-rollout/r/verify", json=_verify_body(), cookies=seeded.cookies))
        async with asyncio.timeout(10):
            await grading.wait()
        # The grading outlasts the lifetime of the task sandbox, which verify already stopped.
        server._open_task_sessions["rs-1"] = monotonic() - 1
        await _checkpoint(http, config, tmp_path / "ckpt")
        expired = server._task_sessions["rs-1"].expired
        finish.set()
        graded = await verify

    restored = make_server(provider, config)
    async with _http(restored) as http:
        await _restore(http, config, tmp_path / "ckpt")
        again = await restored.capture_task_sandbox_patch(
            "rs-1", AsyncMock(side_effect=AssertionError("captured again"))
        )

    assert not expired, "the sweep leaves a session alone while its verify runs"
    assert graded.status_code == 200 and graded.json()["reward"] == 1.0, graded.text
    assert again == "hello", "a verify replayed after the restore grades the capture the checkpoint saved"
    assert _snapshotted(provider) == [] and _running_forks(provider) == []


async def test_a_session_past_its_deadline_whose_verify_claimed_the_task_sandbox_is_left_out_of_a_commit(
    tmp_path: Path,
) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    await _seeded(server, ["a", "b"])
    # A verify claims the sandbox of "a" and takes it out of the registry until it releases it. It still runs,
    # and the checkpointer still tracks it.
    server._claim_task_sandbox("a")
    sandbox = server._session_id_to_sandbox.pop("a")
    server._open_task_sessions["a"] = monotonic() - 1

    states = await server.export_session_states(["a", "b"], checkpoint_id="c1")

    # Past the deadline the provider may have ended it, and a snapshot that fails would fail the commit for "b" too.
    assert sorted(states) == ["b"] and _snapshotted(provider) == [server._session_id_to_sandbox["b"].sandbox_id]
    await server._release_task_sandbox("a", sandbox)
    await server.shutdown()


async def test_task_state_a_checkpoint_cannot_store_fails_the_export_before_any_snapshot(tmp_path: Path) -> None:
    class UnstorableTaskState(NotesServer):
        def _export_task_sandbox_state(self, session_id: str) -> dict[str, JsonValue]:
            return {"note": object()}

    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"), server_type=UnstorableTaskState)

    with pytest.raises(TypeError, match="exported task state a checkpoint cannot store"):
        await _exported(server, ["a", "b"])

    # Refused before any sandbox was snapshotted: every later commit would fail the same way.
    assert _snapshotted(provider) == [] and provider.snapshots == {}
    assert server._export_locks._locks == {}, "the refusal released every session"
    await server.shutdown()


async def test_a_seed_with_another_model_than_the_declared_task_model_is_refused(tmp_path: Path) -> None:
    class _OtherTask(BaseModel):
        note: str

    provider = FakeSnapshotProvider()
    server = _started(make_server(provider, checkpointing_config(tmp_path / "state")))

    # A restore would read the saved task data with task_model, not with the model this seed used.
    with pytest.raises(TypeError, match="seeds sessions with _OtherTask but declares task_model = _Task"):
        await server.seed_task_sandbox_session(_request("rs-1"), _seed(), _OtherTask)

    assert provider.boxes == {} and server._task_sessions == {}


async def test_a_session_is_released_once_its_own_snapshot_is_done(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = await _seeded(make_server(provider, checkpointing_config(tmp_path / "state")), ["a", "b"])
    sandbox_ids = {session_id: sandbox.sandbox_id for session_id, sandbox in server._session_id_to_sandbox.items()}
    snapshotting, finish = asyncio.Event(), asyncio.Event()
    snapshot = provider.snapshot

    async def slow_snapshot(handle: Any, **kwargs: Any) -> str:
        if handle.sandbox_id == sandbox_ids["b"]:
            snapshotting.set()
            await finish.wait()
        return await snapshot(handle, **kwargs)

    provider.snapshot = slow_snapshot
    export = asyncio.create_task(server.export_session_states(["a", "b"], checkpoint_id="c1"))
    async with asyncio.timeout(10):
        await snapshotting.wait()
        # The snapshot of "a" is done while that of "b" still runs: a verify of "a" captures and stops its sandbox.
        captured = await server.capture_task_sandbox_patch("a", lambda: server._read_note("a"))
        await server.stop_task_sandbox_after_capture("a")
    released_early = not export.done()
    # "b" is still being snapshotted, so a stop of its sandbox waits.
    stop = asyncio.create_task(server.stop_task_sandbox_after_capture("b"))
    await asyncio.sleep(0.05)
    held = not stop.done() and sandbox_ids["b"] in provider.live()
    finish.set()
    states = await export
    await stop

    assert released_early and captured == "hello", "the capture of 'a' did not wait for the snapshot of 'b'"
    assert held, "the stop of 'b' waits for its own snapshot"
    # What "a" exports was read before its lock was released: the capture and the stop after it changed nothing.
    assert states["a"]["sandboxes"]["task"]["snapshot_id"] in provider.snapshots
    assert (states["a"]["capture_ready"], states["a"]["task_state"]) == (False, {"note": "hello"})
    assert states["b"]["sandboxes"]["task"]["snapshot_id"] in provider.snapshots
    assert provider.live() == [] and server._export_locks._locks == {}


async def test_a_snapshot_that_keeps_failing_fails_the_commit_and_releases_every_session(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    server.config.sandbox_checkpoint.snapshot_retry_delay_s = 0
    await _seeded(server, ["a", "b"])
    sandbox_ids = {session_id: sandbox.sandbox_id for session_id, sandbox in server._session_id_to_sandbox.items()}
    snapshot = provider.snapshot

    async def failing_snapshot(handle: Any, **kwargs: Any) -> str:
        if handle.sandbox_id == sandbox_ids["b"]:
            raise RuntimeError("snapshot push failed")
        return await snapshot(handle, **kwargs)

    provider.snapshot = failing_snapshot
    with pytest.raises(SandboxCheckpointError):
        await server.export_session_states(["a", "b"], checkpoint_id="c1")

    # The snapshot of "a" was taken and stays for a release to delete; nothing is left locked.
    assert [taken["sandbox_id"] for taken in provider.snapshots.values()] == [sandbox_ids["a"]]
    assert server._export_locks._locks == {}
    await server.shutdown()


async def test_a_session_that_expires_while_a_commit_waits_for_another_session_is_not_exported(
    tmp_path: Path,
) -> None:
    provider = FakeSnapshotProvider()
    server = await _seeded(make_server(provider, checkpointing_config(tmp_path / "state")), ["a", "b"])
    sandbox_ids = {session_id: sandbox.sandbox_id for session_id, sandbox in server._session_id_to_sandbox.items()}

    # The commit chose "a" and holds its lock, then waits for the lock of "b", which a stop or a capture holds.
    async with server._export_locks.locked("b"):
        export = asyncio.create_task(server.export_session_states(["a", "b"], checkpoint_id="c1"))
        await asyncio.sleep(0.05)
        # Meanwhile "a" passes its deadline and a request sweeps it: the sweep marks it, then waits to stop it.
        server._open_task_sessions["a"] = 0.0
        sweep = asyncio.create_task(server._close_abandoned_task_sessions(monotonic()))
        await asyncio.sleep(0.05)
        waited = not export.done() and not sweep.done() and server._task_sessions["a"].expired
    states = await export
    await sweep

    assert waited
    # Checked again once every lock is held: "a" expired, and its sandbox is being stopped, not snapshotted.
    assert sorted(states) == ["b"] and _snapshotted(provider) == [sandbox_ids["b"]]
    assert provider.live() == [sandbox_ids["b"]]
    await server.shutdown()


# --- a restore's install ---------------------------------------------------------------------------------------------


async def test_a_restored_session_s_seed_reply_is_built_with_its_task_state_in_place(tmp_path: Path) -> None:
    class ReplyReadsTaskState(NotesServer):
        async def _task_sandbox_response(self, session_id: str, record: Any) -> ResourcesSeedSessionResponse:
            # As a server whose seed reply depends on what its setup recorded for the session.
            assert self._notes[session_id] == "hello"
            return await super()._task_sandbox_response(session_id, record)

    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    states = await _exported(make_server(provider, config, server_type=ReplyReadsTaskState), ["rs-1"])
    restored = make_server(provider, config, server_type=ReplyReadsTaskState)

    await restored.restore_session_states(states)

    assert restored._task_sessions["rs-1"].response.sandbox_access is not None
    await restored.shutdown()


async def test_a_restore_cancelled_through_its_request_s_scope_still_stops_the_forks(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    states = await _exported(make_server(provider, config), ["a", "b"])
    restored = make_server(provider, config)
    response = restored._task_sandbox_response
    installing = asyncio.Event()

    async def hanging(session_id: str, record: Any) -> ResourcesSeedSessionResponse:
        if session_id == "b":
            installing.set()
            await asyncio.Event().wait()
        return await response(session_id, record)

    object.__setattr__(restored, "_task_sandbox_response", hanging)

    # A client that disconnects cancels the request through its scope. Unlike one task.cancel(), that cancels every
    # await until the scope is left, the rollback's stops included unless they are shielded.
    with anyio.CancelScope() as scope:

        async def disconnect() -> None:
            await installing.wait()
            scope.cancel()

        disconnecting = asyncio.create_task(disconnect())
        await restored.restore_session_states(states)
    await disconnecting

    assert scope.cancelled_caught
    assert _running_forks(provider) == []
    assert restored._task_sessions == {} and restored._session_id_to_sandbox == {} and restored._notes == {}
    assert restored._open_task_sessions == {} and restored._checkpointer().keys == []


async def test_a_restored_session_nobody_seeds_again_is_closed_at_its_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    states = await _exported(make_server(provider, config), ["rs-1"])
    restored = make_server(provider, config)
    await restored.restore_session_states(states)
    assert len(_running_forks(provider)) == 1

    # The replacement attempt never comes. Past the session's lifetime, the next request closes the session.
    later = monotonic() + restored._session_lifetime_s() + 1
    monkeypatch.setattr("resources_servers.swebench.sandbox_sessions.monotonic", lambda: later)
    await restored.seed_task_sandbox_session(_request("other"), _seed(session_id="other"), _Task)

    assert _running_forks(provider) == [] and restored._task_sessions["rs-1"].expired
    assert restored._checkpointer().keys == ["other/task"]
    await restored.shutdown()
