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
    lost_session_verification,
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


class InPlaceGrader(NotesServer):
    """A server whose grader needs processes the agent started."""

    restore_policy: ClassVar[str] = "before_agent_close"


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
    await check_sandbox_session_checkpoints(make_server, _seed(attempt=attempt), tmp_path / "state")


async def test_a_server_that_restores_only_before_its_agent_closed_conforms(tmp_path: Path) -> None:
    def make(provider: Any, global_config: dict) -> NotesServer:
        return make_server(provider, global_config, server_type=InPlaceGrader)

    await check_sandbox_session_checkpoints(make, _seed(), tmp_path / "state")


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

    # Refused before the sandbox was snapshotted: every later commit would fail the same way.
    assert not [call for call in provider.calls if call[0] == "snapshot"]


async def test_a_server_whose_fork_cannot_continue_after_the_agent_closed_says_so_on_its_seed(tmp_path: Path) -> None:
    server = make_server(FakeSnapshotProvider(), checkpointing_config(tmp_path), server_type=InPlaceGrader)
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


def _close_path(seed: ResourcesSeedSessionRequest) -> tuple[str, dict]:
    body = {"resources_session_id": seed.resources_session_id, "episode_id": seed.episode_id.model_dump(mode="json")}
    return f"/ng-rollout/{seed.episode_id.capture_key}/close_session", body


def _running_forks(provider: FakeSnapshotProvider) -> list[str]:
    return [
        sandbox_id
        for sandbox_id, box in provider.boxes.items()
        if box["from_snapshot"] is not None and box["state"] == "running"
    ]


async def _exported(server: NotesServer, session_ids: list[str]) -> dict[str, JsonValue]:
    for session_id in session_ids:
        body = _seed(session_id=session_id)
        await server.seed_task_sandbox_session(_request(session_id), body, _Task)
    return await server.export_session_states(session_ids, checkpoint_id="c1")


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
        await snapshotting.wait()
        path, body = _close_path(_seed())
        close = asyncio.create_task(http.post(path, json=body))
        await asyncio.sleep(0.05)
        waited = not close.done() and len(provider.live()) == 1
        finish.set()
        states = await export
        closed = await close

    assert waited, "the close does not stop a sandbox a commit is snapshotting"
    assert states["rs-1"]["sandbox"] is not None, "the commit exported the session with its snapshot"
    assert closed.status_code == 200, closed.text
    assert provider.live() == [] and server._checkpointer().keys == []


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
    assert states["rs-1"]["sandbox"]["snapshot_id"] in provider.snapshots


async def test_a_new_process_deletes_the_snapshots_a_crashed_one_took(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider()
    config = checkpointing_config(tmp_path / "state")
    await _exported(make_server(provider, config), ["rs-1"])
    assert len(provider.snapshots) == 1

    # It has created no sandbox: the index under state_dir names what the crashed process took.
    replacement = make_server(provider, config)
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

    assert states["rs-1"]["sandbox"] is None and states["rs-1"]["capture"] == captured == "hello"
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
    # The install failed at "a", so only the checkpointer knows the fork of "b".
    assert len(_running_forks(provider)) == 2 and restored._task_sessions == {}
    assert sorted(restored._session_id_to_sandbox) == ["a"] and restored._checkpointer().keys == ["a", "b"]

    provider.close = close
    await restored.shutdown()
    assert _running_forks(provider) == []


@pytest.mark.parametrize(
    "change, error",
    [
        ({"workdir": None}, "invalid notes session state"),
        ({"unknown": 1}, "invalid notes session state"),
        ({"seed": {"task_data": {}}}, "invalid notes session state"),
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


def test_a_misspelt_restore_policy_fails_at_startup(tmp_path: Path) -> None:
    class Misspelt(NotesServer):
        restore_policy: ClassVar[str] = "before_agent_closed"

    server = make_server(FakeSnapshotProvider(), checkpointing_config(tmp_path), server_type=Misspelt)
    # Sent as it is, the environment server would read it as "any_step" and continue what it must start over.
    with pytest.raises(ValueError, match="restore_policy must be one of"):
        server.setup_webserver()


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
