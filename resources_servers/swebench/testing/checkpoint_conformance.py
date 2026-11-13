# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conformance checks for ``CheckpointedSandboxSessionResourcesServer`` subclasses. Import them from tests.

:func:`check_sandbox_session_checkpoints` drives a subclass through its real routes, checkpoint control routes and
middleware, in process, against :class:`FakeSnapshotProvider`, a provider with durable snapshots that behaves like
OpenSandbox's: a snapshot is an independent copy that lives until deleted, and ``create_from_snapshot`` forks it
under a new id. A server's own tests call it with a factory that builds the server on the provider it is given, and
one seed request the server accepts.
"""

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxSnapshotInfo,
    SandboxSnapshotNotFoundError,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.server_runtime import ResourcesSeedSessionRequest
from nemo_gym.server_runtime._checkpoint import sandbox as sandbox_checkpoints
from nemo_gym.server_runtime._checkpoint.sandbox import SandboxCheckpointError


TOKEN = "conformance-token"
_AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakeSandboxProvider:
    """An in-process provider: each sandbox records the commands it ran and the files uploaded to it.

    ``exec_results`` answers a command; the default answers ``pwd`` with ``/testbed`` and everything else with an
    empty success. A stopped sandbox refuses every call, as a deleted one does.
    """

    name = "fake"

    def __init__(self, exec_results: Optional[Callable[[str], Optional[SandboxExecResult]]] = None) -> None:
        self.boxes: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self._exec_results = exec_results
        self._counter = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    def _box(self, sandbox_id: str) -> dict[str, Any]:
        box = self.boxes.get(sandbox_id)
        if box is None or box["state"] == "stopped":
            raise RuntimeError(f"sandbox {sandbox_id} is gone")
        return box

    def live(self) -> list[str]:
        """The sandboxes that are still running."""
        return sorted(sandbox_id for sandbox_id, box in self.boxes.items() if box["state"] == "running")

    def created_from_image(self) -> int:
        return sum(1 for box in self.boxes.values() if box["from_snapshot"] is None)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        if spec.image is None:
            raise ValueError("create needs an image; a fork goes through create_from_snapshot")
        return self._add_box([], from_snapshot=None)

    def _add_box(self, commands: list[str], *, from_snapshot: Optional[str]) -> SandboxHandle:
        sandbox_id = self._next_id("sb")
        self.calls.append(("create", sandbox_id))
        self.boxes[sandbox_id] = {"state": "running", "commands": commands, "from_snapshot": from_snapshot}
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        self._box(handle.sandbox_id)["commands"].append(command)
        if self._exec_results is not None:
            result = self._exec_results(command)
            if result is not None:
                return result
        if command.strip() == "pwd":
            return SandboxExecResult(stdout="/testbed\n", stderr=None, return_code=0)
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        self._box(handle.sandbox_id)["commands"].append(f"upload {target_path}")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        self._box(handle.sandbox_id)
        target_path.write_bytes(b"")

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        box = self.boxes.get(handle.sandbox_id)
        return SandboxStatus.RUNNING if box is not None and box["state"] == "running" else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        self.calls.append(("close", handle.sandbox_id))
        if handle.sandbox_id in self.boxes:
            self.boxes[handle.sandbox_id]["state"] = "stopped"

    async def aclose(self) -> None:
        pass

    async def serialize_handle(self, handle: SandboxHandle, *, scope: Optional[str] = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        sandbox_id = str(descriptor["sandbox_id"])
        self._box(sandbox_id)
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)


class FakeSnapshotProvider(FakeSandboxProvider):
    """A :class:`FakeSandboxProvider` with durable snapshots, listed by name; ``fail_snapshots`` fails the next."""

    name = "fake-snapshots"

    def __init__(self, exec_results: Optional[Callable[[str], Optional[SandboxExecResult]]] = None) -> None:
        super().__init__(exec_results)
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.fail_snapshots = 0

    async def create_from_snapshot(self, snapshot_id: str, spec: SandboxSpec) -> SandboxHandle:
        snapshot = self.snapshots.get(snapshot_id)
        if snapshot is None:
            raise SandboxSnapshotNotFoundError(f"provider {self.name!r} has no snapshot {snapshot_id!r}")
        return self._add_box(list(snapshot["commands"]), from_snapshot=snapshot_id)

    async def snapshot(self, handle: SandboxHandle, *, name: Optional[str] = None) -> str:
        self.calls.append(("snapshot", handle.sandbox_id))
        if self.fail_snapshots:
            self.fail_snapshots -= 1
            raise RuntimeError("snapshot push failed")
        box = self._box(handle.sandbox_id)
        snapshot_id = self._next_id("snap")
        self.snapshots[snapshot_id] = {
            "name": name or snapshot_id,
            "sandbox_id": handle.sandbox_id,
            "commands": list(box["commands"]),
        }
        return snapshot_id

    async def wait_snapshot_ready(self, snapshot_id: str) -> None:
        if snapshot_id not in self.snapshots:
            raise RuntimeError(f"snapshot {snapshot_id} is gone")

    async def delete_snapshot(self, snapshot_id: str) -> None:
        self.calls.append(("delete_snapshot", snapshot_id))
        self.snapshots.pop(snapshot_id, None)

    async def list_snapshots(self, prefix: str) -> list[SandboxSnapshotInfo]:
        return [
            SandboxSnapshotInfo(snapshot_id=snapshot_id, name=snapshot["name"], sandbox_id=snapshot["sandbox_id"])
            for snapshot_id, snapshot in self.snapshots.items()
            if snapshot["name"].startswith(prefix)
        ]


def checkpointing_config(state_dir: Path) -> dict[str, Any]:
    """The global ``checkpoint`` block a conformance server runs with."""
    return {"checkpoint": {"enabled": True, "control_auth_token": TOKEN, "state_dir": str(state_dir)}}


def _control(checkpoint_id: str, **extra: Any) -> dict[str, Any]:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 30, **extra}


async def check_sandbox_session_checkpoints(
    make_server: Callable[[FakeSandboxProvider, dict[str, Any]], Any],
    seed: ResourcesSeedSessionRequest,
    state_dir: Path,
) -> None:
    """Check the checkpoint contract of a server that checkpoints its task sandboxes; raise ``AssertionError``.

    ``make_server(provider, global_config)`` builds the server on ``provider`` (``shared_provider()`` must return
    it) with ``global_config`` as its global configuration. ``seed`` is an episode seed the server accepts for a
    fresh ``resources_session_id``.

    Checked: the episode seed lends the task sandbox and is idempotent, and a conflicting seed is refused; a commit
    snapshots the sandbox and indexes the snapshot; after a crash, a restore forks the snapshot, the replacement
    attempt's seed gets the fork without running the setup again, the patch verify captured survives, and a seed of
    any other attempt is refused; a restore whose fork fails leaves no fork running; a commit after verify stopped
    the sandbox exports the session without one; a release deletes the snapshots of every checkpoint it does not
    keep, including ones a crashed process indexed and ones it never indexed; a close stops the sandbox and fences a
    late seed; shutdown stops what is left; and on a provider that cannot snapshot the server runs
    ``restart_only`` and says so on its seed reply.
    """
    import httpx

    seed_path = f"/ng-rollout/{seed.episode_id.capture_key}/seed_session"
    seed_body = seed.model_dump(mode="json")
    session_id = seed.resources_session_id
    provider = FakeSnapshotProvider()
    config = checkpointing_config(state_dir)

    def client_of(server: Any) -> httpx.AsyncClient:
        app = server.setup_webserver()
        assert server.shared_provider() is provider, "make_server must build the server on the given provider"
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://resources")

    async def control(http: httpx.AsyncClient, operation: str, checkpoint_id: str, **extra: Any) -> dict[str, Any]:
        reply = await http.post(
            f"/ng-control/v1/checkpoint/{operation}", json=_control(checkpoint_id, **extra), headers=_AUTH
        )
        assert reply.status_code == 200, f"{operation} failed: {reply.status_code} {reply.text}"
        return reply.json()

    # -- seed, commit -------------------------------------------------------------------------------------------
    first = make_server(provider, config)
    async with client_of(first) as http:
        seeded = await http.post(seed_path, json=seed_body)
        assert seeded.status_code == 200, f"episode seed failed: {seeded.text}"
        access = seeded.json().get("sandbox_access")
        assert access is not None, "the seed reply lends the session's task sandbox"
        assert seeded.headers.get("x-ng-checkpoint-restart") is None, "a provider that snapshots exports sessions"
        if first.restore_policy == "before_agent_close":
            assert seeded.headers.get("x-ng-checkpoint-restore-policy") == "before_agent_close"
        retried = await http.post(seed_path, json=seed_body)
        assert retried.status_code == 200 and retried.json()["sandbox_access"] == access, "an identical retry"
        # Derived from the seed, so it names another episode whatever attempt the seed has.
        other_attempt = seed.episode_id.attempt + 1
        other = seed.model_copy(update={"episode_id": seed.episode_id.model_copy(update={"attempt": other_attempt})})
        conflicting = await http.post(
            f"/ng-rollout/{other.episode_id.capture_key}/seed_session", json=other.model_dump(mode="json")
        )
        assert conflicting.status_code == 409, "a seed naming the session with another episode is refused"
        created = provider.created_from_image()

        calls: list[str] = []

        async def counted() -> str:
            calls.append("capture")
            return "artifact"

        # Verify's capture, before it stops the sandbox: a commit then saves both.
        assert await first.capture_task_sandbox_patch(session_id, counted) == "artifact"
        assert await first.capture_task_sandbox_patch(session_id, counted) == "artifact"
        assert calls == ["capture"], "an episode session captures its patch once"

        await control(http, "prepare", "c1")
        committed = await control(http, "commit", "c1", checkpoint_dir=str(state_dir / "ckpt-c1"))
        await control(http, "resume", "c1")
        assert committed["episode_ids"] == [seed.episode_id.capture_key]
        indexed = first._checkpointer().indexed_snapshots
        assert list(indexed) == ["c1"] and len(indexed["c1"]) == 1, f"the commit indexes its snapshot: {indexed}"
        assert len(provider.live()) == 1, "a commit leaves the sandbox running"
        crashed_sandbox = provider.live()[0]

    # -- a crash, then a restore into a fork ----------------------------------------------------------------------
    restored = make_server(provider, config)
    async with client_of(restored) as http:
        scope = [seed.episode_id.model_dump(mode="json")]
        await control(http, "restore", "r1", checkpoint_dir=str(state_dir / "ckpt-c1"), episode_ids=scope)
        await control(http, "resume", "r1")
        replacement = seed.model_copy(
            update={"episode_id": seed.episode_id.model_copy(update={"attempt": seed.episode_id.attempt + 1})}
        )
        reseeded = await http.post(
            f"/ng-rollout/{replacement.episode_id.capture_key}/seed_session", json=replacement.model_dump(mode="json")
        )
        assert reseeded.status_code == 200, f"the replacement attempt's seed failed: {reseeded.text}"
        fork = reseeded.json()["sandbox_access"]
        assert fork["connection"]["descriptor"] != access["connection"]["descriptor"], "the seed lends the fork"
        assert provider.created_from_image() == created, (
            "the replacement attempt's seed does not set the task up again"
        )
        assert await restored.capture_task_sandbox_patch(session_id, counted) == "artifact", (
            "the captured patch survives a restore"
        )
        assert calls == ["capture"], "a restored session does not capture its patch again"
        later = seed.model_copy(
            update={"episode_id": seed.episode_id.model_copy(update={"attempt": seed.episode_id.attempt + 2})}
        )
        refused = await http.post(
            f"/ng-rollout/{later.episode_id.capture_key}/seed_session", json=later.model_dump(mode="json")
        )
        assert refused.status_code == 409, "only the attempt a restore continues gets the restored session"
        await restored._checkpointer().settle()
        assert crashed_sandbox not in provider.live(), "the restore stops the sandbox the fork supersedes"

        # Verify stops the sandbox after its capture: the next commit exports the session without a sandbox.
        await restored.stop_task_sandbox_after_capture(session_id)
        await control(http, "prepare", "c2")
        committed = await control(http, "commit", "c2", checkpoint_dir=str(state_dir / "ckpt-c2"))
        await control(http, "resume", "c2")
        assert committed["episode_ids"] == [replacement.episode_id.capture_key], "the session is still exported"
        assert session_id not in restored._session_id_to_sandbox, "verify stopped the sandbox"
        assert "c2" not in restored._checkpointer().indexed_snapshots, "nothing is snapshotted after the stop"

        # Release: a crashed process's indexed snapshot and an unindexed one with this server's prefix go too.
        stray = await provider.snapshot(
            await provider.create(SandboxSpec(image="stray")),
            # Named as this server names a snapshot of a checkpoint it never indexed, which a crash in a commit leaves.
            name=restored._checkpointer().snapshot_name("dropped", "stray", 1),
        )
        await control(http, "release", "c2", kept=[])
        await restored._checkpointer().settle()
        assert provider.snapshots == {}, f"release deletes every snapshot of dropped checkpoints: {provider.snapshots}"
        assert stray not in provider.snapshots
        assert restored._checkpointer().indexed_snapshots == {}, "release empties the index"

        closed = await http.post(
            f"/ng-rollout/{replacement.episode_id.capture_key}/close_session",
            json={"resources_session_id": session_id, "episode_id": replacement.episode_id.model_dump(mode="json")},
        )
        assert closed.status_code == 200, f"close failed: {closed.text}"
        assert session_id not in restored._session_id_to_sandbox
        late = await http.post(
            f"/ng-rollout/{replacement.episode_id.capture_key}/seed_session", json=replacement.model_dump(mode="json")
        )
        assert late.status_code == 409, "a seed after close is refused"

    # -- a restore whose fork fails leaves nothing running --------------------------------------------------------
    rollback_provider = FakeSnapshotProvider()
    rollback = make_server(rollback_provider, checkpointing_config(state_dir / "rollback"))
    good = await rollback_provider.snapshot(await rollback_provider.create(SandboxSpec(image="x")), name="good")

    def session_state(session_id: str, snapshot_id: str) -> dict[str, Any]:
        sandbox = {
            "provider_name": rollback_provider.name,
            "descriptor": {"sandbox_id": "gone"},
            "snapshot_id": snapshot_id,
            "spec": {},
            "snapshot_ready_ts": 0.0,
        }
        session_seed = seed.model_copy(update={"resources_session_id": session_id})
        return {"seed": session_seed.model_dump(mode="json"), "workdir": "/testbed", "sandbox": sandbox}

    states = {"ok": session_state("ok", good), "missing": session_state("missing", "snap-missing")}
    live_before = set(rollback_provider.live())
    # A fork retries a snapshot the provider does not have yet; without the waits here, which test nothing.
    backoff_s = sandbox_checkpoints.FORK_NOT_FOUND_BACKOFF_S
    sandbox_checkpoints.FORK_NOT_FOUND_BACKOFF_S = 0.0
    try:
        await rollback.restore_session_states(states)
    except SandboxCheckpointError as error:
        assert "snap-missing" in str(error), f"the restore fails on the snapshot that is missing: {error}"
    else:
        raise AssertionError("a restore whose fork fails must raise")
    finally:
        sandbox_checkpoints.FORK_NOT_FOUND_BACKOFF_S = backoff_s
    forked = [sandbox_id for sandbox_id, box in rollback_provider.boxes.items() if box["from_snapshot"] == good]
    assert forked, "the restore forked the snapshot it has before the other fork failed"
    assert set(rollback_provider.live()) == live_before, "a failed restore stops the forks it made"
    assert "ok" not in rollback._session_id_to_sandbox, "a failed restore installs nothing"

    # -- a close stops a task sandbox that is still running -------------------------------------------------------
    closing_provider = FakeSnapshotProvider()
    provider = closing_provider  # client_of checks the server is built on this provider
    closing = make_server(closing_provider, checkpointing_config(state_dir / "closing"))
    async with client_of(closing) as http:
        seeded = await http.post(seed_path, json=seed_body)
        assert seeded.status_code == 200, f"episode seed failed: {seeded.text}"
        started = set(closing_provider.live())
        assert started, "the seed starts the task sandbox"
        closed = await http.post(
            f"/ng-rollout/{seed.episode_id.capture_key}/close_session",
            json={"resources_session_id": session_id, "episode_id": seed.episode_id.model_dump(mode="json")},
        )
        assert closed.status_code == 200, f"close failed: {closed.text}"
        assert not started & set(closing_provider.live()), "a close stops the session's running task sandbox"
    await closing.shutdown()

    # -- shutdown, and a provider that cannot snapshot ------------------------------------------------------------
    plain = FakeSandboxProvider()
    provider = plain  # client_of checks the server is built on this provider
    restart_only = make_server(plain, checkpointing_config(state_dir / "plain"))
    app_client = client_of(restart_only)
    async with app_client as http:
        status = await http.get("/ng-control/v1/checkpoint/status", headers=_AUTH)
        assert status.json()["mode"] == "restart_only", "a provider that cannot snapshot makes the server restart_only"
        seeded = await http.post(seed_path, json=seed_body)
        assert seeded.status_code == 200
        assert seeded.headers.get("x-ng-checkpoint-restart") == "1", "its seed reply makes the episode a restart"
    await restart_only.shutdown()
    assert plain.live() == [], "shutdown stops every live sandbox"
