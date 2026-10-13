# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SandboxSessionCheckpointer against a fake provider whose snapshots are independent filesystem copies."""

import asyncio
import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.checkpoint import (
    FORK_NOT_FOUND_RETRIES,
    SandboxCheckpointError,
    SandboxCheckpointState,
    SandboxSessionCheckpointer,
    spec_from_json,
    spec_to_json,
)
from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)


class FakeSnapshotProvider:
    """OpenSandbox-shaped fake: an explicit snapshot copies the filesystem, create can fork a snapshot, connect is
    by id. Snapshots live until deleted, whatever the source sandbox does."""

    name = "fake-snapshots"

    def __init__(self) -> None:
        self.boxes: dict[str, dict[str, Any]] = {}
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self.fail: dict[tuple[str, str], Exception] = {}
        self.snapshot_gate: asyncio.Event | None = None
        self.in_flight_snapshots = 0
        self.max_in_flight_snapshots = 0
        self._counter = 0

    def _box(self, handle: SandboxHandle) -> dict[str, Any]:
        box = self.boxes.get(handle.sandbox_id)
        if box is None or box["state"] == "stopped":
            raise RuntimeError(f"sandbox {handle.sandbox_id} is gone")
        return box

    def _maybe_fail(self, op: str, sandbox_id: str) -> None:
        self.calls.append((op, sandbox_id))
        error = self.fail.pop((op, sandbox_id), None)
        if error is not None:
            raise error

    def files(self, sandbox_id: str) -> list[str]:
        return list(self.boxes[sandbox_id]["fs"])

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self._counter += 1
        sandbox_id = f"sb-{self._counter}"
        snapshot_id = spec.provider_options.get("snapshot_id")
        self._maybe_fail("create", sandbox_id)
        if (spec.image is None) == (snapshot_id is None):
            raise ValueError("exactly one of image or snapshot_id must be specified")  # as OpenSandbox does
        if snapshot_id is not None and snapshot_id not in self.snapshots:
            raise RuntimeError(f"SNAPSHOT::NOT_FOUND {snapshot_id}")
        fs = list(self.snapshots[snapshot_id]["fs"]) if snapshot_id is not None else []
        self.boxes[sandbox_id] = {"fs": fs, "state": "running", "spec": spec}
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        box = self._box(handle)
        box["fs"].append(command)
        return SandboxExecResult(stdout="ok", stderr=None, return_code=0)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        pass

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        pass

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        box = self.boxes.get(handle.sandbox_id)
        if box is None or box["state"] == "stopped":
            return SandboxStatus.STOPPED
        return SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        self._maybe_fail("close", handle.sandbox_id)
        if handle.sandbox_id in self.boxes:
            self.boxes[handle.sandbox_id]["state"] = "stopped"

    async def aclose(self) -> None:
        pass

    async def snapshot(self, handle: SandboxHandle, *, name: str | None = None) -> str:
        self._maybe_fail("snapshot", handle.sandbox_id)
        box = self._box(handle)
        self._counter += 1
        snapshot_id = f"snap-{self._counter}"
        # The snapshot exists on the backend from here on, whatever happens to the wait for it.
        self.snapshots[snapshot_id] = {"sandbox_id": handle.sandbox_id, "fs": list(box["fs"]), "name": name}
        return snapshot_id

    async def wait_snapshot_ready(self, snapshot_id: str) -> None:
        self.in_flight_snapshots += 1
        self.max_in_flight_snapshots = max(self.max_in_flight_snapshots, self.in_flight_snapshots)
        try:
            if self.snapshot_gate is not None:
                await self.snapshot_gate.wait()
        finally:
            self.in_flight_snapshots -= 1
        self._maybe_fail("wait_snapshot_ready", snapshot_id)

    async def delete_snapshot(self, snapshot_id: str) -> None:
        self._maybe_fail("delete_snapshot", snapshot_id)
        self.snapshots.pop(snapshot_id, None)  # 404 is not an error

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        sandbox_id = str(descriptor["sandbox_id"])
        self._maybe_fail("connect", sandbox_id)
        box = self.boxes.get(sandbox_id)
        if box is None or box["state"] == "stopped":
            raise RuntimeError(f"sandbox {sandbox_id} is gone")
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)


class NoSnapshotProvider(FakeSnapshotProvider):
    """A backend without snapshots: its sandboxes cannot be checkpointed."""

    name = "fake-plain"

    snapshot = None  # type: ignore[assignment]
    wait_snapshot_ready = None  # type: ignore[assignment]
    delete_snapshot = None  # type: ignore[assignment]


SPEC = SandboxSpec(image="img:1", workdir="/work", env={"A": "1"}, resources={"cpu": 2}, ports=[8080])
# What a checkpoint record keeps of SPEC: the snapshot stands in for the image.
RESTORED_SPEC = dataclasses.replace(SPEC, image=None)


async def seeded(provider: FakeSnapshotProvider, *sessions: str, **kwargs: Any) -> SandboxSessionCheckpointer:
    checkpointer = SandboxSessionCheckpointer(provider, **kwargs)
    for session_id in sessions:
        sandbox = await checkpointer.create(session_id, SPEC)
        await sandbox.exec(f"{session_id}: one")
    return checkpointer


def ops(provider: FakeSnapshotProvider, op: str) -> list[str]:
    return [sandbox_id for name, sandbox_id in provider.calls if name == op]


def state_of(exported: dict[str, Any], session_id: str) -> SandboxCheckpointState:
    return SandboxCheckpointState.model_validate(exported[session_id])


# -- export -------------------------------------------------------------------------------------------------


async def test_export_snapshots_each_sandbox_and_leaves_it_running() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1", "s2")

    states = await checkpointer.export(["s1", "s2"])

    assert set(states) == {"s1", "s2"}
    for session_id in states:
        state = state_of(states, session_id)
        sandbox_id = state.descriptor["sandbox_id"]
        assert provider.boxes[sandbox_id]["state"] == "running"
        assert provider.snapshots[state.snapshot_id]["sandbox_id"] == sandbox_id
        assert provider.snapshots[state.snapshot_id]["fs"] == [f"{session_id}: one"]
        assert provider.snapshots[state.snapshot_id]["name"].startswith(f"ng-{session_id}-")
        assert state.provider_name == provider.name
        assert state.snapshot_ids == [state.snapshot_id]
        assert spec_from_json(state.spec) == RESTORED_SPEC
    # The episode continues in the same sandbox, and the snapshot does not follow it.
    await checkpointer.get("s1").exec("s1: two")
    assert provider.files("sb-1") == ["s1: one", "s1: two"]
    assert provider.snapshots[state_of(states, "s1").snapshot_id]["fs"] == ["s1: one"]


def test_spec_round_trips_every_field_a_fork_reads() -> None:
    spec = SandboxSpec(
        image="img:1",
        ttl_s=120,
        ready_timeout_s=30.5,
        workdir="/work",
        env={"A": "1", "TOKEN": "secret"},
        files={"/a": "contents"},
        metadata={"nemo.nvidia.com/shm": "1Gi"},
        resources=SandboxResources(cpu=2, memory_mib=512, disk_gib=10, gpu=1, gpu_type="H100"),
        entrypoint=["/bin/sh", "-c", "sleep inf"],
        provider_options={
            "network_policy": {"egress": "none"},
            "image_auth": {"username": "u", "password": "p"},
            "snapshot_id": "stale",
        },
        ports=[8080, 9090],
    )
    data = spec_to_json(spec)
    # Dropped: the snapshot has the files and replaces the image, so its registry credentials go too.
    assert data["files"] == {} and data["image"] is None
    assert data["provider_options"] == {"network_policy": {"egress": "none"}}
    assert data["ports"] == [8080, 9090]
    assert data["resources"] == {"cpu": 2, "memory_mib": 512, "disk_gib": 10, "gpu": 1, "gpu_type": "H100"}
    # The rest, the environment included, is exactly what the fork is created with.
    restored = spec_from_json(json.loads(json.dumps(data)))
    assert restored == dataclasses.replace(
        spec, image=None, files={}, provider_options={"network_policy": {"egress": "none"}}
    )


async def test_export_leaves_out_sessions_it_no_longer_holds() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    assert set(await checkpointer.export(["s1", "gone"])) == {"s1"}
    assert ops(provider, "snapshot") == ["sb-1"]


async def test_export_bounds_concurrent_snapshots() -> None:
    provider = FakeSnapshotProvider()
    provider.snapshot_gate = asyncio.Event()
    checkpointer = await seeded(provider, "s1", "s2", "s3", "s4", "s5", parallelism=2)

    export = asyncio.create_task(checkpointer.export(["s1", "s2", "s3", "s4", "s5"]))
    for _ in range(50):
        await asyncio.sleep(0)
    assert provider.in_flight_snapshots == 2
    provider.snapshot_gate.set()
    states = await export

    assert len(states) == 5 and provider.max_in_flight_snapshots == 2


async def test_a_failed_snapshot_fails_the_export_and_leaves_every_sandbox_running() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1", "s2")
    provider.fail[("snapshot", "sb-2")] = RuntimeError("registry push failed")

    with pytest.raises(SandboxCheckpointError, match="'s2'.*could not be snapshotted"):
        await checkpointer.export(["s1", "s2"])

    assert provider.boxes["sb-1"]["state"] == "running" and provider.boxes["sb-2"]["state"] == "running"
    # The episodes continue untouched. The snapshot already taken is recorded on its session, so it goes with
    # the episode: marked at the end, deleted by the next successful export.
    await checkpointer.get("s1").exec("s1: two")
    await checkpointer.get("s2").exec("s2: two")
    assert provider.files("sb-1") == ["s1: one", "s1: two"] and provider.files("sb-2") == ["s2: one", "s2: two"]
    assert len(provider.snapshots) == 1
    await checkpointer.stop("s1", forget_snapshots=True)
    await checkpointer.export(["s2"])
    assert {s["sandbox_id"] for s in provider.snapshots.values()} == {"sb-2"}


async def test_export_waits_for_every_sibling_before_it_fails() -> None:
    provider = FakeSnapshotProvider()
    provider.snapshot_gate = asyncio.Event()
    checkpointer = await seeded(provider, "s1", "s2")
    provider.fail[("snapshot", "sb-2")] = RuntimeError("registry push failed")

    export = asyncio.create_task(checkpointer.export(["s1", "s2"]))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not export.done() and provider.in_flight_snapshots == 1, "s2 failed at once; s1 is still being taken"
    provider.snapshot_gate.set()
    with pytest.raises(SandboxCheckpointError, match="'s2'.*could not be snapshotted"):
        await export

    assert provider.in_flight_snapshots == 0
    [snapshot] = provider.snapshots.values()
    assert snapshot["sandbox_id"] == "sb-1"


async def test_a_snapshot_that_does_not_become_ready_fails_the_export_and_stays_recorded() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    provider.fail[("wait_snapshot_ready", "snap-2")] = TimeoutError("not ready after 240s")

    with pytest.raises(SandboxCheckpointError, match="'s1'.*snapshot 'snap-2' did not become ready"):
        await checkpointer.export(["s1"])

    # The backend has the snapshot whatever the wait said; the episode's end still finds it.
    assert "snap-2" in provider.snapshots
    await checkpointer.stop("s1", forget_snapshots=True)
    assert checkpointer.pending_snapshot_deletions == ["snap-2"]


async def test_an_export_cancelled_mid_wait_still_records_the_snapshot() -> None:
    """A commit that misses its deadline is cancelled by the controller's resume; the snapshot exists anyway."""
    provider = FakeSnapshotProvider()
    provider.snapshot_gate = asyncio.Event()
    checkpointer = await seeded(provider, "s1")

    export = asyncio.create_task(checkpointer.export(["s1"]))
    for _ in range(20):
        await asyncio.sleep(0)
    export.cancel()
    with pytest.raises(asyncio.CancelledError):
        await export

    assert "snap-2" in provider.snapshots
    await checkpointer.stop("s1", forget_snapshots=True)
    await checkpointer.export([])
    assert provider.snapshots == {}, "deleted with the episode, not orphaned"


async def test_export_leaves_out_a_tracked_sandbox_that_was_never_started() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    unstarted = AsyncSandbox(provider, SPEC, owns_provider=False)
    checkpointer.add("s2", unstarted, SPEC)

    assert set(await checkpointer.export(["s1", "s2"])) == {"s1"}
    assert ops(provider, "snapshot") == ["sb-1"]
    assert "s2" in checkpointer and checkpointer.get("s2") is unstarted


def test_add_rejects_a_duplicate_or_a_sandbox_that_is_not_on_the_shared_provider() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = SandboxSessionCheckpointer(provider)
    checkpointer.add("s1", AsyncSandbox(provider, SPEC, owns_provider=False), SPEC)

    with pytest.raises(ValueError, match="'s1' already has a sandbox"):
        checkpointer.add("s1", AsyncSandbox(provider, SPEC, owns_provider=False), SPEC)
    # A sandbox that owned the provider would close it for every session when stopped.
    with pytest.raises(ValueError, match="'s2' owns its provider"):
        checkpointer.add("s2", AsyncSandbox(provider, SPEC), SPEC)
    with pytest.raises(ValueError, match="'s2' is not on the checkpointer's provider"):
        checkpointer.add("s2", AsyncSandbox(FakeSnapshotProvider(), SPEC, owns_provider=False), SPEC)
    assert checkpointer.session_ids == ["s1"]


async def test_create_rejects_a_session_that_already_has_a_sandbox() -> None:
    checkpointer = await seeded(FakeSnapshotProvider(), "s1")
    with pytest.raises(ValueError, match="'s1' already has a sandbox"):
        await checkpointer.create("s1", SPEC)


async def test_a_backend_without_snapshots_cannot_be_checkpointed() -> None:
    provider = NoSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    with pytest.raises(SandboxCheckpointError, match="does not support snapshots"):
        await checkpointer.export(["s1"])
    assert provider.boxes["sb-1"]["state"] == "running"


# -- restore ------------------------------------------------------------------------------------------------


async def test_restore_validates_every_state_before_touching_the_backend() -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1")).export(["s1"])
    calls_before = list(provider.calls)

    fresh = SandboxSessionCheckpointer(provider)
    with pytest.raises(SandboxCheckpointError, match="'s2'.*invalid checkpoint state"):
        await fresh.restore({"s1": states["s1"], "s2": {"bad": 1}})

    assert provider.calls == calls_before and fresh.session_ids == []


async def test_restore_forks_the_snapshot_and_the_fork_holds_the_checkpoints_files() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    states = await checkpointer.export(["s1"])
    # The live rollout continues after the checkpoint, then the process dies.
    await checkpointer.get("s1").exec("s1: two")
    assert provider.files("sb-1") == ["s1: one", "s1: two"]

    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)
    restored = fresh.get("s1")
    await restored.exec("s1: two (replayed)")

    assert restored.handle.sandbox_id == "sb-3"  # sb-1 is the original, snap-2 its snapshot
    assert provider.files("sb-3") == ["s1: one", "s1: two (replayed)"]
    await fresh.settle()
    assert provider.boxes["sb-1"]["state"] == "stopped", "the superseded sandbox is stopped"
    # The fork inherited the snapshot it came from: when its episode ends, that snapshot goes too.
    await fresh.stop("s1", forget_snapshots=True)
    assert provider.boxes["sb-3"]["state"] == "stopped" and fresh.pending_snapshot_deletions == ["snap-2"]
    await fresh.export([])
    assert provider.snapshots == {}


async def test_a_fork_inherits_every_snapshot_of_the_session_not_only_the_one_it_came_from() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    await checkpointer.export(["s1"])
    states = await checkpointer.export(["s1"])
    assert state_of(states, "s1").snapshot_ids == ["snap-2", "snap-3"]

    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)
    await fresh.stop("s1", forget_snapshots=True)
    await fresh.export([])

    assert provider.snapshots == {}, "the earlier checkpoint's snapshot of the session went too"


async def test_restore_returns_before_the_superseded_sandboxes_are_stopped() -> None:
    """Stopping the crashed process's sandboxes is best effort and must not hold up the deadline-bound install."""
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1")).export(["s1"])
    gate = asyncio.Event()
    original_close = provider.close

    async def slow_close(handle: SandboxHandle) -> None:
        if handle.sandbox_id == "sb-1":
            await gate.wait()
        await original_close(handle)

    provider.close = slow_close  # type: ignore[method-assign]

    fresh = SandboxSessionCheckpointer(provider)
    await asyncio.wait_for(fresh.restore(states), timeout=1)

    assert fresh.session_ids == ["s1"] and provider.boxes["sb-1"]["state"] == "running"
    gate.set()
    await fresh.settle()
    assert provider.boxes["sb-1"]["state"] == "stopped"


async def test_restore_forks_even_when_the_live_sandbox_looks_untouched() -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1")).export(["s1"])

    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)

    # There is no cheap way to prove sb-1 is still at the checkpoint, so the restore never reuses it.
    assert fresh.get("s1").handle.sandbox_id == "sb-3"
    await fresh.settle()
    assert provider.files("sb-3") == ["s1: one"] and provider.boxes["sb-1"]["state"] == "stopped"


async def test_restore_works_when_the_old_sandbox_is_gone() -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1")).export(["s1"])
    del provider.boxes["sb-1"]  # parked at the stop, or its TTL expired

    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)

    assert provider.files(fresh.get("s1").handle.sandbox_id) == ["s1: one"]


async def test_restore_tolerates_a_failure_to_stop_the_old_sandbox() -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1")).export(["s1"])
    provider.fail[("close", "sb-1")] = RuntimeError("backend busy")

    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)
    await fresh.settle()

    assert fresh.session_ids == ["s1"] and provider.boxes["sb-1"]["state"] == "running"


async def test_restore_rolls_back_when_one_sandbox_cannot_be_rebuilt() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1", "s2")
    states = await checkpointer.export(["s1", "s2"])
    # Forks are sb-5 and sb-6 (sb-1, sb-2 are originals; snap-3, snap-4 their snapshots). One of them fails.
    provider.fail[("create", "sb-6")] = RuntimeError("quota")

    fresh = SandboxSessionCheckpointer(provider, parallelism=1)
    with pytest.raises(SandboxCheckpointError, match="could not be re-created"):
        await fresh.restore(states)

    assert fresh.session_ids == []
    assert provider.boxes["sb-5"]["state"] == "stopped"
    # The originals are left alone: only a successful restore stops them.
    assert provider.boxes["sb-1"]["state"] == "running" and provider.boxes["sb-2"]["state"] == "running"


async def test_restore_retries_a_fork_the_backend_cannot_find_yet(monkeypatch: pytest.MonkeyPatch) -> None:
    """A snapshot store that is not fully shared answers "not found" from some replicas; a later try may succeed."""
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1")).export(["s1"])
    monkeypatch.setattr("nemo_gym.sandbox.checkpoint.FORK_NOT_FOUND_BACKOFF_S", 0.0)
    # The first two creates land on a replica without the snapshot (sb-3, sb-4 are the failed attempts' ids).
    provider.fail[("create", "sb-3")] = RuntimeError("Create sandbox failed: HTTP 404: SNAPSHOT::NOT_FOUND")
    provider.fail[("create", "sb-4")] = RuntimeError("Create sandbox failed: HTTP 404: SNAPSHOT::NOT_FOUND")

    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)

    assert fresh.get("s1").handle.sandbox_id == "sb-5" and provider.files("sb-5") == ["s1: one"]
    assert ops(provider, "create").count("sb-3") == 1 and ops(provider, "create").count("sb-4") == 1


async def test_not_found_retries_stop_after_the_configured_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1")).export(["s1"])
    provider.snapshots.clear()
    monkeypatch.setattr("nemo_gym.sandbox.checkpoint.FORK_NOT_FOUND_BACKOFF_S", 0.0)
    creates_before = len(ops(provider, "create"))

    fresh = SandboxSessionCheckpointer(provider)
    with pytest.raises(SandboxCheckpointError, match="NOT_FOUND"):
        await fresh.restore(states)

    assert len(ops(provider, "create")) == creates_before + FORK_NOT_FOUND_RETRIES + 1
    assert fresh.session_ids == [] and provider.boxes["sb-1"]["state"] == "running"


async def test_a_fork_error_other_than_not_found_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1")).export(["s1"])
    monkeypatch.setattr("nemo_gym.sandbox.checkpoint.FORK_NOT_FOUND_BACKOFF_S", 0.0)
    creates_before = len(ops(provider, "create"))
    # The fake pops the failure after one use: a retry would succeed and the count below would notice.
    provider.fail[("create", "sb-3")] = RuntimeError("quota exceeded")

    with pytest.raises(SandboxCheckpointError, match="could not be re-created from snapshot 'snap-2': quota"):
        await SandboxSessionCheckpointer(provider).restore(states)

    assert len(ops(provider, "create")) == creates_before + 1


async def test_the_not_found_backoff_does_not_hold_a_parallelism_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session waiting for its snapshot to appear must not keep another session's fork from starting."""
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1", "s2")).export(["s1", "s2"])
    monkeypatch.setattr("nemo_gym.sandbox.checkpoint.FORK_NOT_FOUND_BACKOFF_S", 0.2)
    # Originals sb-1, sb-2; snapshots snap-3, snap-4. s1's first fork attempt (sb-5) is not found yet.
    provider.fail[("create", "sb-5")] = RuntimeError("HTTP 404: SNAPSHOT::NOT_FOUND")

    fresh = SandboxSessionCheckpointer(provider, parallelism=1)
    restore = asyncio.create_task(fresh.restore(states))
    await asyncio.sleep(0.05)
    assert "sb-6" in provider.boxes, "s2 forked while s1 was sleeping out its backoff"
    await restore

    assert sorted(fresh.session_ids) == ["s1", "s2"]
    assert len(ops(provider, "create")) == 5  # two seeds, s1's failed attempt, s2's fork, s1's retry


async def test_restore_fails_typed_when_the_snapshot_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1")).export(["s1"])
    provider.snapshots.clear()  # reaped by a sweep that should not have, or a per-replica store
    monkeypatch.setattr("nemo_gym.sandbox.checkpoint.FORK_NOT_FOUND_BACKOFF_S", 0.0)

    with pytest.raises(SandboxCheckpointError, match="'s1'.*could not be re-created from snapshot 'snap-2'"):
        await SandboxSessionCheckpointer(provider).restore(states)
    assert provider.boxes["sb-1"]["state"] == "running"


async def test_restore_refuses_a_state_from_another_provider_or_a_session_already_held() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    states = await checkpointer.export(["s1"])

    with pytest.raises(SandboxCheckpointError, match="already has a sandbox"):
        await checkpointer.restore(states)
    foreign = {**states["s1"], "provider_name": "other"}
    with pytest.raises(SandboxCheckpointError, match="created by provider 'other'"):
        await SandboxSessionCheckpointer(provider).restore({"s1": foreign})
    plain = {**states["s1"], "provider_name": NoSnapshotProvider.name}
    with pytest.raises(SandboxCheckpointError, match="does not support snapshots"):
        await SandboxSessionCheckpointer(NoSnapshotProvider()).restore({"s1": plain})


async def test_a_pause_based_record_is_rejected() -> None:
    provider = FakeSnapshotProvider()
    legacy = {
        "provider_name": provider.name,
        "descriptor": {"sandbox_id": "sb-1"},
        "snapshot_id": None,
        "spec": spec_to_json(SPEC),
        "paused_at": 1.0,
        "restore_point": "paused",
    }
    with pytest.raises(SandboxCheckpointError, match="invalid checkpoint state"):
        await SandboxSessionCheckpointer(provider).restore({"s1": legacy})


# -- stop and park --------------------------------------------------------------------------------------------


async def test_stop_frees_the_sandbox_keeps_its_snapshots_and_tolerates_repeats() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    await checkpointer.export(["s1"])

    await checkpointer.stop("s1")
    await checkpointer.stop("s1")
    await checkpointer.stop("never-existed")

    assert provider.boxes["sb-1"]["state"] == "stopped" and "s1" not in checkpointer
    # A retire keeps the snapshots: a retained checkpoint may still restore the attempt. The sweep reclaims them.
    assert len(provider.snapshots) == 1 and ops(provider, "delete_snapshot") == []


async def test_a_finished_episodes_snapshots_are_deleted_by_the_next_export_and_only_those() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1", "s2")
    await checkpointer.export(["s1", "s2"])
    await checkpointer.export(["s1"])
    assert len(provider.snapshots) == 3

    await checkpointer.stop("s1", forget_snapshots=True)

    # Not yet: the controller may still be writing the episode's result, and the latest checkpoint names them.
    assert provider.boxes["sb-1"]["state"] == "stopped" and len(provider.snapshots) == 3
    assert sorted(checkpointer.pending_snapshot_deletions) == ["snap-3", "snap-5"]
    await checkpointer.export(["s2"])
    # The new checkpoint names only s2's snapshots, so s1's are gone and s2's (two by now) are kept.
    assert {s["sandbox_id"] for s in provider.snapshots.values()} == {"sb-2"} and len(provider.snapshots) == 2
    assert sorted(ops(provider, "delete_snapshot")) == ["snap-3", "snap-5"]
    assert checkpointer.pending_snapshot_deletions == []


async def test_a_failed_export_keeps_the_pending_deletions_for_the_next_one() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1", "s2")
    await checkpointer.export(["s1", "s2"])
    await checkpointer.stop("s1", forget_snapshots=True)
    provider.fail[("snapshot", "sb-2")] = RuntimeError("registry push failed")

    with pytest.raises(SandboxCheckpointError):
        await checkpointer.export(["s2"])
    assert checkpointer.pending_snapshot_deletions == ["snap-3"], "the previous checkpoint still names it"

    await checkpointer.export(["s2"])
    assert "snap-3" not in provider.snapshots


async def test_a_failed_snapshot_delete_does_not_fail_the_export() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1", "s2")
    states = await checkpointer.export(["s1", "s2"])
    provider.fail[("delete_snapshot", state_of(states, "s1").snapshot_id)] = RuntimeError("registry down")
    await checkpointer.stop("s1", forget_snapshots=True)

    exported = await checkpointer.export(["s2"])

    assert set(exported) == {"s2"} and "s1" not in checkpointer
    assert state_of(states, "s1").snapshot_id in provider.snapshots, "left for the sweep"
    assert checkpointer.pending_snapshot_deletions == []


async def test_a_failed_stop_keeps_the_session_for_a_retry() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    provider.fail[("close", "sb-1")] = RuntimeError("backend down")

    with pytest.raises(RuntimeError, match="backend down"):
        await checkpointer.stop("s1")
    assert "s1" in checkpointer

    await checkpointer.stop("s1")
    assert provider.boxes["sb-1"]["state"] == "stopped" and "s1" not in checkpointer


async def test_a_failed_stop_keeps_the_snapshots_with_the_session_until_it_succeeds() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    await checkpointer.export(["s1"])
    provider.fail[("close", "sb-1")] = RuntimeError("backend down")

    with pytest.raises(RuntimeError, match="backend down"):
        await checkpointer.stop("s1", forget_snapshots=True)
    assert checkpointer.pending_snapshot_deletions == []

    await checkpointer.stop("s1", forget_snapshots=True)
    assert checkpointer.pending_snapshot_deletions == ["snap-2"]


async def test_park_kills_the_exported_sandboxes_and_tolerates_failures() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1", "s2", "s3")
    states = await checkpointer.export(["s1", "s2"])
    provider.fail[("close", "sb-1")] = RuntimeError("backend busy")

    await checkpointer.park(["s1", "s2", "gone"])

    # s2 is stopped, s1's failure is logged, s3 was not exported and is left alone; the checkpoint restores either.
    assert provider.boxes["sb-1"]["state"] == "running" and provider.boxes["sb-2"]["state"] == "stopped"
    assert provider.boxes["sb-3"]["state"] == "running"
    assert checkpointer.session_ids == ["s3"]
    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)
    await fresh.settle()
    assert {provider.files(fresh.get(s).handle.sandbox_id)[0] for s in ("s1", "s2")} == {"s1: one", "s2: one"}


async def test_park_can_leave_the_sandboxes_running() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1", on_stop="none")
    await checkpointer.export(["s1"])

    await checkpointer.park(["s1"])

    assert provider.boxes["sb-1"]["state"] == "running" and checkpointer.session_ids == ["s1"]


def test_on_stop_and_parallelism_are_validated() -> None:
    with pytest.raises(ValueError, match="on_stop"):
        SandboxSessionCheckpointer(FakeSnapshotProvider(), on_stop="pause")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="parallelism"):
        SandboxSessionCheckpointer(FakeSnapshotProvider(), parallelism=0)
    assert SandboxSessionCheckpointer(provider := FakeSnapshotProvider()).provider is provider


class NonConnectableSnapshotProvider(FakeSnapshotProvider):
    """Snapshots, but no connect-by-descriptor: a fresh process cannot reach the superseded sandbox."""

    name = "fake-snapshots-no-connect"

    connect = None  # type: ignore[assignment]
    serialize_handle = None  # type: ignore[assignment]


async def test_a_provider_without_connect_exports_the_id_and_skips_the_superseded_stop() -> None:
    provider = NonConnectableSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    states = await checkpointer.export(["s1"])
    assert state_of(states, "s1").descriptor == {"sandbox_id": "sb-1"}

    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)
    await fresh.settle()

    assert provider.files(fresh.get("s1").handle.sandbox_id) == ["s1: one"]
    assert provider.boxes["sb-1"]["state"] == "running", "nothing can reach it from here; the TTL ends it"
    assert ops(provider, "connect") == [] and ops(provider, "close") == []


# -- borrowed access ------------------------------------------------------------------------------------------


async def test_access_names_the_current_sandbox_which_a_restore_replaces() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    states = await checkpointer.export(["s1"])

    access = await checkpointer.access("s1", provider_config_ref="sandbox", workdir="/work")

    assert access.connection.provider_config_ref == "sandbox" and access.workdir == "/work"
    assert access.connection.descriptor["sandbox_id"] == "sb-1"
    with pytest.raises(KeyError):
        await checkpointer.access("unknown", provider_config_ref="sandbox", workdir="/work")

    # After a restore the access names the fork: a borrower must ask again.
    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)
    await fresh.settle()
    refreshed = await fresh.access("s1", provider_config_ref="sandbox", workdir="/work")
    assert refreshed.connection.descriptor["sandbox_id"] != "sb-1"
    assert refreshed.connection.descriptor["sandbox_id"] == fresh.get("s1").handle.sandbox_id
