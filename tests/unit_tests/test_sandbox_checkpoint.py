# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SandboxSessionCheckpointer against a fake provider whose snapshots are independent filesystem copies."""

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.sandbox.checkpoint import (
    SandboxCheckpointError,
    SandboxCheckpointState,
    SandboxSessionCheckpointer,
    spec_from_json,
    spec_to_json,
)
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus


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
        self.in_flight_snapshots += 1
        self.max_in_flight_snapshots = max(self.max_in_flight_snapshots, self.in_flight_snapshots)
        try:
            if self.snapshot_gate is not None:
                await self.snapshot_gate.wait()
        finally:
            self.in_flight_snapshots -= 1
        self._counter += 1
        snapshot_id = f"snap-{self._counter}"
        self.snapshots[snapshot_id] = {"sandbox_id": handle.sandbox_id, "fs": list(box["fs"]), "name": name}
        return snapshot_id

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
    delete_snapshot = None  # type: ignore[assignment]


SPEC = SandboxSpec(image="img:1", workdir="/work", env={"A": "1"}, resources={"cpu": 2}, ports=[8080])


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
        assert spec_from_json(state.spec) == SPEC
    # The episode continues in the same sandbox, and the snapshot does not follow it.
    await checkpointer.get("s1").exec("s1: two")
    assert provider.files("sb-1") == ["s1: one", "s1: two"]
    assert provider.snapshots[state_of(states, "s1").snapshot_id]["fs"] == ["s1: one"]


async def test_spec_round_trips_without_files() -> None:
    spec = SandboxSpec(image="img", files={"/a": "contents"}, resources={"cpu": 1, "memory_mib": 512}, ports=(1, 2))
    data = spec_to_json(spec)
    assert data["files"] == {} and data["ports"] == [1, 2]
    assert spec_from_json(data) == SandboxSpec(image="img", resources={"cpu": 1, "memory_mib": 512}, ports=(1, 2))


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
    # The episodes continue untouched; the snapshot already taken is an orphan for the sweep.
    await checkpointer.get("s1").exec("s1: two")
    await checkpointer.get("s2").exec("s2: two")
    assert provider.files("sb-1") == ["s1: one", "s1: two"] and provider.files("sb-2") == ["s2: one", "s2: two"]
    assert len(provider.snapshots) == 1


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
    assert provider.boxes["sb-1"]["state"] == "stopped", "the superseded sandbox is stopped"
    # The fork inherited the snapshot it came from: when its episode ends, that snapshot goes too.
    await fresh.stop("s1", forget_snapshots=True)
    assert provider.snapshots == {} and provider.boxes["sb-3"]["state"] == "stopped"


async def test_restore_forks_even_when_the_live_sandbox_looks_untouched() -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1")).export(["s1"])

    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)

    # There is no cheap way to prove sb-1 is still at the checkpoint, so the restore never reuses it.
    assert fresh.get("s1").handle.sandbox_id == "sb-3"
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


async def test_restore_fails_typed_when_the_snapshot_is_gone() -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1")).export(["s1"])
    provider.snapshots.clear()  # reaped by a sweep that should not have, or a per-replica store

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


async def test_stop_for_a_finished_episode_deletes_exactly_its_snapshots() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1", "s2")
    await checkpointer.export(["s1", "s2"])
    await checkpointer.export(["s1"])
    assert len(provider.snapshots) == 3

    await checkpointer.stop("s1", forget_snapshots=True)

    assert provider.boxes["sb-1"]["state"] == "stopped"
    [remaining] = provider.snapshots.values()
    assert remaining["sandbox_id"] == "sb-2"
    assert len(ops(provider, "delete_snapshot")) == 2


async def test_a_failed_snapshot_delete_does_not_fail_the_stop() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    states = await checkpointer.export(["s1"])
    provider.fail[("delete_snapshot", state_of(states, "s1").snapshot_id)] = RuntimeError("registry down")

    await checkpointer.stop("s1", forget_snapshots=True)

    assert "s1" not in checkpointer and provider.boxes["sb-1"]["state"] == "stopped"
    assert len(provider.snapshots) == 1, "left for the sweep"


async def test_a_failed_stop_keeps_the_session_for_a_retry() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    provider.fail[("close", "sb-1")] = RuntimeError("backend down")

    with pytest.raises(RuntimeError, match="backend down"):
        await checkpointer.stop("s1")
    assert "s1" in checkpointer

    await checkpointer.stop("s1")
    assert provider.boxes["sb-1"]["state"] == "stopped" and "s1" not in checkpointer


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
    assert {provider.files(fresh.get(s).handle.sandbox_id)[0] for s in ("s1", "s2")} == {"s1: one", "s2: one"}


async def test_park_can_leave_the_sandboxes_running() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1", on_stop="none")
    await checkpointer.export(["s1"])

    await checkpointer.park(["s1"])

    assert provider.boxes["sb-1"]["state"] == "running" and checkpointer.session_ids == ["s1"]


def test_on_stop_is_validated() -> None:
    with pytest.raises(ValueError, match="on_stop"):
        SandboxSessionCheckpointer(FakeSnapshotProvider(), on_stop="pause")  # type: ignore[arg-type]


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
    refreshed = await fresh.access("s1", provider_config_ref="sandbox", workdir="/work")
    assert refreshed.connection.descriptor["sandbox_id"] != "sb-1"
    assert refreshed.connection.descriptor["sandbox_id"] == fresh.get("s1").handle.sandbox_id
