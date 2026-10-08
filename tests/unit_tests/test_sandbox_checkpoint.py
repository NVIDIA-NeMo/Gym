# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SandboxSessionCheckpointer against a fake pause/resume provider whose snapshots are filesystem copies."""

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
    """OpenSandbox-shaped fake: pause snapshots the filesystem, create can fork a snapshot, connect is by id."""

    name = "fake-snapshots"

    def __init__(self, *, keeps_snapshots: bool = True) -> None:
        self.keeps_snapshots = keeps_snapshots
        self.boxes: dict[str, dict[str, Any]] = {}
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self.fail: dict[tuple[str, str], Exception] = {}
        self.pause_gate: asyncio.Event | None = None
        self.in_flight_pauses = 0
        self.max_in_flight_pauses = 0
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
        fs = list(self.snapshots[snapshot_id]["fs"]) if snapshot_id is not None else []
        self.boxes[sandbox_id] = {"fs": fs, "state": "running", "spec": spec}
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        box = self._box(handle)
        if box["state"] != "running":
            raise RuntimeError(f"sandbox {handle.sandbox_id} is {box['state']}")
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
        return SandboxStatus.PAUSED if box["state"] == "paused" else SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        self._maybe_fail("close", handle.sandbox_id)
        if handle.sandbox_id in self.boxes:
            self.boxes[handle.sandbox_id]["state"] = "stopped"

    async def aclose(self) -> None:
        pass

    async def pause(self, handle: SandboxHandle) -> None:
        self._maybe_fail("pause", handle.sandbox_id)
        box = self._box(handle)
        self.in_flight_pauses += 1
        self.max_in_flight_pauses = max(self.max_in_flight_pauses, self.in_flight_pauses)
        try:
            if self.pause_gate is not None:
                await self.pause_gate.wait()
        finally:
            self.in_flight_pauses -= 1
        box["state"] = "paused"
        if self.keeps_snapshots:
            self._counter += 1
            snapshot_id = f"snap-{self._counter}"
            self.snapshots[snapshot_id] = {
                "sandbox_id": handle.sandbox_id,
                "fs": list(box["fs"]),
                "order": self._counter,
            }

    async def resume(self, handle: SandboxHandle) -> None:
        self._maybe_fail("resume", handle.sandbox_id)
        box = self._box(handle)
        if box["state"] != "paused":
            raise RuntimeError(f"sandbox {handle.sandbox_id} is not paused")
        box["state"] = "running"

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        sandbox_id = str(descriptor["sandbox_id"])
        self._maybe_fail("connect", sandbox_id)
        box = self.boxes.get(sandbox_id)
        if box is None or box["state"] == "stopped":
            raise RuntimeError(f"sandbox {sandbox_id} is gone")
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def latest_snapshot_id(self, handle: SandboxHandle) -> str | None:
        mine = [
            (snap["order"], snapshot_id)
            for snapshot_id, snap in self.snapshots.items()
            if snap["sandbox_id"] == handle.sandbox_id
        ]
        return max(mine)[1] if mine else None


class FrozenOnlyProvider(FakeSnapshotProvider):
    """A backend whose pause only freezes: no snapshot and no lookup capability, like OpenSandbox on Docker."""

    name = "fake-frozen"

    def __init__(self) -> None:
        super().__init__(keeps_snapshots=False)

    latest_snapshot_id = None  # type: ignore[assignment]


SPEC = SandboxSpec(image="img:1", workdir="/work", env={"A": "1"}, resources={"cpu": 2}, ports=[8080])


async def seeded(provider: FakeSnapshotProvider, *sessions: str, **kwargs: Any) -> SandboxSessionCheckpointer:
    checkpointer = SandboxSessionCheckpointer(provider, **kwargs)
    for session_id in sessions:
        sandbox = await checkpointer.create(session_id, SPEC)
        await sandbox.exec(f"{session_id}: one")
    return checkpointer


def ops(provider: FakeSnapshotProvider, op: str) -> list[str]:
    return [sandbox_id for name, sandbox_id in provider.calls if name == op]


# -- export -------------------------------------------------------------------------------------------------


async def test_export_pauses_each_sandbox_and_records_its_snapshot() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1", "s2")

    states = await checkpointer.export(["s1", "s2"])

    assert set(states) == {"s1", "s2"}
    for session_id, raw in states.items():
        state = SandboxCheckpointState.model_validate(raw)
        sandbox_id = state.descriptor["sandbox_id"]
        assert provider.boxes[sandbox_id]["state"] == "paused"
        assert provider.snapshots[state.snapshot_id]["sandbox_id"] == sandbox_id
        assert provider.snapshots[state.snapshot_id]["fs"] == [f"{session_id}: one"]
        assert state.provider_name == provider.name
        assert spec_from_json(state.spec) == SPEC


async def test_spec_round_trips_without_files() -> None:
    spec = SandboxSpec(image="img", files={"/a": "contents"}, resources={"cpu": 1, "memory_mib": 512}, ports=(1, 2))
    data = spec_to_json(spec)
    assert data["files"] == {} and data["ports"] == [1, 2]
    assert spec_from_json(data) == SandboxSpec(image="img", resources={"cpu": 1, "memory_mib": 512}, ports=(1, 2))


async def test_export_leaves_out_sessions_it_no_longer_holds() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    assert set(await checkpointer.export(["s1", "gone"])) == {"s1"}
    assert ops(provider, "pause") == ["sb-1"]


async def test_export_bounds_concurrent_pauses() -> None:
    provider = FakeSnapshotProvider()
    provider.pause_gate = asyncio.Event()
    checkpointer = await seeded(provider, "s1", "s2", "s3", "s4", "s5", parallelism=2)

    export = asyncio.create_task(checkpointer.export(["s1", "s2", "s3", "s4", "s5"]))
    for _ in range(50):
        await asyncio.sleep(0)
    assert provider.in_flight_pauses == 2
    provider.pause_gate.set()
    states = await export

    assert len(states) == 5 and provider.max_in_flight_pauses == 2


async def test_a_failed_pause_fails_the_export_and_leaves_every_sandbox_resumable() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1", "s2")
    provider.fail[("pause", "sb-2")] = RuntimeError("backend busy")

    with pytest.raises(SandboxCheckpointError, match="'s2'.*could not be paused"):
        await checkpointer.export(["s1", "s2"])

    assert provider.boxes["sb-1"]["state"] == "paused" and provider.boxes["sb-2"]["state"] == "running"
    # The next use resumes what was paused, and leaves alone what the failed pause never froze.
    await (await checkpointer.ensure_running("s1")).exec("s1: two")
    await (await checkpointer.ensure_running("s2")).exec("s2: two")
    assert ops(provider, "resume") == ["sb-1"]
    assert provider.files("sb-1") == ["s1: one", "s1: two"] and provider.files("sb-2") == ["s2: one", "s2: two"]


# -- live path ----------------------------------------------------------------------------------------------


async def test_ensure_running_resumes_once_under_concurrent_tool_calls() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    await checkpointer.export(["s1"])

    sandboxes = await asyncio.gather(*(checkpointer.ensure_running("s1") for _ in range(5)))
    await sandboxes[0].exec("s1: two")

    assert ops(provider, "resume") == ["sb-1"]
    assert provider.files("sb-1") == ["s1: one", "s1: two"]
    assert await checkpointer.ensure_running("s1") is sandboxes[0] and ops(provider, "resume") == ["sb-1"]


async def test_ensure_running_is_a_plain_lookup_outside_a_checkpoint() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    assert (await checkpointer.ensure_running("s1")) is checkpointer.get("s1")
    assert ops(provider, "resume") == []
    with pytest.raises(KeyError):
        await checkpointer.ensure_running("unknown")


# -- restore ------------------------------------------------------------------------------------------------


async def test_restore_validates_every_state_before_touching_the_backend() -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1")).export(["s1"])
    calls_before = list(provider.calls)

    fresh = SandboxSessionCheckpointer(provider)
    with pytest.raises(SandboxCheckpointError, match="'s2'.*invalid checkpoint state"):
        await fresh.restore({"s1": states["s1"], "s2": {"bad": 1}})

    assert provider.calls == calls_before and fresh.session_ids == []


async def test_restore_forks_from_the_snapshot_when_the_sandbox_moved_on() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    states = await checkpointer.export(["s1"])
    # The live rollout continues after the checkpoint, then the process dies.
    await (await checkpointer.ensure_running("s1")).exec("s1: two")
    assert provider.files("sb-1") == ["s1: one", "s1: two"]

    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)
    restored = await fresh.ensure_running("s1")
    await restored.exec("s1: two (replayed)")

    assert restored.handle.sandbox_id == "sb-3"  # sb-1 is the original, snap-2 its snapshot
    assert provider.files("sb-3") == ["s1: one", "s1: two (replayed)"]
    assert provider.boxes["sb-1"]["state"] == "stopped"
    assert ops(provider, "resume") == ["sb-1"]  # only the live process's lazy resume; the restore forked


async def test_restore_resumes_a_sandbox_still_paused_at_the_checkpoint() -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1")).export(["s1"])

    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)
    await (await fresh.ensure_running("s1")).exec("s1: two")

    assert ops(provider, "create") == ["sb-1"]
    assert provider.files("sb-1") == ["s1: one", "s1: two"] and provider.boxes["sb-1"]["state"] == "running"


async def test_restore_forks_when_the_sandbox_is_paused_at_a_later_checkpoint() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    first = await checkpointer.export(["s1"])
    await (await checkpointer.ensure_running("s1")).exec("s1: two")
    await checkpointer.export(["s1"])  # a later checkpoint leaves sb-1 paused at a newer snapshot

    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(first)

    restored = fresh.get("s1")
    assert restored.handle.sandbox_id != "sb-1"
    assert provider.files(restored.handle.sandbox_id) == ["s1: one"]
    assert provider.boxes["sb-1"]["state"] == "stopped"


async def test_restore_falls_through_to_the_fork_when_the_old_sandbox_is_unreachable() -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, "s1")).export(["s1"])
    del provider.boxes["sb-1"]  # its TTL expired

    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)

    assert provider.files(fresh.get("s1").handle.sandbox_id) == ["s1: one"]


async def test_restore_rolls_back_when_one_sandbox_cannot_be_rebuilt() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1", "s2")
    states = await checkpointer.export(["s1", "s2"])
    for session_id in ("s1", "s2"):
        await (await checkpointer.ensure_running(session_id)).exec("drift")
    # Forks are sb-5 and sb-6 (sb-1, sb-2 are originals; snap-3, snap-4 their snapshots). One of them fails.
    provider.fail[("create", "sb-6")] = RuntimeError("quota")

    fresh = SandboxSessionCheckpointer(provider, parallelism=1)
    with pytest.raises(SandboxCheckpointError, match="could not be re-created"):
        await fresh.restore(states)

    assert fresh.session_ids == []
    assert provider.boxes["sb-5"]["state"] == "stopped"
    # The originals are left for the retry: only a successful fork stops its predecessor.
    assert provider.boxes["sb-2"]["state"] == "running"


async def test_restore_refuses_a_state_from_another_provider_or_a_session_already_held() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    states = await checkpointer.export(["s1"])

    with pytest.raises(SandboxCheckpointError, match="already has a sandbox"):
        await checkpointer.restore(states)
    foreign = {**states["s1"], "provider_name": "other"}
    with pytest.raises(SandboxCheckpointError, match="created by provider 'other'"):
        await SandboxSessionCheckpointer(provider).restore({"s1": foreign})


async def test_a_backend_without_snapshots_restores_only_while_still_paused() -> None:
    provider = FrozenOnlyProvider()
    checkpointer = await seeded(provider, "s1")
    states = await checkpointer.export(["s1"])
    assert SandboxCheckpointState.model_validate(states["s1"]).snapshot_id is None

    # Still frozen: resume it.
    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)
    assert provider.boxes["sb-1"]["state"] == "running"

    # Moved on since: nothing to rebuild from.
    await (await fresh.ensure_running("s1")).exec("drift")
    with pytest.raises(SandboxCheckpointError, match="no snapshot to re-create from"):
        await SandboxSessionCheckpointer(provider).restore(states)


# -- stop ---------------------------------------------------------------------------------------------------


async def test_stop_frees_the_sandbox_and_tolerates_repeats() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")

    await checkpointer.stop("s1")
    await checkpointer.stop("s1")
    await checkpointer.stop("never-existed")

    assert provider.boxes["sb-1"]["state"] == "stopped" and "s1" not in checkpointer


async def test_a_failed_stop_keeps_the_session_for_a_retry() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    provider.fail[("close", "sb-1")] = RuntimeError("backend down")

    with pytest.raises(RuntimeError, match="backend down"):
        await checkpointer.stop("s1")
    assert "s1" in checkpointer

    await checkpointer.stop("s1")
    assert provider.boxes["sb-1"]["state"] == "stopped" and "s1" not in checkpointer
