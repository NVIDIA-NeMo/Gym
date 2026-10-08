# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SandboxSessionCheckpointer against a fake pause/resume provider whose snapshots are filesystem copies."""

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.sandbox.api import AsyncSandbox
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
    snapshot_survives_resume = True

    def __init__(self, *, keeps_snapshots: bool = True, consumes_snapshot_on_resume: bool = False) -> None:
        self.keeps_snapshots = keeps_snapshots
        self.consumes_snapshot_on_resume = consumes_snapshot_on_resume
        if consumes_snapshot_on_resume:
            self.snapshot_survives_resume = False
        self.lookup_error: Exception | None = None
        self.lookup_delay_s = 0.0
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
        if (spec.image is None) == (snapshot_id is None):
            raise ValueError("exactly one of image or snapshot_id must be specified")  # as OpenSandbox does
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
        if self.consumes_snapshot_on_resume:
            for snapshot_id in [s for s, snap in self.snapshots.items() if snap["sandbox_id"] == handle.sandbox_id]:
                del self.snapshots[snapshot_id]

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
        if self.lookup_delay_s:
            await asyncio.sleep(self.lookup_delay_s)
        if self.lookup_error is not None:
            raise self.lookup_error
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
    with pytest.raises(SandboxCheckpointError, match="resumed after the checkpoint; this backend keeps no durable"):
        await SandboxSessionCheckpointer(provider).restore(states)


# -- a backend whose resume consumes the snapshot (OpenSandbox on Kubernetes) ----------------------------------


def consuming_provider() -> FakeSnapshotProvider:
    return FakeSnapshotProvider(consumes_snapshot_on_resume=True)


async def test_export_records_the_paused_restore_point_and_a_best_effort_snapshot_id() -> None:
    provider = consuming_provider()
    checkpointer = await seeded(provider, "s1")
    assert checkpointer.restore_point == "paused"

    state = SandboxCheckpointState.model_validate((await checkpointer.export(["s1"]))["s1"])

    assert state.restore_point == "paused"
    # The snapshot id is still recorded for the operator tooling while the listing can see it.
    assert state.snapshot_id in provider.snapshots
    assert state.expires_at is None, "the spec sets no TTL"


async def test_export_tolerates_a_failing_or_slow_snapshot_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = consuming_provider()
    checkpointer = await seeded(provider, "s1", "s2")
    provider.lookup_error = RuntimeError("listing is per replica and this one is empty")
    first = SandboxCheckpointState.model_validate((await checkpointer.export(["s1"]))["s1"])

    provider.lookup_error = None
    provider.lookup_delay_s = 0.2
    monkeypatch.setattr("nemo_gym.sandbox.checkpoint.SNAPSHOT_LOOKUP_TIMEOUT_S", 0.01)
    second = SandboxCheckpointState.model_validate((await checkpointer.export(["s2"]))["s2"])

    # Neither failure fails the commit: the sandboxes are paused and the states say so, without a snapshot.
    assert first.snapshot_id is None and second.snapshot_id is None
    assert provider.boxes["sb-1"]["state"] == "paused" and provider.boxes["sb-2"]["state"] == "paused"


async def test_restore_resumes_in_place_while_still_paused_without_consulting_the_listing() -> None:
    provider = consuming_provider()
    states = await (await seeded(provider, "s1")).export(["s1"])
    provider.lookup_error = RuntimeError("listing unavailable")

    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)
    await (await fresh.ensure_running("s1")).exec("s1: two")

    assert fresh.get("s1").handle.sandbox_id == "sb-1"
    assert provider.files("sb-1") == ["s1: one", "s1: two"]
    assert ops(provider, "create") == ["sb-1"], "nothing was forked"


async def test_restore_fails_typed_once_the_sandbox_resumed_after_the_checkpoint() -> None:
    provider = consuming_provider()
    checkpointer = await seeded(provider, "s1")
    states = await checkpointer.export(["s1"])
    # The run continued (eager resume consumed the snapshot), then the process died.
    await (await checkpointer.ensure_running("s1")).exec("drift")
    assert provider.snapshots == {}

    with pytest.raises(SandboxCheckpointError, match="'s1'.*resumed after the checkpoint") as error:
        await SandboxSessionCheckpointer(provider).restore(states)

    assert "keeps no durable snapshot" in str(error.value)
    assert ops(provider, "create") == ["sb-1"], "no fork was attempted"
    assert provider.boxes["sb-1"]["state"] == "running", "the live sandbox is left alone"


async def test_restore_names_the_ttl_when_the_sandbox_expired() -> None:
    provider = consuming_provider()
    checkpointer = SandboxSessionCheckpointer(provider)
    spec = SandboxSpec(image="img:1", ttl_s=3600)
    await (await checkpointer.create("s1", spec)).exec("s1: one")
    states = await checkpointer.export(["s1"])
    state = SandboxCheckpointState.model_validate(states["s1"])
    assert state.expires_at is not None and 0 < state.expires_at - state.paused_at <= 3600

    # Long after: the backend reaped the sandbox.
    del provider.boxes["sb-1"]
    expired = {"s1": {**states["s1"], "expires_at": state.paused_at - 1}}
    with pytest.raises(SandboxCheckpointError, match="reached its TTL at .* and the checkpoint keeps no snapshot"):
        await SandboxSessionCheckpointer(provider).restore(expired)

    # Within the TTL but gone anyway: the error says so without blaming the TTL.
    with pytest.raises(SandboxCheckpointError, match="is unreachable and the checkpoint keeps no snapshot"):
        await SandboxSessionCheckpointer(provider).restore(states)


async def test_add_dates_the_ttl_estimate_from_the_given_creation_time() -> None:
    provider = consuming_provider()
    checkpointer = SandboxSessionCheckpointer(provider)
    spec = SandboxSpec(image="img:1", ttl_s=100)
    handle = await provider.create(spec)
    sandbox = await AsyncSandbox.connect({"sandbox_id": handle.sandbox_id}, provider=provider, owns_provider=False)
    checkpointer.add("s1", sandbox, spec, created_at=1_000.0)

    state = SandboxCheckpointState.model_validate((await checkpointer.export(["s1"]))["s1"])
    assert state.expires_at == 1_100.0


async def test_a_snapshot_backend_still_forks_after_a_resume() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    assert checkpointer.restore_point == "snapshot"
    states = await checkpointer.export(["s1"])
    assert SandboxCheckpointState.model_validate(states["s1"]).restore_point == "snapshot"
    await (await checkpointer.ensure_running("s1")).exec("drift")

    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)
    assert provider.files(fresh.get("s1").handle.sandbox_id) == ["s1: one"]


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


# -- eager resume ---------------------------------------------------------------------------------------------


async def test_resume_paused_resumes_what_a_checkpoint_paused_and_tolerates_failures() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1", "s2", "s3")
    await checkpointer.export(["s1", "s2"])
    provider.fail[("resume", "sb-1")] = RuntimeError("backend busy")

    await checkpointer.resume_paused(["s1", "s2", "s3", "gone"])

    # s2 resumed, s3 was never paused, s1's failure is logged and left for the next use.
    assert provider.boxes["sb-2"]["state"] == "running" and provider.boxes["sb-3"]["state"] == "running"
    assert provider.boxes["sb-1"]["state"] == "paused"
    assert ops(provider, "resume") == ["sb-1", "sb-2"]
    await (await checkpointer.ensure_running("s1")).exec("s1: two")
    assert provider.boxes["sb-1"]["state"] == "running" and provider.files("sb-1") == ["s1: one", "s1: two"]


# -- borrowed access ------------------------------------------------------------------------------------------


async def test_access_names_the_current_sandbox_and_resumes_it_first() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, "s1")
    states = await checkpointer.export(["s1"])

    access = await checkpointer.access("s1", provider_config_ref="sandbox", workdir="/work")

    assert access.connection.provider_config_ref == "sandbox" and access.workdir == "/work"
    assert access.connection.descriptor["sandbox_id"] == "sb-1"
    assert provider.boxes["sb-1"]["state"] == "running", "a borrower is about to use it"

    # After a restore that forked the sandbox, the access names the fork: a borrower must ask again.
    await (await checkpointer.ensure_running("s1")).exec("drift")
    fresh = SandboxSessionCheckpointer(provider)
    await fresh.restore(states)
    refreshed = await fresh.access("s1", provider_config_ref="sandbox", workdir="/work")
    assert refreshed.connection.descriptor["sandbox_id"] != "sb-1"
    assert refreshed.connection.descriptor["sandbox_id"] == fresh.get("s1").handle.sandbox_id
