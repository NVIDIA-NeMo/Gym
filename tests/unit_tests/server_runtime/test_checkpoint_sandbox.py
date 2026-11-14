# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SandboxSessionCheckpointer against a fake provider whose snapshots are independent filesystem copies."""

import asyncio
import dataclasses
import errno
import fcntl
import json
import logging
import os
import shutil
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSidecarSpec,
    SandboxSnapshotInfo,
    SandboxSnapshotInUseError,
    SandboxSnapshotNotFoundError,
    SandboxSpec,
    SandboxStatus,
    SupportsSandboxSnapshot,
)
from nemo_gym.server_runtime._checkpoint import sandbox as checkpoint_sandbox
from nemo_gym.server_runtime._checkpoint.sandbox import (
    FORK_NOT_FOUND_RETRIES,
    SandboxCheckpointConfig,
    SandboxCheckpointError,
    SandboxCheckpointState,
    SandboxSessionCheckpointer,
    SnapshotDeleteIncompleteError,
    checkpoint_state_dir,
    name_hash,
    read_state_dir_id,
    spec_from_json,
    spec_to_json,
)


pytestmark = pytest.mark.sandbox


class FakeSnapshotProvider:
    """OpenSandbox-shaped fake: an explicit snapshot copies the filesystem, create_from_snapshot forks one, connect
    is by id, and snapshots can be listed by name. Snapshots live until deleted, whatever the source sandbox does."""

    name = "fake-snapshots"

    def __init__(self) -> None:
        self.boxes: dict[str, dict[str, Any]] = {}
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self.fail: dict[tuple[str, str], Exception] = {}
        # Holds every readiness wait while set to an unset event; holds every submit while submit_gate is unset.
        self.snapshot_gate: asyncio.Event | None = None
        self.submit_gate: asyncio.Event | None = None
        self.in_flight_snapshots = 0
        self.max_in_flight_snapshots = 0
        self.in_flight_submits = 0
        self.max_in_flight_submits = 0
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

    def add_snapshot(self, name: str) -> str:
        """A snapshot some process took without indexing it, such as one that died right after the service replied."""
        self._counter += 1
        snapshot_id = f"snap-{self._counter}"
        self.snapshots[snapshot_id] = {"sandbox_id": "elsewhere", "fs": [], "name": name}
        return snapshot_id

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        return self._create(spec, snapshot_id=None)

    async def create_from_snapshot(self, snapshot_id: str, spec: SandboxSpec) -> SandboxHandle:
        return self._create(spec, snapshot_id=snapshot_id)

    def _create(self, spec: SandboxSpec, *, snapshot_id: str | None) -> SandboxHandle:
        self._counter += 1
        sandbox_id = f"sb-{self._counter}"
        self._maybe_fail("create", sandbox_id)
        if snapshot_id is not None and snapshot_id not in self.snapshots:
            raise SandboxSnapshotNotFoundError(f"no snapshot {snapshot_id}")
        fs = list(self.snapshots[snapshot_id]["fs"]) if snapshot_id is not None else []
        self.boxes[sandbox_id] = {"fs": fs, "state": "running", "spec": spec, "snapshot_id": snapshot_id}
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
        self.in_flight_submits += 1
        self.max_in_flight_submits = max(self.max_in_flight_submits, self.in_flight_submits)
        try:
            if self.submit_gate is not None:
                await self.submit_gate.wait()
        finally:
            self.in_flight_submits -= 1
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

    async def list_snapshots(self, prefix: str) -> list[SandboxSnapshotInfo]:
        self.calls.append(("list_snapshots", prefix))
        return [
            SandboxSnapshotInfo(snapshot_id=snapshot_id, name=snapshot["name"])
            for snapshot_id, snapshot in self.snapshots.items()
            if (snapshot["name"] or "").startswith(prefix)
        ]

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        # A descriptor minted with a scope carries it, as on a provider that limits what its holder may do.
        return {"sandbox_id": handle.sandbox_id, **({"scope": scope} if scope is not None else {})}

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
    list_snapshots = None  # type: ignore[assignment]


SPEC = SandboxSpec(image="img:1", workdir="/work", env={"A": "1"}, resources={"cpu": 2}, ports=[8080])
# What a checkpoint record keeps of SPEC: the snapshot stands in for the image.
RESTORED_SPEC = dataclasses.replace(SPEC, image=None)


def make_checkpointer(state_dir: Path, provider: Any, **kwargs: Any) -> SandboxSessionCheckpointer:
    """A checkpointer whose index lives under ``state_dir``; a second one over the same directory is a restart."""
    kwargs.setdefault("snapshot_retry_delay_s", 0.0)
    return SandboxSessionCheckpointer(provider, state_dir=state_dir, server_name="server", **kwargs)


async def seeded(
    provider: FakeSnapshotProvider, state_dir: Path, *keys: str, **kwargs: Any
) -> SandboxSessionCheckpointer:
    checkpointer = make_checkpointer(state_dir, provider, **kwargs)
    for key in keys:
        sandbox = await checkpointer.create(key, SPEC)
        await sandbox.exec(f"{key}: one")
    return checkpointer


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state"


async def until(condition: Any) -> None:
    """Yield to the event loop, and to the index's file writes on worker threads, until ``condition()`` holds."""
    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0.001)


def ops(provider: FakeSnapshotProvider, op: str) -> list[str]:
    return [sandbox_id for name, sandbox_id in provider.calls if name == op]


def state_of(exported: dict[str, Any], key: str) -> SandboxCheckpointState:
    return SandboxCheckpointState.model_validate(exported[key])


def test_the_fake_has_the_snapshot_capability() -> None:
    assert isinstance(FakeSnapshotProvider(), SupportsSandboxSnapshot)
    assert not isinstance(NoSnapshotProvider(), SupportsSandboxSnapshot)


# -- export -------------------------------------------------------------------------------------------------


async def test_export_snapshots_each_sandbox_and_leaves_it_running(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", "s2")

    before = time.time()
    states = await checkpointer.export(["s1", "s2"], checkpoint_id="c1")
    after = time.time()

    assert set(states) == {"s1", "s2"}
    for key in states:
        state = state_of(states, key)
        assert states[key]["version"] == 1 and before <= states[key]["snapshot_ready_ts"] <= after
        sandbox_id = state.descriptor["sandbox_id"]
        assert provider.boxes[sandbox_id]["state"] == "running"
        assert provider.snapshots[state.snapshot_id]["sandbox_id"] == sandbox_id
        assert provider.snapshots[state.snapshot_id]["fs"] == [f"{key}: one"]
        assert provider.snapshots[state.snapshot_id]["name"] == checkpointer.snapshot_name("c1", key, 1)
        assert state.provider_name == provider.name
        assert spec_from_json(state.spec) == RESTORED_SPEC
    assert checkpointer.indexed_snapshots == {"c1": sorted(state_of(states, s).snapshot_id for s in states)}
    # The episode continues in the same sandbox, and the snapshot does not follow it.
    await checkpointer.get("s1").exec("s1: two")
    assert provider.files("sb-1") == ["s1: one", "s1: two"]
    assert provider.snapshots[state_of(states, "s1").snapshot_id]["fs"] == ["s1: one"]


def test_snapshot_names_are_short_deterministic_and_carry_run_server_checkpoint_and_key(state_dir: Path) -> None:
    checkpointer = make_checkpointer(state_dir, FakeSnapshotProvider())
    state_dir_id = (state_dir / "state_dir_id").read_text()

    name = checkpointer.snapshot_name("checkpoint-0001.with_dots", "resources-session-" + "f" * 32 + "/main", 2)

    assert name == checkpointer.snapshot_name(
        "checkpoint-0001.with_dots", "resources-session-" + "f" * 32 + "/main", 2
    )
    assert checkpointer.snapshot_name_prefix == f"ng-{state_dir_id}-{name_hash('server')}-"
    assert name == (
        f"ng-{state_dir_id}-{name_hash('server')}-{name_hash('checkpoint-0001.with_dots')}-"
        f"{name_hash('resources-session-' + 'f' * 32 + '/main')}-2"
    )
    assert len(name) <= 63 and all(char.isalnum() or char == "-" for char in name)
    # Another key, checkpoint or attempt gives another name.
    assert len({name, checkpointer.snapshot_name("c", "k", 2), checkpointer.snapshot_name("c", "k", 3)}) == 3


def test_every_process_sharing_a_state_dir_reads_the_same_state_dir_id(tmp_path: Path) -> None:
    first = read_state_dir_id(tmp_path / "a")
    assert read_state_dir_id(tmp_path / "a") == first
    assert read_state_dir_id(tmp_path / "b") != first
    results: list[str] = []
    threads = [threading.Thread(target=lambda: results.append(read_state_dir_id(tmp_path / "race"))) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(results)) == 1, "of racing processes exactly one writes the run id"
    assert sorted(path.name for path in (tmp_path / "race").iterdir()) == ["state_dir_id"], "no temporary file is left"
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "state_dir_id").write_text("not hex!")
    with pytest.raises(ValueError, match="state directory ID"):
        read_state_dir_id(tmp_path / "bad")


def test_a_mark_survives_a_concurrent_unmark_emptying_the_same_checkpoint(tmp_path: Path) -> None:
    """A failed attempt's snapshot is unmarked while the next attempt of the same checkpoint is being marked."""
    index = checkpoint_sandbox._SnapshotIndex(tmp_path / "snapshots")
    stop = threading.Event()

    def churn() -> None:
        while not stop.is_set():
            index.mark("c1", "failed")
            index.unmark("c1", "failed")

    thread = threading.Thread(target=churn)
    thread.start()
    try:
        for attempt in range(300):
            index.mark("c1", f"snap-{attempt}")
            assert (tmp_path / "snapshots" / "c1" / f"snap-{attempt}").exists()
    finally:
        stop.set()
        thread.join()
    assert len(index.snapshot_ids()["c1"]) == 300


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
        sidecars=[SandboxSidecarSpec(name="db", image="postgres:16", env={"POSTGRES_PASSWORD": "x"})],
    )
    data = spec_to_json(spec)
    # Dropped: the snapshot has the files and replaces the image, so its registry credentials go too.
    assert data["files"] == {} and data["image"] is None
    assert data["provider_options"] == {"network_policy": {"egress": "none"}}
    assert data["ports"] == [8080, 9090]
    assert data["resources"] == {"cpu": 2, "memory_mib": 512, "disk_gib": 10, "gpu": 1, "gpu_type": "H100"}
    assert [sidecar["name"] for sidecar in data["sidecars"]] == ["db"]
    # The rest, the environment included, is exactly what the fork is created with.
    restored = spec_from_json(json.loads(json.dumps(data)))
    assert restored == dataclasses.replace(
        spec, image=None, files={}, provider_options={"network_policy": {"egress": "none"}}
    )


async def test_a_key_without_a_live_sandbox_exports_none(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", "s2")
    # s2's sandbox was stopped once its owner captured what it grades; "gone" never had one.
    await checkpointer.stop("s2")

    states = await checkpointer.export(["s1", "s2", "gone"], checkpoint_id="c1")

    assert states["s2"] is None and states["gone"] is None and states["s1"] is not None
    assert ops(provider, "snapshot") == ["sb-1"]


async def test_export_bounds_concurrent_submits_but_not_readiness_waits(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", "s2", "s3", "s4", "s5", parallelism=2)

    # Submits are bounded by the parallelism.
    provider.submit_gate = asyncio.Event()
    provider.snapshot_gate = asyncio.Event()
    export = asyncio.create_task(checkpointer.export(["s1", "s2", "s3", "s4", "s5"], checkpoint_id="c1"))
    await until(lambda: provider.in_flight_submits == 2)
    for _ in range(50):
        await asyncio.sleep(0.001)
    assert provider.in_flight_submits == 2 and provider.in_flight_snapshots == 0
    # Readiness waits are not: every snapshot is waited for at once, holding no slot.
    provider.submit_gate.set()
    await until(lambda: provider.in_flight_snapshots == 5)
    provider.snapshot_gate.set()
    states = await export

    assert len(states) == 5 and provider.max_in_flight_submits == 2 and provider.max_in_flight_snapshots == 5


async def test_a_failed_snapshot_fails_the_export_and_leaves_every_sandbox_running(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", "s2", snapshot_attempts=1)
    provider.fail[("snapshot", "sb-2")] = RuntimeError("registry push failed")

    with pytest.raises(SandboxCheckpointError, match="'s2'.*could not be snapshotted"):
        await checkpointer.export(["s1", "s2"], checkpoint_id="c1")

    assert provider.boxes["sb-1"]["state"] == "running" and provider.boxes["sb-2"]["state"] == "running"
    # The episodes continue untouched.
    await checkpointer.get("s1").exec("s1: two")
    await checkpointer.get("s2").exec("s2: two")
    assert provider.files("sb-1") == ["s1: one", "s1: two"] and provider.files("sb-2") == ["s2: one", "s2: two"]
    # The snapshot already taken is indexed under the failed checkpoint, which no controller keeps.
    assert checkpointer.indexed_snapshots == {"c1": ["snap-3"]}
    await checkpointer.delete_snapshots(frozenset())
    await checkpointer.settle()
    assert provider.snapshots == {} and checkpointer.indexed_snapshots == {}


async def test_export_waits_for_every_sibling_before_it_fails(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    provider.snapshot_gate = asyncio.Event()
    checkpointer = await seeded(provider, state_dir, "s1", "s2", snapshot_attempts=1)
    provider.fail[("snapshot", "sb-2")] = RuntimeError("registry push failed")

    export = asyncio.create_task(checkpointer.export(["s1", "s2"], checkpoint_id="c1"))
    await until(lambda: provider.in_flight_snapshots == 1)
    assert not export.done(), "s2 failed at once; s1 is still being taken"
    provider.snapshot_gate.set()
    with pytest.raises(SandboxCheckpointError, match="'s2'.*could not be snapshotted"):
        await export

    assert provider.in_flight_snapshots == 0
    [snapshot] = provider.snapshots.values()
    assert snapshot["sandbox_id"] == "sb-1"


async def test_a_snapshot_that_does_not_become_ready_fails_the_export_and_is_deleted(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", snapshot_attempts=1)
    provider.fail[("wait_snapshot_ready", "snap-2")] = TimeoutError("not ready after 240s")

    with pytest.raises(SandboxCheckpointError, match="'s1'.*snapshot 'snap-2' did not become ready"):
        await checkpointer.export(["s1"], checkpoint_id="c1")

    # The backend has the snapshot whatever the wait said: it is deleted at once, best effort, and leaves the index.
    await checkpointer.settle()
    assert "snap-2" not in provider.snapshots and checkpointer.indexed_snapshots == {}


async def test_a_failed_attempt_whose_delete_fails_stays_indexed_for_a_release(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", snapshot_attempts=1)
    provider.fail[("wait_snapshot_ready", "snap-2")] = TimeoutError("not ready after 240s")
    provider.fail[("delete_snapshot", "snap-2")] = RuntimeError("HTTP 503")

    with pytest.raises(SandboxCheckpointError):
        await checkpointer.export(["s1"], checkpoint_id="c1")
    await checkpointer.settle()
    assert "snap-2" in provider.snapshots and checkpointer.indexed_snapshots == {"c1": ["snap-2"]}

    await checkpointer.delete_snapshots(frozenset())
    await checkpointer.settle()
    assert provider.snapshots == {} and checkpointer.indexed_snapshots == {}


async def test_a_snapshot_that_fails_is_taken_again_and_every_attempt_is_indexed(state_dir: Path) -> None:
    # OpenSandbox's commit job can fail (CommitJobFailed) while the sandbox keeps running.
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    provider.fail[("wait_snapshot_ready", "snap-2")] = RuntimeError("CommitJobFailed")
    provider.fail[("delete_snapshot", "snap-2")] = RuntimeError("HTTP 503")

    exported = await checkpointer.export(["s1"], checkpoint_id="c1")
    await checkpointer.settle()

    assert state_of(exported, "s1").snapshot_id == "snap-3", "the second snapshot is the one a restore forks"
    assert provider.snapshots["snap-3"]["name"] == checkpointer.snapshot_name("c1", "s1", 2)
    assert checkpointer.indexed_snapshots == {"c1": ["snap-2", "snap-3"]}, "a release deletes the failed one too"


async def test_a_refused_snapshot_submit_is_retried(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", snapshot_attempts=2)
    provider.fail[("snapshot", "sb-1")] = RuntimeError("INVALID_SOURCE_STATE")

    exported = await checkpointer.export(["s1"], checkpoint_id="c1")

    assert state_of(exported, "s1").snapshot_id == "snap-2"
    assert ops(provider, "snapshot") == ["sb-1", "sb-1"]


async def test_the_export_fails_once_every_snapshot_attempt_failed(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", snapshot_attempts=2)
    provider.fail[("wait_snapshot_ready", "snap-2")] = RuntimeError("CommitJobFailed")
    provider.fail[("wait_snapshot_ready", "snap-3")] = RuntimeError("CommitJobFailed")

    with pytest.raises(SandboxCheckpointError, match="'s1'.*snapshot 'snap-3' did not become ready"):
        await checkpointer.export(["s1"], checkpoint_id="c1")

    assert ops(provider, "snapshot") == ["sb-1", "sb-1"]
    assert provider.boxes["sb-1"]["state"] == "running"
    await checkpointer.settle()


async def test_an_export_cancelled_mid_wait_still_indexes_the_snapshot(state_dir: Path) -> None:
    """A commit that misses its deadline is cancelled by the controller's resume; the snapshot exists anyway."""
    provider = FakeSnapshotProvider()
    provider.snapshot_gate = asyncio.Event()
    checkpointer = await seeded(provider, state_dir, "s1")

    export = asyncio.create_task(checkpointer.export(["s1"], checkpoint_id="c1"))
    await until(lambda: provider.in_flight_snapshots == 1)
    # Indexed the moment the service returned its id, before it is ready.
    assert checkpointer.indexed_snapshots == {"c1": ["snap-2"]}
    export.cancel()
    with pytest.raises(asyncio.CancelledError):
        await export

    await checkpointer.delete_snapshots(frozenset({"c0"}))
    await checkpointer.settle()
    assert provider.snapshots == {}, "deleted by the release that does not keep its checkpoint, not orphaned"


async def test_export_maps_a_tracked_sandbox_that_was_never_started_to_none(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    unstarted = AsyncSandbox(provider, SPEC, owns_provider=False)
    checkpointer.add("s2", unstarted, SPEC)

    states = await checkpointer.export(["s1", "s2"], checkpoint_id="c1")
    assert states["s2"] is None and states["s1"] is not None
    assert ops(provider, "snapshot") == ["sb-1"]
    assert "s2" in checkpointer and checkpointer.get("s2") is unstarted


def test_add_rejects_a_duplicate_or_a_sandbox_that_is_not_on_the_shared_provider(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = make_checkpointer(state_dir, provider)
    checkpointer.add("s1", AsyncSandbox(provider, SPEC, owns_provider=False), SPEC)

    with pytest.raises(ValueError, match="'s1' already has a sandbox"):
        checkpointer.add("s1", AsyncSandbox(provider, SPEC, owns_provider=False), SPEC)
    # A sandbox that owned the provider would close it for every session when stopped.
    with pytest.raises(ValueError, match="'s2' owns its provider"):
        checkpointer.add("s2", AsyncSandbox(provider, SPEC), SPEC)
    with pytest.raises(ValueError, match="'s2' is not on the checkpointer's provider"):
        checkpointer.add("s2", AsyncSandbox(FakeSnapshotProvider(), SPEC, owns_provider=False), SPEC)
    assert checkpointer.keys == ["s1"]


async def test_create_rejects_a_key_that_already_has_a_sandbox(state_dir: Path) -> None:
    checkpointer = await seeded(FakeSnapshotProvider(), state_dir, "s1")
    with pytest.raises(ValueError, match="'s1' already has a sandbox"):
        await checkpointer.create("s1", SPEC)


async def test_two_creates_for_one_key_at_once_start_one_sandbox(state_dir: Path) -> None:
    """A retried request can reach create while the first is still starting; the key has one sandbox either way."""
    provider = FakeSnapshotProvider()
    checkpointer = make_checkpointer(state_dir, provider)
    gate = asyncio.Event()
    original_create = provider.create

    async def held_create(spec: SandboxSpec) -> SandboxHandle:
        await gate.wait()
        return await original_create(spec)

    provider.create = held_create  # type: ignore[method-assign]

    first = asyncio.create_task(checkpointer.create("s1", SPEC))
    other = asyncio.create_task(checkpointer.create("s2", SPEC))
    await asyncio.sleep(0)
    with pytest.raises(ValueError, match="'s1' already has a sandbox"):
        await checkpointer.create("s1", SPEC)
    gate.set()
    sandbox = await first
    await other

    assert checkpointer.get("s1") is sandbox and sorted(provider.boxes) == ["sb-1", "sb-2"]
    await checkpointer.stop("s1")
    await checkpointer.stop("s2")
    assert all(box["state"] == "stopped" for box in provider.boxes.values()), "no sandbox is left unreachable"


async def test_a_create_that_failed_or_was_cancelled_can_be_retried(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = make_checkpointer(state_dir, provider)
    provider.fail[("create", "sb-1")] = RuntimeError("quota")
    with pytest.raises(RuntimeError, match="quota"):
        await checkpointer.create("s1", SPEC)

    gate = asyncio.Event()
    original_create = provider.create

    async def held_create(spec: SandboxSpec) -> SandboxHandle:
        await gate.wait()
        return await original_create(spec)

    provider.create = held_create  # type: ignore[method-assign]
    cancelled = asyncio.create_task(checkpointer.create("s1", SPEC))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    provider.create = original_create  # type: ignore[method-assign]

    await checkpointer.create("s1", SPEC)
    assert checkpointer.keys == ["s1"]


async def test_a_spec_that_cannot_be_written_fails_the_export_before_any_snapshot(state_dir: Path) -> None:
    """A tuple, a path or bytes in the spec is not JSON: every commit would fail, each after its snapshot."""
    provider = FakeSnapshotProvider()
    checkpointer = make_checkpointer(state_dir, provider)
    spec = dataclasses.replace(SPEC, provider_options={"volumes": ({"name": "data"},)})
    await checkpointer.create("s1", spec)

    with pytest.raises(SandboxCheckpointError, match="'s1' has a spec that cannot be checkpointed"):
        await checkpointer.export(["s1"], checkpoint_id="c1")

    assert ops(provider, "snapshot") == [] and provider.snapshots == {}
    assert checkpointer.indexed_snapshots == {}


async def test_a_sandbox_cannot_start_from_a_snapshot_on_a_backend_without_snapshots() -> None:
    provider = NoSnapshotProvider()
    sandbox = AsyncSandbox(provider, SPEC, owns_provider=False)

    with pytest.raises(NotImplementedError, match="'fake-plain' does not support snapshots"):
        await sandbox.start(snapshot_id="snap-1")

    assert provider.boxes == {} and not sandbox.started, "refused before anything was created"


async def test_settle_only_waits_and_cancelling_it_leaves_the_background_work_running(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1")).export(["s1"], checkpoint_id="c1")
    gate = asyncio.Event()
    original_close = provider.close

    async def held_close(handle: SandboxHandle) -> None:
        await gate.wait()
        await original_close(handle)

    provider.close = held_close  # type: ignore[method-assign]
    fresh = make_checkpointer(state_dir, provider)
    await fresh.restore(states)  # stops the superseded sb-1 in the background

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(fresh.settle(), timeout=0.05)
    assert provider.boxes["sb-1"]["state"] == "running" and len(fresh._background) == 1
    gate.set()
    await fresh.settle()

    assert provider.boxes["sb-1"]["state"] == "stopped", "the stop the cancelled settle was waiting for still ran"


def test_a_mark_is_made_again_when_an_unmark_removes_its_folder_between_the_mkdir_and_the_touch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The folder is removed exactly where a concurrent unmark of the checkpoint's only other snapshot would."""
    index = checkpoint_sandbox._SnapshotIndex(tmp_path / "snapshots")
    real_touch = Path.touch
    removed: list[Path] = []

    def touch_after_an_unmark(self: Path, *args: object, **kwargs: object) -> None:
        if not removed:
            removed.append(self.parent)
            self.parent.rmdir()
        real_touch(self, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", touch_after_an_unmark)

    index.mark("c1", "snap-1")

    assert removed == [tmp_path / "snapshots" / "c1"]
    assert index.snapshot_ids() == {"c1": ["snap-1"]}


async def test_a_mark_that_keeps_losing_its_folder_fails_the_export_after_a_bounded_number_of_tries(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    tries: list[Path] = []

    def touch_never_lands(self: Path, *args: object, **kwargs: object) -> None:
        tries.append(self)
        self.parent.rmdir()
        raise FileNotFoundError(str(self))

    monkeypatch.setattr(Path, "touch", touch_never_lands)

    with pytest.raises(SandboxCheckpointError, match="snapshot 'snap-2' could not be indexed"):
        await checkpointer.export(["s1"], checkpoint_id="c1")

    assert len(tries) == checkpoint_sandbox._MARK_ATTEMPTS
    assert ops(provider, "snapshot") == ["sb-1"], "an index failure is not retried as a failed snapshot"


async def test_a_backend_without_snapshots_cannot_be_checkpointed(state_dir: Path) -> None:
    provider = NoSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    with pytest.raises(SandboxCheckpointError, match="does not support snapshots"):
        await checkpointer.export(["s1"], checkpoint_id="c1")
    assert provider.boxes["sb-1"]["state"] == "running"


def test_the_config_is_validated_and_builds_a_checkpointer(state_dir: Path) -> None:
    with pytest.raises(ValueError, match="parallelism"):
        make_checkpointer(state_dir, FakeSnapshotProvider(), parallelism=0)
    with pytest.raises(ValueError, match="snapshot_attempts"):
        make_checkpointer(state_dir, FakeSnapshotProvider(), snapshot_attempts=0)
    with pytest.raises(ValueError, match="on_stop"):
        SandboxCheckpointConfig.model_validate({"on_stop": "kill"})
    assert SandboxCheckpointConfig() == SandboxCheckpointConfig(
        parallelism=16, snapshot_attempts=3, snapshot_retry_delay_s=5.0
    )
    provider = FakeSnapshotProvider()
    checkpointer = SandboxSessionCheckpointer.from_config(
        provider, SandboxCheckpointConfig(parallelism=3), state_dir=state_dir, server_name="server"
    )
    assert checkpointer.provider is provider


def test_an_absolute_state_dir_is_required_once_checkpointing_is_on() -> None:
    assert checkpoint_state_dir({}) is None
    assert checkpoint_state_dir({"checkpoint": {"enabled": False}}) is None
    enabled = {"enabled": True, "control_auth_token": "t"}
    with pytest.raises(ValueError, match="checkpoint.state_dir is required"):
        checkpoint_state_dir({"checkpoint": enabled})
    # Each server and each restart would resolve a relative directory against its own working directory.
    with pytest.raises(ValueError, match="must be an absolute path, got 'run/state'"):
        checkpoint_state_dir({"checkpoint": {**enabled, "state_dir": "run/state"}})
    assert checkpoint_state_dir({"checkpoint": {**enabled, "state_dir": "/x/state"}}) == Path("/x/state")


# -- restore ------------------------------------------------------------------------------------------------


async def test_restore_validates_every_state_before_touching_the_backend(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1")).export(["s1"], checkpoint_id="c1")
    calls_before = list(provider.calls)

    fresh = make_checkpointer(state_dir, provider)
    with pytest.raises(SandboxCheckpointError, match="'s2'.*invalid checkpoint state"):
        await fresh.restore({"s1": states["s1"], "s2": {"bad": 1}})

    assert provider.calls == calls_before and fresh.keys == []


async def test_restore_forks_the_snapshot_and_the_fork_holds_the_checkpoints_files(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    states = await checkpointer.export(["s1"], checkpoint_id="c1")
    # The live rollout continues after the checkpoint, then the process dies.
    await checkpointer.get("s1").exec("s1: two")
    assert provider.files("sb-1") == ["s1: one", "s1: two"]

    fresh = make_checkpointer(state_dir, provider)
    await fresh.restore(states)
    restored = fresh.get("s1")
    await restored.exec("s1: two (replayed)")

    assert restored.sandbox_id == "sb-3"  # sb-1 is the original, snap-2 its snapshot
    assert provider.files("sb-3") == ["s1: one", "s1: two (replayed)"]
    await fresh.settle()
    assert provider.boxes["sb-1"]["state"] == "stopped", "the superseded sandbox is stopped"
    # The episode ends in the fork; the checkpoint still names the snapshot, so only a release deletes it.
    await fresh.stop("s1")
    assert provider.boxes["sb-3"]["state"] == "stopped" and "snap-2" in provider.snapshots
    await fresh.delete_snapshots(frozenset())
    await fresh.settle()
    assert provider.snapshots == {}


async def test_a_none_state_restores_no_sandbox(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", "s2")
    await checkpointer.stop("s2")
    states = await checkpointer.export(["s1", "s2"], checkpoint_id="c1")

    fresh = make_checkpointer(state_dir, provider)
    await fresh.restore(states)

    assert fresh.keys == ["s1"] and "s2" not in fresh
    await fresh.settle()


async def test_after_a_crash_the_next_process_releases_what_the_dead_one_created(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")
    states = await checkpointer.export(["s1"], checkpoint_id="c2")
    # c3's export takes its snapshot, then the process dies before anything publishes c3.
    provider.snapshot_gate = asyncio.Event()
    dying = asyncio.create_task(checkpointer.export(["s1"], checkpoint_id="c3"))
    await until(lambda: provider.in_flight_snapshots == 1)
    dying.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dying

    fresh = make_checkpointer(state_dir, provider)
    await fresh.restore(states)
    await fresh.settle()
    await fresh.delete_snapshots(frozenset({"c2"}))
    await fresh.settle()

    # The new process never heard of c1 or c3; the index did. c2 still restores.
    assert fresh.indexed_snapshots == {"c2": ["snap-3"]} and set(provider.snapshots) == {"snap-3"}
    again = make_checkpointer(state_dir, provider)
    await again.restore(states)
    assert provider.files(again.get("s1").sandbox_id) == ["s1: one"]
    await again.settle()


async def test_restore_returns_before_the_superseded_sandboxes_are_stopped(state_dir: Path) -> None:
    """Stopping the crashed process's sandboxes is best effort and must not hold up the deadline-bound install."""
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1")).export(["s1"], checkpoint_id="c1")
    gate = asyncio.Event()
    original_close = provider.close

    async def slow_close(handle: SandboxHandle) -> None:
        if handle.sandbox_id == "sb-1":
            await gate.wait()
        await original_close(handle)

    provider.close = slow_close  # type: ignore[method-assign]

    fresh = make_checkpointer(state_dir, provider)
    await asyncio.wait_for(fresh.restore(states), timeout=1)

    assert fresh.keys == ["s1"] and provider.boxes["sb-1"]["state"] == "running"
    gate.set()
    await fresh.settle()
    assert provider.boxes["sb-1"]["state"] == "stopped"


async def test_restore_works_when_the_old_sandbox_is_gone(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1")).export(["s1"], checkpoint_id="c1")
    del provider.boxes["sb-1"]  # its lease expired after the crash

    fresh = make_checkpointer(state_dir, provider)
    await fresh.restore(states)
    await fresh.settle()

    assert provider.files(fresh.get("s1").sandbox_id) == ["s1: one"]


async def test_restore_tolerates_a_failure_to_stop_the_old_sandbox(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1")).export(["s1"], checkpoint_id="c1")
    provider.fail[("close", "sb-1")] = RuntimeError("backend busy")

    fresh = make_checkpointer(state_dir, provider)
    await fresh.restore(states)
    await fresh.settle()

    assert fresh.keys == ["s1"] and provider.boxes["sb-1"]["state"] == "running"


async def test_restore_rolls_back_when_one_sandbox_cannot_be_rebuilt(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", "s2")
    states = await checkpointer.export(["s1", "s2"], checkpoint_id="c1")
    # Forks are sb-5 and sb-6 (sb-1, sb-2 are originals; snap-3, snap-4 their snapshots). One of them fails.
    provider.fail[("create", "sb-6")] = RuntimeError("quota")

    fresh = make_checkpointer(state_dir, provider, parallelism=1)
    with pytest.raises(SandboxCheckpointError, match="could not be re-created"):
        await fresh.restore(states)

    assert fresh.keys == []
    assert provider.boxes["sb-5"]["state"] == "stopped", "the fork this restore made is rolled back"
    # The originals are left alone: only a successful restore stops them.
    assert provider.boxes["sb-1"]["state"] == "running" and provider.boxes["sb-2"]["state"] == "running"
    # A retried restore starts over cleanly.
    await fresh.restore(states)
    assert sorted(fresh.keys) == ["s1", "s2"]
    await fresh.settle()


async def test_restore_retries_a_fork_the_backend_cannot_find_yet(
    monkeypatch: pytest.MonkeyPatch, state_dir: Path
) -> None:
    """A snapshot store that is not fully shared answers "not found" from some replicas; a later try may succeed."""
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1")).export(["s1"], checkpoint_id="c1")
    monkeypatch.setattr(checkpoint_sandbox, "FORK_NOT_FOUND_BACKOFF_S", 0.0)
    # The first two creates land on a replica without the snapshot (sb-3, sb-4 are the failed attempts' ids).
    provider.fail[("create", "sb-3")] = SandboxSnapshotNotFoundError("snapshot not on this replica yet")
    provider.fail[("create", "sb-4")] = SandboxSnapshotNotFoundError("snapshot not on this replica yet")

    fresh = make_checkpointer(state_dir, provider)
    await fresh.restore(states)

    assert fresh.get("s1").sandbox_id == "sb-5" and provider.files("sb-5") == ["s1: one"]
    await fresh.settle()


async def test_restore_forks_through_the_providers_create_from_snapshot(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1")).export(["s1"], checkpoint_id="c1")

    fresh = make_checkpointer(state_dir, provider)
    await fresh.restore(states)

    box = provider.boxes[fresh.get("s1").sandbox_id]
    assert box["snapshot_id"] == state_of(states, "s1").snapshot_id
    # The provider decides how a snapshot replaces the image: the spec carries no provider convention for it.
    assert box["spec"].image is None and "snapshot_id" not in box["spec"].provider_options
    await fresh.settle()


async def test_not_found_retries_stop_after_the_configured_attempts(
    monkeypatch: pytest.MonkeyPatch, state_dir: Path
) -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1")).export(["s1"], checkpoint_id="c1")
    provider.snapshots.clear()
    monkeypatch.setattr(checkpoint_sandbox, "FORK_NOT_FOUND_BACKOFF_S", 0.0)
    creates_before = len(ops(provider, "create"))

    fresh = make_checkpointer(state_dir, provider)
    with pytest.raises(SandboxCheckpointError, match="no snapshot snap-2"):
        await fresh.restore(states)

    assert len(ops(provider, "create")) == creates_before + FORK_NOT_FOUND_RETRIES + 1
    assert fresh.keys == [] and provider.boxes["sb-1"]["state"] == "running"


async def test_a_fork_error_other_than_not_found_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, state_dir: Path
) -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1")).export(["s1"], checkpoint_id="c1")
    monkeypatch.setattr(checkpoint_sandbox, "FORK_NOT_FOUND_BACKOFF_S", 0.0)
    creates_before = len(ops(provider, "create"))
    # The fake pops the failure after one use: a retry would succeed and the count below would notice.
    provider.fail[("create", "sb-3")] = RuntimeError("quota exceeded")

    with pytest.raises(SandboxCheckpointError, match="could not be re-created from snapshot 'snap-2': quota"):
        await make_checkpointer(state_dir, provider).restore(states)

    assert len(ops(provider, "create")) == creates_before + 1


async def test_restore_refuses_a_state_from_another_provider_or_a_key_already_held(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    states = await checkpointer.export(["s1"], checkpoint_id="c1")

    with pytest.raises(SandboxCheckpointError, match="already has a sandbox"):
        await checkpointer.restore(states)
    foreign = {**states["s1"], "provider_name": "other"}
    with pytest.raises(SandboxCheckpointError, match="created by provider 'other'"):
        await make_checkpointer(state_dir, provider).restore({"s1": foreign})
    plain = {**states["s1"], "provider_name": NoSnapshotProvider.name}
    with pytest.raises(SandboxCheckpointError, match="does not support snapshots"):
        await make_checkpointer(state_dir, NoSnapshotProvider()).restore({"s1": plain})


async def test_a_restore_cancelled_after_a_fork_completed_stops_that_fork(state_dir: Path) -> None:
    """A restore cancelled at shutdown must not leave the forks it already made running."""
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1", "s2")).export(["s1", "s2"], checkpoint_id="c1")
    # sb-1, sb-2 are the originals and snap-3, snap-4 their snapshots: the forks are sb-5, then sb-6.
    gate = asyncio.Event()
    second_started = asyncio.Event()
    original_fork = provider.create_from_snapshot

    async def fork_then_hang(snapshot_id: str, spec: SandboxSpec) -> SandboxHandle:
        if "sb-5" in provider.boxes:
            second_started.set()
            await gate.wait()
        return await original_fork(snapshot_id, spec)

    provider.create_from_snapshot = fork_then_hang  # type: ignore[method-assign]

    fresh = make_checkpointer(state_dir, provider, parallelism=1)
    restore = asyncio.create_task(fresh.restore(states))
    await asyncio.wait_for(second_started.wait(), timeout=5)
    assert provider.boxes["sb-5"]["state"] == "running", "the first fork is complete; the second is in flight"
    restore.cancel()
    with pytest.raises(asyncio.CancelledError):
        await restore
    await fresh.settle()

    assert fresh.keys == [], "nothing is installed"
    assert provider.boxes["sb-5"]["state"] == "stopped" and "sb-6" not in provider.boxes
    assert provider.boxes["sb-1"]["state"] == "running" and provider.boxes["sb-2"]["state"] == "running"
    # The keys are not left taken: a retried restore installs both.
    provider.create_from_snapshot = original_fork  # type: ignore[method-assign]
    await fresh.restore(states)
    assert sorted(fresh.keys) == ["s1", "s2"]
    await fresh.settle()


async def test_a_fork_whose_rollback_stop_failed_is_logged_and_stopped_by_the_next_restore(
    state_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1", "s2")).export(["s1", "s2"], checkpoint_id="c1")
    # The forks are sb-5 and sb-6: the second cannot be created, and the rollback's stop of the first fails once.
    provider.fail[("create", "sb-6")] = RuntimeError("quota")
    provider.fail[("close", "sb-5")] = RuntimeError("HTTP 503")

    fresh = make_checkpointer(state_dir, provider, parallelism=1)
    with caplog.at_level(logging.WARNING, logger=checkpoint_sandbox.__name__):
        with pytest.raises(SandboxCheckpointError, match="could not be re-created.*quota"):
            await fresh.restore(states)

    assert provider.boxes["sb-5"]["state"] == "running" and fresh.keys == []
    assert "fork sb-5 of a restore that failed was not stopped" in caplog.text and "HTTP 503" in caplog.text
    assert provider.boxes["sb-1"]["state"] == "running" and provider.boxes["sb-2"]["state"] == "running"

    # The retried restore stops the fork the failed one left, then installs both sandboxes.
    await fresh.restore(states)
    assert provider.boxes["sb-5"]["state"] == "stopped"
    assert sorted(fresh.keys) == ["s1", "s2"]
    await fresh.settle()


async def failed_restore_with_an_unstopped_fork(
    state_dir: Path, provider: FakeSnapshotProvider
) -> SandboxSessionCheckpointer:
    """A restore that failed and whose rollback could not stop its one completed fork, sb-5."""
    states = await (await seeded(provider, state_dir, "s1", "s2")).export(["s1", "s2"], checkpoint_id="c1")
    provider.fail[("create", "sb-6")] = RuntimeError("quota")
    provider.fail[("close", "sb-5")] = RuntimeError("HTTP 503")
    fresh = make_checkpointer(state_dir, provider, parallelism=1)
    with pytest.raises(SandboxCheckpointError, match="quota"):
        await fresh.restore(states)
    assert provider.boxes["sb-5"]["state"] == "running"
    return fresh


async def test_settle_stops_a_fork_whose_rollback_stop_failed_without_another_restore(state_dir: Path) -> None:
    """After a failed restore the controller restarts the rollouts from their inputs: no later restore comes."""
    provider = FakeSnapshotProvider()
    fresh = await failed_restore_with_an_unstopped_fork(state_dir, provider)

    await fresh.settle()

    assert provider.boxes["sb-5"]["state"] == "stopped" and fresh._unstopped_forks == []
    assert ops(provider, "close").count("sb-5") == 2, "the failed stop and the one that worked"
    await fresh.settle()
    assert ops(provider, "close").count("sb-5") == 2, "a stopped fork is not stopped again"


async def test_a_fork_whose_stop_keeps_failing_is_tried_once_per_settle_and_stays_tracked(
    state_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    provider = FakeSnapshotProvider()
    fresh = await failed_restore_with_an_unstopped_fork(state_dir, provider)
    provider.fail[("close", "sb-5")] = RuntimeError("still 503")

    with caplog.at_level(logging.WARNING, logger=checkpoint_sandbox.__name__):
        await asyncio.wait_for(fresh.settle(), timeout=5)

    assert provider.boxes["sb-5"]["state"] == "running" and len(fresh._unstopped_forks) == 1
    assert ops(provider, "close").count("sb-5") == 2, "one more try, not a loop"
    assert "still 503" in caplog.text
    await fresh.settle()
    assert provider.boxes["sb-5"]["state"] == "stopped" and fresh._unstopped_forks == []


async def test_a_rollback_cancelled_while_it_stops_a_fork_keeps_the_fork_tracked(state_dir: Path) -> None:
    """A cancel during the stop must not leave a running fork that nothing records."""
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1", "s2")).export(["s1", "s2"], checkpoint_id="c1")
    provider.fail[("create", "sb-6")] = RuntimeError("quota")
    stopping = asyncio.Event()
    close = provider.close

    async def close_that_hangs(handle: SandboxHandle) -> None:
        if handle.sandbox_id == "sb-5":
            stopping.set()
            await asyncio.Event().wait()
        await close(handle)

    provider.close = close_that_hangs  # type: ignore[method-assign]
    fresh = make_checkpointer(state_dir, provider, parallelism=1)
    restore = asyncio.create_task(fresh.restore(states))
    await asyncio.wait_for(stopping.wait(), timeout=5)
    restore.cancel()
    with pytest.raises(asyncio.CancelledError):
        await restore

    assert provider.boxes["sb-5"]["state"] == "running" and len(fresh._unstopped_forks) == 1
    provider.close = close  # type: ignore[method-assign]
    await fresh.settle()
    assert provider.boxes["sb-5"]["state"] == "stopped" and fresh._unstopped_forks == []


async def test_a_retry_of_unstopped_forks_that_is_cancelled_keeps_them_tracked(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    fresh = await failed_restore_with_an_unstopped_fork(state_dir, provider)
    stopping = asyncio.Event()
    close = provider.close

    async def close_that_hangs(handle: SandboxHandle) -> None:
        stopping.set()
        await asyncio.Event().wait()

    provider.close = close_that_hangs  # type: ignore[method-assign]
    settle = asyncio.create_task(fresh.settle())
    await asyncio.wait_for(stopping.wait(), timeout=5)
    settle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await settle

    assert len(fresh._unstopped_forks) == 1, "emptying the list before the stops would have lost the fork"
    provider.close = close  # type: ignore[method-assign]
    await fresh.settle()
    assert provider.boxes["sb-5"]["state"] == "stopped" and fresh._unstopped_forks == []


async def test_aclose_stops_every_sandbox_and_every_fork_a_failed_restore_left(state_dir: Path) -> None:
    """A server's shutdown: the keys' sandboxes and the forks no key names, which only the checkpointer knows."""
    provider = FakeSnapshotProvider()
    fresh = await failed_restore_with_an_unstopped_fork(state_dir, provider)
    await fresh.create("live-1", SPEC)
    await fresh.create("live-2", SPEC)
    snapshots_before = dict(provider.snapshots)

    await fresh.aclose()

    assert fresh.keys == [] and fresh._unstopped_forks == []
    running = sorted(sandbox_id for sandbox_id, box in provider.boxes.items() if box["state"] == "running")
    assert running == ["sb-1", "sb-2"], "only the first process's own sandboxes, which this one never tracked"
    assert provider.snapshots == snapshots_before, "a shutdown deletes no snapshot: the controller may restore them"


async def test_aclose_stops_the_rest_when_one_stop_fails_and_a_second_call_retries_it(
    state_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", "s2", "s3")
    provider.fail[("close", "sb-2")] = RuntimeError("HTTP 503")

    with caplog.at_level(logging.WARNING, logger=checkpoint_sandbox.__name__):
        await checkpointer.aclose()

    assert checkpointer.keys == ["s2"], "the one that was not stopped stays tracked"
    assert [provider.boxes[sandbox_id]["state"] for sandbox_id in ("sb-1", "sb-2", "sb-3")] == [
        "stopped",
        "running",
        "stopped",
    ]
    assert "sandbox of s2 was not stopped at shutdown" in caplog.text and "HTTP 503" in caplog.text
    await checkpointer.aclose()
    assert checkpointer.keys == [] and provider.boxes["sb-2"]["state"] == "stopped"


async def test_restore_refuses_a_state_of_a_format_version_it_does_not_know(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1")).export(["s1"], checkpoint_id="c1")
    calls_before = list(provider.calls)

    fresh = make_checkpointer(state_dir, provider)
    with pytest.raises(
        SandboxCheckpointError, match="'s1' has checkpoint state version 2; this server reads version 1"
    ):
        await fresh.restore({"s1": {**states["s1"], "version": 2}})

    assert provider.calls == calls_before and fresh.keys == []


async def test_a_key_being_restored_cannot_be_created_or_added_meanwhile(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1")).export(["s1"], checkpoint_id="c1")
    gate = asyncio.Event()
    forking = asyncio.Event()
    original_fork = provider.create_from_snapshot

    async def held_fork(snapshot_id: str, spec: SandboxSpec) -> SandboxHandle:
        forking.set()
        await gate.wait()
        return await original_fork(snapshot_id, spec)

    provider.create_from_snapshot = held_fork  # type: ignore[method-assign]
    fresh = make_checkpointer(state_dir, provider)
    restore = asyncio.create_task(fresh.restore(states))
    await asyncio.wait_for(forking.wait(), timeout=5)

    with pytest.raises(ValueError, match="'s1' already has a sandbox"):
        await fresh.create("s1", SPEC)
    with pytest.raises(ValueError, match="'s1' already has a sandbox"):
        fresh.add("s1", AsyncSandbox(provider, SPEC, owns_provider=False), SPEC)
    gate.set()
    await restore

    assert fresh.get("s1").sandbox_id == "sb-3", "the restore's fork is the key's sandbox"
    await fresh.settle()


# -- stop -----------------------------------------------------------------------------------------------------


async def test_stop_stops_the_sandbox_keeps_its_snapshots_and_tolerates_repeats(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")

    await checkpointer.stop("s1")
    await checkpointer.stop("s1")
    await checkpointer.stop("never-existed")

    assert provider.boxes["sb-1"]["state"] == "stopped" and "s1" not in checkpointer
    # A stop never deletes a snapshot: a checkpoint the controller keeps may still restore the attempt.
    assert len(provider.snapshots) == 1 and ops(provider, "delete_snapshot") == []
    assert checkpointer.indexed_snapshots == {"c1": ["snap-2"]}


async def test_a_failed_stop_keeps_the_key_for_a_retry(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    provider.fail[("close", "sb-1")] = RuntimeError("backend down")

    with pytest.raises(RuntimeError, match="backend down"):
        await checkpointer.stop("s1")
    assert "s1" in checkpointer

    await checkpointer.stop("s1")
    assert provider.boxes["sb-1"]["state"] == "stopped" and "s1" not in checkpointer


def test_the_index_directory_is_created_at_construction_and_a_bad_one_fails_there(tmp_path: Path) -> None:
    make_checkpointer(tmp_path / "state", FakeSnapshotProvider())
    assert (tmp_path / "state" / "server" / "snapshots").is_dir()

    (tmp_path / "file").write_text("")
    with pytest.raises(OSError):
        make_checkpointer(tmp_path / "file", FakeSnapshotProvider())
    with pytest.raises(ValueError, match="server name"):
        SandboxSessionCheckpointer(FakeSnapshotProvider(), state_dir=tmp_path, server_name="a/b")


class NonConnectableSnapshotProvider(FakeSnapshotProvider):
    """Snapshots, but no connect-by-descriptor: a fresh process cannot reach the superseded sandbox."""

    name = "fake-snapshots-no-connect"

    connect = None  # type: ignore[assignment]
    serialize_handle = None  # type: ignore[assignment]


async def test_a_provider_without_connect_exports_the_id_and_skips_the_superseded_stop(state_dir: Path) -> None:
    provider = NonConnectableSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    states = await checkpointer.export(["s1"], checkpoint_id="c1")
    assert state_of(states, "s1").descriptor == {"sandbox_id": "sb-1"}

    fresh = make_checkpointer(state_dir, provider)
    await fresh.restore(states)
    await fresh.settle()

    assert provider.files(fresh.get("s1").sandbox_id) == ["s1: one"]
    assert provider.boxes["sb-1"]["state"] == "running", "nothing can reach it from here; its lifetime ends it"
    assert ops(provider, "connect") == [] and ops(provider, "close") == []


# -- release --------------------------------------------------------------------------------------------------


async def test_release_deletes_exactly_the_checkpoints_it_does_not_keep(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", "s2")
    for checkpoint_id in ("c1", "c2", "c3"):
        await checkpointer.export(["s1", "s2"], checkpoint_id=checkpoint_id)
    kept = checkpointer.indexed_snapshots["c2"] + checkpointer.indexed_snapshots["c3"]

    await checkpointer.delete_snapshots(frozenset({"c2", "c3", "never-written"}))
    await checkpointer.settle()

    assert sorted(provider.snapshots) == sorted(kept)
    assert set(checkpointer.indexed_snapshots) == {"c2", "c3"}
    # The sandboxes are not the release's business.
    assert provider.boxes["sb-1"]["state"] == "running" and provider.boxes["sb-2"]["state"] == "running"


async def test_release_of_nothing_kept_empties_the_index(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")
    await checkpointer.export(["s1"], checkpoint_id="c2")

    await checkpointer.delete_snapshots(frozenset())
    await checkpointer.settle()

    assert provider.snapshots == {} and list((state_dir / "server" / "snapshots").iterdir()) == []


async def test_release_sweeps_this_servers_snapshots_the_index_never_recorded(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c2")
    # A process died between the service's reply and the index write, for a checkpoint that was dropped (c1) and
    # for one that is kept (c2): the name says which.
    dropped = provider.add_snapshot(checkpointer.snapshot_name("c1", "s1", 1))
    kept = provider.add_snapshot(checkpointer.snapshot_name("c2", "s9", 1))
    # Not this checkpointer's: another server, another run sharing the provider, and a name it never gives.
    other_server = SandboxSessionCheckpointer(provider, state_dir=state_dir, server_name="other")
    foreign = [
        provider.add_snapshot(other_server.snapshot_name("c1", "s1", 1)),
        provider.add_snapshot(f"ng-00000000-{name_hash('server')}-{name_hash('c1')}-{name_hash('s1')}-1"),
        provider.add_snapshot(f"{checkpointer.snapshot_name_prefix}manual-backup"),
        provider.add_snapshot("unrelated"),
    ]

    await checkpointer.delete_snapshots(frozenset({"c2"}))
    await checkpointer.settle()

    assert dropped not in provider.snapshots
    assert kept in provider.snapshots and all(snapshot_id in provider.snapshots for snapshot_id in foreign)
    assert ops(provider, "list_snapshots") == [checkpointer.snapshot_name_prefix]
    assert set(checkpointer.indexed_snapshots) == {"c2"}


async def test_a_snapshot_already_gone_leaves_the_index(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")
    provider.snapshots.clear()  # the delete answers 404, which is success

    await checkpointer.delete_snapshots(frozenset())
    await checkpointer.settle()

    assert checkpointer.indexed_snapshots == {}


async def test_a_failed_delete_keeps_its_file_and_the_next_release_retries_it(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", "s2")
    await checkpointer.export(["s1", "s2"], checkpoint_id="c1")
    provider.fail[("delete_snapshot", "snap-3")] = SandboxSnapshotInUseError("HTTP 409: snapshot in use")

    # Every delete is tried before the release reports the one that failed.
    with pytest.raises(
        SnapshotDeleteIncompleteError, match=r"1 snapshots of checkpoints outside \['c2'\] were not deleted"
    ) as raised:
        await checkpointer.delete_snapshots(frozenset({"c2"}))
    assert raised.value.failed == 1
    assert checkpointer.indexed_snapshots == {"c1": ["snap-3"]} and set(provider.snapshots) == {"snap-3"}

    await checkpointer.delete_snapshots(frozenset({"c2"}))
    assert checkpointer.indexed_snapshots == {} and provider.snapshots == {}


async def test_a_failed_listing_keeps_the_indexed_deletes(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")

    async def broken_listing(prefix: str) -> list[SandboxSnapshotInfo]:
        raise RuntimeError("HTTP 503")

    provider.list_snapshots = broken_listing  # type: ignore[method-assign]

    # The release reports that it could not look for unindexed snapshots, after the indexed ones are deleted.
    with pytest.raises(RuntimeError, match="HTTP 503"):
        await checkpointer.delete_snapshots(frozenset())

    assert provider.snapshots == {} and checkpointer.indexed_snapshots == {}


def hold_delete(provider: FakeSnapshotProvider) -> tuple[asyncio.Event, list[str]]:
    """Hold every ``delete_snapshot`` of ``provider`` until the returned gate is set; the list records each start."""
    gate = asyncio.Event()
    deleting: list[str] = []
    original_delete = provider.delete_snapshot

    async def held_delete(snapshot_id: str) -> None:
        deleting.append(snapshot_id)
        await gate.wait()
        await original_delete(snapshot_id)

    provider.delete_snapshot = held_delete  # type: ignore[method-assign]
    return gate, deleting


async def test_release_returns_only_once_its_deletes_finished(state_dir: Path) -> None:
    """The participant awaits release before it admits a commit, so a later checkpoint is never among the dropped."""
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")
    kept = state_of(await checkpointer.export(["s1"], checkpoint_id="c2"), "s1").snapshot_id
    gate, deleting = hold_delete(provider)

    release = asyncio.create_task(checkpointer.delete_snapshots(frozenset({"c2"})))
    await until(lambda: deleting)
    for _ in range(20):
        await asyncio.sleep(0.001)
    assert not release.done(), "the release is still deleting, so its caller still excludes the next commit"
    gate.set()
    await release

    # The checkpoint committed once the release returned is not something that release could have dropped.
    later = state_of(await checkpointer.export(["s1"], checkpoint_id="c3"), "s1").snapshot_id
    assert deleting == ["snap-2"] and set(provider.snapshots) == {kept, later}
    assert checkpointer.indexed_snapshots == {"c2": [kept], "c3": [later]}
    assert checkpointer._background == set(), "a release leaves nothing running behind it"


async def test_a_release_cancelled_while_deleting_leaves_the_rest_for_the_next_release(state_dir: Path) -> None:
    """A release that misses its deadline is cancelled; what it had not deleted is still indexed."""
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")
    gate, deleting = hold_delete(provider)

    release = asyncio.create_task(checkpointer.delete_snapshots(frozenset()))
    await until(lambda: deleting)
    release.cancel()
    with pytest.raises(asyncio.CancelledError):
        await release
    assert checkpointer.indexed_snapshots == {"c1": ["snap-2"]} and "snap-2" in provider.snapshots

    gate.set()
    await asyncio.wait_for(checkpointer.delete_snapshots(frozenset()), timeout=5)
    assert provider.snapshots == {} and checkpointer.indexed_snapshots == {}


def lock_index(checkpointer: SandboxSessionCheckpointer) -> int:
    """Hold the index's release lock as another worker's release would; returns the descriptor to close."""
    descriptor = os.open(checkpointer._index.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return descriptor


async def test_a_release_waits_for_another_process_holding_the_index_lock(state_dir: Path) -> None:
    """Workers of one server share its index: a release of one waits for another worker's release to finish."""
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")
    held = lock_index(checkpointer)
    try:
        release = asyncio.create_task(checkpointer.delete_snapshots(frozenset()))
        for _ in range(50):
            await asyncio.sleep(0.002)
        assert not release.done()
        assert ops(provider, "delete_snapshot") == [], "nothing is deleted while another process holds the lock"
    finally:
        os.close(held)
    await asyncio.wait_for(release, timeout=5)
    assert provider.snapshots == {} and checkpointer.indexed_snapshots == {}


async def test_a_release_cancelled_while_waiting_for_the_index_lock_leaves_it_free(
    monkeypatch: pytest.MonkeyPatch, state_dir: Path
) -> None:
    """A release cancelled at its deadline while another worker holds the lock must not end up holding it."""
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")
    held = lock_index(checkpointer)
    flock = fcntl.flock
    lock_tries: list[tuple[int, bool]] = []

    def recording_flock(descriptor: int, operation: int) -> None:
        lock_tries.append((threading.get_ident(), bool(operation & fcntl.LOCK_NB)))
        flock(descriptor, operation)

    monkeypatch.setattr(checkpoint_sandbox.fcntl, "flock", recording_flock)
    try:
        release = asyncio.create_task(checkpointer.delete_snapshots(frozenset()))
        # Until it has tried the held lock a second time: it is waiting, not giving up.
        for _ in range(2000):
            if len(lock_tries) >= 2:
                break
            await asyncio.sleep(0.002)
        release.cancel()
        with pytest.raises(asyncio.CancelledError):
            await release
    finally:
        os.close(held)

    # A blocking flock in a worker thread cannot be cancelled: every cancelled wait would leave one executor thread
    # blocked until the lock is free, and the next export needs those threads to write its index.
    assert len(lock_tries) > 1, "the lock was tried again while another worker held it"
    assert set(lock_tries) == {(threading.get_ident(), True)}, "every try is non-blocking, on the event loop's thread"
    monkeypatch.undo()
    # Nothing of the cancelled release holds the lock: this checkpointer and another process both take it at once.
    os.close(lock_index(checkpointer))
    await asyncio.wait_for(checkpointer.delete_snapshots(frozenset()), timeout=5)
    assert ops(provider, "delete_snapshot") == ["snap-2"]
    assert provider.snapshots == {} and checkpointer.indexed_snapshots == {}


async def test_a_release_that_cannot_lock_the_index_says_why_and_deletes_nothing(
    monkeypatch: pytest.MonkeyPatch, state_dir: Path
) -> None:
    """A filesystem without flock (ENOSYS on Lustre mounted without it) must not fail every release in silence."""
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")
    seen: list[int] = []

    def no_flock(descriptor: int, operation: int) -> None:
        seen.append(descriptor)
        raise OSError(errno.ENOSYS, "Function not implemented")

    monkeypatch.setattr(checkpoint_sandbox.fcntl, "flock", no_flock)

    with pytest.raises(RuntimeError, match="cannot lock the snapshot index .*snapshots.lock: .*flock") as raised:
        await checkpointer.delete_snapshots(frozenset())

    assert "Function not implemented" in str(raised.value)
    assert checkpointer.indexed_snapshots == {"c1": ["snap-2"]} and "snap-2" in provider.snapshots
    [descriptor] = seen
    with pytest.raises(OSError) as closed:
        os.fstat(descriptor)
    assert closed.value.errno == errno.EBADF, "a release that cannot lock does not leak its descriptor"


async def test_the_sweep_leaves_names_under_its_prefix_that_it_never_gives(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c2")
    prefix = checkpointer.snapshot_name_prefix
    foreign = [
        provider.add_snapshot(f"{prefix}manual01-backup-daily"),  # three parts and eight characters, but no hashes
        provider.add_snapshot(f"{prefix}{name_hash('c1')}-backup-1"),
        provider.add_snapshot(f"{prefix}{name_hash('c1')}-{name_hash('s1')}-latest"),
        provider.add_snapshot(f"{prefix}{name_hash('c1')}-{name_hash('s1')}-0"),
        # A name this checkpointer gives, followed by more: a copy someone made of one of its snapshots.
        provider.add_snapshot(f"{checkpointer.snapshot_name('c1', 's1', 1)}-backup"),
        provider.add_snapshot(f"{checkpointer.snapshot_name('c1', 's1', 10)}.old"),
    ]
    # Names it does give, for a dropped checkpoint, are still swept, whatever the attempt.
    orphans = [provider.add_snapshot(checkpointer.snapshot_name("c1", "s1", attempt)) for attempt in (1, 10)]

    await checkpointer.delete_snapshots(frozenset({"c2"}))

    assert all(snapshot_id not in provider.snapshots for snapshot_id in orphans)
    assert [snapshot_id for snapshot_id in foreign if snapshot_id not in provider.snapshots] == []


async def test_a_release_after_the_index_was_lost_still_sweeps_by_name_and_keeps_kept_checkpoints(
    state_dir: Path,
) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")
    kept = state_of(await checkpointer.export(["s1"], checkpoint_id="c2"), "s1").snapshot_id
    shutil.rmtree(state_dir / "server" / "snapshots")
    assert checkpointer.indexed_snapshots == {}

    await checkpointer.delete_snapshots(frozenset({"c2"}))

    # The names, not the index, say which checkpoint each snapshot belongs to.
    assert set(provider.snapshots) == {kept}
    assert ops(provider, "delete_snapshot") == ["snap-2"]


async def test_a_release_on_a_backend_without_snapshots_leaves_the_index_alone(state_dir: Path) -> None:
    """A server restarted on a provider that cannot delete snapshots must keep the record of what it owes."""
    provider = FakeSnapshotProvider()
    await (await seeded(provider, state_dir, "s1")).export(["s1"], checkpoint_id="c1")

    plain = make_checkpointer(state_dir, NoSnapshotProvider())
    await plain.delete_snapshots(frozenset())
    assert plain.indexed_snapshots == {"c1": ["snap-2"]}

    # A later process on the snapshot provider still finds and deletes it.
    again = make_checkpointer(state_dir, provider)
    await again.delete_snapshots(frozenset())
    assert provider.snapshots == {} and again.indexed_snapshots == {}


async def test_a_release_skips_a_checkpoint_folder_that_vanishes_while_the_index_is_read(
    monkeypatch: pytest.MonkeyPatch, state_dir: Path
) -> None:
    """An unmark of a failed attempt, which takes no lock, can empty and remove a folder while a release lists it."""
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")
    await checkpointer.export(["s1"], checkpoint_id="c2")
    vanishing = state_dir / "server" / "snapshots" / "c2"
    real_iterdir = Path.iterdir

    def iterdir(self: Path) -> Any:
        if self == vanishing:
            shutil.rmtree(vanishing)
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", iterdir)

    await checkpointer.delete_snapshots(frozenset({"c2"}))

    assert ops(provider, "delete_snapshot") == ["snap-2"], "the other dropped checkpoint is still released"
    assert checkpointer.indexed_snapshots == {}


async def test_servers_sharing_a_state_dir_release_only_their_own_snapshots(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    first = SandboxSessionCheckpointer(provider, state_dir=state_dir, server_name="first")
    second = SandboxSessionCheckpointer(provider, state_dir=state_dir, server_name="second")
    await first.create("s1", SPEC)
    await second.create("s1", SPEC)
    await first.export(["s1"], checkpoint_id="c1")
    await second.export(["s1"], checkpoint_id="c1")

    await first.delete_snapshots(frozenset())
    await first.settle()

    assert first.indexed_snapshots == {} and second.indexed_snapshots == {"c1": ["snap-4"]}
    assert set(provider.snapshots) == {"snap-4"}


async def test_a_snapshot_id_that_cannot_name_a_file_fails_the_export(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")

    async def odd_snapshot(handle: SandboxHandle, *, name: str | None = None) -> str:
        return "../escape"

    provider.snapshot = odd_snapshot  # type: ignore[method-assign]
    before = sorted(path.relative_to(state_dir) for path in state_dir.rglob("*"))

    with pytest.raises(SandboxCheckpointError, match="could not be indexed"):
        await checkpointer.export(["s1"], checkpoint_id="c1")
    # ``<index>/c1/../escape`` would be ``<state_dir>/server/snapshots/escape``: nothing was written anywhere.
    assert sorted(path.relative_to(state_dir) for path in state_dir.rglob("*")) == before
    assert checkpointer.indexed_snapshots == {}


async def test_without_a_state_dir_sandboxes_are_tracked_but_never_exported() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = SandboxSessionCheckpointer(provider, state_dir=None, server_name="server")
    await checkpointer.create("s1", SPEC)

    with pytest.raises(SandboxCheckpointError, match="without a state_dir"):
        await checkpointer.export(["s1"], checkpoint_id="c1")
    await checkpointer.delete_snapshots(frozenset())
    await checkpointer.stop("s1")

    assert provider.snapshots == {} and checkpointer.indexed_snapshots == {}
    assert checkpointer.snapshot_name_prefix is None
    assert provider.boxes["sb-1"]["state"] == "stopped"


# -- checkpoints no release drops -----------------------------------------------------------------------------


def unreleased_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if "no release has dropped" in record.getMessage()]


async def test_an_export_warns_once_the_index_holds_more_than_a_few_checkpoints_and_again_when_that_doubles(
    state_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Nothing but a release deletes a snapshot, so a controller that never releases leaves every one for good."""
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    warned_at: list[int] = []
    with caplog.at_level(logging.WARNING, logger=checkpoint_sandbox.__name__):
        for number in range(1, 15):
            await checkpointer.export(["s1"], checkpoint_id=f"c{number}")
            if len(unreleased_warnings(caplog)) > len(warned_at):
                warned_at.append(number)

    # Not at every commit of a long run: at the sixth checkpoint, and next once there are more than twice as many.
    assert warned_at == [6, 13]
    first = unreleased_warnings(caplog)[0]
    assert "server server holds 6 checkpoints" in first and checkpointer.snapshot_name_prefix in first


async def test_a_controller_that_releases_is_never_warned_and_one_that_stops_releasing_is(
    state_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    with caplog.at_level(logging.WARNING, logger=checkpoint_sandbox.__name__):
        for number in range(1, 21):
            await checkpointer.export(["s1"], checkpoint_id=f"c{number}")
            await checkpointer.delete_snapshots(frozenset({f"c{number}", f"c{number - 1}"}))
        assert unreleased_warnings(caplog) == [], "the index never held more than the two checkpoints kept"

        # The releases stop: the warning comes once the index has grown past the threshold again.
        for number in range(21, 26):
            await checkpointer.export(["s1"], checkpoint_id=f"c{number}")
    assert len(unreleased_warnings(caplog)) == 1 and "holds 6 checkpoints" in unreleased_warnings(caplog)[0]


async def test_a_checkpointer_without_a_state_dir_has_no_index_to_warn_of(caplog: pytest.LogCaptureFixture) -> None:
    checkpointer = SandboxSessionCheckpointer(FakeSnapshotProvider(), state_dir=None, server_name="server")
    with caplog.at_level(logging.WARNING, logger=checkpoint_sandbox.__name__):
        assert await checkpointer.export([], checkpoint_id="c1") == {}
    assert unreleased_warnings(caplog) == []


# -- the sandbox's own snapshot ---------------------------------------------------------------------------------


async def test_a_sandbox_snapshots_itself_through_its_provider_only_while_it_is_started() -> None:
    """The checkpointer snapshots through the sandbox, which checks that it is started, not through its handle."""
    provider = FakeSnapshotProvider()
    sandbox = AsyncSandbox(provider, SPEC, owns_provider=False)
    assert not sandbox.started
    with pytest.raises(RuntimeError, match="has not been started"):
        await sandbox.snapshot(name="early")

    await sandbox.start()
    await sandbox.exec("one")
    assert sandbox.started
    snapshot_id = await sandbox.snapshot(name="ng-named")
    assert provider.snapshots[snapshot_id] == {"sandbox_id": "sb-1", "fs": ["one"], "name": "ng-named"}

    await sandbox.stop()
    assert not sandbox.started and sandbox.sandbox_id == "sb-1", "the id outlives the stop; started does not"
    with pytest.raises(RuntimeError, match="has not been started"):
        await sandbox.snapshot(name="late")
    assert len(provider.snapshots) == 1


async def test_a_sandbox_on_a_provider_without_snapshots_refuses_to_snapshot() -> None:
    sandbox = AsyncSandbox(NoSnapshotProvider(), SPEC, owns_provider=False)
    await sandbox.start()
    with pytest.raises(NotImplementedError, match="'fake-plain' does not support snapshots"):
        await sandbox.snapshot(name="ng-named")
    await sandbox.stop()


# -- borrowed access ------------------------------------------------------------------------------------------


async def test_access_names_the_current_sandbox_which_a_restore_replaces(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    states = await checkpointer.export(["s1"], checkpoint_id="c1")

    access = await checkpointer.access("s1", provider_config_ref="sandbox", workdir="/work")

    assert access.connection.provider_config_ref == "sandbox" and access.workdir == "/work"
    assert access.connection.descriptor["sandbox_id"] == "sb-1"
    # A borrower may operate the sandbox, not destroy it: the owner still checkpoints it and stops it itself.
    assert access.connection.descriptor["scope"] == "operate"
    # The checkpoint's own descriptor is not a borrower's: a restore stops the superseded sandbox with it.
    assert "scope" not in state_of(states, "s1").descriptor
    with pytest.raises(KeyError):
        await checkpointer.access("unknown", provider_config_ref="sandbox", workdir="/work")

    # After a restore the access names the fork: a borrower must be given it again.
    fresh = make_checkpointer(state_dir, provider)
    await fresh.restore(states)
    await fresh.settle()
    refreshed = await fresh.access("s1", provider_config_ref="sandbox", workdir="/work")
    assert refreshed.connection.descriptor["sandbox_id"] != "sb-1"
    assert refreshed.connection.descriptor["sandbox_id"] == fresh.get("s1").sandbox_id
