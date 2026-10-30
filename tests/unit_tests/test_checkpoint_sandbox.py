# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SandboxSessionCheckpointer against a fake provider whose snapshots are independent filesystem copies."""

import asyncio
import dataclasses
import json
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from nemo_gym._checkpoint import sandbox as checkpoint_sandbox
from nemo_gym._checkpoint.sandbox import (
    FORK_NOT_FOUND_RETRIES,
    SandboxCheckpointConfig,
    SandboxCheckpointError,
    SandboxCheckpointState,
    SandboxSessionCheckpointer,
    checkpoint_state_dir,
    name_hash,
    read_run_id,
    spec_from_json,
    spec_to_json,
)
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSidecarSpec,
    SandboxSpec,
    SandboxStatus,
    SupportsSandboxSnapshot,
)


pytestmark = pytest.mark.sandbox


class FakeSnapshotProvider:
    """OpenSandbox-shaped fake: an explicit snapshot copies the filesystem, create can fork a snapshot, connect is
    by id, and snapshots can be listed by name. Snapshots live until deleted, whatever the source sandbox does."""

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

    async def list_snapshots(self, prefix: str) -> list[dict[str, Any]]:
        self.calls.append(("list_snapshots", prefix))
        return [
            {"id": snapshot_id, "name": snapshot["name"]}
            for snapshot_id, snapshot in self.snapshots.items()
            if (snapshot["name"] or "").startswith(prefix)
        ]

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

    states = await checkpointer.export(["s1", "s2"], checkpoint_id="c1")

    assert set(states) == {"s1", "s2"}
    for key in states:
        state = state_of(states, key)
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
    run_id = (state_dir / "run_id").read_text()

    name = checkpointer.snapshot_name("checkpoint-0001.with_dots", "resources-session-" + "f" * 32 + "/main", 2)

    assert name == checkpointer.snapshot_name(
        "checkpoint-0001.with_dots", "resources-session-" + "f" * 32 + "/main", 2
    )
    assert checkpointer.snapshot_name_prefix == f"ng-{run_id}-{name_hash('server')}-"
    assert name == (
        f"ng-{run_id}-{name_hash('server')}-{name_hash('checkpoint-0001.with_dots')}-"
        f"{name_hash('resources-session-' + 'f' * 32 + '/main')}-2"
    )
    assert len(name) <= 63 and all(char.isalnum() or char == "-" for char in name)
    # Another key, checkpoint or attempt gives another name.
    assert len({name, checkpointer.snapshot_name("c", "k", 2), checkpointer.snapshot_name("c", "k", 3)}) == 3


def test_every_process_sharing_a_state_dir_reads_the_same_run_id(tmp_path: Path) -> None:
    first = read_run_id(tmp_path / "a")
    assert read_run_id(tmp_path / "a") == first
    assert read_run_id(tmp_path / "b") != first
    results: list[str] = []
    threads = [threading.Thread(target=lambda: results.append(read_run_id(tmp_path / "race"))) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(set(results)) == 1, "of racing processes exactly one writes the run id"
    assert sorted(path.name for path in (tmp_path / "race").iterdir()) == ["run_id"], "no temporary file is left"
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "run_id").write_text("not hex!")
    with pytest.raises(ValueError, match="run ID"):
        read_run_id(tmp_path / "bad")


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
    await checkpointer.release(frozenset())
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

    await checkpointer.release(frozenset())
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

    await checkpointer.release(frozenset({"c0"}))
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


def test_the_state_dir_is_required_once_checkpointing_is_on() -> None:
    assert checkpoint_state_dir({}) is None
    assert checkpoint_state_dir({"checkpoint": {"enabled": False}}) is None
    enabled = {"enabled": True, "control_auth_token": "t"}
    with pytest.raises(ValueError, match="checkpoint.state_dir is required"):
        checkpoint_state_dir({"checkpoint": enabled})
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

    assert restored.handle.sandbox_id == "sb-3"  # sb-1 is the original, snap-2 its snapshot
    assert provider.files("sb-3") == ["s1: one", "s1: two (replayed)"]
    await fresh.settle()
    assert provider.boxes["sb-1"]["state"] == "stopped", "the superseded sandbox is stopped"
    # The episode ends in the fork; the checkpoint still names the snapshot, so only a release deletes it.
    await fresh.stop("s1")
    assert provider.boxes["sb-3"]["state"] == "stopped" and "snap-2" in provider.snapshots
    await fresh.release(frozenset())
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
    await fresh.release(frozenset({"c2"}))
    await fresh.settle()

    # The new process never heard of c1 or c3; the index did. c2 still restores.
    assert fresh.indexed_snapshots == {"c2": ["snap-3"]} and set(provider.snapshots) == {"snap-3"}
    again = make_checkpointer(state_dir, provider)
    await again.restore(states)
    assert provider.files(again.get("s1").handle.sandbox_id) == ["s1: one"]


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

    assert provider.files(fresh.get("s1").handle.sandbox_id) == ["s1: one"]


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


async def test_restore_retries_a_fork_the_backend_cannot_find_yet(
    monkeypatch: pytest.MonkeyPatch, state_dir: Path
) -> None:
    """A snapshot store that is not fully shared answers "not found" from some replicas; a later try may succeed."""
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1")).export(["s1"], checkpoint_id="c1")
    monkeypatch.setattr(checkpoint_sandbox, "FORK_NOT_FOUND_BACKOFF_S", 0.0)
    # The first two creates land on a replica without the snapshot (sb-3, sb-4 are the failed attempts' ids).
    provider.fail[("create", "sb-3")] = RuntimeError("Create sandbox failed: HTTP 404: SNAPSHOT::NOT_FOUND")
    provider.fail[("create", "sb-4")] = RuntimeError("Create sandbox failed: HTTP 404: SNAPSHOT::NOT_FOUND")

    fresh = make_checkpointer(state_dir, provider)
    await fresh.restore(states)

    assert fresh.get("s1").handle.sandbox_id == "sb-5" and provider.files("sb-5") == ["s1: one"]


async def test_not_found_retries_stop_after_the_configured_attempts(
    monkeypatch: pytest.MonkeyPatch, state_dir: Path
) -> None:
    provider = FakeSnapshotProvider()
    states = await (await seeded(provider, state_dir, "s1")).export(["s1"], checkpoint_id="c1")
    provider.snapshots.clear()
    monkeypatch.setattr(checkpoint_sandbox, "FORK_NOT_FOUND_BACKOFF_S", 0.0)
    creates_before = len(ops(provider, "create"))

    fresh = make_checkpointer(state_dir, provider)
    with pytest.raises(SandboxCheckpointError, match="NOT_FOUND"):
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


# -- stop -----------------------------------------------------------------------------------------------------


async def test_stop_frees_the_sandbox_keeps_its_snapshots_and_tolerates_repeats(state_dir: Path) -> None:
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

    assert provider.files(fresh.get("s1").handle.sandbox_id) == ["s1: one"]
    assert provider.boxes["sb-1"]["state"] == "running", "nothing can reach it from here; its lifetime ends it"
    assert ops(provider, "connect") == [] and ops(provider, "close") == []


# -- release --------------------------------------------------------------------------------------------------


async def test_release_deletes_exactly_the_checkpoints_it_does_not_keep(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", "s2")
    for checkpoint_id in ("c1", "c2", "c3"):
        await checkpointer.export(["s1", "s2"], checkpoint_id=checkpoint_id)
    kept = checkpointer.indexed_snapshots["c2"] + checkpointer.indexed_snapshots["c3"]

    await checkpointer.release(frozenset({"c2", "c3", "never-written"}))
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

    await checkpointer.release(frozenset())
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

    await checkpointer.release(frozenset({"c2"}))
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

    await checkpointer.release(frozenset())
    await checkpointer.settle()

    assert checkpointer.indexed_snapshots == {}


async def test_a_failed_delete_keeps_its_file_and_the_next_release_retries_it(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1", "s2")
    await checkpointer.export(["s1", "s2"], checkpoint_id="c1")
    provider.fail[("delete_snapshot", "snap-3")] = RuntimeError("HTTP 409: snapshot in use")

    await checkpointer.release(frozenset({"c2"}))
    await checkpointer.settle()
    assert checkpointer.indexed_snapshots == {"c1": ["snap-3"]} and set(provider.snapshots) == {"snap-3"}

    await checkpointer.release(frozenset({"c2"}))
    await checkpointer.settle()
    assert checkpointer.indexed_snapshots == {} and provider.snapshots == {}


async def test_a_failed_listing_keeps_the_indexed_deletes(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")

    async def broken_listing(prefix: str) -> list[dict[str, Any]]:
        raise RuntimeError("HTTP 503")

    provider.list_snapshots = broken_listing  # type: ignore[method-assign]

    await checkpointer.release(frozenset())
    await checkpointer.settle()

    assert provider.snapshots == {} and checkpointer.indexed_snapshots == {}


async def test_release_returns_before_its_deletes_and_releases_run_one_after_another(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")
    await checkpointer.export(["s1"], checkpoint_id="c2")
    gate = asyncio.Event()
    deleting: list[str] = []
    original_delete = provider.delete_snapshot

    async def slow_delete(snapshot_id: str) -> None:
        deleting.append(snapshot_id)
        await gate.wait()
        await original_delete(snapshot_id)

    provider.delete_snapshot = slow_delete  # type: ignore[method-assign]

    await asyncio.wait_for(checkpointer.release(frozenset({"c2"})), timeout=1)
    await asyncio.wait_for(checkpointer.release(frozenset()), timeout=1)
    await until(lambda: deleting)
    for _ in range(20):
        await asyncio.sleep(0.001)
    # The second release waits for the first, which is still deleting c1's snapshot.
    assert deleting == ["snap-2"]
    gate.set()
    await checkpointer.settle()

    # The second release read the index after the first finished, so it deletes only c2's snapshot.
    assert deleting == ["snap-2", "snap-3"]
    assert provider.snapshots == {} and checkpointer.indexed_snapshots == {}


async def test_a_release_waits_for_another_process_holding_the_index_lock(state_dir: Path) -> None:
    """Workers of one server share its index: a release of one waits for another worker's release to finish."""
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    await checkpointer.export(["s1"], checkpoint_id="c1")
    index = checkpointer._index
    held = index.lock()  # as another worker's release would hold it
    try:
        await checkpointer.release(frozenset())
        for _ in range(50):
            await asyncio.sleep(0.002)
        assert ops(provider, "delete_snapshot") == [], "nothing is deleted while another process holds the lock"
    finally:
        index.unlock(held)
    await checkpointer.settle()
    assert provider.snapshots == {} and checkpointer.indexed_snapshots == {}


async def test_servers_sharing_a_state_dir_release_only_their_own_snapshots(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    first = SandboxSessionCheckpointer(provider, state_dir=state_dir, server_name="first")
    second = SandboxSessionCheckpointer(provider, state_dir=state_dir, server_name="second")
    await first.create("s1", SPEC)
    await second.create("s1", SPEC)
    await first.export(["s1"], checkpoint_id="c1")
    await second.export(["s1"], checkpoint_id="c1")

    await first.release(frozenset())
    await first.settle()

    assert first.indexed_snapshots == {} and second.indexed_snapshots == {"c1": ["snap-4"]}
    assert set(provider.snapshots) == {"snap-4"}


async def test_a_snapshot_id_that_cannot_name_a_file_fails_the_export(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")

    async def odd_snapshot(handle: SandboxHandle, *, name: str | None = None) -> str:
        return "../escape"

    provider.snapshot = odd_snapshot  # type: ignore[method-assign]

    with pytest.raises(SandboxCheckpointError, match="could not be indexed"):
        await checkpointer.export(["s1"], checkpoint_id="c1")
    assert not (state_dir / "server" / "escape").exists()


async def test_without_a_state_dir_sandboxes_are_tracked_but_never_exported() -> None:
    provider = FakeSnapshotProvider()
    checkpointer = SandboxSessionCheckpointer(provider, state_dir=None, server_name="server")
    await checkpointer.create("s1", SPEC)

    with pytest.raises(SandboxCheckpointError, match="without a state_dir"):
        await checkpointer.export(["s1"], checkpoint_id="c1")
    await checkpointer.release(frozenset())
    await checkpointer.stop("s1")

    assert provider.snapshots == {} and checkpointer.indexed_snapshots == {}
    assert checkpointer.snapshot_name_prefix is None
    assert provider.boxes["sb-1"]["state"] == "stopped"


# -- borrowed access ------------------------------------------------------------------------------------------


async def test_access_names_the_current_sandbox_which_a_restore_replaces(state_dir: Path) -> None:
    provider = FakeSnapshotProvider()
    checkpointer = await seeded(provider, state_dir, "s1")
    states = await checkpointer.export(["s1"], checkpoint_id="c1")

    access = await checkpointer.access("s1", provider_config_ref="sandbox", workdir="/work")

    assert access.connection.provider_config_ref == "sandbox" and access.workdir == "/work"
    assert access.connection.descriptor["sandbox_id"] == "sb-1"
    with pytest.raises(KeyError):
        await checkpointer.access("unknown", provider_config_ref="sandbox", workdir="/work")

    # After a restore the access names the fork: a borrower must be given it again.
    fresh = make_checkpointer(state_dir, provider)
    await fresh.restore(states)
    await fresh.settle()
    refreshed = await fresh.access("s1", provider_config_ref="sandbox", workdir="/work")
    assert refreshed.connection.descriptor["sandbox_id"] != "sb-1"
    assert refreshed.connection.descriptor["sandbox_id"] == fresh.get("s1").handle.sandbox_id
