# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invariants of ``SandboxSessionCheckpointer`` over random sequences of operations with injected backend faults.

Example tests check what each operation does on its own. This test checks what must hold however tool calls,
exports, parks, releases, restores into fresh processes, and stops interleave while the backend fails at random:
no live session points at a stopped sandbox, every snapshot of a kept checkpoint exists and holds the files its
session had at the checkpoint, every snapshot on the backend is in the index, a failed restore leaves nothing
behind, and every kept checkpoint restores into a fresh checkpointer with byte-equal files, whatever happened to
the live sandbox after the commit. Once faults stop, a release leaves exactly the kept checkpoints' snapshots.
A failing seed reproduces on its own.
"""

import os
import random
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.sandbox.checkpoint import SandboxCheckpointError, SandboxCheckpointState, SandboxSessionCheckpointer
from nemo_gym.sandbox.providers.base import SandboxSpec
from tests.unit_tests.test_sandbox_checkpoint import FakeSnapshotProvider


SEEDS = range(int(os.environ.get("NEMO_GYM_SANDBOX_SEQUENCE_SEEDS", "40")))
SPEC = SandboxSpec(image="img")
CHAOS_OPS = {"snapshot", "wait_snapshot_ready", "create", "connect", "close", "delete_snapshot"}


class ChaosProvider(FakeSnapshotProvider):
    """The fake provider, failing a share of its calls at random while armed."""

    def __init__(self, rng: random.Random, fault_rate: float) -> None:
        super().__init__()
        self.rng = rng
        self.fault_rate = fault_rate
        self.armed = True

    def _maybe_fail(self, op: str, sandbox_id: str) -> None:
        super()._maybe_fail(op, sandbox_id)
        if self.armed and op in CHAOS_OPS and self.rng.random() < self.fault_rate:
            raise RuntimeError(f"chaos: {op} on {sandbox_id}")


def _states(exported: dict[str, Any]) -> dict[str, SandboxCheckpointState]:
    return {session_id: SandboxCheckpointState.model_validate(raw) for session_id, raw in exported.items()}


def check_live_invariants(
    provider: ChaosProvider, live: SandboxSessionCheckpointer, expected_files: dict[str, list[str]]
) -> None:
    for session_id in live.session_ids:
        box = provider.boxes[live.get(session_id).handle.sandbox_id]
        # A live session's sandbox is running: nothing a checkpoint does pauses or stops it.
        assert box["state"] == "running", (session_id, box["state"])
        assert box["fs"] == expected_files[session_id], session_id
    assert set(live.session_ids) == set(expected_files)


def check_committed_invariants(
    provider: ChaosProvider, live: SandboxSessionCheckpointer, committed: dict[str, dict[str, Any]]
) -> None:
    for exported in committed.values():
        for session_id, state in _states(exported).items():
            assert state.snapshot_id in provider.snapshots, (session_id, state.snapshot_id)
    # Nothing on the backend escapes the index, so a release can always reach it.
    indexed = {snapshot_id for ids in live.indexed_snapshots.values() for snapshot_id in ids}
    assert set(provider.snapshots) == indexed


@pytest.mark.parametrize("seed", SEEDS)
async def test_random_sequences_keep_the_sandbox_invariants(seed: int, tmp_path: Path) -> None:
    rng = random.Random(seed)
    provider = ChaosProvider(rng, fault_rate=rng.choice([0.0, 0.1, 0.25]))

    def checkpointer(**kwargs: Any) -> SandboxSessionCheckpointer:
        # Every process, the live one and each one a crash starts, shares the index on disk.
        return SandboxSessionCheckpointer(provider, state_dir=tmp_path, server_name="server", **kwargs)

    live = checkpointer(parallelism=rng.choice([1, 2, 4]))
    expected_files: dict[str, list[str]] = {}
    # Each checkpoint the controller keeps, by id: the exported states, and the files each session had then.
    committed: dict[str, dict[str, Any]] = {}
    committed_files: dict[str, dict[str, list[str]]] = {}
    log: list[str] = []
    counter = 0

    for _ in range(rng.randint(15, 50)):
        op = rng.choices(
            ["tool", "create", "export", "stop", "release", "crash_restore", "park_restore"],
            weights=[5, 2, 2, 1, 1, 1, 1],
        )[0]
        try:
            if op == "create" or (op == "tool" and not expected_files):
                counter += 1
                session_id = f"s{counter}"
                await live.create(session_id, SPEC)
                expected_files[session_id] = []
                log.append(f"create {session_id}")
            elif op == "tool":
                session_id = rng.choice(sorted(expected_files))
                counter += 1
                await live.get(session_id).exec(f"{session_id}: line {counter}")
                expected_files[session_id].append(f"{session_id}: line {counter}")
                log.append(f"tool {session_id}")
            elif op == "export":
                chosen = [s for s in sorted(expected_files) if rng.random() < 0.7] + (
                    ["gone"] if rng.random() < 0.3 else []
                )
                snapshot_of = dict(expected_files)
                counter += 1
                checkpoint_id = f"c{counter}"
                exported = await live.export(chosen, checkpoint_id=checkpoint_id)
                assert set(exported) == set(chosen) & set(expected_files)
                for session_id, state in _states(exported).items():
                    assert provider.boxes[state.descriptor["sandbox_id"]]["state"] == "running"
                    assert provider.snapshots[state.snapshot_id]["fs"] == snapshot_of[session_id]
                committed[checkpoint_id] = exported
                committed_files[checkpoint_id] = {s: list(snapshot_of[s]) for s in exported}
                log.append(f"export {checkpoint_id} {sorted(exported)}")
            elif op == "release":
                kept = sorted(c for c in committed if rng.random() < 0.6)
                await live.release(kept)
                await live.settle()
                committed = {c: committed[c] for c in kept}
                committed_files = {c: committed_files[c] for c in kept}
                # A dropped checkpoint survives only where a delete failed, for the next release to retry.
                assert provider.fault_rate or set(live.indexed_snapshots) <= set(kept), log
                log.append(f"release {kept}")
            elif op == "stop" and expected_files:
                session_id = rng.choice(sorted(expected_files))
                await live.stop(session_id)
                assert session_id not in live
                expected_files.pop(session_id)
                log.append(f"stop {session_id}")
            elif op in ("crash_restore", "park_restore") and committed:
                index = rng.choice(sorted(committed))
                if op == "park_restore":
                    # The controller stopped after this commit: the exported sandboxes were killed first.
                    await live.park(list(committed[index]))
                    log.append(f"park #{index}")
                before = provider._counter
                fresh = checkpointer(parallelism=rng.choice([1, 2, 4]))
                try:
                    await fresh.restore(committed[index])
                except SandboxCheckpointError:
                    # A failed restore installs nothing and issues a stop for every sandbox it created; the stop
                    # itself may be one of the injected failures, and the sandbox then ends with its lifetime.
                    assert fresh.session_ids == []
                    for sandbox_id, box in provider.boxes.items():
                        if int(sandbox_id.split("-")[1]) > before:
                            assert box["state"] == "stopped" or ("close", sandbox_id) in provider.calls, (
                                f"restore leaked {sandbox_id} without trying to stop it"
                            )
                    log.append(f"{op} #{index} failed")
                else:
                    for session_id, state in _states(committed[index]).items():
                        box = provider.boxes[fresh.get(session_id).handle.sandbox_id]
                        assert box["state"] == "running"
                        assert box["fs"] == provider.snapshots[state.snapshot_id]["fs"]
                    log.append(f"{op} #{index}")
                # The crashed process is gone either way; what the restore did not bring back restarts from input.
                live = fresh
                expected_files = (
                    {s: list(files) for s, files in committed_files[index].items()} if fresh.session_ids else {}
                )
        except (SandboxCheckpointError, RuntimeError) as error:
            log.append(f"{op} failed: {error}")
        check_live_invariants(provider, live, expected_files)
        check_committed_invariants(provider, live, committed)

    # Every kept checkpoint restores into a fresh process, with the files its sessions had at the checkpoint,
    # whatever happened to the live sandboxes since.
    provider.armed = False
    for checkpoint_id, exported in committed.items():
        fresh = checkpointer()
        await fresh.restore(exported)
        for session_id in exported:
            files = committed_files[checkpoint_id][session_id]
            assert provider.boxes[fresh.get(session_id).handle.sandbox_id]["fs"] == files, (seed, log)
        await fresh.settle()
    # Without faults, a release leaves exactly the snapshots of the checkpoints it keeps, and an index to match.
    await live.release(sorted(committed))
    await live.settle()
    kept_snapshots = {state.snapshot_id for exported in committed.values() for state in _states(exported).values()}
    assert set(provider.snapshots) == kept_snapshots, (seed, log)
    assert set(live.indexed_snapshots) == {c for c, exported in committed.items() if exported}, (seed, log)
    await live.release([])
    await live.settle()
    assert provider.snapshots == {} and live.indexed_snapshots == {}, (seed, log)
