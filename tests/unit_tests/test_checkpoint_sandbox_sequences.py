# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invariants of ``SandboxSessionCheckpointer`` over random sequences of operations with injected backend faults.

Example tests check what each operation does on its own. This test checks what must hold however tool calls,
exports, restores into fresh processes, and stops interleave while the backend fails at random: no sandbox is
leaked or left paused and forgotten, every committed snapshot exists and holds the files its session had at
the checkpoint, a failed restore leaves nothing behind, and every committed checkpoint restores into a fresh
checkpointer with byte-equal files. A failing seed reproduces on its own.
"""

import os
import random
from typing import Any

import pytest

from nemo_gym.sandbox.checkpoint import SandboxCheckpointError, SandboxCheckpointState, SandboxSessionCheckpointer
from nemo_gym.sandbox.providers.base import SandboxSpec
from tests.unit_tests.test_sandbox_checkpoint import FakeSnapshotProvider


SEEDS = range(int(os.environ.get("NEMO_GYM_SANDBOX_SEQUENCE_SEEDS", "40")))
SPEC = SandboxSpec(image="img")
CHAOS_OPS = {"pause", "resume", "create", "connect", "close"}


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
        entry = live._entries[session_id]
        box = provider.boxes[entry.sandbox.handle.sandbox_id]
        # A live session's sandbox is running, or paused by a checkpoint that ensure_running will resume.
        assert box["state"] in ("running", "paused"), (session_id, box["state"])
        if box["state"] == "paused":
            assert entry.paused_by_checkpoint, f"{session_id} is paused and forgotten"
        assert box["fs"] == expected_files[session_id], session_id
    assert set(live.session_ids) == set(expected_files)


def check_committed_invariants(provider: ChaosProvider, committed: list[dict[str, Any]]) -> None:
    for exported in committed:
        for session_id, state in _states(exported).items():
            assert state.snapshot_id in provider.snapshots, (session_id, state.snapshot_id)


@pytest.mark.parametrize("seed", SEEDS)
async def test_random_sequences_keep_the_sandbox_invariants(seed: int) -> None:
    rng = random.Random(seed)
    provider = ChaosProvider(rng, fault_rate=rng.choice([0.0, 0.1, 0.25]))
    live = SandboxSessionCheckpointer(provider, parallelism=rng.choice([1, 2, 4]))
    expected_files: dict[str, list[str]] = {}
    # Each committed checkpoint: the exported states, and the files each session had when it was exported.
    committed: list[dict[str, Any]] = []
    committed_files: list[dict[str, list[str]]] = []
    log: list[str] = []
    counter = 0

    for _ in range(rng.randint(15, 50)):
        op = rng.choices(["tool", "create", "export", "stop", "crash_restore"], weights=[5, 2, 2, 1, 1])[0]
        try:
            if op == "create" or (op == "tool" and not expected_files):
                counter += 1
                session_id = f"s{counter}"
                await live.create(session_id, SPEC)
                expected_files[session_id] = []
                log.append(f"create {session_id}")
            elif op == "tool":
                session_id = rng.choice(sorted(expected_files))
                sandbox = await live.ensure_running(session_id)
                counter += 1
                await sandbox.exec(f"{session_id}: line {counter}")
                expected_files[session_id].append(f"{session_id}: line {counter}")
                log.append(f"tool {session_id}")
            elif op == "export":
                chosen = [s for s in sorted(expected_files) if rng.random() < 0.7] + (
                    ["gone"] if rng.random() < 0.3 else []
                )
                snapshot_of = dict(expected_files)
                exported = await live.export(chosen)
                assert set(exported) == set(chosen) & set(expected_files)
                for session_id, state in _states(exported).items():
                    box = provider.boxes[state.descriptor["sandbox_id"]]
                    assert box["state"] == "paused"
                    assert provider.snapshots[state.snapshot_id]["fs"] == snapshot_of[session_id]
                committed.append(exported)
                committed_files.append({s: list(snapshot_of[s]) for s in exported})
                log.append(f"export {sorted(exported)}")
            elif op == "stop" and expected_files:
                session_id = rng.choice(sorted(expected_files))
                await live.stop(session_id)
                assert session_id not in live
                expected_files.pop(session_id)
                log.append(f"stop {session_id}")
            elif op == "crash_restore" and committed:
                index = rng.randrange(len(committed))
                before = provider._counter
                fresh = SandboxSessionCheckpointer(provider, parallelism=rng.choice([1, 2, 4]))
                try:
                    await fresh.restore(committed[index])
                except SandboxCheckpointError:
                    # A failed restore installs nothing and issues a stop for every sandbox it created; the stop
                    # itself may be one of the injected failures, which the orphan sweep is for.
                    assert fresh.session_ids == []
                    for sandbox_id, box in provider.boxes.items():
                        if int(sandbox_id.split("-")[1]) > before:
                            assert box["state"] == "stopped" or ("close", sandbox_id) in provider.calls, (
                                f"restore leaked {sandbox_id} without trying to stop it"
                            )
                    log.append(f"crash_restore #{index} failed")
                    # The crashed process is gone either way; its rollouts restart from input.
                    live, expected_files = fresh, {}
                else:
                    for session_id, state in _states(committed[index]).items():
                        box = provider.boxes[fresh.get(session_id).handle.sandbox_id]
                        assert box["state"] == "running"
                        assert box["fs"] == provider.snapshots[state.snapshot_id]["fs"]
                    live = fresh
                    expected_files = {s: list(files) for s, files in committed_files[index].items()}
                    log.append(f"crash_restore #{index}")
        except (SandboxCheckpointError, RuntimeError, KeyError) as error:
            log.append(f"{op} failed: {error}")
        check_live_invariants(provider, live, expected_files)
        check_committed_invariants(provider, committed)

    # Every committed checkpoint restores into a fresh process, with the files its sessions had at the checkpoint.
    provider.armed = False
    for exported, files in zip(committed, committed_files):
        fresh = SandboxSessionCheckpointer(provider)
        await fresh.restore(exported)
        for session_id in exported:
            sandbox = await fresh.ensure_running(session_id)
            assert provider.boxes[sandbox.handle.sandbox_id]["fs"] == files[session_id], (seed, log)
