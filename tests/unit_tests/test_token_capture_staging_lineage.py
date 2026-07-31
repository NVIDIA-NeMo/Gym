# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Staging lineage custody + conformance kit (gate-authoritative capture).

Ported from the fork-era ``test_token_capture_gate_primitives.py``. The
lineage state machine is fully exercised standalone, before it is hosted
behind the gate; the conformance fixtures are the same golden sequences every
framework sink/source (and the gate itself) must reproduce byte-exactly.
"""

import pytest

from nemo_gym.token_id_capture.staging.conformance import (
    fixture_names,
    load_fixture,
    run_lineage_conformance,
    run_sink_source_conformance,
)
from nemo_gym.token_id_capture.staging.lineage import (
    DuplicateRolloutError,
    LineageRegistry,
    LineageStateError,
    RolloutLineage,
    UnknownCallError,
    UnknownRolloutError,
)
from nemo_gym.token_id_capture.staging.records import (
    CommitCoords,
    StagedCallRecord,
    StagedCallSnapshot,
    StageResult,
    staging_key,
)


# ---------------------------------------------------------------------------
# lineage: the pure state machine, standalone
# ---------------------------------------------------------------------------


def _commit(lineage: RolloutLineage, call_id: str, *, parent: str | None, delta: int, wv: int = 1) -> None:
    prev_len = 0 if parent is None else lineage.committed_parent_len(parent)
    lineage.commit(
        CommitCoords(
            rollout_id=lineage.rollout_id,
            call_id=call_id,
            parent_call_id=parent,
            delta_len=delta,
            cum_len=prev_len + delta,
            digest="d-" + call_id,
            staging_key=staging_key(lineage.rollout_id, call_id),
            weight_version=wv,
        ),
        now=0.0,
    )


def test_lineage_happy_path_manifest_order_and_terminal_default() -> None:
    lineage = RolloutLineage("r0", now=0.0)
    lineage.admit("c1", parent_call_id=None, mode="text", now=0.0)
    _commit(lineage, "c1", parent=None, delta=5)
    admitted = lineage.admit("c2", parent_call_id="c1", mode="token_in", now=1.0)
    assert admitted.prev_len == 5
    _commit(lineage, "c2", parent="c1", delta=5)
    receipt = lineage.seal(reward=1.0, now=2.0)
    assert [record.call_id for record in receipt.manifest] == ["c1", "c2"]
    assert receipt.terminal_call_id == "c2"
    assert receipt.manifest[1].cum_len == 10
    assert not receipt.capture_poisoned


def test_lineage_fork_topology_and_explicit_terminal() -> None:
    lineage = RolloutLineage("r1", now=0.0)
    lineage.admit("c1", parent_call_id=None, mode="text", now=0.0)
    _commit(lineage, "c1", parent=None, delta=5)
    lineage.admit("c2", parent_call_id="c1", mode="token_in", now=0.0)
    lineage.admit("c3", parent_call_id="c1", mode="token_in", now=0.0)  # fork off interior
    _commit(lineage, "c2", parent="c1", delta=5)
    _commit(lineage, "c3", parent="c1", delta=4)
    lineage.admit("c4", parent_call_id=None, mode="text", now=0.0)  # fingerprint-miss root
    _commit(lineage, "c4", parent=None, delta=9)
    receipt = lineage.seal(reward=1.0, terminal_call_id="c2", now=1.0)
    assert [r.call_id for r in receipt.manifest] == ["c1", "c2", "c3", "c4"]
    assert receipt.terminal_call_id == "c2"
    assert [r.parent_call_id for r in receipt.manifest] == [None, "c1", "c1", None]


def test_lineage_admission_rules() -> None:
    lineage = RolloutLineage("r2", now=0.0)
    # token-in with no parent is a contract violation (the gate falls back to text).
    with pytest.raises(LineageStateError, match="requires a committed parent"):
        lineage.admit("c1", parent_call_id=None, mode="token_in", now=0.0)
    lineage.admit("c1", parent_call_id=None, mode="text", now=0.0)
    # duplicate call id
    with pytest.raises(LineageStateError, match="already admitted"):
        lineage.admit("c1", parent_call_id=None, mode="text", now=0.0)
    # child of an uncommitted parent cannot be admitted (fail-closed ordering)
    with pytest.raises(LineageStateError, match="not committed"):
        lineage.admit("c2", parent_call_id="c1", mode="token_in", now=0.0)
    # child of an unknown parent
    with pytest.raises(UnknownCallError):
        lineage.admit("c3", parent_call_id="ghost", mode="token_in", now=0.0)


def test_lineage_commit_validations() -> None:
    lineage = RolloutLineage("r3", now=0.0)
    lineage.admit("c1", parent_call_id=None, mode="text", now=0.0)
    # length chaining
    with pytest.raises(LineageStateError, match="do not chain"):
        lineage.commit(
            CommitCoords(
                rollout_id="r3",
                call_id="c1",
                parent_call_id=None,
                delta_len=5,
                cum_len=6,
                digest="d",
                staging_key="r3/c1",
                weight_version=1,
            )
        )
    # wrong rollout
    with pytest.raises(LineageStateError, match="ingested at rollout"):
        lineage.commit(
            CommitCoords(
                rollout_id="other",
                call_id="c1",
                parent_call_id=None,
                delta_len=5,
                cum_len=5,
                digest="d",
                staging_key="o/c1",
                weight_version=1,
            )
        )
    # empty delta
    with pytest.raises(LineageStateError, match="empty delta"):
        lineage.commit(
            CommitCoords(
                rollout_id="r3",
                call_id="c1",
                parent_call_id=None,
                delta_len=0,
                cum_len=0,
                digest="d",
                staging_key="r3/c1",
                weight_version=1,
            )
        )
    # parent mismatch vs admission
    with pytest.raises(LineageStateError, match="does not match admitted parent"):
        lineage.commit(
            CommitCoords(
                rollout_id="r3",
                call_id="c1",
                parent_call_id="c0",
                delta_len=5,
                cum_len=5,
                digest="d",
                staging_key="r3/c1",
                weight_version=1,
            )
        )
    _commit(lineage, "c1", parent=None, delta=5)
    # double commit
    with pytest.raises(LineageStateError, match="cannot commit"):
        _commit(lineage, "c1", parent=None, delta=5)
    # unknown call
    with pytest.raises(UnknownCallError):
        lineage.commit(
            CommitCoords(
                rollout_id="r3",
                call_id="nope",
                parent_call_id=None,
                delta_len=1,
                cum_len=1,
                digest="d",
                staging_key="r3/nope",
                weight_version=1,
            )
        )


def test_lineage_capture_failure_poisons_rollout_but_serves_on() -> None:
    lineage = RolloutLineage("r4", now=0.0)
    lineage.admit("c1", parent_call_id=None, mode="text", now=0.0)
    _commit(lineage, "c1", parent=None, delta=3)
    lineage.admit("c2", parent_call_id="c1", mode="token_in", now=0.0)
    lineage.commit(
        CommitCoords(
            rollout_id="r4",
            call_id="c2",
            parent_call_id="c1",
            delta_len=0,
            cum_len=3,
            digest="",
            staging_key="r4/c2",
            weight_version=1,
            disposition="capture_failed",
        )
    )
    assert lineage.capture_poisoned
    assert lineage.call_state("c2") == "failed"
    receipt = lineage.seal(reward=0.0, now=1.0)
    assert receipt.capture_poisoned
    assert [r.call_id for r in receipt.manifest] == ["c1"]


def test_lineage_fail_call_and_terminal_validation() -> None:
    lineage = RolloutLineage("r5", now=0.0)
    lineage.admit("c1", parent_call_id=None, mode="text", now=0.0)
    lineage.fail_call("c1", reason="call_timeout", now=0.5)
    assert lineage.call_state("c1") == "failed"
    with pytest.raises(LineageStateError, match="cannot fail"):
        lineage.admit("c2", parent_call_id=None, mode="text", now=0.6)
        _commit(lineage, "c2", parent=None, delta=2)
        lineage.fail_call("c2", reason="late")
    with pytest.raises(LineageStateError, match="not committed"):
        lineage.seal(reward=0.0, terminal_call_id="c1", now=1.0)
    receipt = lineage.seal(reward=0.0, now=1.0)
    assert receipt.terminal_call_id == "c2"


def test_lineage_sealed_and_failed_rollouts_reject_operations() -> None:
    lineage = RolloutLineage("r6", now=0.0)
    lineage.admit("c1", parent_call_id=None, mode="text", now=0.0)
    _commit(lineage, "c1", parent=None, delta=2)
    lineage.seal(reward=0.0, now=1.0)
    with pytest.raises(LineageStateError, match="sealed"):
        lineage.admit("c2", parent_call_id=None, mode="text", now=1.5)
    failed = RolloutLineage("r7", now=0.0)
    failed.fail(reason="dispatch_cancelled", now=0.5)
    with pytest.raises(LineageStateError, match="failed"):
        failed.admit("c1", parent_call_id=None, mode="text", now=1.0)
    # empty seal: no committed calls -> empty manifest, no terminal
    empty = RolloutLineage("r8", now=0.0)
    receipt = empty.seal(reward=None, now=0.1)
    assert receipt.manifest == [] and receipt.terminal_call_id is None


def test_registry_create_only_ttl_and_seal_drop() -> None:
    registry = LineageRegistry(registration_ttl_s=10.0)
    registry.register("r0", now=0.0)
    with pytest.raises(DuplicateRolloutError):
        registry.register("r0", now=1.0)  # NaN-retry re-dispatch fails loudly
    with pytest.raises(UnknownRolloutError):
        registry.get("ghost")
    lineage = registry.get("r0")
    lineage.admit("c1", parent_call_id=None, mode="text", now=1.0)
    _commit(lineage, "c1", parent=None, delta=2)
    receipt = registry.seal("r0", reward=1.0, now=2.0)
    assert receipt.rollout_id == "r0" and "r0" not in registry
    # fail_rollout drops state
    registry.register("r1", now=0.0)
    registry.fail_rollout("r1", reason="shutdown", now=1.0)
    assert "r1" not in registry
    # TTL sweep uses last-touch time
    registry.register("r2", now=100.0)
    registry.register("r3", now=100.0)
    registry.get("r3").admit("c1", parent_call_id=None, mode="text", now=109.0)
    assert registry.expire_stale(now=111.0) == ["r2"]
    assert "r3" in registry and len(registry) == 1
    with pytest.raises(ValueError, match="positive"):
        LineageRegistry(registration_ttl_s=0.0)


# ---------------------------------------------------------------------------
# conformance: fixtures + a reference in-memory sink/source
# ---------------------------------------------------------------------------


class _MemorySink:
    def __init__(self) -> None:
        self.rows: dict[str, StagedCallRecord] = {}

    def stage(self, record: StagedCallRecord) -> StageResult:
        self.rows[record.staging_key] = record
        return StageResult(ok=True, staging_key=record.staging_key)


class _MemorySource:
    def __init__(self, sink: _MemorySink) -> None:
        self._sink = sink

    def fetch(self, staging_keys: list[str]) -> list[StagedCallSnapshot]:
        out = []
        for key in staging_keys:
            record = self._sink.rows[key]
            out.append(
                StagedCallSnapshot(
                    call_id=record.call_id,
                    prev_len=record.prev_len,
                    token_ids_delta=list(record.token_ids_delta),
                    token_mask_delta=list(record.token_mask_delta),
                    logprobs_delta=list(record.generation_logprobs_delta),
                    weight_version=record.weight_version,
                )
            )
        return out


def test_all_fixtures_pass_lineage_conformance() -> None:
    names = fixture_names()
    assert set(names) >= {"worked_example", "single_call", "capture_failed_call", "mixed_weight_versions"}
    for name in names:
        run_lineage_conformance(load_fixture(name))


def test_memory_sink_source_passes_conformance() -> None:
    for name in fixture_names():
        sink = _MemorySink()
        run_sink_source_conformance(load_fixture(name), sink, _MemorySource(sink))


def test_conformance_catches_a_corrupted_store() -> None:
    class _FlippingSource(_MemorySource):
        def fetch(self, staging_keys: list[str]) -> list[StagedCallSnapshot]:
            snapshots = super().fetch(staging_keys)
            snapshots[0].token_ids_delta[0] += 1
            return snapshots

    sink = _MemorySink()
    with pytest.raises(AssertionError, match="digest"):
        run_sink_source_conformance(load_fixture("single_call"), sink, _FlippingSource(sink))
