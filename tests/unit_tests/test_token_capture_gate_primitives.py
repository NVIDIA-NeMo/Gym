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
"""Gate-authoritative capture primitives (S1): digest, lineage, rebuild, conformance.

The lineage state machine is fully exercised here, two stages before it is
hosted behind HTTP; the digest vectors are frozen against the sync prototype's
implementation; the conformance fixtures are the same golden sequences every
framework sink/source and (in S3) the gate itself must reproduce byte-exactly.
"""

import subprocess
import sys

import pytest

from nemo_gym.token_id_capture.conformance import (
    build_fixture_artifacts,
    fixture_names,
    load_fixture,
    run_lineage_conformance,
    run_sink_source_conformance,
)
from nemo_gym.token_id_capture.digest import (
    EMPTY_PREFIX_HASH,
    build_staging_delta,
    compute_staging_digest,
    encode_token_ids,
    hash_token_ids,
)
from nemo_gym.token_id_capture.lineage import (
    DuplicateRolloutError,
    LineageRegistry,
    LineageStateError,
    RolloutLineage,
    UnknownCallError,
    UnknownRolloutError,
)
from nemo_gym.token_id_capture.rebuild import (
    RebuildError,
    linearize,
    main_chain_call_ids,
    snapshots_to_entries,
)
from nemo_gym.token_id_capture.records import (
    CommitCoords,
    StagedCallRecord,
    StagedCallSnapshot,
    StageResult,
    staging_key,
)


# ---------------------------------------------------------------------------
# digest: golden vectors frozen against the sync prototype (rollout_writer.py)
# ---------------------------------------------------------------------------


def test_digest_golden_vectors_match_prototype() -> None:
    # Verified byte-identical against the prototype donor implementation
    # (compute_staging_digest at pranav/tq_gym_prototype 05e0adfa0).
    assert (
        compute_staging_digest(
            rollout_id="g7_r0",
            call_id="c1",
            prev_len=0,
            token_ids_delta=[10, 11, 12, 13, 14],
            token_mask_delta=[0.0, 0.0, 0.0, 1.0, 1.0],
            logprobs_delta=[0.0, 0.0, 0.0, -0.125, -1.75],
        )
        == "6629f7662bf8fbc7de3ddb8c49048c8de9e8d2e8d63cfcfb84556b05c1323aa7"
    )
    assert (
        compute_staging_digest(
            rollout_id="g0_r1",
            call_id="a1",
            prev_len=0,
            token_ids_delta=[5, 6, 7, 8, 9],
            token_mask_delta=[0.0, 0.0, 0.0, 0.0, 1.0],
            logprobs_delta=[0.0, 0.0, 0.0, 0.0, -0.25],
        )
        == "f4cc7cf930dcc821fbc9b87da523f1b487a51cb65adc7cec2eefb97b5cd54884"
    )
    assert EMPTY_PREFIX_HASH == hash_token_ids([])
    assert encode_token_ids([1]) != encode_token_ids([1, 0])  # length-delimited


def test_digest_is_sensitive_to_every_field() -> None:
    base = dict(
        rollout_id="r",
        call_id="c",
        prev_len=3,
        token_ids_delta=[1, 2],
        token_mask_delta=[0.0, 1.0],
        logprobs_delta=[0.0, -1.0],
    )
    reference = compute_staging_digest(**base)
    for mutation in (
        {"rollout_id": "r2"},
        {"call_id": "c2"},
        {"prev_len": 4},
        {"token_ids_delta": [1, 3]},
        {"token_mask_delta": [1.0, 1.0]},
        {"logprobs_delta": [0.0, -1.0000001]},
        {"logprobs_delta": [-0.0, -1.0]},  # -0.0 vs 0.0 must not alias
    ):
        assert compute_staging_digest(**{**base, **mutation}) != reference


def test_encode_token_ids_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        encode_token_ids([-1])


def test_build_staging_delta_layout_and_errors() -> None:
    ids, mask, lps = build_staging_delta(
        prompt_token_ids=[1, 2, 3, 4],
        generated_token_ids=[5, 6],
        generated_logprobs=[-0.5, -0.25],
        prev_len=2,
    )
    assert ids == [3, 4, 5, 6]
    assert mask == [0.0, 0.0, 1.0, 1.0]
    assert lps == [0.0, 0.0, -0.5, -0.25]
    with pytest.raises(ValueError, match="outside prompt length"):
        build_staging_delta(prompt_token_ids=[1], generated_token_ids=[2], generated_logprobs=[-1.0], prev_len=5)
    with pytest.raises(ValueError, match="lengths differ"):
        build_staging_delta(prompt_token_ids=[1], generated_token_ids=[2], generated_logprobs=[], prev_len=0)
    with pytest.raises(ValueError, match="at least one token"):
        build_staging_delta(prompt_token_ids=[1], generated_token_ids=[], generated_logprobs=[], prev_len=1)


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
# rebuild: inverse property, forest semantics, linearize
# ---------------------------------------------------------------------------


def _snapshot(
    call_id: str, prev_len: int, carry: list[int], gen: list[int], parent: str | None = None, wv: int | None = None
) -> StagedCallSnapshot:
    return StagedCallSnapshot(
        call_id=call_id,
        prev_len=prev_len,
        token_ids_delta=carry + gen,
        token_mask_delta=[0.0] * len(carry) + [1.0] * len(gen),
        logprobs_delta=[0.0] * len(carry) + [-1.0] * len(gen),
        parent_call_id=parent,
        weight_version=wv,
    )


def test_rebuild_is_inverse_of_build_staging_delta() -> None:
    prompt = [1, 2, 3, 4, 5]
    generated = [6, 7]
    ids, mask, lps = build_staging_delta(
        prompt_token_ids=prompt, generated_token_ids=generated, generated_logprobs=[-0.5, -2.0], prev_len=0
    )
    root = StagedCallSnapshot(call_id="c1", prev_len=0, token_ids_delta=ids, token_mask_delta=mask, logprobs_delta=lps)
    prompt2 = prompt + generated + [8, 9]
    ids2, mask2, lps2 = build_staging_delta(
        prompt_token_ids=prompt2, generated_token_ids=[10], generated_logprobs=[-0.25], prev_len=7
    )
    child = StagedCallSnapshot(
        call_id="c2",
        prev_len=7,
        token_ids_delta=ids2,
        token_mask_delta=mask2,
        logprobs_delta=lps2,
        parent_call_id="c1",
    )
    entries = snapshots_to_entries("r", [root, child])
    assert entries[0].prompt_token_ids == prompt
    assert entries[0].generation_token_ids == generated
    assert entries[1].prompt_token_ids == prompt2
    assert entries[1].generation_token_ids == [10]
    assert entries[1].generation_log_probs == [-0.25]


def test_rebuild_error_paths() -> None:
    # parent precedes it in no snapshot
    with pytest.raises(RebuildError, match="precedes"):
        snapshots_to_entries("r", [_snapshot("c2", 5, [1], [2], parent="c1")])
    # prev_len != parent length
    ok_root = _snapshot("c1", 0, [1, 2], [3])
    with pytest.raises(RebuildError, match="parent length"):
        snapshots_to_entries("r", [ok_root, _snapshot("c2", 2, [4], [5], parent="c1")])
    # parentless with prev_len > 0
    with pytest.raises(RebuildError, match="self-contained"):
        snapshots_to_entries("r", [_snapshot("c1", 3, [1], [2])])
    # misaligned arrays
    bad = _snapshot("c1", 0, [1], [2])
    bad = bad.model_copy(update={"logprobs_delta": [0.0]})
    with pytest.raises(RebuildError, match="misaligned"):
        snapshots_to_entries("r", [bad])
    # non-contiguous mask
    weird = StagedCallSnapshot(
        call_id="c1",
        prev_len=0,
        token_ids_delta=[1, 2, 3],
        token_mask_delta=[0.0, 1.0, 0.0],
        logprobs_delta=[0.0, -1.0, 0.0],
    )
    with pytest.raises(RebuildError, match="prompt-carry"):
        snapshots_to_entries("r", [weird])


def test_linearize_main_chain_and_manifest_walk() -> None:
    fixture = load_fixture("worked_example")
    records, _, receipt, row = build_fixture_artifacts(fixture)
    assert row is not None
    assert row.call_ids == ["c1", "c2"]
    assert row.token_ids == [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    assert row.token_mask == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert row.prompt_len == 3
    assert row.weight_versions == [4, 4]
    # manifest walk errors
    with pytest.raises(RebuildError, match="not in the manifest"):
        main_chain_call_ids(receipt.manifest, "ghost")
    assert main_chain_call_ids([], None) == []
    # snapshot/manifest mismatch
    snapshots = [
        StagedCallSnapshot(
            call_id=record.call_id,
            prev_len=record.prev_len,
            token_ids_delta=list(record.token_ids_delta),
            token_mask_delta=list(record.token_mask_delta),
            logprobs_delta=list(record.generation_logprobs_delta),
            parent_call_id=record.parent_call_id,
        )
        for record in records
    ]
    with pytest.raises(RebuildError, match="do not match the manifest"):
        linearize("g7_r0", snapshots[:-1], receipt.manifest)
    with pytest.raises(NotImplementedError):
        linearize("g7_r0", snapshots, receipt.manifest, policy="everything")  # type: ignore[arg-type]


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


# ---------------------------------------------------------------------------
# purity: the capture core must import with no heavy dependencies
# ---------------------------------------------------------------------------

CORE_MODULES = [
    "nemo_gym.token_id_capture.records",
    "nemo_gym.token_id_capture.protocols",
    "nemo_gym.token_id_capture.digest",
    "nemo_gym.token_id_capture.lineage",
    "nemo_gym.token_id_capture.rebuild",
    "nemo_gym.token_id_capture.conformance.kit",
]

FORBIDDEN_IMPORTS = ("fastapi", "ray", "torch", "transfer_queue", "aiohttp")


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_module_purity(module: str) -> None:
    """The § 3.0 purity rule: core modules must be importable inside any
    framework's worker process, so importing one may not pull fastapi, ray,
    torch, TQ, or an HTTP client into the process."""
    code = (
        f"import {module}, sys; "
        f"bad = sorted(set(sys.modules) & set({FORBIDDEN_IMPORTS!r})); "
        "assert not bad, f'forbidden imports: {bad}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
