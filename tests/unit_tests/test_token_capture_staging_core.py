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
"""Staging core (gate-authoritative capture): wire records, digest, purity.

Ported from the fork-era ``test_token_capture_gate_primitives.py``; grows with
the staging subpackage (lineage, rebuild, and conformance sections land with
their modules). The digest vectors are frozen against the sync prototype's
implementation and must never change.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.token_id_capture.staging.digest import (
    EMPTY_PREFIX_HASH,
    build_staging_delta,
    compute_staging_digest,
    encode_token_ids,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.rebuild import (
    RebuildError,
    linearize,
    snapshots_to_entries,
)
from nemo_gym.token_id_capture.staging.records import (
    SCHEMA_VERSION,
    CallRecord,
    CommitCoords,
    StagedCallRecord,
    StagedCallSnapshot,
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
# records: wire-schema sanity
# ---------------------------------------------------------------------------


def test_staging_key_and_record_roundtrip() -> None:
    record = StagedCallRecord(
        rollout_id="r0",
        call_id="c0",
        prev_len=0,
        new_len=2,
        weight_version=1,
        digest="d",
        token_ids_delta=[1, 2],
        token_mask_delta=[0.0, 1.0],
        generation_logprobs_delta=[0.0, -0.5],
    )
    assert record.staging_key == staging_key("r0", "c0") == "r0/c0"
    assert record.schema_version == SCHEMA_VERSION
    assert StagedCallRecord.model_validate(record.model_dump()) == record


def test_records_reject_unknown_fields() -> None:
    with pytest.raises(Exception, match="extra|unexpected"):
        CommitCoords(
            rollout_id="r",
            call_id="c",
            parent_call_id=None,
            delta_len=1,
            cum_len=1,
            digest="d",
            staging_key="r/c",
            weight_version=0,
            not_a_field=True,
        )


# ---------------------------------------------------------------------------
# rebuild: inverse property, terminal-aware linearize over run_builder
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


def _manifest_for(rollout_id: str, snapshots: list[StagedCallSnapshot]) -> list[CallRecord]:
    return [
        CallRecord(
            call_id=s.call_id,
            parent_call_id=s.parent_call_id,
            delta_len=len(s.token_ids_delta),
            cum_len=s.prev_len + len(s.token_ids_delta),
            digest="d-" + s.call_id,
            staging_key=staging_key(rollout_id, s.call_id),
            weight_version=s.weight_version or 0,
        )
        for s in snapshots
    ]


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


def test_linearize_two_call_chain() -> None:
    snapshots = [
        _snapshot("c1", 0, [10, 11, 12], [13, 14], wv=4),
        _snapshot("c2", 5, [20, 21, 22], [23, 24], parent="c1", wv=4),
    ]
    row = linearize("g7_r0", snapshots, _manifest_for("g7_r0", snapshots), terminal_hint="c2")
    assert row.call_ids == ["c1", "c2"]
    assert row.token_ids == [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    assert row.token_mask == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert row.logprobs == [0.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, -1.0, -1.0]
    assert row.prompt_len == 3
    assert row.weight_versions == [4, 4]


def test_linearize_terminal_hint_beats_token_mass() -> None:
    # c3 is a sub-agent fork that out-generates the main conversation; the
    # base builder's token-mass pick would deliver it. The sealed terminal
    # call is c2, so the terminal-aware selection must deliver c1 -> c2.
    snapshots = [
        _snapshot("c1", 0, [10, 11], [12], wv=1),
        _snapshot("c2", 3, [20], [21], parent="c1", wv=1),
        _snapshot("c3", 3, [30], [31, 32, 33, 34, 35], parent="c1", wv=1),
    ]
    manifest = _manifest_for("r", snapshots)
    row = linearize("r", snapshots, manifest, terminal_hint="c2")
    assert row.call_ids == ["c1", "c2"]
    # Without the hint the base's main-chain pick wins (and picks the fork).
    row_mass = linearize("r", snapshots, manifest)
    assert row_mass.call_ids == ["c1", "c3"]


def test_linearize_unresolved_final_retry_raises() -> None:
    # Two siblings with identical prompts and no children: a final-call retry
    # no later call can resolve. Training on either would be a guess.
    snapshots = [
        _snapshot("c1", 0, [10], [11], wv=1),
        _snapshot("c2a", 2, [12], [13], parent="c1", wv=1),
        _snapshot("c2b", 2, [12], [14], parent="c1", wv=1),
    ]
    with pytest.raises(RebuildError, match="unresolved retry"):
        linearize("r", snapshots, _manifest_for("r", snapshots), terminal_hint="c2a")


def test_linearize_terminal_missing_from_chains() -> None:
    snapshots = [_snapshot("c1", 0, [10], [11], wv=1)]
    with pytest.raises(RebuildError, match="not the leaf"):
        linearize("r", snapshots, _manifest_for("r", snapshots), terminal_hint="ghost")


def test_linearize_snapshot_manifest_mismatch() -> None:
    snapshots = [
        _snapshot("c1", 0, [10], [11], wv=1),
        _snapshot("c2", 2, [12], [13], parent="c1", wv=1),
    ]
    manifest = _manifest_for("r", snapshots)
    with pytest.raises(RebuildError, match="manifest of"):
        linearize("r", snapshots[:-1], manifest)
    with pytest.raises(RebuildError, match="diverges"):
        linearize("r", snapshots, list(reversed(manifest)))


# ---------------------------------------------------------------------------
# purity: the staging core must import with no heavy dependencies
# ---------------------------------------------------------------------------


def _staging_modules() -> list[str]:
    """Every module in the ``staging`` subpackage, discovered from disk so a
    new core module is covered automatically (no hand-maintained list)."""
    import nemo_gym.token_id_capture.staging as staging

    root = Path(staging.__file__).parent
    modules = ["nemo_gym.token_id_capture.staging"]
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).with_suffix("")
        parts = [p for p in rel.parts if p != "__init__"]
        if parts:
            modules.append(".".join(["nemo_gym.token_id_capture.staging", *parts]))
    return modules


CORE_MODULES = _staging_modules()

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
