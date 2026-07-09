# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests: capture store ordering/spill/manifest + builder correctness.

Run: pytest tests/ -q   (from the Gym repo root with the bundle overlaid)
"""

from __future__ import annotations

import threading

import pytest

from nemo_gym.observability.capture_reader import LocalCaptureReader
from nemo_gym.observability.capture_store import LocalJsonlCaptureStore
from nemo_gym.observability.records import ModelCallRecord, TokenEntry, record_from_envelope
from nemo_gym.trajectory.registry import get_builder, list_builders, register_builder
from nemo_gym.trajectory.strategies import BuildOutput


RID = "toy.00000.r00.a0"


def _step(i: int) -> ModelCallRecord:
    return ModelCallRecord(
        rollout_id=RID,
        request_id=f"r{i}",
        model="m",
        tokens_in=10 * i,
        tokens_out=i,
        request={"i": i},
        response={"output": [], "i": i},
    )


def test_append_is_totally_ordered_across_threads(tmp_path):
    store = LocalJsonlCaptureStore(str(tmp_path), fsync_per_record=False)
    n_threads, per_thread = 8, 25

    def writer(t):
        for i in range(per_thread):
            store.append(_step(t * 1000 + i))

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    envs = list(store.read_envelopes(RID))
    seqs = [e["seq"] for e in envs]
    assert len(envs) == n_threads * per_thread
    assert sorted(seqs) == list(range(n_threads * per_thread))  # dense, unique
    assert all(isinstance(record_from_envelope(e), ModelCallRecord) for e in envs)


def test_blob_spill_and_kind_dispatch(tmp_path):
    store = LocalJsonlCaptureStore(str(tmp_path))
    big = "x" * (70 * 1024)
    store.append(ModelCallRecord(rollout_id=RID, request_id="r0", request={"screenshot": big}, response={}))
    store.append(
        TokenEntry(
            rollout_id=RID,
            request_id="r0",
            prompt_token_ids=[1, 2],
            generation_token_ids=[3],
            generation_log_probs=[-0.1],
        )
    )
    envs = list(store.read_envelopes(RID))
    assert [e["kind"] for e in envs] == ["model_call", "tokens"]
    blob_ref = envs[0]["data"]["request"]["screenshot"]
    assert set(blob_ref) == {"$blob", "bytes"}
    assert store.resolve_blob(blob_ref["$blob"]).decode() == big
    only_tokens = LocalCaptureReader(store).records(RID, kinds={"tokens"})
    assert len(only_tokens) == 1 and isinstance(only_tokens[0], TokenEntry)


def test_manifest_written_once(tmp_path):
    a = LocalJsonlCaptureStore(str(tmp_path), run_id="run-a")
    b = LocalJsonlCaptureStore(str(tmp_path), run_id="ignored")
    assert a.manifest == b.manifest and a.manifest["run_id"] == "run-a"


# ---------------------------------------------------------------------------
# Strategy fixtures: token-faithful synthetic rollouts. Convention: prompt of
# call k+1 literally extends (prompt+gen [+interstitial]) of its parent — the
# property real engines give us and prefix merging relies on.
# ---------------------------------------------------------------------------


def _entry(seq: int, prompt: list[int], gen: list[int]) -> TokenEntry:
    return TokenEntry(
        rollout_id=RID,
        request_id=f"r{seq}",
        seq=seq,
        prompt_token_ids=prompt,
        generation_token_ids=gen,
        generation_log_probs=[-0.1] * len(gen),
    )


P0, G1, T1, G2, T2, G3 = [1, 2, 3], [10, 11], [20], [12, 13], [21, 22], [14]
SX, G4 = [30], [15, 16]


def linear_rollout() -> list[TokenEntry]:
    e1 = _entry(0, P0, G1)
    e2 = _entry(1, P0 + G1 + T1, G2)
    e3 = _entry(2, P0 + G1 + T1 + G2 + T2, G3)
    return [e1, e2, e3]


def branching_rollout() -> list[TokenEntry]:
    # e4 branches off after call 1 (sub-agent): prompt = P0+G1+SX
    return linear_rollout() + [_entry(3, P0 + G1 + SX, G4)]


def test_per_request_makes_one_chain_per_call():
    out = get_builder("per_request")(branching_rollout())
    assert len(out.chains) == 4 and not out.quarantined
    ids, mask, lps, _ = out.chains[0].flatten()
    assert ids == P0 + G1 and mask == [0] * len(P0) + [1] * len(G1)
    assert all(lp is not None for lp, m in zip(lps, mask) if m == 1)


def test_prefix_merging_linear_single_chain():
    out = get_builder("prefix_merging")(linear_rollout())
    assert len(out.chains) == 1 and out.chains[0].chain_id == "main"
    ids, mask, _, _ = out.chains[0].flatten()
    assert ids == P0 + G1 + T1 + G2 + T2 + G3
    # provenance mask: generated 1, prompt/interstitial 0
    expected = [0] * len(P0) + [1] * len(G1) + [0] * len(T1) + [1] * len(G2) + [0] * len(T2) + [1] * len(G3)
    assert mask == expected


def test_prefix_merging_forest_and_main_selection():
    out = get_builder("prefix_merging")(branching_rollout())
    by_id = {c.chain_id: c for c in out.chains}
    assert set(by_id) == {"main", "branch-0"}
    main_ids, _, _, _ = by_id["main"].flatten()
    branch_ids, branch_mask, _, _ = by_id["branch-0"].flatten()
    assert main_ids == P0 + G1 + T1 + G2 + T2 + G3
    assert branch_ids == P0 + G1 + SX + G4
    assert branch_mask == [0] * len(P0) + [1] * len(G1) + [0] * len(SX) + [1] * len(G4)


def test_compaction_splits_new_root():
    e1 = _entry(0, P0, G1)
    rewritten = _entry(1, [1, 2, 99], G2)  # shares a partial prefix only
    out = get_builder("prefix_merging")([e1, rewritten])
    assert len(out.chains) == 2
    assert out.notes["roots"] == 2 and out.notes["compaction_roots"] == 1


def test_ambiguous_identical_prefix_quarantined():
    e1 = _entry(0, P0, G1)
    e1_dup = _entry(1, P0, G1)  # identical cumulative sequence
    child = _entry(2, P0 + G1 + [40], [50])  # two identical candidate parents
    out = get_builder("prefix_merging")([e1, e1_dup, child])
    assert child.request_id in out.quarantined
    for c in out.chains:  # quarantined subtree not emitted
        assert all(link.entry.request_id != child.request_id for link in c.links)


def test_equivalence_on_append_only_rollouts():
    """CI check from the spec: on append-only rollouts, the mask-1 token
    multiset (with logprobs) is identical across strategies."""
    entries = linear_rollout()

    def mask1(out):
        pairs = []
        for c in out.chains:
            ids, mask, lps, _ = c.flatten()
            pairs += [(i, lp) for i, m, lp in zip(ids, mask, lps) if m == 1]
        return sorted(pairs)

    assert mask1(get_builder("per_request")(entries)) == mask1(get_builder("prefix_merging")(entries))


def test_logprob_length_mismatch_raises():
    bad = TokenEntry(
        rollout_id=RID,
        request_id="r0",
        seq=0,
        prompt_token_ids=P0,
        generation_token_ids=G1,
        generation_log_probs=[-0.1],
    )
    out = get_builder("per_request")([bad])
    with pytest.raises(ValueError, match="mismatch"):
        out.chains[0].flatten()


def test_builder_plugin_registry():
    """Users can register custom trajectory builders and look them up by name."""
    assert {"per_request", "prefix_merging"} <= set(list_builders())

    def custom(entries):
        return BuildOutput(chains=[], notes={"builder": "custom_test"})

    register_builder("custom_test", custom)
    assert "custom_test" in list_builders()
    assert get_builder("custom_test") is custom
    with pytest.raises(ValueError, match="already registered"):
        register_builder("per_request", custom)
    with pytest.raises(KeyError, match="unknown trajectory builder"):
        get_builder("does_not_exist")
