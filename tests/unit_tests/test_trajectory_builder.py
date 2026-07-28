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
"""Trajectory builder: chaining, loss masks, and the NeMo-RL projection."""

import pytest

from nemo_gym.token_id_capture import (
    Trajectory,
    assert_nemo_rl_contiguity,
    build_trajectories,
    classify_side_calls,
    compute_digest,
    prefix_merging,
    project_main_chain_response,
    stamp_lineage,
    token_id_capture_dirs_from_config,
    trajectories_for_rollout,
)
from nemo_gym.token_id_capture.records import TokenEntry
from nemo_gym.token_id_capture.store import TokenCaptureStore


def _entry(mcid, prompt, gen, lp=None):
    return TokenEntry(
        rollout_id="t0-r0",
        model_call_id=mcid,
        model="m",
        prompt_token_ids=prompt,
        generation_token_ids=gen,
        generation_log_probs=lp if lp is not None else [-0.1] * len(gen),
    )


# An append-only 3-call rollout: each call's prompt extends the prior prompt+generation
# plus interstitial tokens (tool output / new user turn).
CALL1 = _entry("c1", [1, 2, 3], [10, 11])
CALL2 = _entry("c2", [1, 2, 3, 10, 11, 4, 5], [12])
CALL3 = _entry("c3", [1, 2, 3, 10, 11, 4, 5, 12, 6], [13, 14])
APPEND_ONLY = [CALL1, CALL2, CALL3]


def test_prefix_merging_builds_one_contiguous_main_chain():
    trajs = build_trajectories("t0-r0", APPEND_ONLY, builder="prefix_merging", reward=1.0)
    assert len(trajs) == 1
    t = trajs[0]
    assert t.chain_id == "main"
    # The flat stream is the final cumulative sequence.
    assert t.token_ids == [1, 2, 3, 10, 11, 4, 5, 12, 6, 13, 14]
    # Generated tokens are masked 1, everything re-fed to a prompt is masked 0.
    assert t.loss_mask == [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    # Log probs are present exactly at the generated positions.
    assert [lp is not None for lp in t.log_probs] == [bool(m) for m in t.loss_mask]
    assert t.reward == 1.0
    assert t.provenance["n_calls"] == 3


def test_order_independent():
    import random

    shuffled = list(APPEND_ONLY)
    random.Random(0).shuffle(shuffled)
    a = build_trajectories("t0-r0", APPEND_ONLY, builder="prefix_merging")[0]
    b = build_trajectories("t0-r0", shuffled, builder="prefix_merging")[0]
    assert a.token_ids == b.token_ids and a.loss_mask == b.loss_mask


def test_per_request_marks_the_same_generated_tokens():
    # Both builders must agree on which tokens were generated (mask 1).
    def generated(trajs: list[Trajectory]):
        out = []
        for t in trajs:
            out += [tid for tid, m in zip(t.token_ids, t.loss_mask) if m == 1]
        return sorted(out)

    merged = build_trajectories("t0-r0", APPEND_ONLY, builder="prefix_merging")
    per_req = build_trajectories("t0-r0", APPEND_ONLY, builder="per_request")
    assert len(per_req) == 3
    assert generated(merged) == generated(per_req) == sorted([10, 11, 12, 13, 14])


def test_projection_is_nemo_rl_contiguous():
    out = prefix_merging(APPEND_ONLY)
    response = project_main_chain_response("t0-r0", out, model="m")
    assert [len(i["prompt_token_ids"]) for i in response["output"]] == [3, 7, 9]
    assert response["usage"] == {"input_tokens": 3, "output_tokens": 5}
    assert_nemo_rl_contiguity(response)  # must not raise


def test_contiguity_assert_catches_a_gap():
    broken = {
        "output": [
            {"type": "message", "prompt_token_ids": [1, 2, 3], "generation_token_ids": [10]},
            # prompt does not extend [1,2,3,10]:
            {"type": "message", "prompt_token_ids": [1, 2, 3, 99], "generation_token_ids": [11]},
        ]
    }
    with pytest.raises(AssertionError):
        assert_nemo_rl_contiguity(broken)


def _content_entry(mcid, prompt, gen, text):
    lp = [-0.1] * len(gen)
    return TokenEntry(
        rollout_id="t0-r0",
        model_call_id=mcid,
        model="m",
        prompt_token_ids=prompt,
        generation_token_ids=gen,
        generation_log_probs=lp,
        output_items=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
                "prompt_token_ids": prompt,
                "generation_token_ids": gen,
                "generation_log_probs": lp,
            }
        ],
    )


def test_projection_carries_content_and_stays_contiguous():
    entries = [
        _content_entry("c1", [1, 2, 3], [10, 11], "first turn"),
        _content_entry("c2", [1, 2, 3, 10, 11, 4, 5], [12], "second turn"),
    ]
    out = prefix_merging(entries)
    resp = project_main_chain_response("t0-r0", out, model="m")
    texts = [item["content"][0]["text"] for item in resp["output"]]
    assert texts == ["first turn", "second turn"]  # content preserved (not token-only)
    assert [len(i["prompt_token_ids"]) for i in resp["output"]] == [3, 7]
    assert_nemo_rl_contiguity(resp)  # prompts still contiguous with content attached


def test_projection_handles_content_only_leading_item():
    # A single call whose output is an assistant text message (no token fields) followed by a
    # tool call that carries the token fields -- the real shape when a model narrates before a
    # tool call. Usage must be read from the token-bearing item, not output[0].
    entry = TokenEntry(
        rollout_id="t0-r0",
        model_call_id="c1",
        model="m",
        prompt_token_ids=[1, 2, 3],
        generation_token_ids=[10, 11],
        generation_log_probs=[-0.1, -0.1],
        output_items=[
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "let me check"}]},
            {"type": "function_call", "name": "grep", "arguments": "{}", "call_id": "x"},
        ],
    )
    out = prefix_merging([entry])
    resp = project_main_chain_response("t0-r0", out, model="m")
    assert resp["output"][0]["type"] == "message"  # content-only leading item preserved
    assert "prompt_token_ids" not in resp["output"][0]
    assert resp["usage"] == {"input_tokens": 3, "output_tokens": 2}  # counts from the token-bearing item
    assert_nemo_rl_contiguity(resp)


def test_retry_sibling_is_dropped_and_main_chain_is_deterministic():
    # c2a and c2b are a retry pair (identical prompt, divergent generation). c3 extends c2a.
    c1 = _entry("c1", [1, 2, 3], [10, 11])
    c2a = _entry("c2a", [1, 2, 3, 10, 11, 4], [12])
    c2b = _entry("c2b", [1, 2, 3, 10, 11, 4], [99])
    c3 = _entry("c3", [1, 2, 3, 10, 11, 4, 12, 5], [13])
    out = prefix_merging([c1, c2a, c2b, c3])
    assert "c2b" in out.quarantined  # unextended retry sibling dropped
    main = next(c for c in out.chains if c.chain_id == "main")
    assert [link.entry.model_call_id for link in main.links] == ["c1", "c2a", "c3"]
    assert_nemo_rl_contiguity(project_main_chain_response("t0-r0", out))


def test_spans_mark_each_generation():
    trajs = build_trajectories("t0-r0", APPEND_ONLY, builder="prefix_merging")
    t = trajs[0]
    # One span per call, each covering exactly the mask-1 (generated) positions.
    assert [call for _, _, call in t.spans] == ["c1", "c2", "c3"]
    for start, end, _ in t.spans:
        assert all(t.loss_mask[i] == 1 for i in range(start, end))
    assert t.provenance["trained_token_fraction"] > 0


def test_consumer_reads_store_and_builds(tmp_path):
    # The co-located consumer: write the rollout's tokens, then build from the store files.
    store = TokenCaptureStore(tmp_path)
    for e in APPEND_ONLY:
        store.append(e.model_copy(update={"rollout_id": "t0-r0"}))
    dirs = token_id_capture_dirs_from_config({"token_id_capture_enabled": True, "token_id_capture_dir": str(tmp_path)})
    assert dirs == [tmp_path]
    merged = trajectories_for_rollout("t0-r0", dirs, builder="prefix_merging", reward=1.0)
    assert merged is not None
    assert merged["builder"] == "prefix_merging"
    assert len(merged["trajectories"]) == 1
    assert merged["trajectories"][0]["token_ids"] == [1, 2, 3, 10, 11, 4, 5, 12, 6, 13, 14]
    assert len(merged["nemo_rl_response"]["output"]) == 3


def test_reward_components_ride_the_trajectory(tmp_path):
    # Multi-objective (GDPO): the scalar reward and the named components both ride the trajectory,
    # copied from the verifier result. Token records never carry them.
    components = {"correctness": 1.0, "integer": 1.0, "format": 0.0}
    trajs = build_trajectories("t0-r0", APPEND_ONLY, reward=2.0, reward_components=components)
    assert trajs[0].reward == 2.0
    assert trajs[0].reward_components == components
    # Single-objective (GRPO) leaves components None, so the trainer path is unchanged.
    assert build_trajectories("t0-r0", APPEND_ONLY, reward=1.0)[0].reward_components is None

    store = TokenCaptureStore(tmp_path)
    for e in APPEND_ONLY:
        store.append(e)
    dirs = token_id_capture_dirs_from_config({"token_id_capture_enabled": True, "token_id_capture_dir": str(tmp_path)})
    merged = trajectories_for_rollout("t0-r0", dirs, reward=2.0, reward_components=components)
    assert merged["trajectories"][0]["reward_components"] == components


def test_consumer_noop_when_disabled_or_absent(tmp_path):
    assert token_id_capture_dirs_from_config({}) == []
    assert trajectories_for_rollout("t0-r0", []) is None
    # Enabled dir but no file for this rollout -> None (graceful no-op).
    dirs = token_id_capture_dirs_from_config({"token_id_capture_enabled": True, "token_id_capture_dir": str(tmp_path)})
    assert trajectories_for_rollout("missing", dirs) is None


def test_ambiguous_parents_are_quarantined():
    # Two roots with identical prompt+generation, then a call extending that shared
    # sequence: its parent is ambiguous, so the subtree is quarantined, not guessed.
    a = _entry("a", [1, 2], [7, 8])
    b = _entry("b", [1, 2], [7, 8])
    child = _entry("child", [1, 2, 7, 8, 9], [20])
    out = prefix_merging([a, b, child])
    assert "child" in out.quarantined
    # The quarantined child is excluded from every emitted chain.
    for chain in out.chains:
        assert all(link.entry.model_call_id != "child" for link in chain.links)


# --- side calls and chain selection -------------------------------------------


def test_a_short_side_call_does_not_replace_the_rollout():
    """A conversation-title call must not become the delivered chain.

    Claude Code generates a title (and probes quota) on a tiny prompt, while the
    rollout's first real call carries the full system prompt and tool
    definitions. Entries are processed by increasing prompt length, so the title
    call is the first root; selecting the main chain from the first root would
    deliver the title and relabel the whole rollout a branch. Nothing would
    error -- the trainer would receive a contiguous, token-bearing response
    containing a generated title, with the rollout's reward attached.
    """
    title = _entry("title", [9000, 9001], [7, 7, 7])
    real_1 = _entry("real1", list(range(100, 160)), [200, 201, 202, 203])
    real_2 = _entry("real2", list(range(100, 160)) + [200, 201, 202, 203, 500], [300, 301, 302])

    out = prefix_merging([title, real_1, real_2])
    main = next(c for c in out.chains if c.chain_id == "main")

    assert [link.entry.model_call_id for link in main.links] == ["real1", "real2"]
    assert out.notes["chains"] == 2
    # The dropped chain is reported rather than silently discarded.
    assert out.notes["generated_tokens_captured"] == 10
    assert out.notes["generated_tokens_delivered"] == 7
    assert out.notes["delivered_fraction"] == 0.7


def test_post_compaction_chain_is_reported_as_dropped():
    """A rewritten context starts a new root. Only one chain is delivered today,
    so what is left behind has to show up in the metrics."""
    call_1 = _entry("c1", [1, 2, 3], [4, 5])
    call_2 = _entry("c2", [1, 2, 3, 4, 5, 6], [7])
    # Compaction: the prompt no longer extends anything captured.
    call_3 = _entry("c3", [90, 91], [92, 93, 94, 95])

    out = prefix_merging([call_1, call_2, call_3])
    assert out.notes["chains"] == 2
    assert out.notes["generated_tokens_captured"] == 7
    assert out.notes["delivered_fraction"] < 1.0


# --- recorded parent links ----------------------------------------------------


def _with_lineage(entry, parent_call_id=None):
    stamp_lineage(entry, parent_call_id)
    return entry


def test_recorded_parent_link_resolves_a_final_call_retry_exactly():
    """Two siblings share a prompt and differ only in their generation.

    Prefix inference cannot tell which one the harness kept, because both are
    equally valid children. A recorded parent link on the next call names the
    survivor, so the other is provably unused rather than tie-broken.
    """
    root = _with_lineage(_entry("root", [1, 2], [3]))
    kept = _with_lineage(_entry("kept", [1, 2, 3, 4], [5]), parent_call_id="root")
    dropped = _with_lineage(_entry("dropped", [1, 2, 3, 4], [9]), parent_call_id="root")
    # The next call continued `kept`, and says so.
    nxt = _with_lineage(_entry("next", [1, 2, 3, 4, 5, 6], [7]), parent_call_id="kept")

    out = prefix_merging([root, kept, dropped, nxt])
    main = next(c for c in out.chains if c.chain_id == "main")
    assert [link.entry.model_call_id for link in main.links] == ["root", "kept", "next"]
    assert "dropped" in out.quarantined
    # Resolved, so nothing is flagged for masking.
    assert out.notes["unresolved_retries"] == []


def test_unresolvable_final_retry_is_flagged_not_silently_tie_broken():
    """A retry of the LAST call has no successor to name the survivor. Neither
    inference nor a parent link can resolve it, so it must be reported so the
    caller can mask the rollout instead of training on a generation the client
    may never have received."""
    root = _with_lineage(_entry("root", [1, 2], [3]))
    a = _with_lineage(_entry("a", [1, 2, 3, 4], [5]), parent_call_id="root")
    b = _with_lineage(_entry("b", [1, 2, 3, 4], [9]), parent_call_id="root")

    out = prefix_merging([root, a, b])
    assert sorted(out.notes["unresolved_retries"]) == ["a", "b"]


def test_a_stale_parent_link_fails_verification_and_falls_back():
    """A rerun that appended onto a previous attempt's records must not merge two
    attempts. The digest check catches the bad edge; the builder falls back to
    inference and reports that it did."""
    root = _with_lineage(_entry("root", [1, 2], [3]))
    child = _entry("child", [1, 2, 3, 4], [5])
    stamp_lineage(child, "root")
    # Corrupt the recorded parent's digest, as a stale record would.
    root.digest = compute_digest([42, 42, 42])

    out = prefix_merging([root, child])
    assert out.notes["parent_link_fallbacks"] == {"parent_digest_mismatch": 1}
    # Inference still finds the right parent, so the chain is intact.
    main = next(c for c in out.chains if c.chain_id == "main")
    assert [link.entry.model_call_id for link in main.links] == ["root", "child"]


def test_parent_link_and_inference_agree_on_a_clean_rollout():
    """Parity: with and without recorded links, the same rollout must stitch the
    same way. This is what makes the lineage fields safe to add before anything
    populates them."""
    plain = [
        _entry("c1", [1, 2, 3], [4, 5]),
        _entry("c2", [1, 2, 3, 4, 5, 6], [7]),
        _entry("c3", [1, 2, 3, 4, 5, 6, 7, 8], [9, 10]),
    ]
    linked = [
        _with_lineage(_entry("c1", [1, 2, 3], [4, 5])),
        _with_lineage(_entry("c2", [1, 2, 3, 4, 5, 6], [7]), parent_call_id="c1"),
        _with_lineage(_entry("c3", [1, 2, 3, 4, 5, 6, 7, 8], [9, 10]), parent_call_id="c2"),
    ]
    inferred = prefix_merging(plain)
    recorded = prefix_merging(linked)
    assert [c.flatten()[0] for c in inferred.chains] == [c.flatten()[0] for c in recorded.chains]
    assert [c.flatten()[1] for c in inferred.chains] == [c.flatten()[1] for c in recorded.chains]


def test_malformed_capture_masks_the_rollout_instead_of_raising(tmp_path):
    """The callers are a rollout-collection loop and NeMo-RL's training loop; an
    escaping exception there kills a whole step's batch rather than dropping one
    sample."""
    store = TokenCaptureStore(tmp_path)
    bad = _entry("c1", [1, 2, 3], [4, 5])
    bad.generation_log_probs = [-0.1]  # one log prob for two generated tokens
    store.append(bad)

    built = trajectories_for_rollout("t0-r0", [tmp_path])
    assert built is not None
    assert built["mask_sample"] is True
    assert built["nemo_rl_response"] is None
    assert "ValueError" in built["error"]


def test_incomplete_capture_masks_the_rollout(tmp_path):
    """A rollout that lost a call can still stitch into a clean-looking chain --
    it is just missing a turn. The marker is what makes that visible."""
    store = TokenCaptureStore(tmp_path)
    store.append(_entry("c1", [1, 2, 3], [4, 5]))
    store.mark_incomplete("t0-r0", "c2")

    built = trajectories_for_rollout("t0-r0", [tmp_path])
    assert built["mask_sample"] is True
    assert built["metrics"]["capture_incomplete"] is True


def test_clean_rollout_is_not_masked_and_reports_full_delivery(tmp_path):
    store = TokenCaptureStore(tmp_path)
    store.append(_entry("c1", [1, 2, 3], [4, 5]))
    store.append(_entry("c2", [1, 2, 3, 4, 5, 6], [7]))

    built = trajectories_for_rollout("t0-r0", [tmp_path])
    assert built["mask_sample"] is False
    assert built["metrics"]["delivered_fraction"] == 1.0
    assert built["metrics"]["quarantined_calls"] == 0


# --- side calls ---------------------------------------------------------------


def _sc(mcid, prompt, gen, requested_model=""):
    entry = _entry(mcid, prompt, gen)
    entry.requested_model = requested_model
    return entry


def test_side_calls_are_excluded_by_self_calibration():
    """No configuration: whichever model generated the most tokens is the policy
    model, and calls asking for a different one are side calls."""
    real_1 = _sc("r1", [100, 101], [1, 2, 3, 4, 5], "big-policy-model")
    real_2 = _sc("r2", [100, 101, 1, 2, 3, 4, 5, 6], [7, 8, 9], "big-policy-model")
    title = _sc("title", [9000], [42], "tiny-title-model")

    kept, excluded = classify_side_calls([real_1, real_2, title])

    assert [e.model_call_id for e in kept] == ["r1", "r2"]
    assert [e.model_call_id for e in excluded] == ["title"]


def test_side_calls_are_excluded_by_explicit_pattern():
    real = _sc("r1", [100], [1, 2, 3], "claude-sonnet-4")
    title = _sc("title", [9000], [42], "claude-3-5-haiku-20241022")

    kept, excluded = classify_side_calls([real, title], side_call_model_patterns=("haiku",))

    assert [e.model_call_id for e in kept] == ["r1"]
    assert [e.model_call_id for e in excluded] == ["title"]


def test_records_without_a_requested_model_are_all_kept():
    """Backward compatibility: records written before the field existed must not
    be reclassified as side calls."""
    entries = [_entry("c1", [1, 2, 3], [4, 5]), _entry("c2", [1, 2, 3, 4, 5, 6], [7])]
    kept, excluded = classify_side_calls(entries)
    assert len(kept) == 2 and excluded == []


def test_single_model_rollout_is_left_alone():
    entries = [_sc("c1", [1, 2, 3], [4, 5], "m"), _sc("c2", [1, 2, 3, 4, 5, 6], [7], "m")]
    kept, excluded = classify_side_calls(entries)
    assert len(kept) == 2 and excluded == []


def test_consumer_excludes_side_calls_and_reports_the_count(tmp_path):
    store = TokenCaptureStore(tmp_path)
    store.append(_sc("r1", [100, 101], [1, 2, 3, 4, 5], "policy"))
    store.append(_sc("r2", [100, 101, 1, 2, 3, 4, 5, 6], [7, 8, 9], "policy"))
    store.append(_sc("title", [9000], [42], "titler"))

    built = trajectories_for_rollout("t0-r0", [tmp_path])

    assert built["metrics"]["side_calls_excluded"] == 1
    assert built["metrics"]["n_calls"] == 2
    call_ids = [span[2] for span in built["trajectories"][0]["spans"]]
    assert "title" not in call_ids
