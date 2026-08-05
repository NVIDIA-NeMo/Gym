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
"""Trajectory builder: chaining calls into one contiguous Responses projection."""

import pytest

from nemo_gym.token_id_capture import (
    assert_prefix_contiguity,
    compute_digest,
    per_request,
    prefix_merging,
    project_chain_to_output_items,
    project_main_chain_response,
    stamp_lineage,
    token_id_capture_dirs_from_config,
    trajectories_for_rollout,
)
from nemo_gym.token_id_capture.records import TokenEntry
from nemo_gym.token_id_capture.store import TokenCaptureStore


def _entry(mcid, prompt, gen, parent=None, lp=None, created_at=0.0):
    e = TokenEntry(
        rollout_id="t0-r0",
        model_call_id=mcid,
        model="m",
        prompt_token_ids=prompt,
        generation_token_ids=gen,
        generation_log_probs=lp if lp is not None else [-0.1] * len(gen),
        # Stamped when the call completed. Chain selection orders on it.
        created_at=created_at,
    )
    stamp_lineage(e, parent)
    return e


# An append-only 3-call rollout: each call's prompt extends the prior prompt+generation
# plus interstitial tokens (tool output / new user turn).
CALL1 = _entry("c1", [1, 2, 3], [10, 11])
CALL2 = _entry("c2", [1, 2, 3, 10, 11, 4, 5], [12])
CALL3 = _entry("c3", [1, 2, 3, 10, 11, 4, 5, 12, 6], [13, 14])
APPEND_ONLY = [CALL1, CALL2, CALL3]


def _generated_tokens(response: dict) -> list[int]:
    """What the policy sampled, read back off the projection."""
    out: list[int] = []
    for item in response["output"]:
        out += item.get("generation_token_ids") or []
    return sorted(out)


def test_prefix_merging_builds_one_contiguous_main_chain():
    out = prefix_merging(APPEND_ONLY)
    assert [c.chain_id for c in out.chains] == ["main"]

    response = project_main_chain_response("t0-r0", out, model="m")
    # Each item's prompt is the running cumulative sequence, so the loss mask a trainer needs
    # follows from the structure: prompt positions are context, generation positions are sampled.
    assert [i["prompt_token_ids"] for i in response["output"]] == [
        [1, 2, 3],
        [1, 2, 3, 10, 11, 4, 5],
        [1, 2, 3, 10, 11, 4, 5, 12, 6],
    ]
    assert [i["generation_token_ids"] for i in response["output"]] == [[10, 11], [12], [13, 14]]
    # A log probability for every sampled token.
    assert [len(i["generation_log_probs"]) for i in response["output"]] == [2, 1, 2]


def test_order_independent():
    import random

    shuffled = list(APPEND_ONLY)
    random.Random(0).shuffle(shuffled)
    a = project_main_chain_response("t0-r0", prefix_merging(APPEND_ONLY), model="m")
    b = project_main_chain_response("t0-r0", prefix_merging(shuffled), model="m")
    assert a["output"] == b["output"]


def test_per_request_marks_the_same_generated_tokens():
    # Both builders must agree on which tokens the policy sampled.
    merged = prefix_merging(APPEND_ONLY)
    per_req = per_request(APPEND_ONLY)
    assert len(per_req.chains) == 3

    merged_tokens = _generated_tokens(project_main_chain_response("t0-r0", merged, model="m"))
    per_req_tokens = sorted(
        tok
        for chain in per_req.chains
        for item in project_chain_to_output_items(chain)
        for tok in (item.get("generation_token_ids") or [])
    )
    assert merged_tokens == sorted([10, 11, 12, 13, 14])
    assert per_req_tokens == sorted([10, 11, 12, 13, 14])


def test_projection_is_prefix_contiguous():
    out = prefix_merging(APPEND_ONLY)
    response = project_main_chain_response("t0-r0", out, model="m")
    assert [len(i["prompt_token_ids"]) for i in response["output"]] == [3, 7, 9]
    assert response["usage"] == {"input_tokens": 3, "output_tokens": 5}
    assert_prefix_contiguity(response)  # must not raise


def test_projection_uses_the_recorded_carrier():
    """The record holds the arrays once; projection puts the chain-correct values back on
    the item they came off, leaving the other content items token-free."""
    entry = _entry("m1", [1, 2, 3], [4, 5])
    entry.output_items = [
        {"type": "message", "content": "thinking out loud"},
        {"type": "function_call", "name": "f", "arguments": "{}"},
    ]
    entry.token_item_index = 1
    projected = project_chain_to_output_items(prefix_merging([entry]).chains[0])
    assert projected[1]["prompt_token_ids"] == [1, 2, 3]
    assert projected[1]["generation_token_ids"] == [4, 5]
    assert "prompt_token_ids" not in projected[0]
    assert projected[0]["content"] == "thinking out loud"


def test_projection_falls_back_for_records_without_a_carrier_index():
    """Records written before the arrays moved off the items carry them inline and set no
    index. Scanning for the item that has them projects those identically."""
    entry = _entry("m1", [1, 2, 3], [4, 5])
    entry.output_items = [
        {"type": "message", "content": "thinking out loud"},
        {"type": "function_call", "name": "f", "arguments": "{}", "generation_token_ids": [4, 5]},
    ]
    projected = project_chain_to_output_items(prefix_merging([entry]).chains[0])
    assert projected[1]["prompt_token_ids"] == [1, 2, 3]
    assert "prompt_token_ids" not in projected[0]


def test_contiguity_assert_catches_a_gap():
    broken = {
        "output": [
            {"type": "message", "prompt_token_ids": [1, 2, 3], "generation_token_ids": [10]},
            # prompt does not extend [1,2,3,10]:
            {"type": "message", "prompt_token_ids": [1, 2, 3, 99], "generation_token_ids": [11]},
        ]
    }
    with pytest.raises(AssertionError):
        assert_prefix_contiguity(broken)


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
    assert_prefix_contiguity(resp)  # prompts still contiguous with content attached


def test_projection_handles_content_only_leading_item():
    # A single call whose output is an assistant text message (no token fields) followed by a
    # tool call that carries the token fields, the real shape when a model narrates before a
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
    assert_prefix_contiguity(resp)


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
    assert_prefix_contiguity(project_main_chain_response("t0-r0", out))


def test_consumer_reads_store_and_builds(tmp_path):
    # The co-located consumer: write the rollout's tokens, then build from the store files.
    store = TokenCaptureStore(tmp_path)
    for e in APPEND_ONLY:
        store.append(e.model_copy(update={"rollout_id": "t0-r0"}))
    dirs = token_id_capture_dirs_from_config({"token_id_capture": {"enabled": True, "dir": str(tmp_path)}})
    assert dirs == [tmp_path]
    merged = trajectories_for_rollout("t0-r0", dirs, builder="prefix_merging")
    assert merged is not None
    assert merged["builder"] == "prefix_merging"
    # Three calls become three contiguous output items on one Responses payload.
    output = merged["rebuilt_response"]["output"]
    assert len(output) == 3
    assert output[-1]["prompt_token_ids"] + output[-1]["generation_token_ids"] == [
        1,
        2,
        3,
        10,
        11,
        4,
        5,
        12,
        6,
        13,
        14,
    ]


def test_consumer_noop_when_disabled_or_absent(tmp_path):
    assert token_id_capture_dirs_from_config({}) == []
    assert trajectories_for_rollout("t0-r0", []) is None
    # Enabled dir but no file for this rollout -> None (graceful no-op).
    dirs = token_id_capture_dirs_from_config({"token_id_capture": {"enabled": True, "dir": str(tmp_path)}})
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


def test_the_earliest_root_becomes_the_delivered_chain():
    """The rollout's own first call completes before anything it goes on to do.

    Prompt length cannot decide this: the first real call carries the whole system prompt and
    tool definitions, so it is among the longest, while an auxiliary call is tiny. Generated
    length cannot either: an orchestrator that delegates generates less than what it delegates to.
    """
    # A short auxiliary call, dispatched after the agent's first turn was already served.
    side = _entry("side", [9000, 9001], [7, 7, 7], created_at=200.0)
    real_1 = _entry("real1", list(range(100, 160)), [200, 201, 202, 203], created_at=100.0)
    real_2 = _entry("real2", list(range(100, 160)) + [200, 201, 202, 203, 500], [300, 301, 302], created_at=150.0)

    out = prefix_merging([side, real_1, real_2])
    main = next(c for c in out.chains if c.chain_id == "main")

    assert [link.entry.model_call_id for link in main.links] == ["real1", "real2"]
    assert out.notes.chains == 2
    # The dropped chain is reported rather than silently discarded.
    assert out.notes.generated_tokens_captured == 10
    assert out.notes.generated_tokens_delivered == 7
    assert out.notes.delivered_fraction == 0.7


def test_an_orchestrator_is_delivered_over_the_sub_agent_it_delegates_to():
    """An orchestrator generates little and delegates the work, so the chain that generated
    the most is the wrong one to deliver. It starts first, which is what selection uses."""
    orchestrator = _entry("orch", list(range(100, 160)), [1, 2], created_at=100.0)
    sub_agent = _entry("sub", [9000, 9001], list(range(500, 560)), created_at=120.0)

    out = prefix_merging([orchestrator, sub_agent])
    main = next(c for c in out.chains if c.chain_id == "main")

    assert [link.entry.model_call_id for link in main.links] == ["orch"]
    # Most of the rollout's generation was not delivered, and that is visible rather than silent.
    assert out.notes.chains == 2
    assert out.notes.delivered_fraction < 0.1


def test_an_auxiliary_call_that_finishes_first_is_selected_and_reported():
    """A known limitation, pinned so it is not mistaken for correct behaviour.

    A harness that issues an auxiliary call before the agent's first turn, and whose short prompt
    returns sooner, has that call selected as the root. Selection cannot tell the two apart,
    because nothing in a record says which agent made the call. The rollout is not silently wrong:
    it reports more than one chain and a low delivered fraction.
    """
    side = _entry("side", [9000, 9001], [7, 7, 7], created_at=90.0)
    real = _entry("real", list(range(100, 160)), [200, 201, 202, 203], created_at=100.0)

    out = prefix_merging([side, real])
    main = next(c for c in out.chains if c.chain_id == "main")

    assert [link.entry.model_call_id for link in main.links] == ["side"]
    assert out.notes.chains == 2
    assert out.notes.delivered_fraction < 0.5


def test_selection_is_deterministic_when_timestamps_tie():
    """Records written in the same clock tick must not make the delivered chain arbitrary."""
    a = _entry("aaa", [1, 2], [3], created_at=100.0)
    b = _entry("bbb", [7, 8], [9], created_at=100.0)

    first = prefix_merging([a, b])
    second = prefix_merging([b, a])

    def main_id(out):
        return next(c for c in out.chains if c.chain_id == "main").links[0].entry.model_call_id

    assert main_id(first) == main_id(second) == "aaa"


def test_post_compaction_chain_is_reported_as_dropped():
    """A rewritten context starts a new root. Only one chain is delivered today,
    so what is left behind has to show up in the metrics."""
    call_1 = _entry("c1", [1, 2, 3], [4, 5])
    call_2 = _entry("c2", [1, 2, 3, 4, 5, 6], [7])
    # Compaction: the prompt no longer extends anything captured.
    call_3 = _entry("c3", [90, 91], [92, 93, 94, 95])

    out = prefix_merging([call_1, call_2, call_3])
    assert out.notes.chains == 2
    assert out.notes.generated_tokens_captured == 7
    assert out.notes.delivered_fraction < 1.0


# --- recorded parent links ----------------------------------------------------


def test_malformed_capture_masks_the_rollout_instead_of_raising(tmp_path):
    """The callers are a rollout-collection loop and a training framework's loop; an
    escaping exception there kills a whole step's batch rather than dropping one
    sample."""
    store = TokenCaptureStore(tmp_path)
    bad = _entry("c1", [1, 2, 3], [4, 5])
    bad.generation_log_probs = [-0.1]  # one log prob for two generated tokens
    store.append(bad)

    built = trajectories_for_rollout("t0-r0", [tmp_path])
    assert built is not None
    assert built["mask_sample"] is True
    assert built["rebuilt_response"] is None
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


def test_a_call_that_generated_nothing_is_not_a_parent():
    """A filtered call and its retry share a prompt. If the filtered one is treated as a parent the
    pair reads as a two-turn chain, and retry resolution never compares them because it only looks
    at siblings under a shared parent."""
    filtered = _entry("filtered", [1, 2, 3], [])
    retry = _entry("retry", [1, 2, 3], [9, 9])

    out = prefix_merging([filtered, retry])
    main = next(c for c in out.chains if c.chain_id == "main")

    assert [link.entry.model_call_id for link in main.links] == ["retry"]
    assert out.notes.empty_generation_calls == ["filtered"]
    assert out.notes.delivered_fraction == 1.0


def test_a_rollout_of_only_empty_generations_builds_nothing():
    out = prefix_merging([_entry("a", [1, 2], []), _entry("b", [1, 2, 3], [])])
    assert out.chains == []
    assert out.notes.empty_generation_calls == ["a", "b"]


def test_the_builder_runs_once_per_rollout(tmp_path, monkeypatch):
    """The metrics and the trajectories must come from the same chaining pass, and chaining is
    quadratic in call count when parent links are absent."""
    import nemo_gym.token_id_capture.consumer as consumer_module

    store = TokenCaptureStore(tmp_path)
    store.append(_entry("c1", [1, 2, 3], [4, 5]))
    store.append(_entry("c2", [1, 2, 3, 4, 5, 6], [7]))

    calls = []
    real_run_builder = consumer_module.run_builder

    def counting_run_builder(entries, builder="prefix_merging"):
        calls.append(builder)
        return real_run_builder(entries, builder)

    monkeypatch.setattr(consumer_module, "run_builder", counting_run_builder)
    built = trajectories_for_rollout("t0-r0", [tmp_path])

    assert calls == ["prefix_merging"]
    assert built["metrics"]["n_calls"] == 2


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
    assert out.notes.unresolved_retries == []


def test_unresolvable_final_retry_is_flagged_not_silently_tie_broken():
    """A retry of the LAST call has no successor to name the survivor. Neither
    inference nor a parent link can resolve it, so it must be reported so the
    caller can mask the rollout instead of training on a generation the client
    may never have received."""
    root = _with_lineage(_entry("root", [1, 2], [3]))
    a = _with_lineage(_entry("a", [1, 2, 3, 4], [5]), parent_call_id="root")
    b = _with_lineage(_entry("b", [1, 2, 3, 4], [9]), parent_call_id="root")

    out = prefix_merging([root, a, b])
    assert sorted(out.notes.unresolved_retries) == ["a", "b"]


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
    assert out.notes.parent_link_fallbacks == {"parent_digest_mismatch": 1}
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

    def shape(out):
        return [([link.entry.model_call_id for link in c.links], c.root_prompt) for c in out.chains]

    assert shape(inferred) == shape(recorded)


def test_a_recorded_parent_is_verified_not_trusted():
    """A rerun appends to the same rollout file, so a record can name a parent from a previous
    attempt whose tokens are an unrelated sequence. Chaining the two would train on a
    conversation that never happened, so the digest check has to reject it and fall back."""
    previous_attempt = _entry("old", [1, 2, 3], [4, 5])
    this_attempt = _entry("new", [90, 91, 92, 93], [94], parent="old")

    out = prefix_merging([previous_attempt, this_attempt])

    assert "parent_digest_mismatch" in out.notes.parent_link_fallbacks
    for chain in out.chains:
        assert [link.entry.model_call_id for link in chain.links] != ["old", "new"]


def test_a_correct_parent_link_is_used():
    """The other half: a link that verifies must be honoured, or the recorded lineage buys
    nothing over inferring the parent from token prefixes."""
    parent = _entry("p", [1, 2], [3, 4])
    child = _entry("c", [1, 2, 3, 4, 5], [6], parent="p")

    out = prefix_merging([parent, child])

    assert out.notes.parent_link_fallbacks == {}
    assert any([link.entry.model_call_id for link in chain.links] == ["p", "c"] for chain in out.chains)


def test_a_chain_that_breaks_is_split_and_reported():
    """Two calls whose prompts do not extend each other are separate sequences. Stitching them
    would hand the trainer positions the policy never generated in that order, so they become
    separate chains and delivered_fraction reports what was left behind."""
    first = _entry("a", [1, 2, 3], [4, 5])
    unrelated = _entry("b", [80, 81], [82])

    out = prefix_merging([first, unrelated])

    assert len(out.chains) > 1
    assert out.notes.delivered_fraction < 1.0
    assert_prefix_contiguity(project_main_chain_response("r0", out, model="m"))
