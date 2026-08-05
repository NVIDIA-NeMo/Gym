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
    per_request,
    prefix_merging,
    project_chain_to_output_items,
    project_main_chain_response,
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


def test_consumer_reads_store_and_builds(tmp_path):
    # The co-located consumer: write the rollout's tokens, then build from the store files.
    store = TokenCaptureStore(tmp_path)
    for e in APPEND_ONLY:
        store.append(e.model_copy(update={"rollout_id": "t0-r0"}))
    dirs = token_id_capture_dirs_from_config({"token_id_capture_enabled": True, "token_id_capture_dir": str(tmp_path)})
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
    error, because the trainer would receive a contiguous, token-bearing response
    containing a generated title, with the rollout's reward attached.
    """
    title = _entry("title", [9000, 9001], [7, 7, 7])
    real_1 = _entry("real1", list(range(100, 160)), [200, 201, 202, 203])
    real_2 = _entry("real2", list(range(100, 160)) + [200, 201, 202, 203, 500], [300, 301, 302])

    out = prefix_merging([title, real_1, real_2])
    main = next(c for c in out.chains if c.chain_id == "main")

    assert [link.entry.model_call_id for link in main.links] == ["real1", "real2"]
    assert out.notes.chains == 2
    # The dropped chain is reported rather than silently discarded.
    assert out.notes.generated_tokens_captured == 10
    assert out.notes.generated_tokens_delivered == 7
    assert out.notes.delivered_fraction == 0.7


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
