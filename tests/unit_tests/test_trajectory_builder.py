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
    prefix_merging,
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
