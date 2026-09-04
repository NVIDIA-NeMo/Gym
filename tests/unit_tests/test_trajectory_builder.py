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
"""Test trajectory building and contiguous Responses projection."""

import asyncio
import threading

import pytest

from nemo_gym.token_id_capture import (
    CaptureContext,
    FileLineageStore,
    InMemoryLineageStore,
    ParentResolutionStatus,
    TokenCaptureSnapshot,
    assert_prefix_contiguity,
    compute_digest,
    prefix_merging,
    project_chain_to_output_items,
    project_main_chain_response,
    reset_token_sink,
    set_token_sink,
    stamp_continuation,
    stamp_lineage,
    token_id_capture_dirs_from_config,
    trajectories_for_rollout,
    trajectories_from_source,
)
from nemo_gym.token_id_capture.records import TokenEntry
from nemo_gym.token_id_capture.sink import commit_entry, resolve_parent
from nemo_gym.token_id_capture.store import TokenCaptureStore


def _entry(mcid, prompt, gen, parent=None, lp=None, created_at=0.0):
    e = TokenEntry(
        rollout_id="t0-r0",
        model_call_id=mcid,
        model="m",
        prompt_token_ids=prompt,
        generation_token_ids=gen,
        generation_log_probs=lp if lp is not None else [-0.1] * len(gen),
        # Capture stamps this value when the call completes.
        # Chain selection uses this value.
        created_at=created_at,
    )
    # Stamp the way a current writer does.
    # Every supported record carries a parent decision.
    status = ParentResolutionStatus.RESOLVED if parent is not None else ParentResolutionStatus.ROOT
    stamp_lineage(e, parent, parent_resolution=status)
    return e


# This append-only rollout has three calls.
# Each prompt extends the previous prompt and generation.
# Interstitial tokens represent tool output or a new user turn.
CALL1 = _entry("c1", [1, 2, 3], [10, 11])
CALL2 = _entry("c2", [1, 2, 3, 10, 11, 4, 5], [12], parent="c1")
CALL3 = _entry("c3", [1, 2, 3, 10, 11, 4, 5, 12, 6], [13, 14], parent="c2")
APPEND_ONLY = [CALL1, CALL2, CALL3]


def _generated_tokens(response: dict) -> list[int]:
    """Return policy-sampled tokens from the projection."""
    out: list[int] = []
    for item in response["output"]:
        out += item.get("generation_token_ids") or []
    return sorted(out)


def test_prefix_merging_builds_one_contiguous_main_chain():
    out = prefix_merging(APPEND_ONLY)
    assert [c.chain_id for c in out.chains] == ["main"]

    response = project_main_chain_response("t0-r0", out, model="m")
    # Each prompt is the cumulative sequence.
    # Prompt positions are context.
    # Generation positions are sampled tokens.
    assert [i["prompt_token_ids"] for i in response["output"]] == [
        [1, 2, 3],
        [1, 2, 3, 10, 11, 4, 5],
        [1, 2, 3, 10, 11, 4, 5, 12, 6],
    ]
    assert [i["generation_token_ids"] for i in response["output"]] == [[10, 11], [12], [13, 14]]
    # Every sampled token has a log probability.
    assert [len(i["generation_log_probs"]) for i in response["output"]] == [2, 1, 2]


def test_prefix_merging_handles_a_thousand_turns_without_recursive_traversal():
    entries = []
    prompt = [1]
    for turn in range(1_100):
        generation = [10_000 + turn]
        parent = f"call-{turn - 1:04d}" if turn else None
        entries.append(_entry(f"call-{turn:04d}", list(prompt), generation, parent=parent))
        prompt.extend(generation)
        prompt.append(20_000 + turn)

    out = prefix_merging(entries)

    assert len(out.chains) == 1
    assert len(out.chains[0].links) == len(entries)
    assert out.notes.delivered_fraction == 1.0


def test_order_independent():
    import random

    shuffled = list(APPEND_ONLY)
    random.Random(0).shuffle(shuffled)
    a = project_main_chain_response("t0-r0", prefix_merging(APPEND_ONLY), model="m")
    b = project_main_chain_response("t0-r0", prefix_merging(shuffled), model="m")
    assert a["output"] == b["output"]


def test_projection_is_prefix_contiguous():
    out = prefix_merging(APPEND_ONLY)
    response = project_main_chain_response("t0-r0", out, model="m")
    assert [len(i["prompt_token_ids"]) for i in response["output"]] == [3, 7, 9]
    assert response["usage"] == {"input_tokens": 3, "output_tokens": 5}
    assert_prefix_contiguity(response)  # This must not raise.


def test_projection_uses_the_recorded_carrier():
    """Restore token arrays to their recorded carrier item.

    Leave other content items without token fields.
    """
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
    """Support records that keep token arrays inline.

    Older records do not set a carrier index.
    Scan those records for the token-bearing item.
    """
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
            # This prompt does not extend [1, 2, 3, 10].
            {"type": "message", "prompt_token_ids": [1, 2, 3, 99], "generation_token_ids": [11]},
        ]
    }
    with pytest.raises(AssertionError):
        assert_prefix_contiguity(broken)


def _content_entry(mcid, prompt, gen, text, parent=None):
    lp = [-0.1] * len(gen)
    entry = TokenEntry(
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
    status = ParentResolutionStatus.RESOLVED if parent is not None else ParentResolutionStatus.ROOT
    stamp_lineage(entry, parent, parent_resolution=status)
    return entry


def test_projection_carries_content_and_stays_contiguous():
    entries = [
        _content_entry("c1", [1, 2, 3], [10, 11], "first turn"),
        _content_entry("c2", [1, 2, 3, 10, 11, 4, 5], [12], "second turn", parent="c1"),
    ]
    out = prefix_merging(entries)
    resp = project_main_chain_response("t0-r0", out, model="m")
    texts = [item["content"][0]["text"] for item in resp["output"]]
    assert texts == ["first turn", "second turn"]  # Preserve content.
    assert [len(i["prompt_token_ids"]) for i in resp["output"]] == [3, 7]
    assert_prefix_contiguity(resp)  # Content does not break prompt contiguity.


def test_projection_handles_content_only_leading_item():
    # This call emits assistant text before a tool call.
    # The assistant text has no token fields.
    # The tool call carries the token fields.
    # Usage comes from the token-bearing item.
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
    assert resp["output"][0]["type"] == "message"  # Preserve the leading content-only item.
    assert "prompt_token_ids" not in resp["output"][0]
    assert resp["usage"] == {"input_tokens": 3, "output_tokens": 2}  # Count the token-bearing item.
    assert_prefix_contiguity(resp)


def test_consumer_reads_store_and_builds(tmp_path):
    # The colocated consumer freezes the local store before building.
    store = TokenCaptureStore(tmp_path)
    for e in APPEND_ONLY:
        store.append(e.model_copy(update={"rollout_id": "t0-r0"}))
    dirs = token_id_capture_dirs_from_config({"token_id_capture": {"enabled": True, "dir": str(tmp_path)}})
    assert dirs == [tmp_path]
    merged = trajectories_for_rollout("t0-r0", dirs, builder="prefix_merging")
    assert merged is not None
    assert merged["builder"] == "prefix_merging"
    # Three calls become contiguous items in one Responses payload.
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


def test_consumer_reads_and_freezes_an_external_source():
    class Source:
        async def freeze(self, rollout_id):
            return TokenCaptureSnapshot(
                rollout_id=rollout_id,
                entries=tuple(APPEND_ONLY),
                incomplete=False,
                snapshot_id="snapshot-1",
                version=4,
            )

        async def drop(self, rollout_id, *, snapshot_id, version):
            return True

        async def close(self):
            return None

    built = asyncio.run(trajectories_from_source("t0-r0", Source()))

    assert built["mask_sample"] is False
    assert built["_capture_snapshot"] == {
        "snapshot_id": "snapshot-1",
        "version": 4,
    }


def test_external_source_offloads_trajectory_assembly(monkeypatch):
    import nemo_gym.token_id_capture.consumer as consumer_module

    assemble_threads = []
    original_assemble = consumer_module._assemble

    def recording_assemble(*args, **kwargs):
        assemble_threads.append(threading.get_ident())
        return original_assemble(*args, **kwargs)

    monkeypatch.setattr(consumer_module, "_assemble", recording_assemble)

    class Source:
        async def freeze(self, rollout_id):
            return TokenCaptureSnapshot(
                rollout_id=rollout_id,
                entries=tuple(APPEND_ONLY),
                incomplete=False,
                snapshot_id="snapshot-thread",
                version=1,
            )

        async def drop(self, rollout_id, *, snapshot_id, version):
            return True

        async def close(self):
            return None

    async def consume():
        event_loop_thread = threading.get_ident()
        built = await trajectories_from_source("t0-r0", Source())
        return event_loop_thread, built

    event_loop_thread, built = asyncio.run(consume())

    assert built["mask_sample"] is False
    assert assemble_threads
    assert assemble_threads[0] != event_loop_thread


def test_consumer_masks_an_incomplete_external_snapshot():
    class Source:
        async def freeze(self, rollout_id):
            return TokenCaptureSnapshot(
                rollout_id=rollout_id,
                entries=(APPEND_ONLY[0],),
                incomplete=True,
                snapshot_id="snapshot-2",
                version=2,
            )

        async def drop(self, rollout_id, *, snapshot_id, version):
            return True

        async def close(self):
            return None

    built = asyncio.run(trajectories_from_source("t0-r0", Source()))

    assert built["mask_sample"] is True
    assert built["metrics"]["capture_incomplete"] is True


def test_consumer_masks_an_empty_external_snapshot():
    class Source:
        async def freeze(self, rollout_id):
            return TokenCaptureSnapshot(
                rollout_id=rollout_id,
                entries=(),
                incomplete=False,
                snapshot_id="snapshot-empty",
                version=1,
            )

        async def drop(self, rollout_id, *, snapshot_id, version):
            return True

        async def close(self):
            return None

    built = asyncio.run(trajectories_from_source("t0-r0", Source()))

    assert built["mask_sample"] is True
    assert built["error"] == "capture contains no token records"


def test_consumer_masks_an_external_source_failure():
    class Source:
        async def freeze(self, rollout_id):
            raise RuntimeError("transport unavailable")

        async def drop(self, rollout_id, *, snapshot_id, version):
            return True

        async def close(self):
            return None

    built = asyncio.run(trajectories_from_source("t0-r0", Source()))

    assert built["mask_sample"] is True
    assert "transport unavailable" in built["error"]


def test_consumer_noop_when_disabled_or_absent(tmp_path):
    assert token_id_capture_dirs_from_config({}) == []
    assert trajectories_for_rollout("t0-r0", []) is None
    # Missing records are unsafe after capture is configured.
    dirs = token_id_capture_dirs_from_config({"token_id_capture": {"enabled": True, "dir": str(tmp_path)}})
    missing = trajectories_for_rollout("missing", dirs)
    assert missing["mask_sample"] is True
    assert missing["rebuilt_response"] is None


# --- side calls and chain selection -------------------------------------------


def test_the_earliest_root_becomes_the_delivered_chain():
    """The rollout's own first call completes before anything it goes on to do.

    Prompt length cannot identify this call.
    The first task call contains the system prompt and tool definitions.
    An auxiliary call can have a much shorter prompt.
    Generation length cannot identify this call either.
    An orchestrator can generate fewer tokens than its delegate.
    """
    # This short auxiliary call starts after the first agent turn completes.
    side = _entry("side", [9000, 9001], [7, 7, 7], created_at=200.0)
    real_1 = _entry("real1", list(range(100, 160)), [200, 201, 202, 203], created_at=100.0)
    real_2 = _entry(
        "real2", list(range(100, 160)) + [200, 201, 202, 203, 500], [300, 301, 302], parent="real1", created_at=150.0
    )

    out = prefix_merging([side, real_1, real_2])
    main = next(c for c in out.chains if c.chain_id == "main")

    assert [link.entry.model_call_id for link in main.links] == ["real1", "real2"]
    assert out.notes.chains == 2
    # Metrics report the dropped chain.
    assert out.notes.generated_tokens_captured == 10
    assert out.notes.generated_tokens_delivered == 7
    assert out.notes.delivered_fraction == 0.7


def test_an_orchestrator_is_delivered_over_the_sub_agent_it_delegates_to():
    """Deliver the orchestrator instead of its sub-agent.

    The orchestrator generates few tokens before delegation.
    The longest generation is therefore the wrong selection criterion.
    Selection uses the earlier start.
    """
    orchestrator = _entry("orch", list(range(100, 160)), [1, 2], created_at=100.0)
    sub_agent = _entry("sub", [9000, 9001], list(range(500, 560)), created_at=120.0)

    out = prefix_merging([orchestrator, sub_agent])
    main = next(c for c in out.chains if c.chain_id == "main")

    assert [link.entry.model_call_id for link in main.links] == ["orch"]
    # Metrics expose the undelivered generation.
    assert out.notes.chains == 2
    assert out.notes.delivered_fraction < 0.1


def test_an_auxiliary_call_that_finishes_first_is_selected_and_reported():
    """Document the known auxiliary-call selection limitation.

    A harness issues an auxiliary call before the first agent turn.
    The short auxiliary call finishes first.
    Selection chooses that call as the root.
    Records do not identify the calling agent.
    Metrics report multiple chains and a low delivered fraction.
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
    """Report a post-compaction chain as dropped.

    A rewritten context starts a new root.
    Only one chain is delivered.
    Metrics report the remaining chain.
    """
    call_1 = _entry("c1", [1, 2, 3], [4, 5])
    call_2 = _entry("c2", [1, 2, 3, 4, 5, 6], [7], parent="c1")
    # A compacted context resolves UNRESOLVED at request time (edited history).
    call_3 = _entry("c3", [90, 91], [92, 93, 94, 95])
    stamp_lineage(call_3, None, parent_resolution=ParentResolutionStatus.UNRESOLVED)

    out = prefix_merging([call_1, call_2, call_3])
    assert out.notes.chains == 2
    assert out.notes.generated_tokens_captured == 7
    assert out.notes.delivered_fraction < 1.0


# --- recorded parent links ----------------------------------------------------


def test_malformed_capture_masks_the_rollout_instead_of_raising(tmp_path):
    """Mask a malformed capture instead of raising.

    An escaping exception can fail a rollout collection or training batch.
    """
    store = TokenCaptureStore(tmp_path)
    bad = _entry("c1", [1, 2, 3], [4, 5])
    bad.generation_log_probs = [-0.1]  # One log probability covers two generated tokens.
    store.append(bad)

    built = trajectories_for_rollout("t0-r0", [tmp_path])
    assert built is not None
    assert built["mask_sample"] is True
    assert built["rebuilt_response"] is None
    assert "ValidationError" in built["error"]


def test_incomplete_capture_masks_the_rollout(tmp_path):
    """Mask an incomplete capture.

    A missing call can still produce a clean-looking chain.
    The incomplete marker exposes the missing turn.
    """
    store = TokenCaptureStore(tmp_path)
    store.append(_entry("c1", [1, 2, 3], [4, 5]))
    asyncio.run(store.mark_incomplete("t0-r0", "c2"))

    built = trajectories_for_rollout("t0-r0", [tmp_path])
    assert built["mask_sample"] is True
    assert built["metrics"]["capture_incomplete"] is True


def test_clean_rollout_is_not_masked_and_reports_full_delivery(tmp_path):
    store = TokenCaptureStore(tmp_path)
    store.append(_entry("c1", [1, 2, 3], [4, 5]))
    store.append(_entry("c2", [1, 2, 3, 4, 5, 6], [7], parent="c1"))

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
    """Exclude an empty generation from parent inference.

    A filtered call and its retry share a prompt.
    Treating the filtered call as a parent creates a false two-turn chain.
    Retry resolution compares only siblings.
    """
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


def test_a_rollout_of_only_empty_generations_is_masked(tmp_path):
    store = TokenCaptureStore(tmp_path)
    store.append(_entry("a", [1, 2], []))
    store.append(_entry("b", [1, 2, 3], []))
    built = trajectories_for_rollout("t0-r0", [tmp_path])
    assert built["mask_sample"] is True
    assert built["rebuilt_response"] is None


def test_multiple_roots_are_masked_instead_of_rewarding_an_auxiliary_chain(tmp_path):
    store = TokenCaptureStore(tmp_path)
    store.append(_entry("side", [90], [91], created_at=1.0))
    store.append(_entry("task", [1, 2], [3], created_at=2.0))
    built = trajectories_for_rollout("t0-r0", [tmp_path])
    assert built["mask_sample"] is True
    assert built["metrics"]["roots"] == 2
    assert built["metrics"]["chains"] == 2


def test_the_builder_runs_once_per_rollout(tmp_path, monkeypatch):
    """Run the builder once per rollout.

    Metrics and trajectories must use the same chaining pass.
    Chaining is quadratic without parent links.
    """
    import nemo_gym.token_id_capture.consumer as consumer_module

    store = TokenCaptureStore(tmp_path)
    store.append(_entry("c1", [1, 2, 3], [4, 5]))
    store.append(_entry("c2", [1, 2, 3, 4, 5, 6], [7]))

    calls = []
    real_run_builder = consumer_module.run_builder

    def counting_run_builder(entries, builder="prefix_merging", **kwargs):
        calls.append(builder)
        return real_run_builder(entries, builder, **kwargs)

    monkeypatch.setattr(consumer_module, "run_builder", counting_run_builder)
    built = trajectories_for_rollout("t0-r0", [tmp_path])

    assert calls == ["prefix_merging"]
    assert built["metrics"]["n_calls"] == 2


def _with_lineage(entry, parent_call_id=None):
    status = ParentResolutionStatus.RESOLVED if parent_call_id is not None else ParentResolutionStatus.ROOT
    stamp_lineage(entry, parent_call_id, parent_resolution=status)
    return entry


def test_recorded_parent_link_resolves_a_final_call_retry_exactly():
    """Use a recorded parent link to resolve retries exactly.

    The siblings share a prompt and differ only in their generation.
    Prefix matching cannot identify which generation the harness kept.
    The next call's parent link identifies the survivor.
    """
    root = _with_lineage(_entry("root", [1, 2], [3]))
    kept = _with_lineage(_entry("kept", [1, 2, 3, 4], [5]), parent_call_id="root")
    dropped = _with_lineage(_entry("dropped", [1, 2, 3, 4], [9]), parent_call_id="root")
    # The next call explicitly continues ``kept``.
    nxt = _with_lineage(_entry("next", [1, 2, 3, 4, 5, 6], [7]), parent_call_id="kept")

    out = prefix_merging([root, kept, dropped, nxt])
    main = next(c for c in out.chains if c.chain_id == "main")
    assert [link.entry.model_call_id for link in main.links] == ["root", "kept", "next"]
    assert "dropped" in out.quarantined
    # Exact resolution leaves no retry unresolved.
    assert out.notes.unresolved_retries == []


def test_unresolvable_final_retry_is_flagged_not_silently_tie_broken():
    """Report a final-call retry that has no successor."""
    root = _with_lineage(_entry("root", [1, 2], [3]))
    a = _with_lineage(_entry("a", [1, 2, 3, 4], [5]), parent_call_id="root")
    b = _with_lineage(_entry("b", [1, 2, 3, 4], [9]), parent_call_id="root")

    out = prefix_merging([root, a, b])
    assert sorted(out.notes.unresolved_retries) == ["a", "b"]


def test_a_stale_parent_link_becomes_an_incomplete_fragment():
    """Preserve a call without inventing an edge across a digest mismatch."""
    root = _with_lineage(_entry("root", [1, 2], [3]))
    child = _entry("child", [1, 2, 3, 4], [5])
    stamp_lineage(child, "root", parent_resolution=ParentResolutionStatus.RESOLVED)
    # Simulate a stale record by corrupting the parent digest.
    root.digest = compute_digest([42, 42, 42])

    out = prefix_merging([root, child])
    assert out.notes.parent_link_failures == {"parent_digest_mismatch": 1}
    assert out.notes.unresolved_parent_calls == ["child"]
    assert sorted([link.entry.model_call_id for chain in out.chains for link in chain.links]) == ["child", "root"]
    assert "child" not in out.quarantined


def test_a_missing_filtered_parent_recovers_through_prefix_matching():
    """Recover a verified ancestor when the recorded parent is absent.

    The builder filters calls that generated no tokens.
    A child can therefore name a parent that is missing from the build.
    Prefix matching may reconnect that child to a verified surviving ancestor.
    A digest mismatch remains an unresolved boundary.
    """
    root = _with_lineage(_entry("root", [1, 2], [3]))
    empty = _with_lineage(_entry("empty", [1, 2, 3, 4], []), parent_call_id="root")
    child = _with_lineage(_entry("child", [1, 2, 3, 4, 5], [6]), parent_call_id="empty")

    out = prefix_merging([root, empty, child])

    assert out.notes.parent_link_failures == {"parent_call_id_missing_recovered": 1}
    assert out.notes.unresolved_parent_calls == []
    assert len(out.chains) == 1
    assert "child" not in out.quarantined


def test_a_duplicated_snapshot_entry_is_one_call():
    """Collapse identical entries delivered by an at-least-once transport.

    A duplicate must not become a second root that masks a complete rollout.
    """
    root = _with_lineage(_entry("root", [1, 2], [3]))
    child = _with_lineage(_entry("child", [1, 2, 3, 4], [5]), parent_call_id="root")
    duplicate = _with_lineage(_entry("child", [1, 2, 3, 4], [5]), parent_call_id="root")

    out = prefix_merging([root, child, duplicate])

    assert len(out.chains) == 1
    assert out.notes.unresolved_parent_calls == []


def test_unresolved_parent_never_uses_prefix_fallback():
    root = _with_lineage(_entry("root", [1, 2], [3]))
    child = _entry("child", [1, 2, 3, 4], [5])
    stamp_lineage(child, None, parent_resolution=ParentResolutionStatus.UNRESOLVED)

    out = prefix_merging([root, child])

    assert len(out.chains) == 2
    assert out.notes.unresolved_parent_calls == ["child"]


def test_explicit_root_never_uses_prefix_fallback():
    first = _with_lineage(_entry("first", [1, 2], [3]))
    root = _with_lineage(_entry("root", [1, 2, 3, 4], [5]))

    out = prefix_merging([first, root])

    assert len(out.chains) == 2
    assert out.notes.unresolved_parent_calls == []


def test_a_recorded_parent_is_verified_not_trusted():
    """Quarantine a recorded parent with unrelated tokens."""
    previous_attempt = _entry("old", [1, 2, 3], [4, 5])
    this_attempt = _entry("new", [90, 91, 92, 93], [94], parent="old")

    out = prefix_merging([previous_attempt, this_attempt])

    assert "parent_digest_mismatch" in out.notes.parent_link_failures
    for chain in out.chains:
        assert [link.entry.model_call_id for link in chain.links] != ["old", "new"]


def test_a_correct_parent_link_is_used():
    """Use a recorded parent link that passes verification."""
    parent = _entry("p", [1, 2], [3, 4])
    child = _entry("c", [1, 2, 3, 4, 5], [6], parent="p")

    out = prefix_merging([parent, child])

    assert out.notes.parent_link_failures == {}
    assert any([link.entry.model_call_id for link in chain.links] == ["p", "c"] for chain in out.chains)


def test_a_chain_that_breaks_is_split_and_reported():
    """Split calls whose prompts do not extend each other.

    These calls are separate sequences.
    Joining them would create an order that the policy never generated.
    The delivered fraction reports the omitted chain.
    """
    first = _entry("a", [1, 2, 3], [4, 5])
    unrelated = _entry("b", [80, 81], [82])

    out = prefix_merging([first, unrelated])

    assert len(out.chains) > 1
    assert out.notes.delivered_fraction < 1.0
    assert_prefix_contiguity(project_main_chain_response("r0", out, model="m"))


def _commit_delta_entry(
    store: TokenCaptureStore,
    lineage: FileLineageStore,
    rollout_id: str,
    call_id: str,
    prompt: list[int],
    generation: list[int],
    request: list[dict],
) -> CaptureContext:
    entry = TokenEntry(
        rollout_id=rollout_id,
        model_call_id=call_id,
        prompt_token_ids=prompt,
        generation_token_ids=generation,
        generation_log_probs=[-0.1] * len(generation),
        output_items=[{"type": "message", "role": "assistant", "content": f"answer {call_id}"}],
    )
    stamp_continuation(entry, request)
    context = CaptureContext(
        rollout_id=rollout_id,
        model_call_id=call_id,
        token_sink=store,
        lineage_store=lineage,
        delta_records=True,
    )
    token = set_token_sink(context)
    try:
        asyncio.run(resolve_parent(request))
        asyncio.run(commit_entry(entry))
    finally:
        reset_token_sink(token)
    return context


def test_delta_records_store_suffixes_and_reconstruct_exact_prompts(tmp_path):
    store = TokenCaptureStore(tmp_path)
    lineage = FileLineageStore(tmp_path)
    rollout_id = "delta"
    request1 = [{"role": "user", "content": "q1"}]
    _commit_delta_entry(store, lineage, rollout_id, "c1", [1, 2, 3], [4, 5], request1)
    request2 = request1 + [
        {"role": "assistant", "content": "answer c1"},
        {"role": "user", "content": "q2"},
    ]
    second = _commit_delta_entry(
        store,
        lineage,
        rollout_id,
        "c2",
        [1, 2, 3, 4, 5, 6, 7],
        [8],
        request2,
    )
    request3 = request2 + [
        {"role": "assistant", "content": "answer c2"},
        {"role": "user", "content": "q3"},
    ]
    third = _commit_delta_entry(
        store,
        lineage,
        rollout_id,
        "c3",
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [10, 11],
        request3,
    )

    assert second.parent_resolution is not None
    assert second.parent_resolution.status == ParentResolutionStatus.RESOLVED
    assert third.parent_resolution is not None
    assert third.parent_resolution.status == ParentResolutionStatus.RESOLVED
    entries = {entry.model_call_id: entry for entry in store.read_entries(rollout_id)}
    assert entries["c1"].prompt_is_delta is False
    assert entries["c2"].prompt_is_delta is True
    assert entries["c2"].prompt_token_ids == [6, 7]
    assert entries["c3"].prompt_is_delta is True
    assert entries["c3"].prompt_token_ids == [9]

    built = asyncio.run(trajectories_from_source(rollout_id, store))
    assert built["mask_sample"] is False
    output = built["rebuilt_response"]["output"]
    assert [item["generation_token_ids"] for item in output] == [[4, 5], [8], [10, 11]]
    assert output[2]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_delta_materialization_handles_a_thousand_turns_without_recursion():
    root = _entry("c0", [1], [2])
    entries = [root]
    cumulative = [1, 2]
    for turn in range(1, 1_101):
        suffix = [10_000 + turn]
        generation = [20_000 + turn]
        full_prompt = cumulative + suffix
        entry = _entry(f"c{turn}", full_prompt, generation, parent=f"c{turn - 1}")
        entry.prompt_token_ids = suffix
        entry.prompt_is_delta = True
        entries.append(entry)
        cumulative = full_prompt + generation

    out = prefix_merging(entries)

    assert len(out.chains) == 1
    assert len(out.chains[0].links) == len(entries)
    assert out.chains[0].links[-1].entry.prompt_token_ids == cumulative[:-1]


def test_broken_delta_chain_masks_instead_of_guessing(tmp_path):
    store = TokenCaptureStore(tmp_path)
    orphan = TokenEntry(
        rollout_id="orphan",
        model_call_id="child",
        prompt_token_ids=[9],
        generation_token_ids=[10],
        generation_log_probs=[-0.1],
        output_items=[{"type": "message", "role": "assistant", "content": "a"}],
        prompt_is_delta=True,
        parent_call_id="missing-parent",
        parent_resolution=ParentResolutionStatus.RESOLVED,
    )
    stamp_continuation(orphan, [{"role": "user", "content": "q"}])
    stamp_lineage(
        orphan,
        "missing-parent",
        parent_resolution=ParentResolutionStatus.RESOLVED,
        cumulative=[1, 2, 9, 10],
    )
    store.append(orphan)

    built = asyncio.run(trajectories_from_source("orphan", store))
    out = prefix_merging(store.read_entries("orphan"))

    assert built["mask_sample"] is True
    assert out.notes.parent_link_failures["delta_chain_unreconstructable"] == 1
    assert out.notes.unresolved_parent_calls == ["child"]


def test_in_memory_lineage_refuses_delta_records():
    entry = TokenEntry(
        rollout_id="r",
        model_call_id="child",
        prompt_token_ids=[9],
        generation_token_ids=[10],
        generation_log_probs=[-0.1],
        output_items=[{"type": "message", "role": "assistant", "content": "answer"}],
        prompt_is_delta=True,
        parent_call_id="parent",
        parent_resolution=ParentResolutionStatus.RESOLVED,
    )
    stamp_continuation(entry, [{"role": "user", "content": "q"}])
    stamp_lineage(
        entry,
        "parent",
        parent_resolution=ParentResolutionStatus.RESOLVED,
        cumulative=[1, 9, 10],
    )

    with pytest.raises(ValueError, match="durable-log-backed"):
        asyncio.run(InMemoryLineageStore().put(entry))
