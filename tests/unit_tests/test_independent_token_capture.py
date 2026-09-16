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

from unittest.mock import AsyncMock

import pytest

from nemo_gym.token_id_capture.builder import independent_calls, project_independent_call_responses
from nemo_gym.token_id_capture.config import TokenIdCaptureSettings
from nemo_gym.token_id_capture.consumer import _assemble, trajectories_from_source
from nemo_gym.token_id_capture.delivery import (
    TRAINING_RESPONSES_KEY,
    finalize_rollout_token_capture,
    retire_rollout_token_capture,
)
from nemo_gym.token_id_capture.protocols import TokenCaptureSnapshot
from nemo_gym.token_id_capture.records import ParentResolutionStatus, TokenEntry, stamp_lineage


def entry(call_id, prompt, generated, *, parent=None):
    record = TokenEntry(
        rollout_id="rollout",
        model_call_id=call_id,
        response_id=f"response-{call_id}",
        prompt_token_ids=prompt,
        generation_token_ids=generated,
        generation_log_probs=[-0.5] * len(generated),
    )
    return stamp_lineage(
        record,
        parent,
        parent_resolution=ParentResolutionStatus.RESOLVED if parent else ParentResolutionStatus.ROOT,
    )


def build(entries, **kwargs):
    return _assemble("rollout", entries, "independent_calls", "policy", **kwargs)


def generated_items(built):
    return [
        item
        for response in built["rebuilt_responses"]
        for item in response["output"]
        if item.get("generation_token_ids")
    ]


def test_independent_mode_is_opt_in():
    assert TokenIdCaptureSettings().builder == "prefix_merging"
    records = [entry("a", [1], [2]), entry("subagent", [9], [10])]
    assert _assemble("rollout", records, "prefix_merging", "policy")["mask_sample"]
    assert not build(records)["mask_sample"]


def test_retokenization_compaction_and_subagents_preserve_each_exact_context():
    records = [entry("a", [1, 2], [3, 4]), entry("b", [1, 2, 30, 40, 5], [6]), entry("subagent", [9], [10])]
    records[1].parent_resolution = ParentResolutionStatus.UNRESOLVED
    built = build(records)
    assert not built["mask_sample"]
    assert [(item["prompt_token_ids"], item["generation_token_ids"]) for item in generated_items(built)] == [
        ([1, 2], [3, 4]),
        ([1, 2, 30, 40, 5], [6]),
        ([9], [10]),
    ]
    assert built["metrics"]["generated_tokens_delivered"] == 4
    assert built["metrics"]["delivered_fraction"] == 1.0


def test_delta_prompt_is_materialized_before_delivery_without_mutating_snapshot():
    root = entry("a", [1], [2])
    child = entry("b", [1, 2, 3], [4], parent="a")
    child.prompt_is_delta = True
    child.prompt_token_ids = [3]
    built = build([child, root])
    assert not built["mask_sample"]
    assert generated_items(built)[1]["prompt_token_ids"] == [1, 2, 3]
    assert child.prompt_token_ids == [3]
    assert child.prompt_is_delta
    assert built["metrics"]["generated_tokens_delivered"] == 2


@pytest.mark.parametrize("corruption", ["missing_parent", "digest", "length", "delta_cycle", "unproven_delta"])
def test_corrupt_provenance_masks_instead_of_fabricating_context(corruption):
    root = entry("a", [1], [2])
    child = entry("b", [1, 2, 3], [4], parent="a")
    if corruption == "missing_parent":
        child.parent_call_id = "absent"
        child.prefix_supplied = True
    elif corruption == "digest":
        child.digest = "wrong"
    elif corruption == "length":
        child.cum_len = 999
    elif corruption == "delta_cycle":
        child.prompt_is_delta = True
        child.parent_call_id = "b"
    else:
        child.prompt_is_delta = True
        child.digest = None
    built = build([root, child])
    assert built["mask_sample"]
    assert built["rebuilt_response"] is None
    assert "error" in built


@pytest.mark.parametrize(
    "field,value",
    [("generation_log_probs", [-1.0]), ("routed_experts", [[1]]), ("ng_generation_weight_version_end", 7)],
)
def test_duplicate_call_id_with_conflicting_training_metadata_is_rejected(field, value):
    original = entry("a", [1], [2])
    conflict = original.model_copy(update={field: value})
    assert build([original, conflict])["mask_sample"]


def test_transport_duplicates_and_inline_carriers_do_not_duplicate_actions():
    original = entry("a", [1], [2, 3])
    original.output_items = [
        {"type": "reasoning", "generation_token_ids": [2, 3]},
        {"type": "message", "generation_token_ids": [2, 3]},
    ]
    built = build([original, original.model_copy(update={"created_at": 12.0})])
    assert not built["mask_sample"]
    assert len(generated_items(built)) == 1
    assert built["metrics"]["generated_tokens_delivered"] == 2


def test_distinct_calls_with_identical_prompts_keep_every_sampled_action():
    first = entry("a", [1], [2])
    retry = entry("b", [1], [3])
    child = entry("c", [1, 3, 4], [5], parent="b")
    built = build([first, retry, child])
    assert not built["mask_sample"]
    assert [item["generation_token_ids"] for item in generated_items(built)] == [[2], [3], [5]]
    first_child = entry("d", [1, 2, 4], [6], parent="a")
    both = build([first, retry, child, first_child], explicit_terminal_call_id="c")
    assert not both["mask_sample"]
    assert both["metrics"]["generated_tokens_delivered"] == 4


def test_terminal_selection_does_not_discard_identical_prompt_subagents():
    records = [entry("a", [1], [2]), entry("retry", [1], [3]), entry("main", [9], [10])]
    built = build(records, explicit_terminal_call_id="main")
    assert not built["mask_sample"]
    assert built["unresolved_retries"] == []
    assert [item["generation_token_ids"] for item in generated_items(built)] == [[2], [10], [3]]


def test_terminal_witness_preserves_other_completed_calls():
    built = build([entry("a", [1], [2]), entry("b", [1], [3])], declared_response_id="response-b")
    assert not built["mask_sample"]
    assert [item["generation_token_ids"] for item in generated_items(built)] == [[2], [3]]


def test_identical_outputs_from_distinct_calls_are_not_transport_duplicates():
    built = build([entry("a", [1], [2]), entry("b", [1], [2])])
    assert not built["mask_sample"]
    assert len(generated_items(built)) == 2
    assert built["metrics"]["generated_tokens_delivered"] == 2


def test_unknown_declared_terminal_is_not_ignored():
    assert build([entry("a", [1], [2])], declared_response_id="absent")["mask_sample"]


def test_message_parent_match_does_not_force_token_continuity():
    first = entry("a", [1], [2])
    retokenized = entry("b", [10, 20, 3], [4], parent="a")
    built = build([first, retokenized])
    assert not built["mask_sample"]
    assert generated_items(built)[1]["prompt_token_ids"] == [10, 20, 3]
    retokenized.prefix_supplied = True
    assert build([first, retokenized])["mask_sample"]


@pytest.mark.parametrize("logprobs", [[], [float("inf")], [float("nan")], [float("-inf")]])
def test_missing_or_nonfinite_generated_logprobs_mask(logprobs):
    record = entry("a", [1], [2])
    record.generation_log_probs = logprobs
    assert build([record])["mask_sample"]


def test_router_cache_and_refit_metadata_survive_delta_materialization():
    root = entry("a", [1], [2])
    child = entry("b", [1, 2, 3], [4], parent="a")
    metadata = {
        "routed_experts": [[[1]], [[2]], [[3]], [[-1]]],
        "ng_generation_replica_id": "replica-2",
        "ng_generation_weight_version": 7,
        "ng_generation_weight_version_end": 8,
        "ng_kv_cache_scheduler_block_size": 16,
        "ng_kv_cache_hash_block_size": 16,
        "ng_kv_cache_num_cached_tokens": 2,
    }
    child = child.model_copy(update={**metadata, "prompt_is_delta": True, "prompt_token_ids": [3]})
    item = generated_items(build([root, child]))[1]
    assert {key: item[key] for key in metadata} == metadata


def test_independent_projection_refuses_chained_builder_output():
    from nemo_gym.token_id_capture.builder import prefix_merging

    with pytest.raises(ValueError, match="independent_calls"):
        project_independent_call_responses("rollout", prefix_merging([entry("a", [1], [2])]))
    assert len(independent_calls([entry("a", [1], [2])]).chains) == 1


@pytest.mark.asyncio
async def test_exact_call_finalize_preserves_native_response_and_retires_only_after_handoff():
    source = AsyncMock()
    source.freeze.return_value = TokenCaptureSnapshot("rollout", (entry("a", [1], [2]),), False, "frozen", 4)
    source.drop.return_value = True
    native = {"output": [{"generation_token_ids": [99]}]}
    result = {"_ng_rollout_id": "rollout", "response": native, "reward": 0.75}
    built = await finalize_rollout_token_capture(result, source, builder="independent_calls")
    assert not built["mask_sample"]
    assert result["response"] is native
    assert result["reward"] == 0.75
    assert result[TRAINING_RESPONSES_KEY][0]["output"][0]["generation_token_ids"] == [2]
    source.drop.assert_not_called()
    assert await retire_rollout_token_capture("rollout", source, built)
    source.drop.assert_awaited_once_with("rollout", snapshot_id="frozen", version=4)


@pytest.mark.asyncio
async def test_incomplete_snapshot_is_never_retired_even_with_independent_calls():
    source = AsyncMock()
    source.freeze.return_value = TokenCaptureSnapshot("rollout", (entry("a", [1], [2]),), True, "frozen", 4)
    built = await trajectories_from_source("rollout", source, builder="independent_calls")
    assert built["mask_sample"]
    assert built["metrics"]["capture_incomplete"]
    assert not await retire_rollout_token_capture("rollout", source, built)
    source.drop.assert_not_called()
