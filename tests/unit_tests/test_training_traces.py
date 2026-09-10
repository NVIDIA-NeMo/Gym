# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check complete rollout delivery, custody, and reconstruction-independent ownership."""

import copy
from itertools import permutations
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from nemo_gym.global_config import ROLLOUT_ID_KEY_NAME
from nemo_gym.token_id_capture.builder import run_builder
from nemo_gym.token_id_capture.config import TokenIdCaptureConfig
from nemo_gym.token_id_capture.consumer import _assemble, trajectories_for_rollout, trajectories_from_source
from nemo_gym.token_id_capture.delivery import (
    capture_build_can_retire,
    finalize_rollout_token_capture,
    retire_rollout_token_capture,
)
from nemo_gym.token_id_capture.protocols import TokenCaptureSnapshot
from nemo_gym.token_id_capture.records import ParentResolutionStatus, TokenEntry, stamp_lineage
from nemo_gym.token_id_capture.store import TokenCaptureStore
from nemo_gym.token_id_capture.training_traces import TrainingTraceBatch, project_training_traces


def _entry(call_id, prompt, generated, parent=None):
    entry = TokenEntry(
        rollout_id="r1",
        model_call_id=call_id,
        model="policy",
        prompt_token_ids=prompt,
        generation_token_ids=generated,
        generation_log_probs=[-token / 100 for token in generated],
        response_id=f"response-{call_id}",
    )
    return stamp_lineage(
        entry,
        parent,
        parent_resolution=ParentResolutionStatus.RESOLVED if parent else ParentResolutionStatus.ROOT,
    )


def _branching_calls():
    return [
        _entry("a", [1, 2], [10, 11]),
        _entry("b", [1, 2, 10, 11, 3], [12], "a"),
        _entry("c", [1, 2, 10, 11, 4], [13, 14], "a"),
        _entry("d", [5, 6], [15]),
    ]


def _build(entries, builder="prefix_merging", **kwargs):
    return _assemble("r1", entries, builder, "policy", delivery="all_traces", **kwargs)


def _owned(envelope):
    return {
        span["model_call_id"]: (
            trace["token_ids"][span["start"] : span["end"]],
            trace["generation_logprobs"][span["start"] : span["end"]],
        )
        for trace in envelope["traces"]
        for span in trace["sampled_spans"]
    }


@pytest.mark.parametrize("builder", ["prefix_merging", "per_request"])
def test_interleaved_forks_and_independent_roots_preserve_every_sample_once(builder):
    entries = _branching_calls()
    expected = {entry.model_call_id: (entry.generation_token_ids, entry.generation_log_probs) for entry in entries}
    results = []
    for shuffled in permutations(entries):
        result = _build(list(shuffled), builder, explicit_terminal_call_id="c")
        assert not result["mask_sample"]
        envelope = result["training_traces"]
        assert _owned(envelope) == expected
        assert sum(sum(trace["loss_mask"]) for trace in envelope["traces"]) == 6
        assert result["metrics"]["delivered_fraction"] == 1.0
        results.append(envelope)
    assert all(result == results[0] for result in results)
    assert len(results[0]["traces"]) == (4 if builder == "per_request" else 3)


def test_prefix_projection_masks_ancestor_copy_but_retains_exact_context():
    envelope = _build(_branching_calls())["training_traces"]
    second_branch = next(trace for trace in envelope["traces"] if trace["model_call_ids"] == ["a", "c"])
    assert second_branch["token_ids"] == [1, 2, 10, 11, 4, 13, 14]
    assert second_branch["loss_mask"] == [0, 0, 0, 0, 0, 1, 1]
    assert second_branch["generation_logprobs"][-2:] == [-0.13, -0.14]


def test_representation_modes_own_identical_sampled_data():
    entries = _branching_calls()
    assert _owned(_build(entries, "per_request")["training_traces"]) == _owned(
        _build(entries, "prefix_merging")["training_traces"]
    )


@pytest.mark.parametrize("builder", ["prefix_merging", "per_request"])
def test_compaction_root_preserved_and_legacy_delivery_unchanged(builder):
    entries = [_entry("a", [1], [10]), _entry("b", [1, 10, 2], [11], "a"), _entry("c", [3], [12])]
    result = _build(entries, builder, explicit_terminal_call_id="c")
    assert set(_owned(result["training_traces"])) == {"a", "b", "c"}
    legacy = _assemble("r1", entries, "prefix_merging", "policy", explicit_terminal_call_id="c")
    assert not legacy["mask_sample"]
    assert "training_traces" not in legacy
    assert legacy["rebuilt_response"]["output"][0]["generation_token_ids"] == [12]


@pytest.mark.parametrize("builder", ["prefix_merging", "per_request"])
def test_duplicate_delivery_deduplicated_and_differing_logprobs_rejected(builder):
    entry = _entry("a", [1], [10])
    result = _build([entry, entry.model_copy(deep=True)], builder)
    assert not result["mask_sample"]
    assert len(_owned(result["training_traces"])) == 1
    conflict = entry.model_copy(update={"generation_log_probs": [-0.9]})
    bad = _build([entry, conflict], builder)
    assert bad["mask_sample"] and "training_traces" not in bad


@pytest.mark.parametrize("builder", ["prefix_merging", "per_request"])
def test_ambiguous_off_terminal_retry_masks_complete_rollout(builder):
    entries = [_entry("a", [1], [10]), _entry("b", [1], [11]), _entry("terminal", [2], [12])]
    result = _build(entries, builder, explicit_terminal_call_id="terminal")
    assert result["mask_sample"] and "training_traces" not in result


@pytest.mark.parametrize("builder", ["prefix_merging", "per_request"])
def test_terminal_and_extended_sibling_both_have_retention_evidence(builder):
    entries = [_entry("a", [1], [10]), _entry("b", [1], [11]), _entry("c", [1, 11, 2], [12], "b")]
    result = _build(entries, builder, explicit_terminal_call_id="a")
    assert set(_owned(result["training_traces"])) == {"a", "b", "c"}


@pytest.mark.parametrize("builder", ["prefix_merging", "per_request"])
def test_retry_selected_by_terminal_or_child_excludes_abandoned_generation(builder):
    entries = [_entry("a", [1], [10]), _entry("b", [1], [11])]
    result = _build(entries, builder, explicit_terminal_call_id="b")
    assert set(_owned(result["training_traces"])) == {"b"}
    entries.append(_entry("c", [1, 11, 2], [12], "b"))
    result = _build(entries, builder)
    assert set(_owned(result["training_traces"])) == {"b", "c"}


@pytest.mark.parametrize("builder", ["prefix_merging", "per_request"])
def test_delta_prompts_equal_full_prompt_reconstruction(builder):
    first = _entry("a", [1], [10])
    full = _entry("b", [1, 10, 2], [11], "a")
    delta = full.model_copy(update={"prompt_token_ids": [2], "prompt_is_delta": True})
    assert _build([first, delta], builder)["training_traces"] == _build([first, full], builder)["training_traces"]
    bad = _build([delta], builder)
    assert bad["mask_sample"] and "training_traces" not in bad


@pytest.mark.parametrize(
    "update",
    [
        {"rollout_id": "different"},
        {"parent_resolution": ParentResolutionStatus.UNRESOLVED},
        {"digest": "bad-proof"},
        {"cum_len": 999},
        {"generation_log_probs": [float("nan")]},
        {"routed_experts": [[0]]},
    ],
)
def test_unsupported_or_unproven_evidence_is_not_delivered(update):
    result = _build([_entry("a", [1], [10]).model_copy(update=update)])
    assert result["mask_sample"] and "training_traces" not in result


def test_unknown_terminal_and_mixed_policy_models_fail_closed():
    assert _build([_entry("a", [1], [10])], declared_response_id="missing")["mask_sample"]
    result = _build([_entry("a", [1], [10]), _entry("b", [2], [11]).model_copy(update={"model": "other"})])
    assert result["mask_sample"]


def test_empty_prompt_and_empty_generation_cannot_create_unconditioned_loss():
    assert _build([_entry("a", [], [10])])["mask_sample"]
    assert _build([_entry("a", [1], [])])["mask_sample"]


@pytest.mark.parametrize("builder", ["prefix_merging", "per_request"])
async def test_source_file_and_finalizer_share_envelope_and_retirement_boundary(tmp_path, builder):
    store = TokenCaptureStore(tmp_path)
    for entry in _branching_calls():
        store.append(entry)
    file_result = trajectories_for_rollout("r1", [tmp_path], builder=builder, delivery="all_traces")
    source_result = await trajectories_from_source("r1", store, builder=builder, delivery="all_traces")
    assert file_result["training_traces"] == source_result["training_traces"]
    response = {"output": [{"generation_token_ids": [99], "content": "scored answer"}]}
    record = {ROLLOUT_ID_KEY_NAME: "r1", "response": copy.deepcopy(response), "reward": 0.75}
    built = await finalize_rollout_token_capture(record, store, builder=builder, delivery="all_traces")
    assert record["response"] == response and record["reward"] == 0.75
    assert record["training_traces"] == file_result["training_traces"]
    assert capture_build_can_retire(built)
    assert len(store.read_entries("r1")) == 4
    assert await retire_rollout_token_capture("r1", store, built)
    assert store.read_entries("r1") == []
    await store.close()


async def test_incomplete_source_masks_and_removes_stale_envelope():
    source = AsyncMock()
    source.freeze.return_value = TokenCaptureSnapshot("r1", tuple(_branching_calls()), True, "frozen", 1)
    result = {ROLLOUT_ID_KEY_NAME: "r1", "response": {}, "training_traces": {"stale": True}}
    with pytest.warns(UserWarning, match="incompletely"):
        built = await finalize_rollout_token_capture(result, source, delivery="all_traces")
    assert result["mask_sample"] and "training_traces" not in result
    assert not capture_build_can_retire(built)


@pytest.mark.parametrize(
    "block,match",
    [
        ({"builder": "per_request"}, "per_request requires"),
        (
            {"enabled": True, "delivery": "all_traces", "external_staging": True, "rebuild_response": False},
            "external_staging",
        ),
        ({"delivery": "unknown"}, "Input should be"),
    ],
)
def test_unsupported_config_rejected_before_training(block, match):
    with pytest.raises(ValidationError, match=match):
        TokenIdCaptureConfig.model_validate({"token_id_capture": block})


def test_default_configuration_is_legacy_and_per_request_is_registered():
    settings = TokenIdCaptureConfig().token_id_capture
    assert settings.delivery == "main_chain" and settings.builder == "prefix_merging"
    assert run_builder(_branching_calls(), "per_request").notes.builder == "per_request"


@pytest.mark.parametrize(
    "damage",
    [
        "double-owner",
        "span-outside",
        "unowned-mask",
        "first-token",
        "nan",
        "length",
        "call-ids",
        "overlap",
        "trace-ids",
        "unowned-call",
    ],
)
def test_wire_contract_rejects_invalid_ownership_and_alignment(damage):
    envelope = _build(_branching_calls())["training_traces"]
    first = envelope["traces"][0]
    if damage == "double-owner":
        duplicate = copy.deepcopy(first)
        duplicate["trace_id"] = "duplicate"
        envelope["traces"].append(duplicate)
    elif damage == "span-outside":
        first["sampled_spans"][0]["end"] = 1000
    elif damage == "unowned-mask":
        first["loss_mask"][1] = 1
    elif damage == "first-token":
        first["loss_mask"][0] = 1
    elif damage == "nan":
        first["generation_logprobs"][0] = float("nan")
    elif damage == "length":
        first["generation_logprobs"].pop()
    elif damage == "call-ids":
        first["model_call_ids"].append(first["model_call_ids"][0])
    elif damage == "overlap":
        first["sampled_spans"].append(copy.deepcopy(first["sampled_spans"][0]))
    elif damage == "trace-ids":
        envelope["traces"][1]["trace_id"] = first["trace_id"]
    else:
        first["model_call_ids"].append("missing-owner")
    with pytest.raises(ValidationError):
        TrainingTraceBatch.model_validate(envelope)


def test_direct_projection_rejects_wrong_rollout_and_unavailable_terminal():
    entries = _branching_calls()
    with pytest.raises(ValueError, match="different rollout"):
        project_training_traces("wrong", run_builder(entries, all_traces=True))
    with pytest.raises(ValueError, match="terminal"):
        project_training_traces("r1", run_builder(entries, terminal_call_id="absent", all_traces=True))
