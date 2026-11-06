# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental prefix byte coverage, ownership, transactional forks and lineage."""

from dataclasses import replace

import pytest

from nemo_gym.token_id_capture.staging.digest import EMPTY_EXTRAS_DIGEST
from nemo_gym.token_id_capture.staging.incremental_prefix import (
    IncrementalPrefixDigest,
    PrefixMetadata,
)
from nemo_gym.token_id_capture.staging.records import StagedCallBaseSnapshot


def prefix():
    return IncrementalPrefixDigest(prefix_token_ids=[7, 8], parent_chain_hash="ab" * 32)


def metadata():
    return PrefixMetadata(2, "rollout", "call", "parent", "token_in", 2, 3)


def snapshot(state, *, tokens, mask, logprobs):
    meta = metadata()
    identity = state.identity(meta, extras_digest=EMPTY_EXTRAS_DIGEST)
    return StagedCallBaseSnapshot(
        digest_version=3,
        rollout_id=meta.rollout_id,
        model_call_id=meta.model_call_id,
        parent_call_id=meta.parent_call_id,
        mode=meta.mode,
        prev_len=meta.prev_len,
        delta_len=len(tokens),
        cum_len=meta.prev_len + len(tokens),
        weight_version=meta.weight_version,
        digest=identity.digest,
        token_ids_delta=tokens,
        token_mask_delta=mask,
        generation_log_probs_delta=logprobs,
        extras_digest=EMPTY_EXTRAS_DIGEST,
        chain_hash=identity.chain_hash,
        cumulative_hash=identity.cumulative_hash,
    )


@pytest.mark.parametrize("cut", [1, 2, 4])
def test_incremental_chunks_equal_independent_full_snapshot(cut):
    tokens = [10, 11, 12, 13]
    mask = [0.0, 1.0, 1.0, 1.0]
    probs = [0.0, -0.1, -0.2, -0.3]
    partial = prefix()
    partial.append(tokens=tokens[:cut], mask=mask[:cut], logprobs=probs[:cut])
    fork = partial.copy()
    fork.append(tokens=tokens[cut:], mask=mask[cut:], logprobs=probs[cut:])
    whole = prefix()
    whole.append(tokens=tokens, mask=mask, logprobs=probs)
    assert whole.identity(metadata(), extras_digest=EMPTY_EXTRAS_DIGEST) == fork.identity(
        metadata(), extras_digest=EMPTY_EXTRAS_DIGEST
    )
    assert partial.count == cut
    restored = snapshot(fork, tokens=tokens, mask=mask, logprobs=probs)
    assert restored.digest_version == 3 and restored.token_ids_delta == tokens


@pytest.mark.parametrize(
    "field,value",
    [
        ("token_ids_delta", [10, 99]),
        ("token_mask_delta", [1.0, 0.0]),
        ("generation_log_probs_delta", [-0.1, -0.9]),
        ("weight_version", 4),
        ("rollout_id", "other"),
        ("model_call_id", "other"),
        ("parent_call_id", "other"),
        ("chain_hash", "aa" * 32),
        ("cumulative_hash", "bb" * 32),
        ("extras_digest", "cc" * 32),
        ("delta_len", 3),
        ("cum_len", 99),
    ],
)
def test_snapshot_detects_each_column_and_metadata_corruption(field, value):
    state = prefix()
    state.append(tokens=[10, 11], mask=[1.0, 1.0], logprobs=[-0.1, -0.2])
    original = snapshot(state, tokens=[10, 11], mask=[1.0, 1.0], logprobs=[-0.1, -0.2])
    wire = original.model_dump()
    wire[field] = value
    with pytest.raises(ValueError):
        StagedCallBaseSnapshot.model_validate(wire)


@pytest.mark.parametrize(
    "tokens,mask,probs",
    [
        ([True], [1.0], [-0.1]),
        ([-1], [1.0], [-0.1]),
        ([1], [0.5], [-0.1]),
        ([1], [0.0], [-0.1]),
        ([1], [1.0], [float("nan")]),
        ([1], [1.0], [1e99]),
        ([1], [1.0], []),
    ],
)
def test_invalid_append_cannot_advance_committed_hashes(tokens, mask, probs):
    state = prefix()
    state.append(tokens=[9], mask=[1.0], logprobs=[-0.1])
    identity = state.identity(metadata(), extras_digest=EMPTY_EXTRAS_DIGEST)
    with pytest.raises((ValueError, TypeError)):
        state.append(tokens=tokens, mask=mask, logprobs=probs)
    assert state.count == 1
    assert state.identity(metadata(), extras_digest=EMPTY_EXTRAS_DIGEST) == identity


def test_identity_binds_exact_parent_tokens_and_chain():
    left = prefix()
    right = IncrementalPrefixDigest(prefix_token_ids=[7, 9], parent_chain_hash="ab" * 32)
    other = IncrementalPrefixDigest(prefix_token_ids=[7, 8], parent_chain_hash="cd" * 32)
    for state in (left, right, other):
        state.append(tokens=[10], mask=[1.0], logprobs=[-0.2])
    assert len({s.identity(metadata(), extras_digest=EMPTY_EXTRAS_DIGEST).digest for s in (left, right, other)}) == 3
    with pytest.raises(ValueError, match="parent length"):
        left.identity(replace(metadata(), prev_len=3), extras_digest=EMPTY_EXTRAS_DIGEST)
