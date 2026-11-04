# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt carry must remain zero for every mask layout and wire round trip."""

import math
from itertools import product

import pytest
from pydantic import ValidationError

from nemo_gym.token_id_capture.staging.digest import compute_chain_hash, hash_token_ids
from nemo_gym.token_id_capture.staging.records import StagedCallRecord, StagedCallSnapshot


def _record(mask: list[float], logprobs: list[float]) -> StagedCallRecord:
    tokens = list(range(len(mask)))
    return StagedCallRecord.from_components(
        rollout_id="r",
        model_call_id="c",
        parent_call_id=None,
        mode="text",
        prev_len=0,
        delta_len=len(tokens),
        cum_len=len(tokens),
        weight_version=7,
        token_ids_delta=tokens,
        token_mask_delta=mask,
        generation_log_probs_delta=logprobs,
        extras=None,
        extras_digest=None,
        chain_hash=compute_chain_hash(None, tokens),
        cumulative_hash=hash_token_ids(tokens),
    )


@pytest.mark.parametrize("mask", [list(values) for values in product((0.0, 1.0), repeat=6)])
def test_every_mask_layout_enforces_only_prompt_logprobs(mask: list[float]) -> None:
    logprobs = [0.0 if value == 0.0 else -0.25 for value in mask]
    record = _record(mask, logprobs)
    assert StagedCallSnapshot.model_validate_json(record.model_dump_json()).model_dump() == record.model_dump()
    for index, value in enumerate(mask):
        modified = logprobs.copy()
        modified[index] = -0.5
        if value == 0.0:
            with pytest.raises(ValidationError, match="prompt-carry log probabilities"):
                _record(mask, modified)
        else:
            assert _record(mask, modified).generation_log_probs_delta[index] == -0.5


@pytest.mark.parametrize("bad_mask", [-1.0, 0.5, 2.0, math.nan, math.inf, -math.inf])
def test_invalid_mask_domain_fails(bad_mask: float) -> None:
    with pytest.raises(ValidationError, match="token_mask_delta"):
        _record([1.0, bad_mask, 1.0], [-0.25, 0.0, -0.25])


@pytest.mark.parametrize("mask", [[1.0] * 3, [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
@pytest.mark.parametrize("bad_logprob", [math.nan, math.inf, -math.inf, 1e39])
def test_fast_and_general_paths_preserve_float_checks(mask: list[float], bad_logprob: float) -> None:
    logprobs = [0.0 if value == 0.0 else -0.25 for value in mask]
    logprobs[2] = bad_logprob
    with pytest.raises((ValidationError, ValueError)):
        _record(mask, logprobs)


def test_negative_zero_preserves_its_digest_bytes() -> None:
    positive = _record([0.0, 1.0], [0.0, -0.25])
    negative = _record([-0.0, 1.0], [-0.0, -0.25])
    assert positive.digest != negative.digest
    assert StagedCallSnapshot.model_validate_json(negative.model_dump_json()).model_dump() == negative.model_dump()


@pytest.mark.parametrize("mask,logprobs", [([0, 1.0], [0.0, -0.25]), ([0.0, 1.0], [0, -0.25])])
def test_local_build_still_requires_exact_float_columns(mask: list[float], logprobs: list[float]) -> None:
    with pytest.raises(ValueError, match="Python floats"):
        _record(mask, logprobs)
