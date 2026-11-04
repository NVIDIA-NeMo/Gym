# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local hashing optimization retains the frozen wire checks and exact types."""

import math
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from nemo_gym.token_id_capture.staging import records
from nemo_gym.token_id_capture.staging.digest import (
    compute_chain_hash,
    compute_extras_digest,
    encode_token_ids,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.records import (
    StagedCallRecord,
    StagedCallSnapshot,
)


def components():
    tokens = [1, 2]
    extras = {"routes": [[1, 2]], "flag": True}
    return dict(
        rollout_id="r",
        model_call_id="c",
        parent_call_id=None,
        mode="text",
        prev_len=0,
        delta_len=2,
        cum_len=2,
        weight_version=7,
        token_ids_delta=tokens,
        token_mask_delta=[0.0, 1.0],
        generation_log_probs_delta=[0.0, -0.25],
        extras=extras,
        extras_digest=compute_extras_digest(extras),
        chain_hash=compute_chain_hash(None, tokens),
        cumulative_hash=hash_token_ids(tokens),
    )


def test_derived_extras_hashes_once_and_keeps_wire_checks(monkeypatch):
    kwargs = components()
    expected = StagedCallRecord.from_components(**kwargs).model_dump()
    extras_hash = Mock(wraps=records.compute_extras_digest)
    staged_hash = Mock(wraps=records.compute_staging_digest)
    monkeypatch.setattr(records, "compute_extras_digest", extras_hash)
    monkeypatch.setattr(records, "compute_staging_digest", staged_hash)
    record = StagedCallRecord.from_components(**{**kwargs, "extras_digest": None})
    assert record.model_dump() == expected
    assert extras_hash.call_count == staged_hash.call_count == 1
    assert StagedCallSnapshot.model_validate_json(record.model_dump_json()).model_dump() == expected
    assert extras_hash.call_count == staged_hash.call_count == 2
    record.extras["flag"] = False
    with pytest.raises(ValidationError, match="extras_digest does not match"):
        StagedCallSnapshot.model_validate_json(record.model_dump_json())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("token_ids_delta", [1, -1]),
        ("token_ids_delta", [1, 2**64]),
        ("token_ids_delta", [True, 2]),
        ("token_mask_delta", [0.0, 0.5]),
        ("generation_log_probs_delta", [0.1, -0.25]),
        ("generation_log_probs_delta", [0.0, math.nan]),
        ("generation_log_probs_delta", [0.0, 1e39]),
        ("delta_len", 3),
    ],
)
def test_derived_local_build_retains_invalid_column_rejections(field, value):
    with pytest.raises((ValueError, ValidationError)):
        StagedCallRecord.from_components(**{**components(), "extras_digest": None, field: value})


def test_wire_cannot_opt_into_local_hash_derivation():
    record = StagedCallRecord.from_components(**components()).model_dump()
    for context in (None, {"derive_extras": True}):
        with pytest.raises(ValidationError):
            StagedCallSnapshot.model_validate({**record, "extras_digest": None}, context=context)
        with pytest.raises(ValidationError, match="digest does not match"):
            StagedCallSnapshot.model_validate({**record, "digest": "0" * 64}, context=context)


def test_exact_type_check_never_hashes_or_coerces_invalid_token_types():
    class Meta(type):
        def __hash__(self):
            raise AssertionError("type hashing must not occur")

    class Token(int, metaclass=Meta):
        pass

    with pytest.raises(ValueError, match="unsigned 64-bit"):
        encode_token_ids([1, Token(2)])
