# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local packing preserves wire digests and rejects malformed token columns."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import field_validator

from nemo_gym.token_id_capture.staging.digest import (
    PackedTokenIds,
    compute_chain_hash,
    encode_token_ids,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.records import CaptureAdmission, StagedCallRecord


def build(*, child: bool = False, length: int = 17, **overrides) -> StagedCallRecord:
    prefix = [5, 6] if child else []
    admission = CaptureAdmission(
        rollout_id="rollout",
        model_call_id="call",
        mode="token_in" if child else "text",
        prev_len=len(prefix),
        parent_call_id="parent" if child else None,
        parent_chain_hash=compute_chain_hash(None, prefix) if child else None,
        required_prefix_token_ids=prefix,
    )
    fields = dict(
        admission=admission,
        weight_version=3,
        prefix_token_ids=prefix,
        token_ids_delta=list(range(length)),
        token_mask_delta=[1.0] * length,
        generation_log_probs_delta=[-0.1] * length,
        extras={"route_chunks": ["a", "b"]},
    )
    fields.update(overrides)
    return StagedCallRecord.from_token_components(**fields)


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("length", [1, 17, 32768])
def test_local_builder_is_wire_identical_to_original(child: bool, length: int) -> None:
    result = build(child=child, length=length)
    prefix = [5, 6] if child else []
    parent_chain = compute_chain_hash(None, prefix) if child else None
    fields = result.model_dump()
    for name in ("schema_version", "digest_version", "extras_digest_version", "digest"):
        fields.pop(name)
    fields["chain_hash"] = compute_chain_hash(parent_chain, result.token_ids_delta)
    fields["cumulative_hash"] = hash_token_ids(prefix + result.token_ids_delta)
    original = StagedCallRecord.from_components(**fields)
    assert result.model_dump() == original.model_dump()
    assert StagedCallRecord.model_validate_json(result.model_dump_json()).model_dump() == result.model_dump()


@pytest.mark.parametrize(
    "field,values",
    [
        ("token_ids_delta", [True]),
        ("token_ids_delta", [-1]),
        ("token_ids_delta", [2**64]),
        ("token_ids_delta", [1.0]),
        ("token_mask_delta", [1]),
        ("token_mask_delta", [0.5]),
        ("generation_log_probs_delta", [0]),
        ("generation_log_probs_delta", [float("nan")]),
        ("generation_log_probs_delta", [float("inf")]),
        ("generation_log_probs_delta", [1e100]),
        ("generation_log_probs_delta", []),
    ],
)
def test_invalid_columns_still_rejected(field: str, values: list) -> None:
    with pytest.raises((ValueError, TypeError)):
        build(length=1, **{field: values})


def test_prompt_carry_and_prefix_lengths_are_checked() -> None:
    with pytest.raises(ValueError, match="prompt-carry"):
        build(length=1, token_mask_delta=[0.0], generation_log_probs_delta=[-1.0])
    with pytest.raises(ValueError, match="prefix length"):
        build(child=True, prefix_token_ids=[5])
    result = build(length=2, token_mask_delta=[-0.0, 1.0], generation_log_probs_delta=[-0.0, -1e-45])
    StagedCallRecord.model_validate_json(result.model_dump_json())


def test_packed_tokens_own_values_and_combine_identically() -> None:
    values = [0, 2**64 - 1]
    packed = PackedTokenIds(values)
    values[0] = 9
    assert list(packed) == [0, 2**64 - 1]
    assert packed.encoded == encode_token_ids([0, 2**64 - 1])
    assert packed.prepend([5, 6]).encoded == encode_token_ids([5, 6, 0, 2**64 - 1])
    assert compute_chain_hash(None, packed) == compute_chain_hash(None, list(packed))
    assert hash_token_ids(packed) == hash_token_ids(list(packed))
    with pytest.raises(AttributeError):
        packed.encoded = b"corrupt"


@pytest.mark.parametrize("field", ["token_ids_delta", "generation_log_probs_delta", "digest", "chain_hash", "extras"])
def test_wire_tampering_still_rejected(field: str) -> None:
    payload = build().model_dump()
    if field == "token_ids_delta":
        payload[field][0] += 1
    elif field == "generation_log_probs_delta":
        payload[field][0] = -3.0
    elif field == "extras":
        payload[field] = {"route_chunks": ["corrupt"]}
    else:
        payload[field] = "f" * 64
    with pytest.raises(ValueError):
        StagedCallRecord.model_validate(payload)


def test_parallel_builds_have_no_shared_cache() -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda length: build(length=length), range(1, 17)))
    assert len({record.digest for record in results}) == 16
    for record in results:
        StagedCallRecord.model_validate_json(record.model_dump_json())


def test_custom_field_validation_cannot_reuse_stale_float_bytes() -> None:
    class ChangedMaskRecord(StagedCallRecord):
        @field_validator("token_mask_delta")
        @classmethod
        def replace_positive_zero(cls, values: list[float]) -> list[float]:
            return [-0.0 if value == 0.0 else value for value in values]

    admission = CaptureAdmission(rollout_id="r", model_call_id="c", mode="text")
    result = ChangedMaskRecord.from_token_components(
        admission=admission,
        weight_version=0,
        prefix_token_ids=[],
        token_ids_delta=[5],
        token_mask_delta=[0.0],
        generation_log_probs_delta=[0.0],
        extras=None,
    )
    # The ordinary wire validator must accept the digest of the *changed* column.
    StagedCallRecord.model_validate_json(result.model_dump_json())
