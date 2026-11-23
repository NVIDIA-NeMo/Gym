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

"""Packed local construction preserves the validated schema-v3 record bytes."""

from typing import Any

import pytest
from pydantic import field_validator

from nemo_gym.models.token_capture.staging.digest import (
    PackedFloat32Values,
    PackedTokenIds,
    compute_chain_hash,
    compute_extras_digest,
    hash_token_ids,
)
from nemo_gym.models.token_capture.staging.records import CaptureAdmission, StagedCallRecord


@pytest.mark.parametrize("length", [1, 17, 8192])
@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("extras", [None, {"nested": ["x", -0.0, {"v": 4}]}])
def test_packed_record_matches_explicit_hash_construction(
    length: int, child: bool, extras: dict[str, Any] | None
) -> None:
    prefix = [9, 8] if child else []
    admission = CaptureAdmission(
        capture_key="episode-a1",
        model_call_id="call",
        parent_call_id="parent" if child else None,
        prev_len=len(prefix),
        mode="token_in" if child else "text",
        required_prefix_token_ids=prefix,
        parent_chain_hash=compute_chain_hash(None, prefix) if child else None,
    )
    tokens = list(range(length))
    masks = [0.0] * (length // 3) + [1.0] * (length - length // 3)
    logprobs = [0.0 if mask == 0 else -0.123456789 for mask in masks]
    packed = StagedCallRecord.from_token_components(
        admission=admission,
        weight_version=7,
        prefix_token_ids=prefix,
        token_ids_delta=tokens,
        token_mask_delta=masks,
        generation_log_probs_delta=logprobs,
        extras=extras,
    )
    expected = StagedCallRecord.from_components(
        capture_key=admission.capture_key,
        model_call_id=admission.model_call_id,
        parent_call_id=admission.parent_call_id,
        mode=admission.mode,
        prev_len=len(prefix),
        delta_len=length,
        cum_len=len(prefix) + length,
        weight_version=7,
        token_ids_delta=tokens,
        token_mask_delta=masks,
        generation_log_probs_delta=logprobs,
        extras=extras,
        extras_digest=compute_extras_digest(extras),
        chain_hash=compute_chain_hash(admission.parent_chain_hash, tokens),
        cumulative_hash=hash_token_ids(prefix + tokens),
    )
    assert packed.model_dump_json() == expected.model_dump_json()
    assert StagedCallRecord.model_validate_json(packed.model_dump_json()) == expected
    tokens[0] = 999
    logprobs[-1] = -99.0
    assert packed.token_ids_delta[0] != 999
    assert packed.generation_log_probs_delta[-1] != -99.0
    wire = packed.model_dump()
    wire["token_ids_delta"][0] += 1
    with pytest.raises(ValueError, match="digest does not match"):
        StagedCallRecord.model_validate(wire)


@pytest.mark.parametrize("bad", [0, True, float("nan"), float("inf"), 1e100])
def test_packed_local_builder_rejects_noncanonical_logprobs(bad: Any) -> None:
    with pytest.raises(ValueError):
        StagedCallRecord.from_token_components(
            admission=CaptureAdmission(capture_key="r", model_call_id="c", mode="text"),
            weight_version=0,
            prefix_token_ids=[],
            token_ids_delta=[1],
            token_mask_delta=[1.0],
            generation_log_probs_delta=[bad],
            extras=None,
        )


def _local_record(
    *, masks: list[float], logprobs: list[float], record_type: type[StagedCallRecord] = StagedCallRecord
) -> StagedCallRecord:
    return record_type.from_token_components(
        admission=CaptureAdmission(capture_key="episode", model_call_id="call", mode="text"),
        weight_version=0,
        prefix_token_ids=[],
        token_ids_delta=list(range(len(masks))),
        token_mask_delta=masks,
        generation_log_probs_delta=logprobs,
        extras={"value": 1},
    )


@pytest.mark.parametrize("masks", [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [-0.0, 1.0]])
def test_prompt_mask_patterns_preserve_wire_integrity(masks: list[float]) -> None:
    logprobs = [0.0 if mask == 0.0 else -0.123456789 for mask in masks]
    record = _local_record(masks=masks, logprobs=logprobs)
    assert StagedCallRecord.model_validate_json(record.model_dump_json()) == record
    for index, mask in enumerate(masks):
        if mask == 0.0:
            mutated = list(logprobs)
            mutated[index] = -0.25
            with pytest.raises(ValueError, match="prompt-carry log probabilities must be 0.0"):
                _local_record(masks=masks, logprobs=mutated)


def test_custom_field_validator_does_not_reuse_stale_encoding() -> None:
    class AdjustedRecord(StagedCallRecord):
        @field_validator("generation_log_probs_delta")
        @classmethod
        def adjust(cls, values: list[float]) -> list[float]:
            return [value / 2 for value in values]

    record = _local_record(masks=[1.0], logprobs=[-0.25], record_type=AdjustedRecord)
    assert record.generation_log_probs_delta == [-0.125]
    assert StagedCallRecord.model_validate_json(record.model_dump_json()) == StagedCallRecord.model_validate(
        record.model_dump()
    )


@pytest.mark.parametrize("bad", [True, 1.0, -1, 2**64])
def test_packed_ids_validate_before_reuse(bad: Any) -> None:
    with pytest.raises(ValueError, match="unsigned 64-bit integer"):
        PackedTokenIds([bad])


def test_packed_encodings_own_inputs_and_hide_payload_in_repr() -> None:
    tokens = [1, 2]
    values = [-0.0, -0.123456789]
    packed_tokens = PackedTokenIds(tokens)
    packed_values = PackedFloat32Values(values, field="logprobs")
    tokens[0] = 99
    values[0] = 0.0
    assert packed_tokens[:] == (1, 2)
    assert packed_values[:][0].hex() == (-0.0).hex()
    assert not packed_values.matches(values)
    assert not packed_values.matches([0.0, -0.123456789])
    assert len(repr(PackedTokenIds(list(range(8192))))) < 200
    assert len(repr(PackedFloat32Values([-0.125] * 8192, field="logprobs"))) < 200
    with pytest.raises(AttributeError):
        packed_tokens.encoded = b"changed"


def test_local_prefix_length_is_checked() -> None:
    with pytest.raises(ValueError, match="prefix length does not match prev_len"):
        StagedCallRecord.from_token_components(
            admission=CaptureAdmission(capture_key="episode", model_call_id="call", mode="text"),
            weight_version=0,
            prefix_token_ids=[9],
            token_ids_delta=[1],
            token_mask_delta=[1.0],
            generation_log_probs_delta=[-0.25],
            extras=None,
        )


@pytest.mark.parametrize("context", [None, {}, {"build": True}, object()])
def test_wire_validation_cannot_bypass_integrity_with_context(context: Any) -> None:
    record = _local_record(masks=[1.0], logprobs=[-0.25])
    payload = record.model_dump()
    payload["digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest does not match staged call contents"):
        StagedCallRecord.model_validate(payload, context=context)
    payload = record.model_dump()
    payload["extras"] = {"value": 2}
    with pytest.raises(ValueError, match="extras_digest does not match extras"):
        StagedCallRecord.model_validate(payload, context=context)
