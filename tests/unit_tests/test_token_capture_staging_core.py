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
"""Contract tests for framework-owned token staging."""

import math
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.token_id_capture.staging.conformance import assert_golden_vectors, load_golden_vectors
from nemo_gym.token_id_capture.staging.digest import (
    EMPTY_EXTRAS_DIGEST,
    STAGING_DIGEST_VERSION,
    STAGING_SCHEMA_VERSION,
    build_staging_delta,
    compute_extras_digest,
    compute_staging_digest,
    encode_token_ids,
)
from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    CaptureAdmission,
    CommitCoords,
    RolloutReceipt,
    StagedCallRecord,
    StagedCallSnapshot,
    StageResult,
    staging_key,
)


def _digest_args() -> dict:
    extras_digest = compute_extras_digest({"routed_experts": [[[1, 2]], [[3, 4]]]})
    return {
        "schema_version": STAGING_SCHEMA_VERSION,
        "digest_version": STAGING_DIGEST_VERSION,
        "extras_digest_version": 1,
        "rollout_id": "rollout-1",
        "model_call_id": "call-1",
        "parent_call_id": "parent-1",
        "mode": "token_in",
        "prev_len": 3,
        "delta_len": 2,
        "cum_len": 5,
        "weight_version": 7,
        "token_ids_delta": [10, 11],
        "token_mask_delta": [0.0, 1.0],
        "generation_log_probs_delta": [0.0, -0.25],
        "extras_digest": extras_digest,
        "chain_hash": "1" * 64,
        "cumulative_hash": "2" * 64,
    }


def _record_payload() -> dict:
    args = _digest_args()
    return {
        **args,
        "digest": compute_staging_digest(**args),
        "extras": {"routed_experts": [[[1, 2]], [[3, 4]]]},
    }


def test_digest_golden_vectors() -> None:
    assert_golden_vectors()
    vectors = load_golden_vectors()
    assert compute_extras_digest(None) == EMPTY_EXTRAS_DIGEST
    assert compute_extras_digest({"b": [2, 3], "a": True}) == compute_extras_digest({"a": True, "b": [2, 3]})
    assert compute_extras_digest({"routed_experts": [[[1, 2]], [[3, 4]]]}) == vectors["extras_digest"]
    assert compute_staging_digest(**_digest_args()) == vectors["staged_call_digest"]
    assert encode_token_ids([1]) != encode_token_ids([1, 0])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rollout_id", "rollout-2"),
        ("model_call_id", "call-2"),
        ("parent_call_id", "parent-2"),
        ("mode", "text"),
        ("prev_len", 2),
        ("delta_len", 3),
        ("cum_len", 6),
        ("weight_version", 8),
        ("token_ids_delta", [10, 12]),
        ("token_mask_delta", [1.0, 1.0]),
        ("generation_log_probs_delta", [0.0, -0.5]),
        ("extras_digest", "3" * 64),
        ("chain_hash", "4" * 64),
        ("cumulative_hash", "5" * 64),
    ],
)
def test_digest_rejects_or_changes_every_mutable_field(field: str, value) -> None:
    base = _digest_args()
    reference = compute_staging_digest(**base)
    mutated = {**base, field: value}
    try:
        assert compute_staging_digest(**mutated) != reference
    except ValueError:
        # Structurally inconsistent mutations are rejected before hashing.
        pass


@pytest.mark.parametrize(
    ("version_field", "version"),
    [("schema_version", 1), ("digest_version", 1), ("extras_digest_version", 2)],
)
def test_digest_rejects_unknown_versions(version_field: str, version: int) -> None:
    args = {**_digest_args(), version_field: version}
    with pytest.raises(ValueError, match="unsupported .*digest|unsupported staging"):
        compute_staging_digest(**args)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_digest_rejects_nonfinite_numbers(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_staging_digest(**{**_digest_args(), "generation_log_probs_delta": [0.0, bad]})
    with pytest.raises(ValueError, match="finite"):
        compute_extras_digest({"value": bad})


def test_extras_reject_ambiguous_or_unsupported_types() -> None:
    with pytest.raises(TypeError, match="string keys"):
        compute_extras_digest({1: "bad"})
    with pytest.raises(TypeError, match="unsupported"):
        compute_extras_digest({"value": (1, 2)})
    with pytest.raises(ValueError, match="signed 64-bit"):
        compute_extras_digest({"value": 2**80})


def test_build_staging_delta_layout_and_errors() -> None:
    ids, mask, log_probs = build_staging_delta(
        prompt_token_ids=[1, 2, 3, 4],
        generated_token_ids=[5, 6],
        generated_log_probs=[-0.5, -0.25],
        prev_len=2,
    )
    assert ids == [3, 4, 5, 6]
    assert mask == [0.0, 0.0, 1.0, 1.0]
    assert log_probs == [0.0, 0.0, -0.5, -0.25]
    with pytest.raises(ValueError, match="outside prompt length"):
        build_staging_delta(prompt_token_ids=[1], generated_token_ids=[2], generated_log_probs=[-1.0], prev_len=5)
    with pytest.raises(ValueError, match="lengths differ"):
        build_staging_delta(prompt_token_ids=[1], generated_token_ids=[2], generated_log_probs=[], prev_len=0)


def test_admission_uses_canonical_model_call_identity_and_exact_prefix() -> None:
    admission = CaptureAdmission(
        rollout_id="r",
        model_call_id="c",
        parent_call_id="p",
        prev_len=2,
        mode="token_in",
        required_prefix_token_ids=[10, 11],
        parent_chain_hash="1" * 64,
    )
    assert admission.model_dump(mode="json")["model_call_id"] == "c"
    with pytest.raises(ValidationError, match="length must equal prev_len"):
        CaptureAdmission(
            rollout_id="r",
            model_call_id="c",
            parent_call_id="p",
            prev_len=2,
            mode="token_in",
            required_prefix_token_ids=[10],
            parent_chain_hash="1" * 64,
        )
    with pytest.raises(ValidationError, match="parent's chain hash"):
        CaptureAdmission(
            rollout_id="r",
            model_call_id="c",
            parent_call_id="p",
            prev_len=2,
            mode="token_in",
            required_prefix_token_ids=[10, 11],
        )
    with pytest.raises(ValidationError, match="text admission"):
        CaptureAdmission(
            rollout_id="r",
            model_call_id="c",
            parent_call_id="p",
            prev_len=2,
            mode="text",
            required_prefix_token_ids=[10, 11],
        )
    chain_admission = CaptureAdmission(
        rollout_id="r",
        model_call_id="c",
        parent_call_id="p",
        prev_len=2,
        mode="token_in",
        staging_chain=["r/p"],
        parent_chain_hash="1" * 64,
    )
    assert chain_admission.required_prefix_token_ids == []
    with pytest.raises(ValidationError, match="text admission"):
        CaptureAdmission(
            rollout_id="r",
            model_call_id="c",
            mode="text",
            staging_chain=["r/p"],
        )
    with pytest.raises(ValidationError, match="text admission"):
        CaptureAdmission(
            rollout_id="r",
            model_call_id="c",
            mode="text",
            parent_chain_hash="1" * 64,
        )


def test_staged_record_and_snapshot_round_trip() -> None:
    record = StagedCallRecord.model_validate(_record_payload())
    assert record.staging_key == staging_key("rollout-1", "call-1")
    snapshot = StagedCallSnapshot.model_validate(record.model_dump())
    assert snapshot.extras == record.extras
    assert StagedCallSnapshot.model_validate(snapshot.model_dump()) == snapshot


@pytest.mark.parametrize(
    ("version_field", "version"),
    [
        ("schema_version", 1),
        ("schema_version", 3),
        ("digest_version", 1),
        ("digest_version", 3),
        ("extras_digest_version", 0),
        ("extras_digest_version", 2),
    ],
)
def test_wire_records_reject_old_and_new_versions(version_field: str, version: int) -> None:
    with pytest.raises(ValidationError, match=version_field):
        StagedCallRecord.model_validate({**_record_payload(), version_field: version})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parent_call_id", "other-parent"),
        ("mode", "text"),
        ("weight_version", 8),
        ("token_ids_delta", [10, 12]),
        ("extras", {"routed_experts": [[[9, 9]], [[3, 4]]]}),
    ],
)
def test_staged_record_rejects_digest_covered_mutation(field: str, value) -> None:
    with pytest.raises(ValidationError, match="digest|mode"):
        StagedCallRecord.model_validate({**_record_payload(), field: value})


def test_coords_enforce_staged_and_failed_shapes() -> None:
    payload = _record_payload()
    staged = CommitCoords(
        rollout_id=payload["rollout_id"],
        model_call_id=payload["model_call_id"],
        parent_call_id=payload["parent_call_id"],
        prev_len=payload["prev_len"],
        delta_len=payload["delta_len"],
        cum_len=payload["cum_len"],
        weight_version=payload["weight_version"],
        digest=payload["digest"],
        extras_digest=payload["extras_digest"],
        staging_key="backend/key",
        chain_hash=payload["chain_hash"],
        cumulative_hash=payload["cumulative_hash"],
    )
    assert staged.disposition == "staged"
    with pytest.raises(ValidationError, match="chain_hash"):
        CommitCoords.model_validate({**staged.model_dump(), "chain_hash": None})
    failed = CommitCoords(
        rollout_id="r",
        model_call_id="c",
        prev_len=0,
        delta_len=0,
        cum_len=0,
        weight_version=1,
        disposition="capture_failed",
    )
    assert failed.digest is None
    with pytest.raises(ValidationError, match="cannot carry staged payload"):
        failed.model_copy(update={"digest": "1" * 64}).__class__.model_validate(
            {**failed.model_dump(), "digest": "1" * 64}
        )


def test_stage_result_and_receipt_validate_identity() -> None:
    assert StageResult(ok=True, staging_key="backend/key").staging_key == "backend/key"
    with pytest.raises(ValidationError, match="requires an error"):
        StageResult(ok=False)

    payload = _record_payload()
    manifest = CallRecord(
        model_call_id=payload["model_call_id"],
        parent_call_id=payload["parent_call_id"],
        prev_len=payload["prev_len"],
        delta_len=payload["delta_len"],
        cum_len=payload["cum_len"],
        weight_version=payload["weight_version"],
        digest=payload["digest"],
        extras_digest=payload["extras_digest"],
        staging_key="backend/key",
        mode=payload["mode"],
        response_id="chatcmpl-call-1",
    )
    receipt = RolloutReceipt(
        rollout_id="rollout-1",
        terminal_model_call_id="call-1",
        manifest=[manifest],
        terminal_selection="declared",
    )
    assert receipt.terminal_model_call_id == "call-1"
    with pytest.raises(ValidationError, match="absent"):
        RolloutReceipt(
            rollout_id="rollout-1",
            terminal_model_call_id="ghost",
            manifest=[manifest],
            terminal_selection="declared",
        )


def test_staging_namespace_has_no_serving_or_framework_dependencies() -> None:
    import nemo_gym.token_id_capture.staging as staging

    root = Path(staging.__file__).parent
    modules = ["nemo_gym.token_id_capture.staging"]
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).with_suffix("")
        parts = [part for part in relative.parts if part != "__init__"]
        if parts:
            modules.append(".".join(["nemo_gym.token_id_capture.staging", *parts]))
    forbidden = ("fastapi", "ray", "torch", "transfer_queue", "aiohttp")
    for module in modules:
        code = (
            f"import {module}, sys; "
            f"bad = sorted(set(sys.modules) & set({forbidden!r})); "
            "assert not bad, f'forbidden imports: {bad}'"
        )
        subprocess.run([sys.executable, "-c", code], check=True)
