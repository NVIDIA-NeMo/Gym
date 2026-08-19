# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial receipt verification and terminal-chain rebuild tests."""

import base64
import struct
from typing import Any

import pytest

from nemo_gym.token_id_capture.staging.digest import (
    EXTRAS_DIGEST_VERSION,
    STAGING_DIGEST_VERSION,
    STAGING_SCHEMA_VERSION,
    compute_extras_digest,
    compute_staging_digest,
)
from nemo_gym.token_id_capture.staging.rebuild import (
    ReceiptVerificationError,
    verify_and_linearize,
)
from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    RolloutReceipt,
    StagedCallSnapshot,
)


def _snapshot(
    model_call_id: str,
    *,
    parent_call_id: str | None = None,
    prev_len: int = 0,
    token_ids: list[int],
    masks: list[float],
    logprobs: list[float],
    weight_version: int = 7,
    extras: dict[str, Any] | None = None,
) -> StagedCallSnapshot:
    mode = "text" if parent_call_id is None else "token_in"
    extras_digest = compute_extras_digest(extras)
    delta_len = len(token_ids)
    cum_len = prev_len + delta_len
    digest = compute_staging_digest(
        schema_version=STAGING_SCHEMA_VERSION,
        digest_version=STAGING_DIGEST_VERSION,
        extras_digest_version=EXTRAS_DIGEST_VERSION,
        rollout_id="rollout-1",
        model_call_id=model_call_id,
        parent_call_id=parent_call_id,
        mode=mode,
        prev_len=prev_len,
        delta_len=delta_len,
        cum_len=cum_len,
        weight_version=weight_version,
        token_ids_delta=token_ids,
        token_mask_delta=masks,
        generation_log_probs_delta=logprobs,
        extras_digest=extras_digest,
    )
    return StagedCallSnapshot(
        rollout_id="rollout-1",
        model_call_id=model_call_id,
        parent_call_id=parent_call_id,
        mode=mode,
        prev_len=prev_len,
        delta_len=delta_len,
        cum_len=cum_len,
        weight_version=weight_version,
        digest=digest,
        token_ids_delta=token_ids,
        token_mask_delta=masks,
        generation_log_probs_delta=logprobs,
        extras=extras,
        extras_digest=extras_digest,
    )


def _manifest_row(snapshot: StagedCallSnapshot, *, staging_key: str | None = None) -> CallRecord:
    return CallRecord(
        model_call_id=snapshot.model_call_id,
        parent_call_id=snapshot.parent_call_id,
        prev_len=snapshot.prev_len,
        delta_len=snapshot.delta_len,
        cum_len=snapshot.cum_len,
        weight_version=snapshot.weight_version,
        digest=snapshot.digest,
        extras_digest=snapshot.extras_digest,
        staging_key=staging_key or f"row/{snapshot.model_call_id}",
        mode=snapshot.mode,
        chain_hash=snapshot.chain_hash,
        cumulative_hash=snapshot.cumulative_hash,
    )


def _receipt(
    snapshots: list[StagedCallSnapshot],
    *,
    terminal: str,
    poisoned: bool = False,
) -> RolloutReceipt:
    return RolloutReceipt(
        rollout_id="rollout-1",
        terminal_model_call_id=terminal,
        manifest=[_manifest_row(snapshot) for snapshot in snapshots],
        capture_poisoned=poisoned,
    )


def _branched() -> tuple[RolloutReceipt, list[StagedCallSnapshot]]:
    root = _snapshot(
        "root",
        token_ids=[10, 11, 12],
        masks=[0.0, 0.0, 1.0],
        logprobs=[0.0, 0.0, -0.1],
        weight_version=3,
    )
    main = _snapshot(
        "main",
        parent_call_id="root",
        prev_len=3,
        token_ids=[20, 21],
        masks=[0.0, 1.0],
        logprobs=[0.0, -0.2],
        weight_version=4,
    )
    sibling = _snapshot(
        "sibling",
        parent_call_id="root",
        prev_len=3,
        token_ids=[30, 31, 32],
        masks=[0.0, 1.0, 1.0],
        logprobs=[0.0, -0.3, -0.4],
        weight_version=99,
    )
    snapshots = [root, sibling, main]
    return _receipt(snapshots, terminal="main"), snapshots


def test_verify_and_linearize_selects_only_terminal_ancestry() -> None:
    receipt, snapshots = _branched()
    row = verify_and_linearize(receipt, snapshots)
    assert row.model_call_ids == ["root", "main"]
    assert row.token_ids == [10, 11, 12, 20, 21]
    assert row.token_mask == [0.0, 0.0, 1.0, 0.0, 1.0]
    assert row.logprobs == [0.0, 0.0, -0.1, 0.0, -0.2]
    assert row.prompt_len == 2
    assert row.weight_versions == [3, 4]
    assert row.link_spans == [("root", 2, 1), ("main", 1, 1)]
    assert [(span.start, span.end) for span in row.weight_version_spans] == [(0, 3), (3, 5)]


def test_verification_is_retry_safe_and_deterministic() -> None:
    receipt, snapshots = _branched()
    assert verify_and_linearize(receipt, snapshots) == verify_and_linearize(receipt, snapshots)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("token_ids_delta", [999, 11, 12], "corrupt_digest"),
        ("generation_log_probs_delta", [0.0, 0.0, -9.0], "corrupt_digest"),
        ("weight_version", 8, "wrong_weight_version"),
        ("parent_call_id", "other", "wrong_parent_call_id"),
        ("cum_len", 99, "wrong_cum_len"),
    ],
)
def test_snapshot_mutation_is_rejected(field: str, value: Any, code: str) -> None:
    receipt, snapshots = _branched()
    snapshots = [snapshots[0].model_copy(update={field: value}), *snapshots[1:]]
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize(receipt, snapshots)
    assert error.value.code == code


def test_extras_mutation_is_rejected() -> None:
    routes = [[[1, 2]], [[3, 4]], [[5, 6]]]
    root = _snapshot(
        "root",
        token_ids=[10, 11, 12],
        masks=[0.0, 0.0, 1.0],
        logprobs=[0.0, 0.0, -0.1],
        extras={"routed_experts": routes},
    )
    receipt = _receipt([root], terminal="root")
    corrupt = root.model_copy(update={"extras": {"routed_experts": [[[9, 9]]] * 3}})
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize(receipt, [corrupt])
    assert error.value.code == "corrupt_extras"


@pytest.mark.parametrize(
    ("snapshots_transform", "code"),
    [
        (lambda rows: rows[:-1], "row_count_mismatch"),
        (lambda rows: rows + [rows[0]], "row_count_mismatch"),
        (lambda rows: [rows[0], rows[0], rows[2]], "duplicate_snapshot"),
    ],
)
def test_missing_extra_and_duplicate_rows_are_rejected(snapshots_transform: Any, code: str) -> None:
    receipt, snapshots = _branched()
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize(receipt, snapshots_transform(snapshots))
    assert error.value.code == code


def test_snapshot_order_binds_manifest_keys_to_identities() -> None:
    receipt, snapshots = _branched()
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize(receipt, list(reversed(snapshots)))
    assert error.value.code == "snapshot_order_mismatch"


def test_poisoned_failed_and_unsupported_receipts_are_rejected() -> None:
    receipt, snapshots = _branched()
    poisoned = receipt.model_copy(update={"capture_poisoned": True})
    failed = receipt.model_copy(update={"failure_reason": "stage failed"})
    unsupported = receipt.model_copy(update={"schema_version": 999})
    for candidate, code in (
        (poisoned, "capture_poisoned"),
        (failed, "rollout_failed"),
        (unsupported, "unsupported_schema"),
    ):
        with pytest.raises(ReceiptVerificationError) as error:
            verify_and_linearize(candidate, snapshots)
        assert error.value.code == code


def test_missing_terminal_and_wrong_parent_length_are_rejected() -> None:
    receipt, snapshots = _branched()
    no_terminal = receipt.model_copy(update={"terminal_model_call_id": None})
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize(no_terminal, snapshots)
    assert error.value.code == "missing_terminal"

    bad_main = receipt.manifest[2].model_copy(update={"prev_len": 2, "cum_len": 4})
    bad_snapshot = snapshots[2].model_copy(update={"prev_len": 2, "cum_len": 4})
    wrong_length = receipt.model_copy(update={"manifest": [receipt.manifest[0], receipt.manifest[1], bad_main]})
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize(wrong_length, [snapshots[0], snapshots[1], bad_snapshot])
    assert error.value.code in {"invalid_snapshot", "parent_length_mismatch"}


def _route_envelope(values: list[int], *, shape: str, dtype: str = "int16") -> str:
    encoded = base64.b64encode(struct.pack(f"<{len(values)}h", *values)).decode("ascii")
    return f"nrlre1:{dtype}:{shape}:{encoded}"


def test_routed_expert_envelopes_rebuild_and_missing_spans_use_sentinel() -> None:
    root = _snapshot(
        "root",
        token_ids=[10, 11],
        masks=[0.0, 1.0],
        logprobs=[0.0, -0.1],
        extras={"routed_experts": _route_envelope([1, 2, 3, 4], shape="2x1x2")},
    )
    child = _snapshot(
        "child",
        parent_call_id="root",
        prev_len=2,
        token_ids=[20, 21],
        masks=[0.0, 1.0],
        logprobs=[0.0, -0.2],
    )
    row = verify_and_linearize(_receipt([root, child], terminal="child"), [root, child])
    assert row.routed_experts == [[[1, 2]], [[3, 4]], [[-1, -1]], [[-1, -1]]]
    assert row.routed_experts_dtype == "int16"


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ("nrlre2:int16:1x1x1:AA==", "invalid_routes"),
        (_route_envelope([1, 2], shape="1x1x2"), "route_length_mismatch"),
    ],
)
def test_unknown_or_misaligned_route_envelope_is_rejected(payload: str, code: str) -> None:
    root = _snapshot(
        "root",
        token_ids=[10, 11],
        masks=[0.0, 1.0],
        logprobs=[0.0, -0.1],
        extras={"routed_experts": payload},
    )
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize(_receipt([root], terminal="root"), [root])
    assert error.value.code == code
