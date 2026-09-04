# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial receipt verification and metadata-only terminal-chain rebuild tests."""

from typing import Any

import pytest

from nemo_gym.token_id_capture.staging.digest import (
    EXTRAS_DIGEST_VERSION,
    STAGING_DIGEST_VERSION,
    STAGING_SCHEMA_VERSION,
    compute_chain_hash,
    compute_extras_digest,
    compute_staging_digest,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.rebuild import (
    ReceiptVerificationError,
    verify_and_linearize,
)
from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    RolloutReceipt,
    StagedCallBaseSnapshot,
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
    parent: StagedCallBaseSnapshot | None = None,
    prefix_token_ids: list[int] | None = None,
    chain_hash: str | None = None,
    cumulative_hash: str | None = None,
) -> StagedCallBaseSnapshot:
    """Build a snapshot whose chain digests extend ``parent`` unless overridden.

    ``prefix_token_ids`` are the linearized tokens preceding this delta; they
    default to the parent's own delta (a two-level chain).
    """
    mode = "text" if parent_call_id is None else "token_in"
    if prefix_token_ids is None:
        prefix_token_ids = list(parent.token_ids_delta) if parent is not None else []
    if chain_hash is None:
        chain_hash = compute_chain_hash(parent.chain_hash if parent is not None else None, token_ids)
    if cumulative_hash is None:
        cumulative_hash = hash_token_ids(prefix_token_ids + token_ids)
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
        chain_hash=chain_hash,
        cumulative_hash=cumulative_hash,
    )
    return StagedCallBaseSnapshot(
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
        extras_digest=extras_digest,
        chain_hash=chain_hash,
        cumulative_hash=cumulative_hash,
    )


def _manifest_row(snapshot: StagedCallBaseSnapshot, *, staging_key: str | None = None) -> CallRecord:
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
        response_id=f"chatcmpl-{snapshot.model_call_id}",
    )


def _receipt(
    snapshots: list[StagedCallBaseSnapshot],
    *,
    terminal: str,
    poisoned: bool = False,
) -> RolloutReceipt:
    return RolloutReceipt(
        rollout_id="rollout-1",
        terminal_model_call_id=terminal,
        manifest=[_manifest_row(snapshot) for snapshot in snapshots],
        capture_poisoned=poisoned,
        terminal_selection="declared",
    )


def _branched() -> tuple[RolloutReceipt, list[StagedCallBaseSnapshot]]:
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
        parent=root,
    )
    sibling = _snapshot(
        "sibling",
        parent_call_id="root",
        prev_len=3,
        token_ids=[30, 31, 32],
        masks=[0.0, 1.0, 1.0],
        logprobs=[0.0, -0.3, -0.4],
        weight_version=99,
        parent=root,
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


def test_extras_commitments_cover_the_selected_chain_in_order() -> None:
    receipt, snapshots = _branched()
    row = verify_and_linearize(receipt, snapshots)
    assert [commitment.model_call_id for commitment in row.extras_commitments] == ["root", "main"]
    committed = {record.model_call_id: record for record in receipt.manifest}
    for commitment in row.extras_commitments:
        assert commitment.extras_digest == committed[commitment.model_call_id].extras_digest
        assert commitment.extras_digest_version == EXTRAS_DIGEST_VERSION


def test_committed_extras_verify_and_corrupt_extras_fail_at_point_of_use() -> None:
    """The verifier never reads extras; consumers verify them via commitments."""
    extras = {"routed_experts": [[[1, 2]], [[3, 4]], [[5, 6]]], "note": "x"}
    root = _snapshot(
        "root",
        token_ids=[10, 11, 12],
        masks=[0.0, 0.0, 1.0],
        logprobs=[0.0, 0.0, -0.1],
        extras=extras,
    )
    receipt = _receipt([root], terminal="root")
    row = verify_and_linearize(receipt, [root])
    (commitment,) = row.extras_commitments
    assert compute_extras_digest(extras) == commitment.extras_digest
    corrupted = {**extras, "routed_experts": [[[9, 9]]] * 3}
    assert compute_extras_digest(corrupted) != commitment.extras_digest


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("token_ids_delta", [999, 11, 12], "corrupt_digest"),
        ("token_mask_delta", [0.0, 1.0, 1.0], "corrupt_digest"),
        ("generation_log_probs_delta", [0.0, 0.0, -9.0], "corrupt_digest"),
        ("weight_version", 8, "wrong_weight_version"),
        ("parent_call_id", "other", "wrong_parent_call_id"),
        ("prev_len", 1, "wrong_prev_len"),
        ("cum_len", 99, "wrong_cum_len"),
        ("mode", "token_in", "wrong_mode"),
        ("rollout_id", "rollout-2", "wrong_rollout"),
        ("model_call_id", "other", "snapshot_identity_mismatch"),
        ("digest", "0" * 64, "wrong_digest"),
        ("extras_digest", "0" * 64, "wrong_extras_digest"),
        ("chain_hash", "0" * 64, "wrong_chain_hash"),
        ("cumulative_hash", "0" * 64, "wrong_cumulative_hash"),
        ("schema_version", 999, "wrong_schema_version"),
        ("digest_version", 999, "wrong_digest_version"),
        ("extras_digest_version", 999, "wrong_extras_digest_version"),
    ],
)
def test_snapshot_mutation_is_rejected(field: str, value: Any, code: str) -> None:
    receipt, snapshots = _branched()
    snapshots = [snapshots[0].model_copy(update={field: value}), *snapshots[1:]]
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize(receipt, snapshots)
    assert error.value.code == code


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


def test_non_base_snapshot_values_are_rejected() -> None:
    receipt, snapshots = _branched()
    with pytest.raises(TypeError):
        verify_and_linearize(receipt, [snapshots[0].model_dump(), *snapshots[1:]])  # type: ignore[list-item]


def test_poisoned_failed_and_unsupported_receipts_are_rejected() -> None:
    receipt, snapshots = _branched()
    poisoned = receipt.model_copy(update={"capture_poisoned": True})
    failed = receipt.model_copy(update={"failure_reason": "stage failed"})
    unsupported = receipt.model_copy(update={"schema_version": 999})
    bad_digest_version = receipt.model_copy(update={"digest_version": 999})
    bad_extras_version = receipt.model_copy(update={"extras_digest_version": 999})
    for candidate, code in (
        (poisoned, "capture_poisoned"),
        (failed, "rollout_failed"),
        (unsupported, "unsupported_schema"),
        (bad_digest_version, "unsupported_digest"),
        (bad_extras_version, "unsupported_extras_digest"),
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

    bad_snapshot = _snapshot(
        "main",
        parent_call_id="root",
        prev_len=2,
        token_ids=[20, 21],
        masks=[0.0, 1.0],
        logprobs=[0.0, -0.2],
        weight_version=4,
        parent=snapshots[0],
    )
    bad_main = _manifest_row(
        bad_snapshot,
        staging_key=receipt.manifest[2].staging_key,
    )
    wrong_length = receipt.model_copy(update={"manifest": [receipt.manifest[0], receipt.manifest[1], bad_main]})
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize(wrong_length, [snapshots[0], snapshots[1], bad_snapshot])
    assert error.value.code == "parent_length_mismatch"


def test_missing_parent_and_empty_generation_are_rejected() -> None:
    orphan = _snapshot("orphan", parent_call_id="ghost", prev_len=3, token_ids=[40], masks=[1.0], logprobs=[-0.5])
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize(_receipt([orphan], terminal="orphan"), [orphan])
    assert error.value.code == "missing_parent"

    allcarry = _snapshot("root", token_ids=[10, 11], masks=[0.0, 0.0], logprobs=[0.0, 0.0])
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize(_receipt([allcarry], terminal="root"), [allcarry])
    assert error.value.code == "empty_generation"


def test_out_of_order_mask_is_rejected() -> None:
    shuffled = _snapshot("root", token_ids=[10, 11, 12], masks=[1.0, 0.0, 1.0], logprobs=[-0.1, 0.0, -0.2])
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize(_receipt([shuffled], terminal="root"), [shuffled])
    assert error.value.code == "invalid_mask_order"


def _chained_pair() -> tuple[StagedCallBaseSnapshot, StagedCallBaseSnapshot]:
    root = _snapshot(
        "root",
        token_ids=[10, 11, 12],
        masks=[0.0, 0.0, 1.0],
        logprobs=[0.0, 0.0, -0.1],
    )
    child = _snapshot(
        "child",
        parent_call_id="root",
        prev_len=3,
        token_ids=[20, 21],
        masks=[0.0, 1.0],
        logprobs=[0.0, -0.2],
        parent=root,
    )
    assert root.chain_hash == compute_chain_hash(None, [10, 11, 12])
    assert child.chain_hash == compute_chain_hash(root.chain_hash, [20, 21])
    assert child.cumulative_hash == hash_token_ids([10, 11, 12, 20, 21])
    return root, child


def test_chained_receipt_verifies_and_linearizes() -> None:
    root, child = _chained_pair()
    row = verify_and_linearize(_receipt([root, child], terminal="child"), [root, child])
    assert row.token_ids == [10, 11, 12, 20, 21]


def test_broken_chain_link_is_rejected() -> None:
    root, child = _chained_pair()
    # A child whose declared chain hash does not extend the actual root delta.
    wrong_chain = compute_chain_hash(compute_chain_hash(None, [99]), [20, 21])
    bad_child = _snapshot(
        "child",
        parent_call_id="root",
        prev_len=3,
        token_ids=[20, 21],
        masks=[0.0, 1.0],
        logprobs=[0.0, -0.2],
        chain_hash=wrong_chain,
        cumulative_hash=child.cumulative_hash,
    )
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize(_receipt([root, bad_child], terminal="child"), [root, bad_child])
    assert error.value.code == "chain_hash_mismatch"


def test_terminal_cumulative_hash_mismatch_is_rejected() -> None:
    root, child = _chained_pair()
    bad_child = _snapshot(
        "child",
        parent_call_id="root",
        prev_len=3,
        token_ids=[20, 21],
        masks=[0.0, 1.0],
        logprobs=[0.0, -0.2],
        chain_hash=child.chain_hash,
        cumulative_hash=hash_token_ids([10, 11, 12, 20, 99]),
    )
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize(_receipt([root, bad_child], terminal="child"), [root, bad_child])
    assert error.value.code == "cumulative_hash_mismatch"


def test_chain_digests_are_required_on_staged_rows() -> None:
    root = _snapshot("root", token_ids=[10, 11], masks=[0.0, 1.0], logprobs=[0.0, -0.1])
    for field in ("chain_hash", "cumulative_hash"):
        with pytest.raises(ValueError):
            StagedCallBaseSnapshot(**{**root.model_dump(), field: None})
        with pytest.raises(ValueError):
            CallRecord(**{**_manifest_row(root).model_dump(), field: None})


def test_base_snapshot_validates_strictly_without_extras_bytes() -> None:
    snapshot = _snapshot("root", token_ids=[10, 11], masks=[0.0, 1.0], logprobs=[0.0, -0.1])
    assert snapshot.staging_key == "rollout-1/root"
    # The base model forbids extras entirely; payload bytes never reach it.
    with pytest.raises(ValueError):
        StagedCallBaseSnapshot(**{**snapshot.model_dump(), "extras": {"routed_experts": []}})
    # A tampered token column fails digest recomputation at construction.
    with pytest.raises(ValueError):
        StagedCallBaseSnapshot(**{**snapshot.model_dump(), "token_ids_delta": [99, 11]})
