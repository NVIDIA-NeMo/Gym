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


# ---------------------------------------------------------------------------
# Forest linearization: one row per chain (the terminal plus subagent sessions).
# Ported from 4662c53 (Gerald Shen); the context-rewrite (compaction) cases are
# omitted with the boundary support they exercise.
# ---------------------------------------------------------------------------

from dataclasses import replace  # noqa: E402

from nemo_gym.token_id_capture.staging import rebuild as rebuild_module  # noqa: E402
from nemo_gym.token_id_capture.staging.rebuild import (  # noqa: E402
    CHAIN_KIND_SUBAGENT,
    CHAIN_KIND_TERMINAL,
    CHAIN_KINDS,
    SKIP_ABANDONED_ROOT,
    SKIP_AMBIGUOUS_LEAF,
    LinearizedRollout,
    SkippedChain,
    verify_and_linearize_all,
)
from nemo_gym.token_id_capture.staging.terminal import TerminalSelection, select_terminal_call  # noqa: E402


SUBAGENT = CHAIN_KIND_SUBAGENT
TERMINAL = CHAIN_KIND_TERMINAL


def _forest_receipt(
    snapshots: list[StagedCallBaseSnapshot],
    *,
    terminal: str,
    admitted_at: dict[str, float] | None = None,
) -> RolloutReceipt:
    """Receipt over a multi-root manifest with optional admission stamps."""
    admitted_at = admitted_at or {}
    manifest = [
        CallRecord(**{**_manifest_row(snapshot).model_dump(), "admitted_at": admitted_at.get(snapshot.model_call_id)})
        for snapshot in snapshots
    ]
    return RolloutReceipt(
        rollout_id="rollout-1",
        terminal_model_call_id=terminal,
        manifest=manifest,
        terminal_selection="heuristic",
    )


def _chain_snapshots(prefix: str, deltas: list[tuple[list[int], list[float]]]) -> list[StagedCallBaseSnapshot]:
    """Build a linear chain ``<prefix>0 -> <prefix>1 -> ...`` with correct chain/cumulative digests."""
    rows: list[StagedCallBaseSnapshot] = []
    seen: list[int] = []
    parent: StagedCallBaseSnapshot | None = None
    for index, (token_ids, masks) in enumerate(deltas):
        rows.append(
            _snapshot(
                f"{prefix}{index}",
                parent_call_id=parent.model_call_id if parent is not None else None,
                prev_len=len(seen),
                token_ids=token_ids,
                masks=masks,
                logprobs=[-0.01 * (i + 1) if m else 0.0 for i, m in enumerate(masks)],
                weight_version=10 + index,
                parent=parent,
                prefix_token_ids=list(seen),
            )
        )
        seen.extend(token_ids)
        parent = rows[-1]
    return rows


def _placement_free(row) -> Any:
    """Drop the forest-placement fields so a chain row compares to its single-chain rebuild."""
    return replace(row, chain_index=0, chain_kind="terminal", segment_index=0)


def _single_chain(snapshots: list[StagedCallBaseSnapshot], receipt: RolloutReceipt):
    """``verify_and_linearize`` over just this chain's manifest rows."""
    ids = {snapshot.model_call_id for snapshot in snapshots}
    manifest = [record for record in receipt.manifest if record.model_call_id in ids]
    sub = RolloutReceipt(
        rollout_id="rollout-1",
        terminal_model_call_id=snapshots[-1].model_call_id,
        manifest=manifest,
        terminal_selection="declared",
    )
    return verify_and_linearize(sub, snapshots)


def _rows(result: LinearizedRollout) -> list[tuple]:
    """``(leaf call, kind, segment_index, boundary parent, chain_index)`` per row."""
    return [
        (row.terminal_model_call_id, row.chain_kind, row.segment_index, row.boundary_parent_call_id, row.chain_index)
        for row in result.rows
    ]


def _assert_rows_equal_single(result: LinearizedRollout, chains: list[list[StagedCallBaseSnapshot]], receipt) -> None:
    """Every secondary row equals the single-chain rebuild of its (root-to-leaf ordered) chain."""
    by_leaf = {chain[-1].model_call_id: chain for chain in chains}
    for row in result.rows[1:]:
        chain = by_leaf[row.terminal_model_call_id]
        assert _placement_free(row) == _single_chain(chain, receipt), row.terminal_model_call_id
        assert row.chain_kind in CHAIN_KINDS


def _main_and_subagent() -> tuple[RolloutReceipt, list[StagedCallBaseSnapshot], list, list]:
    main = _chain_snapshots("m", [([10, 11], [0.0, 1.0]), ([20, 21], [0.0, 1.0])])
    sub = _chain_snapshots("u", [([70, 71, 72], [0.0, 0.0, 1.0]), ([80], [1.0])])
    snapshots = main + sub
    receipt = _forest_receipt(
        snapshots, terminal="m1", admitted_at={"m0": 100.0, "u0": 100.5, "u1": 100.7, "m1": 101.0}
    )
    return receipt, snapshots, main, sub


def test_all_main_session_with_two_subagents_yields_three_rows_terminal_first() -> None:
    main = _chain_snapshots("m", [([10, 11], [0.0, 1.0]), ([20, 21], [0.0, 1.0]), ([30], [1.0])])
    explore = _chain_snapshots("u", [([70, 71, 72], [0.0, 0.0, 1.0]), ([80], [1.0])])
    review = _chain_snapshots("v", [([90, 91], [0.0, 1.0])])
    snapshots = main + explore + review
    receipt = _forest_receipt(
        snapshots,
        terminal="m2",
        admitted_at={"m0": 100.0, "m1": 101.0, "u0": 101.5, "u1": 101.7, "v0": 102.5, "m2": 103.0},
    )
    result = verify_and_linearize_all(receipt, snapshots)
    assert result.skipped == []
    assert result.num_roots == 3
    assert result.num_boundary_roots == 0
    assert _rows(result) == [
        ("m2", TERMINAL, 0, None, 0),
        ("u1", SUBAGENT, 0, None, 1),
        ("v0", SUBAGENT, 0, None, 2),
    ]
    # The terminal row is exactly what verify_and_linearize returns for the receipt.
    assert result.rows[0] == verify_and_linearize(receipt, snapshots)
    assert result.terminal is result.rows[0]
    _assert_rows_equal_single(result, [explore, review], receipt)


def test_all_subagent_root_is_a_subagent_row() -> None:
    receipt, snapshots, _, sub = _main_and_subagent()
    result = verify_and_linearize_all(receipt, snapshots)
    assert result.skipped == []
    assert result.num_roots == 2
    assert result.num_boundary_roots == 0
    assert [(row.chain_kind, row.segment_index, row.boundary_parent_call_id) for row in result.rows] == [
        ("terminal", 0, None),
        ("subagent", 0, None),
    ]
    assert result.rows[1].model_call_ids == ["u0", "u1"]
    assert result.rows[1].token_ids == [70, 71, 72, 80]
    assert result.rows[1].token_mask == [0.0, 0.0, 1.0, 1.0]
    assert result.rows[1].prompt_len == 2
    assert _placement_free(result.rows[1]) == _single_chain(sub, receipt)


def test_all_is_deterministic_and_verifies_every_chain() -> None:
    receipt, snapshots, main, sub = _main_and_subagent()
    assert verify_and_linearize_all(receipt, snapshots) == verify_and_linearize_all(receipt, snapshots)
    # A broken link on the *subagent* chain skips that chain with its code; the canonical row survives.
    bad_u1 = _snapshot(
        "u1",
        parent_call_id="u0",
        prev_len=3,
        token_ids=[80],
        masks=[1.0],
        logprobs=[-0.02],
        weight_version=11,
        chain_hash=compute_chain_hash(compute_chain_hash(None, [99]), [80]),
        cumulative_hash=sub[1].cumulative_hash,
    )
    corrupted = [*main, sub[0], bad_u1]
    bad_receipt = _forest_receipt(corrupted, terminal="m1", admitted_at={"m0": 100.0, "u0": 100.5})
    # The terminal chain alone still verifies ...
    assert verify_and_linearize(bad_receipt, corrupted).model_call_ids == ["m0", "m1"]
    # ... and the forest walk keeps it while reporting the inconsistent subagent chain.
    result = verify_and_linearize_all(bad_receipt, corrupted)
    assert [row.terminal_model_call_id for row in result.rows] == ["m1"]
    assert result.skipped == [SkippedChain(root_call_id="u0", reason="chain_hash_mismatch")]
    # The same break on the *terminal* chain stays fatal.
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize_all(bad_receipt.model_copy(update={"terminal_model_call_id": "u1"}), corrupted)
    assert error.value.code == "chain_hash_mismatch"


def test_all_ambiguous_secondary_fork_is_skipped_and_terminal_still_delivered() -> None:
    main = _chain_snapshots("m", [([10, 11], [0.0, 1.0]), ([20, 21], [0.0, 1.0])])
    fork_root = _snapshot("u0", token_ids=[70, 71], masks=[0.0, 1.0], logprobs=[0.0, -0.1])
    fork_a = _snapshot(
        "u1a", parent_call_id="u0", prev_len=2, token_ids=[80], masks=[1.0], logprobs=[-0.2], parent=fork_root
    )
    fork_b = _snapshot(
        "u1b", parent_call_id="u0", prev_len=2, token_ids=[81], masks=[1.0], logprobs=[-0.3], parent=fork_root
    )
    snapshots = main + [fork_root, fork_a, fork_b]
    receipt = _forest_receipt(snapshots, terminal="m1", admitted_at={"m0": 100.0, "u0": 100.5})
    result = verify_and_linearize_all(receipt, snapshots)
    assert result.skipped == [SkippedChain(root_call_id="u0", reason="ambiguous_leaf")]
    assert len(result.rows) == 1
    assert result.rows[0].chain_kind == "terminal"
    assert result.rows[0].model_call_ids == ["m0", "m1"]
    assert result.num_roots == 2


def test_all_secondary_rows_follow_admission_order_with_unstamped_roots_last() -> None:
    main = _chain_snapshots("m", [([10, 11], [0.0, 1.0])])
    late = _chain_snapshots("x", [([20, 21], [0.0, 1.0])])
    early = _chain_snapshots("y", [([30, 31], [0.0, 1.0])])
    unstamped = _chain_snapshots("z", [([40, 41], [0.0, 1.0])])
    snapshots = main + unstamped + late + early
    receipt = _forest_receipt(snapshots, terminal="m0", admitted_at={"m0": 50.0, "x0": 300.0, "y0": 200.0})
    result = verify_and_linearize_all(receipt, snapshots)
    assert [row.terminal_model_call_id for row in result.rows] == ["m0", "y0", "x0", "z0"]
    assert [row.chain_index for row in result.rows] == [0, 1, 2, 3]
    assert {row.chain_kind for row in result.rows[1:]} == {"subagent"}


def test_all_on_a_single_chain_matches_verify_and_linearize_exactly() -> None:
    receipt, snapshots = _branched()  # one root; the dead sibling is a branch, not a root
    result = verify_and_linearize_all(receipt, snapshots)
    assert result.skipped == []
    assert result.num_roots == 1
    assert result.num_boundary_roots == 0
    assert len(result.rows) == 1
    assert result.rows[0] == verify_and_linearize(receipt, snapshots)
    single = result.rows[0]
    assert (single.terminal_model_call_id, single.chain_index, single.chain_kind, single.segment_index) == (
        "main",
        0,
        "terminal",
        0,
    )
    assert single.boundary_parent_call_id is None


def test_all_reports_receipt_level_failures_like_verify_and_linearize() -> None:
    receipt, snapshots = _branched()
    for candidate, code in (
        (receipt.model_copy(update={"capture_poisoned": True}), "capture_poisoned"),
        (receipt.model_copy(update={"terminal_model_call_id": None}), "missing_terminal"),
    ):
        with pytest.raises(ReceiptVerificationError) as error:
            verify_and_linearize_all(candidate, snapshots)
        assert error.value.code == code
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize_all(receipt, snapshots[:-1])
    assert error.value.code == "row_count_mismatch"


def test_all_skips_a_secondary_chain_without_generated_tokens_but_fails_the_terminal(monkeypatch) -> None:
    """``empty_training_row`` is only reachable past the per-call carry check; stub that check."""
    original = rebuild_module._carry_boundary

    def lenient(snapshot):
        if not any(snapshot.token_mask_delta):
            return len(snapshot.token_mask_delta)
        return original(snapshot)

    monkeypatch.setattr(rebuild_module, "_carry_boundary", lenient)
    main = _chain_snapshots("m", [([10, 11], [0.0, 1.0])])
    silent = _chain_snapshots("u", [([20, 21], [0.0, 0.0])])
    snapshots = main + silent
    receipt = _forest_receipt(snapshots, terminal="m0", admitted_at={"m0": 1.0, "u0": 2.0})
    result = verify_and_linearize_all(receipt, snapshots)
    assert result.skipped == [SkippedChain(root_call_id="u0", reason="empty_training_row")]
    assert [row.terminal_model_call_id for row in result.rows] == ["m0"]

    # The terminal chain keeps today's hard error.
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize_all(_forest_receipt(snapshots, terminal="u0"), snapshots)
    assert error.value.code == "empty_training_row"


def test_verify_and_linearize_row_carries_default_placement_fields() -> None:
    receipt, snapshots, _, _ = _main_and_subagent()
    row = verify_and_linearize(receipt, snapshots)
    assert row.terminal_model_call_id == "m1"
    assert row.chain_index == 0
    assert row.chain_kind == "terminal"
    assert row.segment_index == 0
    assert row.boundary_parent_call_id is None
    assert row.call_ids == row.model_call_ids


def test_abandoned_first_call_retry_is_not_published() -> None:
    dead = _chain_snapshots("d", [([10, 11], [0.0, 1.0])])
    live = _chain_snapshots("r", [([10, 11], [0.0, 1.0]), ([12], [1.0])])
    snapshots = dead + live
    receipt = _forest_receipt(snapshots, terminal="r1", admitted_at={"d0": 50.0, "r0": 100.0})
    assert select_terminal_call(receipt.manifest) == TerminalSelection("r1", "selected")
    result = verify_and_linearize_all(receipt, snapshots)
    assert _rows(result) == [("r1", TERMINAL, 0, None, 0)]
    assert result.skipped == [SkippedChain("d0", SKIP_ABANDONED_ROOT)]


def test_retry_fork_at_the_terminal_versus_on_a_subagent_chain() -> None:
    main = _chain_snapshots("r", [([10, 11], [0.0, 1.0]), ([12], [1.0])])
    fork = _snapshot(
        "r1x", parent_call_id="r0", prev_len=2, token_ids=[13], masks=[1.0], logprobs=[-0.9], parent=main[0]
    )
    snapshots = main + [fork]
    receipt = _forest_receipt(snapshots, terminal="r1", admitted_at={"r0": 100.0})
    # A declared terminal makes the fork irrelevant to the terminal chain ...
    assert _rows(verify_and_linearize_all(receipt, snapshots)) == [("r1", TERMINAL, 0, None, 0)]
    # ... while the token-free heuristic stays ambiguous.
    assert select_terminal_call(receipt.manifest) == TerminalSelection(None, "ambiguous_terminal")
    # The same fork on a secondary (subagent) chain skips that chain only.
    sub = _chain_snapshots("u", [([30, 31], [0.0, 1.0]), ([32], [1.0])])
    sub_fork = _snapshot(
        "u1x", parent_call_id="u0", prev_len=2, token_ids=[33], masks=[1.0], logprobs=[-0.9], parent=sub[0]
    )
    snapshots2 = main + sub + [sub_fork]
    receipt2 = _forest_receipt(snapshots2, terminal="r1", admitted_at={"r0": 100.0, "u0": 105.0})
    result = verify_and_linearize_all(receipt2, snapshots2)
    assert _rows(result) == [("r1", TERMINAL, 0, None, 0)]
    assert result.skipped == [SkippedChain("u0", SKIP_AMBIGUOUS_LEAF)]


def test_corrupt_secondary_chains_are_skipped_with_their_code_and_the_rollout_survives() -> None:
    main = _chain_snapshots("m", [([10, 11], [0.0, 1.0]), ([12], [1.0])])
    # Subagent leaf with generation-then-carry mask order (digest valid, custody check fails).
    u0 = _snapshot("u0", token_ids=[30, 31], masks=[0.0, 1.0], logprobs=[0.0, -0.1])
    u1 = _snapshot(
        "u1", parent_call_id="u0", prev_len=2, token_ids=[32, 33], masks=[1.0, 0.0], logprobs=[-0.1, 0.0], parent=u0
    )
    snapshots = main + [u0, u1]
    receipt = _forest_receipt(snapshots, terminal="m1", admitted_at={"m0": 1.0, "u0": 2.0})
    assert verify_and_linearize(receipt, snapshots).model_call_ids == ["m0", "m1"]
    result = verify_and_linearize_all(receipt, snapshots)
    assert _rows(result) == [("m1", TERMINAL, 0, None, 0)]
    assert result.skipped == [SkippedChain("u0", "invalid_mask_order")]
    # A secondary chain whose root has no generated token (extended, so it is not an abandoned root).
    silent = _chain_snapshots("w", [([20, 21], [0.0, 0.0]), ([22], [1.0])])
    receipt = _forest_receipt(main + silent, terminal="m1", admitted_at={"m0": 1.0, "w0": 2.0})
    result = verify_and_linearize_all(receipt, main + silent)
    assert _rows(result) == [("m1", TERMINAL, 0, None, 0)]
    assert result.skipped == [SkippedChain("w0", "empty_generation")]
    # The same defects on the terminal chain stay fatal.
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize_all(
            _forest_receipt(snapshots, terminal="u1", admitted_at={"m0": 1.0, "u0": 2.0}), snapshots
        )
    assert error.value.code == "invalid_mask_order"
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize_all(_forest_receipt(main + silent, terminal="w0"), main + silent)
    assert error.value.code == "empty_generation"


def test_weight_version_spans_and_chain_hashes_restart_per_chain() -> None:
    main = _chain_snapshots("r", [([10, 11], [0.0, 1.0]), ([12], [1.0])])
    sub = _chain_snapshots("s", [([20, 21], [0.0, 1.0]), ([22], [1.0])])
    snapshots = main + sub
    receipt = _forest_receipt(snapshots, terminal="r1", admitted_at={"r0": 100.0, "s0": 110.0})
    result = verify_and_linearize_all(receipt, snapshots)
    assert _rows(result) == [("r1", TERMINAL, 0, None, 0), ("s1", SUBAGENT, 0, None, 1)]
    for row in result.rows:
        assert row.weight_version_spans[0].start == 0
        assert row.weight_version_spans[-1].end == len(row.token_ids)
        assert row.prompt_len == 1
    # A root whose chain hash chains onto another session instead of restarting fails as the terminal chain ...
    bad_root = _snapshot(
        "s0", token_ids=[20, 21], masks=[0.0, 1.0], logprobs=[0.0, -0.01], chain_hash=main[-1].chain_hash
    )
    snapshots2 = main + [bad_root]
    receipt2 = _forest_receipt(snapshots2, terminal="s0", admitted_at={"r0": 100.0, "s0": 110.0})
    with pytest.raises(ReceiptVerificationError) as error:
        verify_and_linearize_all(receipt2, snapshots2)
    assert error.value.code == "chain_hash_mismatch"
    # ... and is skipped with its code when the chain is a secondary (extended) root.
    bad_child = _snapshot(
        "s1", parent_call_id="s0", prev_len=2, token_ids=[22], masks=[1.0], logprobs=[-0.1], parent=bad_root
    )
    snapshots3 = main + [bad_root, bad_child]
    receipt3 = _forest_receipt(snapshots3, terminal="r1", admitted_at={"r0": 100.0, "s0": 110.0})
    result = verify_and_linearize_all(receipt3, snapshots3)
    assert _rows(result) == [("r1", TERMINAL, 0, None, 0)]
    assert result.skipped == [SkippedChain("s0", "chain_hash_mismatch")]


def test_single_call_subagent_beside_an_extended_main_chain_is_published() -> None:
    main = _chain_snapshots("m", [([10, 11], [0.0, 1.0]), ([12], [1.0])])
    sub = _chain_snapshots("u", [([70, 71, 72], [0.0, 0.0, 1.0])])  # a `task` subagent that answered in one call
    snapshots = main + sub
    receipt = _forest_receipt(snapshots, terminal="m1", admitted_at={"m0": 100.0, "u0": 100.5})
    # The heuristic still prefers the extended root ...
    assert select_terminal_call(receipt.manifest) == TerminalSelection("m1", "selected")
    result = verify_and_linearize_all(receipt, snapshots)
    # ... and the single-call session is published, not dropped as an abandoned root.
    assert _rows(result) == [("m1", TERMINAL, 0, None, 0), ("u0", SUBAGENT, 0, None, 1)]
    assert result.rows[1].model_call_ids == ["u0"]
    assert result.rows[1].token_mask == [0.0, 0.0, 1.0]
    assert result.skipped == []
    _assert_rows_equal_single(result, [sub], receipt)


def test_dead_first_call_retry_is_identified_by_its_prompt_tokens() -> None:
    live = _chain_snapshots("r", [([10, 11], [0.0, 1.0]), ([12], [1.0])])
    # Same prompt tokens, different generation: a first attempt the client never saw.
    dead = _chain_snapshots("d", [([10, 99], [0.0, 1.0])])
    receipt = _forest_receipt(dead + live, terminal="r1", admitted_at={"d0": 50.0, "r0": 100.0})
    assert select_terminal_call(receipt.manifest) == TerminalSelection("r1", "selected")
    result = verify_and_linearize_all(receipt, dead + live)
    assert _rows(result) == [("r1", TERMINAL, 0, None, 0)]
    assert result.skipped == [SkippedChain("d0", SKIP_ABANDONED_ROOT)]
    # A different prompt is a different session: published although childless and admitted first.
    other = _chain_snapshots("d", [([20, 99], [0.0, 1.0])])
    receipt = _forest_receipt(other + live, terminal="r1", admitted_at={"d0": 50.0, "r0": 100.0})
    assert select_terminal_call(receipt.manifest) == TerminalSelection("r1", "selected")
    result = verify_and_linearize_all(receipt, other + live)
    assert _rows(result) == [("r1", TERMINAL, 0, None, 0), ("d0", SUBAGENT, 0, None, 1)]
    assert result.skipped == []
    # The whole carry span is compared, not just its first token.
    longer_live = _chain_snapshots("r", [([10, 11, 12], [0.0, 0.0, 1.0]), ([13], [1.0])])
    near_miss = _chain_snapshots("d", [([10, 12, 99], [0.0, 0.0, 1.0])])
    receipt = _forest_receipt(near_miss + longer_live, terminal="r1", admitted_at={"d0": 50.0, "r0": 100.0})
    result = verify_and_linearize_all(receipt, near_miss + longer_live)
    assert _rows(result) == [("r1", TERMINAL, 0, None, 0), ("d0", SUBAGENT, 0, None, 1)]
    same_prompt = _chain_snapshots("d", [([10, 11, 99], [0.0, 0.0, 1.0])])
    receipt = _forest_receipt(same_prompt + longer_live, terminal="r1", admitted_at={"d0": 50.0, "r0": 100.0})
    result = verify_and_linearize_all(receipt, same_prompt + longer_live)
    assert result.skipped == [SkippedChain("d0", SKIP_ABANDONED_ROOT)]


def test_dead_retry_of_a_single_call_terminal_is_abandoned() -> None:
    """The declared terminal counts as extended: a same-prompt sibling the witness did not name is dead."""
    main = _chain_snapshots("m", [([10, 11], [0.0, 1.0]), ([12], [1.0])])
    first = _chain_snapshots("d", [([20, 21], [0.0, 1.0])])
    retry = _chain_snapshots("x", [([20, 22], [0.0, 1.0])])  # same prompt, the session ended here
    snapshots = main + first + retry
    receipt = _forest_receipt(snapshots, terminal="x0", admitted_at={"m0": 100.0, "d0": 105.0, "x0": 106.0})
    result = verify_and_linearize_all(receipt, snapshots)
    assert _rows(result) == [("x0", TERMINAL, 0, None, 0), ("m1", SUBAGENT, 0, None, 1)]
    assert result.skipped == [SkippedChain("d0", SKIP_ABANDONED_ROOT)]


def test_two_single_call_sessions_with_the_same_prompt_and_no_continuation_are_both_kept() -> None:
    """Neither was extended and neither is the terminal: nothing marks one as the retry, so the survivors rule
    keeps the whole pool; a byte-identical pair is still collapsed by cumulative hash."""
    main = _chain_snapshots("m", [([10, 11], [0.0, 1.0]), ([12], [1.0])])
    a = _chain_snapshots("a", [([70, 71], [0.0, 1.0])])
    b = _chain_snapshots("b", [([70, 72], [0.0, 1.0])])  # same prompt, different answer
    receipt = _forest_receipt(main + a + b, terminal="m1", admitted_at={"m0": 100.0, "a0": 101.0, "b0": 102.0})
    result = verify_and_linearize_all(receipt, main + a + b)
    assert _rows(result) == [
        ("m1", TERMINAL, 0, None, 0),
        ("a0", SUBAGENT, 0, None, 1),
        ("b0", SUBAGENT, 0, None, 2),
    ]
    assert result.skipped == []
    identical = _chain_snapshots("b", [([70, 71], [0.0, 1.0])])
    receipt = _forest_receipt(main + a + identical, terminal="m1", admitted_at={"m0": 100.0, "a0": 101.0, "b0": 102.0})
    result = verify_and_linearize_all(receipt, main + a + identical)
    assert _rows(result) == [("m1", TERMINAL, 0, None, 0), ("a0", SUBAGENT, 0, None, 1)]
    assert result.skipped == [SkippedChain("b0", SKIP_ABANDONED_ROOT)]


def test_childless_root_with_an_unreadable_prompt_is_skipped_with_its_code() -> None:
    """A root whose delta is not carry-then-generation joins no retry pool and is skipped when linearized."""
    main = _chain_snapshots("m", [([10, 11], [0.0, 1.0]), ([12], [1.0])])
    bad = _snapshot("u0", token_ids=[10, 11], masks=[1.0, 0.0], logprobs=[-0.1, 0.0])  # same tokens as m0
    snapshots = main + [bad]
    receipt = _forest_receipt(snapshots, terminal="m1", admitted_at={"m0": 1.0, "u0": 2.0})
    result = verify_and_linearize_all(receipt, snapshots)
    assert _rows(result) == [("m1", TERMINAL, 0, None, 0)]
    assert result.skipped == [SkippedChain("u0", "invalid_mask_order")]
    # A root with no prompt token at all is never grouped as a retry of anything.
    bare_a = _chain_snapshots("a", [([70], [1.0])])
    bare_b = _chain_snapshots("b", [([71], [1.0])])
    snapshots = main + bare_a + bare_b
    receipt = _forest_receipt(snapshots, terminal="m1", admitted_at={"m0": 1.0, "a0": 2.0, "b0": 3.0})
    result = verify_and_linearize_all(receipt, snapshots)
    assert _rows(result) == [
        ("m1", TERMINAL, 0, None, 0),
        ("a0", SUBAGENT, 0, None, 1),
        ("b0", SUBAGENT, 0, None, 2),
    ]
    assert result.skipped == []
