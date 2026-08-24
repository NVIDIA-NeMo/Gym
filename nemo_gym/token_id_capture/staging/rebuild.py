# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify sealed receipts and rebuild only their declared terminal ancestry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from nemo_gym.token_id_capture.staging.digest import (
    EXTRAS_DIGEST_VERSION,
    STAGING_DIGEST_VERSION,
    STAGING_SCHEMA_VERSION,
    compute_chain_hash,
    compute_extras_digest,
    compute_staging_digest,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    RolloutReceipt,
    StagedCallSnapshot,
)
from nemo_gym.token_id_capture.staging.routes import (
    MISSING_ROUTE_SENTINEL,
    RoutedExpertsFragment,
    decode_routed_experts,
)


class ReceiptVerificationError(ValueError):
    """A receipt or staged snapshot failed a custody invariant."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


class RebuildError(ReceiptVerificationError):
    """A verified manifest cannot form the declared terminal ancestry."""


@dataclass(frozen=True)
class WeightVersionSpan:
    """Policy version covering one call's newly contributed token span."""

    model_call_id: str
    start: int
    end: int
    weight_version: int


@dataclass(frozen=True)
class LinearizedRow:
    """One verified terminal chain ready for framework publication."""

    rollout_id: str
    token_ids: list[int]
    token_mask: list[float]
    logprobs: list[float]
    model_call_ids: list[str]
    prompt_len: int
    weight_versions: list[int]
    weight_version_spans: list[WeightVersionSpan]
    routed_experts: list[list[list[int]]] | None = None
    routed_experts_dtype: str | None = None
    link_spans: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def call_ids(self) -> list[str]:
        """Compatibility spelling for existing framework consumers."""
        return self.model_call_ids


def _fail(code: str, detail: str) -> ReceiptVerificationError:
    return ReceiptVerificationError(code, detail)


def _verify_versions(receipt: RolloutReceipt) -> None:
    if receipt.schema_version != STAGING_SCHEMA_VERSION:
        raise _fail("unsupported_schema", f"receipt schema {receipt.schema_version}")
    if receipt.digest_version != STAGING_DIGEST_VERSION:
        raise _fail("unsupported_digest", f"receipt digest {receipt.digest_version}")
    if receipt.extras_digest_version != EXTRAS_DIGEST_VERSION:
        raise _fail(
            "unsupported_extras_digest",
            f"receipt extras digest {receipt.extras_digest_version}",
        )


def _compare_manifest_fields(
    receipt: RolloutReceipt,
    record: CallRecord,
    snapshot: StagedCallSnapshot,
) -> None:
    call_id = record.model_call_id
    if snapshot.rollout_id != receipt.rollout_id:
        raise _fail("wrong_rollout", f"snapshot {call_id} belongs to {snapshot.rollout_id}")
    comparisons = {
        "model_call_id": snapshot.model_call_id,
        "parent_call_id": snapshot.parent_call_id,
        "mode": snapshot.mode,
        "prev_len": snapshot.prev_len,
        "delta_len": snapshot.delta_len,
        "cum_len": snapshot.cum_len,
        "weight_version": snapshot.weight_version,
        "digest": snapshot.digest,
        "extras_digest": snapshot.extras_digest,
        "chain_hash": snapshot.chain_hash,
        "cumulative_hash": snapshot.cumulative_hash,
    }
    for field_name, actual in comparisons.items():
        expected = getattr(record, field_name)
        if actual != expected:
            raise _fail(
                f"wrong_{field_name}",
                f"call {call_id}: snapshot {actual!r}, manifest {expected!r}",
            )
    for field_name in ("schema_version", "digest_version", "extras_digest_version"):
        actual = getattr(snapshot, field_name)
        expected = getattr(record, field_name)
        if actual != expected or actual != getattr(receipt, field_name):
            raise _fail(
                f"wrong_{field_name}",
                f"call {call_id}: snapshot={actual}, manifest={expected}, receipt={getattr(receipt, field_name)}",
            )


def _recompute_integrity(snapshot: StagedCallSnapshot) -> None:
    call_id = snapshot.model_call_id
    try:
        extras_digest = compute_extras_digest(snapshot.extras)
        digest = compute_staging_digest(
            schema_version=snapshot.schema_version,
            digest_version=snapshot.digest_version,
            extras_digest_version=snapshot.extras_digest_version,
            rollout_id=snapshot.rollout_id,
            model_call_id=call_id,
            parent_call_id=snapshot.parent_call_id,
            mode=snapshot.mode,
            prev_len=snapshot.prev_len,
            delta_len=snapshot.delta_len,
            cum_len=snapshot.cum_len,
            weight_version=snapshot.weight_version,
            token_ids_delta=snapshot.token_ids_delta,
            token_mask_delta=snapshot.token_mask_delta,
            generation_log_probs_delta=snapshot.generation_log_probs_delta,
            extras_digest=extras_digest,
            chain_hash=snapshot.chain_hash,
            cumulative_hash=snapshot.cumulative_hash,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise _fail("invalid_snapshot", f"call {call_id}: {error}") from error
    if extras_digest != snapshot.extras_digest:
        raise _fail("corrupt_extras", f"call {call_id}: extras digest mismatch")
    if digest != snapshot.digest:
        raise _fail("corrupt_digest", f"call {call_id}: staged digest mismatch")


def _validate_manifest_graph(records: dict[str, CallRecord]) -> None:
    for call_id, record in records.items():
        if record.parent_call_id is not None:
            parent = records.get(record.parent_call_id)
            if parent is None:
                raise RebuildError(
                    "missing_parent",
                    f"call {call_id} names absent parent {record.parent_call_id}",
                )
            if parent.cum_len != record.prev_len:
                raise RebuildError(
                    "parent_length_mismatch",
                    f"call {call_id} starts at {record.prev_len}, parent ends at {parent.cum_len}",
                )
        visited: set[str] = set()
        cursor: CallRecord | None = record
        while cursor is not None:
            if cursor.model_call_id in visited:
                raise RebuildError("lineage_cycle", f"cycle reaches call {cursor.model_call_id}")
            visited.add(cursor.model_call_id)
            cursor = records.get(cursor.parent_call_id) if cursor.parent_call_id is not None else None


def _terminal_chain(
    receipt: RolloutReceipt,
    records: dict[str, CallRecord],
) -> list[CallRecord]:
    terminal = receipt.terminal_model_call_id
    if terminal is None:
        raise RebuildError("missing_terminal", "successful receipt has no terminal call")
    chain: list[CallRecord] = []
    cursor = records.get(terminal)
    if cursor is None:
        raise RebuildError("missing_terminal", f"terminal call {terminal} is absent")
    while cursor is not None:
        chain.append(cursor)
        cursor = records.get(cursor.parent_call_id) if cursor.parent_call_id is not None else None
    chain.reverse()
    return chain


def _carry_boundary(snapshot: StagedCallSnapshot) -> int:
    boundary = 0
    for mask in snapshot.token_mask_delta:
        if mask != 0.0:
            break
        boundary += 1
    if any(mask != 1.0 for mask in snapshot.token_mask_delta[boundary:]):
        raise RebuildError(
            "invalid_mask_order",
            f"call {snapshot.model_call_id} mask is not carry-then-generation",
        )
    if boundary == len(snapshot.token_mask_delta):
        raise RebuildError(
            "empty_generation",
            f"call {snapshot.model_call_id} contains no policy-generated token",
        )
    return boundary


def _decode_selected_routes(
    chain: Sequence[CallRecord],
    snapshots: dict[str, StagedCallSnapshot],
) -> tuple[list[list[list[int]]] | None, str | None]:
    decoded: dict[str, RoutedExpertsFragment] = {}
    for record in chain:
        payload = (snapshots[record.model_call_id].extras or {}).get("routed_experts")
        if payload is None:
            continue
        try:
            fragment = decode_routed_experts(payload)
        except ValueError as error:
            raise RebuildError(
                "invalid_routes",
                f"call {record.model_call_id}: {error}",
            ) from error
        if len(fragment.values) != record.delta_len:
            raise RebuildError(
                "route_length_mismatch",
                f"call {record.model_call_id}: {len(fragment.values)} routes for {record.delta_len} tokens",
            )
        decoded[record.model_call_id] = fragment
    if not decoded:
        return None, None
    template = next(iter(decoded.values()))
    for call_id, fragment in decoded.items():
        if (fragment.num_layers, fragment.topk) != (template.num_layers, template.topk):
            raise RebuildError("route_shape_mismatch", f"call {call_id} route shape changed")
        if fragment.dtype != template.dtype:
            raise RebuildError("route_dtype_mismatch", f"call {call_id} route dtype changed")
    sentinel = [[MISSING_ROUTE_SENTINEL] * template.topk for _ in range(template.num_layers)]
    routes: list[list[list[int]]] = []
    for record in chain:
        fragment = decoded.get(record.model_call_id)
        if fragment is None:
            routes.extend([[list(layer) for layer in sentinel] for _ in range(record.delta_len)])
        else:
            routes.extend(fragment.values)
    return routes, template.dtype


def verify_and_linearize(
    receipt: RolloutReceipt,
    snapshots: Sequence[StagedCallSnapshot],
) -> LinearizedRow:
    """Verify an untrusted staged set and linearize the declared terminal chain."""
    if not isinstance(receipt, RolloutReceipt):
        raise TypeError("receipt must be a RolloutReceipt")
    _verify_versions(receipt)
    if receipt.failure_reason is not None:
        raise _fail("rollout_failed", receipt.failure_reason)
    if receipt.capture_poisoned:
        raise _fail("capture_poisoned", "receipt marks token capture as poisoned")
    if not receipt.manifest:
        raise _fail("empty_manifest", "successful receipt has no committed calls")

    manifest_ids = [record.model_call_id for record in receipt.manifest]
    staging_keys = [record.staging_key for record in receipt.manifest]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise _fail("duplicate_manifest_row", "model_call_id values are not unique")
    if len(staging_keys) != len(set(staging_keys)):
        raise _fail("duplicate_staging_key", "staging keys are not unique")
    if len(snapshots) != len(receipt.manifest):
        raise _fail(
            "row_count_mismatch",
            f"{len(snapshots)} snapshots for {len(receipt.manifest)} manifest rows",
        )

    for snapshot in snapshots:
        if not isinstance(snapshot, StagedCallSnapshot):
            raise TypeError("snapshots must contain StagedCallSnapshot values")

    snapshot_ids = [snapshot.model_call_id for snapshot in snapshots]
    if len(snapshot_ids) != len(set(snapshot_ids)):
        raise _fail("duplicate_snapshot", "model_call_id values are not unique")
    if set(snapshot_ids) != set(manifest_ids):
        missing = sorted(set(manifest_ids) - set(snapshot_ids))
        extra = sorted(set(snapshot_ids) - set(manifest_ids))
        raise _fail("snapshot_identity_mismatch", f"missing={missing}, extra={extra}")
    for record, snapshot in zip(receipt.manifest, snapshots):
        if snapshot.model_call_id != record.model_call_id:
            raise _fail(
                "snapshot_order_mismatch",
                f"key {record.staging_key} expected {record.model_call_id}, received {snapshot.model_call_id}",
            )

    snapshots_by_id = {snapshot.model_call_id: snapshot for snapshot in snapshots}
    records_by_id = {record.model_call_id: record for record in receipt.manifest}
    for record in receipt.manifest:
        snapshot = snapshots_by_id[record.model_call_id]
        _compare_manifest_fields(receipt, record, snapshot)
        _recompute_integrity(snapshot)
    _validate_manifest_graph(records_by_id)
    chain = _terminal_chain(receipt, records_by_id)

    token_ids: list[int] = []
    token_mask: list[float] = []
    logprobs: list[float] = []
    model_call_ids: list[str] = []
    weight_versions: list[int] = []
    weight_version_spans: list[WeightVersionSpan] = []
    link_spans: list[tuple[str, int, int]] = []
    prompt_len = 0
    running_chain_hash: str | None = None
    for index, record in enumerate(chain):
        snapshot = snapshots_by_id[record.model_call_id]
        # Chained-digest verification: each staged delta must extend its
        # parent's chain hash. Rows staged before the chain columns existed
        # carry ``None`` and skip this check.
        running_chain_hash = compute_chain_hash(running_chain_hash, snapshot.token_ids_delta)
        if record.chain_hash is not None and record.chain_hash != running_chain_hash:
            raise RebuildError(
                "chain_hash_mismatch",
                f"call {record.model_call_id} does not extend its parent's staged chain",
            )
        boundary = _carry_boundary(snapshot)
        start = len(token_ids)
        token_ids.extend(snapshot.token_ids_delta)
        token_mask.extend(snapshot.token_mask_delta)
        logprobs.extend(snapshot.generation_log_probs_delta)
        end = len(token_ids)
        if index == 0:
            prompt_len = boundary
        model_call_ids.append(record.model_call_id)
        weight_versions.append(record.weight_version)
        weight_version_spans.append(
            WeightVersionSpan(
                model_call_id=record.model_call_id,
                start=start,
                end=end,
                weight_version=record.weight_version,
            )
        )
        link_spans.append((record.model_call_id, boundary, record.delta_len - boundary))
    if not any(token_mask):
        raise RebuildError("empty_training_row", "terminal chain has no generated tokens")
    # Terminal-only whole-sequence anchor; per-record cumulative checks would
    # rehash O(n^2) tokens for no additional coverage over the chain hashes.
    terminal_cumulative_hash = chain[-1].cumulative_hash
    if terminal_cumulative_hash is not None and terminal_cumulative_hash != hash_token_ids(token_ids):
        raise RebuildError(
            "cumulative_hash_mismatch",
            f"terminal call {chain[-1].model_call_id} cumulative hash does not cover the linearized tokens",
        )

    routed_experts, routed_experts_dtype = _decode_selected_routes(chain, snapshots_by_id)
    if routed_experts is not None and len(routed_experts) != len(token_ids):
        raise RebuildError(
            "route_length_mismatch",
            f"{len(routed_experts)} route rows for {len(token_ids)} tokens",
        )
    return LinearizedRow(
        rollout_id=receipt.rollout_id,
        token_ids=token_ids,
        token_mask=token_mask,
        logprobs=logprobs,
        model_call_ids=model_call_ids,
        prompt_len=prompt_len,
        weight_versions=weight_versions,
        weight_version_spans=weight_version_spans,
        routed_experts=routed_experts,
        routed_experts_dtype=routed_experts_dtype,
        link_spans=link_spans,
    )


def linearize(
    rollout_id: str,
    snapshots: list[StagedCallSnapshot],
    manifest: list[CallRecord],
    *,
    terminal_hint: str | None = None,
) -> LinearizedRow:
    """Compatibility wrapper that still executes the production verifier."""
    receipt = RolloutReceipt(
        rollout_id=rollout_id,
        terminal_model_call_id=terminal_hint,
        manifest=manifest,
        terminal_selection="declared",
    )
    return verify_and_linearize(receipt, snapshots)
