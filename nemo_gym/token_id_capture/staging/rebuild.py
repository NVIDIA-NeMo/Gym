# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify a rollout receipt and rebuild its selected token sequence(s).

The verifier reads token columns from :class:`StagedCallBaseSnapshot`.
It does not fetch or decode optional extras.
For each selected call, it returns an :class:`ExtrasCommitment`.
Consumers fetch the extras and compare their digest before use.

Two entry points share one verification pass:

* :func:`verify_and_linearize` returns the receipt's declared terminal chain
  as a single :class:`LinearizedRow` (the historical contract).
* :func:`verify_and_linearize_all` additionally linearizes every other
  surviving root of the manifest forest (subagent sessions), one row per
  chain, terminal chain first, then the other roots in admission order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from nemo_gym.token_id_capture.staging.digest import (
    EXTRAS_DIGEST_VERSION,
    STAGING_DIGEST_VERSION,
    STAGING_SCHEMA_VERSION,
    compute_chain_hash,
    compute_staging_digest,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    RolloutReceipt,
    StagedCallBaseSnapshot,
)
from nemo_gym.token_id_capture.staging.terminal import (
    children_by_parent,
    descend_to_leaf,
    root_order_key,
)


class ReceiptVerificationError(ValueError):
    """A receipt or staged snapshot failed a custody invariant."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(f"{code}: {detail}")


class RebuildError(ReceiptVerificationError):
    """A verified manifest cannot form the declared terminal ancestry."""


# Chain kinds carried on ``LinearizedRow.chain_kind``.
CHAIN_KIND_TERMINAL = "terminal"
CHAIN_KIND_SUBAGENT = "subagent"
CHAIN_KINDS = (CHAIN_KIND_TERMINAL, CHAIN_KIND_SUBAGENT)

# Reasons carried on ``SkippedChain.reason`` for structural skips. A
# non-terminal chain that fails a per-chain custody check is skipped with the
# :class:`RebuildError` code instead (``chain_hash_mismatch``,
# ``invalid_mask_order``, ``empty_generation``, ``cumulative_hash_mismatch``,
# ``empty_training_row``).
SKIP_AMBIGUOUS_LEAF = "ambiguous_leaf"
SKIP_ABANDONED_ROOT = "abandoned_root"
SKIP_EMPTY_TRAINING_ROW = "empty_training_row"
SKIP_LEAF_ON_TERMINAL_CHAIN = "leaf_on_terminal_chain"


@dataclass(frozen=True)
class WeightVersionSpan:
    """Policy version covering one call's newly contributed token span."""

    model_call_id: str
    start: int
    end: int
    weight_version: int


@dataclass(frozen=True)
class ExtrasCommitment:
    """The receipt-bound extras digest for one selected call.

    A commitment proves what the extras payload must hash to — not that the
    payload was fetched or valid. Consumers verify deferred extras bytes
    against it with ``compute_extras_digest`` at their own point of use.
    """

    model_call_id: str
    extras_digest_version: int
    extras_digest: str


@dataclass(frozen=True)
class LinearizedRow:
    """One verified root-to-leaf chain ready for framework publication.

    This is a proof that the base training row was verified. It carries no
    extras payloads; ``extras_commitments`` lists the selected calls'
    receipt-bound digests root-to-leaf for point-of-use verification.

    Chain placement fields (defaults describe the receipt's terminal chain):

    * ``terminal_model_call_id`` -- the leaf this row was linearized to.
    * ``chain_index`` -- position in ``LinearizedRollout.rows``; 0 is always
      the receipt's terminal chain.
    * ``chain_kind`` -- ``terminal`` or ``subagent`` (any other root of the
      rollout's manifest forest).
    * ``segment_index`` -- position along a context-rewrite sequence; always 0
      here (context rewrites are not linearized as segments).
    * ``boundary_parent_call_id`` -- reserved for context-rewrite roots; always
      ``None`` here.
    """

    rollout_id: str
    token_ids: list[int]
    token_mask: list[float]
    logprobs: list[float]
    model_call_ids: list[str]
    prompt_len: int
    weight_versions: list[int]
    weight_version_spans: list[WeightVersionSpan]
    link_spans: list[tuple[str, int, int]] = field(default_factory=list)
    extras_commitments: list[ExtrasCommitment] = field(default_factory=list)
    terminal_model_call_id: str = ""
    chain_index: int = 0
    chain_kind: str = CHAIN_KIND_TERMINAL
    segment_index: int = 0
    boundary_parent_call_id: str | None = None

    @property
    def call_ids(self) -> list[str]:
        """Compatibility spelling for existing framework consumers."""
        return self.model_call_ids


@dataclass(frozen=True)
class SkippedChain:
    """A non-terminal root that verified but produced no publishable row."""

    root_call_id: str
    reason: str


@dataclass(frozen=True)
class LinearizedRollout:
    """Every publishable chain of one rollout, terminal chain first."""

    rows: list[LinearizedRow]
    skipped: list[SkippedChain] = field(default_factory=list)
    num_roots: int = 0
    # Context-rewrite roots; always 0 here (kept for consumers that report it).
    num_boundary_roots: int = 0

    @property
    def terminal(self) -> LinearizedRow:
        """The receipt's declared terminal chain (always ``rows[0]``)."""
        return self.rows[0]


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
    snapshot: StagedCallBaseSnapshot,
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


def _recompute_integrity(snapshot: StagedCallBaseSnapshot) -> None:
    # Recomputed from the *committed* extras digest: the base row's
    # authenticity never depends on extras bytes being present.
    call_id = snapshot.model_call_id
    try:
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
            extras_digest=snapshot.extras_digest,
            chain_hash=snapshot.chain_hash,
            cumulative_hash=snapshot.cumulative_hash,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise _fail("invalid_snapshot", f"call {call_id}: {error}") from error
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


def _ancestry(records: dict[str, CallRecord], leaf_id: str) -> list[CallRecord]:
    """Root-to-leaf records of the chain ending at ``leaf_id`` (graph already validated)."""
    chain: list[CallRecord] = []
    cursor = records.get(leaf_id)
    while cursor is not None:
        chain.append(cursor)
        cursor = records.get(cursor.parent_call_id) if cursor.parent_call_id is not None else None
    chain.reverse()
    return chain


def _terminal_chain(
    receipt: RolloutReceipt,
    records: dict[str, CallRecord],
) -> list[CallRecord]:
    terminal = receipt.terminal_model_call_id
    if terminal is None:
        raise RebuildError("missing_terminal", "successful receipt has no terminal call")
    if terminal not in records:
        raise RebuildError("missing_terminal", f"terminal call {terminal} is absent")
    return _ancestry(records, terminal)


def _carry_boundary(snapshot: StagedCallBaseSnapshot) -> int:
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


def _verify_receipt_and_snapshots(
    receipt: RolloutReceipt,
    snapshots: Sequence[StagedCallBaseSnapshot],
) -> tuple[dict[str, CallRecord], dict[str, StagedCallBaseSnapshot]]:
    """Receipt-level checks plus per-row binding and integrity for every manifest row.

    Shared by both linearization entry points; runs once per rollout and
    covers rows on every chain, whether or not they are delivered.
    """
    if not isinstance(receipt, RolloutReceipt):
        raise TypeError("receipt must be a RolloutReceipt")
    _verify_versions(receipt)
    if receipt.failure_reason is not None:
        raise _fail("rollout_failed", receipt.failure_reason)
    if receipt.capture_poisoned:
        raise _fail("capture_poisoned", "receipt marks token capture as poisoned")
    if not receipt.manifest:
        raise _fail("empty_manifest", "successful receipt has no committed calls")

    records_by_id: dict[str, CallRecord] = {}
    staging_keys: set[str] = set()
    for record in receipt.manifest:
        if record.model_call_id in records_by_id:
            raise _fail("duplicate_manifest_row", "model_call_id values are not unique")
        if record.staging_key in staging_keys:
            raise _fail("duplicate_staging_key", "staging keys are not unique")
        records_by_id[record.model_call_id] = record
        staging_keys.add(record.staging_key)

    if len(snapshots) != len(receipt.manifest):
        raise _fail(
            "row_count_mismatch",
            f"{len(snapshots)} snapshots for {len(receipt.manifest)} manifest rows",
        )

    # One indexed pass binds each manifest row to its snapshot and validates it.
    snapshots_by_id: dict[str, StagedCallBaseSnapshot] = {}
    for record, snapshot in zip(receipt.manifest, snapshots):
        if not isinstance(snapshot, StagedCallBaseSnapshot):
            raise TypeError("snapshots must contain StagedCallBaseSnapshot values")
        if snapshot.model_call_id in snapshots_by_id:
            raise _fail("duplicate_snapshot", "model_call_id values are not unique")
        if snapshot.model_call_id not in records_by_id:
            raise _fail("snapshot_identity_mismatch", f"snapshot {snapshot.model_call_id} is not in the manifest")
        if snapshot.model_call_id != record.model_call_id:
            raise _fail(
                "snapshot_order_mismatch",
                f"key {record.staging_key} expected {record.model_call_id}, received {snapshot.model_call_id}",
            )
        snapshots_by_id[snapshot.model_call_id] = snapshot
        _compare_manifest_fields(receipt, record, snapshot)
        _recompute_integrity(snapshot)
    _validate_manifest_graph(records_by_id)
    return records_by_id, snapshots_by_id


def _linearize_chain(
    receipt: RolloutReceipt,
    records_by_id: dict[str, CallRecord],
    snapshots_by_id: dict[str, StagedCallBaseSnapshot],
    leaf_id: str,
    *,
    chain_index: int = 0,
    chain_kind: str = CHAIN_KIND_TERMINAL,
) -> LinearizedRow:
    """Verify and concatenate the root-to-``leaf_id`` chain into one row.

    Checks chain-hash continuity link by link, the carry-then-generation mask
    order of every delta, that the chain trains at least one token, and the
    leaf's cumulative hash over the concatenated tokens. Raises
    :class:`RebuildError` on any failure; callers decide whether a failure on
    a non-terminal chain is fatal.
    """
    chain = _ancestry(records_by_id, leaf_id)
    if not chain:
        raise RebuildError("missing_terminal", f"terminal call {leaf_id} is absent")

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
        # parent's chain hash.
        running_chain_hash = compute_chain_hash(running_chain_hash, snapshot.token_ids_delta)
        if record.chain_hash != running_chain_hash:
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
        raise RebuildError("empty_training_row", f"chain ending at {leaf_id} has no generated tokens")
    # Leaf-only whole-sequence anchor; per-record cumulative checks would
    # rehash O(n^2) tokens for no additional coverage over the chain hashes.
    if chain[-1].cumulative_hash != hash_token_ids(token_ids):
        raise RebuildError(
            "cumulative_hash_mismatch",
            f"leaf call {chain[-1].model_call_id} cumulative hash does not cover the linearized tokens",
        )

    extras_commitments = [
        ExtrasCommitment(
            model_call_id=record.model_call_id,
            extras_digest_version=record.extras_digest_version,
            extras_digest=record.extras_digest,
        )
        for record in chain
    ]
    return LinearizedRow(
        rollout_id=receipt.rollout_id,
        token_ids=token_ids,
        token_mask=token_mask,
        logprobs=logprobs,
        model_call_ids=model_call_ids,
        prompt_len=prompt_len,
        weight_versions=weight_versions,
        weight_version_spans=weight_version_spans,
        link_spans=link_spans,
        extras_commitments=extras_commitments,
        terminal_model_call_id=leaf_id,
        chain_index=chain_index,
        chain_kind=chain_kind,
    )


def verify_and_linearize(
    receipt: RolloutReceipt,
    snapshots: Sequence[StagedCallBaseSnapshot],
) -> LinearizedRow:
    """Verify an untrusted staged base set and linearize the declared terminal chain.

    Metadata-only: extras payloads are never read. The returned row's
    ``extras_commitments`` carry the selected calls' receipt-bound digests for
    consumers to verify fetched extras against at their own point of use.
    """
    records_by_id, snapshots_by_id = _verify_receipt_and_snapshots(receipt, snapshots)
    chain = _terminal_chain(receipt, records_by_id)
    return _linearize_chain(receipt, records_by_id, snapshots_by_id, chain[-1].model_call_id)


def _root_prompts(
    roots: Sequence[CallRecord],
    snapshots_by_id: Mapping[str, StagedCallBaseSnapshot],
) -> dict[str, tuple[int, ...]]:
    """Root -> the prompt tokens of its first request (the carry span of the root call's delta).

    Two roots whose first requests carried the same prompt tokens were started
    by the same request; the one the session did not go on with is a retry of
    the other. A root whose delta is not carry-then-generation has no readable
    prompt and joins no retry pool (its chain is skipped with the custody error
    code when it is linearized), nor does a root that carried no prompt token.
    """
    prompts: dict[str, tuple[int, ...]] = {}
    for root in roots:
        snapshot = snapshots_by_id[root.model_call_id]
        try:
            boundary = _carry_boundary(snapshot)
        except RebuildError:
            continue
        if boundary:
            prompts[root.model_call_id] = tuple(snapshot.token_ids_delta[:boundary])
    return prompts


def _abandoned_roots(
    roots: Sequence[CallRecord],
    children: Mapping[str, Sequence[CallRecord]],
    prompts: Mapping[str, tuple[int, ...]],
    terminal_root_id: str,
) -> set[str]:
    """Roots that are dead retries of another root.

    A retry pool is the set of roots whose first request carried identical
    prompt tokens (a retried first call beside the attempt that was
    continued). Inside a pool the terminal heuristic's ``survivors`` rule
    applies: an unextended member (no token child) is eliminated only when
    another member was extended, and the receipt's terminal root counts as
    extended because the session ended there. A childless root with a
    distinct prompt is a genuine single-call session (a subagent that
    answered in one call) and is kept; byte-identical duplicates are
    collapsed later by cumulative hash.
    """

    def extended(record: CallRecord) -> bool:
        call_id = record.model_call_id
        return call_id == terminal_root_id or bool(children.get(call_id))

    by_prompt: dict[tuple[int, ...], list[CallRecord]] = {}
    for root in roots:
        prompt = prompts.get(root.model_call_id)
        if prompt is not None:
            by_prompt.setdefault(prompt, []).append(root)
    abandoned: set[str] = set()
    for pool in by_prompt.values():
        if len(pool) > 1 and any(extended(record) for record in pool):
            abandoned.update(record.model_call_id for record in pool if not extended(record))
    return abandoned


def verify_and_linearize_all(
    receipt: RolloutReceipt,
    snapshots: Sequence[StagedCallBaseSnapshot],
) -> LinearizedRollout:
    """Verify once and linearize every publishable chain of the manifest forest.

    ``rows[0]`` is the receipt's declared terminal chain, built exactly as
    :func:`verify_and_linearize` builds it (and subject to the same errors).
    Every other root (a subagent session) is first checked for being a dead
    retry (an unextended root beside an extended one with the same prompt
    tokens is ``abandoned_root``; a childless root with a prompt of its own is
    a genuine single-call session and is published), then descended to its
    leaf (a surviving fork is ``ambiguous_leaf``), de-duplicated by the leaf's
    cumulative hash (an identical retry is ``abandoned_root``) and linearized
    root-to-leaf with the same per-chain checks. A per-chain custody failure
    on a non-terminal chain skips that chain with the error code; the
    terminal chain's failures raise.

    Rows follow the terminal, then the other roots by admission time
    (unstamped last) and manifest position, all as ``subagent`` rows.
    """
    records_by_id, snapshots_by_id = _verify_receipt_and_snapshots(receipt, snapshots)
    terminal_chain = _terminal_chain(receipt, records_by_id)
    terminal_root_id = terminal_chain[0].model_call_id
    terminal_call_ids = {record.model_call_id for record in terminal_chain}

    manifest = list(receipt.manifest)
    roots = [record for record in manifest if record.parent_call_id is None]
    children = children_by_parent(manifest)
    abandoned = _abandoned_roots(roots, children, _root_prompts(roots, snapshots_by_id), terminal_root_id)
    others = sorted(
        (root for root in roots if root.model_call_id != terminal_root_id),
        key=root_order_key(manifest),
    )

    rows: list[LinearizedRow] = [
        _linearize_chain(receipt, records_by_id, snapshots_by_id, terminal_chain[-1].model_call_id)
    ]
    skipped: list[SkippedChain] = []
    published_hashes = {terminal_chain[-1].cumulative_hash}
    for root in others:
        root_id = root.model_call_id
        if root_id in abandoned:
            skipped.append(SkippedChain(root_id, SKIP_ABANDONED_ROOT))
            continue
        leaf_id = descend_to_leaf(root_id, children)
        if leaf_id is None:
            skipped.append(SkippedChain(root_id, SKIP_AMBIGUOUS_LEAF))
            continue
        if leaf_id in terminal_call_ids:
            # Distinct roots own disjoint trees; this only guards a caller
            # that hands in a terminal chain rooted elsewhere.
            skipped.append(SkippedChain(root_id, SKIP_LEAF_ON_TERMINAL_CHAIN))
            continue
        leaf_hash = records_by_id[leaf_id].cumulative_hash
        if leaf_hash in published_hashes:
            # Byte-identical retry of a published chain: training it twice
            # would double-count the same tokens.
            skipped.append(SkippedChain(root_id, SKIP_ABANDONED_ROOT))
            continue
        try:
            row = _linearize_chain(
                receipt,
                records_by_id,
                snapshots_by_id,
                leaf_id,
                chain_index=len(rows),
                chain_kind=CHAIN_KIND_SUBAGENT,
            )
        except RebuildError as error:
            # The terminal chain was verified on its own above; a corrupt
            # secondary chain says nothing about the canonical row.
            skipped.append(SkippedChain(root_id, error.code))
            continue
        published_hashes.add(leaf_hash)
        rows.append(row)

    return LinearizedRollout(rows=rows, skipped=skipped, num_roots=len(roots))


def linearize(
    rollout_id: str,
    snapshots: list[StagedCallBaseSnapshot],
    manifest: list[CallRecord],
    *,
    terminal_hint: str | None = None,
) -> LinearizedRow:
    """Compatibility wrapper that still executes the production verifier."""
    if terminal_hint is None:
        # ``RolloutReceipt`` rejects an unpoisoned receipt without a terminal;
        # surface the same rebuild failure the verifier reports for that case.
        raise _fail("missing_terminal", "successful receipt has no terminal call")
    receipt = RolloutReceipt(
        rollout_id=rollout_id,
        terminal_model_call_id=terminal_hint,
        manifest=manifest,
        terminal_selection="declared",
    )
    return verify_and_linearize(receipt, snapshots)
