# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ledger and admission invariants for external-staging token capture.

These re-express the gate invariants that survive the gate's removal:
tri-state admission, commit ordering, same-call commit idempotency, and
fail-closed poisoning.
"""

from __future__ import annotations

import pytest

from nemo_gym.token_id_capture.lineage import FileLineageStore, InMemoryLineageStore, _custody_columns
from nemo_gym.token_id_capture.protocols import CaptureLedger
from nemo_gym.token_id_capture.records import compute_digest
from nemo_gym.token_id_capture.sink import (
    UNRESOLVED_PARENT_REASON,
    CaptureContext,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
)
from nemo_gym.token_id_capture.staging.digest import (
    EMPTY_EXTRAS_DIGEST,
    compute_chain_hash,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.records import CallRecord, CaptureLedgerCommit, RolloutManifest


USER_1 = {"role": "user", "content": "solve the task"}
ASSISTANT_1 = {"role": "assistant", "content": "first answer"}
USER_2 = {"role": "user", "content": "tool result"}
ASSISTANT_2 = {"role": "assistant", "content": "second answer"}
USER_3 = {"role": "user", "content": "follow up"}
ASSISTANT_SEEDED = {"role": "assistant", "content": "seeded turn nobody served"}

TOKENS_1 = list(range(900))
STAGING_DIGEST = "a" * 64
CHAIN_HASH_1 = compute_chain_hash(None, TOKENS_1)
CUMULATIVE_HASH_1 = hash_token_ids(TOKENS_1)


def _call_record(
    model_call_id: str,
    *,
    parent_call_id: str | None = None,
    prev_len: int = 0,
    cumulative_hash: str = CUMULATIVE_HASH_1,
    chain_hash: str = CHAIN_HASH_1,
    delta_len: int | None = None,
    admitted_at: float | None = 1_755_600_000.25,
) -> CallRecord:
    if delta_len is None:
        delta_len = len(TOKENS_1) - prev_len
    return CallRecord(
        model_call_id=model_call_id,
        parent_call_id=parent_call_id,
        staging_key=f"r1/{model_call_id}",
        weight_version=17,
        prev_len=prev_len,
        delta_len=delta_len,
        cum_len=prev_len + delta_len,
        digest=STAGING_DIGEST,
        extras_digest=EMPTY_EXTRAS_DIGEST,
        mode="text" if parent_call_id is None else "token_in",
        response_id=f"chatcmpl-{model_call_id}",
        admitted_at=admitted_at,
        chain_hash=chain_hash,
        cumulative_hash=cumulative_hash,
    )


def _commit(
    record: CallRecord,
    request_items: list[dict],
    response_items: list[dict],
    *,
    rollout_id: str = "r1",
    staging_chain: tuple[str, ...] | None = None,
) -> CaptureLedgerCommit:
    return CaptureLedgerCommit(
        rollout_id=rollout_id,
        record=record,
        staging_chain=staging_chain if staging_chain is not None else (record.staging_key,),
        request_items=request_items,
        response_items=response_items,
    )


async def _record_call_1(store, rollout_id: str = "r1") -> None:
    # Token-free custody row, exactly as the external commit hook writes it.
    await store.record(
        _commit(
            _call_record("c1"),
            [USER_1],
            [ASSISTANT_1],
            rollout_id=rollout_id,
            staging_chain=(f"{rollout_id}/c1",),
        )
    )


@pytest.fixture(params=["file", "memory"])
def store(request, tmp_path):
    if request.param == "file":
        return FileLineageStore(tmp_path)
    return InMemoryLineageStore()


def test_stores_implement_capture_ledger(store):
    assert isinstance(store, CaptureLedger)


@pytest.mark.asyncio
async def test_ledger_row_round_trips_token_free_manifest(store):
    await _record_call_1(store)
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert manifest.rollout_id == "r1"
    assert manifest.failures == []
    (record,) = manifest.records
    # The manifest row is exactly the committed ``CallRecord``.
    assert record == _call_record("c1")
    assert record.staging_key == "r1/c1"
    assert record.admitted_at == 1_755_600_000.25
    assert record.digest == STAGING_DIGEST
    assert record.chain_hash == CHAIN_HASH_1
    assert record.cumulative_hash == CUMULATIVE_HASH_1
    # Cumulative token IDs stay off the manifest surface.
    assert "cumulative_token_ids" not in manifest.model_dump()["records"][0]


@pytest.mark.asyncio
async def test_row_without_admitted_at_still_validates(store):
    await store.record(_commit(_call_record("c1", admitted_at=None), [USER_1], [ASSISTANT_1]))
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    (record,) = manifest.records
    assert record.admitted_at is None


@pytest.mark.asyncio
async def test_same_call_commit_is_idempotent_and_conflicts_raise(store):
    await _record_call_1(store)
    await _record_call_1(store)  # identical replay is a no-op
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert len(manifest.records) == 1
    with pytest.raises(ValueError, match="conflicting"):
        await store.record(
            _commit(
                _call_record("c1", cumulative_hash=hash_token_ids(TOKENS_1 + [1])),
                [USER_1],
                [ASSISTANT_1],
            )
        )


@pytest.mark.asyncio
async def test_failure_rows_poison_and_never_resolve(store):
    await _record_call_1(store)
    await store.record_failure("r1", "c2", "worker_capture_failed")
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert [failure.reason for failure in manifest.failures] == ["worker_capture_failed"]
    # The committed parent still resolves; the failure row is invisible to lineage.
    match = (await store.resolve("r1", [USER_1, ASSISTANT_1, USER_2])).match
    assert match is not None and match.model_call_id == "c1"
    assert await store.has_rows("r1")


@pytest.mark.asyncio
async def test_has_rows_is_false_for_untouched_rollout(store):
    assert not await store.has_rows("r-none")
    assert RolloutManifest.model_validate(await store.manifest("r-none")).records == []


async def _admit(store, request_items, rollout_id="r1", model_call_id="c2"):
    context = CaptureContext(
        rollout_id=rollout_id,
        model_call_id=model_call_id,
        token_sink=None,
        lineage_store=store,
        external_staging=True,
    )
    token = set_token_sink(context)
    try:
        await resolve_parent(request_items)
    finally:
        reset_token_sink(token)
    return context


@pytest.mark.asyncio
async def test_admission_match_uses_staging_chain_without_wire_prefix(store):
    await _record_call_1(store)
    context = await _admit(store, [USER_1, ASSISTANT_1, USER_2])
    admission = context.capture_admission
    assert admission is not None and admission.mode == "token_in"
    assert admission.parent_call_id == "c1"
    assert admission.required_prefix_token_ids == []
    assert admission.staging_chain == ["r1/c1"]
    assert admission.prev_len == len(TOKENS_1)
    assert admission.parent_chain_hash == CHAIN_HASH_1
    assert context.parent_staging_chain == ["r1/c1"]
    assert context.parent_chain_hash == CHAIN_HASH_1
    assert context.request_items == [USER_1, ASSISTANT_1, USER_2]


@pytest.mark.asyncio
async def test_staging_chain_grows_across_external_calls(store):
    await _record_call_1(store)
    tokens_2 = TOKENS_1 + [901, 902]
    chain_hash_2 = compute_chain_hash(CHAIN_HASH_1, [901, 902])
    await store.record(
        _commit(
            _call_record(
                "c2",
                parent_call_id="c1",
                prev_len=len(TOKENS_1),
                delta_len=2,
                chain_hash=chain_hash_2,
                cumulative_hash=hash_token_ids(tokens_2),
            ),
            [USER_1, ASSISTANT_1, USER_2],
            [ASSISTANT_2],
            staging_chain=("r1/c1", "r1/c2"),
        )
    )

    context = await _admit(
        store,
        [USER_1, ASSISTANT_1, USER_2, ASSISTANT_2, USER_3],
        model_call_id="c3",
    )

    admission = context.capture_admission
    assert admission is not None
    assert admission.parent_call_id == "c2"
    assert admission.prev_len == len(tokens_2)
    assert admission.staging_chain == ["r1/c1", "r1/c2"]
    assert admission.required_prefix_token_ids == []
    assert admission.parent_chain_hash == chain_hash_2


@pytest.mark.asyncio
async def test_admission_empty_fingerprint_is_text_root(store):
    context = await _admit(store, [USER_1], model_call_id="c1")
    admission = context.capture_admission
    assert admission is not None and admission.mode == "text"
    assert admission.parent_call_id is None


@pytest.mark.asyncio
async def test_admission_seeded_history_on_empty_ledger_is_text_root(store):
    context = await _admit(store, [USER_1, ASSISTANT_SEEDED, USER_2], model_call_id="c1")
    admission = context.capture_admission
    assert admission is not None and admission.mode == "text"


@pytest.mark.asyncio
async def test_admission_unresolved_poisons_instead_of_new_root(store):
    await _record_call_1(store)
    # Assistant history that matches no committed call on a non-empty ledger.
    context = await _admit(store, [USER_1, ASSISTANT_SEEDED, USER_2])
    assert context.capture_admission is None
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert [failure.reason for failure in manifest.failures] == [UNRESOLVED_PARENT_REASON]
    assert manifest.failures[0].model_call_id == "c2"


@pytest.mark.asyncio
async def test_ambiguous_siblings_are_unresolved(store):
    """Two committed calls with identical text but different tokens must never resolve; the call poisons."""
    await _record_call_1(store)
    # Same request and response text as c1, but a different token sequence.
    sibling = _call_record("c1b", cumulative_hash=hash_token_ids(TOKENS_1 + [1]))
    await store.record(_commit(sibling, [USER_1], [ASSISTANT_1]))
    context = await _admit(store, [USER_1, ASSISTANT_1, USER_2])
    assert context.capture_admission is None
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert [failure.reason for failure in manifest.failures] == [UNRESOLVED_PARENT_REASON]


@pytest.mark.asyncio
async def test_commit_ordering_parent_resolvable_only_after_record(store):
    # Before the ledger row exists, the follow-up cannot resolve a parent.
    assert (await store.resolve("r1", [USER_1, ASSISTANT_1, USER_2])).match is None
    await _record_call_1(store)
    match = (await store.resolve("r1", [USER_1, ASSISTANT_1, USER_2])).match
    assert match is not None
    # Custody rows resolve token-free: continuity rides the chain hash.
    assert list(match.cumulative_token_ids) == []
    assert match.prev_len == len(TOKENS_1)
    assert match.chain_hash == CHAIN_HASH_1


@pytest.mark.asyncio
async def test_file_store_cross_handle_visibility(tmp_path):
    writer = FileLineageStore(tmp_path)
    reader = FileLineageStore(tmp_path)
    await _record_call_1(writer)
    await writer.record_failure("r1", "c9", "worker_capture_failed")
    manifest = RolloutManifest.model_validate(await reader.manifest("r1"))
    assert len(manifest.records) == 1 and len(manifest.failures) == 1
    assert await reader.has_rows("r1")


@pytest.mark.asyncio
async def test_lineage_only_rows_do_not_enter_the_manifest(tmp_path):
    """Local-capture rows (no custody columns) resolve but are not manifest rows."""
    import json

    from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint, conversation_digest

    store = FileLineageStore(tmp_path)
    lineage_only_row = {
        "model_call_id": "c1",
        "fingerprint": assistant_fingerprint([USER_1, ASSISTANT_1]),
        "context_len": 1,
        "context_digest": conversation_digest([USER_1]),
        "cumulative_token_ids": TOKENS_1,
        "digest": compute_digest(TOKENS_1),
    }
    (tmp_path / "r1.lineage.jsonl").write_text(
        json.dumps(lineage_only_row, sort_keys=True, separators=(",", ":")) + "\n"
    )
    match = (await store.resolve("r1", [USER_1, ASSISTANT_1, USER_2])).match
    assert match is not None and list(match.cumulative_token_ids) == TOKENS_1
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert manifest.records == [] and manifest.failures == []


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["chain_hash", "cumulative_hash"])
async def test_custody_row_missing_a_chain_digest_poisons_the_manifest(tmp_path, missing):
    """A committed row without either chain digest cannot anchor verification."""
    import json

    from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint, conversation_digest
    from nemo_gym.token_id_capture.records import LEDGER_ROW_MISSING_CHAIN_HASH_REASON

    store = FileLineageStore(tmp_path)
    row = {
        "model_call_id": "c1",
        "fingerprint": assistant_fingerprint([USER_1, ASSISTANT_1]),
        "context_len": 1,
        "context_digest": conversation_digest([USER_1]),
        "digest": CUMULATIVE_HASH_1,
        **{k: v for k, v in _custody_columns(_call_record("c1"), ("r1/c1",)).items() if k != missing},
    }
    (tmp_path / "r1.lineage.jsonl").write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert manifest.records == []
    assert [failure.reason for failure in manifest.failures] == [LEDGER_ROW_MISSING_CHAIN_HASH_REASON]


@pytest.mark.asyncio
async def test_legacy_token_carrying_row_resolves_but_cannot_anchor_a_chain(tmp_path):
    """Pre-chain external rows stay readable; extending them fails closed."""
    import json

    from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint, conversation_digest

    store = FileLineageStore(tmp_path)
    legacy_row = {
        "model_call_id": "c1",
        "fingerprint": assistant_fingerprint([USER_1, ASSISTANT_1]),
        "context_len": 1,
        "context_digest": conversation_digest([USER_1]),
        "cumulative_token_ids": TOKENS_1,
        "digest": compute_digest(TOKENS_1),
        **{
            key: value
            for key, value in _custody_columns(_call_record("c1"), ("r1/c1",)).items()
            if key not in ("chain_hash", "cumulative_hash", "response_id")
        },
    }
    path = tmp_path / "r1.lineage.jsonl"
    path.write_text(json.dumps(legacy_row, sort_keys=True, separators=(",", ":")) + "\n")

    match = (await store.resolve("r1", [USER_1, ASSISTANT_1, USER_2])).match
    assert match is not None
    assert list(match.cumulative_token_ids) == TOKENS_1
    assert match.chain_hash == ""

    context = await _admit(store, [USER_1, ASSISTANT_1, USER_2])
    assert context.capture_admission is None
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    # The pre-response-id custody row is no longer manifest-expressible: it
    # poisons the rollout (fail-closed) instead of being tolerated.
    assert sorted(failure.reason for failure in manifest.failures) == sorted(
        ["ledger_row_missing_response_id", UNRESOLVED_PARENT_REASON]
    )
