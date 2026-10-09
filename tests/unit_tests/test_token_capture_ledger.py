# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ledger and admission invariants for external-staging token capture.

These re-express the gate invariants that survive the gate's removal:
tri-state admission, commit ordering, same-call commit idempotency, and
fail-closed poisoning.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nemo_gym.token_id_capture.fingerprint import FINGERPRINT_VERSION
from nemo_gym.token_id_capture.lineage import FileLineageStore, InMemoryLineageStore, _custody_columns
from nemo_gym.token_id_capture.protocols import CaptureLedger, RolloutRemovalPayload, RolloutRetiredError
from nemo_gym.token_id_capture.records import ParentResolutionStatus, compute_digest
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
from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    CaptureLedgerCommit,
    RolloutManifest,
    RolloutRemoval,
)


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
        fingerprint_version=FINGERPRINT_VERSION,
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


def test_capture_ledger_type_hints_resolve_at_runtime():
    """``CaptureLedgerCommit`` must not hide behind TYPE_CHECKING (get_type_hints resolves it)."""
    import typing

    from nemo_gym.token_id_capture.staging.records import CaptureLedgerCommit

    assert typing.get_type_hints(CaptureLedger.record)["commit"] is CaptureLedgerCommit


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
@pytest.mark.parametrize(
    "changed",
    [
        {"weight_version": 18},
        {"response_id": "chatcmpl-other"},
        {"admitted_at": 1.0},
    ],
)
async def test_recommit_with_changed_metadata_conflicts_even_when_index_agrees(store, changed):
    """Fields the lineage index does not keep still make a re-commit a conflict."""
    await _record_call_1(store)
    with pytest.raises(ValueError, match="conflicting"):
        await store.record(
            _commit(
                _call_record("c1").model_copy(update=changed),
                [USER_1],
                [ASSISTANT_1],
                staging_chain=("r1/c1",),
            )
        )
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert [record.model_call_id for record in manifest.records] == ["c1"]


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
@pytest.mark.parametrize("version", [None, 0, FINGERPRINT_VERSION - 1, FINGERPRINT_VERSION, FINGERPRINT_VERSION + 1])
async def test_file_store_requires_current_fingerprint_version(tmp_path, version):
    import json

    writer = FileLineageStore(tmp_path)
    await _record_call_1(writer)
    path = tmp_path / "r1.lineage.jsonl"
    row = json.loads(path.read_text())
    if version is None:
        row.pop("fingerprint_version")
    else:
        row["fingerprint_version"] = version
    # Keep the matching fingerprint and context; only the stored version differs.
    path.write_text(json.dumps(row) + "\n")

    reader = FileLineageStore(tmp_path)
    resolution = await reader.resolve("r1", [USER_1, ASSISTANT_1, USER_2])
    context = await _admit(reader, [USER_1, ASSISTANT_1, USER_2])
    if version == FINGERPRINT_VERSION:
        assert resolution.status == ParentResolutionStatus.RESOLVED
        assert resolution.match.model_call_id == "c1"
        assert context.capture_admission.staging_chain == ["r1/c1"]
    else:
        assert resolution.status == ParentResolutionStatus.UNRESOLVED
        assert resolution.match is None
        assert context.capture_admission is None
        manifest = RolloutManifest.model_validate(await reader.manifest("r1"))
        assert [failure.reason for failure in manifest.failures] == [UNRESOLVED_PARENT_REASON]


@pytest.mark.asyncio
async def test_file_store_ignores_old_version_when_current_match_is_appended(tmp_path):
    writer = FileLineageStore(tmp_path)
    reader = FileLineageStore(tmp_path)
    old_record = _call_record("old").model_copy(update={"fingerprint_version": FINGERPRINT_VERSION - 1})
    await writer.record(_commit(old_record, [USER_1], [ASSISTANT_1]))
    assert (await reader.resolve("r1", [USER_1, ASSISTANT_1, USER_2])).match is None

    await _record_call_1(writer)
    match = (await reader.resolve("r1", [USER_1, ASSISTANT_1, USER_2])).match
    assert match is not None and match.model_call_id == "c1"
    assert match.staging_chain == ("r1/c1",)


@pytest.mark.asyncio
async def test_lineage_only_rows_do_not_enter_the_manifest(tmp_path):
    """Local-capture rows (no custody columns) resolve but are not manifest rows."""
    import json

    from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint, conversation_digest

    store = FileLineageStore(tmp_path)
    lineage_only_row = {
        "model_call_id": "c1",
        "fingerprint_version": FINGERPRINT_VERSION,
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
async def test_unversioned_legacy_token_carrying_row_cannot_resolve_or_anchor_a_chain(tmp_path):
    """Unversioned pre-chain external rows cannot supply a parent token prefix."""
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
            if key not in ("chain_hash", "cumulative_hash", "response_id", "fingerprint_version")
        },
    }
    path = tmp_path / "r1.lineage.jsonl"
    path.write_text(json.dumps(legacy_row, sort_keys=True, separators=(",", ":")) + "\n")

    match = (await store.resolve("r1", [USER_1, ASSISTANT_1, USER_2])).match
    assert match is None

    context = await _admit(store, [USER_1, ASSISTANT_1, USER_2])
    assert context.capture_admission is None
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    # The pre-response-id custody row is no longer manifest-expressible: it
    # poisons the rollout (fail-closed) instead of being tolerated.
    assert sorted(failure.reason for failure in manifest.failures) == sorted(
        ["ledger_row_missing_response_id", UNRESOLVED_PARENT_REASON]
    )


# --- retire and delete -------------------------------------------------------


@pytest.mark.asyncio
async def test_retire_removes_ledgers_and_reports_absent_rollouts(store):
    await _record_call_1(store)
    await store.record_failure("r2", "c1", "worker_capture_failed")

    result = RolloutRemoval.model_validate(await store.retire(["r1", "r2", "r-none", "r1"]))

    # Duplicates in one batch count once.
    assert result.removed == ["r1", "r2"]
    assert result.absent == ["r-none"]
    for rollout_id in ("r1", "r2"):
        with pytest.raises(RolloutRetiredError):
            await store.has_rows(rollout_id)
        with pytest.raises(RolloutRetiredError):
            await store.manifest(rollout_id)
    # The retired call can no longer anchor a continuation.
    assert (await store.resolve("r1", [USER_1, ASSISTANT_1, USER_2])).match is None


@pytest.mark.asyncio
async def test_retiring_again_is_a_no_op(store):
    await _record_call_1(store)
    await store.retire(["r1"])
    result = RolloutRemoval.model_validate(await store.retire(["r1"]))
    assert result.removed == [] and result.absent == ["r1"]


@pytest.mark.asyncio
async def test_retired_rollouts_discard_later_records_and_failures(store):
    await _record_call_1(store)
    # Retiring a rollout that never recorded a call fences it as well.
    await store.retire(["r1", "r-unstarted"])

    await _record_call_1(store)
    await store.record_failure("r1", "c2", "worker_capture_failed")
    await _record_call_1(store, rollout_id="r-unstarted")

    for rollout_id in ("r1", "r-unstarted"):
        with pytest.raises(RolloutRetiredError):
            await store.has_rows(rollout_id)
    # Other rollouts are unaffected.
    await _record_call_1(store, rollout_id="r2")
    assert await store.has_rows("r2")


@pytest.mark.asyncio
async def test_delete_removes_the_fence_so_the_rollout_id_can_be_reused(store):
    await _record_call_1(store)
    await store.retire(["r1"])

    result = RolloutRemoval.model_validate(await store.delete(["r1"]))
    assert result.removed == [] and result.absent == ["r1"]

    await _record_call_1(store)
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert [record.model_call_id for record in manifest.records] == ["c1"]


@pytest.mark.asyncio
async def test_delete_removes_a_live_ledger(store):
    await _record_call_1(store)
    result = RolloutRemoval.model_validate(await store.delete(["r1"]))
    assert result.removed == ["r1"]
    assert not await store.has_rows("r1")


@pytest.mark.asyncio
async def test_file_store_retirement_is_visible_to_every_worker(tmp_path):
    retirer = FileLineageStore(tmp_path)
    late_writer = FileLineageStore(tmp_path)
    await _record_call_1(late_writer)

    await retirer.retire(["r1"])
    await late_writer.record(_commit(_call_record("c2"), [USER_1], [ASSISTANT_1]))
    # A fresh process sees retirements made before it started.
    await _record_call_1(FileLineageStore(tmp_path))
    await late_writer.record_failure("r1", "c3", "worker_capture_failed")

    # No ledger was recreated; only the fence remains.
    assert sorted(path.name for path in tmp_path.glob("*.lineage.*")) == ["r1.lineage.retired"]


@pytest.mark.asyncio
async def test_file_store_checks_the_fence_only_before_creating_a_ledger(tmp_path):
    store = FileLineageStore(tmp_path)
    with patch.object(store, "_is_retired", wraps=store._is_retired) as is_retired:
        await _record_call_1(store)
        await store.record(_commit(_call_record("c2"), [USER_1], [ASSISTANT_1]))
        await store.record_failure("r1", "c3", "worker_capture_failed")
    # Only the first write, which creates the ledger, looks for a fence: a ledger with rows is not retired.
    assert is_retired.call_count == 1


@pytest.mark.asyncio
async def test_file_store_delete_removes_every_ledger_and_fence_file(tmp_path):
    store = FileLineageStore(tmp_path)
    for rollout_id in ("r1", "r2"):
        await _record_call_1(store, rollout_id=rollout_id)
    await store.retire(["r1"])

    await store.delete(["r1", "r2"])

    assert list(tmp_path.glob("*.lineage.*")) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["retire", "delete"])
async def test_a_bare_rollout_id_string_is_rejected(store, method):
    """A string is a sequence of one-character IDs; treating it as a batch would remove the wrong rollouts."""
    await _record_call_1(store)
    with pytest.raises(TypeError, match="sequence of rollout ids"):
        await getattr(store, method)("r1")
    assert await store.has_rows("r1")


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["retire", "delete"])
async def test_file_store_rejects_the_whole_batch_on_an_invalid_rollout_id(tmp_path, method):
    store = FileLineageStore(tmp_path)
    await _record_call_1(store)
    with pytest.raises(ValueError, match="Invalid rollout id"):
        await getattr(store, method)(["r1", "../escape"])
    assert await store.has_rows("r1")
    assert not (tmp_path / "r1.lineage.retired").exists()


@pytest.mark.asyncio
async def test_retiring_again_writes_nothing_but_syncs_once(tmp_path):
    store = FileLineageStore(tmp_path)
    await _record_call_1(store)
    await store.retire(["r1"])
    fence = tmp_path / "r1.lineage.retired"
    fence_mtime = fence.stat().st_mtime_ns

    with patch.object(store, "_fsync_ledger_root", wraps=store._fsync_ledger_root) as fsync_root:
        result = await store.retire(["r1"])

    # A retried retire writes no metadata. It still syncs once: an overlapping retire may have written the
    # fence without syncing it yet, and this call must not report success before the fence is durable.
    assert result == {"removed": [], "absent": ["r1"]}
    assert fence.stat().st_mtime_ns == fence_mtime
    assert fsync_root.call_count == 1


@pytest.mark.asyncio
async def test_retire_syncs_an_existing_fence_before_deleting_the_ledger(tmp_path):
    """A crashed retire can leave a fence that was never synced next to a ledger that was never deleted."""
    store = FileLineageStore(tmp_path)
    await _record_call_1(store)
    (tmp_path / "r1.lineage.retired").touch()
    events = []
    real_fsync, real_unlink = store._fsync_ledger_root, type(tmp_path).unlink

    def fsync_root():
        events.append("fsync")
        real_fsync()

    def unlink(path, *args, **kwargs):
        if path.name == "r1.lineage.jsonl":
            events.append("unlink ledger")
        return real_unlink(path, *args, **kwargs)

    with patch.object(store, "_fsync_ledger_root", fsync_root), patch.object(type(tmp_path), "unlink", unlink):
        result = await store.retire(["r1"])

    assert result == {"removed": ["r1"], "absent": []}
    assert events[:2] == ["fsync", "unlink ledger"]


@pytest.mark.asyncio
async def test_a_discarded_late_row_names_its_staging_key(tmp_path, caplog):
    """The worker staged the late call's tokens; the warning is the only remaining pointer to them."""
    store = FileLineageStore(tmp_path)
    await _record_call_1(store)
    await store.retire(["r1"])

    with caplog.at_level("WARNING"):
        await store.record(_commit(_call_record("c2"), [USER_1], [ASSISTANT_1], staging_chain=("r1/c1", "r1/c2")))

    assert any("r1/c2" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_reading_a_retired_rollouts_manifest_fails(store):
    """An empty manifest would look like a rollout that made no calls."""
    await _record_call_1(store)
    await store.retire(["r1", "never-recorded"])

    for rollout_id in ("r1", "never-recorded"):
        with pytest.raises(RolloutRetiredError):
            await store.manifest(rollout_id)
    # Delete removes the fence, so the ID reads as unused again.
    await store.delete(["r1"])
    assert RolloutManifest.model_validate(await store.manifest("r1")).records == []


def test_retire_and_delete_declare_their_result_shape():
    import typing

    from nemo_gym.token_id_capture.protocols import TokenSource
    from nemo_gym.token_id_capture.store import TokenCaptureStore

    for method in (
        CaptureLedger.retire,
        CaptureLedger.delete,
        FileLineageStore.retire,
        InMemoryLineageStore.delete,
        TokenSource.retire,
        TokenSource.delete,
        TokenCaptureStore.retire_now,
        TokenCaptureStore.delete_now,
    ):
        assert typing.get_type_hints(method)["return"] is RolloutRemovalPayload


@pytest.mark.asyncio
@pytest.mark.parametrize("request_items", [[USER_1, ASSISTANT_1, USER_2], [USER_1]], ids=["continuation", "root"])
async def test_a_call_on_a_retired_rollout_is_not_admitted_for_staging(store, request_items):
    """A late continuation, or the first call of a duplicate run, would stage tokens no manifest names."""
    await _record_call_1(store)
    await store.retire(["r1"])

    context = await _admit(store, request_items)

    assert context.capture_admission is None
    with pytest.raises(RolloutRetiredError):
        await store.has_rows("r1")


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["retire", "delete"])
async def test_removing_ledgers_forgets_their_cached_rows(tmp_path, method):
    store = FileLineageStore(tmp_path)
    for index in range(5):
        await _record_call_1(store, rollout_id=f"r{index}")

    await getattr(store, method)([f"r{index}" for index in range(5)])

    assert store._ledger_cache == {}
    # A cache that also tracks its size must account for every removal, or it slowly fills with nothing.
    assert getattr(store, "_ledger_cache_weight", 0) == 0


@pytest.mark.asyncio
async def test_file_store_delete_syncs_every_non_empty_batch(tmp_path):
    """An overlapping delete may have removed the files without syncing yet, so a delete that finds nothing still syncs."""
    store = FileLineageStore(tmp_path)
    await _record_call_1(store)

    with patch.object(store, "_fsync_ledger_root", wraps=store._fsync_ledger_root) as fsync_root:
        await store.delete(["r1"])
        await store.delete(["r1", "never-recorded"])

    assert fsync_root.call_count == 2


@pytest.mark.asyncio
async def test_retire_does_not_remove_a_ledger_recreated_after_a_delete_between_its_phases(tmp_path):
    """Between fencing and removing, a delete can clear the fence and a reused rollout ID can record again."""
    store = FileLineageStore(tmp_path)
    await _record_call_1(store)
    real_fsync = store._fsync_ledger_root
    reused = []

    def fsync_root():
        real_fsync()
        if not reused:
            reused.append(True)
            store._delete(["r1"])
            store._record(_commit(_call_record("c1"), [USER_1], [ASSISTANT_1]))

    with patch.object(store, "_fsync_ledger_root", fsync_root):
        await store.retire(["r1"])

    # The new attempt's ledger survives, unfenced.
    assert not (tmp_path / "r1.lineage.retired").exists()
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert [record.model_call_id for record in manifest.records] == ["c1"]
