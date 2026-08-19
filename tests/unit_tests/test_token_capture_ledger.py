# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ledger and admission invariants for external-staging token capture.

These re-express the gate invariants that survive the gate's removal:
tri-state admission, commit ordering, same-call commit idempotency, and
fail-closed poisoning.
"""

from __future__ import annotations

import pytest

from nemo_gym.token_id_capture.lineage import FileLineageStore, InMemoryLineageStore
from nemo_gym.token_id_capture.records import compute_digest
from nemo_gym.token_id_capture.sink import (
    UNRESOLVED_PARENT_REASON,
    CaptureContext,
    reset_token_sink,
    resolve_parent,
    set_token_sink,
)
from nemo_gym.token_id_capture.staging.digest import EMPTY_EXTRAS_DIGEST
from nemo_gym.token_id_capture.staging.records import RolloutManifest


USER_1 = {"role": "user", "content": "solve the task"}
ASSISTANT_1 = {"role": "assistant", "content": "first answer"}
USER_2 = {"role": "user", "content": "tool result"}
ASSISTANT_SEEDED = {"role": "assistant", "content": "seeded turn nobody served"}

TOKENS_1 = list(range(900))
STAGING_DIGEST = "a" * 64


def _custody(model_call_id: str, *, parent_call_id: str | None = None, prev_len: int = 0) -> dict:
    delta_len = len(TOKENS_1) - prev_len
    return dict(
        parent_call_id=parent_call_id,
        staging_key=f"r1/{model_call_id}",
        weight_version=17,
        prev_len=prev_len,
        delta_len=delta_len,
        cum_len=prev_len + delta_len,
        staging_digest=STAGING_DIGEST,
        extras_digest=EMPTY_EXTRAS_DIGEST,
        mode="text" if parent_call_id is None else "token_in",
        logical_request_id=f"lr-{model_call_id}",
    )


async def _record_call_1(store, rollout_id: str = "r1") -> None:
    await store.record(
        rollout_id,
        "c1",
        [USER_1],
        [ASSISTANT_1],
        TOKENS_1,
        compute_digest(TOKENS_1),
        **_custody("c1"),
    )


@pytest.fixture(params=["file", "memory"])
def store(request, tmp_path):
    if request.param == "file":
        return FileLineageStore(tmp_path)
    return InMemoryLineageStore()


@pytest.mark.asyncio
async def test_ledger_row_round_trips_token_free_manifest(store):
    await _record_call_1(store)
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert manifest.rollout_id == "r1"
    assert manifest.failures == []
    (record,) = manifest.records
    assert record.model_call_id == "c1"
    assert record.staging_key == "r1/c1"
    assert record.logical_request_id == "lr-c1"
    assert record.digest == STAGING_DIGEST
    # Cumulative token IDs stay off the manifest surface.
    assert "cumulative_token_ids" not in manifest.model_dump()["records"][0]


@pytest.mark.asyncio
async def test_same_call_commit_is_idempotent_and_conflicts_raise(store):
    await _record_call_1(store)
    await _record_call_1(store)  # identical replay is a no-op
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert len(manifest.records) == 1
    with pytest.raises(ValueError, match="conflicting"):
        await store.record(
            "r1",
            "c1",
            [USER_1],
            [ASSISTANT_1],
            TOKENS_1 + [1],
            compute_digest(TOKENS_1 + [1]),
            **_custody("c1"),
        )


@pytest.mark.asyncio
async def test_failure_rows_poison_and_never_resolve(store):
    await _record_call_1(store)
    await store.record_failure("r1", "c2", "worker_capture_failed")
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert [failure.reason for failure in manifest.failures] == ["worker_capture_failed"]
    # The committed parent still resolves; the failure row is invisible to lineage.
    match = await store.resolve("r1", [USER_1, ASSISTANT_1, USER_2])
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
async def test_admission_match_is_token_in_with_exact_prefix(store):
    await _record_call_1(store)
    context = await _admit(store, [USER_1, ASSISTANT_1, USER_2])
    admission = context.capture_admission
    assert admission is not None and admission.mode == "token_in"
    assert admission.parent_call_id == "c1"
    assert admission.required_prefix_token_ids == TOKENS_1
    assert context.request_items == [USER_1, ASSISTANT_1, USER_2]


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
    """Two committed calls with identical text must never resolve; the call poisons."""
    await _record_call_1(store)
    await store.record(
        "r1",
        "c1b",
        [USER_1],
        [ASSISTANT_1],
        TOKENS_1,
        compute_digest(TOKENS_1),
        **_custody("c1b"),
    )
    context = await _admit(store, [USER_1, ASSISTANT_1, USER_2])
    assert context.capture_admission is None
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert [failure.reason for failure in manifest.failures] == [UNRESOLVED_PARENT_REASON]


@pytest.mark.asyncio
async def test_commit_ordering_parent_resolvable_only_after_record(store):
    # Before the ledger row exists, the follow-up cannot resolve a parent.
    assert await store.resolve("r1", [USER_1, ASSISTANT_1, USER_2]) is None
    await _record_call_1(store)
    match = await store.resolve("r1", [USER_1, ASSISTANT_1, USER_2])
    assert match is not None and list(match.cumulative_token_ids) == TOKENS_1


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
async def test_legacy_lineage_rows_do_not_enter_the_manifest(store):
    # Local-capture record: no custody columns.
    await store.record("r1", "c1", [USER_1], [ASSISTANT_1], TOKENS_1, compute_digest(TOKENS_1))
    manifest = RolloutManifest.model_validate(await store.manifest("r1"))
    assert manifest.records == [] and manifest.failures == []
