# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import shutil

import pytest

from nemo_gym.token_id_capture import lineage as module
from nemo_gym.token_id_capture.fingerprint import FINGERPRINT_VERSION
from nemo_gym.token_id_capture.staging.digest import EMPTY_EXTRAS_DIGEST, compute_chain_hash, hash_token_ids
from nemo_gym.token_id_capture.staging.records import CallRecord, CaptureLedgerCommit


def commit(call_id, request, output, tokens):
    record = CallRecord(
        model_call_id=call_id,
        staging_key=f"r/{call_id}",
        weight_version=7,
        prev_len=0,
        delta_len=len(tokens),
        cum_len=len(tokens),
        digest="a" * 64,
        extras_digest=EMPTY_EXTRAS_DIGEST,
        mode="text",
        chain_hash=compute_chain_hash(None, tokens),
        cumulative_hash=hash_token_ids(tokens),
        response_id=f"response-{call_id}",
        fingerprint_version=FINGERPRINT_VERSION,
    )
    return CaptureLedgerCommit(
        rollout_id="r", record=record, request_items=request, response_items=output, staging_chain=(f"r/{call_id}",)
    )


def scenario():
    request = [{"role": "user", "content": "Synthetic task"}]
    first_output = [{"role": "assistant", "content": "Working"}]
    second = request + first_output + [{"role": "user", "content": "Continue"}]
    reasoning = [
        {
            "type": "reasoning",
            "id": "rs_unique",
            "summary": [{"type": "summary_text", "text": "Plan"}],
            "content": None,
            "encrypted_content": None,
        }
    ]
    echo = [{k: v for k, v in x.items() if v is not None} for x in reasoning]
    third = second + echo + [{"role": "user", "content": "Again"}]
    return commit("a", request, first_output, [1, 2]), commit("b", second, reasoning, [1, 2, 3]), third


@pytest.mark.asyncio
async def test_file_reload_checkpoint_copy_and_idempotent_replay(tmp_path):
    writer = module.FileLineageStore(tmp_path / "live")
    a, b, request = scenario()
    await writer.record(a)
    await writer.record(b)
    for reader in [writer, module.FileLineageStore(tmp_path / "live")]:
        result = await reader.resolve("r", request)
        assert result.reason == "reasoning_output_witness" and result.match.model_call_id == "b"
        assert result.match.prev_len == 3 and result.match.digest == hash_token_ids([1, 2, 3])
        await reader.record(b)
    shutil.copytree(tmp_path / "live", tmp_path / "checkpoint_restored")
    restored = module.FileLineageStore(tmp_path / "checkpoint_restored")
    result = await restored.resolve("r", request)
    assert result.match.model_call_id == "b" and result.match.staging_chain == ("r/b",)
    assert not (await restored.manifest("r"))["failures"]
    exported = writer.export_rows("r")
    imported = module.FileLineageStore(tmp_path / "checkpoint_imported")
    imported.import_rows("r", exported)
    result = await imported.resolve("r", request)
    assert result.reason == "reasoning_output_witness" and result.match.model_call_id == "b"
    assert imported.export_rows("r")[1]["reasoning_witness"] == exported[1]["reasoning_witness"]


@pytest.mark.asyncio
async def test_legacy_rows_and_corrupt_witness_abstain(tmp_path):
    a, b, request = scenario()
    for variant in ["absent", "corrupt", "wrong_call"]:
        root = tmp_path / variant
        writer = module.FileLineageStore(root)
        await writer.record(a)
        await writer.record(b)
        path = writer._ledger_path("r")
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        if variant == "absent":
            rows[1].pop("reasoning_witness")
        elif variant == "corrupt":
            rows[1]["reasoning_witness"]["context_len"] = "wrong"
        else:
            rows[1]["reasoning_witness"]["call_id"] = "other"
        path.write_text("".join(json.dumps(x) + "\n" for x in rows))
        reader = module.FileLineageStore(root)
        result = await reader.resolve("r", request)
        assert result.status.value == "unresolved" and result.reason == "ambiguous"
        if variant == "absent":
            await reader.record(b)
            assert "reasoning_witness" not in json.loads(path.read_text().splitlines()[1])


@pytest.mark.asyncio
async def test_modified_prior_reasoning_cannot_select_parent(tmp_path):
    a, b, request = scenario()
    a.request_items.insert(
        0, {"type": "reasoning", "id": "prior", "summary": [{"type": "summary_text", "text": "Original"}]}
    )
    b.request_items.insert(0, a.request_items[0])
    request.insert(0, {"type": "reasoning", "id": "prior", "summary": [{"type": "summary_text", "text": "Changed"}]})
    writer = module.FileLineageStore(tmp_path)
    await writer.record(a)
    await writer.record(b)
    assert (await writer.resolve("r", request)).status.value == "unresolved"


@pytest.fixture(params=["memory", "file"])
def initial_store(request, tmp_path):
    if request.param == "memory":
        return module.InMemoryLineageStore()
    return module.FileLineageStore(tmp_path / request.param)


async def admit_initial(store, request, call_id="next"):
    from nemo_gym.token_id_capture.sink import CaptureContext, reset_token_sink, resolve_parent, set_token_sink

    context = CaptureContext(
        rollout_id="r", model_call_id=call_id, token_sink=None, lineage_store=store, external_staging=True
    )
    token = set_token_sink(context)
    try:
        await resolve_parent(request)
    finally:
        reset_token_sink(token)
    return context


def initial_reasoning():
    return [{"type": "reasoning", "id": "rs_initial", "summary": [{"type": "summary_text", "text": "Plan"}]}]


@pytest.mark.asyncio
async def test_initial_reasoning_echo_reuses_actual_parent_prefix(initial_store, tmp_path):
    request = [{"role": "user", "content": "Task"}]
    output = initial_reasoning()
    await initial_store.record(commit("initial", request, output, [1, 2, 3]))
    # Exercise actual checkpoint row export/import and a fresh reader, not just
    # an in-memory witness that would disappear on recovery.
    if isinstance(initial_store, module.FileLineageStore):
        restored = module.FileLineageStore(tmp_path / "imported")
        restored.import_rows("r", initial_store.export_rows("r"))
        initial_store = module.FileLineageStore(tmp_path / "imported")
    context = await admit_initial(initial_store, request + output + [{"role": "user", "content": "Continue"}])
    assert context.parent_resolution.reason == "reasoning_output_witness"
    admission = context.capture_admission
    assert admission.mode == "token_in" and admission.parent_call_id == "initial"
    assert admission.prev_len == 3 and admission.staging_chain == ["r/initial"]
    assert admission.parent_chain_hash == compute_chain_hash(None, [1, 2, 3])
    assert not (await initial_store.manifest("r"))["failures"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["context", "id", "summary", "missing_id", "interleaved", "duplicate", "legacy"])
async def test_initial_reasoning_mutation_or_ambiguity_poison_admission(initial_store, mutation):
    from copy import deepcopy

    request = [{"role": "user", "content": "Task"}]
    output = initial_reasoning()
    await initial_store.record(commit("initial", request, output, [1, 2, 3]))
    continuation = deepcopy(request + output)
    if mutation == "context":
        continuation[0]["content"] = "Changed task"
    elif mutation == "id":
        continuation[1]["id"] = "rs_other"
    elif mutation == "summary":
        continuation[1]["summary"][0]["text"] = "Changed reasoning"
    elif mutation == "missing_id":
        continuation[1].pop("id")
    elif mutation == "interleaved":
        continuation.insert(1, {"role": "user", "content": "Inserted"})
    elif mutation == "duplicate":
        await initial_store.record(commit("duplicate", request, output, [1, 2, 4]))
    else:
        if isinstance(initial_store, module.FileLineageStore):
            path = initial_store._ledger_path("r")
            row = json.loads(path.read_text())
            row.pop("reasoning_witness")
            path.write_text(json.dumps(row) + "\n")
            initial_store = module.FileLineageStore(initial_store._ledger_root)
        else:
            initial_store.index.for_rollout("r").by_call_id["initial"].reasoning_witness = None
    context = await admit_initial(initial_store, continuation)
    assert context.parent_resolution.status.value == "unresolved"
    assert context.capture_admission is None
    failures = (await initial_store.manifest("r"))["failures"]
    assert len(failures) == 1 and failures[0]["model_call_id"] == "next"


@pytest.mark.asyncio
async def test_independent_child_root_and_reasoning_chain_stay_separate(initial_store):
    root = [{"role": "user", "content": "Root task"}]
    child = [{"role": "user", "content": "Child task"}]
    root_output = initial_reasoning()
    child_output = [{"type": "reasoning", "id": "rs_child", "summary": []}]
    await initial_store.record(commit("root", root, root_output, [1, 2, 3]))
    child_context = await admit_initial(initial_store, child, "child")
    assert child_context.parent_resolution.status.value == "root"
    assert child_context.capture_admission.mode == "text"
    await initial_store.record(commit("child", child, child_output, [4, 5]))
    root_next = await admit_initial(initial_store, root + root_output, "root-next")
    child_next = await admit_initial(initial_store, child + child_output, "child-next")
    assert root_next.capture_admission.parent_call_id == "root"
    assert child_next.capture_admission.parent_call_id == "child"
    assert not (await initial_store.manifest("r"))["failures"]


@pytest.mark.asyncio
async def test_consecutive_initial_reasoning_uses_latest_exact_context(initial_store):
    request = [{"role": "user", "content": "Task"}]
    first = initial_reasoning()
    await initial_store.record(commit("first", request, first, [1, 2]))
    second_request = request + first + [{"role": "user", "content": "Continue"}]
    second = [{"type": "reasoning", "id": "rs_second", "summary": []}]
    await initial_store.record(commit("second", second_request, second, [1, 2, 3]))
    # A different server-issued output ID cannot compete with the echoed one.
    alternative = [{"type": "reasoning", "id": "rs_alternative", "summary": []}]
    await initial_store.record(commit("alternative", second_request, alternative, [1, 2, 4]))
    context = await admit_initial(initial_store, second_request + second)
    assert context.capture_admission.parent_call_id == "second"
    assert context.capture_admission.prev_len == 3
