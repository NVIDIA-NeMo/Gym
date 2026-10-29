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
