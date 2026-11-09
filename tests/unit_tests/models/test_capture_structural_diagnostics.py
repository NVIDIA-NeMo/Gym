# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structural evidence must explain real rejection without changing custody."""

import json

import pytest

from nemo_gym.models.token_capture import diagnostics
from nemo_gym.models.token_capture.external_capture import VLLMWorkerCaptureHandler, publish_deferred_commit
from nemo_gym.models.token_capture.lineage import FileLineageStore
from nemo_gym.models.token_capture.sink import CaptureContext, reset_token_sink, resolve_parent, set_token_sink
from nemo_gym.models.token_capture.staging.records import CommitCoords


def _rows(path):
    return [
        json.loads(line) for file in path.glob("capture-structure-*.jsonl") for line in file.read_text().splitlines()
    ]


async def _commit(store, call_id, request, output, *, deferred=False):
    context = CaptureContext(
        capture_key="capture",
        model_call_id=call_id,
        token_sink=None,
        lineage_store=store,
        external_staging=True,
        publish_on_delivery=deferred,
    )
    token = set_token_sink(context)
    try:
        await resolve_parent(request)
        admission = context.capture_admission
        assert admission is not None
        coords = CommitCoords(
            capture_key="capture",
            model_call_id=call_id,
            parent_call_id=admission.parent_call_id,
            prev_len=admission.prev_len,
            delta_len=3,
            cum_len=admission.prev_len + 3,
            weight_version=0,
            digest="1" * 64,
            extras_digest="2" * 64,
            staging_key=f"capture/{call_id}",
            chain_hash="3" * 64,
            cumulative_hash=("4" if call_id == "ancestor" else "5") * 64,
        )
        handler = VLLMWorkerCaptureHandler()
        handler.prepare_response({"ng_commit_coords": coords.model_dump(mode="json")})
        await handler.finalize_response(
            {
                "id": f"response-{call_id}",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens", "private": "SECRET"},
                "output": output,
                "headers": {"authorization": "SECRET"},
            }
        )
        return context
    finally:
        reset_token_sink(token)


@pytest.mark.parametrize("mode", ["enabled", "disabled", "write-failure"])
async def test_actual_commits_join_ambiguous_rejection_without_changing_decisions(tmp_path, monkeypatch, mode, caplog):
    destination = tmp_path / "diagnostics"
    if mode == "disabled":
        monkeypatch.delenv("NEMO_GYM_CAPTURE_DIAGNOSTICS_DIR", raising=False)
    else:
        monkeypatch.setenv("NEMO_GYM_CAPTURE_DIAGNOSTICS_DIR", str(destination))
    if mode == "write-failure":
        destination.write_text("not a directory")
    store = FileLineageStore(tmp_path / "ledger")
    prompt = [{"role": "user", "content": "SECRET"} for _ in range(21)]
    tool = {"type": "function_call", "id": "item-tool", "call_id": "tool-a", "name": "shell", "arguments": "SECRET"}
    ancestor = await _commit(store, "ancestor", prompt, [tool])
    assert ancestor.committed
    continuation = prompt + [
        tool,
        {"type": "function_call_output", "call_id": "tool-a", "output": "SECRET"},
        {"role": "user", "content": "SECRET"},
    ]
    reasoning = {"type": "reasoning", "id": "reasoning-b", "summary": [], "encrypted_content": "SECRET"}
    descendant = await _commit(store, "descendant", continuation, [reasoning])
    assert descendant.committed and descendant.parent_call_id == "ancestor"
    rejected = CaptureContext(
        capture_key="capture",
        model_call_id="rejected",
        token_sink=None,
        lineage_store=store,
        external_staging=True,
        diagnostic_previous_response_id="response-descendant",
    )
    token = set_token_sink(rejected)
    try:
        await resolve_parent(continuation + [reasoning])
    finally:
        reset_token_sink(token)
    assert rejected.capture_admission is None
    assert rejected.parent_resolution.reason == "ambiguous"
    manifest = await store.manifest("capture")
    assert len(manifest["records"]) == 2
    if mode == "enabled":
        rows = _rows(destination)
        assert [r["event"] for r in rows] == ["commit", "commit", "rejection"]
        commits = {r["model_call_id"]: r for r in rows[:2]}
        rejection = rows[-1]
        assert rejection["resolution"]["branch"] == "fingerprint"
        candidates = rejection["resolution"]["candidates"]
        assert {c["model_call_id"] for c in candidates} == set(commits)
        assert [c["context_len"] for c in candidates] == [21, 24]
        assert [c["cum_len"] for c in candidates] == [3, 6]
        assert commits["descendant"]["response"]["output"]["items"][0]["id"] == "reasoning-b"
        assert rejection["request"]["items"][-1]["id"] == "reasoning-b"
        assert rejection["previous_response_id"] == commits["descendant"]["response"]["response_id"]
        assert commits["descendant"]["response"]["incomplete_reason"] == "max_output_tokens"
        assert "SECRET" not in json.dumps(rows)
    elif mode == "disabled":
        assert not destination.exists()
    else:
        assert "diagnostic write failed" in caplog.text
    await store.close()


async def test_deferred_commit_diagnostic_waits_for_publication(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_CAPTURE_DIAGNOSTICS_DIR", str(tmp_path / "diag"))
    store = FileLineageStore(tmp_path / "ledger")
    context = await _commit(
        store,
        "ancestor",
        [{"role": "user", "content": "SECRET"}],
        [{"type": "message", "id": "item", "role": "assistant", "content": "SECRET"}],
        deferred=True,
    )
    assert not context.committed and _rows(tmp_path / "diag") == []
    await publish_deferred_commit(context)
    assert context.committed and len(_rows(tmp_path / "diag")) == 1
    await store.close()


def test_bounds_and_missing_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_CAPTURE_DIAGNOSTICS_DIR", str(tmp_path))
    context = CaptureContext(capture_key="capture", model_call_id="call", token_sink=None)
    diagnostics.emit(context, event="rejection")
    assert _rows(tmp_path)[0]["resolution"] is None
    item = {key: "\U00010000" * 1000 for key in ("type", "id", "role", "status", "call_id", "name")}
    context.request_items = [item] * 100
    diagnostics.emit(context, event="rejection")
    row = _rows(tmp_path)[1]
    assert row["truncated"] and row["truncation_reason"] == "record_byte_limit"
    for file in tmp_path.glob("*.jsonl"):
        assert all(len(line) + 1 <= diagnostics.MAX_BYTES for line in file.read_bytes().splitlines())
    context.request_items = [{"type": "message", "id": "x" * 300, "content": "SECRET"}] * 100
    diagnostics.emit(context, event="rejection")
    request = _rows(tmp_path)[2]["request"]
    assert request["truncated"] and request["count"] == 100 and len(request["items"]) == 32
    assert len(request["items"][0]["id"]) == 128
    assert "SECRET" not in json.dumps(request)


async def test_responses_request_boundary_retains_echo_before_resolution(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from nemo_gym.core.responses_api_types import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.models import base_responses_api_model as module

    context = CaptureContext(capture_key="capture", model_call_id="call", token_sink=None)

    async def resolve(items):
        assert context.diagnostic_previous_response_id == "response-from-harness"
        assert len(items) == 1 and items[0]["role"] == "user" and items[0]["content"] == "continue"

    async def responses(body):
        return {"id": "served", "output": []}

    monkeypatch.setattr(module, "resolve_parent", resolve)
    monkeypatch.setattr(module, "register_call_intent", AsyncMock())
    monkeypatch.setattr(module, "capture_tokens", AsyncMock())
    model = SimpleNamespace(_validate_capture_request=lambda params: None, responses=responses)
    params = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "continue"}], previous_response_id="response-from-harness"
    )
    token = set_token_sink(context)
    try:
        await module.SimpleResponsesAPIModel._invoke_responses(model, None, params)
    finally:
        reset_token_sink(token)


def test_actual_candidate_list_is_bounded_without_selecting_a_parent(tmp_path, monkeypatch):
    from nemo_gym.models.token_capture.lineage import RolloutLineage

    monkeypatch.setenv("NEMO_GYM_CAPTURE_DIAGNOSTICS_DIR", str(tmp_path))
    request = [{"role": "user", "content": "SECRET"}, {"role": "assistant", "content": "SECRET2"}]
    lineage = RolloutLineage()
    for index in range(40):
        lineage.record(str(index), request, [], str(index), context_len=1, cum_len=index + 1)
    context = CaptureContext(capture_key="capture", model_call_id="call", token_sink=None)
    token = set_token_sink(context)
    try:
        resolution = lineage.resolve(request)
    finally:
        reset_token_sink(token)
    assert resolution.reason == "ambiguous" and resolution.match is None
    assert context.diagnostic_resolution["candidate_count"] == 40
    assert len(context.diagnostic_resolution["candidates"]) == 32
    assert context.diagnostic_resolution["truncated"]
