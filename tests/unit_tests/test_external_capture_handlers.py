# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External capture strategy lifecycle tests."""

from typing import Any

import pytest

from nemo_gym.token_id_capture.external_capture import (
    MegatronLedgerCaptureHandler,
    VLLMWorkerCaptureHandler,
    make_external_capture_handler,
)
from nemo_gym.token_id_capture.lineage import InMemoryLineageStore
from nemo_gym.token_id_capture.sink import CaptureContext, reset_token_sink, set_token_sink
from nemo_gym.token_id_capture.staging.records import CaptureAdmission, CommitCoords


def _root_context(store: InMemoryLineageStore) -> CaptureContext:
    return CaptureContext(
        rollout_id="rollout-1",
        model_call_id="c1",
        token_sink=None,
        lineage_store=store,
        external_staging=True,
        request_items=[{"role": "user", "content": "go"}],
        capture_admission=CaptureAdmission(
            rollout_id="rollout-1",
            model_call_id="c1",
            mode="text",
        ),
    )


def _transport_payload(**fields: Any) -> dict[str, Any]:
    message = {
        "role": "assistant",
        "content": "done",
        "prompt_token_ids": [10, 11],
        "generation_token_ids": [12],
        "generation_log_probs": [-0.2],
        "routed_experts": {"data": "unused"},
    }
    message.update(fields)
    return {
        "id": "request-1",
        "prompt_token_ids": [10, 11],
        "choices": [
            {
                "token_ids": [12],
                "logprobs": {"content": []},
                "message": message,
            }
        ],
    }


def _assert_transport_fields_stripped(payload: dict[str, Any]) -> None:
    assert "ng_commit_coords" not in payload
    assert "prompt_token_ids" not in payload
    choice = payload["choices"][0]
    assert "token_ids" not in choice
    assert "logprobs" not in choice
    message = choice["message"]
    assert "prompt_token_ids" not in message
    assert "generation_token_ids" not in message
    assert "generation_log_probs" not in message
    assert "routed_experts" not in message


def test_vllm_worker_handler_preserves_worker_staged_request_contract() -> None:
    store = InMemoryLineageStore()
    context = _root_context(store)
    token = set_token_sink(context)
    try:
        payload = VLLMWorkerCaptureHandler().prepare_request({})
    finally:
        reset_token_sink(token)

    assert payload["ng_capture"] == context.capture_admission.model_dump(mode="json")
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 0
    assert payload["return_tokens_as_token_ids"] is True
    assert "return_tokenized_data" not in payload


@pytest.mark.asyncio
async def test_vllm_worker_handler_commits_valid_staged_coordinates() -> None:
    store = InMemoryLineageStore()
    context = _root_context(store)
    payload = _transport_payload()
    payload["ng_commit_coords"] = CommitCoords(
        rollout_id="rollout-1",
        model_call_id="c1",
        prev_len=0,
        delta_len=3,
        cum_len=3,
        weight_version=7,
        digest="0" * 64,
        extras_digest="1" * 64,
        staging_key="r0/c1",
        chain_hash="2" * 64,
        cumulative_hash="3" * 64,
    ).model_dump(mode="json")
    token = set_token_sink(context)
    try:
        await VLLMWorkerCaptureHandler().finalize_response(payload)
    finally:
        reset_token_sink(token)

    manifest = await store.manifest("rollout-1")
    assert manifest["records"][0]["staging_key"] == "r0/c1"
    assert manifest["records"][0]["weight_version"] == 7
    assert manifest["records"][0]["chain_hash"] == "2" * 64
    assert manifest["records"][0]["cumulative_hash"] == "3" * 64
    assert manifest["records"][0]["response_id"] == "request-1"
    assert context.committed is True
    _assert_transport_fields_stripped(payload)


@pytest.mark.asyncio
async def test_vllm_worker_handler_poison_and_cleanup_survive_failed_capture() -> None:
    store = InMemoryLineageStore()
    context = _root_context(store)
    payload = _transport_payload()
    payload["ng_commit_coords"] = CommitCoords(
        rollout_id="rollout-1",
        model_call_id="c1",
        prev_len=0,
        delta_len=0,
        cum_len=0,
        weight_version=7,
        disposition="capture_failed",
    ).model_dump(mode="json")
    token = set_token_sink(context)
    try:
        await VLLMWorkerCaptureHandler().finalize_response(payload)
    finally:
        reset_token_sink(token)

    manifest = await store.manifest("rollout-1")
    assert manifest["failures"] == [
        {
            "schema_version": 2,
            "model_call_id": "c1",
            "reason": "worker_capture_failed",
        }
    ]
    assert context.committed is False
    _assert_transport_fields_stripped(payload)


@pytest.mark.asyncio
async def test_megatron_handler_poisons_invalid_reference_and_always_cleans_transport() -> None:
    store = InMemoryLineageStore()
    context = _root_context(store)
    payload = _transport_payload()
    payload.pop("id")
    token = set_token_sink(context)
    try:
        await MegatronLedgerCaptureHandler().finalize_response(payload)
    finally:
        reset_token_sink(token)

    manifest = await store.manifest("rollout-1")
    assert manifest["failures"][0]["reason"] == "invalid_megatron_ledger_reference"
    assert context.committed is False
    _assert_transport_fields_stripped(payload)


@pytest.mark.asyncio
async def test_handlers_strip_unadmitted_capture_responses() -> None:
    store = InMemoryLineageStore()
    for handler in (VLLMWorkerCaptureHandler(), MegatronLedgerCaptureHandler()):
        context = _root_context(store)
        context.capture_admission = None
        payload = _transport_payload()
        payload["ng_commit_coords"] = {"unused": True}
        token = set_token_sink(context)
        try:
            await handler.finalize_response(payload)
        finally:
            reset_token_sink(token)
        _assert_transport_fields_stripped(payload)


@pytest.mark.asyncio
async def test_handlers_leave_uncorrelated_traffic_untouched() -> None:
    for handler in (VLLMWorkerCaptureHandler(), MegatronLedgerCaptureHandler()):
        payload = _transport_payload()
        await handler.finalize_response(payload)
        assert payload["prompt_token_ids"] == [10, 11]
        assert payload["choices"][0]["message"]["generation_token_ids"] == [12]


def test_factory_selects_the_typed_backend_strategy() -> None:
    assert isinstance(make_external_capture_handler("vllm_worker"), VLLMWorkerCaptureHandler)
    assert isinstance(make_external_capture_handler("megatron_ledger"), MegatronLedgerCaptureHandler)
