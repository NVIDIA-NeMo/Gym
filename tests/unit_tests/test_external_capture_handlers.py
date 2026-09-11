# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External capture strategy lifecycle tests."""

from typing import Any

import pytest

from nemo_gym.token_id_capture.external_capture import (
    MegatronWorkerCaptureHandler,
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
        # Megatron ``return_tokenized_data`` echo, absent from vLLM payloads.
        "compact_prompt_token_ids": [10, 11],
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
    assert "compact_prompt_token_ids" not in message


@pytest.mark.parametrize(
    ("handler", "request_payload", "metadata_field", "token_return_field"),
    [
        (VLLMWorkerCaptureHandler(), {}, None, "return_tokens_as_token_ids"),
        (MegatronWorkerCaptureHandler(), {}, "request_metadata", "return_tokenized_data"),
        (
            MegatronWorkerCaptureHandler(),
            {"request_metadata": {"caller_metadata": "preserved"}},
            "request_metadata",
            "return_tokenized_data",
        ),
    ],
    ids=["vllm", "megatron", "megatron-existing-metadata"],
)
def test_handler_prepares_worker_staged_request(
    handler, request_payload: dict[str, Any], metadata_field, token_return_field
) -> None:
    store = InMemoryLineageStore()
    context = _root_context(store)
    token = set_token_sink(context)
    try:
        payload = handler.prepare_request(request_payload)
    finally:
        reset_token_sink(token)

    capture_container = payload if metadata_field is None else payload[metadata_field]
    assert capture_container["ng_capture"] == context.capture_admission.model_dump(mode="json")
    if metadata_field is not None:
        assert capture_container == {
            **request_payload.get(metadata_field, {}),
            "ng_capture": context.capture_admission.model_dump(mode="json"),
        }
        assert "ng_capture" not in payload
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 0
    assert payload[token_return_field] is True
    other_token_return_field = (
        "return_tokenized_data" if token_return_field == "return_tokens_as_token_ids" else "return_tokens_as_token_ids"
    )
    assert other_token_return_field not in payload


@pytest.mark.parametrize(
    ("request_payload", "error"),
    [
        ({"n": 2}, "requires n=1"),
        ({"request_metadata": []}, "request_metadata must be an object"),
    ],
)
def test_megatron_handler_rejects_invalid_request_contract(request_payload: dict[str, Any], error: str) -> None:
    store = InMemoryLineageStore()
    context = _root_context(store)
    token = set_token_sink(context)
    try:
        with pytest.raises(ValueError, match=error):
            MegatronWorkerCaptureHandler().prepare_request(request_payload)
    finally:
        reset_token_sink(token)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "coords_kwargs", "drop_response_id", "expected_failure"),
    [
        (
            VLLMWorkerCaptureHandler(),
            {
                "delta_len": 3,
                "cum_len": 3,
                "digest": "0" * 64,
                "extras_digest": "1" * 64,
                "staging_key": "r0/c1",
                "chain_hash": "2" * 64,
                "cumulative_hash": "3" * 64,
            },
            False,
            None,
        ),
        (
            VLLMWorkerCaptureHandler(),
            {"delta_len": 0, "cum_len": 0, "disposition": "capture_failed"},
            False,
            "worker_capture_failed",
        ),
        (
            MegatronWorkerCaptureHandler(),
            None,
            True,
            "worker_response_missing_commit_coordinates",
        ),
    ],
    ids=["vllm-staged", "vllm-capture-failed", "megatron-missing-coordinates"],
)
async def test_handler_finalization_updates_lineage_and_cleans_transport(
    handler, coords_kwargs, drop_response_id, expected_failure
) -> None:
    store = InMemoryLineageStore()
    context = _root_context(store)
    payload = _transport_payload()
    if coords_kwargs is not None:
        payload["ng_commit_coords"] = CommitCoords(
            rollout_id="rollout-1",
            model_call_id="c1",
            prev_len=0,
            weight_version=7,
            **coords_kwargs,
        ).model_dump(mode="json")
    if drop_response_id:
        payload.pop("id")
    token = set_token_sink(context)
    try:
        await handler.finalize_response(payload)
    finally:
        reset_token_sink(token)

    manifest = await store.manifest("rollout-1")
    assert context.committed is (expected_failure is None)
    if expected_failure is None:
        record = manifest["records"][0]
        assert record["staging_key"] == "r0/c1"
        assert record["weight_version"] == 7
        assert record["chain_hash"] == "2" * 64
        assert record["cumulative_hash"] == "3" * 64
        assert record["response_id"] == "request-1"
    else:
        assert manifest["failures"] == [
            {
                "schema_version": 2,
                "model_call_id": "c1",
                "reason": expected_failure,
            }
        ]
    _assert_transport_fields_stripped(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler",
    [VLLMWorkerCaptureHandler(), MegatronWorkerCaptureHandler()],
    ids=["vllm", "megatron"],
)
async def test_handlers_strip_unadmitted_capture_responses(handler) -> None:
    store = InMemoryLineageStore()
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
@pytest.mark.parametrize(
    "handler",
    [VLLMWorkerCaptureHandler(), MegatronWorkerCaptureHandler()],
    ids=["vllm", "megatron"],
)
async def test_handlers_leave_uncorrelated_traffic_untouched(handler) -> None:
    payload = _transport_payload()
    await handler.finalize_response(payload)
    assert payload["prompt_token_ids"] == [10, 11]
    assert payload["choices"][0]["message"]["generation_token_ids"] == [12]


@pytest.mark.parametrize(
    ("backend", "handler_type"),
    [("vllm_worker", VLLMWorkerCaptureHandler), ("megatron_worker", MegatronWorkerCaptureHandler)],
)
def test_factory_selects_the_typed_backend_strategy(backend, handler_type) -> None:
    assert isinstance(make_external_capture_handler(backend), handler_type)
