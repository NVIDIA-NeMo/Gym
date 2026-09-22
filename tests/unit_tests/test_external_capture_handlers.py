# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External capture strategy lifecycle tests."""

import logging
from dataclasses import dataclass
from typing import Any

import pytest

from nemo_gym.token_id_capture.external_capture import (
    MegatronWorkerCaptureHandler,
    VLLMWorkerCaptureHandler,
    make_external_capture_handler,
)
from nemo_gym.token_id_capture.lineage import InMemoryLineageStore
from nemo_gym.token_id_capture.sink import CaptureContext, reset_token_sink, set_token_sink
from nemo_gym.token_id_capture.staging.records import (
    INVALID_COMMIT_COORDS_REASON,
    WORKER_CAPTURE_FAILED_REASON,
    WORKER_MISSING_COMMIT_COORDS_REASON,
    CaptureAdmission,
    CaptureLedgerCommit,
    CommitCoords,
)


HANDLER_CLASSES = pytest.mark.parametrize(
    "handler_cls",
    [VLLMWorkerCaptureHandler, MegatronWorkerCaptureHandler],
    ids=["vllm", "megatron"],
)


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
        # Megatron prompt-form echo, absent from vLLM payloads; stripped defensively.
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
        (MegatronWorkerCaptureHandler(), {}, "offload_params", None),
        (
            MegatronWorkerCaptureHandler(),
            {"offload_params": {"caller_metadata": "preserved"}},
            "offload_params",
            None,
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
    if token_return_field is None:
        # Megatron stages the delta on the worker; Gym asks for no token echo.
        assert "return_tokenized_data" not in payload
        assert "return_tokens_as_token_ids" not in payload
    else:
        assert payload[token_return_field] is True
        assert "return_tokenized_data" not in payload


@pytest.mark.parametrize(
    ("request_payload", "error"),
    [
        ({"n": 2}, "requires n=1"),
        ({"offload_params": []}, "offload_params must be an object"),
        (
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:,"}}]}]},
            "does not support multimodal content",
        ),
        (
            {"messages": [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "data:,"}}]}]},
            "does not support multimodal content",
        ),
        (
            {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": ""}}]}]},
            "does not support multimodal content",
        ),
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


def _staged_coords(**overrides: Any) -> dict[str, Any]:
    """Return a valid ``staged`` acknowledgement for the ``_root_context`` call, with overrides."""
    kwargs: dict[str, Any] = {
        "rollout_id": "rollout-1",
        "model_call_id": "c1",
        "prev_len": 0,
        "weight_version": 7,
        "delta_len": 3,
        "cum_len": 3,
        "digest": "0" * 64,
        "extras_digest": "1" * 64,
        "staging_key": "r0/c1",
        "chain_hash": "2" * 64,
        "cumulative_hash": "3" * 64,
    }
    kwargs.update(overrides)
    return CommitCoords(**kwargs).model_dump(mode="json")


@dataclass(frozen=True)
class _FinalizeCase:
    """One worker-acknowledgement scenario, run against every handler backend.

    ``coords_payload`` is placed verbatim on ``ng_commit_coords`` (``None``
    means the worker sent no acknowledgement at all). ``expected_failure`` is
    the poison reason the ledger must carry, or ``None`` for a committed call.
    """

    coords_payload: Any
    expected_failure: str | None
    drop_response_id: bool = False


_FINALIZE_CASES = {
    "staged": _FinalizeCase(coords_payload=_staged_coords(), expected_failure=None),
    # ``drop_response_id`` is deliberately False: finalization returns at the
    # missing-coordinates branch before the envelope id is ever inspected.
    "missing-coordinates": _FinalizeCase(
        coords_payload=None,
        expected_failure=WORKER_MISSING_COMMIT_COORDS_REASON,
    ),
    "capture-failed": _FinalizeCase(
        coords_payload=CommitCoords(
            rollout_id="rollout-1",
            model_call_id="c1",
            prev_len=0,
            weight_version=7,
            delta_len=0,
            cum_len=0,
            disposition="capture_failed",
        ).model_dump(mode="json"),
        expected_failure=WORKER_CAPTURE_FAILED_REASON,
    ),
    "malformed-not-a-dict": _FinalizeCase(
        coords_payload=["not", "a", "mapping"],
        expected_failure=INVALID_COMMIT_COORDS_REASON,
    ),
    "malformed-missing-fields": _FinalizeCase(
        coords_payload={"rollout_id": "rollout-1", "model_call_id": "c1"},
        expected_failure=INVALID_COMMIT_COORDS_REASON,
    ),
    "mismatched-rollout-id": _FinalizeCase(
        coords_payload=_staged_coords(rollout_id="rollout-other"),
        expected_failure=INVALID_COMMIT_COORDS_REASON,
    ),
    "mismatched-model-call-id": _FinalizeCase(
        coords_payload=_staged_coords(model_call_id="c9"),
        expected_failure=INVALID_COMMIT_COORDS_REASON,
    ),
    # Self-consistent child coordinates (parent set, prev_len > 0) that
    # diverge from the parentless text admission on the context.
    "mismatched-parent-and-prev-len": _FinalizeCase(
        coords_payload=_staged_coords(parent_call_id="c0", prev_len=2, cum_len=5),
        expected_failure=INVALID_COMMIT_COORDS_REASON,
    ),
    "missing-response-id": _FinalizeCase(
        coords_payload=_staged_coords(),
        expected_failure=INVALID_COMMIT_COORDS_REASON,
        drop_response_id=True,
    ),
}


def _assert_poisoned(
    manifest: dict[str, Any],
    context: CaptureContext,
    payload: dict[str, Any],
    reason: str,
) -> None:
    """Assert a call failed closed: not committed, exactly one poison row, transport scrubbed."""
    assert context.committed is False
    assert manifest["records"] == []
    assert manifest["failures"] == [
        {
            "schema_version": 2,
            "model_call_id": "c1",
            "reason": reason,
        }
    ]
    _assert_transport_fields_stripped(payload)


async def _prepare_and_finalize(handler, context: CaptureContext, payload: dict[str, Any]) -> None:
    token = set_token_sink(context)
    try:
        handler.prepare_response(payload)
        _assert_transport_fields_stripped(payload)
        await handler.finalize_response(payload)
    finally:
        reset_token_sink(token)


@pytest.mark.asyncio
@HANDLER_CLASSES
@pytest.mark.parametrize("case", list(_FINALIZE_CASES.values()), ids=list(_FINALIZE_CASES))
async def test_handler_finalization_updates_lineage_and_cleans_transport(handler_cls, case: _FinalizeCase) -> None:
    handler = handler_cls()
    store = InMemoryLineageStore()
    context = _root_context(store)
    payload = _transport_payload()
    if case.coords_payload is not None:
        payload["ng_commit_coords"] = case.coords_payload
    if case.drop_response_id:
        payload.pop("id")

    await _prepare_and_finalize(handler, context, payload)

    manifest = await store.manifest("rollout-1")
    if case.expected_failure is None:
        assert context.committed is True
        assert manifest["failures"] == []
        record = manifest["records"][0]
        assert record["staging_key"] == "r0/c1"
        assert record["weight_version"] == 7
        assert record["chain_hash"] == "2" * 64
        assert record["cumulative_hash"] == "3" * 64
        assert record["response_id"] == "request-1"
        _assert_transport_fields_stripped(payload)
    else:
        _assert_poisoned(manifest, context, payload, case.expected_failure)


class _FaultyLedger(InMemoryLineageStore):
    """Ledger whose writes can be made to raise, to exercise the poison fallback paths."""

    def __init__(self, *, record_fails: bool, record_failure_fails: bool) -> None:
        super().__init__()
        self._record_fails = record_fails
        self._record_failure_fails = record_failure_fails
        self.record_failure_calls: list[tuple[str, str, str]] = []

    async def record(self, commit: CaptureLedgerCommit) -> None:
        if self._record_fails:
            raise RuntimeError("ledger write failed")
        await super().record(commit)

    async def record_failure(self, rollout_id: str, model_call_id: str, reason: str) -> None:
        self.record_failure_calls.append((rollout_id, model_call_id, reason))
        if self._record_failure_fails:
            raise RuntimeError("ledger poison write failed")
        await super().record_failure(rollout_id, model_call_id, reason)


@pytest.mark.asyncio
@HANDLER_CLASSES
async def test_handler_poisons_call_when_ledger_record_raises(handler_cls) -> None:
    handler = handler_cls()
    store = _FaultyLedger(record_fails=True, record_failure_fails=False)
    context = _root_context(store)
    payload = _transport_payload()
    payload["ng_commit_coords"] = _staged_coords()

    await _prepare_and_finalize(handler, context, payload)

    manifest = await store.manifest("rollout-1")
    _assert_poisoned(manifest, context, payload, INVALID_COMMIT_COORDS_REASON)
    assert store.record_failure_calls == [("rollout-1", "c1", INVALID_COMMIT_COORDS_REASON)]


@pytest.mark.asyncio
@HANDLER_CLASSES
async def test_handler_logs_and_continues_when_poisoning_also_fails(handler_cls, caplog) -> None:
    handler = handler_cls()
    store = _FaultyLedger(record_fails=True, record_failure_fails=True)
    context = _root_context(store)
    payload = _transport_payload()
    payload["ng_commit_coords"] = _staged_coords()

    with caplog.at_level(logging.ERROR, logger="nemo_gym.token_id_capture.external_capture"):
        # Must return normally: a valid completion is never turned into a harness failure.
        await _prepare_and_finalize(handler, context, payload)

    assert context.committed is False
    _assert_transport_fields_stripped(payload)
    assert store.record_failure_calls == [("rollout-1", "c1", INVALID_COMMIT_COORDS_REASON)]
    manifest = await store.manifest("rollout-1")
    assert manifest["records"] == []
    assert manifest["failures"] == []
    assert (
        f"{handler._BACKEND_LABEL} worker capture acknowledgement failed for rollout rollout-1 call c1" in caplog.text
    )
    assert (
        f"Could not poison rollout rollout-1 call c1 after a failed {handler._BACKEND_LABEL} worker acknowledgement"
        in caplog.text
    )


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
        handler.prepare_response(payload)
        await handler.finalize_response(payload)
    finally:
        reset_token_sink(token)
    _assert_transport_fields_stripped(payload)
    assert context.external_commit_coords == {"unused": True}
    assert (await store.manifest("rollout-1"))["records"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler",
    [VLLMWorkerCaptureHandler(), MegatronWorkerCaptureHandler()],
    ids=["vllm", "megatron"],
)
async def test_handlers_leave_uncorrelated_traffic_untouched(handler) -> None:
    payload = _transport_payload()
    handler.prepare_response(payload)
    await handler.finalize_response(payload)
    assert payload["prompt_token_ids"] == [10, 11]
    assert payload["choices"][0]["message"]["generation_token_ids"] == [12]


@pytest.mark.parametrize(
    ("backend", "handler_type"),
    [("vllm_worker", VLLMWorkerCaptureHandler), ("megatron_worker", MegatronWorkerCaptureHandler)],
)
def test_factory_selects_the_typed_backend_strategy(backend, handler_type) -> None:
    assert isinstance(make_external_capture_handler(backend), handler_type)
