# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend strategies for framework-owned token capture."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol, cast

from nemo_gym.token_id_capture.adapters.megatron import MegatronCaptureAdapter
from nemo_gym.token_id_capture.config import ExternalStagingBackend
from nemo_gym.token_id_capture.lineage import (
    LINEAGE_FINGERPRINT_VERSION,
    assistant_fingerprint,
)
from nemo_gym.token_id_capture.protocols import CaptureLedger
from nemo_gym.token_id_capture.records import (
    TOKEN_FIELDS,
    compute_digest,
    response_to_output_items,
    strip_token_fields,
)
from nemo_gym.token_id_capture.sink import (
    NG_CAPTURE_FIELD,
    NG_COMMIT_COORDS_FIELD,
    CaptureContext,
    current_capture_context,
    mark_external_ledger_capture_recorded,
    mark_external_staging_committed,
)
from nemo_gym.token_id_capture.staging.digest import compute_chain_hash, hash_token_ids
from nemo_gym.token_id_capture.staging.records import CaptureAdmission, CommitCoords


LOGGER = logging.getLogger(__name__)


class ExternalCaptureHandler(Protocol):
    """Prepare and finalize one backend-specific external capture call."""

    def prepare_request(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Attach capture instructions to an admitted engine request."""
        ...

    async def finalize_response(self, response_payload: dict[str, Any]) -> None:
        """Commit lineage and remove capture-only response fields."""
        ...


def _strip_capture_transport_fields(payload: dict[str, Any]) -> None:
    """Keep token IDs, logprobs, routes, and coordinates off the agent hop."""
    payload.pop(NG_COMMIT_COORDS_FIELD, None)
    payload.pop("prompt_token_ids", None)
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        choice.pop("logprobs", None)
        choice.pop("token_ids", None)
        message = choice.get("message")
        if isinstance(message, dict):
            for field_name in TOKEN_FIELDS:
                message.pop(field_name, None)


class _BaseExternalCaptureHandler(ABC):
    """Own the lifecycle shared by external capture backends."""

    _INVALID_CAPTURE_REASON: str
    _CAPTURE_ERROR_MESSAGE: str
    _POISON_ERROR_MESSAGE: str

    def prepare_request(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        """Attach capture instructions to an engine-bound request.

        An unadmitted call (``UNRESOLVED`` — already poisoned in the ledger)
        is forwarded as plain traffic: the backend captures nothing and the
        completion still serves the agent.
        """
        context = current_capture_context()
        if context is None or not context.external_staging:
            return request_payload
        admission = context.capture_admission
        if admission is None:
            return request_payload
        return self._prepare_admitted_request(request_payload, admission)

    @abstractmethod
    def _prepare_admitted_request(
        self,
        request_payload: dict[str, Any],
        admission: CaptureAdmission,
    ) -> dict[str, Any]:
        """Attach backend-specific fields after shared admission checks."""

    async def finalize_response(self, response_payload: dict[str, Any]) -> None:
        context = current_capture_context()
        if context is None or not context.external_staging or context.lineage_store is None:
            return
        ledger = cast(CaptureLedger, context.lineage_store)
        try:
            admission = context.capture_admission
            if admission is None:
                # UNRESOLVED — the ledger already carries this call's poison row.
                return
            try:
                await self._finalize_admitted_response(
                    response_payload,
                    context=context,
                    ledger=ledger,
                    admission=admission,
                )
            except Exception:
                # Backend/framework payloads are an external integrity boundary.
                # Poison capture without turning a valid model completion into a
                # harness failure.
                LOGGER.exception(
                    self._CAPTURE_ERROR_MESSAGE,
                    context.rollout_id,
                    context.model_call_id,
                )
                try:
                    await ledger.record_failure(
                        context.rollout_id,
                        context.model_call_id,
                        self._INVALID_CAPTURE_REASON,
                    )
                except Exception:
                    LOGGER.exception(
                        self._POISON_ERROR_MESSAGE,
                        context.rollout_id,
                        context.model_call_id,
                    )
        finally:
            _strip_capture_transport_fields(response_payload)

    @abstractmethod
    async def _finalize_admitted_response(
        self,
        response_payload: dict[str, Any],
        *,
        context: CaptureContext,
        ledger: CaptureLedger,
        admission: CaptureAdmission,
    ) -> None:
        """Publish backend-specific custody for an admitted response."""


class VLLMWorkerCaptureHandler(_BaseExternalCaptureHandler):
    """Commit lineage after a vLLM worker durably stages the token delta."""

    _INVALID_CAPTURE_REASON = "invalid_worker_commit_coordinates"
    _CAPTURE_ERROR_MESSAGE = "Worker capture acknowledgement failed for rollout %s call %s"
    _POISON_ERROR_MESSAGE = "Could not poison rollout %s call %s after a failed acknowledgement"

    def _prepare_admitted_request(
        self,
        request_payload: dict[str, Any],
        admission: CaptureAdmission,
    ) -> dict[str, Any]:
        request_payload[NG_CAPTURE_FIELD] = admission.model_dump(mode="json")
        request_payload.update(
            logprobs=True,
            top_logprobs=0,
            return_tokens_as_token_ids=True,
        )
        if admission.mode == "token_in":
            request_payload["required_prefix_token_ids"] = list(admission.required_prefix_token_ids)
        return request_payload

    async def _finalize_admitted_response(
        self,
        response_payload: dict[str, Any],
        *,
        context: CaptureContext,
        ledger: CaptureLedger,
        admission: CaptureAdmission,
    ) -> None:
        """Publish the worker's coordinates as a ledger row.

        The ordering invariant the external sink requires — a call must not
        become a lineage parent until its staged record is durable — holds
        structurally: the worker stages before acknowledging, so the ledger
        row (which is what makes the call resolvable) is written only after
        the coordinates arrive. The shared lifecycle strips custody fields
        after this method returns.
        """
        coords_payload = response_payload.pop(NG_COMMIT_COORDS_FIELD, None)
        if coords_payload is None:
            await ledger.record_failure(
                context.rollout_id,
                context.model_call_id,
                "worker_response_missing_commit_coordinates",
            )
            return
        coords = CommitCoords.model_validate(coords_payload)
        if coords.rollout_id != context.rollout_id or coords.model_call_id != context.model_call_id:
            raise ValueError(
                f"coordinates for {coords.rollout_id}/{coords.model_call_id} do not match the "
                f"active capture context {context.rollout_id}/{context.model_call_id}"
            )
        if coords.disposition == "capture_failed":
            await ledger.record_failure(
                context.rollout_id,
                context.model_call_id,
                "worker_capture_failed",
            )
            return
        if coords.parent_call_id != admission.parent_call_id or coords.prev_len != admission.prev_len:
            raise ValueError(f"coordinates for {coords.model_call_id} diverge from admission")
        # The served envelope id is the terminal-attribution join key: the
        # agent proves which response it kept by possessing it. Observe the
        # payload's own id; never mint one. A served completion without an
        # id is a stamping bug and fails closed (poisons the call below).
        response_id = str(response_payload.get("id") or "")
        if not response_id:
            raise ValueError(f"served response for {coords.model_call_id} carries no envelope id")
        child_staging_chain = list(context.parent_staging_chain) + [str(coords.staging_key)]
        response_items, _ = strip_token_fields(response_to_output_items(response_payload))
        # Content-witness keys, hashed while the response is still
        # server-side: this call's own output, and request + output (the
        # cumulative reading). Unfingerprintable content abstains (None)
        # rather than poisoning a valid completion.
        try:
            output_fingerprint = assistant_fingerprint(list(response_items)) or None
            continuation_fingerprint = (
                assistant_fingerprint(list(context.request_items or []) + list(response_items)) or None
            )
        except (TypeError, ValueError):
            output_fingerprint = None
            continuation_fingerprint = None
        # Custody rows are token-free: the worker's chained ``chain_hash``
        # replaces the cumulative token array, and its whole-sequence
        # ``cumulative_hash`` becomes the row digest. Finalization
        # re-verifies both against the staged deltas in TQ.
        await ledger.record(
            context.rollout_id,
            context.model_call_id,
            list(context.request_items or []),
            response_items,
            [],
            coords.cumulative_hash or "",
            parent_call_id=coords.parent_call_id,
            staging_key=coords.staging_key,
            weight_version=coords.weight_version,
            prev_len=coords.prev_len,
            delta_len=coords.delta_len,
            cum_len=coords.cum_len,
            staging_digest=coords.digest,
            extras_digest=coords.extras_digest,
            mode=admission.mode,
            logical_request_id=context.logical_request_id,
            admitted_at=context.admitted_at,
            staging_chain=child_staging_chain,
            chain_hash=coords.chain_hash,
            cumulative_hash=coords.cumulative_hash,
            response_id=response_id,
            output_fingerprint=output_fingerprint,
            continuation_fingerprint=continuation_fingerprint,
            fingerprint_version=LINEAGE_FINGERPRINT_VERSION,
        )
        mark_external_staging_committed(
            rollout_id=coords.rollout_id,
            model_call_id=coords.model_call_id,
        )


class MegatronLedgerCaptureHandler(_BaseExternalCaptureHandler):
    """Record MInf ledger custody now for deferred rollout-end staging."""

    _INVALID_CAPTURE_REASON = "invalid_megatron_ledger_reference"
    _CAPTURE_ERROR_MESSAGE = "Megatron ledger capture failed for rollout %s call %s"
    _POISON_ERROR_MESSAGE = "Could not poison rollout %s call %s after a failed MInf reference"

    def __init__(self, adapter: MegatronCaptureAdapter | None = None) -> None:
        self._adapter = adapter or MegatronCaptureAdapter()

    def _prepare_admitted_request(
        self,
        request_payload: dict[str, Any],
        admission: CaptureAdmission,
    ) -> dict[str, Any]:
        request_payload.update(
            logprobs=True,
            top_logprobs=0,
            return_tokenized_data=True,
        )
        if admission.mode == "token_in":
            self._adapter.enter_prefix(request_payload, list(admission.required_prefix_token_ids))
        return request_payload

    async def _finalize_admitted_response(
        self,
        response_payload: dict[str, Any],
        *,
        context: CaptureContext,
        ledger: CaptureLedger,
        admission: CaptureAdmission,
    ) -> None:
        pending = self._adapter.pending_capture(response_payload, admission)
        response_items, _ = strip_token_fields(response_to_output_items(response_payload))
        cumulative = list(pending.cumulative_token_ids)
        chain_hash = compute_chain_hash(
            admission.parent_chain_hash,
            pending.token_ids_delta,
        )
        await ledger.record(
            context.rollout_id,
            context.model_call_id,
            list(context.request_items or []),
            response_items,
            cumulative,
            compute_digest(cumulative),
            parent_call_id=admission.parent_call_id,
            prev_len=admission.prev_len,
            delta_len=pending.delta_len,
            cum_len=pending.cum_len,
            mode=admission.mode,
            logical_request_id=context.logical_request_id
            or (str(response_payload["id"]) if response_payload.get("id") else None),
            admitted_at=context.admitted_at,
            ledger_request_uid=pending.request_uid,
            chain_hash=chain_hash,
            cumulative_hash=hash_token_ids(cumulative),
            response_id=pending.request_uid,
        )
        mark_external_ledger_capture_recorded(
            rollout_id=context.rollout_id,
            model_call_id=context.model_call_id,
        )


def make_external_capture_handler(backend: ExternalStagingBackend) -> ExternalCaptureHandler:
    """Create the external capture strategy selected by typed configuration."""
    if backend == "vllm_worker":
        return VLLMWorkerCaptureHandler()
    if backend == "megatron_ledger":
        return MegatronLedgerCaptureHandler()
    raise ValueError(f"Unsupported external staging backend: {backend}")
