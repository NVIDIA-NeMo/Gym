# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-owned, engine-neutral capture with stage-before-response ordering."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

from nemo_gym.token_id_capture.staging.digest import (
    EXTRAS_DIGEST_VERSION,
    STAGING_DIGEST_VERSION,
    build_staging_delta,
    compute_chain_hash,
    compute_extras_digest,
    compute_staging_digest,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.protocols import (
    CaptureAdapter,
    StagingSink,
    WeightVersionProvider,
)
from nemo_gym.token_id_capture.staging.records import (
    CaptureAdmission,
    CommitCoords,
    StagedCallRecord,
    StageResult,
)


LOGGER = logging.getLogger(__name__)


class CaptureError(Exception):
    """A caller violated the worker capture lifecycle."""


class StreamingUnsupportedError(CaptureError):
    """Worker-owned staging currently accepts only complete responses."""


@dataclass
class ActiveCall:
    """One admitted call, the policy version present at admission, and its resolved prefix.

    ``prefix_token_ids`` is the exact token prefix the engine prompt must begin
    with. ``begin_call`` resolves it from the admission's inline prefix or from
    the caller-supplied ids fetched for a ``staging_chain``. Empty for a text root.
    """

    admission: CaptureAdmission
    weight_version: int
    prefix_token_ids: list[int] = field(default_factory=list)
    completed: bool = field(default=False, init=False)

    @property
    def rollout_id(self) -> str:
        return self.admission.rollout_id

    @property
    def model_call_id(self) -> str:
        return self.admission.model_call_id


class RolloutTokenCapture:
    """Build, stage, and acknowledge exact per-call token deltas."""

    def __init__(
        self,
        *,
        sink: StagingSink,
        weight_version_fn: WeightVersionProvider,
        adapter: CaptureAdapter | None = None,
    ) -> None:
        self._sink = sink
        self._weight_version_fn = weight_version_fn
        self._adapter = adapter
        # Guards only the per-call single-completion transition. Sink writes
        # run unlocked and may overlap across calls (StagingSink contract).
        self._completion_lock = threading.Lock()

    @property
    def adapter(self) -> CaptureAdapter | None:
        return self._adapter

    def begin_call(
        self,
        admission: CaptureAdmission,
        *,
        prefix_token_ids: list[int] | None = None,
        stream: bool = False,
    ) -> ActiveCall:
        """Admit a typed gate contract and stamp its generation weight version.

        ``prefix_token_ids`` is the resolved parent prefix for a ``token_in`` call.
        It is required when the admission stores its prefix as a ``staging_chain``:
        the caller fetches and concatenates those staged deltas and passes the ids here,
        the same list it hands to ``CaptureAdapter.enter_prefix``.
        When the admission carries the prefix inline the argument may be omitted;
        if given it must match. A text root accepts no prefix.
        Violations are caller bugs and raise ``CaptureError``; they never poison the call.
        """
        if not isinstance(admission, CaptureAdmission):
            raise TypeError("admission must be a CaptureAdmission")
        if stream:
            raise StreamingUnsupportedError(
                f"rollout {admission.rollout_id} call {admission.model_call_id}: "
                "token capture does not support streaming responses"
            )
        resolved_prefix = self._resolve_prefix(admission, prefix_token_ids)
        weight_version = self._weight_version_fn()
        if type(weight_version) is not int or weight_version < 0:
            raise CaptureError(f"weight_version_fn must return a non-negative int, got {weight_version!r}")
        return ActiveCall(admission=admission, weight_version=weight_version, prefix_token_ids=resolved_prefix)

    @staticmethod
    def _resolve_prefix(admission: CaptureAdmission, prefix_token_ids: list[int] | None) -> list[int]:
        where = f"rollout {admission.rollout_id} call {admission.model_call_id}"
        if prefix_token_ids is not None:
            if not isinstance(prefix_token_ids, list) or any(
                type(token_id) is not int or token_id < 0 for token_id in prefix_token_ids
            ):
                raise CaptureError(f"{where}: prefix_token_ids must be a list of non-negative ints")
        if admission.mode == "text":
            if prefix_token_ids:
                raise CaptureError(f"{where}: a text root admission accepts no prefix_token_ids")
            return []
        inline = list(admission.required_prefix_token_ids)
        if prefix_token_ids is None:
            if not inline:
                raise CaptureError(
                    f"{where}: staging_chain admission requires the caller to pass the resolved prefix_token_ids"
                )
            return inline
        if len(prefix_token_ids) != admission.prev_len:
            raise CaptureError(
                f"{where}: prefix_token_ids length {len(prefix_token_ids)} does not equal prev_len {admission.prev_len}"
            )
        if inline and inline != list(prefix_token_ids):
            raise CaptureError(f"{where}: prefix_token_ids conflict with the admission's inline prefix")
        return list(prefix_token_ids)

    def complete_call(
        self,
        call: ActiveCall,
        *,
        prompt_token_ids: list[int],
        generated_token_ids: list[int],
        generated_logprobs: list[float],
        extras: dict[str, Any] | None = None,
    ) -> CommitCoords:
        """Stage a normalized delta before returning lightweight coordinates."""
        self._claim_completion(call)
        admission = call.admission
        try:
            if admission.mode == "token_in" and prompt_token_ids[: admission.prev_len] != call.prefix_token_ids:
                raise ValueError("generation prompt does not begin with the gate-authorized token prefix")
            token_ids_delta, token_mask_delta, logprobs_delta = build_staging_delta(
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=generated_token_ids,
                generated_log_probs=generated_logprobs,
                prev_len=admission.prev_len,
            )
            delta_len = len(token_ids_delta)
            cum_len = admission.prev_len + delta_len
            # For a continuation, the comparison above verifies that the prompt starts with the required prefix.
            # Appending the generated IDs therefore yields the complete sequence for both hashes.
            chain_hash = compute_chain_hash(admission.parent_chain_hash, token_ids_delta)
            cumulative_hash = hash_token_ids(list(prompt_token_ids) + list(generated_token_ids))
            extras_digest = compute_extras_digest(extras)
            digest = compute_staging_digest(
                schema_version=admission.schema_version,
                digest_version=STAGING_DIGEST_VERSION,
                extras_digest_version=EXTRAS_DIGEST_VERSION,
                rollout_id=admission.rollout_id,
                model_call_id=admission.model_call_id,
                parent_call_id=admission.parent_call_id,
                mode=admission.mode,
                prev_len=admission.prev_len,
                delta_len=delta_len,
                cum_len=cum_len,
                weight_version=call.weight_version,
                token_ids_delta=token_ids_delta,
                token_mask_delta=token_mask_delta,
                generation_log_probs_delta=logprobs_delta,
                extras_digest=extras_digest,
                chain_hash=chain_hash,
                cumulative_hash=cumulative_hash,
            )
            record = StagedCallRecord(
                rollout_id=admission.rollout_id,
                model_call_id=admission.model_call_id,
                parent_call_id=admission.parent_call_id,
                mode=admission.mode,
                prev_len=admission.prev_len,
                delta_len=delta_len,
                cum_len=cum_len,
                weight_version=call.weight_version,
                digest=digest,
                token_ids_delta=token_ids_delta,
                token_mask_delta=token_mask_delta,
                generation_log_probs_delta=logprobs_delta,
                extras=extras,
                extras_digest=extras_digest,
                chain_hash=chain_hash,
                cumulative_hash=cumulative_hash,
            )
        except (TypeError, ValueError, OverflowError):
            LOGGER.exception(
                "token capture could not build rollout %s call %s",
                admission.rollout_id,
                admission.model_call_id,
            )
            return self._failed_coords(call)
        try:
            # Unlocked: the completion claim above already made this call the
            # sole stager, and cross-call ordering comes from stage-before-ack
            # (a child is only admitted after its parent's coords returned).
            # Serializing here would head-of-line block every concurrent
            # completion on the worker behind one sink round trip.
            result = self._sink.stage(record)
            if not isinstance(result, StageResult):
                raise TypeError(f"StagingSink.stage returned {type(result).__name__}, expected StageResult")
        except Exception:
            # The sink is framework code outside Gym's exception hierarchy.
            # This deliberately broad boundary keeps capture failure from
            # failing the model completion.
            LOGGER.exception(
                "token staging failed for rollout %s call %s",
                admission.rollout_id,
                admission.model_call_id,
            )
            return self._failed_coords(call)
        if not result.ok:
            LOGGER.warning(
                "token staging sink rejected rollout %s call %s: %s",
                admission.rollout_id,
                admission.model_call_id,
                result.error,
            )
            return self._failed_coords(call)
        return CommitCoords(
            rollout_id=admission.rollout_id,
            model_call_id=admission.model_call_id,
            parent_call_id=admission.parent_call_id,
            prev_len=admission.prev_len,
            delta_len=delta_len,
            cum_len=cum_len,
            weight_version=call.weight_version,
            disposition="staged",
            digest=digest,
            extras_digest=extras_digest,
            staging_key=result.staging_key,
            chain_hash=chain_hash,
            cumulative_hash=cumulative_hash,
        )

    def complete_call_from_response(
        self,
        call: ActiveCall,
        response_payload: dict[str, Any],
    ) -> CommitCoords:
        """Extract engine-native material and stage it as one atomic lifecycle step."""
        if self._adapter is None:
            raise CaptureError("complete_call_from_response requires a CaptureAdapter")
        try:
            prompt_token_ids = self._adapter.extract_prompt_ids(response_payload)
            generated_token_ids, generated_logprobs = self._adapter.extract_generation(response_payload)
            extras = self._adapter.extract_extras(response_payload)
        except Exception:
            # Adapters are engine/framework boundaries and may expose native
            # exception types. Extraction failure poisons capture only.
            LOGGER.exception(
                "token extraction failed for rollout %s call %s",
                call.rollout_id,
                call.model_call_id,
            )
            self._claim_completion(call)
            return self._failed_coords(call)
        return self.complete_call(
            call,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            generated_logprobs=generated_logprobs,
            extras=extras,
        )

    def fail_call(self, call: ActiveCall, *, reason: str) -> CommitCoords:
        """Poison one admitted call that failed before durable staging."""
        self._claim_completion(call)
        LOGGER.warning(
            "token capture failed for rollout %s call %s: %s",
            call.rollout_id,
            call.model_call_id,
            reason,
        )
        return self._failed_coords(call)

    def _claim_completion(self, call: ActiveCall) -> None:
        with self._completion_lock:
            if call.completed:
                raise CaptureError(f"rollout {call.rollout_id} call {call.model_call_id} was already completed")
            call.completed = True

    @staticmethod
    def _failed_coords(call: ActiveCall) -> CommitCoords:
        admission = call.admission
        return CommitCoords(
            rollout_id=admission.rollout_id,
            model_call_id=admission.model_call_id,
            parent_call_id=admission.parent_call_id,
            prev_len=admission.prev_len,
            delta_len=0,
            cum_len=admission.prev_len,
            weight_version=call.weight_version,
            disposition="capture_failed",
        )


class CaptureHost:
    """Minimal serving-layer seam used by ``install_capture``."""

    def __init__(self) -> None:
        self.token_capture: RolloutTokenCapture | None = None

    def install_token_capture(self, capture: RolloutTokenCapture) -> None:
        self.token_capture = capture


def install_capture(
    serving_layer: Any,
    *,
    sink: StagingSink,
    weight_version_fn: WeightVersionProvider,
    adapter: CaptureAdapter | None = None,
) -> RolloutTokenCapture:
    """Wire worker-owned staging into one inference serving layer."""
    if not isinstance(sink, StagingSink):
        raise TypeError(f"sink does not implement StagingSink: {type(sink)!r}")
    if not callable(weight_version_fn):
        raise TypeError(f"weight_version_fn must be callable, got {type(weight_version_fn)!r}")
    install = getattr(serving_layer, "install_token_capture", None)
    if not callable(install):
        raise TypeError(f"serving layer {type(serving_layer)!r} does not expose install_token_capture(capture)")
    capture = RolloutTokenCapture(
        sink=sink,
        weight_version_fn=weight_version_fn,
        adapter=adapter,
    )
    install(capture)
    return capture
