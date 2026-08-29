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
    """One admitted call and the policy version present at admission."""

    admission: CaptureAdmission
    weight_version: int
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
        self._lock = threading.Lock()

    @property
    def adapter(self) -> CaptureAdapter | None:
        return self._adapter

    def begin_call(self, admission: CaptureAdmission, *, stream: bool = False) -> ActiveCall:
        """Admit a typed gate contract and stamp its generation weight version."""
        if not isinstance(admission, CaptureAdmission):
            raise TypeError("admission must be a CaptureAdmission")
        if stream:
            raise StreamingUnsupportedError(
                f"rollout {admission.rollout_id} call {admission.model_call_id}: "
                "token capture does not support streaming responses"
            )
        weight_version = self._weight_version_fn()
        if type(weight_version) is not int or weight_version < 0:
            raise CaptureError(f"weight_version_fn must return a non-negative int, got {weight_version!r}")
        return ActiveCall(admission=admission, weight_version=weight_version)

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
            if (
                admission.mode == "token_in"
                and prompt_token_ids[: admission.prev_len] != admission.required_prefix_token_ids
            ):
                raise ValueError("generation prompt does not begin with the gate-authorized token prefix")
            token_ids_delta, token_mask_delta, logprobs_delta = build_staging_delta(
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=generated_token_ids,
                generated_log_probs=generated_logprobs,
                prev_len=admission.prev_len,
            )
            delta_len = len(token_ids_delta)
            cum_len = admission.prev_len + delta_len
            # The prefix custody proof above makes prompt + generation the
            # exact cumulative sequence, so both hashes are computable here
            # without the gate ever materializing tokens.
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
            with self._lock:
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
        with self._lock:
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
