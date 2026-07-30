# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The engine-blind capture core a worker hosts (``RolloutTokenCapture``).

One instance lives in each inference worker process. Per model call the
serving layer drives two hooks:

* ``begin_call`` -- when the request is admitted for generation: validates the
  call identity the gate forwarded (rollout id, gate-minted call id, parent,
  ``prev_len``), rejects streaming (unsupported in the MVP), and stamps the
  framework's current weight version.
* ``complete_call`` -- after generation, BEFORE the response is released:
  builds the per-call staging delta and its digest, stages the
  ``StagedCallRecord`` through the framework's ``TokenSink`` (the design's
  only heavy hop), and assembles the ``CommitCoords`` that must ride the
  response back to the gate.

Fail-closed ordering is this module's contract: ``complete_call`` returns
coords only after ``sink.stage`` reported the bytes durable. Any capture
failure -- a bad delta, a sink error -- degrades to
``disposition="capture_failed"`` coords instead of raising, so the completion
is still served and the agent survives; the gate poisons the rollout and the
finalizer emits a placeholder (`token_capture.on_capture_failure` is
framework policy, applied downstream).

Everything engine-specific -- how prefix ids enter a request, how exact ids
and logprobs come off a response -- is behind the ``CaptureAdapter`` protocol
(``adapters/vllm.py`` first), so this module is tested against a mock adapter
and a mock sink, and an SGLang adapter inherits the ordering for free.

This module is part of the dependency-free capture core: stdlib + pydantic
(via ``records``) only.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from nemo_gym.token_id_capture.staging.digest import build_staging_delta, compute_staging_digest
from nemo_gym.token_id_capture.staging.protocols import CaptureAdapter, TokenSink, WeightVersionProvider
from nemo_gym.token_id_capture.staging.records import (
    CaptureMode,
    CommitCoords,
    StagedCallRecord,
    staging_key,
)


LOGGER = logging.getLogger(__name__)


class CaptureError(Exception):
    """Base for capture contract violations (caller bugs, not data failures)."""


class StreamingUnsupportedError(CaptureError):
    """The MVP captures non-streaming completions only."""


@dataclass
class ActiveCall:
    """One admitted model call, between ``begin_call`` and ``complete_call``.

    ``weight_version`` is stamped at ``begin_call`` -- the version the weights
    had when generation started, which is what staleness accounting needs even
    if a refit lands mid-generation.
    """

    rollout_id: str
    call_id: str
    parent_call_id: Optional[str]
    prev_len: int
    mode: CaptureMode
    weight_version: int
    completed: bool = field(default=False)


class RolloutTokenCapture:
    """Engine-blind per-worker capture: record + digest build, fail-closed
    stage-then-respond ordering, coords assembly."""

    def __init__(
        self,
        *,
        sink: TokenSink,
        weight_version_fn: WeightVersionProvider,
        adapter: Optional[CaptureAdapter] = None,
    ) -> None:
        self._sink = sink
        self._weight_version_fn = weight_version_fn
        self._adapter = adapter
        # Serving layers complete calls from concurrent request handlers.
        self._lock = threading.Lock()

    @property
    def adapter(self) -> Optional[CaptureAdapter]:
        return self._adapter

    # -- per-call hooks ------------------------------------------------------

    def begin_call(
        self,
        *,
        rollout_id: str,
        call_id: str,
        parent_call_id: Optional[str] = None,
        prev_len: int = 0,
        mode: CaptureMode,
        stream: bool = False,
    ) -> ActiveCall:
        """Validate one admitted call and stamp its weight version.

        The gate already resolved the marker and applied the serving rule;
        the worker only enforces local invariants: streaming is unsupported,
        a token-in call must name its committed parent and a positive
        ``prev_len``, and a text-mode call is a fresh root.
        """
        if stream:
            raise StreamingUnsupportedError(
                f"rollout {rollout_id} call {call_id}: token capture does not support streaming responses"
            )
        if mode == "token_in":
            if parent_call_id is None:
                raise CaptureError(f"rollout {rollout_id} call {call_id}: token-in capture requires a parent call id")
            if prev_len <= 0:
                raise CaptureError(
                    f"rollout {rollout_id} call {call_id}: token-in capture requires prev_len > 0, got {prev_len}"
                )
        else:
            if parent_call_id is not None or prev_len != 0:
                raise CaptureError(
                    f"rollout {rollout_id} call {call_id}: text-mode capture is a new root "
                    f"(parent={parent_call_id!r}, prev_len={prev_len})"
                )
        return ActiveCall(
            rollout_id=rollout_id,
            call_id=call_id,
            parent_call_id=parent_call_id,
            prev_len=prev_len,
            mode=mode,
            weight_version=int(self._weight_version_fn()),
        )

    def complete_call(
        self,
        call: ActiveCall,
        *,
        prompt_token_ids: list[int],
        generated_token_ids: list[int],
        generated_logprobs: list[float],
        extras: Optional[dict[str, Any]] = None,
    ) -> CommitCoords:
        """Stage one finished call and assemble the coords for its response.

        Returns ``disposition="staged"`` coords only after the sink reported
        the record durable; every failure returns ``capture_failed`` coords
        (never raises), so the serving layer can still release the completion
        while the gate poisons the rollout.
        """
        if call.completed:
            raise CaptureError(f"rollout {call.rollout_id} call {call.call_id} was already completed")
        call.completed = True
        try:
            token_ids_delta, token_mask_delta, logprobs_delta = build_staging_delta(
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=generated_token_ids,
                generated_logprobs=generated_logprobs,
                prev_len=call.prev_len,
            )
            digest = compute_staging_digest(
                rollout_id=call.rollout_id,
                call_id=call.call_id,
                prev_len=call.prev_len,
                token_ids_delta=token_ids_delta,
                token_mask_delta=token_mask_delta,
                logprobs_delta=logprobs_delta,
            )
            record = StagedCallRecord(
                rollout_id=call.rollout_id,
                call_id=call.call_id,
                parent_call_id=call.parent_call_id,
                prev_len=call.prev_len,
                new_len=call.prev_len + len(token_ids_delta),
                weight_version=call.weight_version,
                digest=digest,
                token_ids_delta=token_ids_delta,
                token_mask_delta=token_mask_delta,
                generation_logprobs_delta=logprobs_delta,
                extras=extras,
            )
            with self._lock:
                result = self._sink.stage(record)
            if not result.ok:
                return self._failed_coords(call, f"sink_rejected:{result.error}")
        except Exception as error:  # fail-closed, never break the completion
            LOGGER.exception("token capture failed for rollout %s call %s", call.rollout_id, call.call_id)
            return self._failed_coords(call, f"{type(error).__name__}: {error}")
        return CommitCoords(
            rollout_id=call.rollout_id,
            call_id=call.call_id,
            parent_call_id=call.parent_call_id,
            delta_len=len(token_ids_delta),
            cum_len=call.prev_len + len(token_ids_delta),
            digest=digest,
            staging_key=record.staging_key,
            weight_version=call.weight_version,
            disposition="staged",
            token_ids_delta=token_ids_delta,
        )

    def complete_call_from_response(
        self,
        call: ActiveCall,
        response_payload: dict[str, Any],
        *,
        extras: Optional[dict[str, Any]] = None,
    ) -> CommitCoords:
        """Adapter-driven ``complete_call``: extract the exact prompt ids and
        the generated ids + logprobs natively off the engine response."""
        if self._adapter is None:
            raise CaptureError("complete_call_from_response requires a CaptureAdapter")
        try:
            prompt_token_ids = self._adapter.extract_prompt_ids(response_payload)
            generated_token_ids, generated_logprobs = self._adapter.extract_generation(response_payload)
        except Exception as error:  # fail-closed: extraction is part of capture
            LOGGER.exception("token extraction failed for rollout %s call %s", call.rollout_id, call.call_id)
            call.completed = True
            return self._failed_coords(call, f"{type(error).__name__}: {error}")
        return self.complete_call(
            call,
            prompt_token_ids=prompt_token_ids,
            generated_token_ids=generated_token_ids,
            generated_logprobs=generated_logprobs,
            extras=extras,
        )

    def fail_call(self, call: ActiveCall, *, reason: str) -> CommitCoords:
        """Coords for a call that failed before or during generation."""
        call.completed = True
        return self._failed_coords(call, reason)

    # -- internals -----------------------------------------------------------

    def _failed_coords(self, call: ActiveCall, reason: str) -> CommitCoords:
        return CommitCoords(
            rollout_id=call.rollout_id,
            call_id=call.call_id,
            parent_call_id=call.parent_call_id,
            delta_len=0,
            cum_len=call.prev_len,
            digest="",
            staging_key=staging_key(call.rollout_id, call.call_id),
            weight_version=call.weight_version,
            disposition="capture_failed",
        )


class CaptureHost:
    """The one-method seam a serving layer implements to receive the capture
    instance ``install_capture`` builds (kept as a plain base class rather
    than a protocol so hosts inherit the storage behavior)."""

    token_capture: Optional[RolloutTokenCapture] = None

    def install_token_capture(self, capture: RolloutTokenCapture) -> None:
        self.token_capture = capture


def install_capture(
    serving_layer: Any,
    *,
    sink: TokenSink,
    weight_version_fn: WeightVersionProvider,
    adapter: Optional[CaptureAdapter] = None,
) -> RolloutTokenCapture:
    """Wire gate-authoritative capture into a worker's serving layer.

    The single call a framework makes at worker startup. ``serving_layer``
    must expose ``install_token_capture(capture)`` (subclass ``CaptureHost``
    or implement the method); the built ``RolloutTokenCapture`` is also
    returned for hosts that prefer to hold it directly.
    """
    if not isinstance(sink, TokenSink):
        raise TypeError(f"sink does not implement the TokenSink protocol: {type(sink)!r}")
    if not callable(weight_version_fn):
        raise TypeError(f"weight_version_fn must be callable, got {type(weight_version_fn)!r}")
    install = getattr(serving_layer, "install_token_capture", None)
    if install is None:
        raise TypeError(
            f"serving layer {type(serving_layer)!r} does not expose install_token_capture(capture); "
            "subclass nemo_gym.token_id_capture.staging.capture.CaptureHost or implement the method"
        )
    capture = RolloutTokenCapture(sink=sink, weight_version_fn=weight_version_fn, adapter=adapter)
    install(capture)
    return capture
