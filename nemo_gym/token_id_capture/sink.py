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

"""Capture training tokens from one complete model response.

Streaming responses omit token ids from the wire.
The model server still holds the complete response before streaming.
Middleware provides a request-scoped token sink.
The model server passes its complete response to ``capture_tokens``.
The sink writes a ``TokenEntry``.
Its ``model_call_id`` joins the corresponding evaluation record.
Untagged traffic has no capture context.
"""

from __future__ import annotations

import logging
import threading
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint
from nemo_gym.token_id_capture.lineage import stamp_continuation
from nemo_gym.token_id_capture.protocols import LineageResolution, LineageStore, TokenSink
from nemo_gym.token_id_capture.records import (
    ParentResolutionStatus,
    TokenEntry,
    extract_token_fields,
    response_to_output_items,
    stamp_lineage,
    strip_token_fields,
)


logger = logging.getLogger(__name__)


@dataclass
class CaptureContext:
    """Describe one in-flight training-token capture.

    The context identifies the rollout and model call.
    ``token_sink`` receives the resulting record.
    A framework may provide any ``TokenSink`` implementation.
    Parent resolution runs once for each call.
    Prefix supply and token capture read the same immutable decision.
    """

    rollout_id: str
    model_call_id: str
    # ``None`` means another process owns record staging.
    # The context still carries the capture identity.
    token_sink: TokenSink | None
    lineage_store: LineageStore | None = None
    model: str = ""
    # ``commit_entry`` sets this after another capture path records the call.
    committed: bool = False
    # Store resolved continuations as parent-relative suffixes.
    delta_records: bool = False
    # This records the model server's intent to request prefix supply.
    prefix_requested: bool = False
    # This records proven application based on generation-time prompt_token_ids.
    prefix_supplied: bool = False
    # Resolve the parent once before dispatch.
    # Downstream inference and capture share this immutable decision.
    parent_resolution: LineageResolution | None = None

    @property
    def parent_call_id(self) -> str | None:
        match = self.parent_resolution.match if self.parent_resolution is not None else None
        return match.model_call_id if match is not None else None

    @property
    def parent_tokens(self) -> list[int]:
        match = self.parent_resolution.match if self.parent_resolution is not None else None
        return list(match.cumulative_token_ids) if match is not None else []


_CAPTURE_CONTEXT: ContextVar[CaptureContext | None] = ContextVar("nemo_gym_capture_context", default=None)

# Worker-level health counters are logged periodically.
_STATS_LOCK = threading.Lock()
_RESOLUTION_COUNTS = {"root": 0, "resolved": 0, "unresolved": 0}
_CAPTURE_FAILURES = [0]
_RESOLVER_UNAVAILABLE_NOTED = [False]


def _count_resolution(status_value: str) -> None:
    with _STATS_LOCK:
        _RESOLUTION_COUNTS[status_value] = _RESOLUTION_COUNTS.get(status_value, 0) + 1
        total = sum(_RESOLUTION_COUNTS.values())
    if total % 1000 == 0:
        logger.info("token-capture resolutions: %s", dict(_RESOLUTION_COUNTS))


def capture_health_snapshot() -> dict:
    """Return worker-level capture health for metrics endpoints."""
    with _STATS_LOCK:
        return {"resolutions": dict(_RESOLUTION_COUNTS), "capture_failures": _CAPTURE_FAILURES[0]}


def set_token_sink(context: CaptureContext) -> Token:
    return _CAPTURE_CONTEXT.set(context)


def current_capture_context() -> CaptureContext | None:
    """Return the capture context for the in-flight call.

    Return ``None`` for untagged traffic.
    Framework inference workers use this identity for staged records.
    """
    return _CAPTURE_CONTEXT.get()


def reset_token_sink(token: Token) -> None:
    _CAPTURE_CONTEXT.reset(token)


async def resolve_parent(request_messages: list | None) -> None:
    """Resolve which recorded call this request continues.

    Use the request representation received from the harness.
    Resolve once before dialect conversion or dispatch.
    Prefix supply and capture then share one parent decision.
    Return without work for untagged traffic.
    Every attempted resolution records a root, resolved, or unresolved decision.
    An unresolved decision includes its reason.
    """
    context = _CAPTURE_CONTEXT.get()
    if context is None or request_messages is None:
        return
    try:
        if not assistant_fingerprint(request_messages):
            context.parent_resolution = LineageResolution(ParentResolutionStatus.ROOT)
        elif context.lineage_store is None:
            context.parent_resolution = LineageResolution(
                ParentResolutionStatus.UNRESOLVED,
                reason="resolver_unavailable",
            )
            # Startup requires an explicit unresolved-continuation opt-in.
            # Emit one warning and track later calls in the counters.
            with _STATS_LOCK:
                first = not _RESOLVER_UNAVAILABLE_NOTED[0]
                _RESOLVER_UNAVAILABLE_NOTED[0] = True
            if first:
                logger.warning(
                    "No lineage resolver is available: every continuation resolves UNRESOLVED "
                    "and multi-call rollouts will be masked (allow_unresolved_continuations is set)."
                )
        else:
            context.parent_resolution = await context.lineage_store.resolve(context.rollout_id, request_messages)
        _count_resolution(context.parent_resolution.status.value)
    except Exception:
        logger.warning("Could not resolve a parent for rollout %s.", context.rollout_id, exc_info=True)
        context.parent_resolution = LineageResolution(
            ParentResolutionStatus.UNRESOLVED,
            reason="lookup_error",
        )


async def register_call_intent() -> None:
    """Record durable call intent before dispatch starts generation.

    ``begin_call`` is an optional sink extension.
    A dangling intent identifies a lost entry.
    Failure happens before generation and propagates to the caller.
    The harness can retry without spending inference compute.
    Sinks without ``begin_call`` cannot report a missing final entry this way.
    """
    context = _CAPTURE_CONTEXT.get()
    if context is None or context.token_sink is None:
        return
    begin = getattr(context.token_sink, "begin_call", None)
    if begin is None:
        return
    await begin(context.rollout_id, context.model_call_id)


async def capture_tokens(
    response: Any,
    request_messages: list | None = None,
) -> None:
    """Record a ``TokenEntry`` from a complete model response.

    Accept a Pydantic model or dictionary.
    Return without work when no capture context exists.
    Mark local capture incomplete when required token ids are absent.
    Await the write before the model call returns.
    """
    context = _CAPTURE_CONTEXT.get()
    if context is None:
        return
    # Guard response decoding and record validation.
    # Either failure leaves the rollout short one call.
    # Capture errors must not fail the model call.
    try:
        if hasattr(response, "model_dump"):
            payload = response.model_dump()
        elif isinstance(response, dict):
            payload = response
        else:
            await _capture_missing(context, f"the response is a {type(response).__name__}")
            return
        info = extract_token_fields(payload)
        if info is None:
            await _capture_missing(context, "the response carries no token ids")
            return
        # Keep content on the output items.
        # Store token arrays only on the entry.
        content_items, token_item_index = strip_token_fields(response_to_output_items(payload))
        # Reuse the parent selected before dispatch.
        # Resolve here only when the caller skipped the pre-dispatch step.
        if context.parent_resolution is None and request_messages is not None:
            await resolve_parent(request_messages)
        resolution = context.parent_resolution
        if resolution is None:
            resolution = LineageResolution(
                ParentResolutionStatus.UNRESOLVED,
                reason="not_attempted",
            )
        entry = TokenEntry(
            rollout_id=context.rollout_id,
            model_call_id=context.model_call_id,
            model=context.model or str(payload.get("model") or ""),
            prompt_token_ids=info["prompt_token_ids"],
            generation_token_ids=info["generation_token_ids"],
            generation_log_probs=info["generation_log_probs"],
            routed_experts=info.get("routed_experts"),
            # Preserve content for text-based training penalties.
            output_items=content_items,
            token_item_index=token_item_index,
            # Observe the served payload's own id; never mint one.
            # The Anthropic mapping reuses this id on its outer envelope,
            # so the recorded id matches what the client received in every dialect.
            response_id=str(payload.get("id") or "") or None,
            created_at=time.time(),
            prefix_requested=context.prefix_requested,
            prefix_supplied=context.prefix_supplied,
        )
        if request_messages is not None:
            stamp_continuation(entry, list(request_messages))
    except Exception:
        await _capture_failed(context, "build")
        return
    await commit_entry(entry, parent_resolution=resolution)


async def commit_entry(
    entry: TokenEntry,
    *,
    parent_resolution: LineageResolution | None = None,
) -> None:
    """Durably record a finished entry against the in-flight call.

    ``capture_tokens`` extracts arrays from a served response.
    Engine-side capture may already have those arrays.
    Engine-side callers can use this method directly.
    Return without work when no capture context exists.
    Capture failures mark the rollout incomplete.
    This method never fails the model call.
    """
    context = _CAPTURE_CONTEXT.get()
    if context is None:
        return
    if entry.rollout_id != context.rollout_id or entry.model_call_id != context.model_call_id:
        logger.warning(
            "Training-token capture identity mismatch for model call %s of rollout %s.",
            context.model_call_id,
            context.rollout_id,
        )
        await _mark_incomplete(context)
        return
    if context.token_sink is None:
        context.committed = True
        return
    try:
        # Use the resolution decided before dispatch.
        # Engine-side callers may pass their own resolution.
        resolution = parent_resolution or context.parent_resolution
        if resolution is None:
            resolution = LineageResolution(
                ParentResolutionStatus.UNRESOLVED,
                reason="not_attempted",
            )
        # The digest always describes the full sequence.
        # Delta storage changes representation, not lineage identity.
        # The parent decision is persisted with the same sink write.
        cumulative = None
        if (
            context.delta_records
            and not entry.prompt_is_delta
            and resolution.status == ParentResolutionStatus.RESOLVED
            and resolution.match is not None
            and resolution.match.cumulative_token_ids
        ):
            parent_cum = list(resolution.match.cumulative_token_ids)
            prompt = list(entry.prompt_token_ids)
            # Store a suffix only when the prompt extends the exact parent tokens.
            # Otherwise retain the full prompt and preserve a safe reconstruction anchor.
            if len(prompt) >= len(parent_cum) and prompt[: len(parent_cum)] == parent_cum:
                cumulative = prompt + list(entry.generation_token_ids)
                entry.prompt_token_ids = prompt[len(parent_cum) :]
                entry.prompt_is_delta = True
        stamp_lineage(
            entry,
            resolution.match.model_call_id if resolution.match is not None else None,
            parent_resolution=resolution.status,
            cumulative=cumulative,
        )
        entry.parent_resolution_reason = resolution.reason or ""
        await context.token_sink.put(entry)
        context.committed = True
    except Exception:
        await _capture_failed(context, "write")


async def _capture_failed(context: CaptureContext, stage: str) -> None:
    """Report a capture failure without letting it reach the model call.

    Bad token payloads must not fail the model call.
    Mark the rollout so consumers can mask the sample.
    Call this only from an ``except`` block.
    """
    with _STATS_LOCK:
        _CAPTURE_FAILURES[0] += 1
        failures = _CAPTURE_FAILURES[0]
    if failures % 10 == 0:
        logger.error("Training-token capture has failed %d times in this worker.", failures)
    logger.warning(
        "Training-token capture failed to %s the record for model call %s of rollout %s.",
        stage,
        context.model_call_id,
        context.rollout_id,
        exc_info=True,
    )
    await _mark_incomplete(context)


async def _capture_missing(context: CaptureContext, reason: str) -> None:
    """Mark the rollout when a call this process should have recorded produced nothing.

    A response with no token ids is a hole in the chain rather than traffic to skip.
    The builder reads the gap between one call's tokens and the next call's prompt as tool output.
    A skipped call's generated tokens then enter the next prompt with mask zero.
    Policy tokens would train as if the environment produced them.

    Two cases are not holes and are left alone.
    A committed call was recorded by another capture path.
    A context without a sink delegates completeness to external staging.
    """
    if context.committed or context.token_sink is None:
        return
    logger.warning(
        "Training-token capture has no token ids for model call %s of rollout %s: %s.",
        context.model_call_id,
        context.rollout_id,
        reason,
    )
    await _mark_incomplete(context)


async def _mark_incomplete(context: CaptureContext) -> None:
    """Mark the rollout, or say loudly why it could not be marked.

    A missing ``mark_incomplete`` method can hide incomplete capture.
    Log that condition as an error.
    """
    mark = getattr(context.token_sink, "mark_incomplete", None)
    if mark is None:
        logger.error(
            "Sink %s does not implement mark_incomplete. Rollout %s cannot be marked incomplete "
            "and may be trained on with a missing call.",
            type(context.token_sink).__name__,
            context.rollout_id,
        )
        return
    try:
        await mark(context.rollout_id, context.model_call_id)
    except Exception:
        logger.warning("Could not mark rollout %s incomplete.", context.rollout_id, exc_info=True)
