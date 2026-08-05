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
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from nemo_gym.token_id_capture.lineage import LineageIndex
from nemo_gym.token_id_capture.protocols import TokenSink
from nemo_gym.token_id_capture.records import (
    TokenEntry,
    cumulative_tokens,
    extract_token_fields,
    response_to_output_items,
    stamp_lineage,
    strip_token_fields,
)


logger = logging.getLogger(__name__)


# Which recorded call each request continues, per rollout. Process-wide because the
# capture path is per request; bounded, so an abandoned rollout cannot leak. Losing an
# entry costs a fallback to prefix matching, never a wrong answer.
#
# Being process-wide means it does not span uvicorn workers. With num_workers > 1 the
# calls of one rollout can be handled by different workers, and a call landing on a
# worker that did not record its parent resolves nothing: parent_call_id stays unset and
# the builder matches the parent by token prefix instead. Prefix supply, which needs a
# resolved parent, does not fire for those calls. Both degrade rather than break, and
# parent_link_fallbacks reports the rate. The file store is unaffected because it is keyed
# per rollout and appends under a file lock, which holds across processes.
_LINEAGE = LineageIndex()


def lineage_index() -> LineageIndex:
    return _LINEAGE


@dataclass
class CaptureContext:
    """Describe one in-flight training-token capture.

    The context identifies the rollout and model call.
    ``token_sink`` receives the resulting record.
    A framework may provide any ``TokenSink`` implementation.
    """

    rollout_id: str
    model_call_id: str
    # ``None`` means another process owns record staging.
    # The context still carries the capture identity.
    token_sink: TokenSink | None
    model: str = ""
    # ``commit_entry`` sets this after another capture path records the call.
    committed: bool = False
    # Which recorded call this request continues, resolved once before the request is
    # dispatched. Both consumers read it from here rather than resolving for themselves:
    # prefix supply needs the tokens before the engine is called, capture needs the id
    # afterwards, and resolving twice invites the two to disagree. ``parent_resolved``
    # separates "resolved to nothing" from "never resolved".
    parent_resolved: bool = False
    parent_call_id: str | None = None
    parent_tokens: list[int] = field(default_factory=list)


_CAPTURE_CONTEXT: ContextVar[CaptureContext | None] = ContextVar("nemo_gym_capture_context", default=None)


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


def resolve_parent(request_messages: list | None) -> None:
    """Resolve which recorded call this request continues.

    Use the request representation received from the harness.
    Resolve once before dialect conversion or dispatch.
    Prefix supply and capture then share one parent decision.
    Return without work for untagged traffic.
    A miss leaves the parent link unset.
    """
    context = _CAPTURE_CONTEXT.get()
    if context is None or request_messages is None:
        return
    context.parent_resolved = True
    try:
        parent = _LINEAGE.for_rollout(context.rollout_id).resolve(request_messages)
    except Exception:
        logger.warning("Could not resolve a parent for rollout %s.", context.rollout_id, exc_info=True)
        return
    if parent is None:
        return
    context.parent_call_id = parent.call_id
    context.parent_tokens = list(parent.cum_tokens)


async def capture_tokens(
    response: Any,
    parent_call_id: str | None = None,
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
        # Which call does this one continue? Normally decided before dispatch by
        # ``resolve_parent``, so the record names the same call whose tokens were supplied.
        # A caller that did not resolve first still gets a link, from the same messages.
        lineage = _LINEAGE.for_rollout(context.rollout_id)
        if parent_call_id is None:
            if context.parent_resolved:
                parent_call_id = context.parent_call_id
            elif request_messages is not None:
                parent = lineage.resolve(request_messages)
                parent_call_id = parent.call_id if parent is not None else None
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
            created_at=time.time(),
        )
    except Exception:
        await _capture_failed(context, "build")
        return
    await commit_entry(entry, parent_call_id)
    # Index this call by the conversation a continuation of it would carry, so the next
    # request resolves to it. Indexing lives here rather than in ``commit_entry`` because it
    # is keyed on the request the server saw, which an engine-side caller does not have. The
    # digest it reads is stamped during the commit above.
    if request_messages is not None:
        try:
            # Index the served items as they are, not a turn rebuilt from them. The next request
            # echoes those items back, and the fingerprint canonicalizes both sides the same way,
            # so anything in between is a chance for the two to disagree. One response can also
            # echo as several items, which a single rebuilt turn cannot represent.
            lineage.record(
                context.model_call_id,
                list(request_messages) + list(entry.output_items or []),
                cumulative_tokens(entry),
                entry.digest or "",
                context_len=len(request_messages),
            )
        except Exception:
            # Only costs the next call its parent link, which falls back to prefix matching.
            logger.warning("Could not index lineage for rollout %s.", context.rollout_id, exc_info=True)


async def commit_entry(entry: TokenEntry, parent_call_id: str | None = None) -> None:
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
        # cum_len/digest describe this call and are always computable; the parent
        # link is filled only when the model server resolved one.
        stamp_lineage(entry, parent_call_id)
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
