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

"""Served-layer token capture for one model call.

Token ids are dropped on the wire for streaming responses (Anthropic
``/v1/messages``, OpenAI chat SSE), so the capture middleware, which only sees
the streamed bytes, cannot record them. But the model server holds the
complete response, token ids included, for a moment before it synthesizes the
SSE stream. The middleware therefore hands the model server a per-request "token
sink" through a request-scoped ContextVar; the server calls ``capture_tokens``
on its complete response and the sink writes a ``TokenEntry``.

The sink carries the ``model_call_id`` the middleware minted for the same call,
so a captured ``TokenEntry`` joins its ``ModelCallRecord``. Only the middleware
sets a sink (for rollout-correlated, observed calls), so ordinary untagged
traffic captures nothing.
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
    """What the capture middleware hands the model server for one call: which
    rollout and call this is, and where the record goes.

    ``sink`` is Gym's file store by default and anything satisfying ``TokenSink``
    otherwise, which is how a training framework redirects the write to its own data
    plane without changing the capture code. It is typed as the protocol rather than
    the file store so that redirection is a supported path and not an accident.
    """

    rollout_id: str
    model_call_id: str
    sink: TokenSink
    model: str = ""
    # Which recorded call this request continues, resolved once before the request is
    # dispatched. Both consumers read it from here rather than resolving for themselves:
    # prefix supply needs the tokens before the engine is called, capture needs the id
    # afterwards, and resolving twice invites the two to disagree. ``parent_resolved``
    # separates "resolved to nothing" from "never resolved".
    parent_resolved: bool = False
    parent_call_id: str | None = None
    parent_tokens: list[int] = field(default_factory=list)


_TOKEN_SINK: ContextVar[CaptureContext | None] = ContextVar("nemo_gym_token_sink", default=None)


def set_token_sink(sink: CaptureContext) -> Token:
    return _TOKEN_SINK.set(sink)


def reset_token_sink(token: Token) -> None:
    _TOKEN_SINK.reset(token)


def resolve_parent(request_messages: list | None) -> None:
    """Resolve which recorded call this request continues, before it is dispatched.

    Called with the request as the server received it, ahead of any dialect conversion. The
    index was built from the same representation, so the two agree by construction rather than
    by two code paths canonicalizing their way back to each other. Resolving here also means it
    happens once: the prefix a request is given and the parent its record names are the same
    decision, so they cannot diverge.

    No-op for untagged traffic. A miss is normal and leaves the link unset.
    """
    sink = _TOKEN_SINK.get()
    if sink is None or request_messages is None:
        return
    sink.parent_resolved = True
    try:
        parent = _LINEAGE.for_rollout(sink.rollout_id).resolve(request_messages)
    except Exception:
        logger.warning("Could not resolve a parent for rollout %s.", sink.rollout_id, exc_info=True)
        return
    if parent is None:
        return
    sink.parent_call_id = parent.call_id
    sink.parent_tokens = list(parent.cum_tokens)


async def capture_tokens(
    response: Any,
    parent_call_id: str | None = None,
    request_messages: list | None = None,
) -> None:
    """Record a ``TokenEntry`` from a complete model response when a sink is set.

    ``response`` is a served response as a pydantic model or dict. No-op when no
    sink is active (untagged traffic) or the response carries no token ids. The
    write is awaited, so the entry is durable before the model call returns and a
    post-rollout reader always sees it, with no background writer to drain.
    """
    sink = _TOKEN_SINK.get()
    if sink is None:
        return
    # Everything that reads the response is guarded, not just the write. Decoding a payload and
    # validating a record can fail on malformed token data exactly as writing it can, and the
    # consequence is the same: the rollout is short a call. It is guarded here rather than left
    # to the caller because the caller is the model server's own response path, so an exception
    # escaping this function would fail the model call and break the harness's run.
    try:
        if hasattr(response, "model_dump"):
            payload = response.model_dump()
        elif isinstance(response, dict):
            payload = response
        else:
            return
        info = extract_token_fields(payload)
        if info is None:
            return
        # Content only: the arrays live on the entry, not on the items as well.
        content_items, token_item_index = strip_token_fields(response_to_output_items(payload))
        # Which call does this one continue? Normally decided before dispatch by
        # ``resolve_parent``, so the record names the same call whose tokens were supplied.
        # A caller that did not resolve first still gets a link, from the same messages.
        lineage = _LINEAGE.for_rollout(sink.rollout_id)
        if parent_call_id is None:
            if sink.parent_resolved:
                parent_call_id = sink.parent_call_id
            elif request_messages is not None:
                parent = lineage.resolve(request_messages)
                parent_call_id = parent.call_id if parent is not None else None
        entry = TokenEntry(
            rollout_id=sink.rollout_id,
            model_call_id=sink.model_call_id,
            model=sink.model or str(payload.get("model") or ""),
            prompt_token_ids=info.get("prompt_token_ids") or [],
            generation_token_ids=info.get("generation_token_ids") or [],
            generation_log_probs=info.get("generation_log_probs") or [],
            routed_experts=info.get("routed_experts"),
            # Keep the content (assistant text, tool calls) so the trajectory the trainer
            # reads is not token-only, since text-based penalties need it.
            output_items=content_items,
            token_item_index=token_item_index,
            created_at=time.time(),
        )
    except Exception:
        _capture_failed(sink, "build")
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
                sink.model_call_id,
                list(request_messages) + list(entry.output_items or []),
                cumulative_tokens(entry),
                entry.digest or "",
                context_len=len(request_messages),
            )
        except Exception:
            # Only costs the next call its parent link, which falls back to prefix matching.
            logger.warning("Could not index lineage for rollout %s.", sink.rollout_id, exc_info=True)


async def commit_entry(entry: TokenEntry, parent_call_id: str | None = None) -> None:
    """Durably record a finished entry against the in-flight call.

    Public and separate from ``capture_tokens`` because the two halves are useful apart.
    ``capture_tokens`` reads the arrays off a served response; a framework that captures
    engine-side already has them, and the response Gym sees may carry none at all, so it
    needs this half without the extraction half. Forking it instead would duplicate the
    ordering below, which is the part worth sharing.

    No-op when no sink is active. Never raises: capture is best effort per call, but a
    rollout that lost a call is marked so a consumer masks it rather than training on a
    chain with a hole.
    """
    sink = _TOKEN_SINK.get()
    if sink is None:
        return
    try:
        # cum_len/digest describe this call and are always computable; the parent
        # link is filled only when the model server resolved one.
        stamp_lineage(entry, parent_call_id)
        await sink.sink.put(entry)
    except Exception:
        _capture_failed(sink, "write")


def _capture_failed(sink: CaptureContext, stage: str) -> None:
    """Report a capture failure without letting it reach the model call.

    Capture is best effort per call: a bad token payload must never fail the model call and
    break the harness's run. But a rollout that lost a call must not look identical to a
    complete one, so it is marked, and delivery masks the sample rather than training on a
    chain with a hole. Called only from an ``except`` block, so ``exc_info`` picks up the
    active exception.
    """
    logger.warning(
        "Training-token capture failed to %s the record for model call %s of rollout %s.",
        stage,
        sink.model_call_id,
        sink.rollout_id,
        exc_info=True,
    )
    _mark_incomplete(sink)


def _mark_incomplete(sink: CaptureContext) -> None:
    """Mark the rollout, or say loudly why it could not be marked.

    A sink that does not implement ``mark_incomplete`` would otherwise raise inside the
    failure path above and have the exception swallowed, leaving an incomplete rollout
    that looks complete. That is the one outcome this whole path exists to prevent, so
    it is logged at error rather than passed over.
    """
    mark = getattr(sink.sink, "mark_incomplete", None)
    if mark is None:
        logger.error(
            "Sink %s does not implement mark_incomplete. Rollout %s cannot be marked incomplete "
            "and may be trained on with a missing call.",
            type(sink.sink).__name__,
            sink.rollout_id,
        )
        return
    try:
        mark(sink.rollout_id, sink.model_call_id)
    except Exception:
        logger.warning("Could not mark rollout %s incomplete.", sink.rollout_id, exc_info=True)
