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
``/v1/messages``, OpenAI chat SSE), so the capture middleware -- which only sees
the streamed bytes -- cannot record them. But the model server holds the
complete response WITH token ids for a moment, just before it synthesizes the
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
from dataclasses import dataclass
from typing import Any

from nemo_gym.token_id_capture.lineage import (
    LineageIndex,
    assistant_turn_from_output_items,
)
from nemo_gym.token_id_capture.protocols import TokenSink
from nemo_gym.token_id_capture.records import (
    TokenEntry,
    cumulative_tokens,
    extract_token_fields,
    response_to_output_items,
    stamp_lineage,
    strip_token_fields,
)
from nemo_gym.token_id_capture.store import TokenCaptureStore


logger = logging.getLogger(__name__)


# Which recorded call each request continues, per rollout. Process-wide because the
# capture path is per request; bounded, so an abandoned rollout cannot leak. Losing an
# entry costs a fallback to prefix inference, never a wrong answer.
#
# Being process-wide means it does not span uvicorn workers. With num_workers > 1 the
# calls of one rollout can be handled by different workers, and a call landing on a
# worker that did not record its parent resolves nothing: parent_call_id stays unset and
# the builder infers the parent from token prefixes instead. Prefix supply, which needs a
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

    ``store`` is Gym's file store today; anything satisfying ``TokenSink`` works,
    which is how a training framework redirects the write to its own data plane
    without changing the capture code.
    """

    rollout_id: str
    model_call_id: str
    store: TokenCaptureStore
    model: str = ""
    # Set by the model server when it supplied this call's prefix; read back at
    # capture time so the evidence lands on the record rather than only in a log.
    prefix_supplied: bool = False

    @property
    def sink(self) -> TokenSink:
        return self.store


_TOKEN_SINK: ContextVar[CaptureContext | None] = ContextVar("nemo_gym_token_sink", default=None)


def set_token_sink(sink: CaptureContext) -> Token:
    return _TOKEN_SINK.set(sink)


def current_capture_context() -> CaptureContext | None:
    """The capture context for the in-flight call, or None for untagged traffic."""
    return _TOKEN_SINK.get()


def reset_token_sink(token: Token) -> None:
    _TOKEN_SINK.reset(token)


async def capture_tokens(
    response: Any,
    parent_call_id: str | None = None,
    request_messages: list | None = None,
) -> None:
    """Record a ``TokenEntry`` from a complete model response when a sink is set.

    ``response`` is a served response as a pydantic model or dict. No-op when no
    sink is active (untagged traffic) or the response carries no token ids. The
    write is offloaded and awaited, so the entry is durable before the model call
    returns -- a post-rollout reader always sees it, with no background writer to
    drain.
    """
    sink = _TOKEN_SINK.get()
    if sink is None:
        return
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
    # Which call does this one continue? Resolved from the conversation the request already
    # carries, so the harness needs to send nothing extra. A miss (new conversation, rewritten
    # history, or two recorded calls with byte-identical output) leaves the link unset and the
    # builder infers from token prefixes instead.
    lineage = _LINEAGE.for_rollout(sink.rollout_id)
    if parent_call_id is None and request_messages is not None:
        parent = lineage.resolve(request_messages)
        parent_call_id = parent.call_id if parent is not None else None
    try:
        entry = TokenEntry(
            rollout_id=sink.rollout_id,
            model_call_id=sink.model_call_id,
            model=sink.model or str(payload.get("model") or ""),
            prompt_token_ids=info.get("prompt_token_ids") or [],
            generation_token_ids=info.get("generation_token_ids") or [],
            generation_log_probs=info.get("generation_log_probs") or [],
            routed_experts=info.get("routed_experts"),
            # Keep the content (assistant text, tool calls) so the trajectory the trainer
            # reads is not token-only -- text-based penalties need it.
            output_items=content_items,
            token_item_index=token_item_index,
            created_at=time.time(),
            prefix_supplied=sink.prefix_supplied,
        )
        # cum_len/digest describe this call and are always computable; the parent
        # link is filled only when the model server resolved one.
        stamp_lineage(entry, parent_call_id)
        await sink.sink.put(entry)
        # Index this call by the conversation a continuation of it would carry, so the next
        # request resolves to it.
        if request_messages is not None:
            lineage.record(
                sink.model_call_id,
                list(request_messages) + [assistant_turn_from_output_items(entry.output_items)],
                cumulative_tokens(entry),
                entry.digest or "",
            )
    except Exception:
        # Capture is best-effort per call: a bad token payload must never fail the
        # model call and break the harness's run. But a rollout that lost a call
        # must not look identical to a complete one, so mark it -- delivery reads
        # the marker and masks the sample rather than training on a hole.
        logger.warning(
            "Training-token capture failed for model call %s of rollout %s.",
            sink.model_call_id,
            sink.rollout_id,
            exc_info=True,
        )
        try:
            sink.store.mark_incomplete(sink.rollout_id, sink.model_call_id)
        except Exception:
            logger.warning("Could not mark rollout %s incomplete.", sink.rollout_id, exc_info=True)
