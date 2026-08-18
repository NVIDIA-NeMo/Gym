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
from dataclasses import dataclass
from typing import Any

from nemo_gym.token_id_capture.protocols import TokenSink
from nemo_gym.token_id_capture.records import (
    TokenEntry,
    extract_token_fields,
    response_to_output_items,
    strip_token_fields,
)


logger = logging.getLogger(__name__)


@dataclass
class CaptureContext:
    """Describe one in-flight training-token capture.

    The context identifies the rollout and model call.
    ``sink`` receives the resulting record.
    A framework may provide any ``TokenSink`` implementation.
    """

    rollout_id: str
    model_call_id: str
    # ``None`` means another process owns record staging.
    # The context still carries the capture identity.
    sink: TokenSink | None
    model: str = ""
    # ``commit_entry`` sets this after another capture path records the call.
    committed: bool = False


_TOKEN_SINK: ContextVar[CaptureContext | None] = ContextVar("nemo_gym_token_sink", default=None)


def set_token_sink(sink: CaptureContext) -> Token:
    return _TOKEN_SINK.set(sink)


def current_capture_context() -> CaptureContext | None:
    """Return the capture context for the in-flight call.

    Return ``None`` for untagged traffic.
    Framework inference workers use this identity for staged records.
    """
    return _TOKEN_SINK.get()


def reset_token_sink(token: Token) -> None:
    _TOKEN_SINK.reset(token)


async def capture_tokens(response: Any) -> None:
    """Record a ``TokenEntry`` from a complete model response.

    Accept a Pydantic model or dictionary.
    Return without work when no capture context exists.
    Mark local capture incomplete when required token ids are absent.
    Await the write before the model call returns.
    """
    sink = _TOKEN_SINK.get()
    if sink is None:
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
            await _capture_missing(sink, f"the response is a {type(response).__name__}")
            return
        info = extract_token_fields(payload)
        if info is None:
            await _capture_missing(sink, "the response carries no token ids")
            return
        # Content only: the arrays live on the entry, not on the items as well.
        content_items, token_item_index = strip_token_fields(response_to_output_items(payload))

        entry = TokenEntry(
            rollout_id=sink.rollout_id,
            model_call_id=sink.model_call_id,
            model=sink.model or str(payload.get("model") or ""),
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
        await _capture_failed(sink, "build")
        return
    await commit_entry(entry)


async def commit_entry(entry: TokenEntry) -> None:
    """Durably record a finished entry against the in-flight call.

    ``capture_tokens`` extracts arrays from a served response.
    Engine-side capture may already have those arrays.
    Engine-side callers can use this method directly.
    Return without work when no capture context exists.
    Capture failures mark the rollout incomplete.
    This method never fails the model call.
    """
    sink = _TOKEN_SINK.get()
    if sink is None:
        return
    if entry.rollout_id != sink.rollout_id or entry.model_call_id != sink.model_call_id:
        logger.warning(
            "Training-token capture identity mismatch for model call %s of rollout %s.",
            sink.model_call_id,
            sink.rollout_id,
        )
        await _mark_incomplete(sink)
        return
    if sink.sink is None:
        sink.committed = True
        return
    try:
        await sink.sink.put(entry)
        sink.committed = True
    except Exception:
        await _capture_failed(sink, "write")


async def _capture_failed(sink: CaptureContext, stage: str) -> None:
    """Report a capture failure without letting it reach the model call.

    Bad token payloads must not fail the model call.
    Mark the rollout so consumers can mask the sample.
    Call this only from an ``except`` block.
    """
    logger.warning(
        "Training-token capture failed to %s the record for model call %s of rollout %s.",
        stage,
        sink.model_call_id,
        sink.rollout_id,
        exc_info=True,
    )
    await _mark_incomplete(sink)


async def _capture_missing(sink: CaptureContext, reason: str) -> None:
    """Mark the rollout when a call this process should have recorded produced nothing.

    A response with no token ids is a hole in the chain rather than traffic to skip.
    The builder reads the gap between one call's tokens and the next call's prompt as tool output.
    A skipped call's generated tokens then enter the next prompt with mask zero.
    Policy tokens would train as if the environment produced them.

    Two cases are not holes and are left alone.
    A committed call was recorded by another capture path.
    A context without a sink delegates completeness to external staging.
    """
    if sink.committed or sink.sink is None:
        return
    logger.warning(
        "Training-token capture has no token ids for model call %s of rollout %s: %s.",
        sink.model_call_id,
        sink.rollout_id,
        reason,
    )
    await _mark_incomplete(sink)


async def _mark_incomplete(sink: CaptureContext) -> None:
    """Mark the rollout, or say loudly why it could not be marked.

    A missing ``mark_incomplete`` method can hide incomplete capture.
    Log that condition as an error.
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
        await mark(sink.rollout_id, sink.model_call_id)
    except Exception:
        logger.warning("Could not mark rollout %s incomplete.", sink.rollout_id, exc_info=True)
