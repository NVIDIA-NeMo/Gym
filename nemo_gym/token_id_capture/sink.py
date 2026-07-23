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

import asyncio
import logging
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Optional

from nemo_gym.token_id_capture.records import TokenEntry, extract_token_fields, response_to_output_items
from nemo_gym.token_id_capture.store import TokenCaptureStore


logger = logging.getLogger(__name__)


@dataclass
class TokenSink:
    rollout_id: str
    model_call_id: str
    store: TokenCaptureStore
    model: str = ""


_TOKEN_SINK: ContextVar[Optional[TokenSink]] = ContextVar("nemo_gym_token_sink", default=None)


def set_token_sink(sink: TokenSink) -> Token:
    return _TOKEN_SINK.set(sink)


def reset_token_sink(token: Token) -> None:
    _TOKEN_SINK.reset(token)


async def capture_tokens(response: Any) -> None:
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
            output_items=response_to_output_items(payload),
            created_at=time.time(),
        )
        await asyncio.to_thread(sink.store.append, entry)
    except Exception:
        # Capture is best-effort; a bad token payload must never fail the model call.
        logger.warning("Training-token capture failed for one model call.", exc_info=True)
