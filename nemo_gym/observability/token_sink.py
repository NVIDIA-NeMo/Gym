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

"""Served-layer token capture for streaming model calls.

Streaming responses (Anthropic /v1/messages, OpenAI chat SSE) drop token ids on
the wire, so the gate — which only sees the streamed bytes — cannot record them.
But the model server holds the complete response WITH token ids for a moment,
just before it synthesizes the SSE stream. This module lets the gate hand the
model server a per-request "token sink" (via a request-scoped ContextVar, the
same pattern as the MCP session token), so the server can record a TokenEntry
from that complete response.

The sink carries the request_id the gate stamps on the ModelCallRecord for the same
call, so training can join the streamed TokenEntry to its ModelCallRecord. Only the
gate sets a sink (for rollout-tagged, recorded calls), so nothing is captured
for ordinary un-tagged traffic.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Optional

from nemo_gym.observability.capture_store import LocalJsonlCaptureStore, WriteTracker
from nemo_gym.observability.records import TokenEntry


@dataclass
class TokenSink:
    rollout_id: str
    request_id: str
    store: LocalJsonlCaptureStore
    tracker: WriteTracker
    model: str
    enabled: bool


_TOKEN_SINK: ContextVar[Optional[TokenSink]] = ContextVar("nemo_gym_token_sink", default=None)


def set_token_sink(sink: TokenSink) -> Token:
    return _TOKEN_SINK.set(sink)


def reset_token_sink(token: Token) -> None:
    _TOKEN_SINK.reset(token)


def capture_streamed_tokens(payload: dict) -> None:
    """Record a TokenEntry from a complete response, if a rollout token sink is
    active. Called by the model server on the streaming path before SSE
    synthesis. ``payload`` is the complete response as a dict (Responses-style
    ``output`` items or chat ``choices[*].message``); no-op when no token fields
    are present or no sink is set."""
    sink = _TOKEN_SINK.get()
    if sink is None or not sink.enabled:
        return
    # Lazy import avoids an import cycle with the gate module.
    from nemo_gym.observability.capture_gate import extract_token_fields

    info = extract_token_fields(payload or {})
    if info is None:
        return
    entry = TokenEntry(
        rollout_id=sink.rollout_id,
        request_id=sink.request_id,
        prompt_token_ids=info.get("prompt_token_ids") or [],
        generation_token_ids=info.get("generation_token_ids") or [],
        generation_log_probs=info.get("generation_log_probs") or [],
        routed_experts=info.get("routed_experts"),
        model=sink.model or str((payload or {}).get("model") or ""),
    )
    sink.tracker.track(sink.rollout_id, asyncio.to_thread(sink.store.append, entry))
