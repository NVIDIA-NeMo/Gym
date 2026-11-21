# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The model calls of one game and what they say about its context: chains, drains and helper calls.

The transport records every call of a game here (``GameCallLog.add``): its role, its history id, the response id
the model server returned, a digest of every request message, the game's step and level when the call was sent (when
the transport knows them), and the error of a call that failed. The harness marks its own calls: ``gameplay`` calls
play the game, ``helper`` calls write text the harness keeps (the Kaggle kit harness's ``drain-summary``,
``level-summary``, ``rolling-summary``, ``turn-compaction``; the official harness's ``harness-summary``).

:func:`call_log_entries` is the log as JSON (the rollout's ``agent_call_log``): one entry per call sent, in order.
:func:`harness_log_entries` is the harness's own log lines of the game (the rollout's ``harness_log``): the transport
records every warning the harness logs on the game's thread (``GameCallLog.add_log``), e.g. a drain summary that was
answered but unusable, which no call shows as failed.

:func:`observations` turns the log into the rollout's ``ng_agent_observations``:

- one ``AgentInvocation`` whose ``model_calls`` are the gameplay calls in order (the accepted calls, the references
  a training framework joins to its captured calls to rebuild every context chain);
- one ``ContextCompactionObservation`` per drain: a gameplay request that no longer starts with the previous
  gameplay request (history was dropped), with the summary text the harness put in its place, the call before and
  after it, and the helper calls that wrote the summary.

A gameplay request that rewrites its history without dropping any of it (for example, new level lessons in an
earlier message) also starts a new chain; it is reported as a ``history_rewritten`` gap, not as a compaction.
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from nemo_gym.config import ModelServerRef
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
)


GAMEPLAY_ROLE = "gameplay"
# Helper calls that write the text a drain keeps in place of the dropped history (Kaggle kit harness; official harness).
COMPACTION_HISTORY_IDS = frozenset({"drain-summary", "turn-compaction", "rolling-summary", "harness-summary"})
# The harness's drain summary message (tool_agent.py, turn-compaction patch v5).
DRAIN_SUMMARY_PREFIX = "[Summary of earlier history"
# The official harness's compaction bridge (benchmarking/compaction.py SUMMARY_BRIDGE_TEMPLATE).
OFFICIAL_SUMMARY_PREFIX = "Earlier conversation history was compacted."
DRAIN_SUMMARY_PREFIXES = (DRAIN_SUMMARY_PREFIX, OFFICIAL_SUMMARY_PREFIX)
SUMMARY_MAX_CHARS = 20000
# The harness log kept per game: the first lines, each cut to its first characters (the rest is counted as dropped).
HARNESS_LOG_MAX_LINES = 2000
HARNESS_LOG_LINE_CHARS = 1000


def message_digest(message: Any) -> str:
    return hashlib.sha256(json.dumps(message, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _drain_summary(messages: list[dict[str, Any]]) -> str | None:
    for message in reversed(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if message.get("role") == "user" and isinstance(content, str) and content.startswith(DRAIN_SUMMARY_PREFIXES):
            return content[:SUMMARY_MAX_CHARS]
    return None


@dataclass
class ModelCall:
    index: int
    role: str
    history_id: str
    status: int
    response_id: str | None
    request_digests: list[str]
    prompt_tokens: int | None = None
    drain_summary: str | None = None
    # When the call was sent; observed_at = when its answer (or failure) came back.
    started_at: float | None = None
    observed_at: float = field(default_factory=time.time)
    # The game when the call was sent: step = actions taken + 1 (the Kaggle kit harness's "step"), level = levels
    # completed + 1 (the level on the board). None when the transport does not know them.
    step: int | None = None
    level: int | None = None
    # Why a call got no answer (timeout, connection error, cancelled); status is then 0.
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.status >= 400 or self.error is not None


class GameCallLog:
    """Thread-safe log of one game's model calls, in the order they were sent."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[ModelCall] = []
        self.log_lines: list[dict[str, Any]] = []
        self.log_lines_dropped = 0

    def add(
        self,
        *,
        role: str,
        history_id: str,
        messages: list[dict[str, Any]],
        status: int,
        response: dict[str, Any] | None,
        started_at: float | None = None,
        step: int | None = None,
        level: int | None = None,
        error: str | None = None,
    ) -> None:
        usage = (response or {}).get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        with self._lock:
            self.calls.append(
                ModelCall(
                    index=len(self.calls),
                    role=role,
                    history_id=history_id,
                    status=status,
                    response_id=(response or {}).get("id") if 0 < status < 400 and error is None else None,
                    request_digests=[message_digest(m) for m in messages],
                    prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) and prompt_tokens >= 0 else None,
                    drain_summary=_drain_summary(messages) if role == GAMEPLAY_ROLE else None,
                    started_at=started_at,
                    step=step,
                    level=level,
                    error=error,
                )
            )

    def add_log(
        self,
        *,
        logged_at: float,
        severity: str,
        logger: str,
        message: str,
        step: int | None = None,
        level: int | None = None,
    ) -> None:
        """One line the harness logged while it played this game, with the game's step and level then."""
        with self._lock:
            if len(self.log_lines) >= HARNESS_LOG_MAX_LINES:
                self.log_lines_dropped += 1
                return
            self.log_lines.append(
                {
                    "logged_at": logged_at,
                    "step": step,
                    "level": level,
                    "severity": severity,
                    "logger": logger,
                    "message": message[:HARNESS_LOG_LINE_CHARS],
                }
            )


def harness_log_entries(log: GameCallLog) -> dict[str, Any]:
    """The game's harness log lines as JSON: ``lines`` (``logged_at``, ``step``, ``level``, ``severity``, ``logger``,
    ``message``, in order) and ``dropped`` (lines over ``HARNESS_LOG_MAX_LINES``)."""
    return {"lines": list(log.log_lines), "dropped": log.log_lines_dropped}


def model_unavailable(log: GameCallLog) -> str | None:
    """Why the game did not measure the model, when no model call of it got an answer (an unreachable or failing
    endpoint: every call an HTTP error or no answer): the calls sent and the last failure. None when a call succeeded
    or none was sent."""
    calls = list(log.calls)
    if not calls or not all(c.failed for c in calls):
        return None
    last = calls[-1]
    return f"every model call of the game failed ({len(calls)} calls; last: {last.error or f'HTTP {last.status}'})"


def call_log_entries(log: GameCallLog) -> list[dict[str, Any]]:
    """The log as JSON, one entry per call sent, in order: ``index``, ``role`` (gameplay / helper), ``history_id``,
    ``status`` (HTTP; 0 when no answer came), ``response_id`` (the model server's, None for a failed call),
    ``error``, ``step`` and ``level`` (the game when the call was sent), ``started_at`` / ``completed_at`` (Unix
    seconds, this host's clock) and ``prompt_tokens``."""
    return [
        {
            "index": c.index,
            "role": c.role,
            "history_id": c.history_id,
            "status": c.status,
            "response_id": c.response_id,
            "error": c.error,
            "step": c.step,
            "level": c.level,
            "started_at": c.started_at,
            "completed_at": c.observed_at,
            "prompt_tokens": c.prompt_tokens,
        }
        for c in list(log.calls)
    ]


def observations(
    log: GameCallLog,
    *,
    invocation_id: str,
    model_ref: ModelServerRef,
    status: str,
    stop_reason: str | None,
    source: str = "arc_agi_3_agent",
) -> AgentObservationBundle:
    """The rollout's agent observations (see the module docstring)."""

    def ref(call: ModelCall) -> ModelCallRef | None:
        return ModelCallRef(model_ref=model_ref, response_id=call.response_id) if call.response_id else None

    calls = list(log.calls)
    records: list[Any] = []
    gaps: list[ObservationGap] = []
    gameplay = [c for c in calls if c.role == GAMEPLAY_ROLE and c.response_id]
    for call in calls:
        if call.failed:
            reason = f"HTTP {call.status}" if call.error is None else call.error
            gaps.append(
                ObservationGap(
                    code="model_call_failed",
                    invocation_id=invocation_id,
                    detail=f"call {call.index} ({call.role}/{call.history_id}): {reason}",
                )
            )
    for previous, current in zip(gameplay, gameplay[1:]):
        prefix = previous.request_digests
        if current.request_digests[: len(prefix)] == prefix:
            continue  # the same chain: the request extends the previous one
        helpers = [
            c
            for c in calls
            if previous.index < c.index < current.index
            and c.role != GAMEPLAY_ROLE
            and c.history_id in COMPACTION_HISTORY_IDS
            and c.response_id
        ]
        dropped = len(current.request_digests) <= len(prefix) or current.drain_summary is not None or helpers
        if not dropped:
            gaps.append(
                ObservationGap(
                    code="history_rewritten",
                    invocation_id=invocation_id,
                    detail=f"gameplay call {current.index} rewrote earlier history without dropping it",
                )
            )
            continue
        records.append(
            ContextCompactionObservation(
                invocation_id=invocation_id,
                source_model_ref=model_ref,
                observed_at=current.observed_at,
                trigger="automatic",
                tokens_before=previous.prompt_tokens,
                tokens_after=current.prompt_tokens,
                outcome="completed",
                summary=current.drain_summary,
                before_model_call=ref(previous),
                model_calls=[r for r in (ref(c) for c in helpers) if r is not None],
                after_model_call=ref(current),
            )
        )
    records.insert(
        0,
        AgentInvocation(
            invocation_id=invocation_id,
            status=status,
            stop_reason=stop_reason,
            model_calls=[r for r in (ref(c) for c in gameplay) if r is not None],
        ),
    )
    return AgentObservationBundle(source=source, records=records, gaps=gaps)
