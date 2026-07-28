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

"""The gate: thin hosting of the pure lineage machine inside the model server.

Everything that requires understanding *messages* lives here — and nothing
else does. Per model call the model server drives two hooks:

* ``prepare_call`` — resolve the incoming history's ``ng_call_id`` marker to
  its committed parent, check the recorded fingerprint, and apply the
  serving rule (§ 3.3 of the gate-authoritative design):

      Serve token-in only when the request carries a unique, known marker
      AND the history up to the marker matches the gate's fingerprint for
      that call. Anything else falls back to text mode as a new root —
      correct but wasteful, never silently wrong.

  Returns the admitted call's identity plus the exact ``prefix_ids`` to
  serve (token-in) or ``None`` (text render).
* ``ingest_coords`` — the coords riding the worker's response ARE the
  authoritative commit: extend the rollout's token buffer with the delta
  ids, record the served history's fingerprint for future children, and
  hand back the marker to attach to the assistant message.

The gate holds token ids in flight (``MemoryRolloutTokenBuffer``) and never
logprobs; it never writes the framework's staging storage. State self-clears
at seal / ``fail_rollout`` / registration TTL.

Marker and context carriers (part of the frozen serving rule, § 3.1-3.2):
``NG_CALL_ID_FIELD`` rides assistant messages agent-side (replacing today's
token-array attachment, same carrier); ``NG_CAPTURE_FIELD`` rides the
gate->worker request with the call identity the worker's capture layer needs;
``NG_COMMIT_COORDS_FIELD`` rides the worker->gate response with the
``CommitCoords``.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel

from nemo_gym.token_id_capture.memory_store import MemoryRolloutTokenBuffer
from nemo_gym.token_id_capture.staging.lineage import (
    LineageError,
    LineageRegistry,
    RolloutLineage,
    UnknownRolloutError,
)
from nemo_gym.token_id_capture.staging.records import CaptureMode, CommitCoords, RolloutReceipt


NG_CALL_ID_FIELD = "ng_call_id"
NG_CAPTURE_FIELD = "ng_capture"
NG_COMMIT_COORDS_FIELD = "ng_commit_coords"
NG_ROLLOUT_ID_METADATA_KEY = "ng_rollout_id"

# Message keys that must never participate in fingerprints: capture-plane
# carriers and the legacy token-echo fields.
_CAPTURE_KEYS = (
    NG_CALL_ID_FIELD,
    "prompt_token_ids",
    "generation_token_ids",
    "generation_log_probs",
    "routed_experts",
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class GateError(Exception):
    """A gate contract violation that maps to a request rejection."""


class RolloutGateConfig(BaseModel):
    """Model-server config block enabling gate-authoritative capture.

    Dormant by default; mutually exclusive with the legacy token echo
    (``return_token_id_information``) — enforced loudly at server setup.
    """

    enabled: bool = False
    registration_ttl_s: float = 3600.0


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
        return "".join(parts)
    return str(content)


def normalize_message(message: dict[str, Any]) -> list[Any]:
    """One message's fingerprint contribution.

    Deliberately coarse: role, textual content (assistant reasoning stripped —
    history renders drop ``<think>`` blocks, cf. Qwen3), tool-call
    name/argument pairs, and the tool linkage id. Capture carriers and token
    fields never participate. Too-strict inflates fallbacks; too-loose misses
    history edits — both stay *correct* because mismatch only falls back to a
    text-mode root (§ 3.3); the definition is pinned by the S3 conformance
    tests.
    """
    role = str(message.get("role") or message.get("type") or "")
    text = _content_text(message.get("content"))
    if role == "assistant":
        text = _THINK_RE.sub("", text)
    tool_calls = []
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        tool_calls.append([str(function.get("name") or ""), str(function.get("arguments") or "")])
    return [role, text, tool_calls, str(message.get("tool_call_id") or "")]


def message_fingerprint(messages: list[dict[str, Any]]) -> str:
    """Order-sensitive fingerprint of a normalized message history."""
    payload = json.dumps([normalize_message(m) for m in messages], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(b"ng-history-fp-v1" + payload.encode("utf-8")).hexdigest()


def find_marker(messages: list[dict[str, Any]]) -> tuple[Optional[str], Optional[int]]:
    """The deepest assistant-borne marker and its message index.

    The *last* marker names the parent: a sub-agent forked from turn-1
    history carries turn-1's marker and resolves to that interior node
    exactly (§ 3.1)."""
    for index in reversed(range(len(messages))):
        marker = messages[index].get(NG_CALL_ID_FIELD)
        if marker:
            return str(marker), index
    return None, None


@dataclass(frozen=True)
class GateDecision:
    """What ``prepare_call`` hands the model server for one admitted call."""

    rollout_id: str
    call_id: str
    mode: CaptureMode
    parent_call_id: Optional[str]
    prev_len: int
    # Exact committed prefix to serve (token-in), None for a text render.
    prefix_ids: Optional[list[int]]
    # Why token-in did not apply (None on the happy path or a true first call).
    fallback_reason: Optional[str] = None

    def capture_context(self) -> dict[str, Any]:
        """The ``NG_CAPTURE_FIELD`` payload the worker's capture layer reads."""
        return {
            "rollout_id": self.rollout_id,
            "call_id": self.call_id,
            "parent_call_id": self.parent_call_id,
            "prev_len": self.prev_len,
            "mode": self.mode,
        }


class RolloutCaptureGate:
    """Admission, prefix serving, coords ingestion, and receipts for one
    model server. Single-owner; all methods are synchronous dict/list
    operations, atomic under the server's event loop."""

    def __init__(self, *, registration_ttl_s: float = 3600.0) -> None:
        self._registry = LineageRegistry(registration_ttl_s=registration_ttl_s)
        self._buffer = MemoryRolloutTokenBuffer()
        # rollout_id -> call_id -> fingerprint of (request history + served
        # assistant message), recorded at commit for the serving rule.
        self._fingerprints: dict[str, dict[str, str]] = {}
        self.metrics: dict[str, int] = {
            "registered": 0,
            "token_in": 0,
            "fallback_no_marker": 0,
            "fallback_unknown_marker": 0,
            "fallback_fingerprint_miss": 0,
            "capture_failed": 0,
            "sealed": 0,
            "failed_rollouts": 0,
            "expired_rollouts": 0,
        }

    # -- control plane -------------------------------------------------------

    def register_rollout(self, rollout_id: str) -> None:
        """Create-only (a NaN-retry re-dispatch fails loudly, § 7)."""
        self._registry.register(rollout_id)
        self._buffer.register(rollout_id)
        self._fingerprints[rollout_id] = {}
        self.metrics["registered"] += 1

    def is_registered(self, rollout_id: str) -> bool:
        return rollout_id in self._registry

    def seal_rollout(
        self,
        rollout_id: str,
        *,
        reward: Optional[float],
        terminal_call_id: Optional[str] = None,
    ) -> RolloutReceipt:
        """Seal -> token-free receipt; the gate drops every byte of state."""
        receipt = self._registry.seal(rollout_id, reward=reward, terminal_call_id=terminal_call_id)
        self._drop_state(rollout_id)
        self.metrics["sealed"] += 1
        return receipt

    def fail_rollout(self, rollout_id: str, *, reason: str) -> None:
        self._registry.fail_rollout(rollout_id, reason=reason)
        self._drop_state(rollout_id)
        self.metrics["failed_rollouts"] += 1

    def expire_stale(self) -> list[str]:
        stale = self._registry.expire_stale()
        for rollout_id in stale:
            self._drop_state(rollout_id)
        self.metrics["expired_rollouts"] += len(stale)
        return stale

    # -- data plane (per model call) ------------------------------------------

    def prepare_call(self, rollout_id: str, messages: list[dict[str, Any]]) -> GateDecision:
        """Apply the serving rule and admit one call.

        Raises ``UnknownRolloutError`` for unregistered ids (the SC registers
        every rollout before dispatch, so an unknown id is a contract
        violation, not a fallback) and ``LineageStateError`` for
        sealed/failed rollouts.
        """
        lineage = self._registry.get(rollout_id)
        marker, marker_index = find_marker(messages)

        fallback_reason: Optional[str] = None
        if marker is None:
            # A rollout's true first call has no marker; only later calls
            # count as fallbacks in the metrics.
            fallback_reason = None if not self._fingerprints.get(rollout_id) else "no_marker"
        elif not self._buffer.has_call(rollout_id, marker):
            fallback_reason = "unknown_marker"
        else:
            recorded = self._fingerprints.get(rollout_id, {}).get(marker)
            observed = message_fingerprint([m for m in messages[: marker_index + 1]])
            if recorded is None or observed != recorded:
                fallback_reason = "fingerprint_miss"

        call_id = uuid.uuid4().hex
        if marker is not None and fallback_reason is None:
            admitted = lineage.admit(call_id, parent_call_id=marker, mode="token_in")
            prefix_ids = self._buffer.cumulative_ids(rollout_id, marker)
            self.metrics["token_in"] += 1
            return GateDecision(
                rollout_id=rollout_id,
                call_id=admitted.call_id,
                mode="token_in",
                parent_call_id=marker,
                prev_len=admitted.prev_len,
                prefix_ids=prefix_ids,
            )
        admitted = lineage.admit(call_id, parent_call_id=None, mode="text")
        if fallback_reason is not None:
            self.metrics[f"fallback_{fallback_reason}"] += 1
        return GateDecision(
            rollout_id=rollout_id,
            call_id=admitted.call_id,
            mode="text",
            parent_call_id=None,
            prev_len=0,
            prefix_ids=None,
            fallback_reason=fallback_reason,
        )

    def ingest_coords(
        self,
        coords: CommitCoords,
        *,
        request_messages: list[dict[str, Any]],
        served_message: dict[str, Any],
    ) -> Optional[str]:
        """The authoritative commit for one call.

        Extends the token buffer with the delta ids, records the fingerprint
        of (request history + served assistant message) — exactly what a
        child's history up to the marker echoes — and returns the marker to
        attach, or ``None`` when capture failed (the completion is still
        served; the rollout is poisoned and no marker is released).
        """
        lineage = self._registry.get(coords.rollout_id)
        lineage.commit(coords)
        if coords.disposition == "capture_failed":
            self.metrics["capture_failed"] += 1
            return None
        self._buffer.add_call(
            coords.rollout_id,
            coords.call_id,
            parent_call_id=coords.parent_call_id,
            token_ids_delta=coords.token_ids_delta,
        )
        fingerprint = message_fingerprint([*request_messages, served_message])
        self._fingerprints[coords.rollout_id][coords.call_id] = fingerprint
        return coords.call_id

    def fail_call(self, rollout_id: str, call_id: str, *, reason: str) -> None:
        """Fail one admitted call (timeout / lost response). The rollout
        survives: no marker was released, so no child can exist."""
        try:
            self._registry.get(rollout_id).fail_call(call_id, reason=reason)
        except UnknownRolloutError:
            # The rollout was already sealed/failed/expired underneath the
            # in-flight call; nothing left to mark.
            pass

    # -- internals -------------------------------------------------------------

    def lineage(self, rollout_id: str) -> RolloutLineage:
        return self._registry.get(rollout_id)

    def _drop_state(self, rollout_id: str) -> None:
        self._buffer.drop(rollout_id)
        self._fingerprints.pop(rollout_id, None)


__all__ = [
    "GateDecision",
    "GateError",
    "LineageError",
    "NG_CALL_ID_FIELD",
    "NG_CAPTURE_FIELD",
    "NG_COMMIT_COORDS_FIELD",
    "NG_ROLLOUT_ID_METADATA_KEY",
    "RolloutCaptureGate",
    "find_marker",
    "message_fingerprint",
    "normalize_message",
]
