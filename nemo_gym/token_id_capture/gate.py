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

"""The rollout capture gate: custody hosting over the base lineage stack.

The gate is the model-server-side custody layer of gate-authoritative token
capture. It hosts two structures and keeps them honest with each other:

* the **registry** (``staging/lineage.py``) -- the per-rollout custody state
  machine: create-only registration, admit/commit/fail, seal into a token-free
  ``RolloutReceipt``, TTL expiry. This is what guarantees every rollout ends in
  exactly one accounting outcome.
* the base **``LineageIndex``** (``lineage.py``) -- the token memory: each
  committed call's cumulative sequence, indexed by the assistant-turn
  fingerprint of the conversation a continuation would carry. The gate
  constructs the index with training-sized capacity and installs it as the
  process-wide instance (see ``install_lineage_index``), serves prefixes from
  its nodes, feeds it at coords ingestion, and drops rollouts from it at seal.

Identity is the stack's: ``rollout_id`` arrives via the ``/ng-rollout/<id>``
URL prefix (the capture middleware strips it and mints ``model_call_id``);
the gate never mints ids. Parent resolution is the base's content fingerprint
-- there is no marker in this design revision; the fallback *cause* is
classified here so the marker's re-introduction (behind an upstream resolver
seam) stays a data-driven decision.

Not part of the dependency-free staging core (it imports the base capture
package), but still serving-framework-free: no fastapi imports here. HTTP
hosting lives in ``control_routes.py`` and the model-server app.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel

from nemo_gym.token_id_capture.lineage import LineageIndex, LineageNode, assistant_fingerprint
from nemo_gym.token_id_capture.lineage import RolloutLineage as IndexRolloutLineage
from nemo_gym.token_id_capture.records import compute_digest
from nemo_gym.token_id_capture.staging.lineage import (
    LineageRegistry,
    RolloutLineage,
    UnknownRolloutError,
)
from nemo_gym.token_id_capture.staging.records import CommitCoords, RolloutReceipt


logger = logging.getLogger(__name__)


# The identity carrier the gate attaches to every engine-bound request body and
# the coords carrier the worker rides back on its response. ``ng_capture`` is
# the shape of upstream ask #2181-c1; when that seam lands, the constant stays
# and the attach point moves.
NG_CAPTURE_FIELD = "ng_capture"
NG_COMMIT_COORDS_FIELD = "ng_commit_coords"

# Fallback causes, classified at admission (finding A: ambiguity is counted
# separately from day 1 -- it is the marker trigger).
FALLBACK_CAUSES = ("no_history", "no_match", "ambiguous")


class GateError(Exception):
    """A gate-level contract violation (not a per-call fallback)."""


class RolloutGateConfig(BaseModel):
    """Config for hosting the gate inside a model server.

    ``lineage_max_rollouts`` / ``lineage_max_tokens`` size the base
    ``LineageIndex`` from the training config (in-flight rollouts x max
    context). The defaults are the base's eval-sized defaults; a training
    launch MUST override them (finding M: eviction of a live rollout silently
    degrades token-in, so capacity is load-bearing).

    ``control_auth_token`` protects the ``/ng-control/*`` routes (finding S:
    rollout ids are guessable and visible to sandboxed harnesses; an
    unauthenticated seal with a forged reward would win the state
    transition). Required when the gate is enabled.
    """

    enabled: bool = False
    registration_ttl_s: float = 3600.0
    lineage_max_rollouts: int = 512
    lineage_max_tokens: int = 8_000_000
    control_auth_token: Optional[str] = None


@dataclass(frozen=True)
class GateDecision:
    """The serving-rule outcome for one admitted call."""

    rollout_id: str
    call_id: str
    mode: str  # "token_in" | "text"
    parent_call_id: Optional[str]
    prev_len: int
    prefix_ids: Optional[list[int]]
    fallback_reason: Optional[str] = None

    def capture_context(self) -> dict[str, Any]:
        """The ``ng_capture`` payload the worker keys its capture on."""
        return {
            "rollout_id": self.rollout_id,
            "call_id": self.call_id,
            "parent_call_id": self.parent_call_id,
            "prev_len": self.prev_len,
            "mode": self.mode,
        }


class _CountingLineageIndex(LineageIndex):
    """The base index plus an eviction counter (finding M: eviction of a live
    rollout must be loud, and the base evicts silently)."""

    def __init__(self, max_rollouts: int, max_tokens: int) -> None:
        super().__init__(max_rollouts=max_rollouts, max_tokens=max_tokens)
        self.evictions = 0

    def _evict(self) -> None:
        before = len(self)
        super()._evict()
        self.evictions += before - len(self)


class RolloutCaptureGate:
    """Admission, prefix serving, coords ingestion, and receipts for one
    model server. Single-owner; all methods are synchronous dict/list
    operations, atomic under the server's event loop."""

    def __init__(
        self,
        *,
        registration_ttl_s: float = 3600.0,
        lineage_max_rollouts: int = 512,
        lineage_max_tokens: int = 8_000_000,
    ) -> None:
        self._registry = LineageRegistry(registration_ttl_s=registration_ttl_s)
        self._index = _CountingLineageIndex(max_rollouts=lineage_max_rollouts, max_tokens=lineage_max_tokens)
        self.metrics: dict[str, int] = {
            "registered": 0,
            "token_in": 0,
            "fallback_no_history": 0,
            "fallback_no_match": 0,
            "fallback_ambiguous": 0,
            "capture_failed": 0,
            "sealed": 0,
            "failed_rollouts": 0,
            "expired_rollouts": 0,
            # Model calls that reached a gate-enabled server with no
            # /ng-rollout/ correlation at all: served untouched (eval
            # traffic, pre-registration side calls), but counted -- silent
            # unattributed traffic on a training server is worth noticing.
            "unattributed_calls": 0,
        }
        self._warned_unattributed = False

    def note_unattributed_call(self) -> None:
        """Count an uncorrelated call passing through a gate-enabled server."""
        self.metrics["unattributed_calls"] += 1
        if not self._warned_unattributed:
            self._warned_unattributed = True
            logger.warning(
                "A model call with no /ng-rollout/ correlation reached a gate-enabled server; "
                "it is served on the legacy path and its tokens are not captured. "
                "Counted under gate metric unattributed_calls (warning shown once)."
            )

    @classmethod
    def from_config(cls, config: RolloutGateConfig) -> "RolloutCaptureGate":
        return cls(
            registration_ttl_s=config.registration_ttl_s,
            lineage_max_rollouts=config.lineage_max_rollouts,
            lineage_max_tokens=config.lineage_max_tokens,
        )

    def install_lineage_index(self) -> None:
        """Install the gate's capacity-sized index as the process-wide one.

        Workaround for upstream ask #2180-c1 (``set_lineage_index``): the base
        exposes only a module-global ``sink._LINEAGE`` built with eval-sized
        defaults, and ``_apply_prefix_supply`` / ``capture_tokens`` read it via
        ``lineage_index()``. Replacing the module global before the first
        request is the documented workaround; the attribute name is pinned in
        the migration log's bump checklist. Collapses to one setter call when
        the ask lands.
        """
        from nemo_gym.token_id_capture import sink as _sink  # deferred: avoid import cycle at module load

        _sink._LINEAGE = self._index

    # -- control plane -------------------------------------------------------

    def register_rollout(self, rollout_id: str) -> None:
        """Create-only (a NaN-retry re-dispatch fails loudly, § 7)."""
        self._registry.register(rollout_id)
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
        self._index.drop(rollout_id)
        self.metrics["sealed"] += 1
        return receipt

    def fail_rollout(self, rollout_id: str, *, reason: str) -> None:
        self._registry.fail_rollout(rollout_id, reason=reason)
        self._index.drop(rollout_id)
        self.metrics["failed_rollouts"] += 1

    def expire_stale(self) -> list[str]:
        stale = self._registry.expire_stale()
        for rollout_id in stale:
            self._index.drop(rollout_id)
        self.metrics["expired_rollouts"] += len(stale)
        return stale

    # -- data plane (per model call) ------------------------------------------

    def prepare_call(self, rollout_id: str, call_id: str, messages: list[dict[str, Any]]) -> GateDecision:
        """Apply the serving rule and admit one call.

        ``call_id`` is the middleware-minted ``model_call_id``. The parent is
        the base fingerprint resolution over the gate's index; a miss or an
        ambiguity admits a text-mode root (never wrong, only a cold cache),
        with the cause classified for the metrics. Raises
        ``UnknownRolloutError`` for unregistered ids (the controller registers
        every rollout before dispatch, so an unknown id is a contract
        violation, not a fallback) and ``LineageStateError`` for sealed/failed
        rollouts.
        """
        lineage = self._registry.get(rollout_id)
        index_lineage = self._index.for_rollout(rollout_id)
        parent, cause = self._resolve_with_cause(index_lineage, messages)

        if parent is not None:
            admitted = lineage.admit(call_id, parent_call_id=parent.call_id, mode="token_in")
            self.metrics["token_in"] += 1
            return GateDecision(
                rollout_id=rollout_id,
                call_id=admitted.call_id,
                mode="token_in",
                parent_call_id=parent.call_id,
                prev_len=admitted.prev_len,
                prefix_ids=list(parent.cum_tokens),
            )
        admitted = lineage.admit(call_id, parent_call_id=None, mode="text")
        if cause is not None:
            self.metrics[f"fallback_{cause}"] += 1
            if cause == "no_match":
                # Where does the request first diverge from recorded lineage?
                # depth 0 every time = systematic per-turn drift (e.g. the F5
                # reasoning asymmetry); depth k>0 = localized rewrite, the
                # longest-prefix-matching opportunity.
                matched, incoming = index_lineage.divergence_depth(messages)
                bucket = str(matched) if matched < 5 else "5plus"
                key = f"no_match_matched_depth_{bucket}"
                self.metrics[key] = self.metrics.get(key, 0) + 1
                self.metrics["no_match_incoming_turns"] = (
                    self.metrics.get("no_match_incoming_turns", 0) + incoming
                )
        return GateDecision(
            rollout_id=rollout_id,
            call_id=admitted.call_id,
            mode="text",
            parent_call_id=None,
            prev_len=0,
            prefix_ids=None,
            fallback_reason=cause,
        )

    def _resolve_with_cause(
        self, index_lineage: IndexRolloutLineage, messages: list[dict[str, Any]]
    ) -> tuple[Optional[LineageNode], Optional[str]]:
        """The base ``RolloutLineage.resolve`` with the miss cause classified.

        The base returns ``None`` for both "nothing matches" and "two calls
        match" (finding A); the causes are split here because ambiguity on
        byte-identical assistant turns is routine in agentic loops and is the
        Stage-2 marker trigger. A rollout's true first call (nothing recorded
        yet) is not a fallback at all.
        """
        if not index_lineage.by_call_id:
            return None, None
        fingerprint = assistant_fingerprint(messages)
        if not fingerprint:
            return None, "no_history"
        call_ids = index_lineage.by_fingerprint.get(fingerprint) or []
        if not call_ids:
            return None, "no_match"
        if len(call_ids) > 1:
            return None, "ambiguous"
        node = index_lineage.by_call_id.get(call_ids[0])
        if node is None:
            return None, "no_match"
        return node, None

    def ingest_coords(
        self,
        coords: CommitCoords,
        *,
        request_messages: list[dict[str, Any]],
        served_turn: dict[str, Any],
    ) -> bool:
        """The authoritative commit for one call (the #2124-c3 seam,
        implemented here until ``commit_entry`` lands upstream).

        Commits the coords in the registry, then indexes the call in the base
        ``LineageIndex`` under the fingerprint of (request history + served
        assistant turn) -- exactly what a continuation's history echoes -- with
        the cumulative tokens a child must be served. Returns ``False`` when
        capture failed (the completion is still served; the rollout is
        poisoned and the call is not indexed, so no child can chain onto it).
        """
        lineage = self._registry.get(coords.rollout_id)
        lineage.commit(coords)
        if coords.disposition == "capture_failed":
            self.metrics["capture_failed"] += 1
            return False
        index_lineage = self._index.for_rollout(coords.rollout_id)
        if coords.parent_call_id is not None:
            parent = index_lineage.by_call_id.get(coords.parent_call_id)
            if parent is None:
                # Evicted between admission and commit (finding M). The
                # registry commit above already holds; the index just cannot
                # extend a dropped sequence, so descendants fall back.
                return True
            cum_tokens = list(parent.cum_tokens) + list(coords.token_ids_delta)
        else:
            cum_tokens = list(coords.token_ids_delta)
        index_lineage.record(
            coords.call_id,
            [*request_messages, served_turn],
            cum_tokens,
            compute_digest(cum_tokens),
        )
        return True

    def fail_call(self, rollout_id: str, call_id: str, *, reason: str) -> None:
        """Fail one admitted call (timeout / lost response). The rollout
        survives: the call was never indexed, so no child can chain onto it."""
        try:
            self._registry.get(rollout_id).fail_call(call_id, reason=reason)
        except UnknownRolloutError:
            # The rollout was already sealed/failed/expired underneath the
            # in-flight call; nothing left to mark.
            pass

    # -- introspection ---------------------------------------------------------

    def lineage(self, rollout_id: str) -> RolloutLineage:
        return self._registry.get(rollout_id)

    def snapshot_metrics(self) -> dict[str, int]:
        """Counters plus the index gauges (finding M: eviction must be loud)."""
        return {
            **self.metrics,
            "lineage_rollouts": len(self._index),
            "lineage_tokens": self._index.total_tokens,
            "lineage_evictions": self._index.evictions,
        }


__all__ = [
    "FALLBACK_CAUSES",
    "GateDecision",
    "GateError",
    "NG_CAPTURE_FIELD",
    "NG_COMMIT_COORDS_FIELD",
    "RolloutCaptureGate",
    "RolloutGateConfig",
]
