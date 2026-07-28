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

"""The per-rollout lineage state machine the gate hosts.

``RolloutLineage`` records what the gate knows about one rollout's model
calls: which calls were admitted (and against which committed parent), which
committed (coords ingestion), which failed, and -- at seal -- the token-free
manifest the controller receives. It is deliberately pure: no HTTP, no store,
no token ids (the gate's token buffer lives behind the #2124 store interface,
keyed by the ``call_id``s this machine tracks). That keeps gate hosting thin
and lets the whole lineage contract be unit-tested standalone, two stages
before it touches a server.

Compared to the earlier hash-lineage machine, admission needs no candidate
search and no lease: the incoming request's marker names the parent call
explicitly, the gate resolves and fingerprint-checks it BEFORE admitting, and
each call is committed by the single response that carries its coords. Forks
(two children admitted against one committed parent) are ordinary; identical
siblings are distinct ``call_id``s.

Fail-closed ordering is enforced by construction: a call can only be admitted
against a parent in state ``committed``, and a parent only commits when the
gate ingests the coords riding its response -- after the worker's bytes are
durable in staging.

This module is part of the dependency-free capture core: stdlib + pydantic
(via ``records``) only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional

from nemo_gym.token_id_capture.records import CallRecord, CaptureMode, CommitCoords, RolloutReceipt


class LineageError(Exception):
    """Base for lineage contract violations."""


class DuplicateRolloutError(LineageError):
    """Create-only registration saw an id twice (e.g. a NaN-retry re-dispatch)."""


class UnknownRolloutError(LineageError):
    pass


class UnknownCallError(LineageError):
    pass


class LineageStateError(LineageError):
    """An operation arrived in a state that cannot accept it."""


CallState = Literal["admitted", "committed", "failed"]


@dataclass
class _CallNode:
    call_id: str
    parent_call_id: Optional[str]
    mode: CaptureMode
    prev_len: int
    state: CallState = "admitted"
    updated_at: float = 0.0
    # Set at commit, from the ingested coords.
    cum_len: Optional[int] = None
    delta_len: Optional[int] = None
    digest: Optional[str] = None
    staging_key: Optional[str] = None
    weight_version: Optional[int] = None
    failure_reason: Optional[str] = None


@dataclass(frozen=True)
class AdmittedCall:
    """What admission hands back to the gate: the call identity and the exact
    committed prefix length to serve (``prev_len`` tokens of the parent's
    cumulative sequence; 0 for a root, which serves no prefix)."""

    rollout_id: str
    call_id: str
    parent_call_id: Optional[str]
    mode: CaptureMode
    prev_len: int


class RolloutLineage:
    """State machine for one registered rollout: admit/commit/fail/seal."""

    def __init__(self, rollout_id: str, *, now: Optional[float] = None) -> None:
        timestamp = time.time() if now is None else now
        self.rollout_id = rollout_id
        self.created_at = timestamp
        self.updated_at = timestamp
        self.sealed = False
        self.failure_reason: Optional[str] = None
        self.capture_poisoned = False
        self._nodes: dict[str, _CallNode] = {}
        self._order: list[str] = []

    # -- queries -----------------------------------------------------------

    @property
    def failed(self) -> bool:
        return self.failure_reason is not None

    def call_state(self, call_id: str) -> CallState:
        return self._node(call_id).state

    def committed_parent_len(self, parent_call_id: str) -> int:
        """The cumulative length of a committed call -- the serving rule's
        prefix length for a child admitted against it."""
        node = self._node(parent_call_id)
        if node.state != "committed":
            raise LineageStateError(f"rollout {self.rollout_id}: call {parent_call_id} is {node.state}, not committed")
        assert node.cum_len is not None
        return node.cum_len

    # -- transitions -------------------------------------------------------

    def admit(
        self,
        call_id: str,
        *,
        parent_call_id: Optional[str],
        mode: CaptureMode,
        now: Optional[float] = None,
    ) -> AdmittedCall:
        """Admit one model call. Roots (``parent_call_id=None``) are text-mode
        full renders; children are token-in against a committed parent. The
        gate resolves the marker and checks the fingerprint before calling
        this -- an unresolvable marker admits as a root, never errors here.
        """
        timestamp = self._touch(now)
        if call_id in self._nodes:
            raise LineageStateError(f"rollout {self.rollout_id}: call {call_id} already admitted")
        if parent_call_id is None:
            if mode == "token_in":
                raise LineageStateError(f"rollout {self.rollout_id}: token-in admission requires a committed parent")
            prev_len = 0
        else:
            prev_len = self.committed_parent_len(parent_call_id)
        node = _CallNode(
            call_id=call_id,
            parent_call_id=parent_call_id,
            mode=mode,
            prev_len=prev_len,
            updated_at=timestamp,
        )
        self._nodes[call_id] = node
        self._order.append(call_id)
        return AdmittedCall(
            rollout_id=self.rollout_id,
            call_id=call_id,
            parent_call_id=parent_call_id,
            mode=mode,
            prev_len=prev_len,
        )

    def commit(self, coords: CommitCoords, *, now: Optional[float] = None) -> None:
        """Ingest the coords riding a worker response: the authoritative
        commit. A ``capture_failed`` disposition fails the call and poisons
        the rollout (the completion is still served; the finalizer emits a
        placeholder row)."""
        timestamp = self._touch(now)
        node = self._node(coords.call_id)
        if node.state != "admitted":
            raise LineageStateError(f"rollout {self.rollout_id}: call {coords.call_id} is {node.state}, cannot commit")
        if coords.rollout_id != self.rollout_id:
            raise LineageStateError(f"coords for rollout {coords.rollout_id} ingested at rollout {self.rollout_id}")
        if coords.parent_call_id != node.parent_call_id:
            raise LineageStateError(
                f"rollout {self.rollout_id}: call {coords.call_id} committed parent "
                f"{coords.parent_call_id!r} does not match admitted parent {node.parent_call_id!r}"
            )
        if coords.disposition == "capture_failed":
            node.state = "failed"
            node.failure_reason = "capture_failed"
            node.updated_at = timestamp
            self.capture_poisoned = True
            return
        if coords.delta_len <= 0:
            raise LineageStateError(f"rollout {self.rollout_id}: call {coords.call_id} committed an empty delta")
        if node.prev_len + coords.delta_len != coords.cum_len:
            raise LineageStateError(
                f"rollout {self.rollout_id}: call {coords.call_id} lengths do not chain: "
                f"prev_len={node.prev_len} + delta_len={coords.delta_len} != cum_len={coords.cum_len}"
            )
        node.state = "committed"
        node.delta_len = coords.delta_len
        node.cum_len = coords.cum_len
        node.digest = coords.digest
        node.staging_key = coords.staging_key
        node.weight_version = coords.weight_version
        node.updated_at = timestamp

    def fail_call(self, call_id: str, *, reason: str, now: Optional[float] = None) -> None:
        """Fail one admitted call (e.g. gate call-timeout). The rollout
        survives: no marker was released, so no child can exist."""
        timestamp = self._touch(now)
        node = self._node(call_id)
        if node.state == "committed":
            raise LineageStateError(f"rollout {self.rollout_id}: committed call {call_id} cannot fail")
        node.state = "failed"
        node.failure_reason = reason
        node.updated_at = timestamp

    def fail(self, *, reason: str, now: Optional[float] = None) -> None:
        """Fail the whole rollout (dispatch cancelled, shutdown, TTL)."""
        self._touch(now, allow_sealed=True)
        self.failure_reason = reason

    def seal(
        self,
        *,
        reward: Optional[float],
        terminal_call_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> RolloutReceipt:
        """Seal the rollout and assemble its token-free receipt.

        The manifest lists committed calls in admission order.
        ``terminal_call_id`` defaults to the last committed call; passing an
        uncommitted or unknown id is a contract violation.
        """
        self._touch(now)
        committed = [self._nodes[cid] for cid in self._order if self._nodes[cid].state == "committed"]
        if terminal_call_id is None:
            terminal_call_id = committed[-1].call_id if committed else None
        elif not any(node.call_id == terminal_call_id for node in committed):
            raise LineageStateError(f"rollout {self.rollout_id}: terminal call {terminal_call_id} is not committed")
        self.sealed = True
        manifest = [
            CallRecord(
                call_id=node.call_id,
                parent_call_id=node.parent_call_id,
                delta_len=node.delta_len or 0,
                cum_len=node.cum_len or 0,
                digest=node.digest or "",
                staging_key=node.staging_key or "",
                weight_version=node.weight_version or 0,
                mode=node.mode,
            )
            for node in committed
        ]
        return RolloutReceipt(
            rollout_id=self.rollout_id,
            reward=reward,
            terminal_call_id=terminal_call_id,
            manifest=manifest,
            capture_poisoned=self.capture_poisoned,
            failure_reason=self.failure_reason,
        )

    # -- internals ----------------------------------------------------------

    def _node(self, call_id: str) -> _CallNode:
        node = self._nodes.get(call_id)
        if node is None:
            raise UnknownCallError(f"rollout {self.rollout_id}: unknown call {call_id}")
        return node

    def _touch(self, now: Optional[float], *, allow_sealed: bool = False) -> float:
        if self.sealed and not allow_sealed:
            raise LineageStateError(f"rollout {self.rollout_id} is sealed")
        if self.failure_reason is not None and not allow_sealed:
            raise LineageStateError(f"rollout {self.rollout_id} failed: {self.failure_reason}")
        timestamp = time.time() if now is None else now
        self.updated_at = timestamp
        return timestamp


class LineageRegistry:
    """Create-only registry of in-flight rollouts with a TTL backstop.

    The gate hosts exactly one of these. ``register`` is create-only so a
    re-dispatched batch (e.g. a NaN-logprob retry) fails loudly instead of
    silently splicing two attempts into one lineage. State self-clears at
    ``seal``/``fail_rollout``; ``expire_stale`` sweeps rollouts whose
    registration outlived ``registration_ttl_s`` (the cleanup backstop when a
    controller dies mid-dispatch).
    """

    def __init__(self, *, registration_ttl_s: float) -> None:
        if registration_ttl_s <= 0:
            raise ValueError("registration_ttl_s must be positive")
        self.registration_ttl_s = registration_ttl_s
        self._rollouts: dict[str, RolloutLineage] = {}

    def register(self, rollout_id: str, *, now: Optional[float] = None) -> RolloutLineage:
        if rollout_id in self._rollouts:
            raise DuplicateRolloutError(f"rollout {rollout_id} already registered (create-only)")
        lineage = RolloutLineage(rollout_id, now=now)
        self._rollouts[rollout_id] = lineage
        return lineage

    def get(self, rollout_id: str) -> RolloutLineage:
        lineage = self._rollouts.get(rollout_id)
        if lineage is None:
            raise UnknownRolloutError(f"unknown rollout {rollout_id}")
        return lineage

    def seal(
        self,
        rollout_id: str,
        *,
        reward: Optional[float],
        terminal_call_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> RolloutReceipt:
        """Seal and drop: the receipt is the only thing that survives."""
        receipt = self.get(rollout_id).seal(reward=reward, terminal_call_id=terminal_call_id, now=now)
        del self._rollouts[rollout_id]
        return receipt

    def fail_rollout(self, rollout_id: str, *, reason: str, now: Optional[float] = None) -> None:
        lineage = self.get(rollout_id)
        lineage.fail(reason=reason, now=now)
        del self._rollouts[rollout_id]

    def expire_stale(self, *, now: Optional[float] = None) -> list[str]:
        timestamp = time.time() if now is None else now
        stale = [
            rollout_id
            for rollout_id, lineage in self._rollouts.items()
            if timestamp - lineage.updated_at > self.registration_ttl_s
        ]
        for rollout_id in stale:
            del self._rollouts[rollout_id]
        return stale

    def __len__(self) -> int:
        return len(self._rollouts)

    def __contains__(self, rollout_id: str) -> bool:
        return rollout_id in self._rollouts
