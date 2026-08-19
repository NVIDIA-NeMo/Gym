# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-process rollout gate for worker-owned token staging."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.token_id_capture.protocols import LineageStore
from nemo_gym.token_id_capture.records import compute_digest
from nemo_gym.token_id_capture.sink import CaptureContext
from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    CaptureAdmission,
    CommitCoords,
    RolloutReceipt,
)


LOGGER = logging.getLogger(__name__)

DATA_CAPABILITY_HEADER = "x-nemo-gym-capture-capability"
LOGICAL_REQUEST_HEADER = "x-nemo-gym-logical-request-id"
CONTROL_ROUTE_PREFIX = "/training-token-capture/control"


class GateError(Exception):
    """Base class for rollout gate contract violations."""


class UnknownRolloutError(GateError):
    """The rollout is not live and has no retry tombstone."""


class GateStateError(GateError):
    """The requested transition is invalid for the current rollout state."""


class OperationConflictError(GateError):
    """An idempotency key was reused for a different owner or operation."""


class DataCapabilityError(GateError):
    """A model request did not authenticate to its claimed rollout."""


class RolloutRegistration(BaseModel):
    """Controller-only registration response carrying the data capability."""

    model_config = ConfigDict(extra="forbid")

    rollout_id: str
    owner_id: str
    operation_id: str
    data_capability: str
    capability_id: str
    expires_at: float


class CleanupManifest(BaseModel):
    """Staging keys whose rollout reached a terminal non-training outcome."""

    model_config = ConfigDict(extra="forbid")

    rollout_id: str
    reason: str
    staging_keys: list[str] = Field(default_factory=list)


@dataclass
class _CallState:
    admission: CaptureAdmission
    request_items: list[dict]
    logical_request_id: str | None
    capability_id: str
    status: Literal["admitted", "staged", "failed"] = "admitted"
    manifest_record: CallRecord | None = None
    cumulative_token_ids: list[int] = field(default_factory=list)
    failure_reason: str | None = None


@dataclass
class _RolloutState:
    registration: RolloutRegistration = field(repr=False)
    capability_digest: str = field(repr=False)
    calls: dict[str, _CallState] = field(default_factory=dict)
    logical_requests: dict[str, str] = field(default_factory=dict)
    capture_poisoned: bool = False
    failure_reason: str | None = None


@dataclass(frozen=True)
class _TerminalTombstone:
    owner_id: str
    operation_id: str
    expires_at: float
    receipt: RolloutReceipt | None = None
    cleanup: CleanupManifest | None = None
    request_fingerprint: str = ""
    outcome: Literal["sealed", "failed", "expired"] = "sealed"


class RolloutCaptureGate:
    """Serialize register/admit/commit/seal transitions in one server process."""

    def __init__(
        self,
        *,
        lineage_store: LineageStore,
        registration_ttl_s: float,
        tombstone_ttl_s: float,
    ) -> None:
        if registration_ttl_s <= 0 or tombstone_ttl_s <= 0:
            raise ValueError("gate TTL values must be positive")
        self._lineage_store = lineage_store
        self._registration_ttl_s = registration_ttl_s
        self._tombstone_ttl_s = tombstone_ttl_s
        self._rollouts: dict[str, _RolloutState] = {}
        self._tombstones: dict[str, _TerminalTombstone] = {}
        self._cleanup_manifests: list[CleanupManifest] = []
        self._lock = asyncio.Lock()
        self._metrics: dict[str, int] = {
            "registered": 0,
            "register_retries": 0,
            "admitted": 0,
            "staged": 0,
            "capture_failed": 0,
            "sealed": 0,
            "seal_retries": 0,
            "failed": 0,
            "expired": 0,
            "capability_rejected": 0,
        }

    @staticmethod
    def _validate_identifier(value: str, *, field_name: str) -> None:
        if not value or len(value) > 256:
            raise GateStateError(f"{field_name} must contain 1 to 256 characters")
        if any(not (character.isascii() and (character.isalnum() or character in "._-")) for character in value):
            raise GateStateError(f"{field_name} contains unsupported characters")

    @staticmethod
    def _operation_fingerprint(operation: str, *values: object) -> str:
        payload = json.dumps([operation, *values], separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _sweep_locked(self, now: float) -> list[CleanupManifest]:
        expired: list[CleanupManifest] = []
        for rollout_id, state in list(self._rollouts.items()):
            if state.registration.expires_at > now:
                continue
            manifest = self._cleanup_for(state, reason="registration_expired")
            expired.append(manifest)
            self._cleanup_manifests.append(manifest)
            self._tombstones[rollout_id] = _TerminalTombstone(
                owner_id=state.registration.owner_id,
                operation_id=state.registration.operation_id,
                expires_at=now + self._tombstone_ttl_s,
                outcome="expired",
            )
            del self._rollouts[rollout_id]
            self._metrics["expired"] += 1
        for rollout_id, tombstone in list(self._tombstones.items()):
            if tombstone.expires_at <= now:
                del self._tombstones[rollout_id]
        return expired

    @staticmethod
    def _cleanup_for(state: _RolloutState, *, reason: str) -> CleanupManifest:
        return CleanupManifest(
            rollout_id=state.registration.rollout_id,
            reason=reason,
            staging_keys=[
                call.manifest_record.staging_key for call in state.calls.values() if call.manifest_record is not None
            ],
        )

    @staticmethod
    def _touch(state: _RolloutState, *, now: float, ttl_s: float) -> None:
        state.registration.expires_at = now + ttl_s

    async def register_rollout(
        self,
        rollout_id: str,
        *,
        owner_id: str,
        operation_id: str,
    ) -> RolloutRegistration:
        """Create a rollout or replay the identical live registration."""
        self._validate_identifier(rollout_id, field_name="rollout_id")
        self._validate_identifier(owner_id, field_name="owner_id")
        self._validate_identifier(operation_id, field_name="operation_id")
        now = time.time()
        async with self._lock:
            self._sweep_locked(now)
            existing = self._rollouts.get(rollout_id)
            if existing is not None:
                registration = existing.registration
                if registration.owner_id == owner_id and registration.operation_id == operation_id:
                    self._metrics["register_retries"] += 1
                    return registration.model_copy(deep=True)
                raise OperationConflictError(f"rollout {rollout_id} is already registered by another operation")
            tombstone = self._tombstones.get(rollout_id)
            if tombstone is not None:
                raise GateStateError(f"rollout {rollout_id} already reached {tombstone.outcome}")
            capability = secrets.token_urlsafe(32)
            capability_digest = hashlib.sha256(capability.encode("utf-8")).hexdigest()
            registration = RolloutRegistration(
                rollout_id=rollout_id,
                owner_id=owner_id,
                operation_id=operation_id,
                data_capability=capability,
                capability_id=capability_digest,
                expires_at=now + self._registration_ttl_s,
            )
            self._rollouts[rollout_id] = _RolloutState(
                registration=registration,
                capability_digest=capability_digest,
            )
            self._metrics["registered"] += 1
            return registration.model_copy(deep=True)

    def _authenticate_locked(self, state: _RolloutState, capability: str) -> None:
        supplied = hashlib.sha256(capability.encode("utf-8")).hexdigest() if capability else ""
        if not hmac.compare_digest(supplied, state.capability_digest):
            self._metrics["capability_rejected"] += 1
            raise DataCapabilityError("missing, invalid, or expired data capability")

    async def admit_context(
        self,
        context: CaptureContext,
        *,
        data_capability: str,
        request_items: list[dict],
        logical_request_id: str | None = None,
    ) -> CaptureAdmission:
        """Admit the parent already selected by upstream ``resolve_parent``."""
        if not context.parent_resolved and context.lineage_store is not None:
            raise GateStateError("resolve_parent() must run before gate admission")
        if logical_request_id is not None:
            self._validate_identifier(logical_request_id, field_name="logical_request_id")
        now = time.time()
        async with self._lock:
            self._sweep_locked(now)
            state = self._rollouts.get(context.rollout_id)
            if state is None:
                raise UnknownRolloutError(f"unknown rollout {context.rollout_id}")
            self._authenticate_locked(state, data_capability)
            if context.model_call_id in state.calls:
                raise GateStateError(f"model call {context.model_call_id} was already admitted")
            if logical_request_id is not None and logical_request_id in state.logical_requests:
                raise GateStateError(f"logical request {logical_request_id} was already admitted")
            parent_call_id = context.parent_call_id
            if parent_call_id is None:
                admission = CaptureAdmission(
                    rollout_id=context.rollout_id,
                    model_call_id=context.model_call_id,
                    mode="text",
                )
            else:
                parent = state.calls.get(parent_call_id)
                if parent is None or parent.status != "staged":
                    raise GateStateError(
                        f"resolved parent {parent_call_id} is not committed in rollout {context.rollout_id}"
                    )
                if parent.cumulative_token_ids != context.parent_tokens:
                    raise GateStateError(f"resolved parent {parent_call_id} token prefix diverged from gate state")
                admission = CaptureAdmission(
                    rollout_id=context.rollout_id,
                    model_call_id=context.model_call_id,
                    parent_call_id=parent_call_id,
                    prev_len=len(context.parent_tokens),
                    mode="token_in",
                    required_prefix_token_ids=list(context.parent_tokens),
                )
            state.calls[context.model_call_id] = _CallState(
                admission=admission,
                request_items=[dict(item) for item in request_items],
                logical_request_id=logical_request_id,
                capability_id=state.registration.capability_id,
            )
            if logical_request_id is not None:
                state.logical_requests[logical_request_id] = context.model_call_id
            self._touch(state, now=now, ttl_s=self._registration_ttl_s)
            self._metrics["admitted"] += 1
            return admission.model_copy(deep=True)

    async def commit_coords(
        self,
        coords: CommitCoords,
        *,
        response_items: list[dict],
    ) -> bool:
        """Commit one worker acknowledgement and publish cumulative lineage."""
        now = time.time()
        async with self._lock:
            self._sweep_locked(now)
            state = self._rollouts.get(coords.rollout_id)
            if state is None:
                raise UnknownRolloutError(f"unknown rollout {coords.rollout_id}")
            call = state.calls.get(coords.model_call_id)
            if call is None:
                raise GateStateError(f"model call {coords.model_call_id} was not admitted")
            if call.status != "admitted":
                raise GateStateError(f"model call {coords.model_call_id} already reached {call.status}")
            admission = call.admission
            if coords.parent_call_id != admission.parent_call_id or coords.prev_len != admission.prev_len:
                raise GateStateError(f"coordinates for {coords.model_call_id} diverge from admission")
            if coords.cum_len != coords.prev_len + coords.delta_len:
                raise GateStateError(f"coordinates for {coords.model_call_id} have inconsistent lengths")
            if coords.disposition == "capture_failed":
                if (
                    any(value is not None for value in (coords.digest, coords.extras_digest, coords.staging_key))
                    or coords.delta_len != 0
                    or coords.token_ids_delta
                ):
                    raise GateStateError(f"failed coordinates for {coords.model_call_id} carry staged payload")
                call.status = "failed"
                call.failure_reason = "worker_capture_failed"
                state.capture_poisoned = True
                self._touch(state, now=now, ttl_s=self._registration_ttl_s)
                self._metrics["capture_failed"] += 1
                return False
            if len(coords.token_ids_delta) != coords.delta_len:
                raise GateStateError(f"coordinates for {coords.model_call_id} omit delta token IDs")
            if (
                coords.delta_len == 0
                or coords.digest is None
                or coords.extras_digest is None
                or coords.staging_key is None
            ):
                raise GateStateError(f"staged coordinates for {coords.model_call_id} omit custody metadata")
            if admission.parent_call_id is None:
                cumulative = list(coords.token_ids_delta)
            else:
                parent = state.calls.get(admission.parent_call_id)
                if parent is None or parent.status != "staged":
                    raise GateStateError(f"parent {admission.parent_call_id} is not committed")
                if len(parent.cumulative_token_ids) != coords.prev_len:
                    raise GateStateError(f"parent {admission.parent_call_id} length does not equal prev_len")
                cumulative = list(parent.cumulative_token_ids) + list(coords.token_ids_delta)
            if len(cumulative) != coords.cum_len:
                raise GateStateError(f"coordinates for {coords.model_call_id} diverge from cumulative length")
            lineage_digest = compute_digest(cumulative)
            try:
                await self._lineage_store.record(
                    coords.rollout_id,
                    coords.model_call_id,
                    list(call.request_items),
                    [dict(item) for item in response_items],
                    cumulative,
                    lineage_digest,
                )
            except Exception:
                LOGGER.exception(
                    "lineage publication failed for rollout %s call %s",
                    coords.rollout_id,
                    coords.model_call_id,
                )
                call.status = "failed"
                call.failure_reason = "lineage_record_failed"
                state.capture_poisoned = True
                self._metrics["capture_failed"] += 1
                return False
            call.manifest_record = CallRecord(
                model_call_id=coords.model_call_id,
                parent_call_id=coords.parent_call_id,
                prev_len=coords.prev_len,
                delta_len=coords.delta_len,
                cum_len=coords.cum_len,
                weight_version=coords.weight_version,
                digest=coords.digest,
                extras_digest=coords.extras_digest,
                staging_key=coords.staging_key,
                mode=admission.mode,
            )
            call.cumulative_token_ids = cumulative
            call.status = "staged"
            self._touch(state, now=now, ttl_s=self._registration_ttl_s)
            self._metrics["staged"] += 1
            return True

    async def fail_call(self, rollout_id: str, model_call_id: str, *, reason: str) -> None:
        """Poison an admitted call whose engine request did not return coordinates."""
        now = time.time()
        async with self._lock:
            self._sweep_locked(now)
            state = self._rollouts.get(rollout_id)
            if state is None:
                return
            call = state.calls.get(model_call_id)
            if call is None or call.status != "admitted":
                return
            call.status = "failed"
            call.failure_reason = reason
            state.capture_poisoned = True
            self._touch(state, now=now, ttl_s=self._registration_ttl_s)
            self._metrics["capture_failed"] += 1

    async def seal_rollout(
        self,
        rollout_id: str,
        *,
        owner_id: str,
        operation_id: str,
        reward: float | None,
        terminal_logical_request_id: str,
    ) -> RolloutReceipt:
        """Seal once or replay the identical immutable receipt after a lost response."""
        self._validate_identifier(owner_id, field_name="owner_id")
        self._validate_identifier(operation_id, field_name="operation_id")
        self._validate_identifier(
            terminal_logical_request_id,
            field_name="terminal_logical_request_id",
        )
        now = time.time()
        operation_fingerprint = self._operation_fingerprint(
            "seal",
            reward,
            terminal_logical_request_id,
        )
        async with self._lock:
            self._sweep_locked(now)
            tombstone = self._tombstones.get(rollout_id)
            if tombstone is not None:
                if (
                    tombstone.outcome == "sealed"
                    and tombstone.owner_id == owner_id
                    and tombstone.operation_id == operation_id
                    and tombstone.request_fingerprint == operation_fingerprint
                    and tombstone.receipt is not None
                ):
                    self._metrics["seal_retries"] += 1
                    return tombstone.receipt.model_copy(deep=True)
                raise OperationConflictError(f"rollout {rollout_id} already reached {tombstone.outcome}")
            state = self._rollouts.get(rollout_id)
            if state is None:
                raise UnknownRolloutError(f"unknown rollout {rollout_id}")
            if state.registration.owner_id != owner_id:
                raise OperationConflictError(f"owner {owner_id} cannot seal rollout {rollout_id}")
            pending = [call_id for call_id, call in state.calls.items() if call.status == "admitted"]
            if pending:
                raise GateStateError(f"rollout {rollout_id} still has admitted calls: {pending}")
            terminal_call_id = state.logical_requests.get(terminal_logical_request_id)
            if terminal_call_id is None:
                raise GateStateError(f"terminal logical request {terminal_logical_request_id} was not admitted")
            terminal = state.calls[terminal_call_id]
            manifest = [call.manifest_record for call in state.calls.values() if call.manifest_record is not None]
            receipt = RolloutReceipt(
                rollout_id=rollout_id,
                reward=reward,
                terminal_model_call_id=(terminal_call_id if terminal.status == "staged" else None),
                manifest=manifest,
                capture_poisoned=state.capture_poisoned,
                failure_reason=state.failure_reason,
            )
            self._tombstones[rollout_id] = _TerminalTombstone(
                owner_id=owner_id,
                operation_id=operation_id,
                expires_at=now + self._tombstone_ttl_s,
                receipt=receipt,
                request_fingerprint=operation_fingerprint,
                outcome="sealed",
            )
            del self._rollouts[rollout_id]
            self._metrics["sealed"] += 1
            return receipt.model_copy(deep=True)

    async def fail_rollout(
        self,
        rollout_id: str,
        *,
        owner_id: str,
        operation_id: str,
        reason: str,
    ) -> CleanupManifest:
        """Terminate a rollout and return the staging cleanup handshake."""
        now = time.time()
        operation_fingerprint = self._operation_fingerprint("fail", reason)
        async with self._lock:
            self._sweep_locked(now)
            tombstone = self._tombstones.get(rollout_id)
            if tombstone is not None:
                if (
                    tombstone.owner_id == owner_id
                    and tombstone.operation_id == operation_id
                    and tombstone.request_fingerprint == operation_fingerprint
                    and tombstone.cleanup is not None
                ):
                    return tombstone.cleanup.model_copy(deep=True)
                raise OperationConflictError(f"rollout {rollout_id} already reached {tombstone.outcome}")
            state = self._rollouts.get(rollout_id)
            if state is None:
                raise UnknownRolloutError(f"unknown rollout {rollout_id}")
            if state.registration.owner_id != owner_id:
                raise OperationConflictError(f"owner {owner_id} cannot fail rollout {rollout_id}")
            cleanup = self._cleanup_for(state, reason=reason)
            self._cleanup_manifests.append(cleanup)
            self._tombstones[rollout_id] = _TerminalTombstone(
                owner_id=owner_id,
                operation_id=operation_id,
                expires_at=now + self._tombstone_ttl_s,
                cleanup=cleanup,
                request_fingerprint=operation_fingerprint,
                outcome="failed",
            )
            del self._rollouts[rollout_id]
            self._metrics["failed"] += 1
            return cleanup

    async def expire_stale(self) -> list[CleanupManifest]:
        """Expire live registrations and retain their cleanup manifests."""
        async with self._lock:
            return self._sweep_locked(time.time())

    async def drain_cleanup_manifests(self) -> list[CleanupManifest]:
        """Return and clear cleanup work for the framework staging owner."""
        async with self._lock:
            manifests = [manifest.model_copy(deep=True) for manifest in self._cleanup_manifests]
            self._cleanup_manifests.clear()
            return manifests

    async def snapshot_metrics(self) -> dict[str, int]:
        async with self._lock:
            self._sweep_locked(time.time())
            return {
                **self._metrics,
                "live_rollouts": len(self._rollouts),
                "tombstones": len(self._tombstones),
                "cleanup_backlog": len(self._cleanup_manifests),
            }
