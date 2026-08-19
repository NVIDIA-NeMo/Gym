# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multiworker rollout gate for worker-owned token staging."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from typing import TYPE_CHECKING

from nemo_gym.rollout_correlation import DATA_CAPABILITY_HEADER, LOGICAL_REQUEST_HEADER
from nemo_gym.token_id_capture.gate_store import (
    CleanupManifest,
    GateCallState,
    GateRolloutState,
    GateStateStore,
    GateTombstone,
    RolloutRegistration,
    SharedGateState,
)
from nemo_gym.token_id_capture.protocols import LineageStore
from nemo_gym.token_id_capture.records import compute_digest
from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    CaptureAdmission,
    CommitCoords,
    RolloutReceipt,
)


if TYPE_CHECKING:
    from nemo_gym.token_id_capture.sink import CaptureContext


LOGGER = logging.getLogger(__name__)

CONTROL_ROUTE_PREFIX = "/training-token-capture/control"
NG_CAPTURE_FIELD = "ng_capture"
NG_COMMIT_COORDS_FIELD = "ng_commit_coords"


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


class RolloutCaptureGate:
    """Apply atomic gate transitions through a cross-process state store."""

    def __init__(
        self,
        *,
        lineage_store: LineageStore,
        state_store: GateStateStore,
        registration_ttl_s: float,
        tombstone_ttl_s: float,
    ) -> None:
        if registration_ttl_s <= 0 or tombstone_ttl_s <= 0:
            raise ValueError("gate TTL values must be positive")
        self._lineage_store = lineage_store
        self._state_store = state_store
        self._registration_ttl_s = registration_ttl_s
        self._tombstone_ttl_s = tombstone_ttl_s

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

    def _sweep(self, shared: SharedGateState, now: float) -> list[CleanupManifest]:
        expired: list[CleanupManifest] = []
        for rollout_id, rollout in list(shared.rollouts.items()):
            if rollout.registration.expires_at > now:
                continue
            cleanup = self._cleanup_for(rollout, reason="registration_expired")
            expired.append(cleanup)
            shared.cleanup_manifests.append(cleanup)
            shared.tombstones[rollout_id] = GateTombstone(
                owner_id=rollout.registration.owner_id,
                operation_id=rollout.registration.operation_id,
                expires_at=now + self._tombstone_ttl_s,
                cleanup=cleanup,
                outcome="expired",
            )
            del shared.rollouts[rollout_id]
            shared.metrics["expired"] += 1
        for rollout_id, tombstone in list(shared.tombstones.items()):
            if tombstone.expires_at <= now:
                del shared.tombstones[rollout_id]
        return expired

    @staticmethod
    def _cleanup_for(rollout: GateRolloutState, *, reason: str) -> CleanupManifest:
        staging_keys = {
            call.cleanup_staging_key for call in rollout.calls.values() if call.cleanup_staging_key is not None
        }
        return CleanupManifest(
            rollout_id=rollout.registration.rollout_id,
            reason=reason,
            staging_keys=sorted(staging_keys),
        )

    def _touch(self, rollout: GateRolloutState, *, now: float) -> None:
        rollout.registration.expires_at = now + self._registration_ttl_s

    @staticmethod
    def _authenticate(
        shared: SharedGateState,
        rollout: GateRolloutState,
        capability: str,
    ) -> None:
        supplied = hashlib.sha256(capability.encode("utf-8")).hexdigest() if capability else ""
        if not hmac.compare_digest(supplied, rollout.capability_digest):
            shared.metrics["capability_rejected"] += 1
            raise DataCapabilityError("missing, invalid, or expired data capability")

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
        with self._state_store.transaction() as shared:
            self._sweep(shared, now)
            existing = shared.rollouts.get(rollout_id)
            if existing is not None:
                registration = existing.registration
                if registration.owner_id == owner_id and registration.operation_id == operation_id:
                    shared.metrics["register_retries"] += 1
                    return registration.model_copy(deep=True)
                raise OperationConflictError(f"rollout {rollout_id} is already registered by another operation")
            tombstone = shared.tombstones.get(rollout_id)
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
            shared.rollouts[rollout_id] = GateRolloutState(
                registration=registration,
                capability_digest=capability_digest,
            )
            shared.metrics["registered"] += 1
            return registration.model_copy(deep=True)

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
        with self._state_store.transaction() as shared:
            self._sweep(shared, now)
            rollout = shared.rollouts.get(context.rollout_id)
            if rollout is None:
                raise UnknownRolloutError(f"unknown rollout {context.rollout_id}")
            self._authenticate(shared, rollout, data_capability)
            if context.model_call_id in rollout.calls:
                raise GateStateError(f"model call {context.model_call_id} was already admitted")
            if logical_request_id is not None and logical_request_id in rollout.logical_requests:
                raise GateStateError(f"logical request {logical_request_id} was already admitted")
            if context.parent_call_id is None:
                admission = CaptureAdmission(
                    rollout_id=context.rollout_id,
                    model_call_id=context.model_call_id,
                    mode="text",
                )
            else:
                parent = rollout.calls.get(context.parent_call_id)
                if parent is None or parent.status != "staged":
                    raise GateStateError(
                        f"resolved parent {context.parent_call_id} is not committed in rollout {context.rollout_id}"
                    )
                if parent.cumulative_token_ids != context.parent_tokens:
                    raise GateStateError(
                        f"resolved parent {context.parent_call_id} token prefix diverged from gate state"
                    )
                admission = CaptureAdmission(
                    rollout_id=context.rollout_id,
                    model_call_id=context.model_call_id,
                    parent_call_id=context.parent_call_id,
                    prev_len=len(context.parent_tokens),
                    mode="token_in",
                    required_prefix_token_ids=list(context.parent_tokens),
                )
            rollout.calls[context.model_call_id] = GateCallState(
                admission=admission,
                request_items=[dict(item) for item in request_items],
                logical_request_id=logical_request_id,
                capability_id=rollout.registration.capability_id,
            )
            if logical_request_id is not None:
                rollout.logical_requests[logical_request_id] = context.model_call_id
            self._touch(rollout, now=now)
            shared.metrics["admitted"] += 1
            shared.metrics[admission.mode] += 1
            return admission.model_copy(deep=True)

    @staticmethod
    def _validate_coords(call: GateCallState, coords: CommitCoords) -> None:
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
            return
        if len(coords.token_ids_delta) != coords.delta_len:
            raise GateStateError(f"coordinates for {coords.model_call_id} omit delta token IDs")
        if (
            coords.delta_len == 0
            or coords.digest is None
            or coords.extras_digest is None
            or coords.staging_key is None
        ):
            raise GateStateError(f"staged coordinates for {coords.model_call_id} omit custody metadata")

    @staticmethod
    def _same_committed_coords(call: GateCallState, coords: CommitCoords) -> bool:
        record = call.manifest_record
        if record is None:
            return False
        return (
            record.model_call_id == coords.model_call_id
            and record.parent_call_id == coords.parent_call_id
            and record.prev_len == coords.prev_len
            and record.delta_len == coords.delta_len
            and record.cum_len == coords.cum_len
            and record.weight_version == coords.weight_version
            and record.digest == coords.digest
            and record.extras_digest == coords.extras_digest
            and record.staging_key == coords.staging_key
        )

    async def commit_coords(
        self,
        coords: CommitCoords,
        *,
        response_items: list[dict],
        logical_request_id: str | None = None,
    ) -> bool:
        """Reserve, publish, and atomically finalize one worker acknowledgement."""
        if logical_request_id is not None:
            self._validate_identifier(
                logical_request_id,
                field_name="logical_request_id",
            )
        now = time.time()
        with self._state_store.transaction() as shared:
            self._sweep(shared, now)
            rollout = shared.rollouts.get(coords.rollout_id)
            if rollout is None:
                raise UnknownRolloutError(f"unknown rollout {coords.rollout_id}")
            call = rollout.calls.get(coords.model_call_id)
            if call is None:
                raise GateStateError(f"model call {coords.model_call_id} was not admitted")
            if logical_request_id is not None:
                existing_call_id = rollout.logical_requests.get(logical_request_id)
                if existing_call_id not in (None, coords.model_call_id):
                    raise GateStateError(f"logical request {logical_request_id} belongs to another model call")
                if call.logical_request_id not in (None, logical_request_id):
                    raise GateStateError(f"model call {coords.model_call_id} already has a different logical request")
                call.logical_request_id = logical_request_id
                rollout.logical_requests[logical_request_id] = coords.model_call_id
            self._validate_coords(call, coords)
            if call.status == "staged":
                if self._same_committed_coords(call, coords):
                    return True
                raise GateStateError(f"model call {coords.model_call_id} has conflicting committed coordinates")
            if call.status == "failed":
                raise GateStateError(f"model call {coords.model_call_id} already failed")
            if coords.disposition == "capture_failed":
                if call.status != "admitted":
                    raise GateStateError(f"model call {coords.model_call_id} is already committing")
                call.status = "failed"
                call.failure_reason = "worker_capture_failed"
                rollout.capture_poisoned = True
                rollout.failure_reason = rollout.failure_reason or call.failure_reason
                self._touch(rollout, now=now)
                shared.metrics["capture_failed"] += 1
                return False
            if call.status == "committing":
                if call.pending_coords != coords:
                    raise GateStateError(f"model call {coords.model_call_id} has conflicting pending coordinates")
                cumulative = list(call.cumulative_token_ids)
            else:
                if call.admission.parent_call_id is None:
                    cumulative = list(coords.token_ids_delta)
                else:
                    parent = rollout.calls.get(call.admission.parent_call_id)
                    if parent is None or parent.status != "staged":
                        raise GateStateError(f"parent {call.admission.parent_call_id} is not committed")
                    if len(parent.cumulative_token_ids) != coords.prev_len:
                        raise GateStateError(f"parent {call.admission.parent_call_id} length does not equal prev_len")
                    cumulative = list(parent.cumulative_token_ids) + list(coords.token_ids_delta)
                if len(cumulative) != coords.cum_len:
                    raise GateStateError(f"coordinates for {coords.model_call_id} diverge from cumulative length")
                call.pending_coords = coords.model_copy(deep=True)
                call.cleanup_staging_key = coords.staging_key
                call.cumulative_token_ids = cumulative
                call.status = "committing"
                self._touch(rollout, now=now)
            request_items = [dict(item) for item in call.request_items]

        lineage_digest = compute_digest(cumulative)
        try:
            await self._lineage_store.record(
                coords.rollout_id,
                coords.model_call_id,
                request_items,
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
            with self._state_store.transaction() as shared:
                rollout = shared.rollouts.get(coords.rollout_id)
                call = rollout.calls.get(coords.model_call_id) if rollout is not None else None
                if call is not None and call.status == "committing" and call.pending_coords == coords:
                    call.status = "failed"
                    call.failure_reason = "lineage_record_failed"
                    call.pending_coords = None
                    rollout.capture_poisoned = True
                    rollout.failure_reason = rollout.failure_reason or call.failure_reason
                    shared.metrics["capture_failed"] += 1
            return False

        with self._state_store.transaction() as shared:
            rollout = shared.rollouts.get(coords.rollout_id)
            if rollout is None:
                raise UnknownRolloutError(f"rollout {coords.rollout_id} expired during commit")
            call = rollout.calls.get(coords.model_call_id)
            if call is not None and call.status == "staged":
                if self._same_committed_coords(call, coords):
                    return True
                raise GateStateError(f"model call {coords.model_call_id} has conflicting committed coordinates")
            if call is None or call.status != "committing" or call.pending_coords != coords:
                raise GateStateError(f"commit reservation for {coords.model_call_id} changed")
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
                mode=call.admission.mode,
            )
            call.pending_coords = None
            call.cleanup_staging_key = coords.staging_key
            call.status = "staged"
            self._touch(rollout, now=time.time())
            shared.metrics["staged"] += 1
        return True

    async def fail_call(self, rollout_id: str, model_call_id: str, *, reason: str) -> None:
        """Poison an admitted call whose engine request did not return coordinates."""
        now = time.time()
        with self._state_store.transaction() as shared:
            self._sweep(shared, now)
            rollout = shared.rollouts.get(rollout_id)
            if rollout is None:
                return
            call = rollout.calls.get(model_call_id)
            if call is None or call.status not in ("admitted", "committing"):
                return
            call.status = "failed"
            call.pending_coords = None
            call.failure_reason = reason
            rollout.capture_poisoned = True
            rollout.failure_reason = rollout.failure_reason or reason
            self._touch(rollout, now=now)
            shared.metrics["capture_failed"] += 1

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
        with self._state_store.transaction() as shared:
            self._sweep(shared, now)
            tombstone = shared.tombstones.get(rollout_id)
            if tombstone is not None:
                if (
                    tombstone.outcome == "sealed"
                    and tombstone.owner_id == owner_id
                    and tombstone.operation_id == operation_id
                    and tombstone.request_fingerprint == operation_fingerprint
                    and tombstone.receipt is not None
                ):
                    shared.metrics["seal_retries"] += 1
                    return tombstone.receipt.model_copy(deep=True)
                raise OperationConflictError(f"rollout {rollout_id} already reached {tombstone.outcome}")
            rollout = shared.rollouts.get(rollout_id)
            if rollout is None:
                raise UnknownRolloutError(f"unknown rollout {rollout_id}")
            if rollout.registration.owner_id != owner_id:
                raise OperationConflictError(f"owner {owner_id} cannot seal rollout {rollout_id}")
            pending = [call_id for call_id, call in rollout.calls.items() if call.status in ("admitted", "committing")]
            if pending:
                raise GateStateError(f"rollout {rollout_id} still has in-flight calls: {pending}")
            terminal_call_id = rollout.logical_requests.get(terminal_logical_request_id)
            if terminal_call_id is None:
                raise GateStateError(f"terminal logical request {terminal_logical_request_id} was not admitted")
            terminal = rollout.calls[terminal_call_id]
            manifest = [call.manifest_record for call in rollout.calls.values() if call.manifest_record is not None]
            receipt = RolloutReceipt(
                rollout_id=rollout_id,
                reward=reward,
                terminal_model_call_id=(terminal_call_id if terminal.status == "staged" else None),
                manifest=manifest,
                capture_poisoned=rollout.capture_poisoned,
                failure_reason=rollout.failure_reason,
            )
            shared.tombstones[rollout_id] = GateTombstone(
                owner_id=owner_id,
                operation_id=operation_id,
                expires_at=now + self._tombstone_ttl_s,
                receipt=receipt,
                request_fingerprint=operation_fingerprint,
                outcome="sealed",
            )
            del shared.rollouts[rollout_id]
            shared.metrics["sealed"] += 1
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
        with self._state_store.transaction() as shared:
            self._sweep(shared, now)
            tombstone = shared.tombstones.get(rollout_id)
            if tombstone is not None:
                if (
                    tombstone.owner_id == owner_id
                    and tombstone.operation_id == operation_id
                    and tombstone.request_fingerprint == operation_fingerprint
                    and tombstone.cleanup is not None
                ):
                    shared.metrics["fail_retries"] += 1
                    return tombstone.cleanup.model_copy(deep=True)
                raise OperationConflictError(f"rollout {rollout_id} already reached {tombstone.outcome}")
            rollout = shared.rollouts.get(rollout_id)
            if rollout is None:
                raise UnknownRolloutError(f"unknown rollout {rollout_id}")
            if rollout.registration.owner_id != owner_id:
                raise OperationConflictError(f"owner {owner_id} cannot fail rollout {rollout_id}")
            cleanup = self._cleanup_for(rollout, reason=reason)
            shared.cleanup_manifests.append(cleanup)
            shared.tombstones[rollout_id] = GateTombstone(
                owner_id=owner_id,
                operation_id=operation_id,
                expires_at=now + self._tombstone_ttl_s,
                cleanup=cleanup,
                request_fingerprint=operation_fingerprint,
                outcome="failed",
            )
            del shared.rollouts[rollout_id]
            shared.metrics["failed"] += 1
            return cleanup.model_copy(deep=True)

    async def expire_stale(self) -> list[CleanupManifest]:
        """Expire live registrations and retain their cleanup manifests."""
        with self._state_store.transaction() as shared:
            return [manifest.model_copy(deep=True) for manifest in self._sweep(shared, time.time())]

    async def note_unattributed_call(self) -> None:
        """Count model traffic that bypassed the training-capture path."""
        with self._state_store.transaction() as shared:
            shared.metrics["unattributed_calls"] += 1

    async def drain_cleanup_manifests(self) -> list[CleanupManifest]:
        """Return and clear cleanup work for the framework staging owner."""
        with self._state_store.transaction() as shared:
            manifests = [manifest.model_copy(deep=True) for manifest in shared.cleanup_manifests]
            shared.cleanup_manifests.clear()
            return manifests

    async def snapshot_metrics(self) -> dict[str, int]:
        with self._state_store.transaction() as shared:
            self._sweep(shared, time.time())
            return {
                **shared.metrics,
                "live_rollouts": len(shared.rollouts),
                "tombstones": len(shared.tombstones),
                "cleanup_backlog": len(shared.cleanup_manifests),
            }


__all__ = [
    "CONTROL_ROUTE_PREFIX",
    "DATA_CAPABILITY_HEADER",
    "LOGICAL_REQUEST_HEADER",
    "NG_CAPTURE_FIELD",
    "NG_COMMIT_COORDS_FIELD",
    "CleanupManifest",
    "DataCapabilityError",
    "GateError",
    "GateStateError",
    "OperationConflictError",
    "RolloutCaptureGate",
    "RolloutRegistration",
    "UnknownRolloutError",
]
