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
"""Admission and generation-cut control for one checkpoint participant.

Closing admission rejects new operations before they change server state and
freezes the exact admitted ticket membership of the checkpoint. A configured
generation-cut backend may make those tickets prepare-safe before their HTTP
responses finish. Without one, the historical response-drain behavior remains.

A refused caller receives ``409 checkpoint_parked``.
The caller can re-issue that operation after the checkpoint.

Response lifetime and generation safety are deliberately separate. A request
counts as response-in-flight until its downstream ASGI invocation returns. Its
ticket becomes prepare-safe after a durable completed call, a durable
active-prefix cut, a durable failure, an explicit exclusion, or a no-generation
exit.
"""

import asyncio
import contextvars
import re
import time
from typing import Any, Callable, Literal, Optional
from uuid import uuid4

from starlette.responses import JSONResponse

from nemo_gym._checkpoint.control import AdmissionState, ControlError
from nemo_gym._checkpoint.model_control_contracts import (
    GenerationCutBackend,
    GenerationCutFrozenTicket,
    GenerationCutInventory,
    GenerationCutPrefix,
    GenerationCutReceipt,
    GenerationCutWorkerProof,
)
from nemo_gym.config_types import ROLLOUT_PATH_PREFIX
from nemo_gym.rollout_correlation import (
    ATTEMPT_INDEX_HEADER,
    ROLLOUT_ID_HEADER,
    ROLLOUT_ID_PATTERN,
    capture_key_for,
    split_transport_rollout_id,
)


PLANE_HEADER = "x-nemo-gym-plane"

# Suffixes of the generation routes a policy model server gates. Matching by
# suffix covers the plain routes and their ``/ng-rollout/<id>/...`` twins.
GATED_MODEL_ROUTE_SUFFIXES = ("/v1/responses", "/v1/chat/completions", "/v1/messages")


class AdmissionParkedError(ControlError):
    """The server is draining or paused and this caller can safely park.

    Not a failure: the operation was refused before any state changed, so the
    caller re-issues it after the checkpoint completes.
    """

    code = "checkpoint_parked"


class StaleAttemptError(ControlError):
    """The rollout attempt was force-closed at a checkpoint deadline.

    Its call roots are tombstoned; a late call from the abandoned attempt
    must not write new state under an identity the restore already replaced.
    """

    code = "stale_attempt"


class AdmissionTicket:
    """One admitted operation with independent response and cut lifetimes."""

    __slots__ = (
        "ticket_id",
        "rollout_id",
        "attempt_index",
        "plane",
        "started_ts",
        "task",
        "model_call_id",
        "generation_started",
        "prepare_safe",
        "prepare_safe_reason",
        "response_active",
        "response_started",
        "response_egress_completed",
        "checkpoint_abort_pending",
        "_limiter",
    )

    def __init__(
        self,
        *,
        rollout_id: Optional[str],
        attempt_index: Optional[int],
        plane: Optional[str],
        task: Optional[asyncio.Task],
    ) -> None:
        self.ticket_id = uuid4().hex
        self.rollout_id = rollout_id
        self.attempt_index = attempt_index
        self.plane = plane
        self.started_ts = time.time()
        self.task = task
        self.model_call_id: Optional[str] = None
        self.generation_started = False
        self.prepare_safe = False
        self.prepare_safe_reason: Optional[str] = None
        self.response_active = True
        self.response_started = False
        self.response_egress_completed = False
        self.checkpoint_abort_pending = False

    @property
    def response_completed(self) -> bool:
        return not self.response_active

    def bind_model_call(self, model_call_id: str) -> None:
        self.model_call_id = model_call_id

    def mark_generation_started(self) -> None:
        if (
            not self.generation_started
            and self._limiter.generation_cut_backend is not None
            and self._limiter.state != AdmissionState.ACCEPTING
        ):
            self.mark_no_generation()
            raise AdmissionParkedError(
                "the request was admitted before checkpoint close but had not started generation; "
                "park and re-issue it after the checkpoint completes"
            )
        self.generation_started = True

    def mark_no_generation(self) -> None:
        self._limiter.mark_prepare_safe(self, "no_generation")

    def mark_durable_completed(self) -> None:
        self._limiter.mark_prepare_safe(self, "durable_completed")

    def mark_durable_failure(self) -> None:
        self._limiter.mark_prepare_safe(self, "durable_failure")

    def mark_durable_prefix_cut(self) -> None:
        self._limiter.mark_prepare_safe(self, "durable_prefix")

    def exclude(self) -> None:
        self._limiter.mark_prepare_safe(self, "excluded")


_ADMISSION_TICKET: contextvars.ContextVar[AdmissionTicket | None] = contextvars.ContextVar(
    "nemo_gym_admission_ticket", default=None
)


def current_admission_ticket() -> AdmissionTicket | None:
    """Return the ticket bound around the current downstream ASGI call."""
    return _ADMISSION_TICKET.get()


def bind_current_model_call(model_call_id: str) -> None:
    """Bind the capture-generated call identity to the current admission."""
    ticket = current_admission_ticket()
    if ticket is not None:
        ticket.bind_model_call(model_call_id)


def mark_current_generation_started() -> None:
    """Mark that the current request crossed into generation."""
    ticket = current_admission_ticket()
    if ticket is not None:
        ticket.mark_generation_started()


def mark_current_generation_safe(
    reason: Literal["durable_completed", "durable_failure", "no_generation"] = "durable_completed",
) -> None:
    """Mark the current ticket prepare-safe after its durable terminal event."""
    ticket = current_admission_ticket()
    if ticket is None:
        return
    ticket._limiter.mark_prepare_safe(ticket, reason)


class AdmissionLimiter:
    """Atomic admission state machine for one server process.

    State changes are atomic with the admission test: both happen inside one
    event-loop step, so there is no window where a request is admitted
    against a state that a concurrent control call already changed.
    """

    def __init__(self, generation_cut_backend: GenerationCutBackend | None = None) -> None:
        self.state = AdmissionState.ACCEPTING
        self._inflight: dict[str, AdmissionTicket] = {}
        self._generation_cut_backend = generation_cut_backend
        self._checkpoint_id: Optional[str] = None
        self._checkpoint_tickets: dict[str, AdmissionTicket] = {}
        self._generation_cut_receipt: GenerationCutReceipt | None = None
        self._cut_lock = asyncio.Lock()
        self._admission_tombstones: set[tuple[str, int]] = set()
        self._checkpoint_exclusions: set[tuple[str, int]] = set()
        self._seen_attempts: set[tuple[str, int]] = set()
        self._drained = asyncio.Event()
        self._drained.set()
        self._egress_open = asyncio.Event()
        self._egress_open.set()
        self._listeners: list[Callable[[], None]] = []

    # -- change listeners ----------------------------------------------------
    #
    # A multi-worker deployment reports each worker's in-flight count to a
    # service-level coordinator; the listener fires on every count change so
    # the report is event-driven instead of polled.

    def add_listener(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[], None]) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_listeners(self) -> None:
        for listener in list(self._listeners):
            listener()

    # -- admission -----------------------------------------------------------

    def admit(
        self,
        *,
        rollout_id: Optional[str] = None,
        attempt_index: Optional[int] = None,
        plane: Optional[str] = None,
        task: Optional[asyncio.Task] = None,
    ) -> AdmissionTicket:
        if (rollout_id is None) != (attempt_index is None):
            raise ValueError("rollout_id and attempt_index must be provided together")
        if rollout_id is not None and attempt_index is not None:
            if (rollout_id, attempt_index) in self._admission_tombstones:
                raise StaleAttemptError(
                    f"rollout {rollout_id!r} attempt {attempt_index} was closed at a checkpoint "
                    f"deadline; the restored run dispatched a replacement attempt"
                )
        if self.state != AdmissionState.ACCEPTING:
            raise AdmissionParkedError(
                f"admission is {self.state.value} for a checkpoint; park and re-issue this "
                f"operation after the checkpoint completes"
            )
        if rollout_id is not None and attempt_index is not None:
            self._seen_attempts.add((rollout_id, attempt_index))

        ticket = AdmissionTicket(
            rollout_id=rollout_id,
            attempt_index=attempt_index,
            plane=plane,
            task=task,
        )
        # Kept private and intentionally absent from the public inventory.
        ticket._limiter = self
        self._inflight[ticket.ticket_id] = ticket
        self._drained.clear()
        self._notify_listeners()
        return ticket

    def release(self, ticket: AdmissionTicket, *, response_egress_completed: bool = True) -> None:
        # Idempotent: a force-closed ticket was already removed by abort.
        removed = self._inflight.pop(ticket.ticket_id, None)
        if removed is not None:
            ticket.response_active = False
            ticket.response_egress_completed = response_egress_completed
            if ticket.checkpoint_abort_pending:
                ticket.exclude()
            elif not ticket.prepare_safe:
                if not ticket.generation_started:
                    ticket.mark_no_generation()
                elif response_egress_completed:
                    self.mark_prepare_safe(ticket, "response_complete")
            self._after_inflight_change()

    def _after_inflight_change(self) -> None:
        if self.is_prepare_safe():
            self._drained.set()
        else:
            self._drained.clear()
        if self.state in (AdmissionState.DRAINING, AdmissionState.PAUSED):
            self.state = AdmissionState.PAUSED if self.is_prepare_safe() else AdmissionState.DRAINING
        self._notify_listeners()

    # -- control -------------------------------------------------------------

    def close(self, checkpoint_id: Optional[str] = None) -> None:
        """Stop admission and freeze this checkpoint's ticket membership."""
        if self.state == AdmissionState.ACCEPTING:
            self._checkpoint_id = checkpoint_id
            self._checkpoint_tickets = dict(self._inflight)
            self._generation_cut_receipt = None
            if self._generation_cut_backend is not None:
                self._egress_open.clear()
            self.state = AdmissionState.DRAINING if not self.is_prepare_safe() else AdmissionState.PAUSED
            if self.state == AdmissionState.PAUSED:
                self._drained.set()
            else:
                self._drained.clear()

    def resume(self) -> None:
        self.state = AdmissionState.ACCEPTING
        self._checkpoint_id = None
        self._checkpoint_tickets.clear()
        self._generation_cut_receipt = None
        self._checkpoint_exclusions.clear()
        self._egress_open.set()

    @property
    def generation_cut_backend(self) -> GenerationCutBackend | None:
        return self._generation_cut_backend

    @property
    def generation_cut_receipt(self) -> GenerationCutReceipt | None:
        return self._generation_cut_receipt

    @property
    def generation_cut_ack(self) -> GenerationCutReceipt | None:
        """Compatibility alias for the final durable cut receipt."""
        return self._generation_cut_receipt

    def mark_prepare_safe(self, ticket: AdmissionTicket, reason: str) -> None:
        """Advance one ticket monotonically to a prepare-safe terminal state."""
        if ticket.prepare_safe:
            return
        ticket.prepare_safe = True
        ticket.prepare_safe_reason = reason
        self._after_inflight_change()

    def is_prepare_safe(self) -> bool:
        """Return whether every ticket frozen into the active cut is safe."""
        if self._generation_cut_backend is None:
            return not self._inflight
        return all(self._ticket_checkpoint_ready(ticket) for ticket in self._checkpoint_tickets.values())

    def generation_pending(self) -> int:
        if self._generation_cut_backend is None:
            return len(self._inflight)
        return sum(not self._ticket_checkpoint_ready(ticket) for ticket in self._checkpoint_tickets.values())

    @staticmethod
    def _ticket_checkpoint_ready(ticket: AdmissionTicket) -> bool:
        if ticket.checkpoint_abort_pending:
            return not ticket.response_active
        if ticket.response_started:
            return ticket.response_egress_completed and not ticket.response_active
        return ticket.prepare_safe

    def generation_cut_worker_proof(
        self,
        checkpoint_id: str,
        *,
        coordinator_sequence: int,
        worker_id: str,
    ) -> GenerationCutWorkerProof:
        """Build one sequenced proof over this worker's complete frozen set."""
        if self._checkpoint_id != checkpoint_id:
            raise ValueError("worker generation-cut proof checkpoint does not own frozen admission")
        frozen = [
            GenerationCutFrozenTicket(
                ticket_id=ticket.ticket_id,
                rollout_id=ticket.rollout_id,
                attempt_index=ticket.attempt_index,
                model_call_id=ticket.model_call_id,
                generation_started=ticket.generation_started,
                response_started=ticket.response_started,
            )
            for ticket in self._checkpoint_tickets.values()
        ]
        ready = [
            ticket.ticket_id for ticket in self._checkpoint_tickets.values() if self._ticket_checkpoint_ready(ticket)
        ]
        return GenerationCutWorkerProof.build(
            checkpoint_id=checkpoint_id,
            coordinator_sequence=coordinator_sequence,
            worker_id=worker_id,
            frozen_tickets=frozen,
            ready_ticket_ids=ready,
            generation_cut_receipt=self._generation_cut_receipt,
        )

    def should_gate_response_start(self, ticket: AdmissionTicket) -> bool:
        """Return whether this frozen ticket must wait before response start."""
        return (
            self._generation_cut_backend is not None
            and self.state != AdmissionState.ACCEPTING
            and ticket.ticket_id in self._checkpoint_tickets
            and not ticket.response_started
        )

    async def wait_for_response_egress(self, ticket: AdmissionTicket) -> None:
        """Wait until checkpoint resolution reopens a frozen response."""
        while self.should_gate_response_start(ticket):
            await self._egress_open.wait()

    def mark_response_egress_completed(self, ticket: AdmissionTicket) -> None:
        ticket.response_egress_completed = True
        self._after_inflight_change()

    def generation_cut_inventory(self, checkpoint_id: str, *, server_name: str) -> GenerationCutInventory:
        """Build the deterministic active-prefix inventory for one frozen cut."""
        if self._checkpoint_id not in (None, checkpoint_id):
            raise ValueError(
                f"checkpoint {checkpoint_id!r} does not own the frozen admission membership "
                f"for {self._checkpoint_id!r}"
            )
        prefixes = [
            GenerationCutPrefix(
                ticket_id=ticket.ticket_id,
                rollout_id=ticket.rollout_id,
                attempt_index=ticket.attempt_index,
                model_call_id=ticket.model_call_id,
                admitted_at=ticket.started_ts,
            )
            for ticket in self._checkpoint_tickets.values()
            if ticket.generation_started and not ticket.prepare_safe and not ticket.response_started
        ]
        return GenerationCutInventory.build(
            checkpoint_id=checkpoint_id,
            server_name=server_name,
            active_prefixes=prefixes,
        )

    async def prepare_generation_cut(
        self,
        checkpoint_id: str,
        *,
        server_name: str,
        timeout_s: float,
    ) -> bool:
        """Ask the configured backend to durably cut currently active prefixes."""
        backend = self._generation_cut_backend
        if backend is None or self.is_prepare_safe():
            return self.is_prepare_safe()
        async with self._cut_lock:
            if self._generation_cut_receipt is not None or self.is_prepare_safe():
                return self.is_prepare_safe()
            inventory = self.generation_cut_inventory(checkpoint_id, server_name=server_name)
            if not inventory.active_prefixes:
                return self.is_prepare_safe()
            try:
                receipt = await asyncio.wait_for(
                    backend.checkpoint_generation_cut(inventory),
                    timeout=max(timeout_s, 0.0),
                )
            except asyncio.TimeoutError:
                return False
            receipt = GenerationCutReceipt.model_validate(receipt)
            receipt.validate_for(inventory)
            by_id = {prefix.ticket_id: prefix for prefix in inventory.active_prefixes}
            for result in receipt.prefixes:
                ticket = self._checkpoint_tickets.get(result.ticket_id)
                if ticket is None or result.ticket_id not in by_id:
                    raise ValueError(f"generation-cut ack named unknown ticket {result.ticket_id!r}")
                if result.disposition == "durable_prefix":
                    ticket.mark_durable_prefix_cut()
                else:
                    ticket.mark_durable_failure()
            self._generation_cut_receipt = receipt
            return self.is_prepare_safe()

    def abort_inflight(self, rollout_id: str, attempt_index: int) -> list[str]:
        """Cancel and fence a rollout attempt that missed the prepare deadline.

        The request remains in flight until its ASGI task exits.
        A checkpoint cannot commit while cancelled code may still write.
        """
        identity = (rollout_id, attempt_index)
        self._admission_tombstones.add(identity)
        self._checkpoint_exclusions.add(identity)
        candidates = {**self._checkpoint_tickets, **self._inflight}
        aborted = [
            ticket.ticket_id
            for ticket in candidates.values()
            if ticket.rollout_id == rollout_id
            and ticket.attempt_index == attempt_index
            and (ticket.response_active or not self._ticket_checkpoint_ready(ticket))
        ]
        for ticket_id in aborted:
            ticket = candidates[ticket_id]
            ticket.checkpoint_abort_pending = True
            if not ticket.response_active:
                ticket.exclude()
            else:
                self._after_inflight_change()
            if ticket.task is not None and not ticket.task.done():
                ticket.task.cancel()
        return aborted

    def install_tombstone(self, logical_rollout_id: str, attempt_index: int) -> None:
        """Install a fence for an already-logical identity (checkpoint restore).

        The checkpoint records logical IDs and explicit attempt indices.
        This method preserves those values without parsing capture-key suffixes.
        """
        self._admission_tombstones.add((logical_rollout_id, attempt_index))

    def tombstones(self) -> list[tuple[str, int]]:
        return sorted(self._admission_tombstones)

    def checkpoint_exclusions(self) -> list[tuple[str, int]]:
        """Return attempts aborted during this process's active checkpoint."""
        return sorted(self._checkpoint_exclusions)

    def is_tombstoned(self, rollout_id: Optional[str], attempt_index: Optional[int]) -> bool:
        return (
            rollout_id is not None
            and attempt_index is not None
            and (rollout_id, attempt_index) in self._admission_tombstones
        )

    def seen_attempts(self) -> list[tuple[str, int]]:
        return sorted(self._seen_attempts)

    # -- observation ---------------------------------------------------------

    async def wait_for_drained(self, timeout_s: float) -> bool:
        """Wait until the frozen generation set is prepare-safe."""
        if timeout_s <= 0:
            return self.is_prepare_safe()
        try:
            await asyncio.wait_for(self._drained.wait(), timeout=timeout_s)
            return True
        except asyncio.TimeoutError:
            return False

    def counts(self) -> dict[str, Any]:
        now = time.time()
        return {
            "state": self.state.value,
            "inflight_total": len(self._inflight),
            "response_inflight_total": len(self._inflight),
            "generation_pending_total": self.generation_pending(),
            "waiters_total": 0,
            "inflight": [
                {
                    "rollout_id": ticket.rollout_id,
                    "attempt_index": ticket.attempt_index,
                    "plane": ticket.plane,
                    "age_seconds": round(now - ticket.started_ts, 3),
                }
                for ticket in self._inflight.values()
            ],
        }


class AdmissionMiddleware:
    """Gate a server's data-plane routes behind an ``AdmissionLimiter``.

    Only paths ending in one of ``gated_suffixes`` are gated; control routes
    and liveness stay reachable while the data plane is paused.
    """

    _PATH_IDENTITY = re.compile(rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<capture_key>[^/]+)(?:/|$)")

    def __init__(self, app: Any, limiter: AdmissionLimiter, gated_suffixes: tuple[str, ...]) -> None:
        self._app = app
        self._limiter = limiter
        self._gated_suffixes = tuple(gated_suffixes)

    def _gated(self, scope: dict[str, Any]) -> bool:
        if scope.get("type") != "http":
            return False
        path = scope.get("path", "")
        return path.endswith(self._gated_suffixes)

    @staticmethod
    def _headers(scope: dict[str, Any]) -> dict[str, str]:
        wanted = {ROLLOUT_ID_HEADER, ATTEMPT_INDEX_HEADER, PLANE_HEADER}
        found: dict[str, str] = {}
        for name, value in scope.get("headers") or ():
            key = name.decode("latin-1").lower()
            if key in wanted:
                found[key] = value.decode("latin-1")
        return found

    @classmethod
    def _execution_identity(
        cls, scope: dict[str, Any], headers: dict[str, str]
    ) -> tuple[Optional[str], Optional[int]]:
        rollout_id = headers.get(ROLLOUT_ID_HEADER)
        attempt_raw = headers.get(ATTEMPT_INDEX_HEADER)
        if (rollout_id is None) != (attempt_raw is None):
            raise ValueError("rollout ID and attempt index headers must be sent together")

        attempt_index: Optional[int] = None
        if rollout_id is not None and attempt_raw is not None:
            if ROLLOUT_ID_PATTERN.fullmatch(rollout_id) is None:
                raise ValueError("invalid logical rollout ID header")
            try:
                attempt_index = int(attempt_raw)
            except ValueError as error:
                raise ValueError("invalid attempt index header") from error
            if attempt_index < 0:
                raise ValueError("attempt index must be non-negative")

        match = cls._PATH_IDENTITY.match(scope.get("path", ""))
        capture_key = match.group("capture_key") if match is not None else None
        if capture_key is not None and ROLLOUT_ID_PATTERN.fullmatch(capture_key) is None:
            raise ValueError("invalid capture key in request path")
        if capture_key is not None and rollout_id is not None:
            if capture_key_for(rollout_id, attempt_index or 0) != capture_key:
                raise ValueError("capture path disagrees with execution identity headers")
        elif capture_key is not None:
            rollout_id, attempt_index = split_transport_rollout_id(capture_key)
        return rollout_id, attempt_index

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if not self._gated(scope):
            await self._app(scope, receive, send)
            return

        headers = self._headers(scope)
        try:
            rollout_id, attempt_index = self._execution_identity(scope, headers)
        except ValueError as error:
            response = JSONResponse(
                status_code=409,
                content={"error": {"code": "execution_identity_mismatch", "detail": str(error)}},
            )
            await response(scope, receive, send)
            return

        try:
            ticket = self._limiter.admit(
                rollout_id=rollout_id,
                attempt_index=attempt_index,
                plane=headers.get(PLANE_HEADER),
                task=asyncio.current_task(),
            )
        except ControlError as e:
            response = JSONResponse(
                status_code=e.status_code,
                content={"error": {"code": e.code, "detail": e.detail}},
                headers={"retry-after": "1"},
            )
            await response(scope, receive, send)
            return

        response_started = False
        context_token = _ADMISSION_TICKET.set(ticket)

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                await self._limiter.wait_for_response_egress(ticket)
                ticket.response_started = True
                response_started = True
            await send(message)
            if (
                message.get("type") == "http.response.body"
                and not message.get("more_body", False)
                and ticket.response_started
            ):
                self._limiter.mark_response_egress_completed(ticket)

        try:
            await self._app(scope, receive, tracked_send)
        except AdmissionParkedError as error:
            if response_started:
                raise
            response = JSONResponse(
                status_code=error.status_code,
                content={"error": {"code": error.code, "detail": error.detail}},
                headers={"retry-after": "1"},
            )
            await response(scope, receive, send)
        except asyncio.CancelledError:
            if not self._limiter.is_tombstoned(ticket.rollout_id, ticket.attempt_index):
                raise
            if response_started:
                # HTTP status is immutable after response start. Propagating
                # cancellation closes the transport instead of completing a
                # misleading successful response.
                raise
            response = JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "code": "stale_attempt",
                        "detail": "the rollout attempt was aborted at a checkpoint deadline",
                    }
                },
            )
            await response(scope, receive, send)
        finally:
            # The ASGI call returns only after the response (including a
            # streamed one) has finished sending, so releasing here keeps the
            # request in flight until its response completes.
            try:
                if not ticket.generation_started and not ticket.prepare_safe:
                    self._limiter.mark_prepare_safe(ticket, "no_generation")
                self._limiter.release(
                    ticket,
                    response_egress_completed=ticket.response_egress_completed,
                )
            finally:
                _ADMISSION_TICKET.reset(context_token)
