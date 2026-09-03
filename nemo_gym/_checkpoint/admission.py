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
"""Admission control for one checkpoint participant.

Closing admission rejects new operations before they change server state.
Already accepted operations remain in flight until their response completes.
The server becomes paused only when that count reaches zero.

A refused caller receives ``409 checkpoint_parked``.
The caller can re-issue that operation after the checkpoint.

A request counts as in flight from admission until its response completes.
On capture routes the model server writes durable custody before the
terminal response event, so response completion also marks custody
completion.
"""

import asyncio
import re
import time
from typing import Any, Callable, Optional
from uuid import uuid4

from starlette.responses import JSONResponse

from nemo_gym._checkpoint.control import AdmissionState, ControlError
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
    """One admitted in-flight operation."""

    __slots__ = ("ticket_id", "rollout_id", "attempt_index", "plane", "started_ts", "task")

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


class AdmissionLimiter:
    """Atomic admission state machine for one server process.

    State changes are atomic with the admission test: both happen inside one
    event-loop step, so there is no window where a request is admitted
    against a state that a concurrent control call already changed.
    """

    def __init__(self) -> None:
        self.state = AdmissionState.ACCEPTING
        self._inflight: dict[str, AdmissionTicket] = {}
        self._tombstones: set[tuple[str, int]] = set()
        self._drained = asyncio.Event()
        self._drained.set()
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
            if (rollout_id, attempt_index) in self._tombstones:
                raise StaleAttemptError(
                    f"rollout {rollout_id!r} attempt {attempt_index} was closed at a checkpoint "
                    f"deadline; the restored run dispatched a replacement attempt"
                )

        if self.state != AdmissionState.ACCEPTING:
            raise AdmissionParkedError(
                f"admission is {self.state.value} for a checkpoint; park and re-issue this "
                f"operation after the checkpoint completes"
            )

        ticket = AdmissionTicket(
            rollout_id=rollout_id,
            attempt_index=attempt_index,
            plane=plane,
            task=task,
        )
        self._inflight[ticket.ticket_id] = ticket
        self._drained.clear()
        self._notify_listeners()
        return ticket

    def release(self, ticket: AdmissionTicket) -> None:
        # Idempotent: a force-closed ticket was already removed by abort.
        removed = self._inflight.pop(ticket.ticket_id, None)
        if removed is not None:
            self._after_inflight_change()

    def _after_inflight_change(self) -> None:
        if not self._inflight:
            self._drained.set()
            if self.state == AdmissionState.DRAINING:
                self.state = AdmissionState.PAUSED
        self._notify_listeners()

    # -- control -------------------------------------------------------------

    def close(self) -> None:
        """Stop admitting new root operations. Atomic with the admission test."""
        if self.state == AdmissionState.ACCEPTING:
            self.state = AdmissionState.DRAINING if self._inflight else AdmissionState.PAUSED

    def resume(self) -> None:
        self.state = AdmissionState.ACCEPTING

    def abort_inflight(self, rollout_id: str, attempt_index: int) -> list[str]:
        """Cancel and fence a rollout attempt that missed the prepare deadline.

        The request remains in flight until its ASGI task exits.
        A checkpoint cannot commit while cancelled code may still write.
        """
        self._tombstones.add((rollout_id, attempt_index))
        aborted = [
            ticket.ticket_id
            for ticket in self._inflight.values()
            if ticket.rollout_id == rollout_id and ticket.attempt_index == attempt_index
        ]
        for ticket_id in aborted:
            ticket = self._inflight[ticket_id]
            if ticket.task is not None:
                ticket.task.cancel()
        return aborted

    def install_tombstone(self, logical_rollout_id: str, attempt_index: int) -> None:
        self._tombstones.add((logical_rollout_id, attempt_index))

    def tombstones(self) -> list[tuple[str, int]]:
        return sorted(self._tombstones)

    # -- observation ---------------------------------------------------------

    async def wait_for_drained(self, timeout_s: float) -> bool:
        """Wait until nothing is in flight, up to ``timeout_s``. True if drained."""
        if timeout_s <= 0:
            return not self._inflight
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

        try:
            await self._app(scope, receive, send)
        finally:
            # The ASGI call returns only after the response (including a
            # streamed one) has finished sending, so releasing here keeps the
            # request in flight until its response completes.
            self._limiter.release(ticket)
