# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own web seed requests independently of the HTTP client's lifetime.

This is a web adapter, not a replacement for Gym's proposed core lifecycle
API. Keep task, browser and evaluator ownership in WebSessionManager.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass

from nemo_gym.web.api_models import WebSeedSessionRequest, WebSeedSessionResponse, WebSessionIdentity
from nemo_gym.web.session import CapacityUnavailableError, SessionConflictError, SessionNotFoundError
from nemo_gym.web.session_manager import WebSessionManager


LOG = logging.getLogger(__name__)


class SessionIdentityError(RuntimeError):
    """The request does not authorize access to the named web session."""


@dataclass
class _SessionRequest:
    token_digest: bytes
    created_at: float
    fingerprint: str | None = None
    seed: asyncio.Task[WebSeedSessionResponse] | None = None
    close: asyncio.Task[bool] | None = None
    closing: bool = False
    closed_at: float | None = None
    expiry_close_attempts: int = 0


class WebSessionControl:
    """Deduplicate seed, authorize cookie-less close, and retain tombstones."""

    def __init__(self, manager: WebSessionManager, *, lifetime_seconds: float, max_records: int = 10000) -> None:
        self._manager = manager
        self._lifetime = lifetime_seconds
        self._max_records = max_records
        self._records: dict[str, _SessionRequest] = {}
        self._reaper: asyncio.Task[None] | None = None

    def _record(self, identity: WebSessionIdentity) -> tuple[str, _SessionRequest]:
        # No await between lookup and insertion: concurrent HTTP requests on
        # this event loop cannot claim the same identity with different tokens.
        if identity.session_identity is None or identity.close_token is None:
            raise SessionIdentityError("a session identity and close capability are required")
        session_id = identity.session_identity
        digest = hashlib.sha256(identity.close_token.get_secret_value().encode()).digest()
        record = self._records.get(session_id)
        if record is None:
            self._prune()
            if len(self._records) >= self._max_records:
                raise CapacityUnavailableError("web session identity capacity is full")
            record = _SessionRequest(token_digest=digest, created_at=time.monotonic())
            self._records[session_id] = record
        elif not hmac.compare_digest(record.token_digest, digest):
            raise SessionIdentityError("invalid session close capability")
        return session_id, record

    @staticmethod
    def _consume_exception(task: asyncio.Task) -> None:
        # HTTP cancellation may remove the last waiter, but not our ownership.
        if not task.cancelled():
            task.exception()

    async def seed(self, body: WebSeedSessionRequest) -> WebSeedSessionResponse:
        session_id, record = self._record(body)
        payload = body.model_dump(mode="json", exclude={"session_identity", "close_token"})
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if record.closing:
            raise SessionConflictError("the session identity has already been closed")
        if record.fingerprint is not None and record.fingerprint != fingerprint:
            raise SessionConflictError("the session identity was used with a different seed payload")
        record.fingerprint = fingerprint
        prior_seed = record.seed
        # Replay only a successful seed. A failed one runs again once its
        # cleanup has finished; until then, retries receive the same failure.
        if (
            prior_seed is not None
            and prior_seed.done()
            and (prior_seed.cancelled() or prior_seed.exception() is not None)
            and await self._manager.can_retry_seed(session_id)
        ):
            if record.closing:
                raise SessionConflictError("the session was closed while waiting for admission")
            if record.seed is prior_seed:
                record.seed = None
        if record.seed is None:
            seed_body = body.model_copy(update={"session_identity": None, "close_token": None})
            record.seed = asyncio.create_task(
                self._manager.seed_session(session_id, seed_body), name=f"web-seed-{session_id}"
            )
            record.seed.add_done_callback(self._consume_exception)
        result = await asyncio.shield(record.seed)
        if record.closing:
            raise SessionConflictError("the session was closed while seed was in flight")
        # A completed seed response must not resurrect a manager-expired lease.
        try:
            status = await self._manager.session_status(session_id)
        except SessionNotFoundError:
            record.closing = True
            raise
        if status.status != "ready":
            raise SessionConflictError(f"cannot replay seed while session status={status.status!r}")
        return result

    async def close(self, identity: WebSessionIdentity) -> bool:
        session_id, record = self._record(identity)
        return await self._close_record(session_id, record)

    async def _close_record(self, session_id: str, record: _SessionRequest) -> bool:
        # Close-before-seed creates the same tombstone as a completed close.
        record.closing = True
        if record.closed_at is not None:
            return True
        if record.close is None or record.close.done():
            record.close = asyncio.create_task(self._finish_close(session_id, record), name=f"web-close-{session_id}")
            record.close.add_done_callback(self._consume_exception)
        return await asyncio.shield(record.close)

    async def _finish_close(self, session_id: str, record: _SessionRequest) -> bool:
        if record.seed is not None:
            try:
                await asyncio.shield(record.seed)
            except (Exception, asyncio.CancelledError):  # A failed seed may still have retryable cleanup.
                pass
        closed = await self._manager.close_session(session_id)
        if closed:
            record.closed_at = time.monotonic()
            # Keep only the tombstone, not a completed seed's screenshots or
            # exception traceback, throughout the retry-retention window.
            record.seed = None
        return closed

    def _prune(self) -> None:
        cutoff = time.monotonic() - self._lifetime
        for session_id, record in tuple(self._records.items()):
            if record.closed_at is not None and record.closed_at < cutoff:
                del self._records[session_id]

    def start(self) -> None:
        if self._reaper is None:
            self._reaper = asyncio.create_task(self._reap(), name="web-session-control-reaper")

    async def _reap(self) -> None:
        while True:
            await asyncio.sleep(min(60.0, self._lifetime))
            await self._reap_once()

    async def _reap_once(self) -> None:
        self._prune()
        for session_id, record in tuple(self._records.items()):
            if record.closed_at is not None or time.monotonic() - record.created_at < self._lifetime:
                continue
            # The manager enforces idle expiry. Identity retention must not
            # impose a new wall-clock limit on a healthy, active rollout.
            if not record.closing and await self._manager.is_live_session(session_id):
                continue
            if record.expiry_close_attempts >= 3:
                continue  # Retain evidence; explicit close remains retryable.
            # Do not let one slow seed block expiry of other sessions.
            record.closing = True
            if record.close is None or record.close.done():
                record.expiry_close_attempts += 1
                record.close = asyncio.create_task(self._finish_close(session_id, record))
                record.close.add_done_callback(self._consume_exception)

    async def stop(self, *, timeout: float) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            await asyncio.gather(self._reaper, return_exceptions=True)
            self._reaper = None
        tasks = [asyncio.create_task(self._close_record(sid, record)) for sid, record in self._records.items()]
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=timeout)
            for task in done:
                self._consume_exception(task)
            for task in pending:
                task.add_done_callback(self._consume_exception)
            if pending:
                LOG.error("event=web_session_shutdown_cleanup_pending count=%d", len(pending))
