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
"""Episode sessions for SWE resources servers whose seed starts one task sandbox.

These servers give the agent a task sandbox at seed time. ``verify`` captures the agent's patch from that
sandbox, stops it, and grades the patch in a fresh sandbox. ``SandboxSessionResourcesServer`` adds the episode
session half of the resources session contract (see ``SimpleResourcesServer.close_resources_session``) on top of that
lifecycle, so an Environment Server such as ``single_agent_turn`` can seed a task, hand the agent the task
sandbox through ``sandbox_access``, verify, and close.

A server subclasses it and keeps its agent-run seed for agents that seed through ``/run``. It provides two hooks:

- ``_start_task_sandbox`` starts and prepares the task sandbox, registers it in ``_session_id_to_sandbox``
  as soon as it exists, and returns the directory the agent works in;
- ``_forget_task_sandbox_state`` drops the other per-session state the server keeps, such as the
  untracked files the image ships.

The server runtime routes episode session requests to the worker that owns the session.
"""

import asyncio
import logging
from abc import abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import monotonic
from typing import Any, cast

import anyio
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, ValidationError

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import SESSION_EXPIRED, SESSION_LOST, EpisodeId
from nemo_gym.sandbox import AsyncSandbox, DirectSandboxConnection, SandboxAccess
from nemo_gym.server_runtime import (
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFences,
    ResourcesSessionLocks,
    SimpleResourcesServer,
)


LOG = logging.getLogger(__name__)

# How long an open session lasts when its sandbox config sets no ttl_s. Past the sandbox's lifetime the
# Environment Server that seeded it is gone or has given up on its close, so the session is closed here.
DEFAULT_SESSION_LIFETIME_S = 86400.0
# Ceiling on one sandbox stop, so a hung provider cannot hold a session lock or stall shutdown.
STOP_TIMEOUT_S = 120.0
# Wait before the next request retries an expired session whose stop failed. Without it every request would
# retry the stuck session, and each retry can take up to STOP_TIMEOUT_S. An explicit close or shutdown retries at once.
EXPIRY_RETRY_INTERVAL_S = 60.0
# Sandbox stops that shutdown runs at once, so a server with thousands of open sessions does not flood the provider.
SHUTDOWN_STOP_CONCURRENCY = 64


class AgentRunCloseSessionRequest(BaseModel):
    """Close an agent-run sandbox using the request's session cookie."""

    model_config = ConfigDict(extra="forbid")


@dataclass
class _TaskSession:
    seed: ResourcesSeedSessionRequest | None = None
    task: BaseModel | None = None
    response: ResourcesSeedSessionResponse | None = None
    workdir: str = ""
    expired: bool = False
    verify_request: BaseVerifyRequest | None = None
    capture: object = None
    capture_ready: bool = False
    verification: BaseVerifyResponse | None = None
    # Compatibility for consumers that still use _claim/_release instead of the verify wrapper.
    verified: bool = False
    release_done: asyncio.Event | None = None


class SandboxSessionResourcesServer(SimpleResourcesServer):
    """A resources server whose seed starts one task sandbox, serving episode sessions.

    Task sandboxes live in ``_session_id_to_sandbox``, keyed by session ID. The config must have
    ``sandbox_provider`` and ``sandbox_config``.

    Seed, verify and close share a per-session lock. A session that is never closed expires after its
    sandbox's ``ttl_s`` (or ``DEFAULT_SESSION_LIFETIME_S``), but failed cleanup keeps ownership for a retry.
    Fences of closed sessions last the server runtime's close retry window.
    """

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: dict[str, AsyncSandbox] = {}
        self._verifier_sandboxes: dict[str, AsyncSandbox] = {}
        self._task_sessions: dict[str, _TaskSession] = {}
        self._task_session_locks = ResourcesSessionLocks()
        self._task_session_fences: ResourcesSessionFences[EpisodeId] = ResourcesSessionFences(
            self.config.session_close_retry_window_seconds
        )
        self._open_task_sessions: dict[str, float] = {}
        self._busy_task_sessions: set[str] = set()

    @abstractmethod
    async def _start_task_sandbox(self, session_id: str, task: BaseModel) -> str:
        """Start and prepare the task sandbox for ``session_id`` and return the agent's working directory.

        Register the sandbox in ``_session_id_to_sandbox`` as soon as it exists, so a failed preparation can
        still stop it.
        """

    def _forget_task_sandbox_state(self, session_id: str) -> None:
        """Drop the per-session state, other than the sandbox, that seeding recorded."""

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[Any]:
            try:
                async with parent_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def shutdown(self) -> None:
        """Stop task sandboxes that no close or verify released, so they do not outlive the server.

        Sessions are stopped concurrently: the supervisor gives a server about a second to exit, which a
        sequence of provider stops would exhaust after the first few.
        """
        session_ids = set(self._task_sessions) | set(self._session_id_to_sandbox) | set(self._verifier_sandboxes)
        stopping = asyncio.Semaphore(SHUTDOWN_STOP_CONCURRENCY)
        await asyncio.gather(*(self._stop_abandoned_session(session_id, stopping) for session_id in session_ids))

    async def _stop_abandoned_session(self, session_id: str, stopping: asyncio.Semaphore) -> None:
        async with stopping, self._task_session_locks.locked(session_id):
            try:
                await self._stop_session_sandboxes(session_id)
            except Exception:
                LOG.exception("Failed to stop the abandoned sandboxes of session %s", session_id)
            else:
                self._task_sessions.pop(session_id, None)
                self._open_task_sessions.pop(session_id, None)

    @asynccontextmanager
    async def _locked_task_session(self, session_id: str) -> AsyncIterator[_TaskSession]:
        for expired_id, expired_record in list(self._task_sessions.items()):
            if (
                expired_record.expired
                and expired_id not in self._open_task_sessions
                and expired_id not in self._busy_task_sessions
                and expired_id not in self._session_id_to_sandbox
                and expired_id not in self._verifier_sandboxes
                and self._task_session_fences.get(expired_id) is None
            ):
                self._task_sessions.pop(expired_id, None)
        await self._close_abandoned_task_sessions(monotonic())
        async with self._task_session_locks.locked(session_id):
            record = self._task_sessions.setdefault(session_id, _TaskSession())
            self._busy_task_sessions.add(session_id)
            try:
                yield record
            finally:
                self._busy_task_sessions.discard(session_id)
                if (
                    record.seed is None
                    and session_id not in self._session_id_to_sandbox
                    and session_id not in self._verifier_sandboxes
                ):
                    self._task_sessions.pop(session_id, None)

    async def _close_abandoned_task_sessions(self, now: float) -> None:
        for session_id, deadline in list(self._open_task_sessions.items()):
            if deadline > now or session_id in self._busy_task_sessions:
                continue
            record = self._task_sessions.get(session_id)
            if record is None or record.seed is None:
                continue
            async with self._task_session_locks.locked(session_id):
                # Requests that queued on this lock while another one retried the stop see its new deadline.
                retry_deadline = self._open_task_sessions.get(session_id)
                if (
                    self._task_sessions.get(session_id) is not record
                    or retry_deadline is None
                    or retry_deadline > monotonic()
                ):
                    continue
                record.expired = True
                LOG.warning("Closing resources session %s, which was never closed", session_id)
                try:
                    await self._stop_session_sandboxes(session_id)
                except Exception:
                    LOG.exception("Failed to stop the task sandbox of abandoned session %s", session_id)
                    self._open_task_sessions[session_id] = monotonic() + EXPIRY_RETRY_INTERVAL_S
                    continue
                self._open_task_sessions.pop(session_id, None)
                # Keep the expired marker until the configured fence expires, for SESSION_EXPIRED verification.
                self._task_session_fences.fence(session_id, record.seed.episode_id)

    def _session_lifetime_s(self) -> float:
        ttl_s = self.config.sandbox_config.get("ttl_s")
        return float(ttl_s) if ttl_s else DEFAULT_SESSION_LIFETIME_S

    async def seed_task_sandbox_session(
        self, request: Request, body: ResourcesSeedSessionRequest, task_model: type[BaseModel]
    ) -> ResourcesSeedSessionResponse:
        """Seed an episode session: start the task sandbox and hand it to the agent through ``sandbox_access``.

        Repeating the seed for the same episode and task returns the same sandbox. A seed for a closed
        session, or one that names another episode or task, is rejected.
        """
        session_id = body.resources_session_id
        async with self._locked_task_session(session_id) as record:
            self._task_session_fences.raise_if_fenced(session_id)
            if record.expired:
                raise ResourcesSessionConflictError("Resources session has expired")
            if record.seed is not None and record.seed != body:
                raise ResourcesSessionConflictError("resources_session_id is bound to another episode, task or data")
            if record.verify_request is not None or record.verified:
                raise ResourcesSessionConflictError("Resources session has already started verification")
            if record.response is not None:
                if session_id not in self._session_id_to_sandbox:
                    raise ResourcesSessionConflictError("The seeded task sandbox is no longer available")
                request.session[SESSION_ID_KEY] = session_id
                return record.response
            try:
                task = task_model.model_validate(body.task_data)
            except ValidationError as error:
                raise ResourcesSeedRejectedError(f"Invalid task_data: {error}") from error
            record.seed = body.model_copy(deep=True)
            record.task = task
            self._open_task_sessions.setdefault(session_id, monotonic() + self._session_lifetime_s())
            # A seed whose cleanup failed may have left its sandbox registered; never orphan it.
            await self._stop_session_sandboxes(session_id)
            try:
                record.workdir = await self._start_task_sandbox(session_id, task)
                response = await self._task_sandbox_response(session_id, record)
            except BaseException:
                # The agent cannot reach a sandbox it was never handed, so stop it here.
                if await self._stop_failed_seed_sandbox(session_id):
                    self._task_sessions.pop(session_id, None)
                    self._open_task_sessions.pop(session_id, None)
                raise
            record.response = response
            request.session[SESSION_ID_KEY] = session_id
            return response

    async def _task_sandbox_response(self, session_id: str, record: _TaskSession) -> ResourcesSeedSessionResponse:
        sandbox = self._session_id_to_sandbox.get(session_id)
        if sandbox is None or record.verified:
            raise ResourcesSessionConflictError(f"The task sandbox of {session_id} is no longer available")
        return ResourcesSeedSessionResponse(
            resources_session_id=session_id,
            sandbox_access=SandboxAccess(
                connection=DirectSandboxConnection(
                    provider_config_ref=self.config.sandbox_provider,
                    # The agent may operate the sandbox but not destroy it: verify still needs it. Providers
                    # without leases ignore the scope.
                    descriptor=await sandbox.serialize(scope="operate"),
                ),
                workdir=record.workdir,
            ),
        )

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest | AgentRunCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        """Stop the session's sandboxes after an episode or a cookie-scoped agent run.

        A failed stop raises without fencing, and keeps the sandbox so a retried close stops it.
        Cookie-scoped close cannot close an episode session, which requires its explicit identity.
        """
        if isinstance(body, AgentRunCloseSessionRequest):
            session_id = request.session[SESSION_ID_KEY]
            async with self._locked_task_session(session_id) as record:
                if record.seed is not None or self._task_session_fences.get(session_id) is not None:
                    raise ResourcesSessionConflictError("An episode session requires its explicit close request")
                await self._stop_session_sandboxes(session_id)
                self._task_sessions.pop(session_id, None)
                self._open_task_sessions.pop(session_id, None)
                request.session.pop(SESSION_ID_KEY, None)
                return ResourcesCloseSessionResponse(resources_session_id=session_id)
        session_id = body.resources_session_id
        async with self._locked_task_session(session_id) as record:
            fenced_episode = self._task_session_fences.get(session_id)
            episode = fenced_episode or (record.seed.episode_id if record.seed is not None else None)
            if episode is not None and body.episode_id != episode:
                raise ResourcesSessionConflictError("episode_id does not match the resources session")
            await self._stop_session_sandboxes(session_id)
            self._task_session_fences.fence(session_id, body.episode_id)
            self._task_sessions.pop(session_id, None)
            self._open_task_sessions.pop(session_id, None)
            if request.session.get(SESSION_ID_KEY) == session_id:
                request.session.pop(SESSION_ID_KEY, None)
            return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def _stop_session_sandboxes(self, session_id: str) -> None:
        await self._stop_task_sandbox(session_id)
        await self._stop_verifier_sandbox(session_id)

    async def _stop_task_sandbox(self, session_id: str) -> None:
        record = self._task_sessions.get(session_id)
        if record is not None and record.release_done is not None:
            try:
                with anyio.fail_after(STOP_TIMEOUT_S, shield=True):
                    await record.release_done.wait()
            except TimeoutError:
                # A claim that was never released must not keep close, expiry and shutdown from stopping it.
                LOG.warning("Session %s claimed its task sandbox and never released it; stopping it here", session_id)
        sandbox = self._session_id_to_sandbox.get(session_id)
        if sandbox is not None:
            # Forget the sandbox only once it stopped, so a failed stop can be retried.
            await _stop(sandbox)
            self._session_id_to_sandbox.pop(session_id, None)
        self._forget_task_sandbox_state(session_id)

    async def _stop_failed_seed_sandbox(self, session_id: str) -> bool:
        """Stop the task sandbox of a seed that is failing, and say whether the stop succeeded.

        A failed stop is logged instead of raised, so the seed's own error or cancellation reaches the caller.
        The sandbox stays registered after a failed stop, so a retried seed, close or shutdown stops it.
        """
        try:
            await self._stop_task_sandbox(session_id)
        except Exception:
            LOG.exception("Failed to stop the task sandbox of failed seed %s", session_id)
            return False
        return True

    async def stop_task_sandbox_after_capture(self, session_id: str) -> None:
        """Stop the task sandbox once verify captured its patch; call inside the verify callback.

        An episode session raises when the stop fails: the sandbox stays registered and a retried verify reuses the
        captured patch. A session seeded through ``/run`` has no saved capture and no close, so a failed stop is
        logged and grading goes on; shutdown stops the sandbox that stays registered.
        """
        try:
            await self._stop_task_sandbox(session_id)
        except Exception:
            record = self._task_sessions.get(session_id)
            if record is not None and record.seed is not None:
                raise
            LOG.exception("Failed to stop the task sandbox of session %s after capturing its patch", session_id)

    async def _stop_verifier_sandbox(self, session_id: str) -> None:
        """Stop an owned verifier, retaining its handle when the provider cannot confirm cleanup."""
        sandbox = self._verifier_sandboxes.get(session_id)
        if sandbox is not None:
            await _stop(sandbox)
            self._verifier_sandboxes.pop(session_id, None)

    async def verify_task_sandbox_session[ResponseT: BaseVerifyResponse](
        self,
        request: Request,
        body: BaseVerifyRequest,
        *,
        verify: Callable[[], Awaitable[ResponseT]],
        lost_response: Callable[[str, str], ResponseT],
        allow_seedless: bool = False,
    ) -> ResponseT:
        """Serialize verification with close and replay an episode session's saved verdict.

        ``lost_response`` accepts a reason and failure kind, so each consumer can supply its required
        response fields. Seedless verification is reserved for golden-patch grading.
        """
        session_id = request.session[SESSION_ID_KEY]
        async with self._locked_task_session(session_id) as record:
            if record.expired:
                return lost_response("The task session has expired", SESSION_EXPIRED)
            if self._task_session_fences.get(session_id) is not None:
                return lost_response("The resources session is closed", SESSION_LOST)
            if record.seed is not None:
                if record.response is None or record.task is None:
                    return lost_response("The task session did not finish seeding", SESSION_LOST)
                try:
                    task = type(record.task).model_validate(body.model_dump())
                except ValidationError as error:
                    raise ResourcesSessionConflictError(f"Verification task is invalid: {error}") from error
                fields = set(type(record.task).model_fields)
                if task.model_dump(include=fields) != record.task.model_dump(include=fields):
                    raise ResourcesSessionConflictError("Verification task does not match the seeded task")
                if record.verify_request is not None and record.verify_request != body:
                    raise ResourcesSessionConflictError("Verification input does not match the first request")
                if record.verification is not None:
                    return cast(ResponseT, record.verification)
                if record.verified:
                    return lost_response("The task sandbox was already consumed", SESSION_LOST)
            if not allow_seedless and not record.capture_ready and session_id not in self._session_id_to_sandbox:
                return lost_response("The task sandbox is missing or was never seeded", SESSION_LOST)
            record.verify_request = body.model_copy(deep=True)
            await self._stop_verifier_sandbox(session_id)
            try:
                response = await verify()
                if record.seed is not None:
                    record.verification = response
                return response
            finally:
                # Whatever claimed the task sandbox inside this callback is done with it, released or not.
                if record.release_done is not None:
                    record.release_done.set()
                try:
                    await self._stop_verifier_sandbox(session_id)
                except Exception:
                    LOG.exception("Failed to stop verifier of session %s; close can retry", session_id)

    async def capture_task_sandbox_patch[CaptureT](
        self, session_id: str, capture: Callable[[], Awaitable[CaptureT]]
    ) -> CaptureT:
        """Save an episode session's captured patch before teardown; call inside the verify callback.

        The consumer stops the task with ``stop_task_sandbox_after_capture`` after this returns, outside its
        extraction error handler. A failed stop or grading retry can then reuse the original capture.
        """
        record = self._task_sessions.get(session_id)
        if record is not None and record.seed is not None and record.capture_ready:
            return cast(CaptureT, record.capture)
        result = await capture()
        if record is not None and record.seed is not None:
            record.capture = result
            record.capture_ready = True
        return result

    def _claim_task_sandbox(self, session_id: str) -> None:
        """Mark an episode session's task sandbox as consumed by verify, and reject a repeated verify.

        A session an agent's ``/run`` seeded has no episode record and is left as it was.
        """
        self._task_session_fences.raise_if_fenced(session_id)
        record = self._task_sessions.get(session_id)
        if record is None or record.seed is None:
            return
        if record.verified or record.expired:
            raise ResourcesSessionConflictError(f"The task sandbox of {session_id} was already verified or closed")
        record.verified = True
        record.release_done = asyncio.Event()

    async def _release_task_sandbox(self, session_id: str, sandbox: AsyncSandbox) -> None:
        """Stop a task sandbox verify already unregistered.

        When the stop fails for an open episode session, the sandbox is registered again so its close retries.
        """
        # Restore ownership before awaiting the provider, including for legacy consumers that popped it.
        self._session_id_to_sandbox[session_id] = sandbox
        try:
            await _stop(sandbox)
            self._session_id_to_sandbox.pop(session_id, None)
            self._forget_task_sandbox_state(session_id)
        except BaseException as error:
            if not isinstance(error, Exception):
                raise
            LOG.exception("Failed to stop the task sandbox of session %s", session_id)
        finally:
            record = self._task_sessions.get(session_id)
            if record is not None and record.release_done is not None:
                record.release_done.set()


async def _stop(sandbox: AsyncSandbox) -> None:
    """Stop a sandbox within ``STOP_TIMEOUT_S``, even while the request that owns it is being cancelled.

    A client that disconnects cancels its request handler, and every later await in it, so an unshielded
    cleanup stop would never run and the sandbox would outlive every reference to it.
    """
    with anyio.CancelScope(shield=True), anyio.fail_after(STOP_TIMEOUT_S):
        await sandbox.stop()
