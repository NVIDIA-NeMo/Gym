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
"""Environment Server sessions for SWE resources servers whose seed starts one task sandbox.

These servers give the agent a task sandbox at seed time. ``verify`` captures the agent's patch from that
sandbox, stops it, and grades the patch in a fresh sandbox. ``SandboxSessionResourcesServer`` adds the typed
half of the resources session contract (see ``SimpleResourcesServer.close_resources_session``) on top of that
lifecycle, so an Environment Server such as ``single_agent_turn`` can seed a task, hand the agent the task
sandbox through ``sandbox_access``, verify, and close.

A server subclasses it and keeps its legacy seed for agents that seed through ``/run``. It provides two hooks:

- ``_start_task_sandbox`` starts and prepares the task sandbox, registers it in ``_session_id_to_sandbox``
  as soon as it exists, and returns the directory the agent works in;
- ``_forget_task_sandbox_state`` drops the other per-session state the server keeps, such as the
  untracked files the image ships.

Typed sessions are process-local, so a typed seed requires ``num_workers=1``.
"""

import asyncio
import logging
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

import anyio
from fastapi import FastAPI, Request
from pydantic import BaseModel, JsonValue

from nemo_gym.base_resources_server import (
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.task_materialization import TASK_ID_FIELDS


LOG = logging.getLogger(__name__)

# How long a closed session keeps rejecting seeds. An Environment Server assigns a fresh
# resources_session_id to every episode, so a seed can only reach a closed session as a retry of the
# episode's own seed request, which ends within minutes.
CLOSED_SESSION_RETENTION_S = 3600.0
# How long an open session lasts when its sandbox config sets no ttl_s. Past the sandbox's lifetime the
# Environment Server that seeded it is gone or has given up on its close, so the session is closed here.
DEFAULT_SESSION_LIFETIME_S = 86400.0
# Ceiling on one sandbox stop, so a hung provider cannot hold a session lock or stall shutdown.
STOP_TIMEOUT_S = 120.0


@dataclass
class _TaskSession:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    identity: tuple[EpisodeId, TaskId] | None = None
    workdir: str = ""
    # Set by the first verify, which stops the task sandbox; a repeated verify would grade an empty patch.
    verified: bool = False
    closed_episode_id: EpisodeId | None = None


class SandboxSessionResourcesServer(SimpleResourcesServer):
    """A resources server whose seed starts one task sandbox, serving typed resources sessions.

    Task sandboxes live in ``_session_id_to_sandbox``, keyed by session ID. The config must have
    ``sandbox_provider`` and ``sandbox_config``.

    Each typed session record ends when its close fence expires, ``CLOSED_SESSION_RETENTION_S`` after the
    close, or when its seed fails. A session that is never closed is closed here once its sandbox's
    ``ttl_s`` (or ``DEFAULT_SESSION_LIFETIME_S``) has passed. The task sandbox ends at the first verify or
    at close, whichever comes first. Sandboxes still running at shutdown are stopped then.
    """

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: dict[str, AsyncSandbox] = {}
        self._task_sessions: dict[str, _TaskSession] = {}
        # Closed sessions in close order, with the monotonic time their fence expires.
        self._closed_task_sessions: OrderedDict[str, float] = OrderedDict()
        # Open sessions in seed order, with the monotonic time after which they are treated as abandoned.
        self._open_task_sessions: OrderedDict[str, float] = OrderedDict()

    @abstractmethod
    async def _start_task_sandbox(self, session_id: str, task: Any) -> str:
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
        """Stop task sandboxes that no close or verify released, so they do not outlive the server."""
        self._task_sessions.clear()
        self._closed_task_sessions.clear()
        self._open_task_sessions.clear()
        for session_id in list(self._session_id_to_sandbox):
            try:
                await self._stop_task_sandbox(session_id)
            except Exception:
                LOG.exception("Failed to stop the abandoned task sandbox of session %s", session_id)

    @asynccontextmanager
    async def _locked_task_session(self, session_id: str) -> AsyncIterator[_TaskSession]:
        now = monotonic()
        await self._close_abandoned_task_sessions(now)
        while self._closed_task_sessions:
            expired_id, expires_at = next(iter(self._closed_task_sessions.items()))
            if expires_at > now:
                break
            del self._closed_task_sessions[expired_id]
            self._task_sessions.pop(expired_id, None)
        while True:
            record = self._task_sessions.setdefault(session_id, _TaskSession())
            async with record.lock:
                # A waiter can outlive the record it queued on (a failed seed, an expired fence).
                if self._task_sessions.get(session_id) is not record:
                    continue
                try:
                    yield record
                finally:
                    if record.identity is None and record.closed_episode_id is None:
                        self._task_sessions.pop(session_id, None)
                return

    async def _close_abandoned_task_sessions(self, now: float) -> None:
        while self._open_task_sessions:
            session_id, deadline = next(iter(self._open_task_sessions.items()))
            if deadline > now:
                return
            del self._open_task_sessions[session_id]
            record = self._task_sessions.get(session_id)
            if record is None or record.identity is None or record.closed_episode_id is not None:
                continue
            if record.lock.locked():
                # A seed, close or shutdown of this session is in progress; look again a lifetime later.
                self._open_task_sessions[session_id] = now + self._session_lifetime_s()
                continue
            async with record.lock:
                LOG.warning("Closing resources session %s, which was never closed", session_id)
                try:
                    await self._stop_task_sandbox(session_id)
                except Exception:
                    # The provider ends the sandbox at its ttl_s; keep no reference to it here.
                    LOG.exception("Failed to stop the task sandbox of abandoned session %s", session_id)
                    self._session_id_to_sandbox.pop(session_id, None)
                    self._forget_task_sandbox_state(session_id)
                record.closed_episode_id = record.identity[0]
                self._closed_task_sessions[session_id] = now + CLOSED_SESSION_RETENTION_S

    def _session_lifetime_s(self) -> float:
        ttl_s = self.config.sandbox_config.get("ttl_s")
        return float(ttl_s) if ttl_s else DEFAULT_SESSION_LIFETIME_S

    async def seed_task_sandbox_session(
        self, request: Request, body: ResourcesSeedSessionRequest, task_model: type[BaseModel]
    ) -> ResourcesSeedSessionResponse:
        """Seed a typed session: start the task sandbox and hand it to the agent through ``sandbox_access``.

        Repeating the seed for the same episode and task returns the same sandbox. A seed for a closed
        session, or one that names another episode or task, is rejected.
        """
        if self.config.num_workers not in (None, 1):
            raise ValueError("Typed resources sessions are process-local and require num_workers=1")
        session_id = body.resources_session_id
        request.session[SESSION_ID_KEY] = session_id
        async with self._locked_task_session(session_id) as record:
            if record.closed_episode_id is not None:
                raise ValueError(f"Resources session is already closed: {session_id}")
            if record.identity is not None:
                if record.identity != (body.episode_id, body.task_id):
                    raise ValueError("resources_session_id is already bound to another episode or task")
                return await self._task_sandbox_response(session_id, record)
            row_task_id = _row_task_id(body.task_data)
            if row_task_id is not None and row_task_id != body.task_id.task_id:
                raise ValueError(f"TaskId {body.task_id.task_id!r} does not match the task row's {row_task_id!r}")
            task = task_model.model_validate(body.task_data)
            # A seed whose cleanup failed may have left its sandbox registered; never orphan it.
            await self._stop_task_sandbox(session_id)
            try:
                record.workdir = await self._start_task_sandbox(session_id, task)
                response = await self._task_sandbox_response(session_id, record)
            except BaseException:
                # The agent cannot reach a sandbox it was never handed, so stop it here.
                try:
                    await self._stop_task_sandbox(session_id)
                except Exception:
                    LOG.exception("Failed to stop the task sandbox of failed seed %s", session_id)
                raise
            record.identity = (body.episode_id, body.task_id)
            self._open_task_sessions[session_id] = monotonic() + self._session_lifetime_s()
            return response

    async def _task_sandbox_response(self, session_id: str, record: _TaskSession) -> ResourcesSeedSessionResponse:
        sandbox = self._session_id_to_sandbox.get(session_id)
        if sandbox is None or record.verified:
            raise ValueError(f"The task sandbox of {session_id} is no longer available")
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
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        """Stop the session's task sandbox unless verify already did, and fence later seeds.

        A failed stop raises without fencing, and keeps the sandbox so a retried close stops it.
        """
        session_id = body.resources_session_id
        async with self._locked_task_session(session_id) as record:
            if record.closed_episode_id is not None:
                if body.episode_id != record.closed_episode_id:
                    raise ValueError("episode_id does not match the closed resources session")
            else:
                if record.identity is not None and body.episode_id != record.identity[0]:
                    raise ValueError("episode_id does not match the seeded resources session")
                await self._stop_task_sandbox(session_id)
                record.closed_episode_id = body.episode_id
                self._open_task_sessions.pop(session_id, None)
                self._closed_task_sessions[session_id] = monotonic() + CLOSED_SESSION_RETENTION_S
            request.session.pop(SESSION_ID_KEY, None)
            return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def _stop_task_sandbox(self, session_id: str) -> None:
        sandbox = self._session_id_to_sandbox.get(session_id)
        if sandbox is not None:
            # Forget the sandbox only once it stopped, so a failed stop can be retried.
            await _stop(sandbox)
            self._session_id_to_sandbox.pop(session_id, None)
        self._forget_task_sandbox_state(session_id)

    def _claim_task_sandbox(self, session_id: str) -> None:
        """Mark a typed session's task sandbox as consumed by verify, and reject a repeated verify.

        A legacy session, seeded through ``/run``, has no typed record and is left as it was.
        """
        record = self._task_sessions.get(session_id)
        if record is None:
            return
        if record.verified or record.closed_episode_id is not None:
            raise ValueError(f"The task sandbox of {session_id} was already verified or closed")
        record.verified = True

    async def _release_task_sandbox(self, session_id: str, sandbox: AsyncSandbox) -> None:
        """Stop a task sandbox verify already unregistered.

        When the stop fails for an open typed session, the sandbox is registered again so its close retries.
        """
        try:
            await _stop(sandbox)
        except BaseException as error:
            record = self._task_sessions.get(session_id)
            if record is not None and record.closed_episode_id is None:
                self._session_id_to_sandbox[session_id] = sandbox
            if not isinstance(error, Exception):
                raise
            LOG.exception("Failed to stop the task sandbox of session %s", session_id)


async def _stop(sandbox: AsyncSandbox) -> None:
    """Stop a sandbox within ``STOP_TIMEOUT_S``, even while the request that owns it is being cancelled.

    A client that disconnects cancels its request handler, and every later await in it, so an unshielded
    cleanup stop would never run and the sandbox would outlive every reference to it.
    """
    with anyio.CancelScope(shield=True), anyio.fail_after(STOP_TIMEOUT_S):
        await sandbox.stop()


def _row_task_id(task_data: dict[str, JsonValue]) -> str | None:
    """The task ID that task materialization derives from the row, when the row names one."""
    return next((str(task_data[key]) for key in TASK_ID_FIELDS if task_data.get(key) is not None), None)
