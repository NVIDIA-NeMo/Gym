# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources-owned task sandboxes for the typed single-agent-turn protocol.

The typed SWE protocol follows Gym #4302/#4304 (Apache-2.0):
https://github.com/NVIDIA-NeMo/Gym/pull/4302
Revision: 652d384b0d8d668911c7bbb699cc2be8bada5ca3.
Verification replay and sandbox ownership survive a failed close independently.
"""

import asyncio
import hashlib
import json
import logging
from abc import abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

import anyio
from fastapi import FastAPI, Request
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFence,
    SimpleResourcesServer,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.task_materialization import TASK_ID_FIELDS


LOG = logging.getLogger(__name__)
STOP_TIMEOUT_S = 120.0
VerifyResponseT = TypeVar("VerifyResponseT", bound=BaseVerifyResponse)


@dataclass
class _TaskSession:
    episode_id: EpisodeId
    task_id: TaskId
    seed: ResourcesSeedSessionResponse | None = None
    verification: asyncio.Task[BaseVerifyResponse] | None = None
    verification_digest: str | None = None
    closing: bool = False


class SandboxSessionResourcesServer(SimpleResourcesServer):
    """Add typed seed, replayable verification and close to SWE resources servers.

    Subclasses retain their flat seed/verify API and implement task preparation and
    bookkeeping hooks. They must register a task sandbox before preparing it.
    """

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox: dict[str, AsyncSandbox] = {}
        self._task_sessions: dict[str, _TaskSession] = {}
        self._verifier_cleanup: dict[str, dict[int, AsyncSandbox]] = {}
        self._task_fence = ResourcesSessionFence(self.config.session_close_retry_window_seconds)

    @abstractmethod
    async def _start_task_sandbox(self, session_id: str, task: BaseModel) -> str:
        """Create, register and prepare the task sandbox; return its working directory."""

    def _forget_task_sandbox_state(self, session_id: str) -> None:
        """Release task bookkeeping after its sandbox has stopped."""

    async def seed_task_sandbox_session(
        self, request: Request, body: ResourcesSeedSessionRequest, task_model: type[BaseModel]
    ) -> ResourcesSeedSessionResponse:
        """Lend the owned sandbox, returning the same seed for an identical retry."""
        if self.config.num_workers not in (None, 1):
            raise ValueError("Typed SWE resources sessions require num_workers=1")
        session_id = body.resources_session_id
        request.session[SESSION_ID_KEY] = session_id
        async with self._task_fence.locked(session_id):
            self._task_fence.raise_if_closed(session_id)
            record = self._task_sessions.get(session_id)
            if record is not None:
                if (record.episode_id, record.task_id) != (body.episode_id, body.task_id):
                    raise ResourcesSessionConflictError("resources_session_id belongs to another episode or task")
                if record.closing or record.verification is not None:
                    raise ResourcesSessionConflictError("Task sandbox has already been consumed or is closing")
                if record.seed is not None:
                    return record.seed
            row_id = next((str(body.task_data[k]) for k in TASK_ID_FIELDS if body.task_data.get(k) is not None), None)
            if row_id is not None and row_id != body.task_id.task_id:
                raise ResourcesSessionConflictError("task_id does not match the task row")
            task = task_model.model_validate(body.task_data)
            # A previous failed seed may still own a sandbox whose stop failed.
            await self._stop_task_sandbox(session_id)
            record = _TaskSession(body.episode_id, body.task_id)
            self._task_sessions[session_id] = record
            try:
                workdir = await self._start_task_sandbox(session_id, task)
                record.seed = ResourcesSeedSessionResponse(
                    resources_session_id=session_id,
                    sandbox_access=SandboxAccess(
                        connection=DirectSandboxConnection(
                            provider_config_ref=self.config.sandbox_provider,
                            descriptor=await self._session_id_to_sandbox[session_id].serialize(scope="operate"),
                        ),
                        workdir=workdir,
                    ),
                )
            except BaseException:
                try:
                    await self._stop_task_sandbox(session_id)
                except Exception:
                    LOG.exception("Failed seed still owns task sandbox %s; close will retry", session_id)
                else:
                    self._task_sessions.pop(session_id, None)
                raise
            return record.seed

    async def verify_task_session(
        self, request: Request, body: BaseModel, verify: Callable[[], Awaitable[VerifyResponseT]]
    ) -> VerifyResponseT:
        """Grade once per typed session; replay only an identical verification request."""
        session_id = request.session[SESSION_ID_KEY]
        async with self._task_fence.locked(session_id):
            self._task_fence.raise_if_closed(session_id)
            record = self._task_sessions.get(session_id)
            if record is None:
                task = None  # Flat /run callers retain their existing verification behavior.
            else:
                digest = hashlib.sha256(
                    json.dumps(body.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                if record.verification is not None:
                    if digest != record.verification_digest:
                        raise ResourcesSessionConflictError("Verification retry differs from the original request")
                    task = record.verification
                else:
                    if record.closing or record.seed is None:
                        raise ResourcesSessionConflictError("Task session is closing or was not successfully seeded")
                    record.verification_digest = digest
                    task = asyncio.create_task(verify())
                    record.verification = task
        if task is None:
            return await verify()
        # A disconnected HTTP caller must not cancel the one operation consuming the patch.
        return await asyncio.shield(task)

    async def _stop_task_sandbox(self, session_id: str) -> None:
        sandbox = self._session_id_to_sandbox.get(session_id)
        if sandbox is not None:
            await _stop(sandbox)
            self._session_id_to_sandbox.pop(session_id, None)
        self._forget_task_sandbox_state(session_id)

    async def _release_task_sandbox(self, session_id: str, sandbox: AsyncSandbox) -> None:
        """Release a captured task sandbox, retaining ownership if teardown fails."""
        self._session_id_to_sandbox[session_id] = sandbox
        try:
            await self._stop_task_sandbox(session_id)
        except Exception:
            LOG.exception("Task sandbox stop failed for %s; close will retry", session_id)

    async def _release_verifier_sandbox(self, session_id: str, sandbox: AsyncSandbox | None) -> None:
        """Release a verifier without erasing its verdict on a provider stop failure."""
        if sandbox is None:
            return
        pending = self._verifier_cleanup.setdefault(session_id, {})
        pending[id(sandbox)] = sandbox
        try:
            await _stop(sandbox)
        except Exception:
            LOG.exception("Verifier sandbox stop failed for %s; close will retry", session_id)
        else:
            pending.pop(id(sandbox), None)
            if not pending:
                self._verifier_cleanup.pop(session_id, None)

    async def _close_task_session(self, session_id: str) -> None:
        record = self._task_sessions.get(session_id)
        if record is not None:
            record.closing = True
            task = record.verification
            if task is not None:
                if not task.done() and not task.cancelling():
                    task.cancel()
                # Retain the completed task/result until ALL owned sandboxes are released.
                with anyio.fail_after(STOP_TIMEOUT_S, shield=True):
                    await asyncio.shield(asyncio.gather(task, return_exceptions=True))
        await self._stop_task_sandbox(session_id)
        pending = self._verifier_cleanup.get(session_id, {})
        for key, sandbox in list(pending.items()):
            await _stop(sandbox)
            pending.pop(key)
        self._verifier_cleanup.pop(session_id, None)

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        """Stop owned sandboxes and fence the session only after successful cleanup."""
        session_id = body.resources_session_id
        async with self._task_fence.locked(session_id):
            record = self._task_sessions.get(session_id)
            expected = self._task_fence.closed_episode(session_id) or (record.episode_id if record else None)
            if expected is not None and expected != body.episode_id:
                raise ResourcesSessionConflictError("episode_id does not match the resources session")
            await self._close_task_session(session_id)
            self._task_fence.mark_closed(session_id, body.episode_id)
            self._task_sessions.pop(session_id, None)
            request.session.pop(SESSION_ID_KEY, None)
        return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def shutdown(self) -> None:
        """Retry cleanup for every remaining owned task and verifier sandbox."""
        for session_id in (
            self._task_sessions.keys() | self._session_id_to_sandbox.keys() | self._verifier_cleanup.keys()
        ):
            try:
                async with self._task_fence.locked(session_id):
                    await self._close_task_session(session_id)
            except Exception:
                LOG.exception("Shutdown could not release resources session %s", session_id)
            else:
                self._task_sessions.pop(session_id, None)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[object]:
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app


async def _stop(sandbox: AsyncSandbox) -> None:
    # Shield the actual provider call from the request's AnyIO cancellation scope.
    with anyio.fail_after(STOP_TIMEOUT_S, shield=True):
        await sandbox.stop()
