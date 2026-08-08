# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process BrowserGym session lifecycle and concurrency control."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

from nemo_gym.web.models import WebObservation, WebTask, WebVerifierResult
from nemo_gym.web.protocol import WebEnvironmentBackend
from resources_servers.browsergym_web.artifacts import WebArtifactStore
from resources_servers.browsergym_web.backend import BrowserGymBackend
from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig
from resources_servers.browsergym_web.models import (
    WebEvaluateResponse,
    WebResetRequest,
    WebSeedSessionRequest,
    WebSeedSessionResponse,
    WebSessionStatusResponse,
    WebStepRequest,
    WebStepResponse,
)
from resources_servers.browsergym_web.site_pool import (
    LocalSiteLockPool,
    SiteLease,
    SitePool,
    UnmanagedSitePool,
)


LOG = logging.getLogger("nemo_gym.resources_servers.browsergym_web")


class SessionNotFoundError(KeyError):
    pass


class SessionConflictError(RuntimeError):
    pass


class CapacityUnavailableError(RuntimeError):
    pass


class BenchmarkPreconditionError(RuntimeError):
    """A deterministic task/environment setup failure for the current deployment."""


BackendFactory = Callable[
    [BrowserGymWebResourcesServerConfig, str, WebArtifactStore],
    WebEnvironmentBackend,
]


@dataclass
class WebSessionState:
    session_id: str
    task: WebTask
    backend: WebEnvironmentBackend
    site_lease: SiteLease
    observation: WebObservation
    seed_info: dict[str, Any]
    created_at: float
    last_access_at: float
    status: str = "ready"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    operations: OrderedDict[str, WebStepResponse] = field(default_factory=OrderedDict)
    verifier_result: WebVerifierResult | None = None


class BrowserGymSessionManager:
    """Bind a signed Gym session cookie to one live Playwright context."""

    def __init__(
        self,
        config: BrowserGymWebResourcesServerConfig,
        *,
        backend_factory: BackendFactory = BrowserGymBackend,
        site_pool: SitePool | None = None,
    ) -> None:
        self.config = config
        self._backend_factory = backend_factory
        self._site_pool = site_pool or self._make_site_pool(config)
        self._artifacts = WebArtifactStore(
            config.resolved_artifact_dir(),
            inline_screenshots=config.inline_screenshots,
        )
        self._sessions: dict[str, WebSessionState] = {}
        self._creating: set[str] = set()
        self._lock = asyncio.Lock()
        self._reaper_task: asyncio.Task[None] | None = None
        # BrowserGym 0.14.x owns one process-global Playwright Sync API
        # instance. Playwright's greenlet is bound to the thread that created
        # it, so every reset/step/evaluate/close call must use that same
        # thread. A regular asyncio.to_thread() pool can move consecutive
        # calls between workers and fails under concurrent session creation
        # with ``greenlet.error: cannot switch to a different thread``.
        self._browser_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="browsergym-playwright",
        )
        self._browser_executor_shutdown = False
        self._started_at = time.time()

    async def start(self) -> None:
        self._reaper_task = asyncio.create_task(
            self._reaper_loop(),
            name="browsergym-web-session-reaper",
        )

    async def stop(self) -> None:
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            try:
                await self._reaper_task
            except asyncio.CancelledError:
                pass
            self._reaper_task = None
        async with self._lock:
            session_ids = list(self._sessions)
        await asyncio.gather(
            *(self.close_session(session_id) for session_id in session_ids),
            return_exceptions=True,
        )
        if not self._browser_executor_shutdown:
            self._browser_executor.shutdown(wait=True, cancel_futures=True)
            self._browser_executor_shutdown = True

    async def seed_session(self, session_id: str, body: WebSeedSessionRequest) -> WebSeedSessionResponse:
        self._validate_task(body.task)
        async with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                self._require_same_task(existing.task, body.task, session_id)
                existing.last_access_at = time.time()
                return self._seed_response(existing)
            if session_id in self._creating:
                raise SessionConflictError(f"session {session_id!r} is already being created")
            if len(self._sessions) + len(self._creating) >= self.config.max_sessions:
                raise CapacityUnavailableError(
                    f"BrowserGym session capacity is full (max_sessions={self.config.max_sessions})"
                )
            self._creating.add(session_id)

        lease: SiteLease | None = None
        backend: WebEnvironmentBackend | None = None
        try:
            lease = await self._site_pool.acquire(session_id, body.task)
            backend = self._backend_factory(self.config, session_id, self._artifacts)
            observation, seed_info = await self._reset_backend(backend, body.task)
            now = time.time()
            state = WebSessionState(
                session_id=session_id,
                task=body.task,
                backend=backend,
                site_lease=lease,
                observation=observation,
                seed_info=seed_info,
                created_at=now,
                last_access_at=now,
            )
            async with self._lock:
                self._creating.discard(session_id)
                self._sessions[session_id] = state
            LOG.info(
                "Seeded BrowserGym session=%s benchmark=%s task=%s lease=%s isolated=%s",
                session_id,
                body.task.benchmark.value,
                body.task.task_id,
                lease.lease_id,
                lease.isolated,
            )
            return self._seed_response(state)
        # Client-side seed timeouts cancel this coroutine.  CancelledError is a
        # BaseException on supported Python versions, so catching only
        # Exception leaks the session ID in _creating (and can leak an acquired
        # site lease).  That permanently consumes admission capacity and turns
        # every later rollout into a fast 503 until the server is restarted.
        except BaseException:
            if backend is not None:
                try:
                    await self._run_backend(backend.close)
                except Exception:  # noqa: BLE001
                    LOG.exception("Cleanup failed after BrowserGym session creation error")
            if lease is not None:
                await self._site_pool.release(lease, healthy=False)
            async with self._lock:
                self._creating.discard(session_id)
            raise

    async def reset_session(self, session_id: str, body: WebResetRequest) -> WebSeedSessionResponse:
        state = await self._get_session(session_id)
        self._validate_task(body.task)
        self._require_same_task(state.task, body.task, session_id)
        async with state.lock:
            state.status = "resetting"
            try:
                observation, seed_info = await self._reset_backend(state.backend, body.task)
                state.task = body.task
                state.observation = observation
                state.seed_info = seed_info
                state.operations.clear()
                state.verifier_result = None
                state.status = "ready"
                state.last_access_at = time.time()
                return self._seed_response(state)
            except Exception:
                state.status = "error"
                raise

    async def observe(self, session_id: str) -> WebObservation:
        state = await self._get_session(session_id)
        async with state.lock:
            observation = await self._run_backend(state.backend.observe)
            state.observation = observation
            state.last_access_at = time.time()
            return observation

    async def step(self, session_id: str, body: WebStepRequest) -> WebStepResponse:
        state = await self._get_session(session_id)
        async with state.lock:
            cached = state.operations.get(body.operation_id)
            if cached is not None:
                state.operations.move_to_end(body.operation_id)
                state.last_access_at = time.time()
                return cached
            if state.verifier_result is not None:
                raise SessionConflictError(f"session {session_id!r} has already been evaluated")
            state.status = "stepping"
            try:
                result = await self._run_backend(state.backend.step, body.action)
                response = WebStepResponse(operation_id=body.operation_id, **result.model_dump())
                state.observation = result.observation
                state.operations[body.operation_id] = response
                while len(state.operations) > 128:
                    state.operations.popitem(last=False)
                state.status = "finished" if result.terminated or result.truncated else "ready"
                state.last_access_at = time.time()
                return response
            except Exception:
                state.status = "error"
                raise

    async def evaluate(self, session_id: str, final_answer: str | None = None) -> WebEvaluateResponse:
        state = await self._get_session(session_id)
        async with state.lock:
            if state.verifier_result is not None:
                state.last_access_at = time.time()
                return WebEvaluateResponse(result=state.verifier_result)
            state.status = "evaluating"
            try:
                result = await self._run_backend(state.backend.evaluate, final_answer)
                state.verifier_result = result
                state.status = "evaluated"
                state.last_access_at = time.time()
                return WebEvaluateResponse(result=result)
            except Exception:
                state.status = "error"
                raise

    async def close_session(self, session_id: str) -> bool:
        async with self._lock:
            state = self._sessions.pop(session_id, None)
            self._creating.discard(session_id)
        if state is None:
            return True

        healthy = state.status != "error"
        state.status = "closing"
        try:
            async with state.lock:
                await self._run_backend(state.backend.close)
        except Exception:  # noqa: BLE001
            healthy = False
            LOG.exception("BrowserGym backend close failed for session=%s", session_id)
        finally:
            await self._site_pool.release(state.site_lease, healthy=healthy)
        LOG.info("Closed BrowserGym session=%s lease=%s", session_id, state.site_lease.lease_id)
        return True

    async def session_status(self, session_id: str) -> WebSessionStatusResponse:
        state = await self._get_session(session_id)
        return WebSessionStatusResponse(
            session_id=state.session_id,
            task_id=state.task.task_id,
            benchmark=state.task.benchmark.value,
            status=state.status,
            created_at=state.created_at,
            last_access_at=state.last_access_at,
            site_lease_id=state.site_lease.lease_id,
        )

    async def health(self) -> dict[str, Any]:
        site_pool = await self._site_pool.health()
        async with self._lock:
            return {
                "status": "ok",
                "uptime_seconds": max(0.0, time.time() - self._started_at),
                "sessions": len(self._sessions),
                "creating": len(self._creating),
                "capacity": self.config.max_sessions,
                "site_pool": site_pool,
            }

    async def _get_session(self, session_id: str) -> WebSessionState:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError(session_id)
            state.last_access_at = time.time()
            return state

    async def _run_backend(self, operation: Callable[..., Any], *args: Any) -> Any:
        """Run a BrowserGym operation on its thread-affine Playwright worker."""

        if self._browser_executor_shutdown:
            raise RuntimeError("BrowserGym session manager has already stopped")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._browser_executor, partial(operation, *args))

    async def _reset_backend(
        self, backend: WebEnvironmentBackend, task: WebTask
    ) -> tuple[WebObservation, dict[str, Any]]:
        try:
            return await self._run_backend(backend.reset, task)
        except ValueError as exc:
            # BrowserGym uses ValueError for deterministic reset preconditions,
            # including invalid task configuration and unavailable VWA input
            # images. Retrying the same rollout against the unchanged benchmark
            # deployment cannot repair those conditions.
            raise BenchmarkPreconditionError(str(exc)) from exc

    @staticmethod
    def _make_site_pool(config: BrowserGymWebResourcesServerConfig) -> SitePool:
        if config.site_pool_mode == "local_locks":
            return LocalSiteLockPool()
        return UnmanagedSitePool()

    def _validate_task(self, task: WebTask) -> None:
        if task.benchmark not in self.config.allowed_benchmarks:
            raise ValueError(f"benchmark {task.benchmark.value!r} is disabled by server configuration")
        if task.runtime_profile.value != "browsergym":
            raise ValueError("this resource server only supports the browsergym runtime profile")

    @staticmethod
    def _require_same_task(current: WebTask, requested: WebTask, session_id: str) -> None:
        if current.benchmark != requested.benchmark or current.task_id != requested.task_id:
            raise SessionConflictError(
                f"session {session_id!r} already owns {current.benchmark.value}/{current.task_id}"
            )

    @staticmethod
    def _seed_response(state: WebSessionState) -> WebSeedSessionResponse:
        return WebSeedSessionResponse(
            session_id=state.session_id,
            task_id=state.task.task_id,
            status=state.status,
            observation=state.observation,
            info=state.seed_info
            | {
                "site_lease_id": state.site_lease.lease_id,
                "site_isolated": state.site_lease.isolated,
                "site_lease_metadata": state.site_lease.metadata,
            },
        )

    async def _reaper_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.reaper_interval_seconds)
            cutoff = time.time() - self.config.session_ttl_seconds
            async with self._lock:
                stale = [session_id for session_id, state in self._sessions.items() if state.last_access_at < cutoff]
            if stale:
                LOG.warning("Reaping %d expired BrowserGym session(s)", len(stale))
                await asyncio.gather(
                    *(self.close_session(session_id) for session_id in stale),
                    return_exceptions=True,
                )
