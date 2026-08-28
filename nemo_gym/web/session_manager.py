# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral in-process web session lifecycle and concurrency control."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from nemo_gym.web.api_models import (
    WebEvaluateResponse,
    WebResetRequest,
    WebSeedSessionRequest,
    WebSeedSessionResponse,
    WebSessionStatusResponse,
    WebStepRequest,
    WebStepResponse,
)
from nemo_gym.web.artifacts import WebArtifactStore
from nemo_gym.web.models import WebArtifactRef, WebObservation, WebTask
from nemo_gym.web.operation_runner import ThreadAffineWebOperationRunner, WebOperationRunner
from nemo_gym.web.protocol import WebEnvironmentBackend
from nemo_gym.web.resource_config import WebResourcesServerConfig
from nemo_gym.web.session import (
    BenchmarkPreconditionError,
    CapacityUnavailableError,
    SessionConflictError,
    SessionNotFoundError,
    WebSessionState,
)
from nemo_gym.web.site_pool import LocalSiteLockPool, SiteLease, SitePool, UnmanagedSitePool


LOG = logging.getLogger("nemo_gym.web.session_manager")


BackendFactory = Callable[
    [WebResourcesServerConfig, str, WebArtifactStore],
    WebEnvironmentBackend,
]


class WebSessionManager:
    """Bind a signed Gym session cookie to one live backend instance."""

    def __init__(
        self,
        config: WebResourcesServerConfig,
        *,
        backend_factory: BackendFactory,
        site_pool: SitePool | None = None,
        operation_runner: WebOperationRunner | None = None,
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
        # A supplied runner is a shared override for lightweight unit tests or
        # runtimes that do not own thread-affine browser state. By default each
        # live browser session gets one dedicated worker: Playwright calls for
        # that session stay on one thread, while a slow reset cannot serialize
        # every other session behind the same executor.
        self._shared_operation_runner = operation_runner
        self._started_at = time.time()

    async def start(self) -> None:
        self._reaper_task = asyncio.create_task(
            self._reaper_loop(),
            name="web-session-reaper",
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
        if self._shared_operation_runner is not None:
            await self._shared_operation_runner.close()

    async def seed_session(self, session_id: str, body: WebSeedSessionRequest) -> WebSeedSessionResponse:
        self._validate_task(body.task)
        started = time.monotonic()
        LOG.info(
            "event=web_session_seed_start session=%s benchmark=%s task=%s active=%d creating=%d capacity=%d",
            session_id,
            body.task.benchmark.value,
            body.task.task_id,
            len(self._sessions),
            len(self._creating),
            self.config.max_sessions,
        )
        async with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                self._require_same_task(existing.task, body.task, session_id)
                if existing.status != "ready":
                    raise SessionConflictError(
                        f"session {session_id!r} cannot replay seed while status={existing.status!r}"
                    )
                existing.last_access_at = time.time()
                LOG.info(
                    "event=web_session_seed_cached session=%s benchmark=%s task=%s status=%s",
                    session_id,
                    body.task.benchmark.value,
                    body.task.task_id,
                    existing.status,
                )
                return self._seed_response(existing)
            if session_id in self._creating:
                raise SessionConflictError(f"session {session_id!r} is already being created")
            if len(self._sessions) + len(self._creating) >= self.config.max_sessions:
                raise CapacityUnavailableError(
                    f"web session capacity is full (max_sessions={self.config.max_sessions})"
                )
            self._creating.add(session_id)

        lease: SiteLease | None = None
        backend: WebEnvironmentBackend | None = None
        operation_runner: WebOperationRunner | None = None
        try:
            lease = await self._site_pool.acquire(session_id, body.task)
            backend = self._backend_factory(self.config, session_id, self._artifacts)
            operation_runner = self._shared_operation_runner or self._make_operation_runner(session_id)
            observation, seed_info = await self._reset_backend(operation_runner, backend, body.task)
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
                operation_runner=operation_runner,
            )
            async with self._lock:
                self._creating.discard(session_id)
                self._sessions[session_id] = state
            LOG.info(
                "event=web_session_seed_complete session=%s benchmark=%s task=%s lease=%s isolated=%s "
                "elapsed_seconds=%.3f",
                session_id,
                body.task.benchmark.value,
                body.task.task_id,
                lease.lease_id,
                lease.isolated,
                time.monotonic() - started,
            )
            return self._seed_response(state)
        # Client-side seed timeouts cancel this coroutine.  CancelledError is a
        # BaseException on supported Python versions, so catching only
        # Exception leaks the session ID in _creating (and can leak an acquired
        # site lease).  That permanently consumes admission capacity and turns
        # every later rollout into a fast 503 until the server is restarted.
        except BaseException:
            LOG.exception(
                "event=web_session_seed_failed session=%s benchmark=%s task=%s elapsed_seconds=%.3f",
                session_id,
                body.task.benchmark.value,
                body.task.task_id,
                time.monotonic() - started,
            )
            if backend is not None and operation_runner is not None:
                try:
                    await self._run_backend(operation_runner, backend.close)
                except Exception:  # noqa: BLE001
                    LOG.exception("Cleanup failed after web session creation error")
            if operation_runner is not None and operation_runner is not self._shared_operation_runner:
                await operation_runner.close()
            if lease is not None:
                await self._site_pool.release(lease, healthy=False)
            async with self._lock:
                self._creating.discard(session_id)
            raise

    async def reset_session(self, session_id: str, body: WebResetRequest) -> WebSeedSessionResponse:
        state = await self._get_session(session_id)
        self._validate_task(body.task)
        self._require_same_task(state.task, body.task, session_id)
        cleanup_failed_session = False
        try:
            async with state.lock:
                state.status = "resetting"
                started = time.monotonic()
                LOG.info(
                    "event=web_session_reset_start session=%s benchmark=%s task=%s",
                    session_id,
                    body.task.benchmark.value,
                    body.task.task_id,
                )
                try:
                    observation, seed_info = await self._reset_backend(
                        state.operation_runner,
                        state.backend,
                        body.task,
                    )
                    state.task = body.task
                    state.observation = observation
                    state.seed_info = seed_info
                    state.operations.clear()
                    state.verifier_result = None
                    state.status = "ready"
                    state.last_access_at = time.time()
                    LOG.info(
                        "event=web_session_reset_complete session=%s benchmark=%s task=%s elapsed_seconds=%.3f",
                        session_id,
                        body.task.benchmark.value,
                        body.task.task_id,
                        time.monotonic() - started,
                    )
                    return self._seed_response(state)
                except BaseException:
                    # A failed or cancelled reset can leave browser and site
                    # state partially mutated. Mark the lease unhealthy and
                    # remove the session immediately instead of parking it
                    # until the TTL reaper runs.
                    state.status = "error"
                    cleanup_failed_session = True
                    LOG.exception(
                        "event=web_session_reset_failed session=%s benchmark=%s task=%s elapsed_seconds=%.3f",
                        session_id,
                        body.task.benchmark.value,
                        body.task.task_id,
                        time.monotonic() - started,
                    )
                    raise
        finally:
            # close_session acquires state.lock, so cleanup must happen only
            # after leaving the reset critical section.
            if cleanup_failed_session:
                await self.close_session(session_id)

    async def observe(self, session_id: str) -> WebObservation:
        state = await self._get_session(session_id)
        async with state.lock:
            observation = await self._run_backend(state.operation_runner, state.backend.observe)
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
                LOG.info(
                    "event=web_session_step_cached session=%s task=%s operation=%s",
                    session_id,
                    state.task.task_id,
                    body.operation_id,
                )
                return cached
            if state.verifier_result is not None:
                raise SessionConflictError(f"session {session_id!r} has already been evaluated")
            state.status = "stepping"
            started = time.monotonic()
            LOG.info(
                "event=web_session_step_start session=%s benchmark=%s task=%s operation=%s action=%s terminal=%s",
                session_id,
                state.task.benchmark.value,
                state.task.task_id,
                body.operation_id,
                body.action.name,
                body.action.terminal,
            )
            try:
                result = await self._run_backend(state.operation_runner, state.backend.step, body.action)
                response = WebStepResponse(operation_id=body.operation_id, **result.model_dump())
                state.observation = result.observation
                state.operations[body.operation_id] = response
                while len(state.operations) > 128:
                    state.operations.popitem(last=False)
                state.status = "finished" if result.terminated or result.truncated else "ready"
                state.last_access_at = time.time()
                LOG.info(
                    "event=web_session_step_complete session=%s task=%s operation=%s execution_ok=%s "
                    "terminated=%s truncated=%s elapsed_seconds=%.3f",
                    session_id,
                    state.task.task_id,
                    body.operation_id,
                    result.execution_ok,
                    result.terminated,
                    result.truncated,
                    time.monotonic() - started,
                )
                return response
            except Exception:
                state.status = "error"
                LOG.exception(
                    "event=web_session_step_failed session=%s task=%s operation=%s elapsed_seconds=%.3f",
                    session_id,
                    state.task.task_id,
                    body.operation_id,
                    time.monotonic() - started,
                )
                raise

    async def evaluate(self, session_id: str, final_answer: str | None = None) -> WebEvaluateResponse:
        state = await self._get_session(session_id)
        async with state.lock:
            if state.verifier_result is not None:
                state.last_access_at = time.time()
                return WebEvaluateResponse(result=state.verifier_result)
            state.status = "evaluating"
            started = time.monotonic()
            LOG.info(
                "event=web_session_evaluate_start session=%s benchmark=%s task=%s final_answer_present=%s",
                session_id,
                state.task.benchmark.value,
                state.task.task_id,
                bool(final_answer),
            )
            try:
                result = await self._run_backend(state.operation_runner, state.backend.evaluate, final_answer)
                state.verifier_result = result
                state.status = "evaluated"
                state.last_access_at = time.time()
                LOG.info(
                    "event=web_session_evaluate_complete session=%s task=%s valid_sample=%s reward=%s "
                    "failure_kind=%s elapsed_seconds=%.3f",
                    session_id,
                    state.task.task_id,
                    result.valid_sample,
                    result.reward,
                    result.failure_kind or "none",
                    time.monotonic() - started,
                )
                return WebEvaluateResponse(result=result)
            except Exception:
                state.status = "error"
                LOG.exception(
                    "event=web_session_evaluate_failed session=%s task=%s elapsed_seconds=%.3f",
                    session_id,
                    state.task.task_id,
                    time.monotonic() - started,
                )
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
                await self._run_backend(state.operation_runner, state.backend.close)
        except Exception:  # noqa: BLE001
            healthy = False
            LOG.exception("Web backend close failed for session=%s", session_id)
        finally:
            if state.operation_runner is not self._shared_operation_runner:
                await state.operation_runner.close()
            await self._site_pool.release(state.site_lease, healthy=healthy)
        LOG.info(
            "event=web_session_close session=%s benchmark=%s task=%s lease=%s healthy=%s",
            session_id,
            state.task.benchmark.value,
            state.task.task_id,
            state.site_lease.lease_id,
            healthy,
        )
        return True

    async def recording_artifacts(self, session_id: str) -> list[WebArtifactRef]:
        """Index recordings only after browser close has flushed them to disk."""

        return await asyncio.to_thread(self._artifacts.recording_artifacts, session_id)

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

    async def _run_backend(
        self,
        operation_runner: WebOperationRunner,
        operation: Callable[..., Any],
        *args: Any,
    ) -> Any:
        """Run a backend operation on its thread-affine Playwright worker."""

        return await operation_runner.run(operation, *args)

    async def _reset_backend(
        self,
        operation_runner: WebOperationRunner,
        backend: WebEnvironmentBackend,
        task: WebTask,
    ) -> tuple[WebObservation, dict[str, Any]]:
        try:
            return await self._run_backend(operation_runner, backend.reset, task)
        except ValueError as exc:
            # Backends use ValueError for deterministic task or environment
            # preconditions. Retrying against an unchanged deployment cannot
            # repair those conditions.
            raise BenchmarkPreconditionError(str(exc)) from exc

    def _make_operation_runner(self, session_id: str) -> WebOperationRunner:
        return ThreadAffineWebOperationRunner(thread_name_prefix=f"web-playwright-{session_id[:8]}")

    @staticmethod
    def _make_site_pool(config: WebResourcesServerConfig) -> SitePool:
        if config.site_pool_mode == "local_locks":
            return LocalSiteLockPool()
        return UnmanagedSitePool()

    def _validate_task(self, task: WebTask) -> None:
        if task.benchmark not in self.config.allowed_benchmarks:
            raise ValueError(f"benchmark {task.benchmark.value!r} is disabled by server configuration")

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
                LOG.warning("Reaping %d expired web session(s)", len(stale))
                await asyncio.gather(
                    *(self.close_session(session_id) for session_id in stale),
                    return_exceptions=True,
                )
