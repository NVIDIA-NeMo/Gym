# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral in-process web session lifecycle and concurrency control."""

from __future__ import annotations

import asyncio
import inspect
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
from nemo_gym.web.browser_session import (
    BrowserSessionError,
    BrowserSessionHandle,
    BrowserSessionProvider,
    BrowserSessionSpec,
    RenewableBrowserSessionProvider,
    create_browser_session_provider,
)
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
    [WebResourcesServerConfig, str, WebArtifactStore, BrowserSessionHandle],
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
        browser_session_provider: BrowserSessionProvider | None = None,
    ) -> None:
        self.config = config
        self._backend_factory = backend_factory
        self._site_pool = site_pool or self._make_site_pool(config)
        self._browser_session_provider = browser_session_provider or create_browser_session_provider(
            config.browser_session_provider
        )
        self._artifacts = WebArtifactStore(
            config.resolved_artifact_dir(),
            inline_screenshots=config.inline_screenshots,
        )
        self._sessions: dict[str, WebSessionState] = {}
        self._creating: set[str] = set()
        self._lock = asyncio.Lock()
        self._reaper_task: asyncio.Task[None] | None = None
        self._browser_heartbeat_task: asyncio.Task[None] | None = None
        self._late_browser_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._failed_seed_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._session_cleanup_tasks: dict[str, asyncio.Task[bool]] = {}
        self._browser_leases_acquired = 0
        self._browser_leases_released = 0
        self._browser_release_failures = 0
        self._browser_heartbeat_failures = 0
        self._last_browser_provider_error = ""
        # A supplied runner is a shared override for lightweight unit tests or
        # runtimes that do not own thread-affine browser state. By default each
        # live browser session gets one dedicated worker: Playwright calls for
        # that session stay on one thread, while a slow reset cannot serialize
        # every other session behind the same executor.
        self._shared_operation_runner = operation_runner
        self._started_at = time.time()

    async def start(self) -> None:
        if self._reaper_task is not None:
            return
        await self._call_optional_provider_lifecycle("start")
        self._reaper_task = asyncio.create_task(
            self._reaper_loop(),
            name="web-session-reaper",
        )
        if isinstance(self._browser_session_provider, RenewableBrowserSessionProvider):
            self._browser_heartbeat_task = asyncio.create_task(
                self._browser_heartbeat_loop(),
                name="web-browser-lease-heartbeat",
            )

    async def stop(self) -> None:
        for task in (self._reaper_task, self._browser_heartbeat_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._reaper_task = None
        self._browser_heartbeat_task = None
        async with self._lock:
            session_ids = list(self._sessions)
        await asyncio.gather(
            *(self.close_session(session_id) for session_id in session_ids),
            return_exceptions=True,
        )
        if self._session_cleanup_tasks:
            await asyncio.gather(
                *tuple(self._session_cleanup_tasks.values()),
                return_exceptions=True,
            )
        if self._failed_seed_cleanup_tasks:
            await asyncio.gather(
                *tuple(self._failed_seed_cleanup_tasks),
                return_exceptions=True,
            )
        if self._late_browser_cleanup_tasks:
            done, pending = await asyncio.wait(
                self._late_browser_cleanup_tasks,
                timeout=self.config.browser_release_timeout_seconds,
            )
            for task in done:
                try:
                    task.result()
                except Exception:  # noqa: BLE001
                    LOG.exception("Late browser-acquire cleanup failed during shutdown")
            if pending:
                LOG.error(
                    "event=browser_late_cleanup_pending_on_shutdown count=%d provider=%s",
                    len(pending),
                    getattr(
                        self._browser_session_provider,
                        "name",
                        type(self._browser_session_provider).__name__,
                    ),
                )
        if self._shared_operation_runner is not None:
            await self._shared_operation_runner.close()
        await self._call_optional_provider_lifecycle("close")

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
        browser_lease: BrowserSessionHandle | None = None
        backend: WebEnvironmentBackend | None = None
        operation_runner: WebOperationRunner | None = None
        try:
            lease = await self._site_pool.acquire(session_id, body.task)
            browser_lease = await self._acquire_browser_lease(session_id, body.task)
            backend = self._backend_factory(self.config, session_id, self._artifacts, browser_lease)
            operation_runner = self._shared_operation_runner or self._make_operation_runner(session_id)
            observation, seed_info = await self._reset_backend(operation_runner, backend, body.task)
            now = time.time()
            state = WebSessionState(
                session_id=session_id,
                task=body.task,
                backend=backend,
                browser_lease=browser_lease,
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
                "event=web_session_seed_complete session=%s benchmark=%s task=%s site_lease=%s "
                "browser_lease=%s browser_provider=%s browser_transport=%s isolated=%s elapsed_seconds=%.3f",
                session_id,
                body.task.benchmark.value,
                body.task.task_id,
                lease.lease_id,
                browser_lease.session_id or session_id,
                browser_lease.provider_name or type(self._browser_session_provider).__name__,
                browser_lease.transport,
                lease.isolated,
                time.monotonic() - started,
            )
            return self._seed_response(state)
        # Client-side seed timeouts cancel this coroutine.  CancelledError is a
        # BaseException on supported Python versions, so catching only
        # Exception leaks the session ID in _creating (and can leak an acquired
        # site lease).  That permanently consumes admission capacity and turns
        # every later rollout into a fast 503 until the server is restarted.
        except BaseException as seed_error:
            LOG.exception(
                "event=web_session_seed_failed session=%s benchmark=%s task=%s elapsed_seconds=%.3f",
                session_id,
                body.task.benchmark.value,
                body.task.task_id,
                time.monotonic() - started,
            )
            # Own cleanup independently from the request task. A training
            # worker may cancel the seed RPC while reset is still unwinding;
            # a second cancellation must not strand a browser or site lease.
            cleanup_task = asyncio.create_task(
                self._cleanup_failed_seed(
                    session_id=session_id,
                    backend=backend,
                    operation_runner=operation_runner,
                    browser_lease=browser_lease,
                    site_lease=lease,
                ),
                name=f"web-session-failed-seed-cleanup-{session_id}",
            )
            self._failed_seed_cleanup_tasks.add(cleanup_task)
            cleanup_task.add_done_callback(self._failed_seed_cleanup_tasks.discard)
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                # The cleanup task remains strongly referenced and continues.
                pass
            raise seed_error

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
        created = False
        async with self._lock:
            cleanup_task = self._session_cleanup_tasks.get(session_id)
            if cleanup_task is None:
                state = self._sessions.pop(session_id, None)
                self._creating.discard(session_id)
                if state is not None:
                    cleanup_task = asyncio.create_task(
                        self._close_state(state),
                        name=f"web-session-close-{session_id}",
                    )
                    self._session_cleanup_tasks[session_id] = cleanup_task
                    created = True
        if cleanup_task is None:
            return True

        if created:

            def forget(completed: asyncio.Task[bool]) -> None:
                if self._session_cleanup_tasks.get(session_id) is completed:
                    self._session_cleanup_tasks.pop(session_id, None)

            cleanup_task.add_done_callback(forget)
        # A rollout cancellation must not cancel browser/provider cleanup. The
        # task remains strongly referenced until its done callback executes.
        return await asyncio.shield(cleanup_task)

    async def _close_state(self, state: WebSessionState) -> bool:
        """Close one detached session while attempting every cleanup layer."""

        healthy = state.status != "error"
        state.status = "closing"
        try:
            async with state.lock:
                await self._run_backend(state.operation_runner, state.backend.close)
        except Exception:  # noqa: BLE001
            healthy = False
            LOG.exception("Web backend close failed for session=%s", state.session_id)
        finally:
            if state.operation_runner is not self._shared_operation_runner:
                try:
                    await state.operation_runner.close()
                except Exception:  # noqa: BLE001
                    healthy = False
                    LOG.exception("Web operation runner close failed for session=%s", state.session_id)
            browser_released = await self._release_browser_lease(state.browser_lease)
            healthy = healthy and browser_released
            try:
                await self._site_pool.release(state.site_lease, healthy=healthy)
            except Exception:  # noqa: BLE001
                healthy = False
                LOG.exception("Web site lease release failed for session=%s", state.session_id)
        LOG.info(
            "event=web_session_close session=%s benchmark=%s task=%s site_lease=%s "
            "browser_lease=%s browser_provider=%s healthy=%s",
            state.session_id,
            state.task.benchmark.value,
            state.task.task_id,
            state.site_lease.lease_id,
            state.browser_lease.session_id or state.session_id,
            state.browser_lease.provider_name or type(self._browser_session_provider).__name__,
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
            browser_lease_id=state.browser_lease.session_id,
            browser_provider=state.browser_lease.provider_name,
            browser_transport=state.browser_lease.transport,
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
                "browser_provider": {
                    "name": getattr(
                        self._browser_session_provider,
                        "name",
                        type(self._browser_session_provider).__name__,
                    ),
                    "active_leases": self._browser_leases_acquired - self._browser_leases_released,
                    "acquired": self._browser_leases_acquired,
                    "released": self._browser_leases_released,
                    "release_failures": self._browser_release_failures,
                    "heartbeat_failures": self._browser_heartbeat_failures,
                    "last_error": self._last_browser_provider_error or None,
                },
            }

    async def _get_session(self, session_id: str) -> WebSessionState:
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                raise SessionNotFoundError(session_id)
            state.last_access_at = time.time()
            return state

    async def _cleanup_failed_seed(
        self,
        *,
        session_id: str,
        backend: WebEnvironmentBackend | None,
        operation_runner: WebOperationRunner | None,
        browser_lease: BrowserSessionHandle | None,
        site_lease: SiteLease | None,
    ) -> None:
        """Attempt every cleanup layer after a failed or cancelled seed."""

        if backend is not None and operation_runner is not None:
            try:
                await self._run_backend(operation_runner, backend.close)
            except Exception:  # noqa: BLE001
                LOG.exception("Web backend cleanup failed after seed error session=%s", session_id)
        if operation_runner is not None and operation_runner is not self._shared_operation_runner:
            try:
                await operation_runner.close()
            except Exception:  # noqa: BLE001
                LOG.exception("Web operation runner cleanup failed after seed error session=%s", session_id)
        if browser_lease is not None:
            await self._release_browser_lease(browser_lease)
        if site_lease is not None:
            try:
                await self._site_pool.release(site_lease, healthy=False)
            except Exception:  # noqa: BLE001
                LOG.exception("Web site lease cleanup failed after seed error session=%s", session_id)
        async with self._lock:
            self._creating.discard(session_id)

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

    async def _acquire_browser_lease(self, session_id: str, task: WebTask) -> BrowserSessionHandle:
        provider_name = getattr(
            self._browser_session_provider,
            "name",
            type(self._browser_session_provider).__name__,
        )
        spec = BrowserSessionSpec(
            metadata={
                "rollout_session_id": session_id,
                "benchmark": task.benchmark.value,
                "task_id": task.task_id,
            },
            provider_options=dict(self.config.browser_session_options),
            lease_ttl_seconds=self.config.browser_lease_ttl_seconds,
        )
        started = time.monotonic()
        LOG.info(
            "event=browser_lease_acquire_start session=%s benchmark=%s task=%s provider=%s ttl_seconds=%d",
            session_id,
            task.benchmark.value,
            task.task_id,
            provider_name,
            spec.lease_ttl_seconds,
        )
        acquire_task = asyncio.create_task(
            self._browser_session_provider.acquire(spec),
            name=f"web-browser-acquire-{session_id}",
        )
        try:
            handle = await asyncio.wait_for(
                asyncio.shield(acquire_task),
                timeout=self.config.browser_acquire_timeout_seconds,
            )
        except TimeoutError as exc:
            self._last_browser_provider_error = f"acquire timeout from {provider_name}"
            self._schedule_late_acquire_cleanup(acquire_task, provider_name, session_id)
            raise CapacityUnavailableError(
                f"browser provider {provider_name!r} did not acquire a session within "
                f"{self.config.browser_acquire_timeout_seconds:.1f}s"
            ) from exc
        except asyncio.CancelledError:
            self._schedule_late_acquire_cleanup(acquire_task, provider_name, session_id)
            raise
        except BrowserSessionError as exc:
            self._last_browser_provider_error = str(exc)
            raise CapacityUnavailableError(f"browser provider {provider_name!r} is unavailable: {exc}") from exc
        if not isinstance(handle, BrowserSessionHandle):
            raise TypeError(f"browser provider {provider_name!r} returned an invalid handle")
        if handle.provider_name is None:
            handle.provider_name = provider_name
        if handle.session_id is None:
            handle.session_id = session_id
        self._browser_leases_acquired += 1
        LOG.info(
            "event=browser_lease_acquire_complete session=%s benchmark=%s task=%s provider=%s "
            "lease=%s transport=%s elapsed_seconds=%.3f",
            session_id,
            task.benchmark.value,
            task.task_id,
            handle.provider_name,
            handle.session_id,
            handle.transport,
            time.monotonic() - started,
        )
        return handle

    def _schedule_late_acquire_cleanup(
        self,
        acquire_task: asyncio.Task[BrowserSessionHandle],
        provider_name: str,
        session_id: str,
    ) -> None:
        """Own an acquire that outlived its rollout and release any late handle.

        A provider may implement ``acquire`` with a blocking SDK delegated to a
        thread. Cancelling the awaiting coroutine cannot stop that SDK call.
        Keeping the task alive closes the race between caller timeout and the
        remote session becoming billable.
        """

        cleanup_task = asyncio.create_task(
            self._release_late_acquire(acquire_task, provider_name, session_id),
            name=f"web-browser-late-cleanup-{session_id}",
        )
        self._late_browser_cleanup_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._late_browser_cleanup_tasks.discard)

    async def _release_late_acquire(
        self,
        acquire_task: asyncio.Task[BrowserSessionHandle],
        provider_name: str,
        session_id: str,
    ) -> None:
        started = time.monotonic()
        try:
            handle = await acquire_task
        except asyncio.CancelledError:
            LOG.warning(
                "event=browser_late_acquire_cancelled session=%s provider=%s",
                session_id,
                provider_name,
            )
            return
        except Exception as exc:  # noqa: BLE001
            LOG.warning(
                "event=browser_late_acquire_failed session=%s provider=%s error_type=%s elapsed_seconds=%.3f",
                session_id,
                provider_name,
                type(exc).__name__,
                time.monotonic() - started,
            )
            return
        if not isinstance(handle, BrowserSessionHandle):
            LOG.error(
                "event=browser_late_acquire_invalid_handle session=%s provider=%s elapsed_seconds=%.3f",
                session_id,
                provider_name,
                time.monotonic() - started,
            )
            return
        if handle.provider_name is None:
            handle.provider_name = provider_name
        if handle.session_id is None:
            handle.session_id = session_id
        LOG.warning(
            "event=browser_late_acquire_releasing session=%s provider=%s lease=%s transport=%s elapsed_seconds=%.3f",
            session_id,
            provider_name,
            handle.session_id,
            handle.transport,
            time.monotonic() - started,
        )
        await self._release_browser_lease(handle, count_acquire=True)

    async def _release_browser_lease(
        self,
        handle: BrowserSessionHandle,
        *,
        count_acquire: bool = False,
    ) -> bool:
        provider_name = handle.provider_name or getattr(
            self._browser_session_provider,
            "name",
            type(self._browser_session_provider).__name__,
        )
        started = time.monotonic()
        if count_acquire:
            self._browser_leases_acquired += 1
        try:
            await asyncio.wait_for(
                self._browser_session_provider.release(handle),
                timeout=self.config.browser_release_timeout_seconds,
            )
        except asyncio.CancelledError:
            # Shutdown cancellation must not turn an external browser lease
            # into an orphan.  Run the bounded release in its own task and
            # shield it from the caller's cancellation before propagating.
            cleanup = asyncio.create_task(
                self._release_browser_lease_after_cancellation(handle),
                name=f"web-browser-release-{handle.session_id or 'unknown'}",
            )
            await asyncio.shield(cleanup)
            raise
        except Exception as exc:  # Cleanup must be visible but must not replace an episode verdict.
            self._browser_release_failures += 1
            self._last_browser_provider_error = f"{type(exc).__name__}: {exc}"
            LOG.exception(
                "event=browser_lease_release_failed provider=%s lease=%s transport=%s elapsed_seconds=%.3f",
                provider_name,
                handle.session_id or "unknown",
                handle.transport,
                time.monotonic() - started,
            )
            return False
        self._browser_leases_released += 1
        LOG.info(
            "event=browser_lease_release_complete provider=%s lease=%s transport=%s elapsed_seconds=%.3f",
            provider_name,
            handle.session_id or "unknown",
            handle.transport,
            time.monotonic() - started,
        )
        return True

    async def _release_browser_lease_after_cancellation(self, handle: BrowserSessionHandle) -> bool:
        """Release one lease after its parent rollout has been cancelled."""

        try:
            await asyncio.wait_for(
                self._browser_session_provider.release(handle),
                timeout=self.config.browser_release_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            self._browser_release_failures += 1
            self._last_browser_provider_error = f"{type(exc).__name__}: {exc}"
            LOG.exception(
                "event=browser_lease_release_failed_after_cancellation provider=%s lease=%s transport=%s",
                handle.provider_name or type(self._browser_session_provider).__name__,
                handle.session_id or "unknown",
                handle.transport,
            )
            return False
        self._browser_leases_released += 1
        LOG.info(
            "event=browser_lease_release_complete_after_cancellation provider=%s lease=%s transport=%s",
            handle.provider_name or type(self._browser_session_provider).__name__,
            handle.session_id or "unknown",
            handle.transport,
        )
        return True

    async def _call_optional_provider_lifecycle(self, method_name: str) -> None:
        method = getattr(self._browser_session_provider, method_name, None)
        if method is None:
            return
        result = method()
        if inspect.isawaitable(result):
            await result

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
                "browser_lease_id": state.browser_lease.session_id,
                "browser_provider": state.browser_lease.provider_name,
                "browser_transport": state.browser_lease.transport,
            },
        )

    async def _browser_heartbeat_loop(self) -> None:
        provider = self._browser_session_provider
        if not isinstance(provider, RenewableBrowserSessionProvider):
            return
        while True:
            await asyncio.sleep(self.config.browser_heartbeat_interval_seconds)
            async with self._lock:
                states = list(self._sessions.values())
            if not states:
                continue
            results = await asyncio.gather(
                *(
                    asyncio.wait_for(
                        provider.heartbeat(state.browser_lease),
                        timeout=self.config.browser_heartbeat_timeout_seconds,
                    )
                    for state in states
                ),
                return_exceptions=True,
            )
            expired_sessions: list[str] = []
            for state, result in zip(states, results, strict=True):
                if not isinstance(result, BaseException):
                    state.browser_heartbeat_failures = 0
                    continue
                state.browser_heartbeat_failures += 1
                self._browser_heartbeat_failures += 1
                self._last_browser_provider_error = f"{type(result).__name__}: {result}"
                LOG.error(
                    "event=browser_lease_heartbeat_failed session=%s benchmark=%s task=%s provider=%s "
                    "lease=%s error_type=%s",
                    state.session_id,
                    state.task.benchmark.value,
                    state.task.task_id,
                    state.browser_lease.provider_name or type(provider).__name__,
                    state.browser_lease.session_id or "unknown",
                    type(result).__name__,
                    exc_info=(type(result), result, result.__traceback__),
                )
                if state.browser_heartbeat_failures >= self.config.browser_heartbeat_failure_limit:
                    state.status = "error"
                    expired_sessions.append(state.session_id)
                    LOG.error(
                        "event=browser_lease_heartbeat_limit session=%s benchmark=%s task=%s consecutive_failures=%d",
                        state.session_id,
                        state.task.benchmark.value,
                        state.task.task_id,
                        state.browser_heartbeat_failures,
                    )
            if expired_sessions:
                await asyncio.gather(
                    *(self.close_session(session_id) for session_id in expired_sessions),
                    return_exceptions=True,
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
