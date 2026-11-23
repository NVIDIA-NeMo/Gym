# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from fastapi import FastAPI, HTTPException, Request
from pydantic import ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import SimpleResourcesServer
from resources_servers.webarena.actions import WEB_ARENA_TOOLS
from resources_servers.webarena.actor_pool import BrowserActorPool
from resources_servers.webarena.observations import (
    build_function_output,
    build_observation_message,
)
from resources_servers.webarena.schemas import (
    WebArenaAgentVerifyRequest,
    WebArenaAgentVerifyResponse,
    WebArenaCloseRequest,
    WebArenaCloseResponse,
    WebArenaFinalizeRequest,
    WebArenaFinalizeResponse,
    WebArenaResourcesServerConfig,
    WebArenaSeedSessionRequest,
    WebArenaSeedSessionResponse,
    WebArenaStepRequest,
    WebArenaStepResponse,
    WebArenaTaskRow,
)


logger = logging.getLogger(__name__)
T = TypeVar("T")
_DEDUPLICATION_RETRY_WINDOW_SECONDS = 300.0


class SessionState(str, Enum):
    SEEDING = "seeding"
    LEASED = "leased"
    TERMINATING = "terminating"
    COMPLETED = "completed"
    FINALIZING = "finalizing"
    RECYCLING = "recycling"
    CLOSING = "closing"
    CLOSED = "closed"
    DEAD = "dead"


@dataclass
class SessionRecord:
    env_id: str
    seed_request_id: str
    actor: Any
    task: WebArenaTaskRow
    state: SessionState
    created_at: float
    last_activity_at: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_operations: int = 0
    step_count: int = 0
    reward: float = 0.0
    termination_status: str | None = None
    answer: str | None = None
    evaluator_details: dict[str, Any] = field(default_factory=dict)
    evaluation_completed: bool = False
    closed_at: float | None = None


@dataclass
class _DedupEntry:
    request_id: str
    payload_hash: str
    created_at: float
    future: asyncio.Future[Any]


class SingleFlightDeduplicator:
    def __init__(self, *, ttl_seconds: float):
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, _DedupEntry] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _hash_payload(payload: Any) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    async def execute(
        self,
        scope: str,
        request_id: str,
        payload: Any,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        payload_hash = self._hash_payload(payload)
        owns_operation = False
        async with self._lock:
            self._evict_locked()
            entry = self._entries.get(scope)
            if entry is not None:
                if entry.request_id == request_id and entry.payload_hash != payload_hash:
                    raise HTTPException(
                        status_code=409,
                        detail=f"request_id={request_id!r} was already used with a different payload",
                    )
                if entry.request_id != request_id and not entry.future.done():
                    raise HTTPException(
                        status_code=409,
                        detail=f"Another request is already active for scope={scope!r}",
                    )
                if entry.request_id != request_id:
                    entry = None
            if entry is None:
                future = asyncio.get_running_loop().create_future()
                self._entries[scope] = _DedupEntry(
                    request_id=request_id,
                    payload_hash=payload_hash,
                    created_at=time.monotonic(),
                    future=future,
                )
                owns_operation = True
            else:
                future = entry.future

        if not owns_operation:
            return await asyncio.shield(future)

        try:
            result = await operation()
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
                future.exception()
            raise
        else:
            if not future.done():
                future.set_result(result)
            return result

    def _evict_locked(self) -> None:
        now = time.monotonic()
        for scope, entry in list(self._entries.items()):
            if entry.future.done() and now - entry.created_at > self.ttl_seconds:
                self._entries.pop(scope, None)

    async def forget(self, scope: str) -> None:
        async with self._lock:
            self._entries.pop(scope, None)


@dataclass
class _TerminalEntry:
    finalize_request_id: str
    task: asyncio.Task[WebArenaFinalizeResponse]
    cleanup_task: asyncio.Task[None] | None = None
    completed_at: float | None = None


class TerminalOperationRegistry:
    """Own terminal tasks independently of the request that started them."""

    def __init__(self, *, ttl_seconds: float):
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, _TerminalEntry] = {}
        self._lock = asyncio.Lock()

    async def execute(
        self,
        env_id: str,
        finalize_request_id: str,
        operation: Callable[[], Awaitable[WebArenaFinalizeResponse]],
    ) -> WebArenaFinalizeResponse:
        async with self._lock:
            self._evict_locked()
            entry = self._entries.get(env_id)
            if entry is not None and entry.finalize_request_id != finalize_request_id:
                raise HTTPException(
                    status_code=409,
                    detail=f"Session {env_id} already has a different terminal request",
                )
            if entry is None:
                task = asyncio.create_task(
                    operation(),
                    name=f"webarena-finalize-{env_id}",
                )
                entry = _TerminalEntry(
                    finalize_request_id=finalize_request_id,
                    task=task,
                )
                task.add_done_callback(
                    lambda completed, owned_entry=entry: self._complete_entry(owned_entry, completed)
                )
                self._entries[env_id] = entry
        return await asyncio.shield(entry.task)

    async def task_for(self, env_id: str) -> asyncio.Task[WebArenaFinalizeResponse] | None:
        async with self._lock:
            self._evict_locked()
            entry = self._entries.get(env_id)
            return entry.task if entry is not None else None

    async def start_cleanup(
        self,
        env_id: str,
        operation: Callable[[], Awaitable[None]],
    ) -> None:
        """Start one registry-owned cleanup continuation for a finalized session."""
        async with self._lock:
            entry = self._entries.get(env_id)
            if entry is None:
                raise RuntimeError(f"Cannot start cleanup for unregistered session {env_id}")
            if entry.cleanup_task is not None:
                return
            cleanup_task = asyncio.create_task(
                operation(),
                name=f"webarena-cleanup-{env_id}",
            )
            cleanup_task.add_done_callback(self._observe_task)
            entry.cleanup_task = cleanup_task

    async def cleanup_task_for(self, env_id: str) -> asyncio.Task[None] | None:
        async with self._lock:
            self._evict_locked()
            entry = self._entries.get(env_id)
            return entry.cleanup_task if entry is not None else None

    async def close(self) -> None:
        async with self._lock:
            result_tasks = [entry.task for entry in self._entries.values() if not entry.task.done()]
        for task in result_tasks:
            task.cancel()
        await asyncio.gather(*result_tasks, return_exceptions=True)
        async with self._lock:
            cleanup_tasks = [
                entry.cleanup_task
                for entry in self._entries.values()
                if entry.cleanup_task is not None and not entry.cleanup_task.done()
            ]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    def _evict_locked(self) -> None:
        now = time.monotonic()
        for env_id, entry in list(self._entries.items()):
            cleanup_done = entry.cleanup_task is None or entry.cleanup_task.done()
            if entry.completed_at is not None and cleanup_done and now - entry.completed_at > self.ttl_seconds:
                self._entries.pop(env_id, None)

    @staticmethod
    def _complete_entry(
        entry: _TerminalEntry,
        task: asyncio.Task[WebArenaFinalizeResponse],
    ) -> None:
        entry.completed_at = time.monotonic()
        if not task.cancelled():
            task.exception()

    @staticmethod
    def _observe_task(task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            task.exception()


class WebArenaResourcesServer(SimpleResourcesServer):
    """Stateful resources server backed by a global Ray browser actor pool."""

    ray_enabled = True

    model_config = ConfigDict(arbitrary_types_allowed=True)
    config: WebArenaResourcesServerConfig

    _pool: BrowserActorPool = PrivateAttr()
    _sessions: dict[str, SessionRecord] = PrivateAttr(default_factory=dict)
    _deduplicator: SingleFlightDeduplicator = PrivateAttr()
    _terminal_operations: TerminalOperationRegistry = PrivateAttr()
    _reaper_task: asyncio.Task[None] | None = PrivateAttr(default=None)

    def model_post_init(self, _ctx: Any) -> None:
        self._pool = BrowserActorPool(self.config)
        self._deduplicator = SingleFlightDeduplicator(
            ttl_seconds=_DEDUPLICATION_RETRY_WINDOW_SECONDS,
        )
        self._terminal_operations = TerminalOperationRegistry(
            ttl_seconds=_DEDUPLICATION_RETRY_WINDOW_SECONDS,
        )

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/step")(self.step)
        app.post("/finalize")(self.finalize)
        app.post("/close")(self.close)

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            try:
                await self._pool.prewarm()
                self._reaper_task = asyncio.create_task(self._reaper_loop())
                yield
            finally:
                if self._reaper_task is not None:
                    self._reaper_task.cancel()
                    await asyncio.gather(
                        self._reaper_task,
                        return_exceptions=True,
                    )
                    self._reaper_task = None
                await self._terminal_operations.close()
                await self._pool.close()

        app.router.lifespan_context = lifespan

        return app

    async def seed_session(
        self,
        request: Request,
        body: WebArenaSeedSessionRequest,
    ) -> WebArenaSeedSessionResponse:
        return await self._deduplicator.execute(
            f"seed:{body.request_id}",
            body.request_id,
            body.model_dump(mode="json"),
            lambda: self._seed(body),
        )

    async def _seed(self, body: WebArenaSeedSessionRequest) -> WebArenaSeedSessionResponse:
        env_id = str(uuid.uuid4())
        actor = await self._pool.checkout(env_id)
        now = time.monotonic()
        record = SessionRecord(
            env_id=env_id,
            seed_request_id=body.request_id,
            actor=actor,
            task=body.task,
            state=SessionState.SEEDING,
            created_at=now,
            last_activity_at=now,
            active_operations=1,
        )
        self._sessions[env_id] = record
        try:
            reset_result = await asyncio.wait_for(
                actor.reset.remote(body.task.model_dump(mode="json")),
                timeout=self.config.browser_operation_timeout_seconds,
            )
            record.task = WebArenaTaskRow.model_validate(reset_result["task"])
            record.state = SessionState.LEASED
            record.last_activity_at = time.monotonic()
            obs = build_observation_message(
                screenshot=reset_result["screenshot"],
                task=record.task,
                step=1,
                tab_context=reset_result.get("tab_context"),
                image_format=self.config.screenshot_format,
                jpeg_quality=self.config.screenshot_jpeg_quality,
            )
            return WebArenaSeedSessionResponse(
                env_id=env_id,
                obs=[obs],
                tools=WEB_ARENA_TOOLS,
            )
        except BaseException as exc:
            record.state = SessionState.DEAD
            await self._pool.retire(env_id)
            record.actor = None
            record.closed_at = time.monotonic()
            self._sessions.pop(env_id, None)
            if isinstance(exc, asyncio.CancelledError):
                raise
            logger.exception("Failed to seed WebArena task %s", body.task.task_id)
            raise HTTPException(
                status_code=500,
                detail=f"WebArena seed failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            record.active_operations = 0

    async def step(self, request: Request, body: WebArenaStepRequest) -> WebArenaStepResponse:
        record = self._sessions.get(body.env_id)
        if record is not None:
            await self._deduplicator.forget(f"seed:{record.seed_request_id}")
        return await self._deduplicator.execute(
            f"session:{body.env_id}",
            body.action_request_id,
            body.model_dump(mode="json"),
            lambda: self._step(body),
        )

    async def _step(self, body: WebArenaStepRequest) -> WebArenaStepResponse:
        if len(body.action) != 1:
            raise HTTPException(status_code=400, detail="/step requires exactly one function call")
        record = self._session(body.env_id)
        async with record.lock:
            if record.state not in {SessionState.LEASED, SessionState.TERMINATING}:
                raise HTTPException(
                    status_code=409,
                    detail=f"Session {body.env_id} cannot step while {record.state.value}",
                )
            record.active_operations += 1
            record.last_activity_at = time.monotonic()
            call = body.action[0]
            if call.name == "terminate":
                record.state = SessionState.TERMINATING
            try:
                result = await asyncio.wait_for(
                    record.actor.step.remote(call.model_dump(mode="json")),
                    timeout=self.config.browser_operation_timeout_seconds,
                )
            except BaseException as exc:
                record.state = SessionState.DEAD
                await self._pool.retire(body.env_id)
                record.actor = None
                record.closed_at = time.monotonic()
                self._sessions.pop(body.env_id, None)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise HTTPException(
                    status_code=500,
                    detail=f"WebArena step failed: {type(exc).__name__}: {exc}",
                ) from exc
            finally:
                record.active_operations -= 1
                record.last_activity_at = time.monotonic()

            record.step_count += 1
            obs: list[Any] = []
            if result["function_output"] is not None:
                obs.append(
                    build_function_output(
                        result["call_id"],
                        result["function_output"],
                    )
                )
            if result["done"]:
                record.state = SessionState.COMPLETED
                record.termination_status = result.get("termination_status")
                record.answer = result.get("answer")
            else:
                record.state = SessionState.LEASED
                obs.append(
                    build_observation_message(
                        screenshot=result["screenshot"],
                        task=record.task,
                        step=record.step_count + 1,
                        tab_context=result.get("tab_context"),
                        image_format=self.config.screenshot_format,
                        jpeg_quality=self.config.screenshot_jpeg_quality,
                    )
                )
            return WebArenaStepResponse(
                obs=obs,
                reward=record.reward if result["done"] else 0.0,
                done=bool(result["done"]),
                termination_status=record.termination_status,
                answer=record.answer,
                evaluator_details=record.evaluator_details,
                tool_error=result.get("tool_error"),
            )

    async def close(
        self,
        request: Request,
        body: WebArenaCloseRequest,
    ) -> WebArenaCloseResponse:
        terminal_task = await self._terminal_operations.task_for(body.env_id)
        if terminal_task is not None:
            try:
                await asyncio.shield(terminal_task)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Failed finalization retires its actor before completing. Fall
                # through to preserve /close's idempotent cleanup contract.
                pass
            else:
                cleanup_task = await self._terminal_operations.cleanup_task_for(body.env_id)
                if cleanup_task is not None:
                    try:
                        await asyncio.shield(cleanup_task)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.warning(
                            "Finalized browser cleanup failed for env_id=%s",
                            body.env_id,
                            exc_info=True,
                        )
                return WebArenaCloseResponse(success=True, message="already finalized")
        record = self._sessions.get(body.env_id)
        result = await self._deduplicator.execute(
            f"session:{body.env_id}",
            body.close_request_id,
            body.model_dump(mode="json"),
            lambda: self._close(body.env_id),
        )
        await self._deduplicator.forget(f"session:{body.env_id}")
        if record is not None:
            await self._deduplicator.forget(f"seed:{record.seed_request_id}")
        return result

    async def finalize(
        self,
        request: Request,
        body: WebArenaFinalizeRequest,
    ) -> WebArenaFinalizeResponse:
        record = self._sessions.get(body.env_id)
        if record is not None:
            await self._deduplicator.forget(f"seed:{record.seed_request_id}")
        return await self._terminal_operations.execute(
            body.env_id,
            body.finalize_request_id,
            lambda: self._finalize(body),
        )

    async def _finalize(self, body: WebArenaFinalizeRequest) -> WebArenaFinalizeResponse:
        record = self._session(body.env_id)
        async with record.lock:
            if record.state not in {
                SessionState.LEASED,
                SessionState.TERMINATING,
                SessionState.COMPLETED,
            }:
                raise HTTPException(
                    status_code=409,
                    detail=f"Session {record.env_id} cannot be finalized while {record.state.value}",
                )
            if record.actor is None:
                raise HTTPException(status_code=409, detail=f"Session {record.env_id} has no live browser actor")
            actor = record.actor
            needs_evaluation = not record.evaluation_completed
            record.state = SessionState.FINALIZING
            record.active_operations += 1
            record.last_activity_at = time.monotonic()

        try:
            if needs_evaluation:
                result = await asyncio.wait_for(
                    actor.evaluate.remote(
                        status=record.termination_status,
                        answer=record.answer,
                    ),
                    timeout=self.config.browser_evaluation_timeout_seconds,
                )
                async with record.lock:
                    record.reward = float(result["reward"])
                    record.evaluator_details = dict(result.get("evaluator_details") or {})
                    record.termination_status = result.get("termination_status")
                    record.answer = result.get("answer")
                    record.evaluation_completed = True

            response = WebArenaFinalizeResponse(
                env_id=record.env_id,
                finalize_request_id=body.finalize_request_id,
                reward=record.reward,
                evaluator_details=record.evaluator_details,
                termination_status=record.termination_status,
                answer=record.answer,
            )
            async with record.lock:
                record.state = SessionState.RECYCLING
                record.last_activity_at = time.monotonic()
            await self._terminal_operations.start_cleanup(
                record.env_id,
                lambda: self._cleanup_finalized_record(record),
            )
        except BaseException as exc:
            async with record.lock:
                record.state = SessionState.DEAD
                record.actor = None
                record.active_operations -= 1
                record.closed_at = time.monotonic()
                record.last_activity_at = record.closed_at
            try:
                await self._pool.retire(record.env_id)
            except BaseException:
                logger.exception("Failed to retire uncertain WebArena actor for env_id=%s", record.env_id)
            finally:
                if self._sessions.get(record.env_id) is record:
                    self._sessions.pop(record.env_id, None)
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise HTTPException(
                status_code=500,
                detail=f"WebArena finalization failed: {type(exc).__name__}: {exc}",
            ) from exc

        return response

    async def _cleanup_finalized_record(self, record: SessionRecord) -> None:
        """Release one finalized browser lease without delaying its reward."""
        try:
            await self._pool.release(record.env_id)
        finally:
            async with record.lock:
                record.actor = None
                record.active_operations -= 1
                record.state = SessionState.CLOSED
                record.closed_at = time.monotonic()
                record.last_activity_at = record.closed_at
            if self._sessions.get(record.env_id) is record:
                self._sessions.pop(record.env_id, None)
            await self._deduplicator.forget(f"session:{record.env_id}")

    async def _close(self, env_id: str) -> WebArenaCloseResponse:
        record = self._sessions.get(env_id)
        if record is None:
            return WebArenaCloseResponse(success=True, message="already closed")
        async with record.lock:
            if record.state == SessionState.CLOSED:
                return WebArenaCloseResponse(success=True, message="already closed")
            record.state = SessionState.CLOSING
            await self._pool.release(env_id)
            record.actor = None
            record.state = SessionState.CLOSED
            record.closed_at = time.monotonic()
            record.last_activity_at = record.closed_at
            self._sessions.pop(env_id, None)
        return WebArenaCloseResponse(success=True, message="ok")

    async def verify(
        self,
        request: Request,
        body: WebArenaAgentVerifyRequest,
    ) -> WebArenaAgentVerifyResponse:
        record = self._sessions.get(body.response.env_id)
        if record is not None:
            await self._deduplicator.forget(f"seed:{record.seed_request_id}")
        return await self._deduplicator.execute(
            f"session:{body.response.env_id}",
            body.verify_request_id,
            body.model_dump(mode="json"),
            lambda: self._verify(body),
        )

    async def _verify(self, body: WebArenaAgentVerifyRequest) -> WebArenaAgentVerifyResponse:
        record = self._session(body.response.env_id)
        async with record.lock:
            if not record.evaluation_completed:
                if record.state not in {SessionState.LEASED, SessionState.TERMINATING, SessionState.COMPLETED}:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Session {record.env_id} cannot be evaluated while {record.state.value}",
                    )
                if record.actor is None:
                    raise HTTPException(status_code=409, detail=f"Session {record.env_id} has no live browser actor")
                record.active_operations += 1
                record.last_activity_at = time.monotonic()
                try:
                    result = await asyncio.wait_for(
                        record.actor.evaluate.remote(
                            status=record.termination_status,
                            answer=record.answer,
                        ),
                        timeout=self.config.browser_evaluation_timeout_seconds,
                    )
                except BaseException as exc:
                    record.state = SessionState.DEAD
                    await self._pool.retire(record.env_id)
                    record.actor = None
                    record.closed_at = time.monotonic()
                    self._sessions.pop(record.env_id, None)
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    raise HTTPException(
                        status_code=500,
                        detail=f"WebArena evaluation failed: {type(exc).__name__}: {exc}",
                    ) from exc
                finally:
                    record.active_operations -= 1
                    record.last_activity_at = time.monotonic()

                record.reward = float(result["reward"])
                record.evaluator_details = dict(result.get("evaluator_details") or {})
                record.termination_status = result.get("termination_status")
                record.answer = result.get("answer")
                record.evaluation_completed = True
                record.state = SessionState.COMPLETED

            return WebArenaAgentVerifyResponse(
                **body.model_dump(mode="python"),
                reward=record.reward,
                evaluator_details=record.evaluator_details,
                termination_status=record.termination_status,
                answer=record.answer,
            )

    def _session(self, env_id: str) -> SessionRecord:
        record = self._sessions.get(env_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Unknown WebArena env_id={env_id}")
        return record

    async def _reaper_loop(self) -> None:
        interval = max(5.0, min(60.0, self.config.browser_session_max_seconds / 4))
        while True:
            await asyncio.sleep(interval)
            await self._reap_once()

    async def _reap_once(self) -> None:
        now = time.monotonic()
        for env_id, record in list(self._sessions.items()):
            may_expire = record.state in {
                SessionState.SEEDING,
                SessionState.LEASED,
                SessionState.TERMINATING,
                SessionState.COMPLETED,
                SessionState.FINALIZING,
            }
            expired = now - record.created_at > self.config.browser_session_max_seconds
            if not may_expire or not expired:
                continue
            async with record.lock:
                now = time.monotonic()
                expired = now - record.created_at > self.config.browser_session_max_seconds
                if record.active_operations or not expired:
                    continue
                if record.state == SessionState.COMPLETED:
                    record.state = SessionState.CLOSING
                    await self._pool.release(env_id)
                    record.state = SessionState.CLOSED
                elif record.state in {
                    SessionState.SEEDING,
                    SessionState.LEASED,
                    SessionState.TERMINATING,
                }:
                    record.state = SessionState.DEAD
                    record.reward = 0.0
                    record.termination_status = "session_timeout"
                    await self._pool.retire(env_id)
                else:
                    continue
                record.actor = None
                record.closed_at = now
                record.last_activity_at = now
                self._sessions.pop(env_id, None)
                await self._deduplicator.forget(f"session:{env_id}")
                await self._deduplicator.forget(f"seed:{record.seed_request_id}")


if __name__ == "__main__":
    WebArenaResourcesServer.run_webserver()
