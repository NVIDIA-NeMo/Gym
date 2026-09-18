# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared lifecycle for environment servers."""

import asyncio
import logging
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TypeVar

from anyio import CancelScope
from fastapi import Body, FastAPI
from pydantic import ConfigDict, PositiveFloat, PositiveInt, model_validator
from typing_extensions import Self

from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, BaseRunServerInstanceConfig
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure, EpisodeId
from nemo_gym.server_utils import SimpleServer


LOGGER = logging.getLogger(__name__)

EpisodeRequestT = TypeVar("EpisodeRequestT", bound=BaseEpisodeRequest[Any])
EpisodeResponseT = TypeVar("EpisodeResponseT", bound=BaseEpisodeResponse[Any])
CleanupCallback = Callable[[], Awaitable[None]]


class BaseEnvironmentServerConfig(BaseRunServerInstanceConfig):
    """Configure protocol-neutral episode limits."""

    model_config = ConfigDict(extra="forbid")

    max_concurrent_episodes: PositiveInt | None = None
    queue_timeout_seconds: PositiveFloat | None = None
    default_episode_timeout_seconds: PositiveFloat | None = None
    cleanup_timeout_seconds: PositiveFloat

    @model_validator(mode="after")
    def validate_queue_timeout(self) -> Self:
        if self.max_concurrent_episodes is not None and self.queue_timeout_seconds is None:
            raise ValueError("queue_timeout_seconds is required when max_concurrent_episodes is enabled")
        return self


@dataclass
class _CleanupEntry:
    name: str
    callback: CleanupCallback
    active: bool = True
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def close(self) -> None:
        async with self.lock:
            if not self.active:
                return
            await self.callback()
            self.active = False


@dataclass
class CleanupHandle:
    """Close one registered participant at a protocol boundary."""

    entry: _CleanupEntry
    timeout_seconds: float

    async def close(self) -> None:
        """Run this idempotent callback once.

        The episode timeout bounds calls made during the protocol. The context's
        cleanup timeout bounds callbacks left for final unwinding.
        """
        async with asyncio.timeout(self.timeout_seconds):
            await self.entry.close()


@dataclass
class CleanupContext:
    """Hold bounded process-local cleanup callbacks for one episode.

    Callbacks:
    - are process-local Python objects, although they may issue remote close requests;
    - run sequentially in LIFO order within one total cleanup timeout;
    - must be idempotent because a timed-out remote request may have succeeded;
    - are lost on process or host failure, so remote owners need expiry or reaping.

    Callback failures are logged and do not stop later callbacks. Cleanup has no
    durable retry after this context is discarded.
    """

    episode_id: EpisodeId
    cleanup_timeout_seconds: float
    _cleanups: list[_CleanupEntry] = field(default_factory=list)

    def register_cleanup(self, name: str, callback: CleanupCallback) -> CleanupHandle:
        entry = _CleanupEntry(name=name, callback=callback)
        self._cleanups.append(entry)
        return CleanupHandle(entry, self.cleanup_timeout_seconds)

    async def aclose(self) -> None:
        async def unwind() -> None:
            for entry in reversed(self._cleanups):
                try:
                    await entry.close()
                except Exception:
                    LOGGER.exception(f"Episode cleanup failed: {entry.name}")

        try:
            async with asyncio.timeout(self.cleanup_timeout_seconds):
                await unwind()
        except TimeoutError:
            LOGGER.error(f"Episode cleanup timed out: episode_id={self.episode_id}")


class HandledEpisodeError(Exception):
    """Carry a failure that belongs in the native episode response."""

    def __init__(self, failure: EpisodeFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class BaseEnvironmentServer(SimpleServer, Generic[EpisodeRequestT, EpisodeResponseT]):
    """Expose a typed episode protocol with shared limits and cleanup."""

    config: BaseEnvironmentServerConfig
    request_model: ClassVar[type[EpisodeRequestT]]
    response_model: ClassVar[type[EpisodeResponseT]]
    _admission: asyncio.Semaphore | None = None

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._admission = (
            asyncio.Semaphore(self.config.max_concurrent_episodes)
            if self.config.max_concurrent_episodes is not None
            else None
        )

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()

        async def run_endpoint(body: Any) -> Any:
            return await self.run_request(body)

        run_endpoint.__annotations__["body"] = self.request_model
        run_endpoint.__annotations__["return"] = self.response_model
        app.post("/run", response_model=self.response_model)(run_endpoint)

        async def aggregate_metrics_endpoint(body: Any) -> Any:
            return await self.aggregate_metrics(body)

        aggregate_metrics_endpoint.__annotations__["body"] = AggregateMetricsRequest
        aggregate_metrics_endpoint.__annotations__["return"] = AggregateMetrics
        app.post("/aggregate_metrics", response_model=AggregateMetrics)(aggregate_metrics_endpoint)
        return app

    async def run_request(self, request: EpisodeRequestT) -> EpisodeResponseT:
        acquired = False
        if self._admission is not None:
            try:
                await asyncio.wait_for(
                    self._admission.acquire(),
                    timeout=self.config.queue_timeout_seconds,
                )
            except TimeoutError:
                return self.failure_response(
                    request,
                    EpisodeFailure(
                        message="Episode admission timed out",
                        terminal=False,
                    ),
                )
            acquired = True

        cleanup = CleanupContext(
            episode_id=request.episode_id,
            cleanup_timeout_seconds=float(self.config.cleanup_timeout_seconds),
        )
        response: EpisodeResponseT
        cancelled: asyncio.CancelledError | None = None
        deadline = asyncio.timeout(self.config.default_episode_timeout_seconds)
        try:
            try:
                async with deadline:
                    response = await self.run(request, cleanup)
            except TimeoutError as error:
                if deadline.expired():
                    response = self.failure_response(
                        request,
                        EpisodeFailure(
                            message="Episode timed out",
                            terminal=False,
                        ),
                    )
                else:
                    response = self._unhandled_failure_response(request, error)
            except HandledEpisodeError as error:
                response = self.failure_response(request, error.failure)
            except asyncio.CancelledError as error:
                cancelled = error
            except Exception as error:
                response = self._unhandled_failure_response(request, error)
        finally:
            try:
                with CancelScope(shield=True):
                    await cleanup.aclose()
            finally:
                if acquired and self._admission is not None:
                    self._admission.release()

        if cancelled is not None:
            raise cancelled
        response = self.response_model.model_validate(response)
        self.validate_response_identity(request, response)
        return response

    @abstractmethod
    async def run(self, request: EpisodeRequestT, cleanup: CleanupContext) -> EpisodeResponseT:
        """Run one concrete environment protocol."""

    def _unhandled_failure_response(self, request: EpisodeRequestT, error: Exception) -> EpisodeResponseT:
        LOGGER.exception(f"Unhandled environment server error: episode_id={request.episode_id}")
        message = f"Unhandled environment server error: {type(error).__name__}: {error}"
        return self.failure_response(
            request,
            EpisodeFailure(
                message=message[:2000],
                terminal=True,
            ),
        )

    def failure_response(self, request: EpisodeRequestT, failure: EpisodeFailure) -> EpisodeResponseT:
        return self.response_model.model_validate(
            {
                "episode_id": request.episode_id,
                "task_id": request.task.task_id,
                "failure": failure.model_dump(mode="json"),
            }
        )

    @staticmethod
    def validate_response_identity(request: EpisodeRequestT, response: EpisodeResponseT) -> None:
        if response.episode_id != request.episode_id:
            raise ValueError("response episode_id does not match request")
        if response.task_id != request.task.task_id:
            raise ValueError("response task_id does not match request")

    @abstractmethod
    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Aggregate per-rollout scores into task-level metrics."""
