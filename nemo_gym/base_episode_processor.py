# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared lifecycle for episode-processor servers."""

import asyncio
import logging
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TypeVar

from fastapi import FastAPI
from pydantic import ConfigDict, PositiveFloat, PositiveInt, model_validator
from typing_extensions import Self

from nemo_gym.config_types import BaseRunServerInstanceConfig
from nemo_gym.episode import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure
from nemo_gym.server_utils import ServerClient, SimpleServer


LOGGER = logging.getLogger(__name__)

EpisodeRequestT = TypeVar("EpisodeRequestT", bound=BaseEpisodeRequest[Any])
EpisodeResponseT = TypeVar("EpisodeResponseT", bound=BaseEpisodeResponse[Any])
CleanupCallback = Callable[[], Awaitable[None]]


class BaseEpisodeProcessorConfig(BaseRunServerInstanceConfig):
    """Configure protocol-neutral episode limits."""

    model_config = ConfigDict(extra="forbid")

    max_concurrent_episodes: PositiveInt | None = None
    queue_timeout_seconds: PositiveFloat | None = None
    default_episode_timeout_seconds: PositiveFloat
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
        if not self.entry.active:
            return
        async with asyncio.timeout(self.timeout_seconds):
            await self.entry.callback()
        self.entry.active = False


@dataclass
class EpisodeContext:
    """Hold processor-local cleanup callbacks for one episode.

    Callbacks:
    - are process-local Python objects, although they may issue remote close requests;
    - run sequentially in LIFO order within one total cleanup timeout;
    - must be idempotent because a timed-out remote request may have succeeded;
    - are lost on process or host failure, so remote owners need expiry or reaping.

    Callback failures are logged and do not stop later callbacks. Cleanup has no
    durable retry after this context is discarded.
    """

    request: BaseEpisodeRequest[Any]
    server_client: ServerClient
    cleanup_timeout_seconds: float
    _cleanups: list[_CleanupEntry] = field(default_factory=list)

    def register_cleanup(self, name: str, callback: CleanupCallback) -> CleanupHandle:
        entry = _CleanupEntry(name=name, callback=callback)
        self._cleanups.append(entry)
        return CleanupHandle(entry, self.cleanup_timeout_seconds)

    async def aclose(self) -> None:
        async def unwind() -> None:
            for entry in reversed(self._cleanups):
                if not entry.active:
                    continue
                try:
                    await entry.callback()
                except Exception:
                    LOGGER.exception(f"Episode cleanup failed: {entry.name}")
                else:
                    entry.active = False

        try:
            async with asyncio.timeout(self.cleanup_timeout_seconds):
                await unwind()
        except TimeoutError:
            LOGGER.error(f"Episode cleanup timed out: episode_id={self.request.episode_id}")


class HandledEpisodeError(Exception):
    """Carry a failure that belongs in the native episode response."""

    def __init__(self, failure: EpisodeFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


class BaseEpisodeProcessor(SimpleServer, Generic[EpisodeRequestT, EpisodeResponseT]):
    """Expose a typed episode protocol with shared limits and cleanup."""

    config: BaseEpisodeProcessorConfig
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
            return await self.run(body)

        run_endpoint.__annotations__["body"] = self.request_model
        run_endpoint.__annotations__["return"] = self.response_model
        app.post("/run", response_model=self.response_model)(run_endpoint)
        return app

    async def run(self, request: EpisodeRequestT) -> EpisodeResponseT:
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

        context = EpisodeContext(
            request=request,
            server_client=self.server_client,
            cleanup_timeout_seconds=float(self.config.cleanup_timeout_seconds),
        )
        response: EpisodeResponseT
        cancelled: asyncio.CancelledError | None = None
        try:
            try:
                async with asyncio.timeout(self.config.default_episode_timeout_seconds):
                    response = await self.process(request, context)
            except TimeoutError:
                response = self.failure_response(
                    request,
                    EpisodeFailure(
                        message="Episode timed out",
                        terminal=False,
                    ),
                )
            except HandledEpisodeError as error:
                response = self.failure_response(request, error.failure)
            except asyncio.CancelledError as error:
                cancelled = error
                response = self.failure_response(
                    request,
                    EpisodeFailure(
                        message="Episode request was cancelled",
                        terminal=False,
                    ),
                )
        finally:
            cleanup_task = asyncio.create_task(context.aclose())
            try:
                while not cleanup_task.done():
                    try:
                        await asyncio.shield(cleanup_task)
                    except asyncio.CancelledError as error:
                        cancelled = cancelled or error
            finally:
                if acquired and self._admission is not None:
                    self._admission.release()

        if cancelled is not None:
            raise cancelled
        response = self.response_model.model_validate(response)
        self.validate_response_identity(request, response)
        return response

    @abstractmethod
    async def process(self, request: EpisodeRequestT, context: EpisodeContext) -> EpisodeResponseT:
        """Run one concrete episode protocol."""

    def failure_response(self, request: EpisodeRequestT, failure: EpisodeFailure) -> EpisodeResponseT:
        return self.response_model.model_validate(
            {
                "episode_id": request.episode_id,
                "task_id": request.task.task_id,
                "failure": failure,
            }
        )

    @staticmethod
    def validate_response_identity(request: EpisodeRequestT, response: EpisodeResponseT) -> None:
        if response.episode_id != request.episode_id:
            raise ValueError("response episode_id does not match request")
        if response.task_id != request.task.task_id:
            raise ValueError("response task_id does not match request")
