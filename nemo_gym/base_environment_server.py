# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared lifecycle for environment servers."""

import asyncio
import logging
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TypeVar

from anyio import CancelScope
from fastapi import Body, FastAPI
from pydantic import ConfigDict, JsonValue, PositiveFloat, PositiveInt, model_validator
from typing_extensions import Self

from nemo_gym._checkpoint.control import install_participant
from nemo_gym._checkpoint.environment import EnvironmentParticipant
from nemo_gym._checkpoint.errors import ControlError
from nemo_gym._checkpoint.settings import checkpoint_settings
from nemo_gym._checkpoint.steps import StepMode
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest, BaseRunServerInstanceConfig
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure, EpisodeId
from nemo_gym.rollout_correlation import rollout_context
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
    """Carry a failure that belongs in the episode response."""

    def __init__(self, failure: EpisodeFailure) -> None:
        super().__init__(failure.failure_reason)
        self.failure = failure


class BaseEnvironmentServer(SimpleServer, Generic[EpisodeRequestT, EpisodeResponseT]):
    """Expose a typed episode protocol with shared limits and cleanup."""

    config: BaseEnvironmentServerConfig
    request_model: ClassVar[type[EpisodeRequestT]]
    response_model: ClassVar[type[EpisodeResponseT]]
    _admission: asyncio.Semaphore | None = None
    _checkpoint: EnvironmentParticipant | None = None
    # Whether ``run`` records checkpoint boundaries.
    # Episodes of a protocol that records none cannot be continued, so they are restarts:
    # they never hold up a checkpoint, and start over from their input after a crash.
    checkpoint_boundaries: ClassVar[bool] = False

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._admission = (
            asyncio.Semaphore(self.config.max_concurrent_episodes)
            if self.config.max_concurrent_episodes is not None
            else None
        )

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_environment_checkpoint(app)

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
                        failure_reason="Episode admission timed out",
                        terminal=False,
                        stage="admission",
                    ),
                )
            acquired = True

        cleanup = CleanupContext(
            episode_id=request.episode_id,
            cleanup_timeout_seconds=float(self.config.cleanup_timeout_seconds),
        )
        response: EpisodeResponseT
        cancelled: asyncio.CancelledError | None = None
        # Set once this request owns its episode's checkpoint state.
        # A begin refused as a duplicate or a retired attempt owns nothing,
        # so it must not end the live episode with the same key.
        begun = False
        deadline = asyncio.timeout(self.config.default_episode_timeout_seconds)
        # Every downstream call of this episode, including final cleanup, carries its attempt-qualified
        # rollout id, so Resources and Model Server calls stay correlated with the rollout.
        with rollout_context(request.episode_id.capture_key):
            try:
                try:
                    async with deadline:
                        if self._checkpoint is not None:
                            self._checkpoint.begin(
                                request.episode_id,
                                request.task.model_dump(mode="json"),
                                deadline,
                                restart=not self.checkpoint_boundaries,
                            )
                            begun = True
                        response = await self.run(request, cleanup)
                except TimeoutError as error:
                    if deadline.expired():
                        response = self.failure_response(
                            request,
                            EpisodeFailure(
                                failure_reason="Episode timed out",
                                terminal=False,
                            ),
                        )
                    else:
                        response = self._unhandled_failure_response(request, error)
                except ControlError:
                    # A checkpoint refused this episode before it started; the caller retries it later.
                    raise
                except HandledEpisodeError as error:
                    response = self.failure_response(request, error.failure)
                except asyncio.CancelledError as error:
                    cancelled = error
                except Exception as error:
                    response = self._unhandled_failure_response(request, error)
            finally:
                if begun:
                    # The shield below holds against anyio cancellation, not a retire's native task.cancel().
                    self._checkpoint.finishing(request.episode_id)
                try:
                    with CancelScope(shield=True):
                        try:
                            await cleanup.aclose()
                        finally:
                            # Even a native cancel landing in cleanup must not leave the episode tracked.
                            if begun:
                                await self._checkpoint.end(request.episode_id)
                finally:
                    if acquired and self._admission is not None:
                        self._admission.release()

        if cancelled is not None:
            raise cancelled
        response = self.response_model.model_validate(response)
        self.validate_response_identity(request, response)
        return response

    def setup_environment_checkpoint(self, app: FastAPI) -> None:
        """Take part in partial-rollout checkpoints when they are enabled.

        Every episode is tracked.
        A protocol that records boundaries (``checkpoint_boundaries``) continues after a restore;
        the episodes of a protocol that records none are restarts,
        which never hold up a checkpoint and start over from their input after a crash.
        Call this from any ``setup_webserver`` that builds its own app.
        """
        settings = checkpoint_settings(getattr(self.server_client, "global_config_dict", None))
        if settings is None:
            return
        if (self.config.num_workers or 1) != 1:
            raise ValueError("environment checkpointing requires num_workers=1: episodes live in one process")
        self._checkpoint = EnvironmentParticipant()
        install_participant(
            app,
            self._checkpoint,
            auth_token=settings.control_auth_token,
            lease_grace_seconds=settings.lease_grace_seconds,
            instance_name=self.config.name,
        )

    @abstractmethod
    async def run(self, request: EpisodeRequestT, cleanup: CleanupContext) -> EpisodeResponseT:
        """Run one concrete environment protocol."""

    def checkpoint_continuation(self, request: EpisodeRequestT) -> dict[str, JsonValue] | None:
        """Return the protocol state this attempt continues from a checkpoint, once, or ``None``."""
        return self._checkpoint.continuation(request.episode_id) if self._checkpoint is not None else None

    async def checkpoint_boundary(
        self,
        request: EpisodeRequestT,
        state: dict[str, JsonValue] | Callable[[], dict[str, JsonValue]],
    ) -> None:
        """Record that a protocol step completed; ``state`` names the next step and what it needs.

        While a checkpoint is open the episode parks here until resume.
        A replacement attempt
        that continues a checkpoint receives the latest recorded ``state`` from ``checkpoint_continuation``.
        Pass a function when building ``state`` costs something: it is called only with checkpointing on.
        """
        if self._checkpoint is not None:
            await self._checkpoint.boundary(request.episode_id, state() if callable(state) else state)

    async def checkpoint_restart(self, request: EpisodeRequestT) -> None:
        """Mark this episode as a restart: a server it uses cannot capture its part, such as a restart-only agent.

        Call it when a seed reply says so (``nemo_gym._checkpoint.steps.seed_restarts``).
        The episode then never holds up a checkpoint and is never exported; after a crash,
        the controller starts it over from its input.
        """
        if self._checkpoint is not None:
            await self._checkpoint.mark_restart(request.episode_id)

    def checkpoint_step(self, request: EpisodeRequestT, mode: StepMode) -> AbstractAsyncContextManager[None]:
        """Run one protocol step in ``wait`` or ``replay`` mode (see ``nemo_gym._checkpoint.steps``).

        Use ``replay`` only for a step that is safe to run again after a crash
        and does not change checkpointed state; everything else waits.
        """
        if self._checkpoint is None:
            return nullcontext()
        return self._checkpoint.step(request.episode_id, mode)

    def _unhandled_failure_response(self, request: EpisodeRequestT, error: Exception) -> EpisodeResponseT:
        LOGGER.exception(f"Unhandled environment server error: episode_id={request.episode_id}")
        failure_reason = f"Unhandled environment server error: {type(error).__name__}: {error}"
        return self.failure_response(
            request,
            EpisodeFailure(
                failure_reason=failure_reason[:2000],
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
