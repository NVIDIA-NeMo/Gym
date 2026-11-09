# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared lifecycle for environment servers, and the base for those that serve each episode's tools."""

import asyncio
import json
import logging
from abc import abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, ClassVar, Generic, Optional, TypeVar

from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError
from anyio import CancelScope
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ConfigDict, Field, JsonValue, PositiveFloat, PositiveInt, model_validator
from starlette.datastructures import Headers
from typing_extensions import Self

from nemo_gym.client.http_client import format_exception_with_response_body
from nemo_gym.config.config_types import AggregateMetrics, AggregateMetricsRequest, BaseRunServerInstanceConfig
from nemo_gym.core.episode_context import rollout_context
from nemo_gym.core.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeId
from nemo_gym.core.failure_kinds import EPISODE_FAILED, SESSION_RELEASE_FAILED, FailureStage
from nemo_gym.core.failure_records import EpisodeFailure, FailureBoundary, FailureReport
from nemo_gym.core.failure_transport import reply_failure_report, report_for_current_request
from nemo_gym.core.model_call_sampling import MODEL_SAMPLING_NULL, encode_model_sampling
from nemo_gym.core.typed_exceptions import NgFailureEvent, failure_report_for
from nemo_gym.server_runtime._checkpoint.control import CheckpointMode
from nemo_gym.server_runtime._checkpoint.environment import EnvironmentParticipant
from nemo_gym.server_runtime._checkpoint.errors import ControlError
from nemo_gym.server_runtime._checkpoint.participant_workers import install_server_participant
from nemo_gym.server_runtime._checkpoint.settings import checkpoint_settings
from nemo_gym.server_runtime._checkpoint.steps import StepType
from nemo_gym.server_runtime.base_server import SimpleServer
from nemo_gym.server_runtime.episode_tokens import (
    EPISODE_TOKEN_HEADER,
    EpisodeTokenClaims,
    EpisodeTokens,
    episode_token_secret,
)
from nemo_gym.server_runtime.episode_tools import EpisodeToolset, ToolArgumentsError, ToolCallRecord, ToolSpec
from nemo_gym.server_runtime.request_handling.rollout_context import RolloutContextMiddleware
from nemo_gym.server_runtime.tool_access import (
    TOOL_CALL_ID_HEADER,
    TOOL_ERROR_HEADER,
    DirectHTTPToolAccess,
    ToolAccess,
)
from nemo_gym.telemetry.concurrency import MeteredSemaphore


LOGGER = logging.getLogger(__name__)
# An environment server either records boundaries or restarts its episodes; it has no stateless mode.
_ENVIRONMENT_CHECKPOINT_MODES = ("restart_only", "exported")

EpisodeRequestT = TypeVar("EpisodeRequestT", bound=BaseEpisodeRequest[Any])
EpisodeResponseT = TypeVar("EpisodeResponseT", bound=BaseEpisodeResponse[Any])
CleanupCallback = Callable[[], Awaitable[None]]


class BaseEnvironmentServerConfig(BaseRunServerInstanceConfig):
    """Configure protocol-neutral episode limits."""

    model_config = ConfigDict(extra="forbid")

    max_concurrent_episodes: PositiveInt | None = None
    queue_timeout_seconds: PositiveFloat | None = None
    default_episode_timeout_seconds: PositiveFloat | None = None
    cleanup_timeout_seconds: PositiveFloat = Field(
        description="Timeout for each cleanup callback; total final cleanup may take this limit per registered owner."
    )

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
    - run sequentially in LIFO order, each with its own cleanup timeout;
    - must be idempotent because a timed-out remote request may have succeeded;
    - are lost on process or host failure, so remote owners need expiry or reaping.

    Callback failures or timeouts do not prevent later owners from tearing down
    their resources. Total cleanup is bounded by the number of registered callbacks
    times the per-callback timeout. Cleanup has no
    durable retry after this context is discarded.
    """

    episode_id: EpisodeId
    cleanup_timeout_seconds: float
    _cleanups: list[_CleanupEntry] = field(default_factory=list)
    failure_report: FailureReport = field(default_factory=FailureReport)

    def register_cleanup(self, name: str, callback: CleanupCallback) -> CleanupHandle:
        entry = _CleanupEntry(name=name, callback=callback)
        self._cleanups.append(entry)
        return CleanupHandle(entry, self.cleanup_timeout_seconds)

    async def aclose(self) -> None:
        for entry in reversed(self._cleanups):
            try:
                async with asyncio.timeout(self.cleanup_timeout_seconds):
                    await entry.close()
            except TimeoutError as error:
                self._observe_failure(entry, error)
                # The owner's expiry or reaping takes over; the earlier-registered callbacks still run.
                LOGGER.error(f"Episode cleanup timed out: {entry.name} episode_id={self.episode_id}")
            except Exception as error:
                self._observe_failure(entry, error)
                LOGGER.exception(f"Episode cleanup failed: {entry.name}")

    def _observe_failure(self, entry: _CleanupEntry, error: Exception) -> None:
        observed = NgFailureEvent(
            boundary=FailureBoundary.ENVIRONMENT,
            failure_kind=SESSION_RELEASE_FAILED,
            failure_reason=f"Episode cleanup failed: {entry.name}"[:2000],
            stage="cleanup",
            operation="episode.cleanup",
        )
        observed.__cause__ = error
        # Cleanup is an observation, never a replacement for the episode's result or primary failure.
        self.failure_report = self.failure_report.merged(failure_report_for(observed))


def handled_episode_failure(
    *,
    terminal: bool,
    evidence: dict[str, object] | None = None,
    failure_kind: str | None = None,
    **event_fields: Any,
) -> NgFailureEvent:
    """The environment's handled episode failure: one ``NgFailureEvent`` with the local policy kept apart.

    ``terminal`` and ``evidence`` are runtime outcome data for the response adapter, never event fields; the
    adapter builds the protocol's ``EpisodeFailure`` after cleanup. ``failure_kind`` is the outcome's kind and
    may stay unset; the event then records the registered ``episode_failed``.
    """
    boundary = event_fields.pop("boundary", FailureBoundary.ENVIRONMENT)
    if FailureBoundary(boundary) is not FailureBoundary.ENVIRONMENT:
        raise ValueError("a handled episode failure is an environment failure")
    failure = NgFailureEvent(
        boundary=FailureBoundary.ENVIRONMENT, failure_kind=failure_kind or EPISODE_FAILED, **event_fields
    )
    failure.episode_terminal = terminal
    failure.episode_evidence = evidence if evidence is not None else {}
    failure.episode_failure_kind = failure_kind
    return failure


def is_handled_episode_failure(error: BaseException) -> bool:
    """Whether ``error`` is an environment failure built by ``handled_episode_failure``."""
    return (
        isinstance(error, NgFailureEvent)
        and error.boundary == FailureBoundary.ENVIRONMENT
        and hasattr(error, "episode_terminal")
    )


# Request Timeout, Too Early and Too Many Requests clear on a later attempt.
# Any other 4xx faults the request itself, so a retry is refused the same way.
_RETRYABLE_CLIENT_ERROR_STATUSES = frozenset({408, 425, 429})


def is_retryable_dependency_status(status: int) -> bool:
    """Return whether an HTTP error status from another server is worth another episode attempt.

    408, 425, 429 and every 5xx are retried; every other 4xx is terminal.
    """
    return status in _RETRYABLE_CLIENT_ERROR_STATUSES or status >= 500


def is_retryable_dependency_error(error: BaseException) -> bool:
    """Return whether a failed call to another server is worth another episode attempt.

    An HTTP error is classified by its status (see ``is_retryable_dependency_status``). A refused or dropped
    connection and a timeout are transient. Any other error, such as a reply that fails validation, would fail the
    same way again.
    """
    if isinstance(error, NgFailureEvent) and error.http_error is not None:
        error = error.http_error
    if isinstance(error, ClientResponseError):
        return is_retryable_dependency_status(error.status)
    # A dropped connection mid-body raises ClientPayloadError, which is transient like a refused connection.
    return isinstance(error, (ClientConnectionError, ClientPayloadError, TimeoutError))


def dependency_failure(
    error: BaseException,
    *,
    stage: FailureStage,
    terminal: bool | None = None,
    name_error_type: bool = False,
    **fields: Any,
) -> NgFailureEvent:
    """Turn a failed call to another server into the episode's failure.

    ``failure_reason`` keeps the dependency's own explanation from its response body
    (see ``format_exception_with_response_body``), prefixed with the error's class name when ``name_error_type`` is set,
    and bounded to the failure's 2000 characters. ``terminal`` defaults to
    ``not is_retryable_dependency_error(error)``; pass it only where the stage decides on its own.
    ``fields`` retain partial evidence for the response adapter, such as a partial response.
    """
    description_error = (
        error.http_error if isinstance(error, NgFailureEvent) and error.http_error is not None else error
    )
    failure_reason = format_exception_with_response_body(description_error)
    if name_error_type:
        failure_reason = f"{type(description_error).__name__}: {failure_reason}"
    handled = handled_episode_failure(
        boundary=FailureBoundary.ENVIRONMENT,
        stage=stage,
        operation="episode",
        failure_reason=failure_reason[:2000],
        terminal=not is_retryable_dependency_error(error) if terminal is None else terminal,
        evidence=fields,
    )
    # The dependency's error is this failure's explicit cause, whether or not the caller writes ``from``.
    handled.__cause__ = error
    return handled


class BaseEnvironmentServer(SimpleServer, Generic[EpisodeRequestT, EpisodeResponseT]):
    """Expose a typed episode protocol with shared limits and cleanup."""

    config: BaseEnvironmentServerConfig
    request_model: ClassVar[type[EpisodeRequestT]]
    response_model: ClassVar[type[EpisodeResponseT]]
    _admission: asyncio.Semaphore | None = None
    _checkpoint: EnvironmentParticipant | None = None
    # How this server's episodes take part in partial-rollout checkpoints.
    # A protocol whose ``run`` records checkpoint boundaries declares "exported".
    # Its episodes continue after a restore.
    # The default fails closed: episodes of a protocol that records no boundaries cannot be continued.
    # They are restarts, which never hold up a checkpoint and start over from their input after a crash.
    checkpoint_mode: ClassVar[CheckpointMode] = "restart_only"
    # Whether this protocol puts a training framework's sampling values (the request's ``model_sampling``) in the
    # model URLs of the agents the framework trains, and ``null`` in every other participant's
    # (``model_sampling_segment``). The default fails closed: such an episode is refused, because the agents it trains
    # would sample with other values.
    applies_model_sampling: ClassVar[bool] = False

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._admission = (
            MeteredSemaphore(
                self.config.max_concurrent_episodes,
                component="environment_server",
                limit_name="episodes",
                server_name=self.config.name,
                admission_timeout_s=self.config.queue_timeout_seconds,
            )
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

    async def episode_timeout_seconds(self, request: EpisodeRequestT) -> float | None:
        """The wall-clock bound of this episode: ``default_episode_timeout_seconds`` (``None`` is no bound).

        A protocol whose tasks carry their own budgets may derive a longer one per episode (the Harbor server
        does for a multi-step task whose steps add up to more than the default); it must never raise here, as
        nothing of the episode has started yet.
        """
        return self.config.default_episode_timeout_seconds

    @staticmethod
    def model_sampling_segment(request: BaseEpisodeRequest[Any], *, trained: bool) -> str | None:
        """The ``/ng-sampling/`` value for one participant's model calls in this episode, or ``None`` for none.

        A participant the training framework trains samples with the values the episode carries. Every other
        participant gets ``null``: it keeps its own sampling, or its Model Server's ``sampling_overrides``. An episode
        without values, as in Gym-only evaluation, gives every participant none, so its URLs stay as they are.
        """
        if request.model_sampling is None:
            return None
        return encode_model_sampling(request.model_sampling) if trained else MODEL_SAMPLING_NULL

    async def run_request(self, request: EpisodeRequestT) -> EpisodeResponseT:
        if request.model_sampling is not None and not self.applies_model_sampling:
            return self.failure_response(
                request,
                EpisodeFailure(
                    failure_reason=(
                        f"{type(self).__name__} does not apply a training framework's sampling values, but this "
                        "episode carries them (model_sampling), so the agents the framework trains would sample with "
                        "other values. Run this environment without them, or make it apply them "
                        "(BaseEnvironmentServer.applies_model_sampling)."
                    ),
                    terminal=True,
                    stage="admission",
                ),
            )
        acquired = False
        if self._admission is not None:
            try:
                # The semaphore applies queue_timeout_seconds and counts the timeout.
                await self._admission.acquire()
            except TimeoutError as error:
                return self.failure_response(
                    request,
                    EpisodeFailure(
                        failure_reason="Episode admission timed out",
                        terminal=False,
                        stage="admission",
                    ),
                    report=reply_failure_report(error),
                )
            acquired = True

        cleanup = CleanupContext(
            episode_id=request.episode_id,
            cleanup_timeout_seconds=float(self.config.cleanup_timeout_seconds),
        )
        response: EpisodeResponseT
        handled: NgFailureEvent | None = None
        cancelled: asyncio.CancelledError | None = None
        # Set once this request owns its episode's checkpoint state.
        # A begin refused as a duplicate or a retired attempt owns nothing,
        # so it must not end the live episode with the same key.
        begun = False
        # Opened inside the guard below: a protocol's bound may await a lookup (the Harbor server resolves the
        # task's snapshot), and a request cancelled or failing during it must still release its admission slot.
        deadline: asyncio.Timeout | None = None
        # Every downstream call of this episode, including final cleanup, carries its attempt-qualified
        # rollout id, so Resources and Model Server calls stay correlated with the rollout.
        with rollout_context(request.episode_id.capture_key):
            try:
                try:
                    deadline = asyncio.timeout(await self.episode_timeout_seconds(request))
                    async with deadline:
                        if self._checkpoint is not None:
                            await self._checkpoint.claim(request.episode_id)
                            self._checkpoint.begin(
                                request.episode_id,
                                request.task.model_dump(mode="json"),
                                deadline,
                                restart=self.checkpoint_mode == "restart_only",
                            )
                            begun = True
                        response = await self.run(request, cleanup)
                except TimeoutError as error:
                    if deadline is not None and deadline.expired():
                        response = self.failure_response(
                            request,
                            EpisodeFailure(
                                failure_reason="Episode timed out",
                                terminal=False,
                            ),
                            report=reply_failure_report(error),
                        )
                    else:
                        response = self._unhandled_failure_response(request, error)
                except ControlError:
                    # A checkpoint refused this episode before it started; the caller retries it later.
                    raise
                except NgFailureEvent as error:
                    if not is_handled_episode_failure(error):
                        response = self._unhandled_failure_response(request, error)
                    else:
                        handled = error
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
        if handled is not None:
            response = self.failure_response(request, handled)
        response = self.response_model.model_validate(response)
        if cleanup.failure_report.events:
            response = response.model_copy(
                update={"failure_report": (response.failure_report or FailureReport()).merged(cleanup.failure_report)}
            )
        self.validate_response_identity(request, response)
        return response

    def setup_environment_checkpoint(self, app: FastAPI) -> None:
        """Take part in partial-rollout checkpoints when they are enabled.

        Every episode is tracked.
        A protocol that records boundaries (``checkpoint_mode = "exported"``) continues after a restore;
        the episodes of a protocol that records none are restarts,
        which never hold up a checkpoint and start over from their input after a crash.
        Call this from any ``setup_webserver`` that builds its own app.
        """
        # The declaration is a class attribute nothing else checks: a typo would act as a mode of its own.
        if self.checkpoint_mode not in _ENVIRONMENT_CHECKPOINT_MODES:
            raise ValueError(
                f"{type(self).__name__} declares checkpoint_mode={self.checkpoint_mode!r}; "
                f"expected one of {_ENVIRONMENT_CHECKPOINT_MODES}"
            )
        settings = checkpoint_settings(getattr(self.server_client, "global_config_dict", None))
        if settings is None:
            return
        self._checkpoint = EnvironmentParticipant(mode=self.checkpoint_mode)
        install_server_participant(
            app,
            self._checkpoint,
            num_workers=self.config.num_workers or 1,
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

        Call it when a seed reply says so (``nemo_gym.server_runtime._checkpoint.steps.seed_restarts``).
        The episode then never holds up a checkpoint and is never exported; after a crash,
        the controller starts it over from its input.
        """
        if self._checkpoint is not None:
            await self._checkpoint.mark_restart(request.episode_id)

    def checkpoint_step(self, request: EpisodeRequestT, step_type: StepType) -> AbstractAsyncContextManager[None]:
        """Run one protocol step as a ``wait`` or ``replay`` step (see ``nemo_gym.server_runtime._checkpoint.steps``).

        Use ``replay`` only for a step that is safe to run again after a crash
        and does not change checkpointed state; everything else waits.
        """
        if self._checkpoint is None:
            return nullcontext()
        return self._checkpoint.step(request.episode_id, step_type)

    def _unhandled_failure_response(self, request: EpisodeRequestT, error: Exception) -> EpisodeResponseT:
        LOGGER.exception(f"Unhandled environment server error: episode_id={request.episode_id}")
        failure_reason = f"Unhandled environment server error: {type(error).__name__}: {error}"
        reported = handled_episode_failure(
            boundary=FailureBoundary.ENVIRONMENT,
            failure_reason=failure_reason[:2000],
            operation="episode",
            terminal=True,
        )
        reported.__cause__ = error
        return self.failure_response(request, reported)

    def failure_response(
        self,
        request: EpisodeRequestT,
        failure: EpisodeFailure | NgFailureEvent,
        *,
        report: FailureReport | None = None,
    ) -> EpisodeResponseT:
        """Adapt runtime event, local policy and evidence to the protocol's wire failure model.

        The failure's ``FailureReport`` goes in the reply's ``failure_report``, beside the failure.
        """
        if is_handled_episode_failure(failure):
            report = report_for_current_request(failure)
            fields = failure.episode_evidence | {
                "failure_reason": failure.failure_reason,
                "failure_kind": failure.episode_failure_kind,
                "stage": failure.stage,
                "terminal": failure.episode_terminal,
            }
        else:
            # A protocol may still return a wire failure directly instead of raising a runtime failure.
            fields = failure.model_dump(mode="json")
        reply = {"episode_id": request.episode_id, "task_id": request.task.task_id, "failure": fields}
        if report is not None and (report.events or report.omitted):
            reply["failure_report"] = report.model_dump(mode="json")
        return self.response_model.model_validate(reply)

    @staticmethod
    def validate_response_identity(request: EpisodeRequestT, response: EpisodeResponseT) -> None:
        if response.episode_id != request.episode_id:
            raise ValueError("response episode_id does not match request")
        if response.task_id != request.task.task_id:
            raise ValueError("response task_id does not match request")

    @abstractmethod
    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Aggregate per-rollout scores into task-level metrics."""


class BaseEnvironmentServerWithToolsConfig(BaseEnvironmentServerConfig):
    """Configure how a ``BaseEnvironmentServerWithTools`` serves its episodes' tools."""

    tool_call_timeout_seconds: PositiveFloat | None = Field(
        default=None,
        description=(
            "How long one tool call may run before the server cancels it and answers 504. Calls to one episode run "
            "one at a time, so this also bounds how long a hung call, or one whose client disconnected, makes the "
            "episode's next calls wait. None lets a call run until it returns or the episode ends."
        ),
    )


@dataclass
class ServedTools:
    """The grants that give the agent one episode's tools, and the calls it has made so far."""

    tool_accesses: list[ToolAccess]
    calls: list[ToolCallRecord]


@dataclass
class _OpenEpisode:
    claims: EpisodeTokenClaims
    toolset: EpisodeToolset
    specs: dict[str, ToolSpec]
    calls: list[ToolCallRecord]
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # The call that holds the lock, so the end of the episode can stop it.
    running: Optional[asyncio.Task] = None
    # The first handled episode failure a tool raised, such as a crashed sandbox; the episode fails with it.
    failure: Optional[NgFailureEvent] = None


class _EpisodeTokensOpenOnlyTools:
    """Refuse a request that carries an episode token anywhere but the episode's tools.

    The agent sends the token with every call it makes through the grant, so this stops a model's call named
    ``run`` (or ``../run``) from starting another episode on this server.
    """

    def __init__(self, app: Any, is_tool_path: Callable[[str], bool]) -> None:
        self._app = app
        self._is_tool_path = is_tool_path

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") == "http"
            and EPISODE_TOKEN_HEADER in Headers(scope=scope)
            and not self._is_tool_path(scope["path"])
        ):
            response = JSONResponse(
                status_code=403, content={"detail": "An episode token grants only the episode's tools"}
            )
            await response(scope, receive, send)
            return
        await self._app(scope, receive, send)


class BaseEnvironmentServerWithTools(BaseEnvironmentServer[EpisodeRequestT, EpisodeResponseT]):
    """An Environment Server that serves each episode's tools itself, so it needs no Resources Server.

    Other Environment Servers grant the agent a Resources Server's tools; this one owns the tools, their state and
    verification.

    ``run_episode`` holds the episode's state, and ``serve_tools`` grants the agent the episode's tools until the
    episode ends. A tool call is a separate request to ``POST /<tool name>`` at the server's root, as on a
    Resources Server, with the episode token in the ``X-NeMo-Gym-Episode-Token`` header. A request with an episode
    token reaches only the tools. Calls to one episode run one at a time, each under the episode's rollout, and
    each is recorded in ``ServedTools.calls``. When the episode ends, a call still running is cancelled. A call
    that runs longer than ``tool_call_timeout_seconds`` is cancelled and answered 504. A client that disconnects
    cancels its call, which is recorded.

    The tool route takes every one-segment ``POST`` path that the server does not route when ``setup_webserver``
    returns, so a subclass that adds its own routes gives them more than one path segment.

    An agent may send the model's ``call_id`` in ``X-NeMo-Gym-Tool-Call-Id``; the record keeps it, to join the call
    to the agent's trajectory. Every call runs, as on a Resources Server: an agent's client that resends a call after
    a dropped connection runs it again.

    The base refuses ``checkpoint_mode = "exported"``: a restored attempt has a new episode ID, and the agent's
    grant still names the old one.

    A tool that raises a handled episode failure (``handled_episode_failure``), such as for a crashed sandbox,
    fails the episode with it when ``run_episode`` returns; the agent sees that call fail meanwhile.

    A subclass implements ``run_episode``, not ``run``, so the base tracks the episode's cleanups. When the server
    shuts down, it finishes the cleanups of every episode still running, such as stopping its sandbox.
    """

    config: BaseEnvironmentServerWithToolsConfig
    _episode_tokens: Optional[EpisodeTokens] = None
    _open_episodes: Optional[dict[EpisodeId, _OpenEpisode]] = None
    # Keyed by id(): two attempts of one episode can run at once without checkpointing.
    _running_cleanups: Optional[dict[int, CleanupContext]] = None
    _reserved_tool_names: frozenset[str] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} overrides run; a BaseEnvironmentServerWithTools implements run_episode, so the base "
                "can track the episode's cleanups for the shutdown sweep"
            )
        if cls.checkpoint_mode == "exported":
            raise TypeError(
                f"{cls.__name__} sets checkpoint_mode='exported', which BaseEnvironmentServerWithTools does not "
                "support: an agent's grant names the episode attempt that a restore replaces"
            )

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._episode_tokens = EpisodeTokens(episode_token_secret())
        self._open_episodes = {}
        self._running_cleanups = {}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # Added before RolloutContextMiddleware, so it runs inside it and sees the path without a rollout prefix.
        app.add_middleware(_EpisodeTokensOpenOnlyTools, is_tool_path=self._is_tool_path)
        # A tool call under /ng-rollout/<id>/ (an agent that correlates its calls with the rollout) routes as bare.
        app.add_middleware(RolloutContextMiddleware)
        # A tool cannot take the name of the server's own routes; the catch-all below comes after them.
        self._reserved_tool_names = frozenset(
            route.path.strip("/").split("/")[0] for route in app.routes if getattr(route, "path", "/") != "/"
        )
        app.post("/{tool_name}")(self._tool_call_endpoint)

        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[Any]:
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                await self.finish_running_cleanups()

        app.router.lifespan_context = lifespan
        return app

    def _is_tool_path(self, path: str) -> bool:
        segments = path.strip("/").split("/")
        return len(segments) == 1 and segments[0] not in self._reserved_tool_names

    async def run(self, request: EpisodeRequestT, cleanup: CleanupContext) -> EpisodeResponseT:
        key = id(cleanup)
        self._running_cleanups[key] = cleanup

        async def forget() -> None:
            # A cleanup that failed, such as a sandbox stop that got a provider error, stays tracked, so the shutdown
            # sweep tries it again.
            if not any(entry.active for entry in cleanup._cleanups if entry is not tracking.entry):
                self._running_cleanups.pop(key, None)

        # Registered first, so it runs last: the episode stays tracked until each of its cleanups has finished.
        tracking = cleanup.register_cleanup("shutdown tracking", forget)
        response = await self.run_episode(request, cleanup)
        episode = self._open_episodes.get(request.episode_id)
        if episode is not None and episode.failure is not None:
            # A tool met a failure the episode cannot recover from; the agent saw the call fail.
            raise episode.failure
        return response

    @abstractmethod
    async def run_episode(self, request: EpisodeRequestT, cleanup: CleanupContext) -> EpisodeResponseT:
        """Run one episode, as ``BaseEnvironmentServer.run`` does. Register each resource's release in ``cleanup``."""

    async def finish_running_cleanups(self) -> None:
        """Finish the cleanups of every episode still running; the server runs this when it shuts down.

        Shutdown cancels the running ``/run`` requests and does not wait for them, so a second cancellation can
        interrupt an episode's cleanup partway. Cleanups are idempotent, so this runs each one that has not
        finished, under its episode's rollout. They all start at once, so a sandbox stop does not wait behind a
        call to a Gym server that is shutting down too. The sweep waits ``cleanup_timeout_seconds`` in all, and
        logs each episode whose cleanup it could not confirm.
        """

        async def finish(cleanup: CleanupContext, entry: _CleanupEntry) -> None:
            with rollout_context(cleanup.episode_id.capture_key):
                await entry.close()

        tasks = {
            asyncio.ensure_future(finish(cleanup, entry)): cleanup.episode_id
            for cleanup in list(self._running_cleanups.values())
            for entry in cleanup._cleanups
            if entry.active
        }
        if not tasks:
            return
        await asyncio.wait(tasks, timeout=self.config.cleanup_timeout_seconds)
        unconfirmed: set[EpisodeId] = set()
        for task, episode_id in tasks.items():
            if not task.done():
                task.cancel()
                unconfirmed.add(episode_id)
            elif task.cancelled() or task.exception() is not None:
                unconfirmed.add(episode_id)
            # Consume a failure that lands after the sweep stops waiting.
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        for episode_id in sorted(unconfirmed, key=lambda episode_id: episode_id.capture_key):
            LOGGER.error(f"Episode cleanup unconfirmed at shutdown: episode_id={episode_id.capture_key}")

    def multiple_workers_unsupported_reason(self) -> Optional[str]:
        return "a tool call reaches whichever worker accepts it, and only the worker that runs the episode holds it"

    async def serve_tools(
        self, request: EpisodeRequestT, toolset: EpisodeToolset, cleanup: CleanupContext
    ) -> ServedTools:
        """Grant the agent ``toolset`` for this episode, until the episode's cleanup runs.

        Returns the grants for the agent's seed (``AgentSeedSessionRequest.tool_accesses``) and the list that
        records each call. A call after the episode ends gets 410. Declare the tools to the model with
        ``ToolSpec.function_tool``.
        """
        episode_id = request.episode_id
        if episode_id in self._open_episodes:
            raise ValueError(f"Episode {episode_id.capture_key} already serves its tools")
        listed = list(toolset.specs())
        specs = {spec.name: spec for spec in listed}
        if len(specs) != len(listed):
            raise ValueError("Episode tool names must be unique")
        reserved = sorted(specs.keys() & self._reserved_tool_names)
        if reserved:
            raise ValueError(f"Episode tools cannot take the names of the server's routes: {', '.join(reserved)}")

        token, claims = self._episode_tokens.mint(episode_id)
        episode = _OpenEpisode(claims=claims, toolset=toolset, specs=specs, calls=[])
        self._open_episodes[episode_id] = episode

        async def stop_serving() -> None:
            if self._open_episodes.get(episode_id) is episode:
                del self._open_episodes[episode_id]
            # A call still running would change state that the episode no longer owns, such as a stopped sandbox.
            running = episode.running
            if running is not None and not running.done():
                running.cancel()
                await asyncio.wait({running})

        cleanup.register_cleanup("episode tools", stop_serving)
        base_url = self.server_client._resolve_base_url(self.config.name)
        grant = DirectHTTPToolAccess(
            name=f"{self.config.name}.direct_http",
            required=True,
            base_url=base_url,
            headers={EPISODE_TOKEN_HEADER: token},
        )
        return ServedTools(tool_accesses=[grant], calls=episode.calls)

    async def _tool_call_endpoint(self, tool_name: str, request: Request) -> JSONResponse:
        claims = self._episode_tokens.read(request.headers.get(EPISODE_TOKEN_HEADER, ""))
        if claims is None:
            return _refusal(401, f"A tool call needs a valid {EPISODE_TOKEN_HEADER} header")
        episode = self._open_episodes.get(claims.episode_id)
        if episode is None:
            return _refusal(410, f"Episode {claims.episode_id.capture_key} is not running on this server")
        if not episode.claims.matches(claims):
            return _refusal(403, "The episode token was not granted for this episode")

        try:
            arguments = await request.json()
        except ValueError:
            arguments = None
        call_id = request.headers.get(TOOL_CALL_ID_HEADER) or None
        async with episode.lock:
            # The episode can end while this call waits for its body or for the calls before it.
            if self._open_episodes.get(claims.episode_id) is not episode:
                return _refusal(410, f"Episode {claims.episode_id.capture_key} is not running on this server")
            episode.running = asyncio.ensure_future(self._call_tool(episode, tool_name, arguments, call_id))
            try:
                status, output, is_error = await episode.running
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    raise
                # The end of the episode cancelled the call, not this request.
                return _refusal(410, f"Episode {claims.episode_id.capture_key} ended while the call ran")
            finally:
                episode.running = None
        return JSONResponse(
            status_code=status, content=output, headers={TOOL_ERROR_HEADER: "true"} if is_error else None
        )

    async def _call_tool(
        self, episode: _OpenEpisode, name: str, arguments: Any, call_id: Optional[str] = None
    ) -> tuple[int, JsonValue, bool]:
        """Run one call, under the episode's rollout, and record it, including a call to a tool it does not have."""
        started = perf_counter()
        status, is_error = 200, False

        def record(output: JsonValue) -> None:
            episode.calls.append(
                ToolCallRecord(
                    name=name,
                    arguments=arguments,
                    output=output,
                    is_error=is_error,
                    duration_seconds=perf_counter() - started,
                    call_id=call_id,
                )
            )

        try:
            with rollout_context(episode.claims.episode_id.capture_key):
                if name not in episode.specs:
                    detail = f"Episode {episode.claims.episode_id.capture_key} has no tool {name!r}"
                    status, is_error, output = 404, True, {"detail": detail}
                elif not isinstance(arguments, dict):
                    raise ToolArgumentsError("Tool arguments must be a JSON object")
                else:
                    timeout = self.config.tool_call_timeout_seconds
                    deadline = asyncio.timeout(timeout)
                    try:
                        async with deadline:
                            result = await episode.toolset.call(name, arguments)
                    except TimeoutError:
                        # A TimeoutError the tool raises itself is a failure of the tool, handled below.
                        if not deadline.expired():
                            raise
                        status, is_error = 504, True
                        output = {"detail": f"Tool {name!r} did not finish within {timeout} seconds"}
                    else:
                        output, is_error = result.output, result.is_error
        except ToolArgumentsError as error:
            status, is_error, output = 422, True, {"detail": str(error)}
        except asyncio.CancelledError:
            is_error = True
            record({"detail": f"Tool {name!r} was cancelled"})
            raise
        except Exception as error:
            LOGGER.exception(f"Episode tool {name!r} failed: episode_id={episode.claims.episode_id}")
            status, is_error, output = 500, True, {"detail": f"Tool {name!r} failed: {type(error).__name__}"}
            if is_handled_episode_failure(error) and episode.failure is None:
                episode.failure = error
        if not _is_json(output):
            status, is_error = 500, True
            output = {"detail": f"Tool {name!r} returned a value that is not JSON, such as NaN or Infinity"}
        record(output)
        return status, output, is_error


def _is_json(value: JsonValue) -> bool:
    try:
        json.dumps(value, allow_nan=False)
    except ValueError:
        return False
    return True


def _refusal(status: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail})
