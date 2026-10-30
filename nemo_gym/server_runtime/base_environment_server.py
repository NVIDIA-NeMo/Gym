# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared lifecycle for environment servers, and the base for those that serve each episode's tools."""

import asyncio
import logging
from abc import abstractmethod
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, ClassVar, Generic, Optional, TypeVar

from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError
from anyio import CancelScope
from fastapi import Body, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ConfigDict, Field, JsonValue, PositiveFloat, PositiveInt, model_validator
from typing_extensions import Self

from nemo_gym.client.http_client import format_exception_with_response_body
from nemo_gym.config.config_types import AggregateMetrics, AggregateMetricsRequest, BaseRunServerInstanceConfig
from nemo_gym.core.episode import EpisodeId, rollout_context
from nemo_gym.core.failure_kinds import FailureStage
from nemo_gym.server_runtime._checkpoint.control import CheckpointMode
from nemo_gym.server_runtime._checkpoint.environment import EnvironmentParticipant
from nemo_gym.server_runtime._checkpoint.errors import ControlError
from nemo_gym.server_runtime._checkpoint.participant_workers import install_server_participant
from nemo_gym.server_runtime._checkpoint.settings import checkpoint_settings
from nemo_gym.server_runtime._checkpoint.steps import StepType
from nemo_gym.server_runtime.base_server import SimpleServer
from nemo_gym.server_runtime.episode_tokens import EPISODE_TOKEN_HEADER, EpisodeTokenClaims, EpisodeTokens
from nemo_gym.server_runtime.episode_tools import EpisodeToolset, ToolArgumentsError, ToolCallRecord, ToolSpec
from nemo_gym.server_runtime.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure
from nemo_gym.server_runtime.request_handling.rollout_context import RolloutContextMiddleware
from nemo_gym.server_runtime.tool_access import DirectHTTPToolAccess, ToolAccess


LOGGER = logging.getLogger(__name__)
# An environment server either records boundaries or restarts its episodes; it has no stateless mode.
_ENVIRONMENT_CHECKPOINT_MODES = ("restart_only", "exported")

EpisodeRequestT = TypeVar("EpisodeRequestT", bound=BaseEpisodeRequest[Any])
EpisodeResponseT = TypeVar("EpisodeResponseT", bound=BaseEpisodeResponse[Any])
EpisodeFailureT = TypeVar("EpisodeFailureT", bound=EpisodeFailure)
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

    def register_cleanup(self, name: str, callback: CleanupCallback) -> CleanupHandle:
        entry = _CleanupEntry(name=name, callback=callback)
        self._cleanups.append(entry)
        return CleanupHandle(entry, self.cleanup_timeout_seconds)

    async def aclose(self) -> None:
        for entry in reversed(self._cleanups):
            try:
                async with asyncio.timeout(self.cleanup_timeout_seconds):
                    await entry.close()
            except TimeoutError:
                # The owner's expiry or reaping takes over; the earlier-registered callbacks still run.
                LOGGER.error(f"Episode cleanup timed out: {entry.name} episode_id={self.episode_id}")
            except Exception:
                LOGGER.exception(f"Episode cleanup failed: {entry.name}")


class HandledEpisodeError(Exception):
    """Carry a failure that belongs in the episode response."""

    def __init__(self, failure: EpisodeFailure) -> None:
        super().__init__(failure.failure_reason)
        self.failure = failure


# Request Timeout, Too Early and Too Many Requests clear on a later attempt.
# Any other 4xx faults the request itself, so a retry is refused the same way.
_RETRYABLE_CLIENT_ERROR_STATUSES = frozenset({408, 425, 429})


def is_retryable_dependency_status(status: int) -> bool:
    """Return whether an HTTP error status from another server is worth another episode attempt.

    408, 425, 429 and every 5xx are retried; every other 4xx is final.
    """
    return status in _RETRYABLE_CLIENT_ERROR_STATUSES or status >= 500


def is_retryable_dependency_error(error: BaseException) -> bool:
    """Return whether a failed call to another server is worth another episode attempt.

    An HTTP error is classified by its status (see ``is_retryable_dependency_status``). A refused or dropped
    connection and a timeout are transient. Any other error, such as a reply that fails validation, would fail the
    same way again.
    """
    if isinstance(error, ClientResponseError):
        return is_retryable_dependency_status(error.status)
    # A dropped connection mid-body raises ClientPayloadError, which is transient like a refused connection.
    return isinstance(error, (ClientConnectionError, ClientPayloadError, TimeoutError))


def dependency_failure(
    error: BaseException,
    *,
    failure_model: type[EpisodeFailureT],
    stage: FailureStage,
    terminal: bool | None = None,
    name_error_type: bool = False,
    **fields: Any,
) -> HandledEpisodeError:
    """Turn a failed call to another server into the episode's failure.

    ``failure_reason`` keeps the dependency's own explanation from its response body
    (see ``format_exception_with_response_body``), prefixed with the error's class name when ``name_error_type`` is set,
    and bounded to the failure's 2000 characters. ``terminal`` defaults to
    ``not is_retryable_dependency_error(error)``; pass it only where the stage decides on its own.
    ``fields`` fill the failure model's own fields, such as a partial response.
    """
    failure_reason = format_exception_with_response_body(error)
    if name_error_type:
        failure_reason = f"{type(error).__name__}: {failure_reason}"
    return HandledEpisodeError(
        failure_model(
            stage=stage,
            failure_reason=failure_reason[:2000],
            terminal=not is_retryable_dependency_error(error) if terminal is None else terminal,
            **fields,
        )
    )


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

    async def episode_timeout_seconds(self, request: EpisodeRequestT) -> float | None:
        """The wall-clock bound of this episode: ``default_episode_timeout_seconds`` (``None`` is no bound).

        A protocol whose tasks carry their own budgets may derive a longer one per episode (the Harbor server
        does for a multi-step task whose steps add up to more than the default); it must never raise here, as
        nothing of the episode has started yet.
        """
        return self.config.default_episode_timeout_seconds

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


class BaseEnvironmentServerWithTools(BaseEnvironmentServer[EpisodeRequestT, EpisodeResponseT]):
    """An Environment Server that serves each episode's tools itself, so it needs no Resources Server.

    Other Environment Servers grant the agent a Resources Server's tools; this one owns the tools, their state and
    verification.

    ``run`` holds the episode's state, and ``serve_tools`` grants the agent the episode's tools until the episode
    ends. A tool call is a separate request to ``POST /<tool name>`` at the server's root, as on a Resources
    Server, with the episode token in the ``X-NeMo-Gym-Episode-Token`` header. Calls to one episode run one at a
    time, and each is recorded in ``ServedTools.calls``.
    """

    _episode_tokens: Optional[EpisodeTokens] = None
    _open_episodes: Optional[dict[EpisodeId, _OpenEpisode]] = None
    _reserved_tool_names: frozenset[str] = frozenset()

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._episode_tokens = EpisodeTokens(self.get_session_middleware_key())
        self._open_episodes = {}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # A tool call under /ng-rollout/<id>/ (an agent that correlates its calls with the rollout) routes as bare.
        app.add_middleware(RolloutContextMiddleware)
        # A tool cannot take the name of the server's own routes; the catch-all below comes after them.
        self._reserved_tool_names = frozenset(
            route.path.strip("/").split("/")[0] for route in app.routes if getattr(route, "path", "/") != "/"
        )
        app.post("/{tool_name}")(self._tool_call_endpoint)
        return app

    def multiple_workers_unsupported_reason(self) -> Optional[str]:
        return "a tool call reaches whichever worker accepts it, and only the worker that runs the episode holds it"

    async def serve_tools(
        self, request: EpisodeRequestT, toolset: EpisodeToolset, cleanup: CleanupContext
    ) -> ServedTools:
        """Grant the agent ``toolset`` for this episode, until the episode's cleanup runs.

        Returns the grants for the agent's seed (``AgentSeedSessionRequest.tool_accesses``) and the list that
        records each call. A call after the episode ends gets 410.
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
        if tool_name not in episode.specs:
            return _refusal(404, f"Episode {claims.episode_id.capture_key} has no tool {tool_name!r}")

        try:
            arguments = await request.json()
        except ValueError:
            arguments = None
        async with episode.lock:
            return await self._call_tool(episode, tool_name, arguments)

    async def _call_tool(self, episode: _OpenEpisode, name: str, arguments: Any) -> JSONResponse:
        started = perf_counter()
        status, is_error = 200, False
        try:
            if not isinstance(arguments, dict):
                raise ToolArgumentsError("Tool arguments must be a JSON object")
            result = await episode.toolset.call(name, arguments)
            output, is_error = result.output, result.is_error
        except ToolArgumentsError as error:
            status, is_error, output = 422, True, {"detail": str(error)}
        except Exception as error:
            LOGGER.exception(f"Episode tool {name!r} failed: episode_id={episode.claims.episode_id}")
            status, is_error, output = 500, True, {"detail": f"Tool {name!r} failed: {type(error).__name__}"}
        episode.calls.append(
            ToolCallRecord(
                name=name,
                arguments=arguments,
                output=output,
                is_error=is_error,
                duration_seconds=perf_counter() - started,
            )
        )
        return JSONResponse(status_code=status, content=output)


def _refusal(status: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail})
