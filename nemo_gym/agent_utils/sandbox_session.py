# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-server lifecycle for one supervised harness activation."""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import cast
from uuid import uuid4

from openai.types.responses import ResponseError

from nemo_gym.agent_utils import process_supervisor
from nemo_gym.agent_utils.process_supervisor import CleanupReceipt
from nemo_gym.agent_utils.session_capture import SessionCapture, SessionCaptureConfig
from nemo_gym.agent_utils.supervisor_client import (
    OUTPUT_LOG_FILE,
    STOP_REQUEST_FILE,
    SUPERVISOR_FILE,
    remove_session_directory,
    stop_and_confirm_cleanup,
    supervised_launch_command,
    supervision_timeouts,
)
from nemo_gym.base_responses_api_agent import ModelEndpoint, TokenCapture
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.rollout_observability import AgentObservationBundle, ObservationGap
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.utils import read_text


LOG = logging.getLogger(__name__)


def _reuse_or_start[T](
    task: asyncio.Task[T] | None, factory: Callable[[], Coroutine[object, object, T]]
) -> asyncio.Task[T]:
    if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
        return asyncio.create_task(factory())
    return task


# Observation gap code for an activation answered without running its harness; the gap detail is the reason.
HARNESS_NOT_RUN_GAP = "harness_not_run"


def harness_not_run_response(
    body: NeMoGymResponseCreateParamsNonStreaming, *, model: str, reason: str
) -> NeMoGymResponse:
    """A failed response with an empty assistant message, for an activation whose harness never ran.

    An agent returns this instead of raising when the harness must not run, for example because its session
    capture did not start, so the environment server still verifies and closes the episode. The reason is the
    response's error message; pair it with :func:`harness_not_run_observations` for the close response::

        if state.session.session_capture_failed:
            reason = state.session.token_capture().mask_reason
            state.observations = harness_not_run_observations(source="my_agent", reason=reason)
            return harness_not_run_response(body, model=self.config.model, reason=reason)
        return await state.session.execute(stage_activation=..., collect=...)
    """
    return NeMoGymResponse(
        id=f"resp_{uuid4().hex}",
        created_at=int(time()),
        model=model,
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id=f"msg_{uuid4().hex}",
                content=[NeMoGymResponseOutputText(text="", annotations=[])],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        status="failed",
        error=ResponseError(code="server_error", message=reason[:2000]),
        tool_choice=body.tool_choice,
        tools=body.tools,
        parallel_tool_calls=body.parallel_tool_calls,
    )


def harness_not_run_observations(*, source: str, reason: str) -> AgentObservationBundle:
    """The observations for an activation whose harness never ran: one ``harness_not_run`` gap with the reason."""
    return AgentObservationBundle(source=source, gaps=[ObservationGap(code=HARNESS_NOT_RUN_GAP, detail=reason)])


@dataclass(frozen=True, kw_only=True)
class SandboxCommand:
    """Harness argv and the installed interpreter used to launch its supervisor."""

    argv: list[str]
    python: str


@dataclass(kw_only=True)
class SandboxSession[Artifacts]:
    """Execute, stop, capture, then release an owned or borrowed sandbox.

    Runtime installation belongs in the adapter's seed hook. ``stage_activation``
    stages input and returns a command; ``collect`` copies harness artifacts
    to the agent server after cleanup is confirmed. It may run a bounded snapshot
    command, but must not restart the harness. Parsing responses stays in the adapter.

    A new session stages input and its supervisor, runs the harness process, then
    stops and collects artifacts. Execution retains the sandbox until close
    releases it. A failed borrowed cleanup remains closing and can be retried;
    an owned sandbox can fall back to provider stop. Closed means release succeeded.

    Each stop, collection and release phase is bounded by ``close_timeout`` (or
    the timeout passed to ``close``). Failed cleanup/release retains the handle
    for retry. Failed capture is recorded separately and does not prevent release.
    HTTP request binding and activation retries belong to the agent session layer.

    With ``session_capture`` configured, the adapter awaits ``start_session_capture`` after its runtime is
    ready and before it builds the harness command, and points the harness at the returned endpoint. Close
    collects the capture after the harness has stopped and before the sandbox is released, at most once, and
    ``token_capture`` returns it for the agent's close response. A capture that did not start, failed or timed
    out becomes a masked ``TokenCapture`` instead of an error; when it did not start, ``execute`` refuses to run
    and the adapter answers the activation without the harness (see ``session_capture_failed`` and
    ``harness_not_run_response``).
    """

    sandbox: AsyncSandbox
    session_dir: str
    workdir: str | None
    harness: str
    owns_sandbox: bool = False
    session_capture: SessionCaptureConfig | None = None
    launch_started: bool = field(default=False, init=False)
    cleanup: CleanupReceipt | None = field(default=None, init=False)
    artifacts: Artifacts | None = field(default=None, init=False)
    capture_error: Exception | None = field(default=None, init=False)
    sandbox_stopped: bool = field(default=False, init=False)
    _stage_task: asyncio.Future[SandboxCommand] | None = field(default=None, init=False)
    _exec_task: asyncio.Task[SandboxExecResult] | None = field(default=None, init=False)
    _finalize_task: asyncio.Task[None] | None = field(default=None, init=False)
    _close_task: asyncio.Task[None] | None = field(default=None, init=False)
    _collect: Callable[[], Awaitable[Artifacts]] | None = field(default=None, init=False)
    _capture_attempted: bool = field(default=False, init=False)
    _capture_component: SessionCapture | None = field(default=None, init=False)
    _capture_endpoint: ModelEndpoint | None = field(default=None, init=False)
    _token_capture: TokenCapture | None = field(default=None, init=False)

    @property
    def closing(self) -> bool:
        """Whether close has been requested, including a failed attempt."""
        return self._close_task is not None

    @property
    def closed(self) -> bool:
        """Whether sandbox release completed successfully."""
        task = self._close_task
        return task is not None and task.done() and not task.cancelled() and task.exception() is None

    @property
    def stop_request_path(self) -> str:
        """Marker path an adapter may pass to its harness for cancellation diagnostics."""
        return f"{self.session_dir}/{STOP_REQUEST_FILE}"

    @property
    def session_capture_failed(self) -> bool:
        """Whether a configured session capture cannot run, so the harness must not run either."""
        return self.session_capture is not None and self._capture_endpoint is None and self._token_capture is not None

    def token_capture(self) -> TokenCapture | None:
        """The collected (or masked) session capture; None without a configured capture or before close."""
        return self._token_capture

    async def start_session_capture(self) -> ModelEndpoint | None:
        """Start the configured capture once and return the endpoint the harness must call.

        Returns None when no capture is configured or when it did not start; a failed or cancelled start is
        recorded as a masked ``token_capture`` and the component is aborted. Cancellation still propagates.
        """
        if self.session_capture is None:
            return None
        if self._capture_component is not None:
            raise RuntimeError(f"{self.harness} sandbox session capture was already started")
        if self.closing or self._stage_task is not None:
            raise RuntimeError(f"{self.harness} sandbox session must start its capture before activation")
        self._capture_component = self.session_capture.build()
        try:
            endpoint = await self._capture_component.start(self.sandbox)
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError):
                reason = "start was cancelled"
            else:
                reason = f"{type(error).__name__}: {error}"
                LOG.exception("%s session capture did not start; the harness will not run", self.harness)
            self._token_capture = _masked(f"capture did not start: {reason}")
            await self._abort_capture()
            if not isinstance(error, Exception):
                raise
            return None
        if self._token_capture is not None:
            # Close ran while the capture was starting and has already recorded it as masked.
            await self._abort_capture()
            return None
        self._capture_endpoint = endpoint
        return endpoint

    async def _collect_capture(self) -> None:
        """Collect the running capture once, bounded by its collect timeout; masks instead of raising."""
        if self.session_capture is None or self._token_capture is not None:
            return
        if self._capture_component is None or self._capture_endpoint is None:
            # Never started, or still starting (start aborts it once it returns).
            self._token_capture = _masked("capture was not running when the session closed")
            return
        timeout = self.session_capture.collect_timeout_seconds
        deadline = asyncio.timeout(timeout)
        try:
            async with deadline:
                self._token_capture = await self._capture_component.collect(self.sandbox)
            return
        except TimeoutError as error:
            if not deadline.expired():
                LOG.exception("Collecting the %s session capture failed", self.harness)
                reason = f"capture collection failed: {type(error).__name__}: {error}"
            else:
                LOG.error("Collecting the %s session capture exceeded %ss", self.harness, timeout)
                reason = f"capture collection exceeded {timeout}s"
        except asyncio.CancelledError:
            self._token_capture = _masked("capture collection was cancelled")
            await self._abort_capture()
            raise
        except Exception as error:
            LOG.exception("Collecting the %s session capture failed", self.harness)
            reason = f"capture collection failed: {type(error).__name__}: {error}"
        self._token_capture = _masked(reason)
        await self._abort_capture()

    async def _abort_capture(self) -> None:
        """Abort the capture component, bounded and shielded so a cancelled caller cannot interrupt it."""
        component, config = self._capture_component, self.session_capture
        if component is None or config is None:
            return

        async def abort() -> None:
            try:
                async with asyncio.timeout(config.collect_timeout_seconds):
                    await component.abort(self.sandbox)
            except Exception:
                LOG.exception("Aborting the %s session capture failed", self.harness)

        await asyncio.shield(asyncio.create_task(abort()))

    async def read_output_log(self) -> str:
        """Read combined supervisor/harness diagnostics without masking the original failure."""
        try:
            return await read_text(self.sandbox, path=f"{self.session_dir}/{OUTPUT_LOG_FILE}")
        except Exception:
            LOG.warning("Could not read %s harness output log", self.harness, exc_info=True)
            return ""

    async def _stage(self, stage_activation: Callable[[], Awaitable[SandboxCommand]]) -> SandboxCommand:
        command = await stage_activation()
        await self.sandbox.upload(Path(process_supervisor.__file__), f"{self.session_dir}/{SUPERVISOR_FILE}")
        return command

    async def execute(
        self,
        *,
        stage_activation: Callable[[], Awaitable[SandboxCommand]],
        collect: Callable[[], Awaitable[Artifacts]],
        timeout: float,
        close_timeout: float,
    ) -> Artifacts:
        """Run once, preserving captured artifacts even when execution raises.

        Cancellation stops the remote harness before releasing provider exec.
        Callers that merely stop waiting (such as disconnected HTTP requests)
        must shield their shared activation task.
        """
        if self.closing or self._stage_task is not None:
            raise RuntimeError(f"{self.harness} sandbox session is closing or already activated")
        if self.session_capture is not None and self._capture_endpoint is None:
            raise RuntimeError(f"{self.harness} sandbox session capture is not running; the harness must not run")
        # Register both hooks before yielding, so close can fence preparation too.
        self._collect = collect
        self._stage_task = asyncio.create_task(self._stage(stage_activation))
        try:
            command = await asyncio.shield(self._stage_task)
            if self.closing or self._finalize_task is not None:
                raise RuntimeError(f"{self.harness} sandbox session closed before launch")
            cleanup_timeout, provider_timeout = supervision_timeouts(timeout=timeout, close_timeout=close_timeout)
            launch = supervised_launch_command(
                session_dir=self.session_dir,
                command=command.argv,
                timeout=timeout,
                cleanup_timeout=cleanup_timeout,
                python=command.python,
            )
            self.launch_started = True
            self._exec_task = asyncio.create_task(
                self.sandbox.exec(
                    launch,
                    cwd=self.workdir,
                    timeout_s=provider_timeout,
                )
            )
            result = await asyncio.shield(self._exec_task)
            if result.error_type == "timeout":
                raise TimeoutError(f"{self.harness} sandbox supervisor exceeded its execution deadline")
            if result.error_type:
                raise RuntimeError(f"{self.harness} sandbox execution failed: {result.error_type}: {result.stderr}")
            # A harness process's nonzero exit can still carry useful, adapter-specific output.
        except BaseException:
            try:
                await self._finish(timeout=close_timeout)
            except Exception:
                LOG.exception("%s cleanup remains unconfirmed; close must retry", self.harness)
            raise
        await self._finish(timeout=close_timeout)
        if result.return_code != 0 and self.capture_error is not None:
            raise RuntimeError(
                f"{self.harness} sandbox execution failed (exit {result.return_code}): {result.stderr or ''}"
            ) from self.capture_error
        if self.capture_error is not None:
            raise self.capture_error
        # Collection either returned artifacts (including a valid None) or recorded an error.
        return cast(Artifacts, self.artifacts)

    async def stop_harness(self, *, timeout: float) -> None:
        """Fence a delayed launch or confirm descendant cleanup before transport cancellation."""
        if self.sandbox_stopped or not self.launch_started or self.cleanup is not None:
            return
        try:
            async with asyncio.timeout(timeout):
                self.cleanup = await stop_and_confirm_cleanup(
                    self.sandbox,
                    session_dir=self.session_dir,
                    workdir=self.workdir,
                    timeout=timeout,
                    harness=self.harness,
                )
        except TimeoutError as error:
            raise RuntimeError(f"{self.harness} launch outcome is unknown; cleanup deadline exceeded") from error

    async def _finish(self, *, timeout: float) -> None:
        self._finalize_task = _reuse_or_start(self._finalize_task, lambda: self._finalize(timeout=timeout))
        await asyncio.shield(self._finalize_task)

    async def _cancel_and_wait[T](self, task: asyncio.Future[T], *, timeout: float) -> None:
        if not task.done():
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
        except Exception:
            if not task.done():
                raise
            # Execution/preparation errors are propagated by execute, independently of cleanup.

    async def _finalize(self, *, timeout: float) -> None:
        if self._stage_task is not None:
            await self._cancel_and_wait(self._stage_task, timeout=timeout)
        await self.stop_harness(timeout=timeout)
        if self.launch_started and not self._capture_attempted and self._collect is not None:
            if self.sandbox_stopped:
                self.capture_error = RuntimeError(f"{self.harness} sandbox stopped before artifact capture")
            else:
                try:
                    async with asyncio.timeout(timeout):
                        self.artifacts = await self._collect()
                    self.capture_error = None
                except Exception as error:
                    self.capture_error = error
                    LOG.exception("%s artifact capture failed after confirmed cleanup", self.harness)
            self._capture_attempted = True
        await self._release_exec(timeout=timeout)

    async def _release_exec(self, *, timeout: float) -> None:
        if (self.cleanup is not None or self.sandbox_stopped) and self._exec_task is not None:
            await self._cancel_and_wait(self._exec_task, timeout=timeout)

    async def close(self, *, timeout: float) -> None:
        """Join finalization before release; concurrent/retried closes never duplicate capture."""
        self._close_task = _reuse_or_start(self._close_task, lambda: self._close(timeout=timeout))
        await asyncio.shield(self._close_task)

    async def _close(self, *, timeout: float) -> None:
        try:
            await self._finish(timeout=timeout)
        except Exception:
            if not self.owns_sandbox:
                # The harness may still be calling the capture; a retried close collects once cleanup is confirmed.
                raise
            # Container teardown remains the authority if graceful cleanup is unavailable.
            LOG.exception("%s graceful cleanup failed; stopping owned sandbox", self.harness)
            if not self._capture_attempted:
                self.capture_error = RuntimeError(f"{self.harness} artifact capture unavailable during forced stop")
        # The harness has stopped, or the owned sandbox is about to; the capture must be collected before release.
        await self._collect_capture()
        if self.owns_sandbox:
            if not self.sandbox_stopped:
                async with asyncio.timeout(timeout):
                    await self.sandbox.stop()
                self.sandbox_stopped = True
            await self._release_exec(timeout=timeout)
        else:
            async with asyncio.timeout(timeout):
                await remove_session_directory(
                    self.sandbox,
                    session_dir=self.session_dir,
                    workdir=self.workdir,
                    timeout=timeout,
                    harness=self.harness,
                )
                await self.sandbox.disconnect()


def _masked(reason: str) -> TokenCapture:
    return TokenCapture(masked=True, mask_reason=reason)
