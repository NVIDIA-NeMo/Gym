# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Supervised harness activations over one retained sandbox connection."""

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from time import monotonic
from typing import Literal, TypedDict, cast

from pydantic import TypeAdapter

from nemo_gym.agent_utils.supervisor_client import (
    CONTINUE_REQUEST_FILE,
    DEFAULT_PAUSE_TIMEOUT,
    OUTPUT_LOG_FILE,
    PAUSE_REQUEST_FILE,
    STOP_REQUEST_FILE,
    SUPERVISOR_FILE,
    SUPERVISOR_SCRIPT,
    CleanupReceipt,
    HarnessTerminationError,
    PauseAcknowledgement,
    SupervisorRequestOutcome,
    check_stdlib_runner,
    remove_session_directory,
    request_continue,
    request_pause,
    stop_and_confirm_cleanup,
    supervised_launch_command,
    supervision_timeouts,
)
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.utils import read_text, upload_text


LOG = logging.getLogger(__name__)
# Shared runner uploaded by adapters; reads the bash source written by shell_assignments().
HARNESS_RUNNER_FILE = "harness_runner.sh"
HARNESS_RUNNER_SCRIPT = Path(__file__).with_name(HARNESS_RUNNER_FILE)
# The runner's exit statuses when the CLI never ran: its own failure (bad cwd, setup), and a CLI that
# could not be started. Any other status is the CLI's own.
HARNESS_RUNNER_FAILED = 125
HARNESS_COMMAND_NOT_STARTED = (126, 127)
# Where adapters upload a standard-library Python runner, relative to session_dir. Without a
# .py suffix, a `pkill -f '\.py'` in the task cannot match the runner's command line.
RUNNER_FILE = "sandbox_runner"
_SHELL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _reuse_or_start[T](
    task: asyncio.Task[T] | None, factory: Callable[[], Coroutine[object, object, T]]
) -> asyncio.Task[T]:
    if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
        return asyncio.create_task(factory())
    return task


@dataclass(frozen=True, kw_only=True)
class SandboxCommand:
    """Harness argv, run under the bash process supervisor.

    ``stdlib_runner`` declares a Python runner that needs nothing beyond the standard library.
    Its argv, which must start with the interpreter, then runs under a neutral process name so
    that ``pkill python3`` in the task cannot stop it. The renamed interpreter is the base
    interpreter, outside any virtual environment.
    """

    argv: list[str]
    stdlib_runner: bool = False

    def __post_init__(self) -> None:
        check_stdlib_runner(self.argv, stdlib_runner=self.stdlib_runner)


def shell_assignments(values: Mapping[str, str | Sequence[str] | Mapping[str, str]]) -> str:
    """Bash source assigning each value, for sandbox runners that cannot parse JSON.

    A string becomes a scalar and a sequence an array. A mapping becomes an array of
    ``KEY=VALUE`` words, ready for ``export`` or ``env``. Every word is shell-quoted.
    Words must be strings without NUL, which bash cannot hold.
    """

    def word(item: object) -> str:
        if not isinstance(item, str) or "\0" in item:
            raise ValueError(f"Invalid shell word, which must be a string without NUL: {item!r}")
        return quote(item)

    lines = []
    for name, value in values.items():
        if not _SHELL_NAME.fullmatch(name):
            raise ValueError(f"Invalid shell variable name: {name!r}")
        if isinstance(value, str):
            lines.append(f"{name}={word(value)}")
            continue
        if isinstance(value, Mapping):
            invalid = [key for key in value if not _SHELL_NAME.fullmatch(key)]
            if invalid:
                raise ValueError(f"Invalid environment variable names: {invalid!r}")
            value = [f"{key}={item}" if isinstance(item, str) else item for key, item in value.items()]
        lines.append(f"{name}=({' '.join(word(item) for item in value)})")
    return "".join(f"{line}\n" for line in lines)


@dataclass(frozen=True)
class HarnessPauseResult:
    """The outcome of ``SandboxSession.pause_harness`` or ``continue_harness``.

    ``confirmed``: every process the supervisor reached is paused (or running again), or no launch had started,
    and the launch will honour the request when it starts. ``finished``: the harness had already ended; its cleanup
    receipt answers instead. ``unconfirmed``: the supervisor did not confirm in time or reported ``error``.
    ``pids`` are the processes a pause stopped.
    """

    status: Literal["confirmed", "finished", "unconfirmed"]
    pids: tuple[int, ...] = ()
    error: str | None = None

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"


class SandboxSessionCheckpoint(TypedDict):
    """What ``SandboxSession.reattach`` needs to rebind a session to a sandbox restored from a snapshot."""

    session_root: str
    session_dir: str
    launch_started: bool


_CHECKPOINT_STATE = TypeAdapter(SandboxSessionCheckpoint)


def _pause_result(
    outcome: SupervisorRequestOutcome, acknowledgement: PauseAcknowledgement | None
) -> HarnessPauseResult:
    if outcome == "finished":
        return HarnessPauseResult(status="finished")
    if outcome == "not_launched":
        return HarnessPauseResult(status="confirmed")
    if outcome == "timeout" or acknowledgement is None:
        return HarnessPauseResult(status="unconfirmed", error="the supervisor did not acknowledge in time")
    return HarnessPauseResult(
        status="confirmed" if acknowledgement["confirmed"] else "unconfirmed",
        pids=tuple(acknowledgement["pids"]),
        error=acknowledgement["error"],
    )


@dataclass(kw_only=True)
class SandboxSession[Artifacts]:
    """Execute, stop, capture, then release an owned or borrowed sandbox.

    Runtime installation belongs in the adapter's seed hook. ``stage_activation``
    stages input and returns a command; ``collect`` copies harness artifacts
    to the agent server after cleanup is confirmed. It may run a bounded snapshot
    command, but must not restart the harness. Parsing responses stays in the adapter.

    Each activation stages input and a fresh supervisor, runs the harness process, then
    stops and collects artifacts. Adapters may explicitly rearm it with ``begin_activation``
    after confirmed cleanup, retaining native state outside the per-activation directory.
    The default still permits only one launch. Execution retains the sandbox until close
    releases it. Missing termination evidence remains retryable. A final negative
    receipt skips capture, releases a borrowed connection and leaves teardown to
    Resources; an owned sandbox falls back to provider stop. Closed means release succeeded.

    The supervisor stops the harness process group; a forced stop also reaches the
    harness's descendants in other groups and processes that inherited its
    ``NEMO_GYM_SUPERVISOR`` marker. Processes that had left the harness when the stop
    began, processes in other groups after a natural exit, and a process that left
    between two scans without the marker remain until the sandbox owner destroys the
    sandbox.
    Borrowed sandboxes stay available for Resources verification; owned sandboxes are
    stopped on close. Unconfirmed harness termination prevents capture and verification.

    Each stop, collection and release phase is bounded by ``close_timeout`` (or
    the timeout passed to ``close``). Failed cleanup/release retains the handle
    for retry. Failed capture is recorded separately and does not prevent release.
    HTTP request binding and activation retries belong to the agent session layer.

    A ``pausable`` session's supervisor also pauses and continues the harness in place for a
    checkpoint (``pause_harness`` and ``continue_harness``): every process it reaches, and every
    process carrying its marker, is stopped with SIGSTOP and continued with SIGCONT, and time spent
    paused does not count against the harness timeout. Directories under ``session_root`` survive a
    snapshot of a borrowed sandbox; ``checkpoint_state`` and ``reattach`` rebind a session to the
    sandbox a restore forks from that snapshot.
    """

    sandbox: AsyncSandbox
    session_dir: str
    workdir: str | None
    harness: str
    owns_sandbox: bool = False
    pausable: bool = False
    launch_started: bool = field(default=False, init=False)
    cleanup: CleanupReceipt | None = field(default=None, init=False)
    artifacts: Artifacts | None = field(default=None, init=False)
    capture_error: Exception | None = field(default=None, init=False)
    termination_error: HarnessTerminationError | None = field(default=None, init=False)
    sandbox_stopped: bool = field(default=False, init=False)
    _session_root: str | None = field(default=None, init=False)
    _stage_task: asyncio.Future[SandboxCommand] | None = field(default=None, init=False)
    _exec_task: asyncio.Task[SandboxExecResult] | None = field(default=None, init=False)
    _finalize_task: asyncio.Task[None] | None = field(default=None, init=False)
    _close_task: asyncio.Task[None] | None = field(default=None, init=False)
    _collect: Callable[[], Awaitable[Artifacts]] | None = field(default=None, init=False)
    _capture_attempted: bool = field(default=False, init=False)
    # Sequence numbers of the latest pause and continue requested of the current launch's supervisor,
    # and of the latest continue it confirmed.
    _pause_seq: int = field(default=0, init=False)
    _continue_seq: int = field(default=0, init=False)
    _continued_seq: int = field(default=0, init=False)
    # Holds the provider's polls of the launch's command while the harness is paused, so a sandbox being snapshotted
    # is not asked about it; released before anything else runs a command in the sandbox.
    _parked_polls: ExitStack | None = field(default=None, init=False, repr=False)

    @property
    def session_root(self) -> str:
        """The directory holding every launch directory of this session, removed on close."""
        return self._session_root or self.session_dir

    def checkpoint_state(self) -> SandboxSessionCheckpoint:
        """Where this session's files are, for ``reattach`` after a restore; JSON-serializable."""
        return {
            "session_root": self.session_root,
            "session_dir": self.session_dir,
            "launch_started": self.launch_started,
        }

    @classmethod
    def reattach(
        cls,
        sandbox: AsyncSandbox,
        state: SandboxSessionCheckpoint,
        *,
        workdir: str | None,
        harness: str,
        owns_sandbox: bool = False,
        pausable: bool = False,
    ) -> "SandboxSession[Artifacts]":
        """Rebind a session from ``checkpoint_state`` to ``sandbox``, restored from a snapshot of the original.

        A snapshot keeps files, not processes: no launch recorded in ``state`` still runs in ``sandbox``, so the
        reattached session has none. Rearm it with ``begin_activation`` and a fresh launch directory under
        ``session_root`` before the next ``execute``; ``state["launch_started"]`` says whether the harness had
        already started, for an adapter that resumes the harness's saved conversation.
        """
        state = _CHECKPOINT_STATE.validate_python(state, strict=True)
        session = cls(
            sandbox=sandbox,
            session_dir=state["session_dir"],
            workdir=workdir,
            harness=harness,
            owns_sandbox=owns_sandbox,
            pausable=pausable,
        )
        if state["session_root"] != state["session_dir"]:
            session._session_root = state["session_root"]
        return session

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
    def cleanup_confirmed(self) -> bool:
        """Whether no launch occurred, termination was witnessed, or owned sandbox stopped."""
        return self.sandbox_stopped or (
            self.termination_error is None
            and (not self.launch_started or bool(self.cleanup and self.cleanup["cleanup_confirmed"]))
        )

    @property
    def stop_request_path(self) -> str:
        """Marker path an adapter may pass to its harness for cancellation diagnostics."""
        return f"{self.session_dir}/{STOP_REQUEST_FILE}"

    async def read_output_log(self) -> str:
        """Read combined supervisor/harness diagnostics without masking the original failure."""
        try:
            return await read_text(self.sandbox, path=f"{self.session_dir}/{OUTPUT_LOG_FILE}")
        except Exception:
            LOG.warning("Could not read %s harness output log", self.harness, exc_info=True)
            return ""

    async def _stage(self, stage_activation: Callable[[], Awaitable[SandboxCommand]]) -> SandboxCommand:
        command = await stage_activation()
        await self.sandbox.upload(SUPERVISOR_SCRIPT, f"{self.session_dir}/{SUPERVISOR_FILE}")
        if self.pausable:
            # A pause requested before the launch directory existed reaches the supervisor here, and holds the launch.
            for name, seq in ((PAUSE_REQUEST_FILE, self._pause_seq), (CONTINUE_REQUEST_FILE, self._continue_seq)):
                if seq:
                    await upload_text(self.sandbox, path=f"{self.session_dir}/{name}", text=str(seq))
        return command

    async def execute(
        self,
        *,
        stage_activation: Callable[[], Awaitable[SandboxCommand]],
        collect: Callable[[], Awaitable[Artifacts]],
        timeout: float,
        close_timeout: float,
        deadline: float | None = None,
        pause_allowance_s: float = 0.0,
    ) -> Artifacts:
        """Run once, preserving captured artifacts even when execution raises.

        Cancellation stops the remote harness before releasing provider exec.
        Callers that merely stop waiting (such as disconnected HTTP requests)
        must shield their shared activation task.

        ``pause_allowance_s`` extends the provider's execution deadline by the time the harness may
        spend paused for checkpoints, which the supervisor does not count against ``timeout``. A
        longer pause than the allowance fails the execution, so pause only within it, or stop instead.
        """
        if self.closing or self._stage_task is not None:
            raise RuntimeError(f"{self.harness} sandbox session is closing or already activated")
        # Register both hooks before yielding, so close can fence preparation too.
        self._collect = collect
        self._stage_task = asyncio.create_task(self._stage(stage_activation))
        try:
            async with asyncio.timeout_at(deadline):
                command = await asyncio.shield(self._stage_task)
            if self.closing or self._finalize_task is not None:
                raise RuntimeError(f"{self.harness} sandbox session closed before launch")
            if deadline is not None:
                timeout = min(timeout, deadline - monotonic())
            # The supervisor counts whole centiseconds; less than one cannot run the harness.
            if timeout < 0.01:
                raise TimeoutError(f"{self.harness} activation deadline expired before launch")
            cleanup_timeout, provider_timeout = supervision_timeouts(
                timeout=timeout, close_timeout=close_timeout, pause_allowance=pause_allowance_s
            )
            launch = supervised_launch_command(
                session_dir=self.session_dir,
                command=command.argv,
                timeout=timeout,
                cleanup_timeout=cleanup_timeout,
                stdlib_runner=command.stdlib_runner,
                pausable=self.pausable,
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
            await self._finish(timeout=close_timeout)
        except BaseException:
            if self._finalize_task is None or not self._finalize_task.done():
                try:
                    await self._finish(timeout=close_timeout)
                except Exception:
                    LOG.exception(
                        "%s harness termination was not confirmed; close must release the session", self.harness
                    )
            raise
        if result.return_code != 0 and self.capture_error is not None:
            raise RuntimeError(
                f"{self.harness} sandbox execution failed (exit {result.return_code}): {result.stderr or ''}"
            ) from self.capture_error
        if self.capture_error is not None:
            raise self.capture_error
        # Collection either returned artifacts (including a valid None) or recorded an error.
        return cast(Artifacts, self.artifacts)

    async def begin_activation(self, *, session_dir: str, timeout: float) -> None:
        """Rearm a stopped harness using fresh supervisor files in the same sandbox.

        The adapter retains its native conversation elsewhere in the sandbox. A
        previous launch must be reaped before any activation fields can be reset.
        Close fences all later activations, including when release needs retry.
        """
        if self.closing:
            raise RuntimeError(f"{self.harness} sandbox session is closing")
        if self._stage_task is not None:
            if session_dir == self.session_dir:
                raise ValueError("Repeated activations require a fresh supervisor directory")
            await self._finish(timeout=timeout)
            if self.closing:
                raise RuntimeError(f"{self.harness} sandbox session is closing")
            if not self.cleanup_confirmed:
                raise RuntimeError(f"{self.harness} previous activation cleanup is unconfirmed")
        if self._session_root is None:
            self._session_root = self.session_dir
        self.session_dir = session_dir
        self.launch_started = False
        self.cleanup = None
        self.artifacts = None
        self.capture_error = None
        self._stage_task = None
        self._exec_task = None
        self._finalize_task = None
        self._collect = None
        self._capture_attempted = False
        # The next launch has a fresh supervisor: nothing paused carries over.
        self._pause_seq = self._continue_seq = self._continued_seq = 0
        self._release_parked_polls()

    async def pause_harness(self, *, timeout: float = DEFAULT_PAUSE_TIMEOUT) -> HarnessPauseResult:
        """Pause the harness in place for a checkpoint, without ending it; reads an existing receipt first.

        ``finished`` when the harness had already ended. Before its launch, the pause is recorded and holds the
        launch until ``continue_harness``. Pausing again before a continue requests the same pause again.
        """
        if not self.pausable:
            raise RuntimeError(f"{self.harness} sandbox session was not launched pausable")
        if self.cleanup is not None or self.sandbox_stopped or self.closing:
            return HarnessPauseResult(status="finished")
        if self._pause_seq <= self._continued_seq:
            self._pause_seq = self._continued_seq + 1
        outcome, acknowledgement = await request_pause(
            self.sandbox, session_dir=self.session_dir, workdir=self.workdir, seq=self._pause_seq, timeout=timeout
        )
        result = _pause_result(outcome, acknowledgement)
        if result.confirmed and self.launch_started and self._parked_polls is None:
            # A no-op scope on providers that do not poll command status.
            self._parked_polls = ExitStack()
            self._parked_polls.enter_context(self._polling_scope("parked_exec_polling"))
        return result

    def _polling_scope(self, name: str) -> AbstractContextManager[None]:
        """The sandbox's polling scope ``name``; a sandbox double without one gets a no-op scope."""
        scope = getattr(self.sandbox, name, None)
        return scope() if scope is not None else nullcontext()

    def _release_parked_polls(self) -> None:
        if self._parked_polls is not None:
            parked, self._parked_polls = self._parked_polls, None
            parked.close()

    async def continue_harness(self, *, timeout: float = DEFAULT_PAUSE_TIMEOUT) -> HarnessPauseResult:
        """Continue a harness ``pause_harness`` paused; ``confirmed`` at once when nothing is paused."""
        if self.cleanup is not None or self.sandbox_stopped or self.closing:
            return HarnessPauseResult(status="finished")
        if self._continued_seq >= self._pause_seq:
            return HarnessPauseResult(status="confirmed")
        self._continue_seq = self._pause_seq
        # The continue request is a command in the sandbox itself: the launch's polls are asked again first.
        self._release_parked_polls()
        outcome, acknowledgement = await request_continue(
            self.sandbox, session_dir=self.session_dir, workdir=self.workdir, seq=self._continue_seq, timeout=timeout
        )
        result = _pause_result(outcome, acknowledgement)
        if result.confirmed:
            self._continued_seq = self._continue_seq
        return result

    async def stop_harness(self, *, timeout: float) -> None:
        """Confirm harness termination before capture and transport release."""
        self._release_parked_polls()
        if self.sandbox_stopped or not self.launch_started or self.cleanup is not None:
            return
        if self.pausable:
            # Stopping for a checkpoint: the provider notices the launch's end promptly.
            with self._polling_scope("checkpoint_exec_polling"):
                await self._stop_harness(timeout=timeout)
            return
        await self._stop_harness(timeout=timeout)

    async def _stop_harness(self, *, timeout: float) -> None:
        if self.termination_error is not None:
            raise self.termination_error
        try:
            async with asyncio.timeout(timeout):
                self.cleanup = await stop_and_confirm_cleanup(
                    self.sandbox,
                    session_dir=self.session_dir,
                    workdir=self.workdir,
                    timeout=timeout,
                    harness=self.harness,
                )
        except HarnessTerminationError as error:
            self.termination_error = error
            raise
        except TimeoutError as error:
            raise RuntimeError(f"{self.harness} launch outcome is unknown; termination deadline exceeded") from error

    async def _finish(self, *, timeout: float) -> None:
        """Join process finalization without releasing the sandbox connection."""
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
                    LOG.exception("%s artifact capture failed after confirmed harness termination", self.harness)
            self._capture_attempted = True
        await self._release_exec(timeout=timeout)

    async def _release_exec(self, *, timeout: float) -> None:
        if (
            self.cleanup is not None or self.termination_error is not None or self.sandbox_stopped
        ) and self._exec_task is not None:
            await self._cancel_and_wait(self._exec_task, timeout=timeout)

    async def close(self, *, timeout: float) -> None:
        """Join finalization before release; concurrent/retried closes never duplicate capture."""
        self._close_task = _reuse_or_start(self._close_task, lambda: self._close(timeout=timeout))
        await asyncio.shield(self._close_task)

    async def _close(self, *, timeout: float) -> None:
        self._release_parked_polls()
        try:
            await self._finish(timeout=timeout)
        except Exception as error:
            if not self.owns_sandbox and not isinstance(error, HarnessTerminationError):
                raise
            if self.owns_sandbox:
                LOG.exception("%s session finalization failed; stopping owned sandbox", self.harness)
            else:
                # A negative receipt is final: retrying cannot confirm termination.
                # Release the borrowed client so Resources can tear down the sandbox.
                LOG.exception(
                    "%s harness termination unconfirmed; releasing borrowed session for Resources teardown",
                    self.harness,
                )
            if not self._capture_attempted:
                self.capture_error = error
        if self.owns_sandbox:
            if not self.sandbox_stopped:
                async with asyncio.timeout(timeout):
                    await self.sandbox.stop()
                self.sandbox_stopped = True
            await self._release_exec(timeout=timeout)
        else:
            await self._release_exec(timeout=timeout)
            async with asyncio.timeout(timeout):
                await remove_session_directory(
                    self.sandbox,
                    session_dir=self._session_root or self.session_dir,
                    workdir=self.workdir,
                    timeout=timeout,
                    harness=self.harness,
                )
                await self.sandbox.disconnect()
