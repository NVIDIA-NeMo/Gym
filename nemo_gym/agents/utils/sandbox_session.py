# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Supervised harness activations over one retained sandbox connection."""

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from time import monotonic
from typing import Literal, NotRequired, TypedDict, cast

from pydantic import TypeAdapter

from nemo_gym.agents.utils.supervisor_client import (
    CLEANUP_RECEIPT_FILE,
    CONTINUE_REQUEST_FILE,
    DEFAULT_PAUSE_TIMEOUT,
    EXPIRE_REQUEST_FILE,
    OUTPUT_LOG_FILE,
    PAUSE_REQUEST_FILE,
    STOP_REQUEST_FILE,
    SUPERVISOR_FILE,
    SUPERVISOR_SCRIPT,
    CleanupReceipt,
    PauseAcknowledgement,
    PauseMode,
    SupervisorRequestOutcome,
    is_harness_termination_failure,
    remove_session_directory,
    request_continue,
    request_pause,
    stop_and_confirm_cleanup,
    supervised_launch_command,
    supervision_timeouts,
)
from nemo_gym.core.failure_kinds import PROVIDER_OOM_KILLED
from nemo_gym.core.typed_exceptions import NgFailureEvent
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
# Consecutive failed polls for an adopted launch's receipt before the wait gives up.
_RECEIPT_POLL_FAILURES = 5
_SHELL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _reuse_or_start[T](
    task: asyncio.Task[T] | None, factory: Callable[[], Coroutine[object, object, T]]
) -> asyncio.Task[T]:
    if task is None or (task.done() and (task.cancelled() or task.exception() is not None)):
        return asyncio.create_task(factory())
    return task


@dataclass(frozen=True, kw_only=True)
class SandboxCommand:
    """Harness argv, run under the bash process supervisor."""

    argv: list[str]


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


class NothingCapturedError(RuntimeError):
    """An adapter's capture found nothing because the launch ended before the harness wrote any; logged in one line."""


class ExpiryUnsupportedError(Exception):
    """``expire`` cannot stop this running harness: its sandbox's uploads can wait for capacity the harness holds."""


@dataclass(frozen=True)
class HarnessPauseResult:
    """The outcome of ``SandboxSession.pause_harness`` or ``continue_harness``.

    ``confirmed``: every process the supervisor reached is paused (or running again), or no launch had started,
    and the launch will honour the request when it starts. ``finished``: the harness had already ended; its cleanup
    receipt answers instead. ``unconfirmed``: the supervisor did not confirm in time or reported ``error``.
    ``pids`` are the processes a pause stopped. ``identity`` is the answering supervisor's ``PID.START`` and
    ``mode`` its pause mode, when it reported them.
    """

    status: Literal["confirmed", "finished", "unconfirmed"]
    pids: tuple[int, ...] = ()
    error: str | None = None
    identity: str | None = None
    mode: PauseMode | None = None

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"


class SandboxSessionCheckpoint(TypedDict):
    """What ``SandboxSession.reattach`` needs to rebind a session to a sandbox restored from a snapshot.

    ``pause_seq`` and ``supervisor_identity`` are there while the launch's supervisor holds a confirmed pause: on a
    sandbox whose snapshot keeps its processes, ``adopt`` can take that launch over after the restore.
    """

    session_root: str
    session_dir: str
    launch_started: bool
    pause_seq: NotRequired[int]
    supervisor_identity: NotRequired[str]


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
        identity=acknowledgement.get("identity"),
        mode=acknowledgement.get("mode"),
    )


@dataclass(kw_only=True)
class SandboxSession[Artifacts]:
    """Execute, stop, capture, then release an owned or borrowed sandbox.

    Runtime installation belongs in the adapter's seed hook. ``stage_activation``
    stages input and returns a command. Every sandbox write an activation needs
    belongs there: close cancels staging before it releases the sandbox, but it
    cannot stop a write made before ``execute``. ``collect`` copies harness artifacts
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
    checkpoint (``pause_harness`` and ``continue_harness``), and time spent paused does not count
    against the harness timeout. In ``pause_mode`` ``stop``, every process it reaches, and every
    process carrying its marker, is stopped with SIGSTOP and continued with SIGCONT. In ``hold``, for
    a sandbox whose snapshot keeps its processes, nothing is signalled: the harness keeps running
    while the checkpoint holds its model calls. Directories under ``session_root`` survive a snapshot
    of a borrowed sandbox; ``checkpoint_state`` and ``reattach`` rebind a session to the sandbox a
    restore forks from that snapshot, and when the snapshot also kept the supervisor holding a pause,
    ``adopt`` takes its launch over.
    """

    sandbox: AsyncSandbox
    session_dir: str
    workdir: str | None
    harness: str
    owns_sandbox: bool = False
    pausable: bool = False
    pause_mode: PauseMode = "stop"
    launch_started: bool = field(default=False, init=False)
    cleanup: CleanupReceipt | None = field(default=None, init=False)
    artifacts: Artifacts | None = field(default=None, init=False)
    capture_error: Exception | None = field(default=None, init=False)
    termination_error: NgFailureEvent | None = field(default=None, init=False)
    sandbox_stopped: bool = field(default=False, init=False)
    execution_oom: NgFailureEvent | None = field(default=None, init=False)
    _session_root: str | None = field(default=None, init=False)
    _stage_task: asyncio.Future[SandboxCommand] | None = field(default=None, init=False)
    _exec_task: asyncio.Task[SandboxExecResult] | None = field(default=None, init=False)
    _finalize_task: asyncio.Task[None] | None = field(default=None, init=False)
    _close_task: asyncio.Task[None] | None = field(default=None, init=False)
    _collect: Callable[[], Awaitable[Artifacts]] | None = field(default=None, init=False)
    _capture_attempted: bool = field(default=False, init=False)
    _expire_requested: bool = field(default=False, init=False)
    # The caller's reason for the expiry ``expire`` requested, if it gave one.
    expire_reason: str | None = field(default=None, init=False)
    # An expiry whose marker write failed, possibly after the marker reached the sandbox, and its reason.
    _unconfirmed_expiry: bool = field(default=False, init=False)
    _unconfirmed_expire_reason: str | None = field(default=None, init=False)
    _expire_write: asyncio.Task[None] | None = field(default=None, init=False)
    # Sequence numbers of the latest pause and continue requested of the current launch's supervisor,
    # and of the latest continue it confirmed.
    _pause_seq: int = field(default=0, init=False)
    _continue_seq: int = field(default=0, init=False)
    _continued_seq: int = field(default=0, init=False)
    # Serializes pause and continue requests: a pause asked for while ``adopt`` continues a restored pause waits for
    # that continue, so it takes the next number instead of reading the restored pause's acknowledgement.
    _control: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    # The supervisor that confirmed pause ``_held_seq``, as ``PID.START``.
    _held_identity: str | None = field(default=None, init=False)
    _held_seq: int = field(default=0, init=False)
    # Holds the provider's polls of the launch's command while the harness is paused, so a sandbox being snapshotted
    # is not asked about it; released before anything else runs a command in the sandbox.
    _parked_polls: ExitStack | None = field(default=None, init=False, repr=False)

    @property
    def session_root(self) -> str:
        """The directory holding every launch directory of this session, removed on close."""
        return self._session_root or self.session_dir

    def checkpoint_state(self) -> SandboxSessionCheckpoint:
        """Where this session's files are, for ``reattach`` after a restore; JSON-serializable."""
        state: SandboxSessionCheckpoint = {
            "session_root": self.session_root,
            "session_dir": self.session_dir,
            "launch_started": self.launch_started,
        }
        if self.launch_started and self._held_identity is not None and self._continued_seq < self._held_seq:
            state["pause_seq"] = self._held_seq
            state["supervisor_identity"] = self._held_identity
        return state

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
        pause_mode: PauseMode = "stop",
    ) -> "SandboxSession[Artifacts]":
        """Rebind a session from ``checkpoint_state`` to ``sandbox``, restored from a snapshot of the original.

        The reattached session has no launch of its own. What a snapshot kept depends on the provider. When it
        kept files only, no launch recorded in ``state`` still runs in ``sandbox``: rearm the session with
        ``begin_activation`` and a fresh launch directory under ``session_root`` before the next ``execute``;
        ``state["launch_started"]`` says whether the harness had already started, for an adapter that resumes
        the harness's saved conversation. When it kept the processes too, a launch whose supervisor held a
        confirmed pause (``state["pause_seq"]``) may still run: check it with ``check_restored``, then take it
        over with ``adopt``, or end it with ``abandon_launch`` before a relaunch.
        """
        state = _CHECKPOINT_STATE.validate_python(state, strict=True)
        session = cls(
            sandbox=sandbox,
            session_dir=state["session_dir"],
            workdir=workdir,
            harness=harness,
            owns_sandbox=owns_sandbox,
            pausable=pausable,
            pause_mode=pause_mode,
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

    @property
    def expired(self) -> bool:
        """Whether ``expire`` ended this activation; its receipt then records the timeout."""
        return self._expire_requested and bool(self.cleanup and self.cleanup["timed_out"])

    @property
    def expired_before_launch(self) -> bool:
        """Whether ``expire`` kept the harness from starting, so it left no output to collect."""
        return self.expired and self.cleanup["return_code"] is None and self.cleanup["error"] is None

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

        async def launch() -> asyncio.Task[SandboxExecResult]:
            async with asyncio.timeout_at(deadline):
                command = await asyncio.shield(self._stage_task)
            if self._expire_requested:
                await self._request_expiry(timeout=close_timeout)
            if self.closing or self._finalize_task is not None:
                raise RuntimeError(f"{self.harness} sandbox session closed before launch")
            remaining = timeout if deadline is None else min(timeout, deadline - monotonic())
            # The supervisor counts whole centiseconds; less than one cannot run the harness.
            if remaining < 0.01:
                raise TimeoutError(f"{self.harness} activation deadline expired before launch")
            cleanup_timeout, provider_timeout = supervision_timeouts(
                timeout=remaining, close_timeout=close_timeout, pause_allowance=pause_allowance_s
            )
            command_line = supervised_launch_command(
                session_dir=self.session_dir,
                command=command.argv,
                timeout=remaining,
                cleanup_timeout=cleanup_timeout,
                pausable=self.pausable,
                pause_mode=self.pause_mode,
            )
            self.launch_started = True
            return asyncio.create_task(self.sandbox.exec(command_line, cwd=self.workdir, timeout_s=provider_timeout))

        return await self._run_launch(launch, close_timeout=close_timeout)

    async def adopt(
        self,
        *,
        pause_seq: int,
        collect: Callable[[], Awaitable[Artifacts]],
        timeout: float,
        close_timeout: float,
        pause_allowance_s: float = 0.0,
        poll_window_s: float = 50.0,
    ) -> Artifacts:
        """Take over a launch that ran on through a restore, as ``execute`` would have finished it.

        After ``reattach`` to a sandbox whose snapshot kept its processes, and a ``check_restored`` that found the
        launch's supervisor holding pause ``pause_seq``: continue that pause, then wait for the supervisor's
        cleanup receipt and collect, as ``execute`` does. The process that started the launch is gone with the
        crashed agent server, so the wait polls for the receipt in commands of at most ``poll_window_s`` seconds;
        a command left waiting longer could outlive a provider's stream limit. ``timeout`` bounds the wait like
        the launch's own, plus the supervisor's cleanup phases and ``pause_allowance_s``. Later checkpoints pause
        the same supervisor, from ``pause_seq + 1``.
        """
        if not self.pausable:
            raise RuntimeError(f"{self.harness} sandbox session was not launched pausable")
        if self.closing or self._stage_task is not None:
            raise RuntimeError(f"{self.harness} sandbox session is closing or already activated")
        self._collect = collect
        # Nothing to stage: the launch is already running.
        staged: asyncio.Future[SandboxCommand] = asyncio.get_running_loop().create_future()
        staged.set_result(SandboxCommand(argv=[]))
        self._stage_task = staged

        async def launch() -> asyncio.Task[SandboxExecResult]:
            self.launch_started = True
            self._pause_seq = pause_seq
            self._continue_seq = self._continued_seq = pause_seq - 1
            continued = await self.continue_harness()
            if continued.status == "unconfirmed":
                raise RuntimeError(f"{self.harness} restored launch did not continue: {continued.error}")
            _, provider_timeout = supervision_timeouts(
                timeout=timeout, close_timeout=close_timeout, pause_allowance=pause_allowance_s
            )
            return asyncio.create_task(
                self._wait_for_receipt(deadline=monotonic() + provider_timeout, window_s=poll_window_s)
            )

        return await self._run_launch(launch, close_timeout=close_timeout)

    async def _wait_for_receipt(self, *, deadline: float, window_s: float) -> SandboxExecResult:
        """Poll for the launch's cleanup receipt until ``deadline``; its harness status is the result's code."""
        receipt = quote(f"{self.session_dir}/{CLEANUP_RECEIPT_FILE}")
        failures = 0
        while (window := min(window_s, deadline - monotonic())) > 0:
            result = await self.sandbox.exec(
                f"for _ in $(seq 1 {max(1, int(window))}); do "
                f"[ -f {receipt} ] && {{ cat {receipt}; exit 0; }}; sleep 1; done; exit 3",
                cwd=self.workdir,
                timeout_s=window + 10,
            )
            if result.return_code == 0 and not result.error_type:
                try:
                    code = json.loads(result.stdout or "{}").get("return_code")
                except (ValueError, AttributeError):
                    code = None
                return SandboxExecResult(stdout="", stderr=None, return_code=code if type(code) is int else 0)
            if result.return_code != 3 or result.error_type:
                failures += 1
                if failures >= _RECEIPT_POLL_FAILURES:
                    return SandboxExecResult(
                        stdout=None, stderr=result.stderr, return_code=-1, error_type=result.error_type or "failed"
                    )
                LOG.warning("%s receipt poll failed (%s); retrying", self.harness, result.error_type or result.stderr)
                await asyncio.sleep(1)
            else:
                failures = 0
        return SandboxExecResult(stdout=None, stderr=None, return_code=-1, error_type="timeout")

    async def _run_launch(
        self, launch: Callable[[], Awaitable[asyncio.Task[SandboxExecResult]]], *, close_timeout: float
    ) -> Artifacts:
        """Wait for the launch ``launch`` starts, then finish it: stop, collect, release; artifacts survive errors."""
        try:
            self._exec_task = await launch()
            try:
                result = await asyncio.shield(self._exec_task)
            except NgFailureEvent as error:
                # The provider confirmed that the whole sandbox died during the
                # supervised harness command. In-sandbox cleanup is no longer possible.
                if error.boundary == "sandbox" and error.failure_kind == PROVIDER_OOM_KILLED:
                    self.execution_oom = error
                raise
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
            # An expiry applies to the activation it was requested for; one requested before the first launch stays.
            self._expire_requested = False
            self.expire_reason = None
            self._unconfirmed_expiry = False
            self._expire_write = None
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
        self._pause_seq = self._continue_seq = self._continued_seq = self._held_seq = 0
        self._held_identity = None
        self._release_parked_polls()

    async def pause_harness(self, *, timeout: float = DEFAULT_PAUSE_TIMEOUT) -> HarnessPauseResult:
        """Pause the harness in place for a checkpoint, without ending it; reads an existing receipt first.

        ``finished`` when the harness had already ended. Before its launch, the pause is recorded and holds the
        launch until ``continue_harness``. Pausing again before a continue requests the same pause again.
        """
        if not self.pausable:
            raise RuntimeError(f"{self.harness} sandbox session was not launched pausable")
        async with self._control:
            return await self._pause(timeout=timeout)

    async def _pause(self, *, timeout: float) -> HarnessPauseResult:
        if self.cleanup is not None or self.sandbox_stopped or self.closing:
            return HarnessPauseResult(status="finished")
        if self._pause_seq <= self._continued_seq:
            self._pause_seq = self._continued_seq + 1
        outcome, acknowledgement = await request_pause(
            self.sandbox, session_dir=self.session_dir, workdir=self.workdir, seq=self._pause_seq, timeout=timeout
        )
        result = _pause_result(outcome, acknowledgement)
        if result.confirmed and result.identity is not None:
            self._held_seq, self._held_identity = self._pause_seq, result.identity
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
        async with self._control:
            return await self._continue(timeout=timeout)

    async def _continue(self, *, timeout: float) -> HarnessPauseResult:
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

    async def expire(self, *, timeout: float, reason: str | None = None) -> bool:
        """End the harness as at its time limit, so its receipt records ``timed_out``.

        Unlike ``stop_harness`` (a cancellation), the activation is then graded like a timeout.
        A request made before launch ends the harness as it would launch. Returns False when the
        activation already finished, the session is closing, or expiry was already requested.

        Stopping a running harness takes an upload, which on a provider without
        ``SupportsSandboxIndependentUploads`` can wait behind the very commands it must stop, so there it
        raises ``ExpiryUnsupportedError``. A request before launch works on every provider.
        """
        if self.closing or self.sandbox_stopped or self._expire_requested or self.cleanup is not None:
            return False
        if self._exec_task is not None and self._exec_task.done():
            return False
        if self.launch_started and not self.sandbox.uploads_independent_of_execs:
            raise ExpiryUnsupportedError(
                f"the {self.harness} sandbox's provider cannot stop a running harness: its file uploads can wait "
                "for the capacity running commands hold"
            )
        self._expire_requested = True
        self.expire_reason = reason
        if self.launch_started:
            # A task, so that finalization can wait for a write still in flight before it checks the marker.
            self._expire_write = asyncio.create_task(self._write_expiry(timeout=timeout, reason=reason))
            # Shielded, so a caller that stops waiting (a disconnected stop request) cannot cancel the write before
            # it starts and leave the stop requested but never delivered. The write is bounded by ``timeout`` and,
            # if it fails, makes the stop retryable again.
            await asyncio.shield(self._expire_write)
        return True

    async def _write_expiry(self, *, timeout: float, reason: str | None) -> None:
        try:
            await self._request_expiry(timeout=timeout)
        except BaseException:
            self._expire_requested = False
            self.expire_reason = None
            # The marker may have landed before the write failed; finalization keeps the stop only if it did.
            self._unconfirmed_expiry, self._unconfirmed_expire_reason = True, reason
            raise

    async def _request_expiry(self, *, timeout: float) -> None:
        """Write the marker the supervisor watches, within ``timeout`` including any wait for the provider.

        A file write, not an exec: a provider that bounds its concurrent execs (Local) counts the harness's
        own launch against that bound for as long as it runs, and a stop must not wait for the harness to end.
        """
        try:
            async with asyncio.timeout(timeout):
                await upload_text(self.sandbox, path=f"{self.session_dir}/{EXPIRE_REQUEST_FILE}", text="")
        except TimeoutError as error:
            raise RuntimeError(f"Could not request {self.harness} expiry within {timeout:g} s") from error
        except Exception as error:
            raise RuntimeError(f"Could not request {self.harness} expiry: {error}") from error

    async def stop_harness(self, *, timeout: float) -> None:
        """Confirm harness termination before capture and transport release."""
        self._release_parked_polls()
        if (
            self.sandbox_stopped
            or self.execution_oom is not None
            or not self.launch_started
            or self.cleanup is not None
        ):
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
        except TimeoutError as error:
            raise RuntimeError(f"{self.harness} launch outcome is unknown; termination deadline exceeded") from error
        except NgFailureEvent as error:
            if is_harness_termination_failure(error):
                self.termination_error = error
            raise

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
        await self._confirm_unacknowledged_expiry(timeout=timeout)
        if self.launch_started and not self._capture_attempted and self._collect is not None:
            if self.sandbox_stopped or self.execution_oom is not None:
                self.capture_error = RuntimeError(f"{self.harness} sandbox stopped before artifact capture")
            else:
                try:
                    async with asyncio.timeout(timeout):
                        self.artifacts = await self._collect()
                    self.capture_error = None
                except NothingCapturedError as error:
                    self.capture_error = error
                    LOG.warning("%s captured nothing: %s", self.harness, error)
                except Exception as error:
                    self.capture_error = error
                    LOG.exception("%s artifact capture failed after confirmed harness termination", self.harness)
            self._capture_attempted = True
        await self._release_exec(timeout=timeout)

    async def _confirm_unacknowledged_expiry(self, *, timeout: float) -> None:
        """Keep a stop whose marker write failed only if the marker is in place and the receipt timed out.

        A write still in flight is awaited first, so a failure that arrives during finalization is checked too.
        The whole check, including any wait for provider capacity, is bounded by ``timeout``; without a
        confirmation the stop stays unattributed and finalization goes on. That includes a write still in flight
        when the bound runs out: the harness may have ended at its own deadline, before any marker arrived.
        """
        try:
            async with asyncio.timeout(timeout):
                if self._expire_write is not None and not self._expire_write.done():
                    await asyncio.wait({self._expire_write})
                if (
                    not self._unconfirmed_expiry
                    or self._expire_requested
                    or not (self.cleanup and self.cleanup["timed_out"])
                ):
                    return
                self._unconfirmed_expiry = False
                marker = quote(f"{self.session_dir}/{EXPIRE_REQUEST_FILE}")
                probe = await self.sandbox.exec(
                    f"[ -e {marker} ] && echo present", cwd=self.workdir, timeout_s=timeout
                )
        except Exception:
            if self._expire_write is not None and not self._expire_write.done():
                self._expire_requested, self.expire_reason = False, None
            LOG.warning(
                "Could not confirm the %s expiry marker; the stop stays unattributed", self.harness, exc_info=True
            )
            return
        if probe.return_code == 0 and not probe.error_type and probe.stdout.strip() == "present":
            self._expire_requested, self.expire_reason = True, self._unconfirmed_expire_reason

    async def _release_exec(self, *, timeout: float) -> None:
        if (
            self.cleanup is not None
            or self.termination_error is not None
            or self.sandbox_stopped
            or self.execution_oom is not None
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
            if not self.owns_sandbox and not is_harness_termination_failure(error):
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
                try:
                    if self.cleanup_confirmed and self.execution_oom is None:
                        # A local sandbox stops its workspace, but session runtimes can
                        # live outside it. Reclaim them after capture while exec works.
                        async with asyncio.timeout(timeout):
                            await remove_session_directory(
                                self.sandbox,
                                session_dir=self._session_root or self.session_dir,
                                workdir=self.workdir,
                                timeout=timeout,
                                harness=self.harness,
                            )
                except Exception:
                    # Once stop succeeds, there is no live sandbox in which to retry
                    # removal. Preserve the release result and report any leftovers.
                    LOG.warning(
                        "Could not remove %s session files; stopping owned sandbox", self.harness, exc_info=True
                    )
                finally:
                    async with asyncio.timeout(timeout):
                        await self.sandbox.stop()
                    self.sandbox_stopped = True
            await self._release_exec(timeout=timeout)
        else:
            await self._release_exec(timeout=timeout)
            async with asyncio.timeout(timeout):
                if self.execution_oom is None:
                    await remove_session_directory(
                        self.sandbox,
                        session_dir=self._session_root or self.session_dir,
                        workdir=self.workdir,
                        timeout=timeout,
                        harness=self.harness,
                    )
                await self.sandbox.disconnect()
