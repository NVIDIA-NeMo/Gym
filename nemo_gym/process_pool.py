# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A warm pool of worker processes for CPU-heavy verification.

Resources servers run one asyncio event loop per process. CPU-bound work inside an
``async def verify()`` stalls every other request on that loop, and offloading it to a
thread only helps while the GIL is released, which pure-Python verifiers rarely do. The
servers that already escape to a child process each hand-roll the lifecycle: fork per
request, poll ``is_alive()``, terminate, kill, join, close the pipe.

:class:`WarmProcessPool` centralizes that lifecycle behind one asynchronous call::

    pool = WarmProcessPool(WarmProcessPoolConfig(num_workers=8), initializer=_build_verifier)
    await pool.start()
    reward = await pool.run(_verify_task, expected, generated, timeout=10.0)
    await pool.aclose()

Workers are ``num_workers`` long-lived children started with the ``spawn`` (or ``forkserver``)
start method. ``fork`` is not offered: a server process owns an event loop, thread pools and an
aiohttp client, none of which survive a fork intact. Each child runs ``initializer(*initargs)``
once, reports readiness, then serves tasks one at a time.

Submissions wait in an ``asyncio.Queue`` of at most ``max_pending`` tasks. A full queue makes
:meth:`WarmProcessPool.run` wait, or raise :class:`PoolSaturatedError` once ``queue_timeout``
elapses. Live children never exceed ``num_workers`` plus one per slot that is mid-replacement.

``timeout`` is measured from dispatch to a worker, not from submission, so queue wait and child
execution are reported separately. On expiry the worker is killed and replaced and the caller gets
:class:`TaskTimeoutError`. Cancelling the awaiting task before dispatch drops the submission.
Cancelling it in flight kills and replaces the worker, so a cancelled task never keeps a slot busy.

A worker that times out, crashes, or is cancelled mid-task is killed (``SIGTERM``, then ``SIGKILL``
after ``kill_grace_seconds``), joined, and replaced. Each worker owns its own pipe, so killing one
cannot corrupt another worker's in-flight result. With ``max_tasks_per_worker`` set, a worker is
retired gracefully after that many tasks and a fresh one takes its slot.

The task is pickled in the parent before anything is sent, so an unpicklable function or argument
raises :class:`TaskSerializationError` at the call site. A result the child cannot pickle raises
:class:`ResultSerializationError` in the caller. Exceptions raised by the task arrive as
:class:`TaskFailedError` carrying the child's traceback text; the original exception object is
never unpickled in the parent.

:meth:`WarmProcessPool.aclose` stops accepting work, fails queued submissions with
:class:`PoolClosedError`, waits ``shutdown_grace_seconds`` for in-flight tasks, then kills what
remains and joins every child. No descendants survive it.

Delivery is at most once: a task whose worker dies mid-flight fails with
:class:`WorkerCrashedError` rather than being re-run, because the pool cannot know how far
the child got. Callers that can retry safely decide that themselves.

The pool is for trusted, pure-CPU work. Tasks run with the server's privileges in a
long-lived interpreter that other tasks share, so code that mutates process-global state or
executes untrusted input needs recycling after every task or a fresh interpreter, and that
choice must be made explicitly per server.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import os
import pickle
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, NonNegativeFloat, PositiveFloat, PositiveInt


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public errors
# ---------------------------------------------------------------------------


class ProcessPoolError(Exception):
    """Base class for every error the pool raises."""


class TaskSerializationError(ProcessPoolError):
    """The function or its arguments could not be pickled, so the task never ran."""


class ResultSerializationError(ProcessPoolError):
    """The task ran, but the child could not pickle its return value."""


class TaskFailedError(ProcessPoolError):
    """The task raised inside the worker. ``error_type`` and ``child_traceback`` describe it."""

    def __init__(self, error_type: str, message: str, child_traceback: str) -> None:
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message
        self.child_traceback = child_traceback


class TaskTimeoutError(ProcessPoolError):
    """The task exceeded its deadline. The worker was killed and replaced."""


class WorkerCrashedError(ProcessPoolError):
    """The worker exited while the task was in flight. It was replaced."""

    def __init__(self, message: str, exitcode: Optional[int]) -> None:
        super().__init__(message)
        self.exitcode = exitcode


class WorkerStartError(ProcessPoolError):
    """A worker never reported ready, or its initializer raised."""


class PoolSaturatedError(ProcessPoolError):
    """The pending queue was full for longer than ``queue_timeout``."""


class PoolClosedError(ProcessPoolError):
    """The pool is closed, or was closed while this task was still queued or in flight."""


# ---------------------------------------------------------------------------
# Configuration, stats, timing
# ---------------------------------------------------------------------------


class WarmProcessPoolConfig(BaseModel):
    """Every knob the pool exposes. Defaults suit a verifier doing tens of milliseconds of CPU per call."""

    num_workers: PositiveInt = 4
    """Long-lived worker processes, and therefore the maximum number of tasks executing at once."""

    max_pending: PositiveInt = 1024
    """Submissions allowed to wait for a worker. Beyond this, :meth:`WarmProcessPool.run` waits or raises."""

    default_timeout_seconds: Optional[PositiveFloat] = 30.0
    """Per-task deadline measured from dispatch. ``None`` disables the deadline; prefer not to."""

    default_queue_timeout_seconds: Optional[NonNegativeFloat] = None
    """How long a submission may wait for a queue slot. ``None`` waits indefinitely, ``0`` fails fast."""

    max_tasks_per_worker: Optional[PositiveInt] = None
    """Retire a worker after this many tasks. Bounds memory growth from caches and leaks in the task code."""

    start_method: Literal["spawn", "forkserver"] = "spawn"
    """``fork`` is deliberately unavailable: it copies the server's event loop and locks into the child."""

    worker_start_timeout_seconds: PositiveFloat = 60.0
    """Time allowed for a child to import, run the initializer, and report ready."""

    kill_grace_seconds: PositiveFloat = 1.0
    """Wait after ``SIGTERM`` before ``SIGKILL`` when replacing a worker."""

    shutdown_grace_seconds: PositiveFloat = 5.0
    """Wait for in-flight tasks at :meth:`WarmProcessPool.aclose` before killing their workers."""


@dataclass
class PoolStats:
    """Cumulative counters and current gauges. Snapshot via :meth:`WarmProcessPool.stats`."""

    submitted: int = 0
    completed: int = 0
    failed: int = 0
    timeouts: int = 0
    cancelled_pending: int = 0
    cancelled_in_flight: int = 0
    crashes: int = 0
    restarts: int = 0
    recycles: int = 0
    spawn_failures: int = 0
    saturation_waits: int = 0
    saturation_rejections: int = 0
    task_serialization_errors: int = 0
    result_serialization_errors: int = 0
    queue_wait_seconds_total: float = 0.0
    queue_wait_seconds_max: float = 0.0
    execution_seconds_total: float = 0.0
    execution_seconds_max: float = 0.0
    live_workers: int = 0
    pending: int = 0
    in_flight: int = 0


@dataclass(frozen=True)
class TaskTiming:
    """Where one task spent its time. Queue wait ends when a worker takes it; execution ends with its reply."""

    queue_wait_seconds: float
    execution_seconds: float
    worker_pid: int


# ---------------------------------------------------------------------------
# Wire messages. Explicit, small, and pickled with the highest protocol.
# ---------------------------------------------------------------------------

_SHUTDOWN = b"\x00shutdown"
_READY = "ready"
_INIT_ERROR = "init_error"


@dataclass(frozen=True)
class _TaskMessage:
    task_id: int
    fn: Callable[..., Any]
    args: tuple
    kwargs: dict


@dataclass(frozen=True)
class _ResultMessage:
    task_id: int
    ok: bool
    value: Any = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    child_traceback: Optional[str] = None
    result_unpicklable: bool = False
    task_unpicklable: bool = False


def _dumps(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------


def _worker_main(
    conn: Any,
    initializer: Optional[Callable[..., Any]],
    initargs: tuple,
    *,
    install_signal_handlers: bool = True,
) -> None:
    """Serve tasks from ``conn`` until told to stop or the parent goes away.

    Runs in the child. ``install_signal_handlers`` exists so tests can drive the loop in a
    thread of the test process, where ``signal.signal`` is not permitted.
    """
    if install_signal_handlers:
        # The parent decides when this process dies. Ctrl-C in the parent must not spray
        # tracebacks from every child, and SIGTERM keeps its default (exit) so a wedged
        # pure-Python task can still be stopped without SIGKILL.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        _die_with_parent()

    try:
        if initializer is not None:
            initializer(*initargs)
    except BaseException:
        with contextlib.suppress(Exception):
            conn.send_bytes(_dumps((_INIT_ERROR, traceback.format_exc())))
        return
    conn.send_bytes(_dumps((_READY,)))

    while True:
        try:
            raw = conn.recv_bytes()
        except (EOFError, OSError):
            return
        if raw == _SHUTDOWN:
            return

        task_id = -1
        try:
            message: _TaskMessage = pickle.loads(raw)
            task_id = message.task_id
        except BaseException:
            # The parent pickled this successfully, so the function's module is not importable
            # here. Surface that as a serialization failure, not as a crash.
            result = _ResultMessage(
                task_id=task_id,
                ok=False,
                task_unpicklable=True,
                error_type="TaskSerializationError",
                error_message="task could not be unpickled in the worker",
                child_traceback=traceback.format_exc(),
            )
        else:
            try:
                value = message.fn(*message.args, **message.kwargs)
            except BaseException as exc:
                result = _ResultMessage(
                    task_id=task_id,
                    ok=False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    child_traceback=traceback.format_exc(),
                )
            else:
                result = _ResultMessage(task_id=task_id, ok=True, value=value)

        try:
            payload = _dumps(result)
        except BaseException as exc:
            payload = _dumps(
                _ResultMessage(
                    task_id=task_id,
                    ok=False,
                    result_unpicklable=True,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    child_traceback=traceback.format_exc(),
                )
            )
        try:
            conn.send_bytes(payload)
        except (BrokenPipeError, OSError):
            return


def _die_with_parent() -> None:
    """Linux only, best effort: have the kernel SIGKILL this worker when its parent dies.

    A worker that is idle notices the parent's death as EOF on its pipe and exits. A worker
    that is deep in a hung task never reads the pipe, so without this it would outlive a
    server killed with SIGKILL. macOS has no equivalent; there the daemon flag covers a
    normal exit and a hung task can outlive a SIGKILLed server.
    """
    if sys.platform != "linux":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, int(signal.SIGKILL), 0, 0, 0)
    except Exception:  # pragma: no cover - only reachable on an unusual libc
        return
    if os.getppid() == 1:  # pragma: no cover - the parent died before prctl took effect
        os._exit(0)


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


@dataclass
class _Worker:
    slot: int
    process: Any
    conn: Any
    tasks_done: int = 0

    @property
    def pid(self) -> int:
        try:
            return self.process.pid or -1
        except ValueError:
            return -1

    def is_alive(self) -> bool:
        # ``Process.is_alive`` raises once the handle is closed; a closed handle is not alive.
        try:
            return self.process.is_alive()
        except ValueError:
            return False


@dataclass
class _Pending:
    task_id: int
    payload: bytes
    future: asyncio.Future
    timeout: Optional[float]
    submitted_at: float
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    timing: Optional[TaskTiming] = None


class WarmProcessPool:
    """See the module docstring for semantics. Use as ``async with`` or call :meth:`start` and :meth:`aclose`."""

    def __init__(
        self,
        config: Optional[WarmProcessPoolConfig] = None,
        *,
        initializer: Optional[Callable[..., Any]] = None,
        initargs: tuple = (),
        name: str = "process_pool",
    ) -> None:
        self.config = config or WarmProcessPoolConfig()
        self.name = name
        self._initializer = initializer
        self._initargs = tuple(initargs)
        self._ctx = mp.get_context(self.config.start_method)
        self._stats = PoolStats()
        self._queue: Optional[asyncio.Queue[_Pending]] = None
        self._slot_tasks: list[asyncio.Task] = []
        self._workers: dict[int, _Worker] = {}
        self._in_flight: dict[int, _Pending] = {}
        self._next_task_id = 0
        self._started = False
        self._closing = False
        self._closed = False
        self._start_lock = asyncio.Lock()
        self._closed_event = asyncio.Event()

    # -- lifecycle -----------------------------------------------------------------

    async def __aenter__(self) -> "WarmProcessPool":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def start(self) -> None:
        """Spawn every worker and wait until each reports ready. Fails atomically on any worker.

        Safe to call concurrently and repeatedly; only the first call does the work.
        """
        async with self._start_lock:
            if self._started:
                return
            if self._closed:
                raise PoolClosedError(f"{self.name} is closed")
            self._queue = asyncio.Queue(maxsize=self.config.max_pending)
            outcomes = await asyncio.gather(
                *(self._spawn_worker(slot) for slot in range(self.config.num_workers)), return_exceptions=True
            )
            for outcome in outcomes:
                if isinstance(outcome, _Worker):
                    self._workers[outcome.slot] = outcome
            failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
            if failures:
                await self._kill_all_workers()
                raise failures[0]
            self._stats.live_workers = len(self._workers)
            self._slot_tasks = [
                asyncio.create_task(self._slot_loop(slot), name=f"{self.name}-slot-{slot}")
                for slot in range(self.config.num_workers)
            ]
            self._started = True

    async def aclose(self) -> None:
        """Stop accepting work, drain, and make sure no child survives.

        Safe to call more than once and from several tasks at once: later callers wait until
        the first call has finished, so nobody returns while workers are still being reaped.
        """
        if self._closing:
            await self._closed_event.wait()
            return
        self._closing = True
        self._closed = True
        try:
            await self._aclose()
        finally:
            self._closed_event.set()

    async def _aclose(self) -> None:
        if not self._started:
            await self._kill_all_workers()
            return

        self._fail_queued(PoolClosedError("pool closed before the task was dispatched"))

        # Give in-flight work a bounded chance to finish, then cancel the slot loops. A slot
        # loop cancelled mid-execution kills its worker and fails the task with PoolClosedError.
        deadline = time.monotonic() + self.config.shutdown_grace_seconds
        while self._in_flight and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        for task in self._slot_tasks:
            task.cancel()
        await asyncio.gather(*self._slot_tasks, return_exceptions=True)
        self._slot_tasks = []
        # Anything that slipped past the closed check between the first drain and the loops stopping.
        self._fail_queued(PoolClosedError("pool closed before the task was dispatched"))

        # Idle workers get a polite shutdown; anything still alive after the grace is killed.
        for worker in list(self._workers.values()):
            with contextlib.suppress(OSError, ValueError):
                worker.conn.send_bytes(_SHUTDOWN)
        await self._reap_workers(list(self._workers.values()), grace=self.config.shutdown_grace_seconds)
        self._workers.clear()
        self._stats.live_workers = 0

    def _fail_queued(self, exc: Exception) -> None:
        assert self._queue is not None
        while True:
            try:
                pending = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not pending.future.done():
                pending.future.set_exception(exc)
        self._stats.pending = 0

    # -- submission ------------------------------------------------------------------

    async def run(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        timeout: Any = ...,
        queue_timeout: Any = ...,
        **kwargs: Any,
    ) -> Any:
        """Run ``fn(*args, **kwargs)`` in a worker and return its result.

        ``timeout`` and ``queue_timeout`` belong to the pool. A task that itself takes a keyword
        argument by either name must be wrapped with :func:`functools.partial`.
        """
        value, _ = await self.run_with_timing(fn, *args, timeout=timeout, queue_timeout=queue_timeout, **kwargs)
        return value

    async def run_with_timing(
        self,
        fn: Callable[..., Any],
        /,
        *args: Any,
        timeout: Any = ...,
        queue_timeout: Any = ...,
        **kwargs: Any,
    ) -> tuple[Any, TaskTiming]:
        """Like :meth:`run`, and also report how long the task queued and executed."""
        if self._closed:
            raise PoolClosedError(f"{self.name} is closed")
        if not self._started or self._queue is None:
            raise RuntimeError("WarmProcessPool.start() has not been awaited")
        if timeout is ...:
            timeout = self.config.default_timeout_seconds
        if queue_timeout is ...:
            queue_timeout = self.config.default_queue_timeout_seconds

        task_id = self._next_task_id
        self._next_task_id += 1
        try:
            payload = _dumps(_TaskMessage(task_id=task_id, fn=fn, args=args, kwargs=kwargs))
        except Exception as exc:
            self._stats.task_serialization_errors += 1
            name = getattr(fn, "__qualname__", repr(fn))
            raise TaskSerializationError(f"task {name} is not picklable: {exc}") from exc

        loop = asyncio.get_running_loop()
        pending = _Pending(
            task_id=task_id,
            payload=payload,
            future=loop.create_future(),
            timeout=timeout,
            submitted_at=time.monotonic(),
        )
        pending.future.add_done_callback(lambda fut: pending.cancelled.set() if fut.cancelled() else None)

        if self._queue.full():
            self._stats.saturation_waits += 1
            if queue_timeout == 0:
                self._stats.saturation_rejections += 1
                raise PoolSaturatedError(f"{self.name}: {self.config.max_pending} tasks already pending")
        try:
            await asyncio.wait_for(self._queue.put(pending), timeout=queue_timeout)
        except asyncio.TimeoutError:
            self._stats.saturation_rejections += 1
            raise PoolSaturatedError(
                f"{self.name}: no queue slot within {queue_timeout}s ({self.config.max_pending} pending)"
            ) from None
        self._stats.submitted += 1
        self._stats.pending = self._queue.qsize()

        # If the pool closed between the check above and the put, no slot loop will ever take this.
        if self._closing and not pending.future.done():
            self._fail_queued(PoolClosedError(f"{self.name} closed before the task was dispatched"))

        value = await pending.future
        assert pending.timing is not None
        return value, pending.timing

    def stats(self) -> PoolStats:
        """A copy of the counters and gauges at this instant."""
        snapshot = PoolStats(**vars(self._stats))
        snapshot.pending = self._queue.qsize() if self._queue is not None else 0
        snapshot.in_flight = len(self._in_flight)
        snapshot.live_workers = len(self._workers)
        return snapshot

    # -- slot loop -------------------------------------------------------------------

    async def _slot_loop(self, slot: int) -> None:
        assert self._queue is not None
        while not self._closing:
            pending = await self._queue.get()
            self._stats.pending = self._queue.qsize()
            if pending.future.done():
                # Cancelled (or failed) while it waited; never reached a worker.
                self._stats.cancelled_pending += 1
                continue

            worker = self._workers.get(slot)
            if worker is None:
                # The last replacement failed to spawn. Try again now that there is work; if it
                # fails again, this task pays for it and the slot keeps retrying on the next one.
                try:
                    worker = self._workers[slot] = await self._spawn_worker(slot)
                except WorkerStartError as exc:
                    self._stats.spawn_failures += 1
                    logger.error("%s: slot %d still cannot spawn a worker: %s", self.name, slot, exc)
                    self._fail(pending, exc, time.monotonic() - pending.submitted_at, 0.0)
                    await asyncio.sleep(min(1.0, self.config.kill_grace_seconds))
                    continue

            try:
                replace_reason = await self._execute(worker, pending)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a parent-side bug must not take the slot down with it
                logger.exception("%s: unexpected error while handling a task in slot %d", self.name, slot)
                self._fail(
                    pending,
                    ProcessPoolError(f"{self.name}: internal error handling task {pending.task_id}: {exc!r}"),
                    time.monotonic() - pending.submitted_at,
                    0.0,
                )
                replace_reason = "internal-error"

            if replace_reason is not None:
                self._stats.restarts += 1
                await self._replace_worker(slot, replace_reason)
            elif (
                self.config.max_tasks_per_worker is not None and worker.tasks_done >= self.config.max_tasks_per_worker
            ):
                self._stats.recycles += 1
                await self._replace_worker(slot, "recycle", graceful=True)

    async def _execute(self, worker: _Worker, pending: _Pending) -> Optional[str]:
        """Dispatch one task to ``worker``. Returns a reason to replace the worker, or ``None`` to keep it."""
        dispatched_at = time.monotonic()
        queue_wait = dispatched_at - pending.submitted_at
        self._in_flight[pending.task_id] = pending
        try:
            try:
                worker.conn.send_bytes(pending.payload)
            except (BrokenPipeError, OSError) as exc:
                self._stats.crashes += 1
                self._fail(
                    pending,
                    WorkerCrashedError(
                        f"worker {worker.pid} died before accepting the task: {exc}", worker.process.exitcode
                    ),
                    queue_wait,
                    0.0,
                )
                return "crashed-on-dispatch"

            readable = await _wait_readable(worker.conn, timeout=pending.timeout, until=pending.cancelled)
            execution = time.monotonic() - dispatched_at

            if not readable:
                if pending.cancelled.is_set():
                    self._stats.cancelled_in_flight += 1
                    self._record_timing(queue_wait, execution)
                    return "cancelled"
                self._stats.timeouts += 1
                self._fail(
                    pending,
                    TaskTimeoutError(f"task exceeded {pending.timeout}s in worker {worker.pid}"),
                    queue_wait,
                    execution,
                )
                return "timeout"

            try:
                raw = worker.conn.recv_bytes()
            except (EOFError, OSError):
                # The fd woke us because the child closed it: the worker died mid-task.
                worker.process.join(timeout=0)
                self._stats.crashes += 1
                self._fail(
                    pending,
                    WorkerCrashedError(
                        f"worker {worker.pid} exited with code {worker.process.exitcode} while running the task",
                        worker.process.exitcode,
                    ),
                    queue_wait,
                    execution,
                )
                return "crashed"

            result: _ResultMessage = pickle.loads(raw)
            worker.tasks_done += 1
            pending.timing = TaskTiming(
                queue_wait_seconds=queue_wait, execution_seconds=execution, worker_pid=worker.pid
            )
            self._record_timing(queue_wait, execution)

            if pending.future.done():  # pragma: no cover - a race window a test cannot hit deterministically
                # Cancelled in the instant between the reply arriving and us reading it. The
                # worker is healthy and idle, so keep it.
                self._stats.cancelled_in_flight += 1
                return None

            if result.ok:
                self._stats.completed += 1
                pending.future.set_result(result.value)
            elif result.result_unpicklable:
                self._stats.result_serialization_errors += 1
                pending.future.set_exception(
                    ResultSerializationError(
                        f"result of task {pending.task_id} is not picklable: {result.error_type}: {result.error_message}"
                    )
                )
            elif result.task_unpicklable:
                self._stats.task_serialization_errors += 1
                pending.future.set_exception(
                    TaskSerializationError(
                        f"task {pending.task_id} could not be unpickled in worker {worker.pid}:\n{result.child_traceback}"
                    )
                )
            else:
                self._stats.failed += 1
                pending.future.set_exception(
                    TaskFailedError(
                        result.error_type or "Exception", result.error_message or "", result.child_traceback or ""
                    )
                )
            return None
        except asyncio.CancelledError:
            # The pool is closing under us and this worker may be mid-task: kill it rather than leave it.
            self._fail(
                pending,
                PoolClosedError(f"{self.name} closed while the task was in flight"),
                queue_wait,
                time.monotonic() - dispatched_at,
            )
            self._workers.pop(worker.slot, None)
            await self._reap_workers([worker], grace=0.0)
            raise
        finally:
            self._in_flight.pop(pending.task_id, None)

    def _fail(self, pending: _Pending, exc: Exception, queue_wait: float, execution: float) -> None:
        pending.timing = TaskTiming(queue_wait_seconds=queue_wait, execution_seconds=execution, worker_pid=-1)
        self._record_timing(queue_wait, execution)
        if not pending.future.done():
            pending.future.set_exception(exc)

    def _record_timing(self, queue_wait: float, execution: float) -> None:
        self._stats.queue_wait_seconds_total += queue_wait
        self._stats.queue_wait_seconds_max = max(self._stats.queue_wait_seconds_max, queue_wait)
        self._stats.execution_seconds_total += execution
        self._stats.execution_seconds_max = max(self._stats.execution_seconds_max, execution)

    # -- worker management ---------------------------------------------------------------

    async def _spawn_worker(self, slot: int) -> _Worker:
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=_worker_main,
            args=(child_conn, self._initializer, self._initargs),
            name=f"{self.name}-worker-{slot}",
            daemon=True,
        )
        try:
            # Process.start() blocks for the spawn exec; keep the loop responsive. It also
            # pickles the initializer, so an unpicklable one fails here, in the parent.
            await asyncio.to_thread(process.start)
        except Exception as exc:
            parent_conn.close()
            raise WorkerStartError(f"{self.name}: could not start worker for slot {slot}: {exc}") from exc
        finally:
            child_conn.close()

        worker = _Worker(slot=slot, process=process, conn=parent_conn)
        pid = worker.pid
        readable = await _wait_readable(parent_conn, timeout=self.config.worker_start_timeout_seconds)
        if not readable:
            await self._reap_workers([worker], grace=0.0)
            raise WorkerStartError(
                f"{self.name}: worker {pid} did not report ready within {self.config.worker_start_timeout_seconds}s"
            )
        try:
            message = pickle.loads(parent_conn.recv_bytes())
        except (EOFError, OSError) as exc:
            # Reaping closes the handle, so read the exit code first.
            process.join(timeout=0)
            exitcode = process.exitcode
            await self._reap_workers([worker], grace=0.0)
            raise WorkerStartError(
                f"{self.name}: worker {pid} for slot {slot} exited during startup (exit code {exitcode}); "
                "its traceback, if any, is on this process's stderr"
            ) from exc
        if message[0] != _READY:
            await self._reap_workers([worker], grace=0.0)
            raise WorkerStartError(f"{self.name}: initializer raised in worker for slot {slot}:\n{message[1]}")
        return worker

    async def _replace_worker(self, slot: int, reason: str, *, graceful: bool = False) -> None:
        old = self._workers.pop(slot, None)
        if old is not None:
            log = logger.info if graceful else logger.warning
            log("%s: replacing worker pid=%s in slot %d (%s)", self.name, old.pid, slot, reason)
            if graceful:
                with contextlib.suppress(OSError, ValueError):
                    old.conn.send_bytes(_SHUTDOWN)
                await self._reap_workers([old], grace=self.config.kill_grace_seconds)
            else:
                await self._reap_workers([old], grace=0.0)
        if self._closing:
            return
        try:
            self._workers[slot] = await self._spawn_worker(slot)
        except WorkerStartError as exc:
            self._stats.spawn_failures += 1
            logger.error(
                "%s: could not respawn worker for slot %d, will retry on the next task: %s", self.name, slot, exc
            )

    async def _reap_workers(self, workers: list[_Worker], *, grace: float) -> None:
        """Wait up to ``grace`` for exits, then SIGTERM, wait ``kill_grace_seconds``, then SIGKILL. Always join.

        If the slot loop running this is cancelled part-way (the pool is closing), the reap finishes
        synchronously before the cancellation propagates, so a half-killed worker is never orphaned.
        """
        try:
            deadline = time.monotonic() + grace
            while any(w.is_alive() for w in workers) and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            for w in workers:
                if w.is_alive():
                    with contextlib.suppress(OSError, ValueError):
                        w.process.terminate()
            deadline = time.monotonic() + self.config.kill_grace_seconds
            while any(w.is_alive() for w in workers) and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            for w in workers:
                if w.is_alive():
                    with contextlib.suppress(OSError, ValueError):
                        w.process.kill()
            # SIGKILL is not instantaneous; join off the loop so a slow reap cannot stall requests.
            await asyncio.gather(*(asyncio.to_thread(_join_quietly, w.process) for w in workers))
        except asyncio.CancelledError:
            _kill_and_join_now(workers)
            raise
        finally:
            for w in workers:
                with contextlib.suppress(OSError, ValueError):
                    w.conn.close()
                with contextlib.suppress(ValueError):
                    w.process.close()

    async def _kill_all_workers(self) -> None:
        await self._reap_workers(list(self._workers.values()), grace=0.0)
        self._workers.clear()
        self._stats.live_workers = 0


def _join_quietly(process: Any) -> None:
    with contextlib.suppress(ValueError, OSError, AssertionError):
        process.join(timeout=5.0)


def _kill_and_join_now(workers: list[_Worker]) -> None:
    """Synchronous last resort used when a reap is cancelled: no grace, SIGKILL, join."""
    for w in workers:
        if w.is_alive():
            with contextlib.suppress(OSError, ValueError):
                w.process.kill()
    for w in workers:
        _join_quietly(w.process)


async def _wait_readable(conn: Any, *, timeout: Optional[float], until: Optional[asyncio.Event] = None) -> bool:
    """Wait until ``conn`` has data, ``timeout`` passes, or ``until`` is set. True only when data arrived."""
    loop = asyncio.get_running_loop()
    readable = asyncio.Event()
    fd = conn.fileno()
    loop.add_reader(fd, readable.set)
    waiters = [asyncio.ensure_future(readable.wait())]
    if until is not None:
        waiters.append(asyncio.ensure_future(until.wait()))
    try:
        await asyncio.wait(waiters, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        return readable.is_set()
    finally:
        loop.remove_reader(fd)
        for waiter in waiters:
            waiter.cancel()


def worker_start_method_is_supported(method: str) -> bool:
    """True when the current platform offers ``method``. ``spawn`` is available everywhere."""
    return method in mp.get_all_start_methods()


__all__ = [
    "PoolClosedError",
    "PoolSaturatedError",
    "PoolStats",
    "ProcessPoolError",
    "ResultSerializationError",
    "TaskFailedError",
    "TaskSerializationError",
    "TaskTimeoutError",
    "TaskTiming",
    "WarmProcessPool",
    "WarmProcessPoolConfig",
    "WorkerCrashedError",
    "WorkerStartError",
    "worker_start_method_is_supported",
]
