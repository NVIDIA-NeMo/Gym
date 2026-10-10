# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runtime metrics that show what is slowing a Gym process down.

A Gym server answers every request on one asyncio event loop.
When synchronous work such as validation, serialization, or garbage collection holds that loop, every other request waits, even though no single request looks slow.
These metrics make that pressure visible without inspecting private asyncio internals:

* ``gym.runtime.event_loop.lag`` (histogram, seconds): how late a sleeping coroutine wakes up.
  A coroutine asks to wake after a fixed interval, and the lag is how much later than that it actually runs.
  This is the scheduler-pressure signal.
* ``gym.runtime.event_loop.tasks`` (gauge): how many asyncio tasks exist on the loop.
  This is resident work, not a queue depth: an idle task waiting on the network counts as much as a runnable one.
* ``gym.runtime.gc.pause`` (histogram, seconds), and the ``gym.runtime.gc.collections``, ``gym.runtime.gc.collected``, and ``gym.runtime.gc.uncollectable`` counters, all by ``generation``.
  ``gym.runtime.gc.count`` (gauge, by ``generation``) is the allocation count from ``gc.get_count()``.
* ``process.cpu.time`` (counter, seconds, by ``cpu.mode``), ``process.memory.usage``, ``process.thread.count``, and ``process.open_file_descriptor.count``.
* ``gym.server.requests.in_flight``, ``gym.server.requests.cancelled``, and ``gym.server.requests.failed``.
  The outermost server middleware keeps them as plain integers, and the exporter thread reads them.
* ``gym.http.client.requests.in_flight``: outbound calls through ``server_utils.request`` still waiting for their response headers.

Each kind of work runs where it costs the least:

* The event loop runs one sampler coroutine per interval, which measures lag and counts tasks.
* The exporter's background thread reads process CPU, memory, threads, file descriptors, and GC totals when it exports, through observable instruments.
* The garbage collector's callback only stores timestamps and totals.
  It never calls OpenTelemetry, because it runs inside a collection.
  Pause durations wait in a bounded buffer until the next sample records them.

Process-wide state, which is the GC callback and the observable instruments, is set up once per process.
The last event loop to stop removes it.
Everything is opt-in through ``telemetry.runtime_metrics.enabled``, and every failure is logged and swallowed, because telemetry must never stop a server.
"""

import asyncio
import gc
import logging
import threading
import time
import weakref
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, Optional


logger = logging.getLogger(__name__)

EVENT_LOOP_LAG_INSTRUMENT = "gym.runtime.event_loop.lag"
EVENT_LOOP_TASKS_INSTRUMENT = "gym.runtime.event_loop.tasks"
GC_PAUSE_INSTRUMENT = "gym.runtime.gc.pause"
GC_COLLECTIONS_INSTRUMENT = "gym.runtime.gc.collections"
GC_COLLECTED_INSTRUMENT = "gym.runtime.gc.collected"
GC_UNCOLLECTABLE_INSTRUMENT = "gym.runtime.gc.uncollectable"
GC_COUNT_INSTRUMENT = "gym.runtime.gc.count"
PROCESS_CPU_TIME_INSTRUMENT = "process.cpu.time"
PROCESS_MEMORY_USAGE_INSTRUMENT = "process.memory.usage"
PROCESS_THREAD_COUNT_INSTRUMENT = "process.thread.count"
PROCESS_OPEN_FDS_INSTRUMENT = "process.open_file_descriptor.count"
REQUESTS_IN_FLIGHT_INSTRUMENT = "gym.server.requests.in_flight"
REQUESTS_CANCELLED_INSTRUMENT = "gym.server.requests.cancelled"
REQUESTS_FAILED_INSTRUMENT = "gym.server.requests.failed"
OUTBOUND_IN_FLIGHT_INSTRUMENT = "gym.http.client.requests.in_flight"

EVENT_LOOP_ATTRIBUTE = "nemo.gym.event_loop"
GC_GENERATION_ATTRIBUTE = "generation"
CPU_MODE_ATTRIBUTE = "cpu.mode"

#: Seconds. Lag below half a millisecond is ordinary scheduling, and lag of seconds means the loop was blocked outright.
EVENT_LOOP_LAG_BOUNDARIES_S: tuple[float, ...] = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
)
#: Seconds. A young-generation collection takes microseconds, and a full collection of a large heap can take most of a second.
GC_PAUSE_BOUNDARIES_S: tuple[float, ...] = (
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1,
)

#: Upper bound on GC pauses held between two samples.
#: Pauses beyond it are still counted in ``gym.runtime.gc.collections`` but are not recorded in the pause histogram.
_MAX_BUFFERED_GC_PAUSES = 10_000


class _GcRecorder:
    """Collects GC pause times and totals from ``gc.callbacks`` without calling OpenTelemetry.

    It takes no lock.
    A lock would deadlock when a collection starts on a thread that already holds it, for example while ``drain_pauses`` allocates.
    Python runs one collection at a time, so the callback never races itself, and the buffer swap in ``drain_pauses`` is safe under the GIL.
    """

    def __init__(self) -> None:
        self._started: dict[int, float] = {}
        self._pauses: list[tuple[int, float]] = []
        self.collections = [0, 0, 0]
        self.collected = [0, 0, 0]
        self.uncollectable = [0, 0, 0]

    def __call__(self, phase: str, info: dict[str, int]) -> None:
        generation = info.get("generation", 0)
        if phase == "start":
            self._started[generation] = time.perf_counter()
            return
        started = self._started.pop(generation, None)
        if 0 <= generation <= 2:
            self.collections[generation] += 1
            self.collected[generation] += info.get("collected", 0)
            self.uncollectable[generation] += info.get("uncollectable", 0)
        # A stop without a start, such as one already in progress when the callback was installed, has no pause to report.
        if started is not None and len(self._pauses) < _MAX_BUFFERED_GC_PAUSES:
            self._pauses.append((generation, time.perf_counter() - started))

    def drain_pauses(self) -> list[tuple[int, float]]:
        # A pause appended between these two statements lands in the list being returned.
        pauses = self._pauses
        self._pauses = []
        return pauses


class RequestCounts:
    """Request totals that the server middleware updates with plain integer arithmetic.

    The exporter thread reads them through observable instruments, so a request costs no OpenTelemetry call.
    Only the server's own event loop updates them, so the updates never race.
    """

    def __init__(self) -> None:
        self.in_flight = 0
        self.cancelled = 0
        self.failed = 0
        self.outbound_in_flight = 0


def _observation(value: float, attributes: Optional[dict[str, Any]] = None) -> Any:
    from opentelemetry.metrics import Observation

    return Observation(value, attributes or {})


def _observe(read: Callable[["_ProcessRuntime"], list]) -> Callable[[Any], list]:
    """Wrap a reading of the running observer as an observable-instrument callback.

    It reports nothing while no observer runs, and one unavailable measurement never stops the others from exporting.
    """

    def observe(_options: Any) -> list:
        runtime = _PROCESS_RUNTIME
        if runtime is None:
            return []
        try:
            return read(runtime)
        except Exception:
            logger.debug("nemo-lens: a runtime measurement is unavailable", exc_info=True)
            return []

    return observe


def _by_generation(values: Callable[["_ProcessRuntime"], list[int]]) -> Callable[[Any], list]:
    return _observe(
        lambda runtime: [
            _observation(value, {GC_GENERATION_ATTRIBUTE: generation})
            for generation, value in enumerate(values(runtime))
        ]
    )


def _cpu_times(runtime: "_ProcessRuntime") -> list:
    times = runtime.process.cpu_times()
    return [
        _observation(times.user, {CPU_MODE_ATTRIBUTE: "user"}),
        _observation(times.system, {CPU_MODE_ATTRIBUTE: "system"}),
    ]


class _Instruments:
    """The runtime instruments of one meter, created once so a restarted observer never registers them twice."""

    def __init__(self, meter: Any) -> None:
        self.lag = meter.create_histogram(
            EVENT_LOOP_LAG_INSTRUMENT,
            unit="s",
            description="How late a sleeping coroutine on the event loop woke up.",
            explicit_bucket_boundaries_advisory=list(EVENT_LOOP_LAG_BOUNDARIES_S),
        )
        self.tasks = meter.create_gauge(
            EVENT_LOOP_TASKS_INSTRUMENT,
            unit="{task}",
            description="Asyncio tasks that exist on the event loop. This is resident work, not a queue depth.",
        )
        self.gc_pause = meter.create_histogram(
            GC_PAUSE_INSTRUMENT,
            unit="s",
            description="Duration of one garbage collection.",
            explicit_bucket_boundaries_advisory=list(GC_PAUSE_BOUNDARIES_S),
        )
        meter.create_observable_up_down_counter(
            REQUESTS_IN_FLIGHT_INSTRUMENT,
            callbacks=[_observe(lambda runtime: [_observation(runtime.requests.in_flight)])],
            unit="{request}",
            description="HTTP requests this server is handling.",
        )
        meter.create_observable_counter(
            REQUESTS_CANCELLED_INSTRUMENT,
            callbacks=[_observe(lambda runtime: [_observation(runtime.requests.cancelled)])],
            unit="{request}",
            description="HTTP requests cancelled because the client disconnected before the response finished.",
        )
        meter.create_observable_counter(
            REQUESTS_FAILED_INSTRUMENT,
            callbacks=[_observe(lambda runtime: [_observation(runtime.requests.failed)])],
            unit="{request}",
            description="HTTP requests whose handler raised past every middleware.",
        )
        meter.create_observable_up_down_counter(
            OUTBOUND_IN_FLIGHT_INSTRUMENT,
            callbacks=[_observe(lambda runtime: [_observation(runtime.requests.outbound_in_flight)])],
            unit="{request}",
            description="Outbound calls through server_utils.request that are still waiting for their response headers.",
        )
        meter.create_observable_counter(
            GC_COLLECTIONS_INSTRUMENT,
            callbacks=[_by_generation(lambda runtime: runtime.gc.collections)],
            unit="{collection}",
            description="Garbage collections, by generation.",
        )
        meter.create_observable_counter(
            GC_COLLECTED_INSTRUMENT,
            callbacks=[_by_generation(lambda runtime: runtime.gc.collected)],
            unit="{object}",
            description="Objects the garbage collector freed, by generation.",
        )
        meter.create_observable_counter(
            GC_UNCOLLECTABLE_INSTRUMENT,
            callbacks=[_by_generation(lambda runtime: runtime.gc.uncollectable)],
            unit="{object}",
            description="Objects the garbage collector found but could not free, by generation.",
        )
        meter.create_observable_gauge(
            GC_COUNT_INSTRUMENT,
            callbacks=[_by_generation(lambda _runtime: list(gc.get_count()))],
            unit="{object}",
            description="Allocations counted toward the next collection of each generation, from gc.get_count().",
        )
        meter.create_observable_counter(
            PROCESS_CPU_TIME_INSTRUMENT, callbacks=[_observe(_cpu_times)], unit="s", description="Process CPU time."
        )
        meter.create_observable_gauge(
            PROCESS_MEMORY_USAGE_INSTRUMENT,
            callbacks=[_observe(lambda runtime: [_observation(runtime.process.memory_info().rss)])],
            unit="By",
            description="Resident memory of this process.",
        )
        meter.create_observable_gauge(
            PROCESS_THREAD_COUNT_INSTRUMENT,
            callbacks=[_observe(lambda runtime: [_observation(runtime.process.num_threads())])],
            unit="{thread}",
            description="Threads in this process.",
        )
        meter.create_observable_gauge(
            PROCESS_OPEN_FDS_INSTRUMENT,
            # psutil has no descriptor count on Windows, where this reports nothing.
            callbacks=[_observe(lambda runtime: [_observation(runtime.process.num_fds())])],
            unit="{file_descriptor}",
            description="Open file descriptors in this process.",
        )


#: Instruments per meter, so a meter that is garbage collected takes its instruments with it.
_INSTRUMENTS: "weakref.WeakKeyDictionary[Any, _Instruments]" = weakref.WeakKeyDictionary()


class _ProcessRuntime:
    """The per-process state of the runtime metrics: the GC recorder, the request totals, and this process's handle."""

    def __init__(self, meter: Any) -> None:
        import psutil

        instruments = _INSTRUMENTS.get(meter)
        if instruments is None:
            instruments = _INSTRUMENTS[meter] = _Instruments(meter)
        self.process = psutil.Process()
        self.gc = _GcRecorder()
        self.requests = RequestCounts()
        self.lag = instruments.lag
        self.tasks = instruments.tasks
        self.gc_pause = instruments.gc_pause

    def record_gc_pauses(self) -> None:
        for generation, seconds in self.gc.drain_pauses():
            self.gc_pause.record(seconds, attributes={GC_GENERATION_ATTRIBUTE: generation})


_PROCESS_LOCK = threading.Lock()
_PROCESS_RUNTIME: Optional[_ProcessRuntime] = None
_ACTIVE_LOOPS = 0


def request_counts() -> Optional[RequestCounts]:
    """Return the request totals while an observer is running, else None.

    Server middleware calls this once per request, so it is a plain global read.
    """
    runtime = _PROCESS_RUNTIME
    return runtime.requests if runtime is not None else None


def _acquire_process_runtime(meter: Any) -> _ProcessRuntime:
    global _PROCESS_RUNTIME, _ACTIVE_LOOPS
    with _PROCESS_LOCK:
        if _PROCESS_RUNTIME is None:
            runtime = _ProcessRuntime(meter)
            gc.callbacks.append(runtime.gc)
            _PROCESS_RUNTIME = runtime
        _ACTIVE_LOOPS += 1
        return _PROCESS_RUNTIME


def _release_process_runtime() -> None:
    global _PROCESS_RUNTIME, _ACTIVE_LOOPS
    with _PROCESS_LOCK:
        _ACTIVE_LOOPS -= 1
        if _ACTIVE_LOOPS > 0 or _PROCESS_RUNTIME is None:
            return
        # Remove only this module's callback object, never another library's.
        try:
            gc.callbacks.remove(_PROCESS_RUNTIME.gc)
        except ValueError:
            pass
        _PROCESS_RUNTIME = None


async def _sample(runtime: _ProcessRuntime, interval_seconds: float, loop_name: str) -> None:
    loop = asyncio.get_running_loop()
    attributes = {EVENT_LOOP_ATTRIBUTE: loop_name}
    while True:
        deadline = time.monotonic() + interval_seconds
        await asyncio.sleep(interval_seconds)
        try:
            runtime.lag.record(max(0.0, time.monotonic() - deadline), attributes=attributes)
            runtime.tasks.set(len(asyncio.all_tasks(loop)), attributes=attributes)
            runtime.record_gc_pauses()
        except Exception:
            logger.debug("nemo-lens: runtime sample failed", exc_info=True)


def runtime_metrics_settings() -> Optional[float]:
    """Return the sampling interval when runtime metrics should run in this process, else None."""
    from nemo_gym.telemetry.setup import get_telemetry, is_metrics_exporter_active, runtime_metrics_config_from_env

    config = runtime_metrics_config_from_env()
    telemetry = get_telemetry()
    if not config.enabled or telemetry is None or not is_metrics_exporter_active():
        return None
    return config.interval_seconds


@asynccontextmanager
async def observe_event_loop(loop_name: str) -> AsyncIterator[None]:
    """Sample the running event loop, and this process's runtime, for as long as the context is open.

    ``loop_name`` is a bounded label for the loop, such as ``server``, ``head_server``, or ``driver``.
    Does nothing when runtime metrics are off or telemetry is not exporting.
    """
    sampler: Optional[asyncio.Task] = None
    acquired = False
    try:
        interval_seconds = runtime_metrics_settings()
        if interval_seconds is not None:
            from nemo_gym.telemetry.setup import get_telemetry

            runtime = _acquire_process_runtime(get_telemetry().meter)
            acquired = True
            sampler = asyncio.create_task(
                _sample(runtime, interval_seconds, loop_name), name="nemo-gym-runtime-sampler"
            )
    except Exception:
        logger.warning("nemo-lens: runtime metrics could not start; continuing without them", exc_info=True)
    try:
        yield
    finally:
        if sampler is not None:
            sampler.cancel()
            try:
                await sampler
            except (asyncio.CancelledError, Exception):
                pass
        if acquired:
            _release_process_runtime()


def observe_app_event_loop(app: Any, loop_name: str) -> None:
    """Run :func:`observe_event_loop` for the lifetime of a FastAPI app, keeping any lifespan the app already has."""
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: Any) -> AsyncIterator[Any]:
        async with observe_event_loop(loop_name):
            async with original_lifespan(application) as state:
                yield state

    app.router.lifespan_context = lifespan
