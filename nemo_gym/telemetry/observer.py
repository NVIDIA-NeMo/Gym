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
"""Event-loop, process, request, and rollout-driver metrics that show what is slowing a Gym process down.

A Gym server answers every request on one asyncio event loop.
When synchronous work such as validation, serialization, or garbage collection holds that loop, every other request waits, even though no single request looks slow.
These metrics make that pressure visible without inspecting private asyncio internals:

* ``gym.event_loop.lag_ms`` (histogram, milliseconds): how late a sleeping coroutine wakes up.
  A coroutine asks to wake after a fixed interval, and the lag is how much later than that it actually runs.
  This is the scheduler-pressure signal.
* ``gym.event_loop.tasks`` (gauge, opt-in): how many asyncio tasks exist on the loop.
  This is resident work, not a queue depth: an idle task waiting on the network counts as much as a runnable one.
* ``gym.gc.pause_duration_ms`` (histogram, milliseconds), and the ``gym.gc.collections_total``, ``gym.gc.collected_total``, and ``gym.gc.uncollectable_total`` counters, all by ``nemo.gym.gc.generation``.
  ``gym.gc.count`` (gauge, by ``nemo.gym.gc.generation``) is the allocation count from ``gc.get_count()``.
* ``process.cpu.time`` (counter, seconds, by ``cpu.mode``), ``process.memory.usage``, ``process.thread.count``, and ``process.open_file_descriptor.count``.
  These keep the OpenTelemetry process semantic-convention names, units, and attributes, so standard dashboards recognize them.
* ``gym.server.requests.in_flight``, ``gym.server.requests.unhandled_exceptions_total``, and ``gym.server.responses_total`` by ``http.response.status_code``.
  ``gym.server.requests.cancelled_total`` carries ``http.route`` and ``nemo.gym.request.phase``.
  The outermost server middleware keeps these totals locally, and the exporter thread reads them.
* ``gym.http.client.requests.in_flight``: outbound calls through ``client.http_client.request`` still waiting for their response headers.
* ``gym.driver.rollouts.offered_total``, ``gym.driver.rollouts.resident``, and ``gym.driver.rollouts.active``, plus the ``gym.driver.admission_duration_ms``, ``gym.driver.output_write_duration_ms``, ``gym.driver.output_write_bytes``, and ``gym.driver.materialize_duration_ms`` histograms, from the rollout driver.
* ``gym.driver.rollout.input_items``, ``gym.driver.rollout.output_items``, ``gym.driver.rollout.turns``, and ``gym.driver.rollout.tool_calls``: the size of each completed rollout, recorded once per rollout by the driver.

Each kind of work runs where it costs the least:

* The event loop runs one sampler coroutine per interval, which measures lag, and counts tasks when that is on.
* The exporter's background thread reads process CPU, memory, threads, file descriptors, and GC totals when it exports, through observable instruments.
* The garbage collector's callback only stores timestamps and totals.
  It never calls OpenTelemetry, because it runs inside a collection.
  Pause durations wait in a bounded buffer until the next sample records them.

Process-wide state, which is the GC callback and the observable instruments, is set up once per process.
The last event loop to stop removes it.
These metrics are on whenever telemetry metrics are, because each costs at most a few microseconds per request or rollout.
The one exception is the event-loop task count, which walks every task on the loop and is opt-in through ``telemetry.event_loop_task_count``.
Every failure is logged and swallowed, because telemetry must never stop a server.
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

EVENT_LOOP_LAG_INSTRUMENT = "gym.event_loop.lag_ms"
EVENT_LOOP_TASKS_INSTRUMENT = "gym.event_loop.tasks"
GC_PAUSE_INSTRUMENT = "gym.gc.pause_duration_ms"
GC_COLLECTIONS_INSTRUMENT = "gym.gc.collections_total"
GC_COLLECTED_INSTRUMENT = "gym.gc.collected_total"
GC_UNCOLLECTABLE_INSTRUMENT = "gym.gc.uncollectable_total"
GC_COUNT_INSTRUMENT = "gym.gc.count"
PROCESS_CPU_TIME_INSTRUMENT = "process.cpu.time"
PROCESS_MEMORY_USAGE_INSTRUMENT = "process.memory.usage"
PROCESS_THREAD_COUNT_INSTRUMENT = "process.thread.count"
PROCESS_OPEN_FDS_INSTRUMENT = "process.open_file_descriptor.count"
REQUESTS_IN_FLIGHT_INSTRUMENT = "gym.server.requests.in_flight"
REQUESTS_CANCELLED_INSTRUMENT = "gym.server.requests.cancelled_total"
REQUESTS_UNHANDLED_EXCEPTIONS_INSTRUMENT = "gym.server.requests.unhandled_exceptions_total"
RESPONSES_INSTRUMENT = "gym.server.responses_total"
OUTBOUND_IN_FLIGHT_INSTRUMENT = "gym.http.client.requests.in_flight"
DRIVER_OFFERED_INSTRUMENT = "gym.driver.rollouts.offered_total"
DRIVER_RESIDENT_INSTRUMENT = "gym.driver.rollouts.resident"
DRIVER_ACTIVE_INSTRUMENT = "gym.driver.rollouts.active"
DRIVER_ADMISSION_INSTRUMENT = "gym.driver.admission_duration_ms"
DRIVER_WRITE_DURATION_INSTRUMENT = "gym.driver.output_write_duration_ms"
DRIVER_WRITE_BYTES_INSTRUMENT = "gym.driver.output_write_bytes"
DRIVER_MATERIALIZE_INSTRUMENT = "gym.driver.materialize_duration_ms"
ROLLOUT_INPUT_ITEMS_INSTRUMENT = "gym.driver.rollout.input_items"
ROLLOUT_OUTPUT_ITEMS_INSTRUMENT = "gym.driver.rollout.output_items"
ROLLOUT_TURNS_INSTRUMENT = "gym.driver.rollout.turns"
ROLLOUT_TOOL_CALLS_INSTRUMENT = "gym.driver.rollout.tool_calls"

EVENT_LOOP_ATTRIBUTE = "nemo.gym.event_loop"
GC_GENERATION_ATTRIBUTE = "nemo.gym.gc.generation"
CPU_MODE_ATTRIBUTE = "cpu.mode"
HTTP_STATUS_CODE_ATTRIBUTE = "http.response.status_code"
HTTP_ROUTE_ATTRIBUTE = "http.route"
REQUEST_PHASE_ATTRIBUTE = "nemo.gym.request.phase"

#: Milliseconds. Lag below half a millisecond is ordinary scheduling, and lag of seconds means the loop was blocked outright.
EVENT_LOOP_LAG_BOUNDARIES_MS: tuple[float, ...] = (
    0.5,
    1,
    2.5,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
)
#: Milliseconds. A young-generation collection takes microseconds, and a full collection of a large heap can take most of a second.
GC_PAUSE_BOUNDARIES_MS: tuple[float, ...] = (
    0.1,
    0.25,
    0.5,
    1,
    2.5,
    5,
    10,
    25,
    50,
    100,
    250,
    500,
    1000,
)

#: Upper bound on GC pauses held between two samples.
#: Pauses beyond it are still counted in ``gym.gc.collections_total`` but are not recorded in the pause histogram.
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
    """Request totals that the server middleware and ``client.http_client.request`` update with plain integer arithmetic.

    The exporter thread reads them through observable instruments, so a request costs no OpenTelemetry call.
    The serving event loop updates the inbound totals.
    ``outbound_in_flight`` can also change on another thread's event loop, such as the one the synchronous sandbox API runs on.
    Each update is a single ``+=`` on an attribute with no call in between, which CPython with the GIL never interrupts with another thread, so no update is lost.
    """

    def __init__(self) -> None:
        self.in_flight = 0
        self.cancelled = 0
        self.cancelled_by_route_and_phase: dict[tuple[str, str], int] = {}
        """Cumulative cancellations by registered route template and response-start phase."""
        self.unhandled_exceptions = 0
        self.outbound_in_flight = 0
        self.responses: dict[int, int] = {}
        """Responses by HTTP status code, updated only by the serving event loop. HTTP bounds the set of codes."""

    def count_cancellation(self, route: str, *, response_started: bool) -> None:
        """Count one disconnect; ``route`` must be a registered template or the unmatched sentinel."""
        phase = "response_started" if response_started else "before_response_start"
        key = (route, phase)
        self.cancelled_by_route_and_phase[key] = self.cancelled_by_route_and_phase.get(key, 0) + 1
        self.cancelled += 1

    def count_response(self, status: int) -> None:
        self.responses[status] = self.responses.get(status, 0) + 1


def _observation(value: float, attributes: Optional[dict[str, Any]] = None) -> Any:
    from opentelemetry.metrics import Observation

    return Observation(value, attributes or {})


def _observe(read: Callable[["_ProcessObserver"], list]) -> Callable[[Any], list]:
    """Wrap a reading of the running observer as an observable-instrument callback.

    It reports nothing while no observer runs, and one unavailable measurement never stops the others from exporting.
    """

    def observe(_options: Any) -> list:
        state = _PROCESS_OBSERVER
        if state is None:
            return []
        try:
            return read(state)
        except Exception:
            logger.debug("nemo-lens: an observer measurement is unavailable", exc_info=True)
            return []

    return observe


def _driver_total(read: Callable[["DriverCounts"], int]) -> Callable[[Any], list]:
    """Report a driver total only in a process that has offered rollouts, so servers export no constant-zero series."""
    return _observe(lambda state: [_observation(read(state.driver))] if state.driver.offered else [])


def _by_generation(values: Callable[["_ProcessObserver"], list[int]]) -> Callable[[Any], list]:
    return _observe(
        lambda state: [
            _observation(value, {GC_GENERATION_ATTRIBUTE: generation})
            for generation, value in enumerate(values(state))
        ]
    )


def _cpu_times(state: "_ProcessObserver") -> list:
    times = state.process.cpu_times()
    return [
        _observation(times.user, {CPU_MODE_ATTRIBUTE: "user"}),
        _observation(times.system, {CPU_MODE_ATTRIBUTE: "system"}),
    ]


class _Instruments:
    """The observer's instruments for one meter, created once so a restarted observer never registers them twice."""

    def __init__(self, meter: Any) -> None:
        self.lag = meter.create_histogram(
            EVENT_LOOP_LAG_INSTRUMENT,
            unit="ms",
            description="How late a sleeping coroutine on the event loop woke up.",
            explicit_bucket_boundaries_advisory=list(EVENT_LOOP_LAG_BOUNDARIES_MS),
        )
        self.tasks = meter.create_gauge(
            EVENT_LOOP_TASKS_INSTRUMENT,
            unit="{task}",
            description="Asyncio tasks that exist on the event loop. This is resident work, not a queue depth.",
        )
        self.gc_pause = meter.create_histogram(
            GC_PAUSE_INSTRUMENT,
            unit="ms",
            description="Duration of one garbage collection.",
            explicit_bucket_boundaries_advisory=list(GC_PAUSE_BOUNDARIES_MS),
        )
        meter.create_observable_up_down_counter(
            REQUESTS_IN_FLIGHT_INSTRUMENT,
            callbacks=[_observe(lambda state: [_observation(state.requests.in_flight)])],
            unit="{request}",
            description="HTTP requests this server is handling.",
        )
        meter.create_observable_counter(
            REQUESTS_CANCELLED_INSTRUMENT,
            callbacks=[
                _observe(
                    lambda state: [
                        # Snapshot before iterating: the serving loop can add a bucket during export.
                        # Export only these buckets, not the aggregate, which would double-count cancels.
                        _observation(count, {HTTP_ROUTE_ATTRIBUTE: route, REQUEST_PHASE_ATTRIBUTE: phase})
                        for (route, phase), count in list(state.requests.cancelled_by_route_and_phase.items())
                    ]
                )
            ],
            unit="{request}",
            description="HTTP requests cancelled on client disconnect, by route template and response-start phase.",
        )
        meter.create_observable_counter(
            REQUESTS_UNHANDLED_EXCEPTIONS_INSTRUMENT,
            callbacks=[_observe(lambda state: [_observation(state.requests.unhandled_exceptions)])],
            unit="{request}",
            description="HTTP requests whose handler raised an exception that no middleware turned into a response.",
        )
        meter.create_observable_counter(
            RESPONSES_INSTRUMENT,
            callbacks=[
                _observe(
                    lambda state: [
                        # list() copies the items in one step, so the serving loop can add a code during the export.
                        _observation(count, {HTTP_STATUS_CODE_ATTRIBUTE: status})
                        for status, count in list(state.requests.responses.items())
                    ]
                )
            ],
            unit="{response}",
            description="HTTP responses the server started sending, by status code.",
        )
        meter.create_observable_up_down_counter(
            OUTBOUND_IN_FLIGHT_INSTRUMENT,
            callbacks=[_observe(lambda state: [_observation(state.requests.outbound_in_flight)])],
            unit="{request}",
            description="Outbound calls through client.http_client.request that are still waiting for their response headers.",
        )
        meter.create_observable_counter(
            DRIVER_OFFERED_INSTRUMENT,
            callbacks=[_driver_total(lambda driver: driver.offered)],
            unit="{rollout}",
            description="Rollouts the driver handed to dispatch.",
        )
        meter.create_observable_up_down_counter(
            DRIVER_RESIDENT_INSTRUMENT,
            callbacks=[_driver_total(lambda driver: driver.resident)],
            unit="{rollout}",
            description="Rollout tasks that exist and have not finished, whether waiting for admission or running.",
        )
        meter.create_observable_up_down_counter(
            DRIVER_ACTIVE_INSTRUMENT,
            callbacks=[_driver_total(lambda driver: driver.active)],
            unit="{rollout}",
            description="Rollouts admitted past the driver's concurrency limit and running.",
        )
        self.admission = meter.create_histogram(
            DRIVER_ADMISSION_INSTRUMENT,
            unit="ms",
            description="Time one rollout waited for the driver's concurrency limit before it started.",
            explicit_bucket_boundaries_advisory=list(DRIVER_DURATION_BOUNDARIES_MS),
        )
        self.write_duration = meter.create_histogram(
            DRIVER_WRITE_DURATION_INSTRUMENT,
            unit="ms",
            description="Time to write and flush one rollout's output row.",
            explicit_bucket_boundaries_advisory=list(DRIVER_DURATION_BOUNDARIES_MS),
        )
        self.write_bytes = meter.create_histogram(
            DRIVER_WRITE_BYTES_INSTRUMENT,
            unit="By",
            description="Size of one rollout's output row.",
            explicit_bucket_boundaries_advisory=list(DRIVER_WRITE_BYTES_BOUNDARIES),
        )
        self.materialize = meter.create_histogram(
            DRIVER_MATERIALIZE_INSTRUMENT,
            unit="ms",
            description="Time to read, repeat, and write a run's input rows before dispatch.",
            explicit_bucket_boundaries_advisory=list(DRIVER_DURATION_BOUNDARIES_MS),
        )
        for attribute, name, description in (
            ("input_items", ROLLOUT_INPUT_ITEMS_INSTRUMENT, "Input items one completed rollout started with."),
            (
                "output_items",
                ROLLOUT_OUTPUT_ITEMS_INSTRUMENT,
                "Output items in one completed rollout's final response.",
            ),
            ("turns", ROLLOUT_TURNS_INSTRUMENT, "Agent turns in one completed rollout, from its ng_perf."),
            ("tool_calls", ROLLOUT_TOOL_CALLS_INSTRUMENT, "Tool calls in one completed rollout."),
        ):
            setattr(
                self,
                attribute,
                meter.create_histogram(
                    name,
                    unit="{item}",
                    description=description,
                    explicit_bucket_boundaries_advisory=list(ROLLOUT_COUNT_BOUNDARIES),
                ),
            )
        meter.create_observable_counter(
            GC_COLLECTIONS_INSTRUMENT,
            callbacks=[_by_generation(lambda state: state.gc.collections)],
            unit="{collection}",
            description="Garbage collections, by generation.",
        )
        meter.create_observable_counter(
            GC_COLLECTED_INSTRUMENT,
            callbacks=[_by_generation(lambda state: state.gc.collected)],
            unit="{object}",
            description="Objects the garbage collector freed, by generation.",
        )
        meter.create_observable_counter(
            GC_UNCOLLECTABLE_INSTRUMENT,
            callbacks=[_by_generation(lambda state: state.gc.uncollectable)],
            unit="{object}",
            description="Objects the garbage collector found but could not free, by generation.",
        )
        meter.create_observable_gauge(
            GC_COUNT_INSTRUMENT,
            callbacks=[_by_generation(lambda _state: list(gc.get_count()))],
            unit="{object}",
            description="Allocations counted toward the next collection of each generation, from gc.get_count().",
        )
        meter.create_observable_counter(
            PROCESS_CPU_TIME_INSTRUMENT, callbacks=[_observe(_cpu_times)], unit="s", description="Process CPU time."
        )
        meter.create_observable_gauge(
            PROCESS_MEMORY_USAGE_INSTRUMENT,
            callbacks=[_observe(lambda state: [_observation(state.process.memory_info().rss)])],
            unit="By",
            description="Resident memory of this process.",
        )
        meter.create_observable_gauge(
            PROCESS_THREAD_COUNT_INSTRUMENT,
            callbacks=[_observe(lambda state: [_observation(state.process.num_threads())])],
            unit="{thread}",
            description="Threads in this process.",
        )
        meter.create_observable_gauge(
            PROCESS_OPEN_FDS_INSTRUMENT,
            # psutil has no descriptor count on Windows, where this reports nothing.
            callbacks=[_observe(lambda state: [_observation(state.process.num_fds())])],
            unit="{file_descriptor}",
            description="Open file descriptors in this process.",
        )


#: Instruments per meter, so a meter that is garbage collected takes its instruments with it.
_INSTRUMENTS: "weakref.WeakKeyDictionary[Any, _Instruments]" = weakref.WeakKeyDictionary()


class DriverCounts:
    """Rollout totals that the driver updates with plain integer arithmetic, read by the exporter thread."""

    def __init__(self) -> None:
        self.offered = 0
        """Rollouts handed to dispatch."""

        self.resident = 0
        """Rollout tasks that exist and have not finished, whether waiting for admission or running."""

        self.active = 0
        """Rollouts admitted past the concurrency limit and running."""


#: Milliseconds. Admission can wait for minutes behind a small concurrency limit, and writing one row takes microseconds.
DRIVER_DURATION_BOUNDARIES_MS: tuple[float, ...] = (
    0.1,
    1,
    10,
    100,
    500,
    1000,
    5000,
    10000,
    30000,
    60000,
    300000,
    1800000,
)
#: Counts of items, turns, or calls in one rollout, from a single message to a very long agent trajectory.
ROLLOUT_COUNT_BOUNDARIES: tuple[float, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096)
#: Bytes. A row ranges from a short answer to a long trajectory with token IDs.
DRIVER_WRITE_BYTES_BOUNDARIES: tuple[float, ...] = (1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000)


class _ProcessObserver:
    """The per-process state of the observer: the GC recorder, the request totals, and this process's handle."""

    def __init__(self, meter: Any) -> None:
        import psutil

        instruments = _INSTRUMENTS.get(meter)
        if instruments is None:
            instruments = _INSTRUMENTS[meter] = _Instruments(meter)
        self.process = psutil.Process()
        self.gc = _GcRecorder()
        self.requests = RequestCounts()
        self.driver = DriverCounts()
        self.admission = instruments.admission
        self.write_duration = instruments.write_duration
        self.write_bytes = instruments.write_bytes
        self.materialize = instruments.materialize
        self.input_items = instruments.input_items
        self.output_items = instruments.output_items
        self.turns = instruments.turns
        self.tool_calls = instruments.tool_calls
        self.lag = instruments.lag
        self.tasks = instruments.tasks
        self.gc_pause = instruments.gc_pause

    def record_gc_pauses(self) -> None:
        for generation, seconds in self.gc.drain_pauses():
            self.gc_pause.record(seconds * 1000.0, attributes={GC_GENERATION_ATTRIBUTE: generation})


_PROCESS_LOCK = threading.Lock()
_PROCESS_OBSERVER: Optional[_ProcessObserver] = None
_ACTIVE_LOOPS = 0


def request_counts() -> Optional[RequestCounts]:
    """Return the request totals while an observer is running, else None.

    Server middleware calls this once per request, so it is a plain global read.
    """
    state = _PROCESS_OBSERVER
    return state.requests if state is not None else None


def driver_metrics() -> Optional[_ProcessObserver]:
    """Return the process's observer state while an observer is running, for the rollout driver, else None."""
    return _PROCESS_OBSERVER


class MeteredAdmission:
    """Wraps the driver's concurrency limit to record each rollout's wait and count the rollouts running inside it."""

    def __init__(self, semaphore: Any, observer: _ProcessObserver) -> None:
        self._semaphore = semaphore
        self._observer = observer

    async def __aenter__(self) -> None:
        queued_at = time.perf_counter()
        await self._semaphore.__aenter__()
        self._observer.admission.record((time.perf_counter() - queued_at) * 1000.0)
        self._observer.driver.active += 1

    async def __aexit__(self, *exc_info: Any) -> Any:
        self._observer.driver.active -= 1
        return await self._semaphore.__aexit__(*exc_info)


async def count_resident(awaitable: Any, observer: _ProcessObserver) -> Any:
    """Await one rollout, counting it as resident until it finishes."""
    observer.driver.resident += 1
    try:
        return await awaitable
    finally:
        observer.driver.resident -= 1


def _acquire_process_observer(meter: Any) -> _ProcessObserver:
    global _PROCESS_OBSERVER, _ACTIVE_LOOPS
    with _PROCESS_LOCK:
        if _PROCESS_OBSERVER is None:
            state = _ProcessObserver(meter)
            gc.callbacks.append(state.gc)
            _PROCESS_OBSERVER = state
        _ACTIVE_LOOPS += 1
        return _PROCESS_OBSERVER


def _flush_metrics() -> None:
    try:
        from opentelemetry import metrics

        force_flush = getattr(metrics.get_meter_provider(), "force_flush", None)
        if force_flush is not None:
            force_flush(timeout_millis=5_000)
    except Exception:
        logger.debug("nemo-lens: could not export the final observer metrics", exc_info=True)


def _release_process_observer() -> None:
    global _PROCESS_OBSERVER, _ACTIVE_LOOPS
    with _PROCESS_LOCK:
        _ACTIVE_LOOPS -= 1
        if _ACTIVE_LOOPS > 0 or _PROCESS_OBSERVER is None:
            return
        # The observable totals report only while an observer runs.
        # A short-lived process, such as a driver whose run ends before the first periodic export, would never export them.
        # Exporting once here, while the totals still exist, sends their final values.
        _flush_metrics()
        # Remove only this module's callback object, never another library's.
        try:
            gc.callbacks.remove(_PROCESS_OBSERVER.gc)
        except ValueError:
            pass
        _PROCESS_OBSERVER = None


async def _sample(state: _ProcessObserver, interval_seconds: float, loop_name: str, count_tasks: bool) -> None:
    loop = asyncio.get_running_loop()
    attributes = {EVENT_LOOP_ATTRIBUTE: loop_name}
    while True:
        deadline = time.monotonic() + interval_seconds
        await asyncio.sleep(interval_seconds)
        try:
            state.lag.record(max(0.0, time.monotonic() - deadline) * 1000.0, attributes=attributes)
            if count_tasks:
                # Walks every task on the loop, so it is opt-in: see `TelemetryConfig.event_loop_task_count`.
                state.tasks.set(len(asyncio.all_tasks(loop)), attributes=attributes)
            state.record_gc_pauses()
        except Exception:
            logger.debug("nemo-lens: event-loop sample failed", exc_info=True)


def observer_settings() -> Optional[tuple[float, bool]]:
    """Return the sample interval and whether to count tasks when metrics are exporting in this process, else None."""
    from nemo_gym.telemetry.setup import event_loop_settings_from_env, get_telemetry, is_metrics_exporter_active

    if get_telemetry() is None or not is_metrics_exporter_active():
        return None
    return event_loop_settings_from_env()


@asynccontextmanager
async def observe_event_loop(loop_name: str) -> AsyncIterator[None]:
    """Sample the running event loop, and observe this process, for as long as the context is open.

    ``loop_name`` is a bounded label for the loop, such as ``server``, ``head_server``, or ``driver``.
    Does nothing unless telemetry metrics are exporting.
    """
    sampler: Optional[asyncio.Task] = None
    acquired = False
    try:
        settings = observer_settings()
        if settings is not None:
            from nemo_gym.telemetry.setup import get_telemetry

            interval_seconds, count_tasks = settings
            state = _acquire_process_observer(get_telemetry().meter)
            acquired = True
            sampler = asyncio.create_task(
                _sample(state, interval_seconds, loop_name, count_tasks), name="nemo-gym-event-loop-sampler"
            )
    except Exception:
        logger.warning("nemo-lens: the process observer could not start; continuing without it", exc_info=True)
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
            _release_process_observer()


def observe_app_event_loop(app: Any, loop_name: str) -> None:
    """Run :func:`observe_event_loop` for the lifetime of a FastAPI app, keeping any lifespan the app already has."""
    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: Any) -> AsyncIterator[Any]:
        async with observe_event_loop(loop_name):
            async with original_lifespan(application) as state:
                yield state

    app.router.lifespan_context = lifespan
