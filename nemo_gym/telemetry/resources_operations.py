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
"""A shared way for resources servers to report bounded work: how long it queued, how long it ran, and how it ended.

Many resources servers limit expensive work, such as a verifier subprocess or a judge call, with a semaphore.
When that limit is too small, verification waits in a queue that no request metric shows, and the server looks slow without looking busy.
Wrapping the limited work in :func:`resources_operation` makes that visible:

    async with resources_operation(self._semaphore, operation="verify", executor="subprocess") as op:
        result = await run_the_verifier()
        if it_timed_out:
            op.outcome = "timeout"

It reports, by ``operation`` and ``executor``:

* ``gym.resources.operation.queue_duration_ms`` (histogram, milliseconds): time waiting for the limit.
* ``gym.resources.operation.duration_ms`` (histogram, milliseconds, also by ``outcome``): time running inside the limit.
  The count of each ``outcome``, such as ``timeout`` or ``crashed``, comes from this histogram.
* ``gym.resources.operation.queued`` and ``gym.resources.operation.active``: how many operations wait and run right now.

``operation`` is one of ``verify``, ``tool``, ``judge``, or ``other``, and ``executor`` is one of ``inline``, ``thread``, ``process``, ``subprocess``, or ``other``, so the attributes stay bounded.
Everything is recorded only while metrics are exporting; otherwise the wrapper only enters the limit.
"""

import weakref
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any, AsyncIterator, Optional


OPERATIONS = frozenset({"verify", "tool", "judge", "other"})
EXECUTORS = frozenset({"inline", "thread", "process", "subprocess", "other"})
OUTCOMES = frozenset({"ok", "error", "timeout", "cancelled", "crashed"})

QUEUE_DURATION_INSTRUMENT = "gym.resources.operation.queue_duration_ms"
DURATION_INSTRUMENT = "gym.resources.operation.duration_ms"
QUEUED_INSTRUMENT = "gym.resources.operation.queued"
ACTIVE_INSTRUMENT = "gym.resources.operation.active"

OPERATION_ATTRIBUTE = "nemo.gym.resources.operation"
EXECUTOR_ATTRIBUTE = "nemo.gym.resources.executor"
OUTCOME_ATTRIBUTE = "nemo.gym.resources.operation.outcome"

#: Milliseconds. A verifier subprocess can wait behind its limit for minutes, and a cached check finishes in a millisecond.
OPERATION_BOUNDARIES_MS: tuple[float, ...] = (
    1,
    5,
    10,
    50,
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


class ResourcesOperation:
    """One limited operation. Set ``outcome`` inside the block when the work ended other than by returning normally."""

    def __init__(self) -> None:
        self.outcome: Optional[str] = None


class _OperationMetrics:
    """The instruments and live counts behind :func:`resources_operation`, one set per meter."""

    def __init__(self, meter: Any) -> None:
        from opentelemetry.metrics import Observation

        self.queued: dict[tuple[str, str], int] = {}
        self.active: dict[tuple[str, str], int] = {}
        self.queue_duration = meter.create_histogram(
            QUEUE_DURATION_INSTRUMENT,
            unit="ms",
            description="Time one resources operation waited for its concurrency limit.",
            explicit_bucket_boundaries_advisory=list(OPERATION_BOUNDARIES_MS),
        )
        self.duration = meter.create_histogram(
            DURATION_INSTRUMENT,
            unit="ms",
            description="Time one resources operation ran inside its concurrency limit, by how it ended.",
            explicit_bucket_boundaries_advisory=list(OPERATION_BOUNDARIES_MS),
        )

        def observe(counts: dict[tuple[str, str], int]) -> Any:
            def callback(_options: Any) -> list:
                return [
                    Observation(value, {OPERATION_ATTRIBUTE: operation, EXECUTOR_ATTRIBUTE: executor})
                    for (operation, executor), value in list(counts.items())
                ]

            return callback

        meter.create_observable_up_down_counter(
            QUEUED_INSTRUMENT,
            callbacks=[observe(self.queued)],
            unit="{operation}",
            description="Resources operations waiting for their concurrency limit.",
        )
        meter.create_observable_up_down_counter(
            ACTIVE_INSTRUMENT,
            callbacks=[observe(self.active)],
            unit="{operation}",
            description="Resources operations running inside their concurrency limit.",
        )


#: Instruments per meter, so a meter that is garbage collected takes its instruments with it.
_METRICS: "weakref.WeakKeyDictionary[Any, _OperationMetrics]" = weakref.WeakKeyDictionary()


def _metrics() -> Optional[_OperationMetrics]:
    from nemo_gym.telemetry.observer import driver_metrics
    from nemo_gym.telemetry.setup import get_telemetry

    if driver_metrics() is None:
        return None
    telemetry = get_telemetry()
    if telemetry is None:
        return None
    metrics = _METRICS.get(telemetry.meter)
    if metrics is None:
        metrics = _METRICS[telemetry.meter] = _OperationMetrics(telemetry.meter)
    return metrics


@asynccontextmanager
async def resources_operation(limit: Any, *, operation: str, executor: str) -> AsyncIterator[ResourcesOperation]:
    """Enter ``limit``, which is any async context manager such as an ``asyncio.Semaphore``, and report the operation.

    An exception sets the outcome to ``error``, or ``cancelled`` for a cancellation, unless the block already set one.
    """
    op = ResourcesOperation()
    metrics = _metrics()
    if metrics is None:
        async with limit:
            yield op
        return

    key = (operation if operation in OPERATIONS else "other", executor if executor in EXECUTORS else "other")
    attributes = {OPERATION_ATTRIBUTE: key[0], EXECUTOR_ATTRIBUTE: key[1]}
    metrics.queued[key] = metrics.queued.get(key, 0) + 1
    queued_at = perf_counter()
    try:
        await limit.__aenter__()
    finally:
        metrics.queued[key] -= 1
    metrics.queue_duration.record((perf_counter() - queued_at) * 1000.0, attributes=attributes)
    metrics.active[key] = metrics.active.get(key, 0) + 1
    started = perf_counter()
    exc_info: tuple = (None, None, None)
    try:
        yield op
    except BaseException as error:
        import asyncio

        exc_info = (type(error), error, error.__traceback__)
        if op.outcome is None:
            op.outcome = "cancelled" if isinstance(error, asyncio.CancelledError) else "error"
        raise
    finally:
        metrics.active[key] -= 1
        outcome = op.outcome if op.outcome in OUTCOMES else "ok"
        metrics.duration.record(
            (perf_counter() - started) * 1000.0, attributes={**attributes, OUTCOME_ATTRIBUTE: outcome}
        )
        await limit.__aexit__(*exc_info)
