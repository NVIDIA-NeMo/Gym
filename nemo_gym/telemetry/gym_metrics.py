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
"""Attributed OTel instruments that ``nemo.lens.instruments.gym.record_gym_metrics``
cannot express.

``nemo_gym.telemetry.metrics`` forwards to exactly five fixed, **undimensioned**
instruments at the pinned nemo-lens commit — see that module's docstring. There is no
kwarg on ``record_gym_metrics`` for a new instrument name, and none of them can carry
attributes, which is disqualifying for the metrics this module defines: an undimensioned
queue-wait histogram mixing the rollout driver's semaphore with a judge's would collapse
into the same "answers no question anyone has" failure mode ``metrics.py`` already
rejects for ``gym.server.request_duration_ms``.

This mirrors the precedent set by :mod:`nemo_gym.telemetry.spans` (``client_span``),
which creates a ``SpanKind.CLIENT`` span directly against the OTel API rather than wait
for ``managed_span`` to grow a ``kind=`` parameter. Same move, for metrics: each
instrument here is created directly via ``meter.create_histogram`` /
``meter.create_counter`` and cached per meter. Delete this module in favor of an
attributed ``record_gym_metrics`` if nemo-lens ever grows one.

Every function here is a no-op unless telemetry is initialised *and* exporting, so call
sites do not need their own correctness guard — though, per every other instrumentation
site in Gym, they should still sit under a span-group gate to stay free when disabled.
"""

import logging
import threading
from typing import Any, Optional


logger = logging.getLogger(__name__)

_INSTRUMENT_LOCK = threading.Lock()
#: Instrument cache keyed by ``id(meter)`` so a test that installs a fresh meter (as
#: ``tests/unit_tests/telemetry/test_metrics.py`` does) does not see stale handles bound
#: to a previous provider.
_INSTRUMENTS: dict[int, dict[str, Any]] = {}


def _meter() -> Optional[Any]:
    from nemo_gym.telemetry.setup import get_telemetry

    telemetry = get_telemetry()
    if telemetry is None or not telemetry.is_exporting:
        return None
    try:
        return telemetry.meter
    except Exception:
        logger.debug("nemo-lens: failed to resolve the meter", exc_info=True)
        return None


def _get_or_create(meter: Any, name: str, factory) -> Any:
    key = id(meter)
    with _INSTRUMENT_LOCK:
        bucket = _INSTRUMENTS.setdefault(key, {})
        instrument = bucket.get(name)
        if instrument is None:
            instrument = factory()
            bucket[name] = instrument
        return instrument


def _record_histogram(name: str, unit: str, description: str, value: float, attributes: dict) -> None:
    meter = _meter()
    if meter is None:
        return
    try:
        instrument = _get_or_create(
            meter, name, lambda: meter.create_histogram(name, unit=unit, description=description)
        )
        instrument.record(value, attributes=attributes)
    except Exception:
        logger.debug("nemo-lens: failed to record %s", name, exc_info=True)


def _record_counter(name: str, description: str, attributes: dict, amount: int = 1) -> None:
    meter = _meter()
    if meter is None:
        return
    try:
        instrument = _get_or_create(meter, name, lambda: meter.create_counter(name, unit="1", description=description))
        instrument.add(amount, attributes=attributes)
    except Exception:
        logger.debug("nemo-lens: failed to record %s", name, exc_info=True)


def _record_gauge(name: str, unit: str, description: str, value: float, attributes: dict) -> None:
    meter = _meter()
    if meter is None:
        return
    try:
        instrument = _get_or_create(meter, name, lambda: meter.create_gauge(name, unit=unit, description=description))
        instrument.set(value, attributes=attributes)
    except Exception:
        logger.debug("nemo-lens: failed to record %s", name, exc_info=True)


def record_queue_wait(duration_ms: float, *, site: str) -> None:
    """Record time spent waiting to acquire a concurrency-limiting semaphore.

    ``site`` identifies which semaphore (``"rollout_driver"``, ``"model.<name>"``,
    ``"resources.<name>"``, ``"agent.<name>"``, ``"sandbox.<provider>"``, ...) so a
    dashboard can tell "the rollout driver is queueing" from "a judge server is
    queueing" instead of averaging every semaphore in the fleet into one number.
    """
    _record_histogram(
        "gym.concurrency.queue_wait_duration_ms",
        "ms",
        "Time spent waiting to acquire a Gym concurrency-limiting semaphore.",
        duration_ms,
        {"nemo.gym.concurrency.site": site},
    )


def record_rollout_completed(*, outcome: str) -> None:
    """Increment ``gym.rollout.completed_total``. ``outcome`` is ``success`` or ``failure``.

    A counter, not a rate: throughput and failure rate are both derivable downstream
    (``rate(...)`` in the metrics backend) from this one instrument, the same tradeoff
    ``gym.verify.success_rate`` hit but resolved the other way — that one is a lens gauge
    forced to flatten to a cumulative fraction; this is a local instrument, so it can just
    be a counter.
    """
    _record_counter(
        "gym.rollout.completed_total",
        "Count of completed rollouts by outcome.",
        {"nemo.gym.rollout.outcome": outcome},
    )


def record_sandbox_startup(duration_ms: float, *, provider: str) -> None:
    """Record one sandbox's provisioning time, attributed by provider."""
    _record_histogram(
        "gym.sandbox.startup_duration_ms",
        "ms",
        "Wall-clock time to provision one sandbox.",
        duration_ms,
        {"nemo.gym.sandbox.provider": provider},
    )


def record_sandbox_create_retry(*, provider: str) -> None:
    """Increment ``gym.sandbox.create_retry_total`` for one sandbox-create retry attempt."""
    _record_counter(
        "gym.sandbox.create_retry_total",
        "Count of sandbox-create retry attempts by provider.",
        {"nemo.gym.sandbox.provider": provider},
    )


def record_model_call_duration(duration_ms: float, *, dialect: str, server_name: Optional[str]) -> None:
    """Record one model-server endpoint call's duration, attributed by dialect and server."""
    _record_histogram(
        "gym.model.call_duration_ms",
        "ms",
        "Wall-clock duration of one model-server dialect call.",
        duration_ms,
        {"nemo.gym.model.dialect": dialect, "nemo.gym.server.name": server_name or ""},
    )


def record_model_time_to_first_byte(duration_ms: float, *, dialect: str, server_name: Optional[str]) -> None:
    """Record time to first response byte for one model-server call."""
    _record_histogram(
        "gym.model.time_to_first_byte_ms",
        "ms",
        "Time to first response byte for one model-server call.",
        duration_ms,
        {"nemo.gym.model.dialect": dialect, "nemo.gym.server.name": server_name or ""},
    )


def record_http_timeout(*, internal: bool) -> None:
    """Increment ``gym.http.timeout_total`` for one aiohttp client timeout."""
    _record_counter(
        "gym.http.timeout_total",
        "Count of outbound HTTP calls that timed out.",
        {"nemo.gym.http.internal": internal},
    )


def record_retry(*, reason: str) -> None:
    """Increment ``gym.http.retry_total``. ``reason`` is one of ``server_disconnected``,
    ``client_os_error``, ``timeout``, ``other``."""
    _record_counter(
        "gym.http.retry_total",
        "Count of outbound HTTP request retries by reason.",
        {"nemo.gym.http.retry_reason": reason},
    )


def record_process_cpu_percent(value: float) -> None:
    """This process's CPU utilization (0-100 per logical core; can exceed 100 on a
    multi-threaded workload -- normalize against the ``nemo.gym.host.cpu_count``
    resource attribute for percent-of-node-capacity).

    Call this from *inside* the active span's context (i.e. within the
    ``with managed_span(...) as span:`` block that produced the reading) so the OTel
    SDK's exemplar mechanism can attach that span's trace/span id to this data point.
    Calling it outside a span context still records the gauge, just without a linked
    exemplar. No call-site attributes: process identity is already carried by every
    instrument this process emits via resource attributes
    (``nemo.gym.server.name``/``nemo.gym.server.type``, see
    ``telemetry.setup._build_resource_attributes``).
    """
    _record_gauge(
        "gym.process.cpu.percent",
        "%",
        "This process's CPU utilization, sampled inline at span boundaries.",
        value,
        {},
    )


def record_process_gpu_utilization(value: float, *, index: int, uuid: str) -> None:
    """One GPU's SM utilization (0-100), sampled periodically via ``nvidia-smi`` on a
    background thread (see :mod:`nemo_gym.telemetry.gpu`) -- unlike the CPU gauge, this
    carries no exemplar, since GPU compute often happens in a different process from the
    one holding the active span. Attributed by ``nemo.gym.gpu.index`` (human-readable,
    can shift across driver re-enumeration) and ``nemo.gym.gpu.uuid`` (stable), since one
    process can see more than one GPU and they must stay distinguishable from each
    other, not just from other processes."""
    _record_gauge(
        "gym.process.gpu.utilization_percent",
        "%",
        "GPU compute (SM) utilization, sampled periodically via nvidia-smi.",
        value,
        {"nemo.gym.gpu.index": index, "nemo.gym.gpu.uuid": uuid},
    )


def record_process_gpu_memory_used_mib(value: float, *, index: int, uuid: str) -> None:
    """One GPU's used memory in MiB, as reported by ``nvidia-smi``. See
    :func:`record_process_gpu_utilization` for the attribute/exemplar reasoning."""
    _record_gauge(
        "gym.process.gpu.memory_used_mib",
        "MiB",
        "GPU memory in use, sampled periodically via nvidia-smi.",
        value,
        {"nemo.gym.gpu.index": index, "nemo.gym.gpu.uuid": uuid},
    )


def record_process_gpu_memory_total_mib(value: float, *, index: int, uuid: str) -> None:
    """One GPU's total memory in MiB, as reported by ``nvidia-smi``. Reported as its own
    gauge rather than folded into a used/total ratio: ``nvidia-smi`` already gives both
    numbers natively, and a dashboard can derive a ratio from the pair, but not recover
    the pair from a ratio alone -- keep the more expressive representation. See
    :func:`record_process_gpu_utilization` for the attribute/exemplar reasoning."""
    _record_gauge(
        "gym.process.gpu.memory_total_mib",
        "MiB",
        "GPU total memory, sampled periodically via nvidia-smi.",
        value,
        {"nemo.gym.gpu.index": index, "nemo.gym.gpu.uuid": uuid},
    )


def _reset_for_testing() -> None:
    """Drop cached instruments. Test-only, mirrors ``telemetry.setup._reset_for_testing``."""
    with _INSTRUMENT_LOCK:
        _INSTRUMENTS.clear()
