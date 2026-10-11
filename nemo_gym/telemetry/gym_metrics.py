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

"""Attributed OTel instruments that ``nemo.lens.instruments.gym.record_gym_metrics`` cannot express.

``nemo_gym.telemetry.metrics`` forwards to the five fixed, undimensioned instruments nemo-lens
declares. Anything that needs an attribute (a provider, a site, a class) is created here,
directly on the lens meter, and cached per meter so a re-initialised telemetry handle gets
fresh instruments. Every recorder is a no-op unless telemetry is initialised and exporting,
and never raises into its caller. Call sites use the relevant span-group or metrics-export gate
so disabled telemetry stays off the hot path.

Sandbox lifecycle
-----------------
``gym.sandbox.active`` (up-down counter): sandboxes this process currently holds, ``+1`` when a
provider hands back a handle, ``-1`` when Gym releases it. Summed by the backend across the
processes that hold sandboxes; a gauge would show whichever process wrote last.

``gym.sandbox.startup_duration_ms`` / ``gym.sandbox.exec_duration_ms`` (histograms): wall-clock
of one provisioning and one command. Explicit bucket boundaries up to thirty minutes: the SDK
default stops at ten seconds and a sandbox start routinely takes a minute.

``gym.sandbox.create_retry_total`` (counter): one per create attempt a provider retried.
Retrying is provider-internal, so each provider that retries records it from its own loop.

All four carry ``nemo.gym.sandbox.provider``.

HTTP connection pool
--------------------
``gym.http.connection_pool.queue_duration_ms`` (histogram): connection-acquisition wait
for one queued connection acquisition, combining repeated waiter wakeups within it.
Queued redirect hops, aiohttp reconnects, and Gym retries each record separate samples.
Connection attempts that acquire a slot immediately do not record a queue-duration histogram sample.
Bounded attributes identify the binding connector limit, queue outcome, and destination server.

``gym.http.connection_pool.connect_total`` (observable counter): all connection attempts
(``connect()`` calls), including attempts that later fail or are abandoned.
Compare its value with the queue-duration histogram count to calculate the queued fraction.

Rollout outcomes
----------------
``gym.rollout.completed_total`` (counter): one per rollout attempt the collection driver finished
handling, recorded after it wrote the attempt or decided to omit it. An attempt is one dispatch of
one row in one collection run; a retry on a later resume run is another attempt, and an HTTP-level
re-send of ``/run`` inside one dispatch is not. Only ``run_from_config`` (``gym eval run`` with
serving) records it: ``--no-serve`` initialises no telemetry, and ``run_examples`` callers,
reverification and the GDPVal multistage driver do not call the recorder.

``nemo.gym.rollout.outcome`` is ``scored`` (main output, masked and forced-zero results included),
``failed`` (failures sidecar) or ``omitted`` (no-persist, e.g. kill-shaped). Rows drained by the
dispatch budget never ran and are not counted, unlike the progress metrics, whose ``dropped``
groups ``failed`` with an ``omitted`` that includes drained rows. A failed ``/run`` is a ``failed``
attempt with class ``agent_request_failed`` only with ``route_failures_to_sidecar`` on; otherwise
it aborts the run and is not counted. Coverage comes from the output files, not this counter.

Attributes are bounded: ``nemo.gym.rollout.dispatch.name`` (the environment server, or the agent
when rows name no environment server) and, on attempts that carry a failure class,
``nemo.gym.failure_class``, ``nemo.gym.failure_kind`` (registered or ``<server>:<kind>``
namespaced, any other value recorded as ``unregistered``) and ``nemo.gym.failure_type`` (exception
class name). Free-text reasons are never attributes.
"""

import logging
import threading
from collections.abc import Callable, Sequence
from typing import Any, Optional


logger = logging.getLogger(__name__)

SANDBOX_PROVIDER_ATTRIBUTE = "nemo.gym.sandbox.provider"
SANDBOX_ACTIVE_INSTRUMENT = "gym.sandbox.active"
SANDBOX_STARTUP_INSTRUMENT = "gym.sandbox.startup_duration_ms"
SANDBOX_EXEC_INSTRUMENT = "gym.sandbox.exec_duration_ms"
SANDBOX_CREATE_RETRY_INSTRUMENT = "gym.sandbox.create_retry_total"
HTTP_CONNECTION_POOL_QUEUE_DURATION_INSTRUMENT = "gym.http.connection_pool.queue_duration_ms"
HTTP_CONNECTION_POOL_CONNECT_INSTRUMENT = "gym.http.connection_pool.connect_total"
HTTP_CONNECTION_POOL_QUEUE_CONSTRAINT_ATTRIBUTE = "nemo.gym.http.connection_pool.queue_constraint"
HTTP_CONNECTION_POOL_QUEUE_OUTCOME_ATTRIBUTE = "nemo.gym.http.connection_pool.queue_outcome"
HTTP_DESTINATION_SERVER_NAME_ATTRIBUTE = "nemo.gym.http.destination.server.name"
ROLLOUT_COMPLETED_INSTRUMENT = "gym.rollout.completed_total"
ROLLOUT_OUTCOME_ATTRIBUTE = "nemo.gym.rollout.outcome"
FAILURE_CLASS_ATTRIBUTE = "nemo.gym.failure_class"
FAILURE_KIND_ATTRIBUTE = "nemo.gym.failure_kind"
FAILURE_TYPE_ATTRIBUTE = "nemo.gym.failure_type"
DISPATCH_NAME_ATTRIBUTE = "nemo.gym.rollout.dispatch.name"
UNREGISTERED_FAILURE_KIND = "unregistered"

#: Milliseconds. Provisioning a remote sandbox takes tens of seconds and a long command can run
#: for minutes; the SDK's default boundaries end at 10 s and would put most of both in +Inf.
SANDBOX_DURATION_BOUNDARIES_MS: tuple[float, ...] = (
    250,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    30_000,
    60_000,
    120_000,
    300_000,
    600_000,
    1_800_000,
)

HTTP_CONNECTION_POOL_QUEUE_DURATION_BOUNDARIES_MS: tuple[float, ...] = (
    0.01,
    0.05,
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
    1_000,
    5_000,
    30_000,
)

_INSTRUMENT_LOCK = threading.Lock()
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


def _get_or_create(meter: Any, name: str, factory: Callable[[], Any]) -> Any:
    key = id(meter)
    with _INSTRUMENT_LOCK:
        bucket = _INSTRUMENTS.setdefault(key, {})
        instrument = bucket.get(name)
        if instrument is None:
            instrument = factory()
            bucket[name] = instrument
        return instrument


def _record_histogram(
    name: str,
    unit: str,
    description: str,
    value: float,
    attributes: dict[str, Any],
    *,
    boundaries: Sequence[float] | None = None,
) -> None:
    meter = _meter()
    if meter is None:
        return
    try:
        kwargs: dict[str, Any] = {"unit": unit, "description": description}
        if boundaries is not None:
            kwargs["explicit_bucket_boundaries_advisory"] = list(boundaries)
        instrument = _get_or_create(meter, name, lambda: meter.create_histogram(name, **kwargs))
        instrument.record(value, attributes=attributes)
    except Exception:
        logger.debug("nemo-lens: failed to record %s", name, exc_info=True)


def _record_counter(name: str, description: str, attributes: dict[str, Any], amount: int = 1) -> None:
    meter = _meter()
    if meter is None:
        return
    try:
        instrument = _get_or_create(meter, name, lambda: meter.create_counter(name, unit="1", description=description))
        instrument.add(amount, attributes=attributes)
    except Exception:
        logger.debug("nemo-lens: failed to record %s", name, exc_info=True)


def _record_up_down_counter(name: str, unit: str, description: str, delta: int, attributes: dict[str, Any]) -> None:
    meter = _meter()
    if meter is None:
        return
    try:
        instrument = _get_or_create(
            meter, name, lambda: meter.create_up_down_counter(name, unit=unit, description=description)
        )
        instrument.add(delta, attributes=attributes)
    except Exception:
        logger.debug("nemo-lens: failed to record %s", name, exc_info=True)


def record_sandbox_active(delta: int, *, provider: str) -> None:
    """Add ``delta`` (``+1`` on start, ``-1`` on stop) to ``gym.sandbox.active`` for ``provider``."""
    _record_up_down_counter(
        SANDBOX_ACTIVE_INSTRUMENT,
        "{sandbox}",
        "Sandboxes this process currently holds, from provider create to release.",
        delta,
        {SANDBOX_PROVIDER_ATTRIBUTE: provider},
    )


def record_sandbox_startup(duration_ms: float, *, provider: str) -> None:
    """Record one sandbox's provisioning wall-clock, by provider."""
    _record_histogram(
        SANDBOX_STARTUP_INSTRUMENT,
        "ms",
        "Wall-clock time to provision one sandbox.",
        duration_ms,
        {SANDBOX_PROVIDER_ATTRIBUTE: provider},
        boundaries=SANDBOX_DURATION_BOUNDARIES_MS,
    )


def record_sandbox_exec_duration(duration_ms: float, *, provider: str) -> None:
    """Record one command's wall-clock inside an already-provisioned sandbox, by provider."""
    _record_histogram(
        SANDBOX_EXEC_INSTRUMENT,
        "ms",
        "Wall-clock time to run one command inside a sandbox.",
        duration_ms,
        {SANDBOX_PROVIDER_ATTRIBUTE: provider},
        boundaries=SANDBOX_DURATION_BOUNDARIES_MS,
    )


def record_sandbox_create_retry(*, provider: str) -> None:
    """Count one retried sandbox-create attempt, by provider."""
    _record_counter(
        SANDBOX_CREATE_RETRY_INSTRUMENT,
        "Sandbox-create attempts a provider retried.",
        {SANDBOX_PROVIDER_ATTRIBUTE: provider},
    )


def record_http_connection_pool_queue_duration(
    duration_ms: float,
    *,
    queue_constraint: str,
    queue_outcome: str,
    server_name: str,
) -> None:
    """Record one queued connection acquisition."""
    _record_histogram(
        HTTP_CONNECTION_POOL_QUEUE_DURATION_INSTRUMENT,
        "ms",
        "Time one queued connection acquisition waited for an aiohttp pool slot.",
        duration_ms,
        {
            HTTP_CONNECTION_POOL_QUEUE_CONSTRAINT_ATTRIBUTE: queue_constraint,
            HTTP_CONNECTION_POOL_QUEUE_OUTCOME_ATTRIBUTE: queue_outcome,
            HTTP_DESTINATION_SERVER_NAME_ATTRIBUTE: server_name,
        },
        boundaries=HTTP_CONNECTION_POOL_QUEUE_DURATION_BOUNDARIES_MS,
    )


def register_http_connection_pool_connect_counter(snapshot: Callable[[], dict[str, int]]) -> None:
    """Export cumulative connection-attempt counts without an OTel call on each connect."""
    meter = _meter()
    if meter is None:
        return

    def observe(_options: Any) -> list[Any]:
        from opentelemetry.metrics import Observation

        return [
            Observation(count, {HTTP_DESTINATION_SERVER_NAME_ATTRIBUTE: server_name})
            for server_name, count in snapshot().items()
        ]

    try:
        _get_or_create(
            meter,
            HTTP_CONNECTION_POOL_CONNECT_INSTRUMENT,
            lambda: meter.create_observable_counter(
                HTTP_CONNECTION_POOL_CONNECT_INSTRUMENT,
                callbacks=[observe],
                unit="{connection}",
                description=(
                    "Outbound aiohttp connection attempts (connect() calls), "
                    "including attempts that later fail or are abandoned."
                ),
            ),
        )
    except Exception:
        logger.debug("nemo-lens: failed to register %s", HTTP_CONNECTION_POOL_CONNECT_INSTRUMENT, exc_info=True)


def record_rollout_completed(
    outcome: str,
    *,
    dispatch_name: str,
    failure_class: str | None = None,
    failure_kind: str | None = None,
    failure_type: str | None = None,
) -> None:
    """Count one finished rollout attempt in ``gym.rollout.completed_total``."""
    attributes: dict[str, Any] = {ROLLOUT_OUTCOME_ATTRIBUTE: outcome, DISPATCH_NAME_ATTRIBUTE: dispatch_name}
    if failure_class:
        attributes[FAILURE_CLASS_ATTRIBUTE] = failure_class
    if failure_kind:
        attributes[FAILURE_KIND_ATTRIBUTE] = failure_kind
    if failure_type:
        attributes[FAILURE_TYPE_ATTRIBUTE] = failure_type
    _record_counter(
        ROLLOUT_COMPLETED_INSTRUMENT, "Rollout attempts the driver finished handling, by outcome.", attributes
    )


def _reset_for_testing() -> None:
    """Drop cached instruments. Test-only."""
    with _INSTRUMENT_LOCK:
        _INSTRUMENTS.clear()
