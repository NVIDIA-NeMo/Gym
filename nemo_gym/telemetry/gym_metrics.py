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

Model calls
-----------
Recorded by the model server's capture middleware for every exchange it captures, so they
need ``observability_enabled`` (``gym eval submit`` turns it on). They carry the aggregate
the per-rollout capture files hold in full, keyed by model server and dialect, and ship on the
metrics export interval while the run is in progress.

``gym.model_call.duration_ms`` (histogram): wall-clock of one call, success or failure, with
``nemo.gym.model_call.outcome`` set to ``ok`` or the capture's ``error_category``. Its count is the
call count; filter the outcome for error rates.

``gym.model_call.ttft_ms`` (histogram): time to the first streamed chunk. Streamed responses
only: for a JSON response the first chunk is the whole body and would read as the total latency.

``gym.model_call.tokens`` (histogram): token usage from the response, one sample per
``nemo.gym.model_call.token.type`` (``input``, ``output``, ``reasoning``) the provider reported.
The sum is the token volume; the distribution is the prompt or completion size per call.

``gym.model_call.finish_total`` (counter): responses by ``nemo.gym.model_call.finish_reason``,
in the dialect's own vocabulary (``stop`` / ``length`` / ``tool_calls`` for Chat Completions,
``completed`` or the ``incomplete_details.reason`` for Responses, ``end_turn`` / ``max_tokens`` /
``tool_use`` for Messages). ``length``-class values are the truncation signal.

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
MODEL_CALL_DURATION_INSTRUMENT = "gym.model_call.duration_ms"
MODEL_CALL_TTFT_INSTRUMENT = "gym.model_call.ttft_ms"
MODEL_CALL_TOKENS_INSTRUMENT = "gym.model_call.tokens"
MODEL_CALL_FINISH_INSTRUMENT = "gym.model_call.finish_total"
#: Same key as the model server's span attribute, so span- and metric-derived series share the label.
MODEL_CALL_SERVER_NAME_ATTRIBUTE = "nemo.gym.server.name"
MODEL_CALL_DIALECT_ATTRIBUTE = "nemo.gym.model_call.dialect"
MODEL_CALL_OUTCOME_ATTRIBUTE = "nemo.gym.model_call.outcome"
MODEL_CALL_TOKEN_TYPE_ATTRIBUTE = "nemo.gym.model_call.token.type"
MODEL_CALL_FINISH_REASON_ATTRIBUTE = "nemo.gym.model_call.finish_reason"
MODEL_CALL_OUTCOME_OK = "ok"

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

#: Milliseconds. A reasoning model's call runs for minutes under load; the SDK default ends at 10 s.
MODEL_CALL_DURATION_BOUNDARIES_MS: tuple[float, ...] = (
    100,
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

#: Milliseconds. Time to first chunk is sub-second when the engine has room and tens of seconds when queued.
MODEL_CALL_TTFT_BOUNDARIES_MS: tuple[float, ...] = (
    50,
    100,
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
)

#: Tokens. Spans a short completion to a prompt near a million-token context window.
MODEL_CALL_TOKEN_BOUNDARIES: tuple[float, ...] = (
    16,
    64,
    256,
    1_024,
    4_096,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
    524_288,
    1_048_576,
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


def metrics_exporting() -> bool:
    """True when a recorder below would reach an exporting meter. The call-site gate for work done only for metrics."""
    return _meter() is not None


def record_model_call(
    *,
    server_name: Optional[str],
    dialect: Optional[str],
    latency_ms: Optional[float],
    ttft_ms: Optional[float],
    error_category: Optional[str],
    finish_reason: Optional[str],
    tokens_in: Optional[int],
    tokens_out: Optional[int],
    tokens_reasoning: Optional[int],
) -> None:
    """Record one captured model call: duration with outcome, TTFT, token usage and finish reason.

    ``ttft_ms`` is recorded as given; the caller passes ``None`` for a non-streamed response.
    Token types and the finish reason are recorded only when the provider reported them, so an
    absent count is unknown rather than zero.
    """
    identity = {
        MODEL_CALL_SERVER_NAME_ATTRIBUTE: server_name or "unknown",
        MODEL_CALL_DIALECT_ATTRIBUTE: dialect or "unknown",
    }
    if latency_ms is not None:
        _record_histogram(
            MODEL_CALL_DURATION_INSTRUMENT,
            "ms",
            "Wall-clock time of one model call at a Gym model server, success or failure.",
            latency_ms,
            {**identity, MODEL_CALL_OUTCOME_ATTRIBUTE: error_category or MODEL_CALL_OUTCOME_OK},
            boundaries=MODEL_CALL_DURATION_BOUNDARIES_MS,
        )
    if ttft_ms is not None:
        _record_histogram(
            MODEL_CALL_TTFT_INSTRUMENT,
            "ms",
            "Time to the first streamed chunk of one model call.",
            ttft_ms,
            identity,
            boundaries=MODEL_CALL_TTFT_BOUNDARIES_MS,
        )
    for token_type, count in (("input", tokens_in), ("output", tokens_out), ("reasoning", tokens_reasoning)):
        if count is None:
            continue
        _record_histogram(
            MODEL_CALL_TOKENS_INSTRUMENT,
            "{token}",
            "Token usage one model call reported, by token type.",
            count,
            {**identity, MODEL_CALL_TOKEN_TYPE_ATTRIBUTE: token_type},
            boundaries=MODEL_CALL_TOKEN_BOUNDARIES,
        )
    if finish_reason is not None:
        _record_counter(
            MODEL_CALL_FINISH_INSTRUMENT,
            "Model responses by the finish reason the provider reported.",
            {**identity, MODEL_CALL_FINISH_REASON_ATTRIBUTE: finish_reason},
        )


def _reset_for_testing() -> None:
    """Drop cached instruments. Test-only."""
    with _INSTRUMENT_LOCK:
        _INSTRUMENTS.clear()
