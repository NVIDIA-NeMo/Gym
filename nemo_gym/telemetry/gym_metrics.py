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
and never raises into its caller; call sites still sit under a span-group gate so they cost
nothing when disabled.

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

Rollout outcomes
----------------
``gym.rollout.completed_total`` (counter): one per rollout the driver finished handling,
recorded where the final verdict is known, after the agent's response has been inspected.
``nemo.gym.rollout.outcome`` is ``scored`` (persisted to the main output) or ``dropped``
(left out of the score); dropped rollouts also carry Gym's ``nemo.gym.failure_class`` and,
when it is a bare identifier such as an exception class name, ``nemo.gym.failure_reason``.
A ``/run`` that answered 200 with an infrastructure error inside is ``dropped`` here, which is
why this is not recorded around the HTTP call.
"""

import logging
import re
import threading
from collections.abc import Callable, Sequence
from typing import Any, Optional


logger = logging.getLogger(__name__)

SANDBOX_PROVIDER_ATTRIBUTE = "nemo.gym.sandbox.provider"
SANDBOX_ACTIVE_INSTRUMENT = "gym.sandbox.active"
SANDBOX_STARTUP_INSTRUMENT = "gym.sandbox.startup_duration_ms"
SANDBOX_EXEC_INSTRUMENT = "gym.sandbox.exec_duration_ms"
SANDBOX_CREATE_RETRY_INSTRUMENT = "gym.sandbox.create_retry_total"

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

ROLLOUT_COMPLETED_INSTRUMENT = "gym.rollout.completed_total"
ROLLOUT_OUTCOME_ATTRIBUTE = "nemo.gym.rollout.outcome"
FAILURE_CLASS_ATTRIBUTE = "nemo.gym.failure_class"
FAILURE_REASON_ATTRIBUTE = "nemo.gym.failure_reason"
#: A reason is kept as an attribute only when it looks like an identifier (an exception class
#: name); free-text error messages would fan the series out without bound.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,79}")

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


def record_rollout_completed(
    outcome: str, *, failure_class: str | None = None, failure_reason: str | None = None
) -> None:
    """Count one finished rollout in ``gym.rollout.completed_total``.

    ``outcome`` is ``"scored"`` or ``"dropped"``. For a dropped rollout pass Gym's failure
    class and, if known, the failure reason; the reason is recorded only when it is a bare
    identifier.
    """
    attributes: dict[str, Any] = {ROLLOUT_OUTCOME_ATTRIBUTE: outcome}
    if failure_class:
        attributes[FAILURE_CLASS_ATTRIBUTE] = failure_class
    if failure_reason and _IDENTIFIER.fullmatch(failure_reason):
        attributes[FAILURE_REASON_ATTRIBUTE] = failure_reason
    _record_counter(ROLLOUT_COMPLETED_INSTRUMENT, "Rollouts the driver finished handling, by outcome.", attributes)


def _reset_for_testing() -> None:
    """Drop cached instruments. Test-only."""
    with _INSTRUMENT_LOCK:
        _INSTRUMENTS.clear()
