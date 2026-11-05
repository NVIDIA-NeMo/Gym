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
"""The live state of Gym's concurrency limits: the configured limit, the slots in use, and who is waiting.

A queue-wait duration shows that work waited, not why.
A dashboard that shows "limit 100, active 100, queued 75" for each limit answers why:
the limit is full and work is piling up behind it.
:class:`MeteredSemaphore` is a drop-in ``asyncio.BoundedSemaphore`` that keeps those numbers,
and the process observer exports them, by limit:

* ``gym.concurrency.limit``: the configured number of slots.
* ``gym.concurrency.active``: slots held right now.
* ``gym.concurrency.available``: slots free right now, ``limit - active``.
* ``gym.concurrency.queue_depth``: callers waiting for a slot right now.
* ``gym.concurrency.admission_timeouts_total``: waits that ran out of the limit's admission timeout.
  Reported only for a limit that has an admission timeout, because no other limit can time out.
* ``gym.concurrency.admission_cancelled_total``: waits cancelled before a slot was free,
  for example because the caller's own request was cancelled.

Every point carries ``nemo.gym.concurrency.component`` (:data:`ConcurrencyComponent`),
``nemo.gym.concurrency.limit_name`` (a name the code chose for what the limit bounds, such as ``upstream_requests``),
and ``nemo.gym.server.name`` when the limit belongs to a server.
All three are bounded: no rollout, request, or sandbox ID is ever an attribute.
Limits with the same attributes in one process are summed.

A component without a configured limit creates no ``MeteredSemaphore``, so it reports no series at all rather than a limit of zero.
Acquiring and releasing cost a few integer updates and no OpenTelemetry call.
The exporter's background thread reads the totals through observable instruments,
which exist only while the process observer runs, that is, while metrics are exported.
"""

import asyncio
import logging
import threading
import weakref
from typing import Any, Callable, Literal, NamedTuple, Optional

from nemo_gym.telemetry.gym_metrics import SERVER_NAME_ATTRIBUTE


logger = logging.getLogger(__name__)

LIMIT_INSTRUMENT = "gym.concurrency.limit"
ACTIVE_INSTRUMENT = "gym.concurrency.active"
AVAILABLE_INSTRUMENT = "gym.concurrency.available"
QUEUE_DEPTH_INSTRUMENT = "gym.concurrency.queue_depth"
ADMISSION_TIMEOUTS_INSTRUMENT = "gym.concurrency.admission_timeouts_total"
ADMISSION_CANCELLED_INSTRUMENT = "gym.concurrency.admission_cancelled_total"

COMPONENT_ATTRIBUTE = "nemo.gym.concurrency.component"
LIMIT_NAME_ATTRIBUTE = "nemo.gym.concurrency.limit_name"

#: The kind of Gym component that owns a limit.
ConcurrencyComponent = Literal[
    "rollout_driver",
    "agent_server",
    "model_server",
    "resources_server",
    "environment_server",
    "sandbox_provider",
]


class MeteredSemaphore(asyncio.BoundedSemaphore):
    """An ``asyncio.BoundedSemaphore`` that counts its slots in use and its waiters for export.

    Use it wherever a component has a configured concurrency limit; ``async with``, ``acquire``,
    and ``release`` behave as on the base class.

    Args:
        limit: The configured number of slots, at least one.
        component: The kind of component that owns the limit.
        limit_name: A fixed name for what the limit bounds, such as ``rollouts`` or ``upstream_requests``.
        server_name: The owning server's configured name, or None outside a server.
        admission_timeout_s: How long one caller may wait for a slot before ``acquire`` raises
            ``TimeoutError``, or None to wait indefinitely. Pass the timeout here rather than wrapping
            ``acquire`` in ``asyncio.wait_for``, which this class would count as a cancellation.
    """

    def __init__(
        self,
        limit: int,
        *,
        component: ConcurrencyComponent,
        limit_name: str,
        server_name: Optional[str] = None,
        admission_timeout_s: Optional[float] = None,
    ) -> None:
        if limit < 1:
            raise ValueError(f"A concurrency limit needs at least one slot, got {limit} for {limit_name}")
        if admission_timeout_s is not None and admission_timeout_s <= 0:
            raise ValueError(f"admission_timeout_s must be positive, got {admission_timeout_s} for {limit_name}")
        super().__init__(limit)
        self._limit = limit
        self._active = 0
        self._queued = 0
        self._admission_timeout_s = admission_timeout_s
        self._key = _LimitKey(component, limit_name, server_name)
        _register(self)

    @property
    def limit(self) -> int:
        """The configured number of slots."""
        return self._limit

    @property
    def active(self) -> int:
        """Slots held right now."""
        return self._active

    @property
    def queued(self) -> int:
        """Callers waiting for a slot right now."""
        return self._queued

    @property
    def admission_timeout_s(self) -> Optional[float]:
        """How long one caller may wait for a slot, or None to wait indefinitely."""
        return self._admission_timeout_s

    @property
    def attributes(self) -> dict[str, str]:
        """The metric attributes this limit reports under."""
        return self._key.attributes()

    async def acquire(self) -> Literal[True]:
        """Take a slot, waiting for one if all are held.

        Raises:
            TimeoutError: No slot became free within ``admission_timeout_s``.
        """
        if not self.locked():
            # The base class takes a free slot without suspending, so this caller never queued.
            await super().acquire()
            self._active += 1
            return True
        self._queued += 1
        try:
            async with asyncio.timeout(self._admission_timeout_s):
                await super().acquire()
        except TimeoutError:
            _count_admission_failure(_ADMISSION_TIMEOUTS, self._key)
            raise
        except asyncio.CancelledError:
            _count_admission_failure(_ADMISSION_CANCELLED, self._key)
            raise
        finally:
            self._queued -= 1
        self._active += 1
        return True

    def release(self) -> None:
        """Return a slot; raises ``ValueError`` when no slot is held, as ``asyncio.BoundedSemaphore`` does."""
        # asyncio hands a released slot straight to the next waiter, so a second release before that waiter resumes
        # passes the base class's bound check; the held-slot count catches it.
        if self._active <= 0:
            raise ValueError(f"{self._key.limit_name}: released more slots than were held")
        super().release()
        self._active -= 1


def register_concurrency_instruments(meter: Any) -> None:
    """Create the ``gym.concurrency.*`` instruments on ``meter``. The process observer calls this once per meter."""
    for name, read, unit, description in (
        (LIMIT_INSTRUMENT, lambda state: state.limit, "{slot}", "Configured slots of a Gym concurrency limit."),
        (ACTIVE_INSTRUMENT, lambda state: state.active, "{slot}", "Slots of a Gym concurrency limit held right now."),
        (
            AVAILABLE_INSTRUMENT,
            lambda state: state.limit - state.active,
            "{slot}",
            "Slots of a Gym concurrency limit free right now.",
        ),
        (
            QUEUE_DEPTH_INSTRUMENT,
            lambda state: state.queued,
            "{caller}",
            "Callers waiting for a slot of a Gym concurrency limit right now.",
        ),
    ):
        meter.create_observable_up_down_counter(
            name, callbacks=[_observe_limits(read)], unit=unit, description=description
        )
    meter.create_observable_counter(
        ADMISSION_TIMEOUTS_INSTRUMENT,
        callbacks=[_observe_admission_failures(_ADMISSION_TIMEOUTS)],
        unit="{admission}",
        description="Waits for a Gym concurrency limit that ran out of its admission timeout.",
    )
    meter.create_observable_counter(
        ADMISSION_CANCELLED_INSTRUMENT,
        callbacks=[_observe_admission_failures(_ADMISSION_CANCELLED)],
        unit="{admission}",
        description="Waits for a Gym concurrency limit that were cancelled before a slot was free.",
    )


class _LimitKey(NamedTuple):
    component: str
    limit_name: str
    server_name: Optional[str]

    def attributes(self) -> dict[str, str]:
        attributes = {COMPONENT_ATTRIBUTE: self.component, LIMIT_NAME_ATTRIBUTE: self.limit_name}
        if self.server_name is not None:
            attributes[SERVER_NAME_ATTRIBUTE] = self.server_name
        return attributes


class _LimitState(NamedTuple):
    limit: int
    active: int
    queued: int


#: Guards the registry and the failure totals against the exporter thread reading them while a limit is created.
_LOCK = threading.Lock()
_LIMITS: "weakref.WeakSet[MeteredSemaphore]" = weakref.WeakSet()
#: Cumulative per process, so a total survives the limit that counted it.
#: Only keys of limits that have existed are present, so a missing key is never reported as zero.
_ADMISSION_TIMEOUTS: dict[_LimitKey, int] = {}
_ADMISSION_CANCELLED: dict[_LimitKey, int] = {}


def _register(semaphore: MeteredSemaphore) -> None:
    with _LOCK:
        _LIMITS.add(semaphore)
        _ADMISSION_CANCELLED.setdefault(semaphore._key, 0)
        if semaphore._admission_timeout_s is not None:
            _ADMISSION_TIMEOUTS.setdefault(semaphore._key, 0)


def _count_admission_failure(totals: dict[_LimitKey, int], key: _LimitKey) -> None:
    with _LOCK:
        totals[key] += 1


def _limit_states() -> dict[_LimitKey, _LimitState]:
    """Sum the live limits that share attributes. A garbage-collected limit no longer reports."""
    with _LOCK:
        semaphores = list(_LIMITS)
    states: dict[_LimitKey, _LimitState] = {}
    for semaphore in semaphores:
        previous = states.get(semaphore._key, _LimitState(0, 0, 0))
        states[semaphore._key] = _LimitState(
            previous.limit + semaphore.limit,
            previous.active + semaphore.active,
            previous.queued + semaphore.queued,
        )
    return states


def _observe_limits(read: Callable[[_LimitState], int]) -> Callable[[Any], list]:
    def observe(_options: Any) -> list:
        if not _observer_running():
            return []
        from opentelemetry.metrics import Observation

        try:
            return [Observation(read(state), key.attributes()) for key, state in _limit_states().items()]
        except Exception:
            logger.debug("nemo-lens: the concurrency limit state is unavailable", exc_info=True)
            return []

    return observe


def _observe_admission_failures(totals: dict[_LimitKey, int]) -> Callable[[Any], list]:
    def observe(_options: Any) -> list:
        if not _observer_running():
            return []
        from opentelemetry.metrics import Observation

        with _LOCK:
            snapshot = list(totals.items())
        return [Observation(count, key.attributes()) for key, count in snapshot]

    return observe


def _observer_running() -> bool:
    from nemo_gym.telemetry.observer import driver_metrics

    return driver_metrics() is not None


def _reset_for_testing() -> None:
    with _LOCK:
        _LIMITS.clear()
        _ADMISSION_TIMEOUTS.clear()
        _ADMISSION_CANCELLED.clear()
