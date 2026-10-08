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

"""OTel instruments for training-token capture.

Token capture runs on every captured model call, so recording must stay cheap and the number of
exported series must stay small at scale (one exporter per server worker, many servers per job).
Call sites reach these instruments through ``nemo_gym.token_id_capture.metrics``, which is a no-op
unless a metrics exporter is active.

Instruments
-----------
``gym.token_capture.operation.duration_ms`` (histogram): wall-clock of one capture operation.
Attributes: ``nemo.gym.token_capture.operation`` (a fixed name such as ``sink.put``),
``nemo.gym.token_capture.implementation`` (the class implementing the protocol at that call site,
for example ``TokenCaptureStore`` or a framework's sink), ``nemo.gym.token_capture.size_class``
(the record's token count, bucketed: ``lt_4k``, ``lt_32k``, ``lt_128k``, ``ge_128k``; absent when
the operation has no record), and ``nemo.gym.token_capture.error`` (``error`` or ``cancelled``,
present only when the operation did not complete, so successful calls add no extra series).
Operations timed at a protocol call site include time spent waiting for a worker thread and the
event loop; Gym's file stores also report their in-thread work (``token_store.append``,
``ledger_store.record``), so the difference is queueing. Lock waits (``file_lock.wait``) are
recorded only when the wait exceeded a threshold; ``lock_acquired`` outcomes count every acquisition.

``gym.token_capture.operation.tokens_total`` (counter): tokens of the records each operation
handled. ``duration sum / tokens`` gives milliseconds per token without a per-record histogram.

``gym.token_capture.outcome_total`` (observable counter): capture decisions and failures, with
``nemo.gym.token_capture.outcome`` and a ``nemo.gym.token_capture.reason`` from a fixed set.

``gym.token_capture.digest.calls_total`` / ``gym.token_capture.digest.bytes_total`` (observable
counters): hash computations and the bytes they hashed, by ``nemo.gym.token_capture.digest``.

``gym.token_capture.cache.size`` (observable up-down counter): current size of the in-process
capture caches, by ``nemo.gym.token_capture.cache``.

The observable instruments read integer totals that call sites keep in process memory, so a
high-frequency event costs an increment, not an SDK call.
"""

import logging
import threading
from collections.abc import Callable, Iterable
from typing import Any, Optional

from nemo_gym.telemetry.gym_metrics import _meter


logger = logging.getLogger(__name__)

OPERATION_DURATION_INSTRUMENT = "gym.token_capture.operation.duration_ms"
OPERATION_TOKENS_INSTRUMENT = "gym.token_capture.operation.tokens_total"
OUTCOME_INSTRUMENT = "gym.token_capture.outcome_total"
DIGEST_CALLS_INSTRUMENT = "gym.token_capture.digest.calls_total"
DIGEST_BYTES_INSTRUMENT = "gym.token_capture.digest.bytes_total"
CACHE_SIZE_INSTRUMENT = "gym.token_capture.cache.size"

OPERATION_ATTRIBUTE = "nemo.gym.token_capture.operation"
IMPLEMENTATION_ATTRIBUTE = "nemo.gym.token_capture.implementation"
SIZE_CLASS_ATTRIBUTE = "nemo.gym.token_capture.size_class"
ERROR_ATTRIBUTE = "nemo.gym.token_capture.error"
OUTCOME_ATTRIBUTE = "nemo.gym.token_capture.outcome"
REASON_ATTRIBUTE = "nemo.gym.token_capture.reason"
DIGEST_ATTRIBUTE = "nemo.gym.token_capture.digest"
CACHE_ATTRIBUTE = "nemo.gym.token_capture.cache"

#: Roughly doubling from 10 microseconds to five minutes. Capture spans cached resolves (micro-
#: seconds), record builds over long sequences (hundreds of milliseconds), and fsyncs or lock waits
#: on a loaded shared filesystem (seconds to minutes).
OPERATION_DURATION_BOUNDARIES_MS: tuple[float, ...] = (
    0.01,
    0.1,
    0.25,
    0.5,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1_024,
    2_048,
    4_096,
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    300_000,
)

#: Token-count classes for the duration histogram, so a percentile compares like with like.
SIZE_CLASSES: tuple[tuple[int, str], ...] = ((4_096, "lt_4k"), (32_768, "lt_32k"), (131_072, "lt_128k"))
LARGEST_SIZE_CLASS = "ge_128k"

#: Per-instrument limit on distinct attribute sets, below the SDK's own per-instrument limit. Past
#: it, new combinations are recorded under ``other`` instead of growing the series count.
MAX_ATTRIBUTE_SETS = 512
OVERFLOW_LABEL = "other"


def size_class(tokens: int) -> str:
    for bound, label in SIZE_CLASSES:
        if tokens < bound:
            return label
    return LARGEST_SIZE_CLASS


class _AttributeCache:
    """Attribute dicts reused per combination, capped so they cannot grow without bound."""

    def __init__(self) -> None:
        self._sets: dict[tuple, dict[str, str]] = {}

    def get(self, key: tuple, build: Callable[[], dict[str, str]]) -> dict[str, str]:
        attributes = self._sets.get(key)
        if attributes is not None:
            return attributes
        attributes = build()
        if len(self._sets) >= MAX_ATTRIBUTE_SETS:
            overflow = ("__overflow__", *sorted(attributes))
            cached = self._sets.get(overflow)
            if cached is None:
                cached = {name: OVERFLOW_LABEL for name in attributes}
                self._sets[overflow] = cached
            return cached
        # A racing writer stores an equal dict; either one is fine.
        self._sets[key] = attributes
        return attributes


class TokenCaptureInstruments:
    """The synchronous token-capture instruments of one meter."""

    def __init__(self, meter: Any) -> None:
        self._duration = meter.create_histogram(
            OPERATION_DURATION_INSTRUMENT,
            unit="ms",
            description="Duration of one token-capture operation.",
            explicit_bucket_boundaries_advisory=list(OPERATION_DURATION_BOUNDARIES_MS),
        )
        self._tokens = meter.create_counter(
            OPERATION_TOKENS_INSTRUMENT,
            unit="{token}",
            description="Tokens of the records token-capture operations handled.",
        )
        self._duration_attributes = _AttributeCache()
        self._token_attributes = _AttributeCache()

    def record_operation(
        self, operation: str, implementation: str, error: str, duration_ms: float, tokens: Optional[int]
    ) -> None:
        size = size_class(tokens) if tokens is not None else ""

        def build() -> dict[str, str]:
            attributes = {OPERATION_ATTRIBUTE: operation, IMPLEMENTATION_ATTRIBUTE: implementation}
            if size:
                attributes[SIZE_CLASS_ATTRIBUTE] = size
            if error:
                attributes[ERROR_ATTRIBUTE] = error
            return attributes

        attributes = self._duration_attributes.get((operation, implementation, size, error), build)
        self._duration.record(duration_ms, attributes=attributes)
        if tokens:
            token_attributes = self._token_attributes.get((operation,), lambda: {OPERATION_ATTRIBUTE: operation})
            self._tokens.add(tokens, attributes=token_attributes)


_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"meter": None, "instruments": None, "totals": None}


def set_totals_source(source: Any) -> None:
    """Set the object whose ``outcomes()``, ``digests()``, and ``cache_sizes()`` the observable
    instruments read at each export."""
    _STATE["totals"] = source


def instruments() -> Optional[TokenCaptureInstruments]:
    """Return this process's instruments, or ``None`` when nothing is exporting. Never raises."""
    meter = _meter()
    if meter is None:
        return None
    current = _STATE["instruments"]
    if current is not None and _STATE["meter"] is meter:
        return current
    with _LOCK:
        if _STATE["instruments"] is None or _STATE["meter"] is not meter:
            try:
                _STATE["instruments"] = TokenCaptureInstruments(meter)
                _register_observables(meter)
                _STATE["meter"] = meter
            except Exception:
                logger.debug("nemo-lens: failed to create token-capture instruments", exc_info=True)
                return None
        return _STATE["instruments"]


def _observe(read: Callable[[Any], Iterable[tuple[int, dict[str, str]]]]) -> Callable[[Any], list]:
    from opentelemetry.metrics import Observation

    def callback(_options: Any) -> list:
        source = _STATE["totals"]
        if source is None:
            return []
        try:
            return [Observation(value, attributes) for value, attributes in read(source)]
        except Exception:
            logger.debug("nemo-lens: failed to observe token-capture totals", exc_info=True)
            return []

    return callback


def _register_observables(meter: Any) -> None:
    meter.create_observable_counter(
        OUTCOME_INSTRUMENT,
        callbacks=[
            _observe(
                lambda totals: (
                    (count, {OUTCOME_ATTRIBUTE: outcome, REASON_ATTRIBUTE: reason})
                    for (outcome, reason), count in totals.outcomes()
                )
            )
        ],
        unit="1",
        description="Token-capture decisions and failures.",
    )
    meter.create_observable_counter(
        DIGEST_CALLS_INSTRUMENT,
        callbacks=[
            _observe(lambda totals: ((calls, {DIGEST_ATTRIBUTE: kind}) for kind, calls, _ in totals.digests()))
        ],
        unit="1",
        description="Token-capture hash computations.",
    )
    meter.create_observable_counter(
        DIGEST_BYTES_INSTRUMENT,
        callbacks=[
            _observe(lambda totals: ((nbytes, {DIGEST_ATTRIBUTE: kind}) for kind, _, nbytes in totals.digests()))
        ],
        unit="By",
        description="Bytes hashed by token capture.",
    )
    meter.create_observable_up_down_counter(
        CACHE_SIZE_INSTRUMENT,
        callbacks=[
            _observe(lambda totals: ((size, {CACHE_ATTRIBUTE: cache}) for cache, size in totals.cache_sizes()))
        ],
        unit="1",
        description="Current size of in-process token-capture caches.",
    )


def _reset_for_testing() -> None:
    with _LOCK:
        _STATE.update(meter=None, instruments=None)
