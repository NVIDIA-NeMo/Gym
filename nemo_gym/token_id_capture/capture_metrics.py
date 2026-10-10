# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-overhead token-capture metrics for call sites inside this package.

Capture runs on every captured model call. With no metrics exporter active, a call site pays one
attribute check and gets back a shared no-op timer. With one active:

- an operation costs two clock reads and one histogram record, made in ``__exit__`` so failed and
  cancelled operations are recorded too, labelled ``error`` or ``cancelled``;
- outcome and digest counts are integer increments in process memory, read by observable
  instruments at each export, because they fire far more often than operations do;
- an operation slower than :data:`SLOW_OPERATION_MS` also logs one warning (at most one per
  operation per minute), which survives a crash that loses the last metric export.

Operations are timed where Gym calls a capture protocol (``TokenSink``, ``TokenSource``,
``LineageResolver``, ``CaptureLedger``, ``StagingSink``), labelled with the implementing class, so
framework-provided implementations are measured without changes to them.

This package stays a leaf: nothing from ``nemo_gym.telemetry`` is imported until that package has
already been loaded, which every process that exports metrics does when it sets telemetry up.

Usage::

    with capture_metrics.timed("sink.put", component=sink, tokens=entry.cum_len):
        await sink.put(entry)
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import time
import weakref
from collections.abc import Iterable
from typing import Any, Optional


logger = logging.getLogger(__name__)

#: Operations slower than this also log a warning.
SLOW_OPERATION_MS = 5_000.0
_SLOW_LOG_INTERVAL_S = 60.0

#: Distinct (outcome, reason) pairs kept in process. Past this, a new pair is counted under its
#: outcome with reason ``other`` when that pair already exists, and under ``other``/``other``
#: otherwise, so the totals never hold more than this many pairs plus one.
MAX_OUTCOME_KEYS = 256
OTHER_REASON = "other"

_setup: Any = None
_recorders: Any = None
_CACHE_OWNERS: "weakref.WeakSet[Any]" = weakref.WeakSet()
_SLOW_LOGGED_AT: dict[str, float] = {}

# Totals read by observable instruments at export. Capture runs on many worker threads, and a
# read-modify-write on a dict is not atomic even with the global interpreter lock, so increments
# take this lock; uncontended it costs about a tenth of a microsecond.
_TOTALS_LOCK = threading.Lock()
_OUTCOMES: dict[tuple[str, str], int] = {}
_DIGESTS: dict[str, list[int]] = {}


def _load() -> bool:
    """Bind the telemetry modules once telemetry has been set up in this process."""
    global _setup, _recorders
    setup_module = sys.modules.get("nemo_gym.telemetry.setup")
    if setup_module is None:
        return False
    try:
        from nemo_gym.telemetry import token_capture_metrics as recorders_module
    except Exception:
        _setup = False
        return False
    _setup, _recorders = setup_module, recorders_module
    _recorders.set_totals_source(_Totals())
    return True


def active() -> bool:
    """Whether this process exports metrics. Cheap enough for every captured call."""
    if _setup is None and not _load():
        return False
    if _setup is False:
        return False
    handle = _setup._TELEMETRY_HANDLE
    return handle is not None and handle.is_exporting


def implementation(component: Any) -> str:
    """The class name of a protocol implementation, as a bounded metric label.

    Read from the class on every call rather than cached, so no reference to the class is kept.
    """
    return type(component).__name__


class _NoTimer:
    """Returned when nothing is recorded; setting ``tokens`` on it is ignored."""

    __slots__ = ()

    def __enter__(self) -> "_NoTimer":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    @property
    def tokens(self) -> Optional[int]:
        return None

    @tokens.setter
    def tokens(self, value: Optional[int]) -> None:
        pass


_NO_TIMER = _NoTimer()


class _Timer:
    __slots__ = ("operation", "implementation", "tokens", "threshold_ns", "started")

    def __init__(self, operation: str, implementation: str, tokens: Optional[int], threshold_ms: float) -> None:
        self.operation = operation
        self.implementation = implementation
        self.tokens = tokens
        self.threshold_ns = int(threshold_ms * 1_000_000)
        self.started = 0

    def __enter__(self) -> "_Timer":
        self.started = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        elapsed_ns = time.perf_counter_ns() - self.started
        if exc_type is None:
            error = ""
        elif issubclass(exc_type, asyncio.CancelledError):
            error = "cancelled"
        else:
            error = "error"
        if error or elapsed_ns >= self.threshold_ns:
            _record(self.operation, self.implementation, error, elapsed_ns / 1_000_000, self.tokens)
        return False


def timed(
    operation: str,
    *,
    component: Any = None,
    implementation_name: str = "",
    tokens: Optional[int] = None,
    threshold_ms: float = 0.0,
) -> Any:
    """Time the ``with`` block as one operation. Never raises from recording.

    ``component`` is the protocol implementation called inside the block; its class becomes the
    implementation label. ``tokens`` may also be set on the returned timer inside the block.
    With ``threshold_ms``, a successful operation is recorded only when it took at least that long.
    """
    if not active():
        return _NO_TIMER
    name = implementation(component) if component is not None else implementation_name
    return _Timer(operation, name, tokens, threshold_ms)


def _record(operation: str, implementation_name: str, error: str, duration_ms: float, tokens: Optional[int]) -> None:
    try:
        instruments = _recorders.instruments()
        if instruments is not None:
            instruments.record_operation(operation, implementation_name, error, duration_ms, tokens)
    except Exception:
        pass
    if duration_ms >= SLOW_OPERATION_MS:
        now = time.monotonic()
        if now - _SLOW_LOGGED_AT.get(operation, -_SLOW_LOG_INTERVAL_S) >= _SLOW_LOG_INTERVAL_S:
            _SLOW_LOGGED_AT[operation] = now
            logger.warning(
                "Slow token-capture operation %s (%s): %.0f ms, %s tokens%s.",
                operation,
                implementation_name or "-",
                duration_ms,
                tokens if tokens is not None else "-",
                f", {error}" if error else "",
            )


def count(outcome: str, reason: str = "") -> None:
    """Count one capture decision or failure. ``reason`` should come from a fixed set."""
    if not active():
        return
    with _TOTALS_LOCK:
        key = (outcome, reason)
        if key not in _OUTCOMES and len(_OUTCOMES) >= MAX_OUTCOME_KEYS:
            key = (outcome, OTHER_REASON)
            if key not in _OUTCOMES:
                key = (OTHER_REASON, OTHER_REASON)
        _OUTCOMES[key] = _OUTCOMES.get(key, 0) + 1


def bounded_reason(reason: Optional[str], allowed: frozenset[str]) -> str:
    """Map a reason from an external implementation onto a known set."""
    if not reason:
        return ""
    return reason if reason in allowed else OTHER_REASON


def digest(kind: str, nbytes: int) -> None:
    """Count one hash computation over ``nbytes`` bytes. ``kind`` comes from a fixed set."""
    if not active():
        return
    with _TOTALS_LOCK:
        totals = _DIGESTS.get(kind)
        if totals is None:
            totals = _DIGESTS[kind] = [0, 0]
        totals[0] += 1
        totals[1] += nbytes


def track_cache_owner(owner: Any) -> None:
    """Report ``owner.cache_sizes()`` in the cache-size metric while ``owner`` is alive."""
    _CACHE_OWNERS.add(owner)


class _Totals:
    """What the observable instruments read at each export."""

    def outcomes(self) -> Iterable[tuple[tuple[str, str], int]]:
        with _TOTALS_LOCK:
            return list(_OUTCOMES.items())

    def digests(self) -> Iterable[tuple[str, int, int]]:
        with _TOTALS_LOCK:
            return [(kind, calls, nbytes) for kind, (calls, nbytes) in _DIGESTS.items()]

    def cache_sizes(self) -> Iterable[tuple[str, int]]:
        totals: dict[str, int] = {}
        for owner in list(_CACHE_OWNERS):
            # One owner failing (for example a dict resized by another thread mid-read) must not
            # blank every other cache's size for this export.
            try:
                sizes = list(owner.cache_sizes())
            except Exception:
                logger.debug("Could not read the cache sizes of %s.", type(owner).__name__, exc_info=True)
                continue
            for cache, size in sizes:
                totals[cache] = totals.get(cache, 0) + size
        return list(totals.items())


def _reset_for_testing() -> None:
    with _TOTALS_LOCK:
        _OUTCOMES.clear()
        _DIGESTS.clear()
    _SLOW_LOGGED_AT.clear()
