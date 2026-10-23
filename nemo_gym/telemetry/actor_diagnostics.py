# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, payload-free diagnostics for a training client's async actor.

Synchronous phases are inclusive and must not be summed. Thread CPU applies
only to the synchronous section; the watchdog reports whole-process CPU.
Stack samples contain filenames/functions/line numbers, never source or locals.
The watchdog is best effort: a thread holding the GIL can delay it as well.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)
_CURRENT: ContextVar[ActorDiagnostics | None] = ContextVar("gym_actor_diagnostics", default=None)


@dataclass
class _Timing:
    calls: int = 0
    wall_s: float = 0.0
    thread_cpu_s: float = 0.0
    max_wall_s: float = 0.0
    errors: int = 0


class ActorDiagnostics:
    """Bounded process-local summaries, plus a low-rate loop watchdog.

    Call ``start`` on the actor loop and ``close`` during shutdown. Nothing is
    installed while disabled. GC observations never alter collection policy.
    """

    def __init__(self, *, enabled: bool, report_s: float = 30.0, stall_s: float = 2.0) -> None:
        self.enabled = enabled
        self.report_s = report_s
        self.stall_s = stall_s
        self._lock = threading.RLock()
        self._timings: dict[str, _Timing] = {}
        self._active: list[tuple[str, float]] = []
        self._rollouts = 0
        self._rollouts_started = 0
        self._loop_thread: int | None = None
        self._heartbeat: asyncio.Task | None = None
        self._watchdog: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_beat = time.monotonic()
        self._last_report = time.monotonic()
        self._last_stall = 0.0
        self._loop_lag_max_s = 0.0
        self._gc_started: dict[int, float] = {}
        self._gc_s: dict[int, float] = {}
        self._gc_count: dict[int, int] = {}
        self._process_cpu_started = time.process_time()
        self._started = time.monotonic()

    def start(self) -> None:
        """Install once, from the async actor's event-loop thread."""
        if not self.enabled or self._heartbeat is not None:
            return
        self._loop_thread = threading.get_ident()
        self._last_beat = time.monotonic()
        gc.callbacks.append(self._gc)
        self._heartbeat = asyncio.create_task(self._beat(), name="gym-actor-diagnostics")
        self._watchdog = threading.Thread(target=self._watch, name="gym-actor-watchdog", daemon=True)
        self._watchdog.start()
        self.event("installed")

    async def _beat(self) -> None:
        while True:
            expected = time.monotonic() + 0.5
            await asyncio.sleep(0.5)
            with self._lock:
                self._last_beat = time.monotonic()
                self._loop_lag_max_s = max(self._loop_lag_max_s, self._last_beat - expected)

    def _gc(self, phase: str, info: dict[str, Any]) -> None:
        generation = info["generation"]
        with self._lock:
            if phase == "start":
                self._gc_started[generation] = time.monotonic()
            elif phase == "stop":
                started = self._gc_started.pop(generation, None)
                if started is not None:
                    self._gc_s[generation] = self._gc_s.get(generation, 0.0) + time.monotonic() - started
                    self._gc_count[generation] = self._gc_count.get(generation, 0) + 1

    def _watch(self) -> None:
        while not self._stop.wait(1.0):
            now = time.monotonic()
            with self._lock:
                lag = now - self._last_beat
            if lag >= self.stall_s and now - self._last_stall >= 10.0:
                self._last_stall = now
                frame = sys._current_frames().get(self._loop_thread)
                stack = []
                while frame is not None and len(stack) < 12:
                    stack.append(f"{frame.f_code.co_filename}:{frame.f_lineno}:{frame.f_code.co_name}")
                    frame = frame.f_back
                frame = None
                self.event("loop_stall", heartbeat_age_s=round(lag, 6), stack=stack)
            if now - self._last_report >= self.report_s:
                self._last_report = now
                self.event("summary")

    def event(self, event: str, **fields: Any) -> None:
        """Log cumulative numeric state and explicitly selected metadata only."""
        if not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            data = {
                "event": event,
                "unix_s": time.time(),
                "pid": os.getpid(),
                "elapsed_s": round(now - self._started, 6),
                "process_cpu_s": round(time.process_time() - self._process_cpu_started, 6),
                "active_rollout_calls": self._rollouts,
                "rollout_calls_started": self._rollouts_started,
                "loop_lag_max_s": round(self._loop_lag_max_s, 6),
                "active_sync_phases": [
                    {"phase": name, "age_s": round(now - start, 6)} for name, start in self._active
                ],
                "gc_pause_s": {str(k): round(v, 6) for k, v in self._gc_s.items()},
                "gc_collections": dict(self._gc_count),
                "phases": {
                    name: {
                        "calls": value.calls,
                        "wall_s": round(value.wall_s, 6),
                        "thread_cpu_s": round(value.thread_cpu_s, 6),
                        "max_wall_s": round(value.max_wall_s, 6),
                        "errors": value.errors,
                    }
                    for name, value in self._timings.items()
                },
                **fields,
            }
        logger.warning("gym_actor_probe %s", json.dumps(data, sort_keys=True))

    @contextmanager
    def rollout(self) -> Iterator[None]:
        """Count admitted actor calls through streaming completion/cancellation."""
        if not self.enabled:
            yield
            return
        self.start()
        with self._lock:
            self._rollouts += 1
            self._rollouts_started += 1
        try:
            yield
        finally:
            with self._lock:
                self._rollouts -= 1

    @contextmanager
    def startup(self) -> Iterator[None]:
        """Scope nested synchronous Gym timers to this actor, not other clients."""
        if not self.enabled:
            yield
            return
        token = _CURRENT.set(self)
        try:
            with startup_phase("rollout_start_total"):
                yield
        finally:
            _CURRENT.reset(token)

    def close(self) -> None:
        """Remove owned probes without waiting on the event loop being observed."""
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.cancel()
        if self._gc in gc.callbacks:
            gc.callbacks.remove(self._gc)
        self.event("closed")


@contextmanager
def startup_phase(name: str) -> Iterator[None]:
    """Measure one synchronous startup phase only inside an opted-in scope."""
    probe = _CURRENT.get()
    if probe is None:
        yield
        return
    started, cpu = time.monotonic(), time.thread_time()
    failed = True
    with probe._lock:
        probe._active.append((name, started))
    try:
        yield
        failed = False
    finally:
        elapsed, cpu_elapsed = time.monotonic() - started, time.thread_time() - cpu
        with probe._lock:
            probe._active.pop()
            timing = probe._timings.setdefault(name, _Timing())
            timing.calls += 1
            timing.wall_s += elapsed
            timing.thread_cpu_s += cpu_elapsed
            timing.max_wall_s = max(timing.max_wall_s, elapsed)
            timing.errors += int(failed)
