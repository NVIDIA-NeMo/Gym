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
"""Where server startup time goes, per server, exported through nemo-lens.

Startup is split between two kinds of process, and neither can see the whole of it:

* The **supervisor** (the process running ``gym env start`` / ``gym eval run``) knows when it
  spawned each server and when that server first answered a health probe. It cannot see inside
  the gap.
* Each **server** knows what it did between starting its interpreter and calling
  ``uvicorn.run``. It cannot see the venv setup that ran before it existed.

The two halves meet through environment variables set at spawn. The supervisor stamps the spawn
time and a W3C ``traceparent`` naming its per-server span. The launch shell stamps the time venv
setup finished. The server turns those into its first two stages (``venv_setup``,
``interpreter_start``), times the rest itself, and exports spans and metrics just before serving,
so a server's spans nest under the supervisor's span for it in one trace. A multi-worker server's
workers, which are separate processes that import the server again, report their own spans the same way.

Stages are recorded with :meth:`StageTimeline.mark`: each mark closes the stage that began at the
previous one, so stages are contiguous and sum to the total, and call sites need no extra
indentation. Spans are emitted retroactively with explicit timestamps, because telemetry is not
initialised until partway through startup.

Nothing here may fail a server or the supervisor: every telemetry step is guarded, and with
telemetry off every method is a cheap no-op.
"""

import logging
import os
from dataclasses import dataclass
from time import time_ns
from typing import Any, Dict, List, Optional

from nemo_gym.telemetry._fallbacks import is_span_group_enabled
from nemo_gym.telemetry.gym_metrics import (
    SERVER_NAME_ATTRIBUTE,
    SERVER_TYPE_ATTRIBUTE,
    WORKER_ATTRIBUTE,
    record_server_startup,
    record_server_startup_stage,
)
from nemo_gym.telemetry.gym_metrics import (
    STARTUP_STAGE_ATTRIBUTE as STAGE_ATTRIBUTE,
)
from nemo_gym.telemetry.setup import get_telemetry
from nemo_gym.telemetry.span_groups import GymSpanGroup


logger = logging.getLogger(__name__)

PROCESS_PID_ATTRIBUTE = "process.pid"

#: Wall-clock nanoseconds at which the supervisor spawned this server's launch shell.
STARTUP_SPAWN_NS_ENV = "NEMO_GYM_STARTUP_SPAWN_NS"
#: Wall-clock nanoseconds at which venv setup finished and the server's interpreter was about to launch.
#: Set by the launch shell, so it is absent when a server is started some other way.
STARTUP_SETUP_DONE_NS_ENV = "NEMO_GYM_STARTUP_SETUP_DONE_NS"
#: Wall-clock nanoseconds at which a multi-worker server's main process handed off to Uvicorn. Set in
#: the main process's environment just before ``uvicorn.run``, so the workers it spawns inherit it.
STARTUP_SERVE_NS_ENV = "NEMO_GYM_STARTUP_SERVE_NS"
#: W3C ``traceparent`` of the supervisor's span for this server.
STARTUP_TRACEPARENT_ENV = "NEMO_GYM_STARTUP_TRACEPARENT"

#: Stages recorded before the server's own code runs, in launch order.
VENV_SETUP_STAGE = "venv_setup"
INTERPRETER_START_STAGE = "interpreter_start"
#: A worker's equivalent of the two stages above: from the main process handing off to Uvicorn to the
#: worker entering ``run_webserver``. It covers spawning the process and importing the server module.
WORKER_START_STAGE = "worker_start"
#: From the end of app configuration to Uvicorn running the app's startup, just before it accepts connections.
UVICORN_STARTUP_STAGE = "uvicorn_startup"


@dataclass(frozen=True)
class Stage:
    """One contiguous slice of startup, in wall-clock nanoseconds."""

    name: str
    start_ns: int
    end_ns: int

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6


class StageTimeline:
    """Contiguous stages: each :meth:`mark` closes the stage that began at the previous mark."""

    def __init__(self, start_ns: Optional[int] = None) -> None:
        self.start_ns = time_ns() if start_ns is None else start_ns
        self.stages: List[Stage] = []
        self._cursor_ns = self.start_ns

    def mark(self, name: str) -> None:
        """End the current stage here and name it. A repeated name adds a second stage."""
        now = time_ns()
        self.stages.append(Stage(name, self._cursor_ns, now))
        self._cursor_ns = now

    def add(self, name: str, start_ns: int, end_ns: int) -> None:
        """Record a stage measured elsewhere and move the cursor to its end."""
        self.stages.append(Stage(name, start_ns, end_ns))
        self._cursor_ns = end_ns

    @property
    def end_ns(self) -> int:
        return self.stages[-1].end_ns if self.stages else self.start_ns


def _env_ns(name: str) -> Optional[int]:
    try:
        value = int(os.environ[name])
    except (KeyError, ValueError):
        return None
    return value if value > 0 else None


def _exporting_telemetry() -> Optional[Any]:
    telemetry = get_telemetry()
    if telemetry is None or not telemetry.is_exporting:
        return None
    return telemetry


def _start_span(telemetry: Any, name: str, start_ns: int, parent_context: Any, attributes: Dict[str, Any]) -> Any:
    span = telemetry.tracer.start_span(name, context=parent_context, start_time=start_ns, attributes=attributes)
    return span


def _stage_spans(
    telemetry: Any, timeline: StageTimeline, parent_context: Any, prefix: str, attributes: Dict[str, Any]
) -> None:
    for stage in timeline.stages:
        span = _start_span(
            telemetry,
            f"{prefix}.{stage.name}",
            stage.start_ns,
            parent_context,
            attributes | {STAGE_ATTRIBUTE: stage.name},
        )
        span.end(end_time=stage.end_ns)


class ServerStartupTimeline(StageTimeline):
    """A server process's own startup stages.

    A multi-worker server runs ``run_webserver`` in its main process and again in every Uvicorn
    worker. The main process times the stages up to the hand-off to Uvicorn; each worker times its
    own spawn, import and setup. Each reports its own spans and metrics, marked by ``worker``.
    Only a process that serves requests reports at Uvicorn's startup, so the last stage
    (``uvicorn_startup``) is timed to the moment it starts accepting connections.
    """

    def __init__(
        self, server_name: str, server_type: Optional[str], *, worker: bool = False, enabled: bool = True
    ) -> None:
        super().__init__()
        self.server_name = server_name
        self.server_type = server_type
        self.worker = worker
        self.enabled = enabled
        self._reported = False
        if not enabled:
            return
        entered_ns = self.start_ns
        if worker:
            serve_ns = _env_ns(STARTUP_SERVE_NS_ENV)
            if serve_ns is not None and serve_ns <= entered_ns:
                self.add(WORKER_START_STAGE, serve_ns, entered_ns)
            return
        # Stages that happened before this process could time itself. Both stamps are optional so a
        # server started by hand (no supervisor, no launch shell) still reports its own stages.
        spawn_ns = _env_ns(STARTUP_SPAWN_NS_ENV)
        setup_done_ns = _env_ns(STARTUP_SETUP_DONE_NS_ENV)
        if spawn_ns is not None and setup_done_ns is not None and spawn_ns <= setup_done_ns <= entered_ns:
            self.add(VENV_SETUP_STAGE, spawn_ns, setup_done_ns)
            self.add(INTERPRETER_START_STAGE, setup_done_ns, entered_ns)
        elif spawn_ns is not None and spawn_ns <= entered_ns:
            self.add(INTERPRETER_START_STAGE, spawn_ns, entered_ns)

    def mark(self, name: str) -> None:
        if self.enabled:
            super().mark(name)

    def announce_serve(self) -> None:
        """Stamp the hand-off to Uvicorn into the environment for the workers it is about to spawn."""
        if self.enabled:
            os.environ[STARTUP_SERVE_NS_ENV] = str(time_ns())

    def report_when_serving(self, app: Any) -> None:
        """Report when ``app``'s startup runs, adding the ``uvicorn_startup`` stage.

        Wraps the app's lifespan, so a lifespan the server already set keeps running inside it.
        """
        if not self.enabled:
            return
        from contextlib import asynccontextmanager

        original_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application: Any):
            self.mark(UVICORN_STARTUP_STAGE)
            self.report()
            async with original_lifespan(application) as state:
                yield state

        app.router.lifespan_context = lifespan

    def report(self) -> None:
        """Emit spans and metrics, once. Safe to call from a process that never finishes starting."""
        if not self.enabled or not self.stages or self._reported:
            return
        self._reported = True
        try:
            self._emit_telemetry()
        except Exception:
            logger.debug("nemo-lens: could not report startup telemetry", exc_info=True)

    def _emit_telemetry(self) -> None:
        attributes: Dict[str, Any] = {SERVER_NAME_ATTRIBUTE: self.server_name, WORKER_ATTRIBUTE: self.worker}
        if self.server_type:
            attributes[SERVER_TYPE_ATTRIBUTE] = self.server_type
        for stage in self.stages:
            record_server_startup_stage(
                stage.duration_ms,
                stage=stage.name,
                server_name=self.server_name,
                server_type=self.server_type,
                worker=self.worker,
            )

        telemetry = _exporting_telemetry()
        if telemetry is None or not is_span_group_enabled(GymSpanGroup.STARTUP):
            return
        from opentelemetry import trace as otel_trace
        from opentelemetry.propagate import extract

        carrier = {"traceparent": os.environ.get(STARTUP_TRACEPARENT_ENV, "")}
        parent_context = extract(carrier) if carrier["traceparent"] else None
        root_name = "gym.server.startup.worker" if self.worker else "gym.server.startup"
        attributes[PROCESS_PID_ATTRIBUTE] = os.getpid()
        root = _start_span(telemetry, root_name, self.stages[0].start_ns, parent_context, attributes)
        _stage_spans(telemetry, self, otel_trace.set_span_in_context(root), root_name, attributes)
        root.end(end_time=self.end_ns)


@dataclass
class _ServerRecord:
    server_type: str
    spawn_ns: int
    span: Any = None
    ready_ns: Optional[int] = None


class SupervisorStartup(StageTimeline):
    """The supervisor's view of one startup: its own stages, plus spawn and ready time per server."""

    def __init__(self) -> None:
        super().__init__()
        self._servers: Dict[str, _ServerRecord] = {}
        self._telemetry: Any = None
        self._root_span: Any = None

    def begin_trace(self) -> None:
        """Open the ``gym.startup`` span. Call once telemetry has been initialised."""
        telemetry = _exporting_telemetry()
        if telemetry is None or not is_span_group_enabled(GymSpanGroup.STARTUP):
            return
        try:
            self._root_span = _start_span(telemetry, "gym.startup", self.start_ns, None, {})
            self._telemetry = telemetry
        except Exception:
            logger.debug("nemo-lens: could not open the startup span", exc_info=True)

    def server_env(self, server_name: str, server_type: str) -> Dict[str, str]:
        """Environment for one server about to be spawned. Records its spawn time."""
        record = _ServerRecord(server_type=server_type, spawn_ns=time_ns())
        env = {STARTUP_SPAWN_NS_ENV: str(record.spawn_ns)}
        if self._root_span is not None:
            try:
                from opentelemetry import trace as otel_trace
                from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

                root_context = otel_trace.set_span_in_context(self._root_span)
                record.span = _start_span(
                    self._telemetry,
                    "gym.startup.server",
                    record.spawn_ns,
                    root_context,
                    {SERVER_NAME_ATTRIBUTE: server_name, SERVER_TYPE_ATTRIBUTE: server_type},
                )
                carrier: Dict[str, str] = {}
                TraceContextTextMapPropagator().inject(carrier, context=otel_trace.set_span_in_context(record.span))
                if "traceparent" in carrier:
                    env[STARTUP_TRACEPARENT_ENV] = carrier["traceparent"]
            except Exception:
                logger.debug("nemo-lens: could not open a server startup span", exc_info=True)
        self._servers[server_name] = record
        return env

    def server_ready(self, server_name: str) -> None:
        """Record that ``server_name`` first answered a health probe. Later calls are ignored."""
        record = self._servers.get(server_name)
        if record is None or record.ready_ns is not None:
            return
        record.ready_ns = time_ns()
        record_server_startup(
            (record.ready_ns - record.spawn_ns) / 1e6, server_name=server_name, server_type=record.server_type
        )
        if record.span is not None:
            try:
                record.span.end(end_time=record.ready_ns)
            except Exception:
                logger.debug("nemo-lens: could not close a server startup span", exc_info=True)

    def finish(self) -> None:
        """Close the ``gym.startup`` span with the supervisor's own stages beneath it."""
        if self._root_span is None:
            return
        try:
            from opentelemetry import trace as otel_trace

            _stage_spans(self._telemetry, self, otel_trace.set_span_in_context(self._root_span), "gym.startup", {})
            self._root_span.end(end_time=self.end_ns)
        except Exception:
            logger.debug("nemo-lens: could not close the startup span", exc_info=True)
        self._root_span = None
