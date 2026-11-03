# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dependency-free producer capture, also staged as a standalone sandbox module."""

import asyncio
import json
import logging
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4


if __package__:
    from nemo_gym.utils.jsonl_capture import append_jsonl
else:
    from jsonl_capture import append_jsonl


logger = logging.getLogger(__name__)
ToolStatus = Literal["completed", "failed", "timeout", "cancelled", "incomplete", "unknown"]
TimingBoundary = Literal["lifecycle", "executor", "transcript", "event_receipt"]
_STATUSES = {"completed", "failed", "timeout", "cancelled", "incomplete", "unknown"}


@dataclass
class ClockContext:
    """Comparable samples from one clock owner and continuous recording epoch.

    Share this object only where a common clock is established. A new context has
    a fresh epoch even when the owner string is reused after a process restart.
    """

    location: Literal["host", "sandbox", "producer"]
    owner: str
    discontinuity_tolerance_s: float = 0.1
    _epoch: str = field(default_factory=lambda: uuid4().hex, init=False)
    _offset: float | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.location not in {"host", "sandbox", "producer"} or not self.owner or "/" in self.owner:
            raise ValueError("Clock location and owner must identify one host, sandbox or producer")
        if self.discontinuity_tolerance_s < 0:
            raise ValueError("Clock discontinuity tolerance must be nonnegative")

    def sample(self) -> tuple[float, float, str]:
        """Sample wall/monotonic clocks, rotating the domain after a clock step."""
        with self._lock:
            monotonic = time.monotonic()
            wall = time.time()
            offset = wall - monotonic
            if self._offset is not None and abs(offset - self._offset) > self.discontinuity_tolerance_s:
                self._epoch = uuid4().hex
            self._offset = offset
            return wall, monotonic, f"{self.location}/{self.owner}/{self._epoch}"


@dataclass
class ToolCallHandle:
    """Call-local start identity and outcome; pass the handle between callbacks."""

    span_id: str
    native_call_id: str | None
    tool_call_id: str | None
    tool_name: str | None
    model_call_ref: dict[str, object] | None
    parent_span_id: str | None
    started_at: float
    started_monotonic: float
    clock_domain: str
    _recorder: "ToolCallRecorder" = field(repr=False)
    status: ToolStatus = "unknown"
    failure_id: str | None = None
    _finished: bool = field(default=False, init=False, repr=False)
    _capture_failed: bool = field(default=False, init=False, repr=False)

    def set_outcome(self, status: ToolStatus, *, failure_id: str | None = None) -> None:
        """Attach an explicit adapter-classified outcome without guessing success."""
        if status not in _STATUSES:
            raise ValueError(f"Unknown tool status: {status}")
        with self._recorder._lock:
            if self._finished:
                raise ValueError("Cannot change the outcome after a tool attempt finishes")
            self.status, self.failure_id = status, failure_id


class ToolCallRecorder:
    """Append starts before dispatch and finishes without altering tool execution.

    Each recorder owns a producer identity and sequence. Give independent
    processes distinct producer IDs; file locking permits a shared output path.
    Construct and collect this recorder only when observability is enabled.
    """

    def __init__(
        self,
        *,
        path: str | Path,
        producer_id: str,
        invocation_id: str,
        agent_instance: str | None,
        clock: ClockContext,
        boundary: TimingBoundary,
        timing_source: str,
    ) -> None:
        if not producer_id or not invocation_id or not timing_source:
            raise ValueError("Producer, invocation and timing source must be nonempty")
        if boundary not in {"lifecycle", "executor", "transcript", "event_receipt"}:
            raise ValueError(f"Unknown timing boundary: {boundary}")
        self.path = Path(path)
        self.producer_id = producer_id
        self.invocation_id = invocation_id
        self.agent_instance = agent_instance
        self.clock = clock
        self.boundary = boundary
        self.timing_source = timing_source
        self._sequence = 0
        self._lock = threading.RLock()
        self._failed_spans: set[str] = set()

    @property
    def failed_spans(self) -> tuple[str, ...]:
        """Attempts with write failures, for the adapter's collection diagnostics."""
        with self._lock:
            return tuple(sorted(self._failed_spans))

    def start(
        self,
        *,
        native_call_id: str | None,
        tool_name: str | None,
        model_call_ref: Mapping[str, object] | None = None,
        tool_call_id: str | None = None,
        parent: ToolCallHandle | None = None,
        parent_span_id: str | None = None,
    ) -> ToolCallHandle:
        """Allocate an attempt and flush its start before returning to execution."""
        if parent is not None:
            if parent._recorder is not self:
                raise ValueError("A parent handle must belong to this recorder; use an explicit parent span otherwise")
            if parent_span_id is not None and parent_span_id != parent.span_id:
                raise ValueError("Conflicting parent handle and span")
            parent_span_id = parent.span_id
        wall, monotonic, domain = self.clock.sample()
        handle = ToolCallHandle(
            span_id=uuid4().hex,
            native_call_id=native_call_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            model_call_ref=deepcopy(dict(model_call_ref)) if model_call_ref is not None else None,
            parent_span_id=parent_span_id,
            started_at=wall,
            started_monotonic=monotonic,
            clock_domain=domain,
            _recorder=self,
        )
        self._record(handle, "start", {"started_at": wall, "clock_domain": domain})
        return handle

    def finish(
        self, handle: ToolCallHandle, *, status: ToolStatus | None = None, failure_id: str | None = None
    ) -> None:
        """Persist the observed terminal boundary once; repeated callbacks are harmless."""
        if handle._recorder is not self:
            raise ValueError("Tool handle belongs to another recorder")
        with self._lock:
            if handle._finished:
                return
            if status is not None:
                handle.set_outcome(status, failure_id=failure_id if failure_id is not None else handle.failure_id)
            elif failure_id is not None:
                handle.failure_id = failure_id
            wall, monotonic, domain = self.clock.sample()
            self._record(
                handle,
                "finish",
                {
                    "completed_at": wall,
                    "clock_domain": domain,
                    "duration_ms": (monotonic - handle.started_monotonic) * 1000,
                    "duration_source": "monotonic",
                },
            )
            handle._finished = True

    @contextmanager
    def call(
        self,
        *,
        native_call_id: str | None,
        tool_name: str | None,
        model_call_ref: Mapping[str, object] | None = None,
        tool_call_id: str | None = None,
        parent: ToolCallHandle | None = None,
        parent_span_id: str | None = None,
    ) -> Iterator[ToolCallHandle]:
        """Record a synchronous or awaited operation, preserving its exception/result."""
        handle = self.start(
            native_call_id=native_call_id,
            tool_name=tool_name,
            model_call_ref=model_call_ref,
            tool_call_id=tool_call_id,
            parent=parent,
            parent_span_id=parent_span_id,
        )
        try:
            yield handle
        except asyncio.CancelledError:
            self.finish(handle, status="cancelled")
            raise
        except Exception:
            self.finish(
                handle, status="failed" if handle.status == "unknown" else handle.status, failure_id=handle.failure_id
            )
            raise
        else:
            self.finish(handle)

    def _record(self, handle: ToolCallHandle, kind: str, measurements: dict[str, object]) -> None:
        with self._lock:
            event = {
                "schema_version": "2.0",
                "producer_id": self.producer_id,
                "sequence": self._sequence,
                "span_id": handle.span_id,
                "invocation_id": self.invocation_id,
                "agent_instance": self.agent_instance,
                "native_call_id": handle.native_call_id,
                "tool_call_id": handle.tool_call_id,
                "tool_name": handle.tool_name,
                "model_call_ref": handle.model_call_ref,
                "parent_span_id": handle.parent_span_id,
                "kind": kind,
                "boundary": self.boundary,
                "timing_source": self.timing_source,
                "status": handle.status if kind == "finish" else "incomplete",
                "failure_id": handle.failure_id,
                "capture_failed": handle._capture_failed,
                **measurements,
            }
            self._sequence += 1
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                append_jsonl(self.path, (json.dumps(event, allow_nan=False) + "\n").encode(), durable=False)
            except (OSError, TypeError, ValueError):
                handle._capture_failed = True
                self._failed_spans.add(handle.span_id)
                logger.warning("Could not capture tool attempt %s", handle.span_id, exc_info=True)


def tool_recording_files(directory: str | Path) -> dict[str, str]:
    """Sources for existing SandboxSpec.files or session setup file staging.

    The target directory should also contain the harness's runner, so its Python
    import path resolves ``tool_recording`` without installing Gym in the sandbox.
    """
    source = Path(__file__).parent
    return {
        str(Path(directory) / name): (source / name).read_text() for name in ("tool_recording.py", "jsonl_capture.py")
    }
