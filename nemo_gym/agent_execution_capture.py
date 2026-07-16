# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-side execution evidence associated with model-call capture."""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, Mapping, Optional, Sequence
from uuid import uuid4

import orjson
from pydantic import BaseModel, Field, model_validator

from nemo_gym.base_responses_api_model import _validate_rollout_id, maybe_rollout_id_from_run_body
from nemo_gym.config_types import ModelServerRef


logger = logging.getLogger(__name__)

InvocationStatus = Literal["requested", "started", "completed", "failed", "cancelled", "unknown"]
ToolMeasurementScope = Literal["caller_round_trip", "stream_observation_interval"]
ToolSpanStatus = Literal["returned", "raised"]


class AgentInvocationRecord(BaseModel):
    """One explicitly observed agent invocation."""

    id: str
    parent_id: Optional[str] = None
    source: str
    status: InvocationStatus = "started"
    spawn_response_id: Optional[str] = None


class ToolSpanRecord(BaseModel):
    """One observed tool-call interval."""

    id: str
    agent_invocation_id: str
    model_ref: Optional[ModelServerRef] = None
    model_response_id: Optional[str] = None
    tool_call_id: str
    measurement_scope: ToolMeasurementScope
    clock_id: str
    started_ns: int
    ended_ns: int
    started_at: float
    completed_at: float
    duration_ms: float
    status: ToolSpanStatus
    reported_error: Optional[bool] = None

    @model_validator(mode="after")
    def validate_interval(self) -> "ToolSpanRecord":
        if self.ended_ns < self.started_ns:
            raise ValueError("ended_ns must not precede started_ns")
        return self


class ModelCallLink(BaseModel):
    """Attribution from an invocation to a captured model response."""

    agent_invocation_id: str
    model_ref: ModelServerRef
    response_id: str


class AgentExecutionCapture(BaseModel):
    """Raw agent execution evidence for one rollout."""

    rollout_id: str
    agent_server: str
    agent_invocations: list[AgentInvocationRecord]
    tool_spans: list[ToolSpanRecord] = Field(default_factory=list)
    model_call_links: list[ModelCallLink] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "AgentExecutionCapture":
        _validate_rollout_id(self.rollout_id)
        invocation_by_id = {invocation.id: invocation for invocation in self.agent_invocations}
        if len(invocation_by_id) != len(self.agent_invocations):
            raise ValueError("agent invocation ids must be unique")
        if not invocation_by_id or not any(invocation.parent_id is None for invocation in self.agent_invocations):
            raise ValueError("agent invocations must contain at least one root")

        for invocation in self.agent_invocations:
            if invocation.parent_id is not None and invocation.parent_id not in invocation_by_id:
                raise ValueError(f"unknown parent agent invocation: {invocation.parent_id}")
            seen = {invocation.id}
            parent_id = invocation.parent_id
            while parent_id is not None:
                if parent_id in seen:
                    raise ValueError("agent invocation parent links must not contain a cycle")
                seen.add(parent_id)
                parent_id = invocation_by_id[parent_id].parent_id

        if len({span.id for span in self.tool_spans}) != len(self.tool_spans):
            raise ValueError("tool span ids must be unique")
        for span in self.tool_spans:
            if span.agent_invocation_id not in invocation_by_id:
                raise ValueError(f"tool span references unknown agent invocation: {span.agent_invocation_id}")
        model_call_owners: dict[tuple[str, str, str], str] = {}
        for link in self.model_call_links:
            if link.agent_invocation_id not in invocation_by_id:
                raise ValueError(f"model-call link references unknown agent invocation: {link.agent_invocation_id}")
            identity = (link.model_ref.type, link.model_ref.name, link.response_id)
            existing_owner = model_call_owners.setdefault(identity, link.agent_invocation_id)
            if existing_owner != link.agent_invocation_id:
                raise ValueError("one model response cannot belong to multiple agent invocations")
        return self


class AgentExecutionRecorder:
    """Thread-safe in-memory evidence recorder owned by one Agent Server."""

    ROOT_INVOCATION_ID = "root"

    def __init__(
        self,
        rollout_id: str,
        agent_server: str,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
        wall_clock: Callable[[], float] = time.time,
        clock_id: Optional[str] = None,
    ) -> None:
        self.rollout_id = _validate_rollout_id(rollout_id)
        self.agent_server = agent_server
        self._clock = clock
        self._wall_clock = wall_clock
        self.clock_id = clock_id or f"{agent_server}:{os.getpid()}:{uuid4().hex}:monotonic"
        self._lock = threading.Lock()
        self._next_span_index = 0
        self._agent_invocations = [
            AgentInvocationRecord(id=self.ROOT_INVOCATION_ID, source=agent_server, status="started")
        ]
        self._tool_spans: list[ToolSpanRecord] = []
        self._model_call_links: list[ModelCallLink] = []
        self._warnings: list[str] = []

    def add_agent_invocation(
        self,
        invocation_id: str,
        *,
        source: str,
        parent_id: Optional[str] = ROOT_INVOCATION_ID,
        status: InvocationStatus = "requested",
        spawn_response_id: Optional[str] = None,
    ) -> None:
        invocation = AgentInvocationRecord(
            id=invocation_id,
            parent_id=parent_id,
            source=source,
            status=status,
            spawn_response_id=spawn_response_id,
        )
        with self._lock:
            if any(existing.id == invocation_id for existing in self._agent_invocations):
                raise ValueError(f"duplicate agent invocation id: {invocation_id}")
            self._agent_invocations.append(invocation)

    def set_invocation_status(self, invocation_id: str, status: InvocationStatus) -> None:
        with self._lock:
            for invocation in self._agent_invocations:
                if invocation.id == invocation_id:
                    if invocation.status in {"completed", "failed", "cancelled"}:
                        return
                    invocation.status = status
                    return
        raise ValueError(f"unknown agent invocation id: {invocation_id}")

    def record_tool_span(
        self,
        *,
        tool_call_id: str,
        measurement_scope: ToolMeasurementScope,
        started_ns: int,
        ended_ns: int,
        started_at: float,
        completed_at: float,
        status: ToolSpanStatus,
        reported_error: Optional[bool] = None,
        agent_invocation_id: str = ROOT_INVOCATION_ID,
        model_ref: Optional[ModelServerRef] = None,
        model_response_id: Optional[str] = None,
    ) -> ToolSpanRecord:
        with self._lock:
            span = ToolSpanRecord(
                id=f"tool-{self._next_span_index}",
                agent_invocation_id=agent_invocation_id,
                model_ref=model_ref,
                model_response_id=model_response_id,
                tool_call_id=tool_call_id,
                measurement_scope=measurement_scope,
                clock_id=self.clock_id,
                started_ns=started_ns,
                ended_ns=ended_ns,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=(ended_ns - started_ns) / 1_000_000,
                status=status,
                reported_error=reported_error,
            )
            self._next_span_index += 1
            self._tool_spans.append(span)
            return span

    @contextmanager
    def tool_span(
        self,
        *,
        tool_call_id: str,
        measurement_scope: ToolMeasurementScope,
        agent_invocation_id: str = ROOT_INVOCATION_ID,
        model_ref: Optional[ModelServerRef] = None,
        model_response_id: Optional[str] = None,
    ) -> Iterator[None]:
        try:
            started_ns = self._clock()
            started_at = self._wall_clock()
        except Exception:
            self.add_warning("tool_span_recording_failed")
            logger.warning("Agent execution span recording failed.", exc_info=True)
            yield
            return
        status: ToolSpanStatus = "returned"
        try:
            yield
        except BaseException:
            status = "raised"
            raise
        finally:
            try:
                ended_ns = self._clock()
                self.record_tool_span(
                    tool_call_id=tool_call_id,
                    measurement_scope=measurement_scope,
                    started_ns=started_ns,
                    ended_ns=ended_ns,
                    started_at=started_at,
                    completed_at=self._wall_clock(),
                    status=status,
                    agent_invocation_id=agent_invocation_id,
                    model_ref=model_ref,
                    model_response_id=model_response_id,
                )
            except Exception:
                self.add_warning("tool_span_recording_failed")
                logger.warning("Agent execution span recording failed.", exc_info=True)

    def add_model_call_link(
        self,
        *,
        response_id: str,
        model_ref: ModelServerRef,
        agent_invocation_id: str = ROOT_INVOCATION_ID,
    ) -> None:
        with self._lock:
            self._model_call_links.append(
                ModelCallLink(
                    agent_invocation_id=agent_invocation_id,
                    model_ref=model_ref,
                    response_id=response_id,
                )
            )

    def add_warning(self, warning: str) -> None:
        with self._lock:
            if warning not in self._warnings:
                self._warnings.append(warning)

    def capture(self) -> AgentExecutionCapture:
        with self._lock:
            return AgentExecutionCapture(
                rollout_id=self.rollout_id,
                agent_server=self.agent_server,
                agent_invocations=list(self._agent_invocations),
                tool_spans=list(self._tool_spans),
                model_call_links=list(self._model_call_links),
                warnings=list(self._warnings),
            )


class AgentExecutionCaptureStore:
    """One atomic agent-owned capture file per rollout."""

    def __init__(self, root: str | Path, *, create: bool = True) -> None:
        self._root = Path(root)
        if create:
            self._root.mkdir(parents=True, exist_ok=True)

    def path_for(self, rollout_id: str) -> Path:
        return self._root / f"{_validate_rollout_id(rollout_id)}.agent-capture.json"

    def write(self, capture: AgentExecutionCapture) -> None:
        path = self.path_for(capture.rollout_id)
        payload = orjson.dumps(capture.model_dump(), option=orjson.OPT_APPEND_NEWLINE)
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary_path.write_bytes(payload)
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def read(self, rollout_id: str) -> Optional[AgentExecutionCapture]:
        try:
            payload = self.path_for(rollout_id).read_bytes()
        except FileNotFoundError:
            return None
        return AgentExecutionCapture.model_validate_json(payload)

    def clear(self, rollout_id: str) -> None:
        self.path_for(rollout_id).unlink(missing_ok=True)


def clear_agent_execution_captures_for_rollouts(
    records: Sequence[BaseModel | Mapping[str, Any]], capture_dirs: list[Path]
) -> None:
    for directory in capture_dirs:
        store = AgentExecutionCaptureStore(directory, create=False)
        for record in records:
            rollout_id = maybe_rollout_id_from_run_body(record)
            if rollout_id:
                store.clear(rollout_id)


def merge_agent_execution_capture_into_record(record: dict[str, Any], capture_dirs: list[Path]) -> dict[str, Any]:
    """Attach raw agent evidence without changing the agent response or reward."""
    rollout_id = maybe_rollout_id_from_run_body(record)
    if rollout_id is None:
        return record
    for directory in capture_dirs:
        try:
            capture = AgentExecutionCaptureStore(directory, create=False).read(rollout_id)
        except Exception:
            logger.warning("Unable to read agent execution capture for rollout %s.", rollout_id, exc_info=True)
            continue
        if capture is not None:
            record["ng_agent_execution_capture"] = capture.model_dump()
            break
    return record
