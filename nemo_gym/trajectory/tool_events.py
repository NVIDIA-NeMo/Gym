# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and replay producer capture into the shared tool observation contract."""

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nemo_gym.core.operation_timing import TimingBoundary
from nemo_gym.trajectory.tool_timing import normalize_operation_timing
from nemo_gym.trajectory.trajectory_record import ModelCallRef, ObservationGap, ToolCallObservation
from nemo_gym.utils.jsonl_capture import jsonl_lines


class ToolCaptureEvent(BaseModel):
    """Versioned wire record shared with standalone producer hooks."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    schema_version: Literal["2.0"]
    producer_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    span_id: str = Field(min_length=1)
    invocation_id: str = Field(min_length=1)
    agent_instance: str | None = Field(default=None, min_length=1)
    native_call_id: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    model_call_ref: ModelCallRef | None = None
    parent_span_id: str | None = None
    kind: Literal["start", "finish"]
    boundary: TimingBoundary
    timing_source: str = Field(min_length=1)
    clock_domain: str | None = Field(default=None, min_length=1)
    started_at: float | None = None
    completed_at: float | None = None
    duration_ms: float | None = Field(default=None, ge=0)
    duration_source: Literal["monotonic", "reported", "wall_clock"] | None = None
    status: Literal["completed", "failed", "timeout", "cancelled", "incomplete", "unknown"]
    failure_id: str | None = None
    capture_failed: bool = False

    @model_validator(mode="after")
    def check_event_boundary(self) -> "ToolCaptureEvent":
        if self.kind == "start" and (
            self.completed_at is not None or self.duration_ms is not None or self.status != "incomplete"
        ):
            raise ValueError("A start must not contain terminal measurements or an outcome")
        if self.kind == "finish" and self.started_at is not None:
            raise ValueError("The start endpoint belongs to the persisted start event")
        return self


def tool_event_json_schema() -> dict[str, object]:
    """Generate the external-runtime capture schema from the host decoder."""
    return ToolCaptureEvent.model_json_schema()


def _gap(event: ToolCaptureEvent | None, code: str, detail: str | None = None) -> ObservationGap:
    return ObservationGap(
        code=code,
        invocation_id=event.invocation_id if event else None,
        agent_instance=event.agent_instance if event else None,
        span_id=event.span_id if event else None,
        detail=detail,
    )


def read_tool_events(path: str | Path) -> tuple[list[ToolCallObservation], list[ObservationGap]]:
    """Recover valid records even when a producer died midway through its last append."""
    events: list[dict[str, object]] = []
    gaps: list[ObservationGap] = []
    source = Path(path)
    if not source.exists():
        return [], [_gap(None, "tool_capture_unavailable", source.name)]
    try:
        for index, line in enumerate(jsonl_lines(source)):
            try:
                value = json.loads(line)
            except (ValueError, UnicodeError):
                gaps.append(_gap(None, "tool_capture_record_invalid", f"{source.name}:{index + 1}"))
                continue
            if isinstance(value, dict):
                events.append(value)
            else:
                gaps.append(_gap(None, "tool_capture_record_invalid", f"{source.name}:{index + 1}"))
    except OSError:
        gaps.append(_gap(None, "tool_capture_unavailable", source.name))
    tools, replay_gaps = replay_tool_events(events)
    return tools, [*gaps, *replay_gaps]


def replay_tool_events(
    records: Iterable[Mapping[str, object]],
) -> tuple[list[ToolCallObservation], list[ObservationGap]]:
    """Deduplicate producer sequences and fold starts/finishes without clock synthesis.

    This constructs adapter evidence only. The common trajectory normalizer then
    resolves model, instance, parent and failure ownership after capture merge.
    """
    gaps: list[ObservationGap] = []
    by_sequence: dict[tuple[str, int], tuple[str, ToolCaptureEvent]] = {}
    conflicts: set[tuple[str, int]] = set()
    for raw in records:
        values = dict(raw)
        invalid_timing = False
        # A corrupt measurement must not hide the call's identity or known outcome.
        for name in ("started_at", "completed_at", "duration_ms"):
            value = values.get(name)
            if value is not None and (
                type(value) not in (int, float) or not math.isfinite(value) or (name == "duration_ms" and value < 0)
            ):
                values[name] = None
                invalid_timing = True
        for name in ("clock_domain", "duration_source"):
            value = values.get(name)
            valid = (
                isinstance(value, str) and bool(value.strip())
                if name == "clock_domain"
                else value in ("monotonic", "reported", "wall_clock")
            )
            if value is not None and not valid:
                values[name] = None
                invalid_timing = True
        try:
            event = ToolCaptureEvent.model_validate(values)
            fingerprint = json.dumps(dict(raw), sort_keys=True)
        except (ValidationError, TypeError, ValueError):
            gaps.append(_gap(None, "tool_capture_record_invalid"))
            continue
        if invalid_timing:
            gaps.append(_gap(event, "tool_timing_invalid"))
        key = (event.producer_id, event.sequence)
        previous = by_sequence.get(key)
        if previous is not None and previous[0] != fingerprint:
            conflicts.add(key)
            gaps.append(_gap(previous[1], "tool_capture_sequence_conflict"))
            if previous[1].span_id != event.span_id:
                gaps.append(_gap(event, "tool_capture_sequence_conflict"))
        else:
            by_sequence[key] = (fingerprint, event)

    attempts: dict[tuple[str, str], list[ToolCaptureEvent]] = defaultdict(list)
    for key, (_, event) in sorted(by_sequence.items()):
        if key not in conflicts:
            attempts[event.producer_id, event.span_id].append(event)
    tools: list[ToolCallObservation] = []
    for group in attempts.values():
        exemplar = group[0]
        identity = (
            "invocation_id",
            "agent_instance",
            "native_call_id",
            "tool_call_id",
            "tool_name",
            "model_call_ref",
            "parent_span_id",
            "boundary",
            "timing_source",
        )
        if any(any(getattr(event, name) != getattr(exemplar, name) for name in identity) for event in group[1:]):
            gaps.append(_gap(exemplar, "tool_attempt_identity_conflict"))
            continue
        starts = [event for event in group if event.kind == "start"]
        finishes = [event for event in group if event.kind == "finish"]
        start = starts[0] if len(starts) == 1 else None
        finish = finishes[0] if len(finishes) == 1 else None
        if len(starts) > 1 or len(finishes) > 1:
            gaps.append(_gap(exemplar, "tool_capture_span_conflict"))
        if start is None:
            gaps.append(_gap(exemplar, "tool_start_unavailable"))
        if finish is None:
            gaps.append(_gap(exemplar, "tool_completion_unavailable"))
        domain = start.clock_domain if start else finish.clock_domain if finish else None
        end = finish.completed_at if finish else None
        duration = finish.duration_ms if finish else None
        if start and finish and (start.clock_domain is None or start.clock_domain != finish.clock_domain):
            # Only independently measured elapsed time survives incomparable wall clocks.
            end = None
            if finish.duration_source == "wall_clock":
                duration = None
            gaps.append(
                _gap(
                    exemplar,
                    "tool_clock_discontinuity"
                    if start.clock_domain and finish.clock_domain
                    else "tool_clock_alignment_unavailable",
                )
            )
        timing, codes = normalize_operation_timing(
            boundary=exemplar.boundary,
            timing_source=exemplar.timing_source,
            started_at=start.started_at if start else None,
            completed_at=end,
            duration_ms=duration,
            duration_source=finish.duration_source if finish else None,
            clock_domain=domain,
        )
        gaps.extend(_gap(exemplar, code) for code in codes)
        if any(event.capture_failed for event in group):
            gaps.append(_gap(exemplar, "tool_capture_unavailable"))
        call_id = exemplar.tool_call_id or exemplar.native_call_id
        if call_id is None:
            gaps.append(_gap(exemplar, "tool_call_identity_unavailable"))
            call_id = f"unattributed:{exemplar.span_id}"
        tools.append(
            ToolCallObservation(
                invocation_id=exemplar.invocation_id,
                agent_instance=exemplar.agent_instance,
                tool_call_id=call_id,
                native_tool_call_id=exemplar.native_call_id,
                tool_name=exemplar.tool_name,
                span_id=exemplar.span_id,
                producer_id=exemplar.producer_id,
                model_call_ref=exemplar.model_call_ref,
                parent_span_id=exemplar.parent_span_id,
                failure_id=finish.failure_id if finish else None,
                status=finish.status if finish else "incomplete",
                timings=[timing] if timing is not None else [],
            )
        )
    return tools, gaps
