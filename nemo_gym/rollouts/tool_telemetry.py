# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool spans and metrics derived from a finished rollout's v2 trajectory record.

Nothing is measured again: each tool execution the harness recorded becomes a `gym.tool` span under the rollout's
dispatch span, and a count and duration in the tool metrics. A span needs both recorded endpoints and a duration
needs a recorded duration; a missing one is left out, never written as zero.
"""

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional

from nemo_gym.core.operation_timing import OperationTiming
from nemo_gym.trajectory.trajectory_record import NG_TRAJECTORY_KEY, TrajectoryToolCall


if TYPE_CHECKING:
    from opentelemetry.trace import SpanContext


logger = logging.getLogger(__name__)

TOOL_SPAN_NAME = "gym.tool"
# The boundary a span takes its endpoints from, when a tool has several: the closest to the execution itself first.
_SPAN_BOUNDARY_ORDER = ("executor", "lifecycle", "transcript", "event_receipt")


def _span_timing(tool: TrajectoryToolCall) -> Optional[OperationTiming]:
    timed = [timing for timing in tool.timings if timing.started_at is not None and timing.completed_at is not None]
    return min(timed, key=lambda timing: _SPAN_BOUNDARY_ORDER.index(timing.boundary), default=None)


def _tool_calls(result: Mapping[str, Any]) -> list[TrajectoryToolCall]:
    trajectory = result.get(NG_TRAJECTORY_KEY)
    if not isinstance(trajectory, Mapping) or trajectory.get("schema_version") != "2.0":
        return []
    return [TrajectoryToolCall.model_validate(tool) for tool in trajectory.get("tool_calls") or []]


def record_tool_telemetry(result: Mapping[str, Any], dispatch_span_context: Optional["SpanContext"]) -> None:
    """Count a finished rollout's tool executions and, under its dispatch span, emit one span each.

    A tool whose recorded parent is another tool goes under that tool's span; parents are emitted first, and the
    record guarantees there are no cycles. Never raises: telemetry must not fail a rollout.
    """
    from nemo_gym.telemetry.gym_metrics import record_tool_call

    try:
        tools = _tool_calls(result)
        for tool in tools:
            record_tool_call(
                tool.tool_name,
                status=tool.status,
                durations={timing.boundary: timing.duration_ms for timing in tool.timings},
            )
        if dispatch_span_context is not None and tools:
            _emit_tool_spans(tools, dispatch_span_context)
    except Exception:
        logger.debug("Could not record tool telemetry.", exc_info=True)


def _emit_tool_spans(tools: list[TrajectoryToolCall], dispatch_span_context: "SpanContext") -> None:
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import NonRecordingSpan, set_span_in_context

    from nemo_gym.telemetry._fallbacks import safe_set_span_attributes
    from nemo_gym.telemetry.gym_metrics import tool_span_attributes

    tracer = otel_trace.get_tracer("nemo.gym")
    # A span id is unique within its agent instance, and a parent is resolved within the same instance.
    by_key = {(tool.agent_instance, tool.span_id): tool for tool in tools if tool.span_id}
    # The span context each tool's children go under: its own span, or its nearest emitted ancestor's.
    emitted: dict[tuple, "SpanContext"] = {}

    def emit(tool: TrajectoryToolCall, parent: "SpanContext") -> Optional["SpanContext"]:
        timing = _span_timing(tool)
        if timing is None:
            return None
        span = tracer.start_span(
            TOOL_SPAN_NAME,
            context=set_span_in_context(NonRecordingSpan(parent)),
            start_time=int(timing.started_at * 1e9),
        )
        safe_set_span_attributes(span, tool_span_attributes(tool, timing))
        span.end(end_time=int(timing.completed_at * 1e9))
        return span.get_span_context()

    for tool in tools:
        # Walk up to the nearest ancestor already emitted (iteratively: chains can be long), then emit downwards.
        chain, seen, current = [], set(), tool
        while current is not None:
            key = (current.agent_instance, current.span_id)
            if current.span_id and (key in emitted or key in seen):
                break
            chain.append(current)
            seen.add(key)
            current = by_key.get((current.agent_instance, current.parent_span_id)) if current.parent_span_id else None
        parent = dispatch_span_context
        if current is not None:
            parent = emitted.get((current.agent_instance, current.span_id), dispatch_span_context)
        for link in reversed(chain):
            # A tool without endpoints has no span; its children go under its nearest emitted ancestor instead,
            # the same whichever order the record lists them in.
            parent = emit(link, parent) or parent
            if link.span_id:
                emitted[(link.agent_instance, link.span_id)] = parent
