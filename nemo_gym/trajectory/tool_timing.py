# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize adapter evidence into attempt-owned tool measurements and exact capture joins."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from nemo_gym.core.operation_timing import OperationTiming, TimingBoundary
from nemo_gym.trajectory.model_call_record import tool_request_evidence
from nemo_gym.trajectory.trajectory_record import (
    AgentInvocation,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
    TrajectoryModelCall,
    TrajectoryToolCall,
    tool_parent_cycles,
)


def tool_attempt_id(*, producer_id: str, invocation_id: str, operation_key: str) -> str:
    """Derive a replay-stable ID from persisted operation identity, never timestamp or content."""
    identity = json.dumps([producer_id, invocation_id, operation_key], separators=(",", ":"))
    return "tool-" + hashlib.sha256(identity.encode()).hexdigest()[:32]


def normalize_operation_timing(
    *,
    boundary: TimingBoundary,
    timing_source: str,
    started_at: object = None,
    completed_at: object = None,
    duration_ms: object = None,
    duration_source: str | None = None,
    clock_domain: str | None = None,
) -> tuple[OperationTiming | None, list[str]]:
    """Salvage usable native measurements without clamping invalid endpoints or inventing time."""
    gaps: list[str] = []

    def number(value: object, name: str) -> float | None:
        if value is None:
            return None
        if type(value) not in (int, float) or not math.isfinite(value) or (name == "duration" and value < 0):
            gaps.append("tool_timing_invalid")
            return None
        return float(value)

    start, end = number(started_at, "start"), number(completed_at, "end")
    duration = number(duration_ms, "duration")
    if start is not None and end is not None and end < start:
        end = None
        if duration_source == "wall_clock":
            duration = None
        gaps.append("tool_clock_discontinuity")
    if start is None and end is None and duration is None:
        return None, [*gaps, "tool_timing_unavailable"]
    if duration is not None and duration_source not in {"monotonic", "reported", "wall_clock"}:
        duration_source = None
        gaps.append("tool_duration_source_unavailable")
    if clock_domain is None:
        gaps.append("tool_clock_domain_unavailable")
    return OperationTiming(
        boundary=boundary,
        timing_source=timing_source,
        clock_domain=clock_domain,
        started_at=start,
        completed_at=end,
        duration_ms=duration,
        duration_source=duration_source if duration is not None else None,
    ), gaps


def _matches_reference(call: TrajectoryModelCall, reference: ModelCallRef) -> bool:
    if reference.model_call_id is not None:
        return (
            call.model_call_id == reference.model_call_id
            and (reference.model_ref is None or call.response_metadata.model_ref == reference.model_ref)
            and (reference.response_id is None or call.response_metadata.response_id == reference.response_id)
        )
    return (
        call.response_metadata.model_ref == reference.model_ref
        and call.response_metadata.response_id == reference.response_id
    )


def _unresolved_gap(gap: ObservationGap, reason: str) -> ObservationGap:
    """Retain an unattributable diagnostic without asserting a canonical span reference."""
    return gap.model_copy(
        update={
            "span_id": None,
            "detail": json.dumps(
                {
                    "unresolved_span_reference": {"span_id": gap.span_id, "reason": reason},
                    "detail": gap.detail,
                }
            ),
        }
    )


def tool_snapshot_advances(previous: str, incoming: str) -> bool:
    """A more definite outcome can replace a partial snapshot, never the reverse."""
    return (previous == "unknown" and incoming != "unknown") or (
        previous == "incomplete" and incoming not in {"unknown", "incomplete"}
    )


def tool_snapshot_output(tool: ToolCallObservation, invocations: Sequence[AgentInvocation]) -> Any | None:
    """Read one explicitly linked output from the same source snapshot as the tool."""
    outputs = [
        item.output
        for invocation in invocations
        if invocation.invocation_id == tool.invocation_id
        and (tool.agent_instance is None or invocation.agent_instance in {None, tool.agent_instance})
        for item in invocation.conversation
        if getattr(item, "type", None) in {"function_call_output", "custom_tool_call_output"}
        and item.call_id == tool.tool_call_id
    ]
    return outputs[0] if len(outputs) == 1 else None


def _import_tool_observations(
    observations: Iterable[ToolCallObservation],
    observation_gaps: Iterable[ObservationGap],
    *,
    capture_key: str,
    producer_id: str,
    invocations: Sequence[AgentInvocation],
    observation_invocations: Sequence[AgentInvocation],
) -> tuple[list[TrajectoryToolCall], list[ObservationGap], set[str]]:
    """Import the complete raw span namespace once, before generating any diagnostics.

    Canonical tools already own qualified IDs. Only raw observations participate in
    raw-gap lookup; model/failure IDs and conversation call references are untouched.
    Gaps carry no producer ID, so an ambiguous local reference stays explicitly unresolved.
    """
    observations = list(observations)
    imported: list[TrajectoryToolCall] = []
    raw_targets: list[tuple[ToolCallObservation, TrajectoryToolCall]] = []
    legacy_spans: set[str] = set()
    for observed in observations:
        tool = TrajectoryToolCall.model_validate(observed.model_dump())
        if tool.span_id is None:
            tool.span_id = tool_attempt_id(
                producer_id=tool.producer_id or producer_id,
                invocation_id=tool.invocation_id,
                operation_key=f"legacy:{tool.tool_call_id}",
            )
            legacy_spans.add(tool.span_id)
        elif not isinstance(observed, TrajectoryToolCall):

            def qualified(span_id: str) -> str:
                return tool_attempt_id(
                    producer_id=tool.producer_id or producer_id,
                    invocation_id=capture_key,
                    operation_key=span_id,
                )

            tool.span_id = qualified(tool.span_id)
            if tool.parent_span_id is not None:
                tool.parent_span_id = qualified(tool.parent_span_id)
        owners = [
            inv
            for inv in invocations
            if inv.invocation_id == tool.invocation_id
            and (tool.agent_instance is None or inv.agent_instance in {None, tool.agent_instance})
        ]
        if len(owners) == 1 and owners[0].agent_instance is not None:
            tool.agent_instance = owners[0].agent_instance
        if not isinstance(observed, TrajectoryToolCall):
            raw_targets.append((observed.model_copy(update={"agent_instance": tool.agent_instance}), tool))
        imported.append(tool)

    # Compare qualified identities across retained and incoming attempts. Replays of
    # one attempt count once; a retained owner makes a reused request ID ambiguous.
    attempts_by_request = defaultdict(set)
    for tool in imported:
        attempts_by_request[(tool.invocation_id, tool.tool_call_id)].add((tool.agent_instance, tool.span_id))
    for observed, tool in raw_targets:
        if len(attempts_by_request[(tool.invocation_id, tool.tool_call_id)]) == 1:
            tool.output = tool_snapshot_output(observed, observation_invocations)

    gaps = []
    for gap in observation_gaps:
        if gap.span_id is None:
            gaps.append(gap.model_copy())
            continue
        matches = {(tool.agent_instance, tool.span_id): tool for raw, tool in raw_targets if gap.matches_tool(raw)}
        if len(matches) != 1:
            gaps.append(_unresolved_gap(gap, "ambiguous" if matches else "unmatched"))
            continue
        target = next(iter(matches.values()))
        gaps.append(
            gap.model_copy(
                update={
                    "span_id": target.span_id,
                    "agent_instance": target.agent_instance,
                    "invocation_id": target.invocation_id,
                }
            )
        )
    return imported, gaps, legacy_spans


def normalize_tool_calls(
    tools: Iterable[ToolCallObservation],
    *,
    capture_key: str,
    producer_id: str,
    model_calls: Sequence[TrajectoryModelCall],
    invocations: Sequence[AgentInvocation],
    failure_events: Mapping[str, object] | None = None,
    observation_gaps: Iterable[ObservationGap] = (),
    observation_invocations: Sequence[AgentInvocation] | None = None,
) -> tuple[list[TrajectoryToolCall], list[ObservationGap]]:
    """Resolve one record per attempt after capture merge, retaining uncertainty as scoped gaps.

    Input-only producer identity/message/model references stay in bundles for replay. Native
    call IDs verify an exact model reference; by themselves they never select a model attempt.
    Pass raw bundle diagnostics via observation_gaps so they cross the same import boundary.
    Returned span-scoped gaps resolve to retained tools; unresolved references keep their raw
    span and reason in detail without claiming a canonical span. Canonical input gaps stay
    with their trajectory and do not cross this raw import boundary again.
    """
    if len({inv.invocation_id for inv in invocations}) != len(invocations):
        raise ValueError("invocation_id must be unique; duplicate IDs across agent instances are unsupported")
    failures = failure_events or {}
    imported, gaps, legacy_spans = _import_tool_observations(
        tools,
        observation_gaps,
        capture_key=capture_key,
        producer_id=producer_id,
        invocations=invocations,
        observation_invocations=invocations if observation_invocations is None else observation_invocations,
    )
    merged: dict[tuple[str | None, str], TrajectoryToolCall] = {}
    request_conflicts: set[tuple[str | None, str]] = set()

    def gap(tool: ToolCallObservation, code: str, detail: str | None = None) -> None:
        gaps.append(
            ObservationGap(
                code=code,
                invocation_id=tool.invocation_id,
                agent_instance=tool.agent_instance,
                span_id=tool.span_id,
                detail=detail,
            )
        )

    for tool in imported:
        if tool.span_id in legacy_spans:
            gap(tool, "tool_attempt_identity_unavailable")
        if not tool.timings and any(
            value is not None for value in (tool.started_at, tool.completed_at, tool.duration_ms)
        ):
            # "artifact" describes delivery, not the measured boundary. It can carry
            # lifecycle, executor or transcript evidence, so preserve it diagnostically.
            boundary = {"executor": "executor", "harness": "lifecycle"}.get(tool.timing_source)
            if boundary is None:
                gap(
                    tool,
                    "tool_legacy_boundary_unavailable",
                    json.dumps(
                        {
                            "started_at": tool.started_at,
                            "completed_at": tool.completed_at,
                            "duration_ms": tool.duration_ms,
                            "timing_source": tool.timing_source,
                        }
                    ),
                )
            else:
                timing, timing_gaps = normalize_operation_timing(
                    boundary=boundary,
                    timing_source=f"legacy.{tool.timing_source}",
                    started_at=tool.started_at,
                    completed_at=tool.completed_at,
                    duration_ms=tool.duration_ms,
                )
                tool.timings = [timing] if timing else []
                for code in timing_gaps:
                    gap(tool, code)
        key = (tool.agent_instance, tool.span_id)
        existing = merged.get(key)
        if existing is None:
            merged[key] = tool
            continue
        identity_fields = ("invocation_id", "tool_call_id", "native_tool_call_id", "producer_id", "source_message_id")
        if any(
            getattr(existing, name) is not None
            and getattr(tool, name) is not None
            and getattr(existing, name) != getattr(tool, name)
            for name in identity_fields
        ):
            gap(tool, "tool_attempt_identity_conflict")
            continue
        # Every supplied qualifier constrains replay; combining snapshots must not widen a join.
        qualifiers = {}
        for snapshot in (existing, tool):
            supplied = snapshot.model_call_ref.model_dump(exclude_none=True) if snapshot.model_call_ref else {}
            for name, value in (("model_call_id", snapshot.model_call_id), ("model_ref", snapshot.source_model_ref)):
                if value is not None:
                    value = value.model_dump() if hasattr(value, "model_dump") else value
                    if name in supplied and supplied[name] != value:
                        request_conflicts.add(key)
                    supplied[name] = value
            for name, value in supplied.items():
                if name in qualifiers and qualifiers[name] != value:
                    request_conflicts.add(key)
                qualifiers[name] = value
        if key not in request_conflicts and (qualifiers.get("model_call_id") or qualifiers.get("response_id")):
            existing.model_call_ref = ModelCallRef.model_validate(qualifiers)
        for timing in tool.timings:
            previous = next((value for value in existing.timings if value.boundary == timing.boundary), None)
            if previous is None:
                existing.timings.append(timing)
            elif previous != timing:
                # A later complete snapshot may add evidence to a persisted start. It may not
                # contradict already observed endpoints/source/domain or mix two instruments.
                old, new = previous.model_dump(), timing.model_dump()
                if all(value is None or new.get(name) == value for name, value in old.items()):
                    existing.timings[existing.timings.index(previous)] = timing
                elif not all(value is None or old.get(name) == value for name, value in new.items()):
                    gap(tool, "tool_timing_conflict", timing.boundary)
        for name in (
            "model_call_id",
            "model_call_ref",
            "source_message_id",
            "source_model_ref",
            "native_tool_call_id",
            "parent_span_id",
            "failure_id",
        ):
            if getattr(existing, name) is None:
                setattr(existing, name, getattr(tool, name))
        advances = tool_snapshot_advances(existing.status, tool.status)
        compatible = existing.status == tool.status
        if advances:
            existing.status = tool.status
        elif tool.status not in {"incomplete", "unknown", existing.status}:
            gap(tool, "tool_outcome_conflict")
        if existing.error_type is None and existing.status == tool.status:
            existing.error_type = tool.error_type
        if tool.output is not None and (advances or compatible):
            if advances or existing.output is None or existing.output == tool.output:
                existing.output = tool.output
            else:
                gap(tool, "tool_output_ambiguous")

    attempts = list(merged.values())
    invocation_index = defaultdict(list)
    for invocation in invocations:
        invocation_index[invocation.invocation_id].append(invocation)
    call_ids = Counter(call.model_call_id for call in model_calls if call.model_call_id)
    for tool in attempts:
        owners = [
            inv
            for inv in invocation_index[tool.invocation_id]
            if tool.agent_instance is None or inv.agent_instance in {None, tool.agent_instance}
        ]
        if len(owners) == 1 and owners[0].agent_instance is not None:
            tool.agent_instance = owners[0].agent_instance
        if tool.agent_instance is None:
            gap(tool, "tool_agent_instance_unavailable")
        if not tool.timings and all(value is None for value in (tool.started_at, tool.completed_at, tool.duration_ms)):
            gap(tool, "tool_timing_unavailable")
        for timing in tool.timings:
            if timing.clock_domain is None:
                gap(tool, "tool_clock_domain_unavailable", timing.boundary)
            if timing.duration_ms is not None and timing.duration_source is None:
                gap(tool, "tool_duration_source_unavailable", timing.boundary)
        if tool.status == "incomplete":
            gap(tool, "tool_completion_unavailable")
        if tool.failure_id is not None and tool.failure_id not in failures:
            gap(tool, "tool_failure_detail_unavailable", tool.failure_id)
            tool.failure_id = None
        elif tool.status in {"failed", "timeout", "cancelled"} and tool.failure_id is None:
            gap(tool, "tool_failure_detail_unavailable")

        if (tool.agent_instance, tool.span_id) in request_conflicts:
            tool.model_call_id = None
            gap(tool, "tool_model_call_reference_conflict")
            continue
        reference = tool.model_call_ref
        if tool.model_call_id:
            if reference is not None and reference.model_call_id not in {None, tool.model_call_id}:
                tool.model_call_id = None
                gap(tool, "tool_model_call_reference_conflict")
                continue
            reference = (
                reference.model_copy(update={"model_call_id": tool.model_call_id})
                if reference is not None
                else ModelCallRef(model_call_id=tool.model_call_id)
            )
        candidates: list[TrajectoryModelCall] = []
        if reference is not None:
            candidates = [
                call
                for call in model_calls
                if _matches_reference(call, reference)
                and (tool.source_model_ref is None or call.response_metadata.model_ref == tool.source_model_ref)
            ]
        elif tool.source_message_id and len(owners) == 1:
            owner = owners[0]
            sessions = {owner.invocation_id}
            if owner.harness_session_id:
                sessions.add(owner.harness_session_id)
            candidates = [
                call
                for call in model_calls
                if call.client_assistant_message_id == tool.source_message_id
                and call.client_session_id in sessions
                and (tool.source_model_ref is None or call.response_metadata.model_ref == tool.source_model_ref)
            ]
        known_mismatch = any(
            tool.agent_instance is not None
            and call.agent_instance is not None
            and call.agent_instance != tool.agent_instance
            for call in candidates
        )
        candidates = [
            call
            for call in candidates
            if tool.agent_instance is None or call.agent_instance in {None, tool.agent_instance}
        ]
        native_id = tool.native_tool_call_id or tool.tool_call_id
        # Exact message identity can span failed HTTP retries; the requested native ID selects
        # only successful responses which actually contained this request.
        evidence = {
            id(call): tool_request_evidence(
                call.response,
                status_code=call.response_metadata.status_code,
                response_status=call.response_metadata.response_status,
            )
            for call in candidates
        }
        rejected = {item.kind for item in evidence.values()} & {"malformed", "unsuccessful"}
        candidates = [call for call in candidates if evidence[id(call)].permits_link(native_id)]
        for kind in sorted(rejected):
            gap(tool, f"tool_request_payload_{kind}")
        if len(candidates) != 1 or not candidates[0].model_call_id or call_ids[candidates[0].model_call_id] != 1:
            tool.model_call_id = None
            gap(tool, "tool_model_call_ambiguous" if len(candidates) > 1 else "tool_model_call_unavailable")
            if known_mismatch:
                gap(tool, "tool_agent_instance_mismatch")
            continue
        call = candidates[0]
        tool.model_call_id = call.model_call_id
        if tool.agent_instance is None and call.agent_instance is not None:
            tool.agent_instance = call.agent_instance
        if evidence[id(call)].kind == "omitted":
            gap(tool, "tool_request_payload_unavailable")

    # A captured call can resolve previously unknown attribution. Merge again only when
    # that authoritative resolution made two snapshots identify the same attempt.
    if len({(tool.agent_instance, tool.span_id) for tool in attempts}) != len(attempts):
        attempts, resolved_gaps = normalize_tool_calls(
            attempts,
            capture_key=capture_key,
            producer_id=producer_id,
            model_calls=model_calls,
            invocations=invocations,
            failure_events=failure_events,
        )
        gaps.extend(resolved_gaps)

    for tool in attempts:
        owners = [inv for inv in invocation_index[tool.invocation_id] if inv.agent_instance == tool.agent_instance]
        if tool.parent_span_id is None and len(owners) == 1:
            owner = owners[0]
            if owner.parent_invocation_id is not None and owner.spawned_by_tool_call_id is not None:
                parents = [
                    parent
                    for parent in attempts
                    if parent.agent_instance == tool.agent_instance
                    and parent.invocation_id == owner.parent_invocation_id
                    and parent.tool_call_id == owner.spawned_by_tool_call_id
                ]
                if len(parents) == 1:
                    tool.parent_span_id = parents[0].span_id
                else:
                    gap(tool, "tool_parent_ambiguous" if parents else "tool_parent_unavailable")
    identities = {(tool.agent_instance, tool.span_id) for tool in attempts}
    for tool in attempts:
        if tool.parent_span_id and (tool.agent_instance, tool.parent_span_id) not in identities:
            gap(tool, "tool_parent_unavailable", tool.parent_span_id)
            tool.parent_span_id = None
    cyclic = tool_parent_cycles(attempts)
    for tool in attempts:
        if (tool.agent_instance, tool.span_id) in cyclic:
            gap(tool, "tool_parent_cycle", tool.parent_span_id)
            tool.parent_span_id = None
    # A conversation result belongs to one explicit attempt. The one-to-many legacy join is ambiguous.
    by_request = defaultdict(list)
    for tool in attempts:
        by_request[(tool.invocation_id, tool.tool_call_id)].append(tool)
    for invocation in invocations:
        outputs = defaultdict(list)
        for item in invocation.conversation:
            if getattr(item, "type", None) in {"function_call_output", "custom_tool_call_output"}:
                outputs[item.call_id].append(item.output)
        for call_id, values in outputs.items():
            candidates = by_request.get((invocation.invocation_id, call_id), [])
            matches = []
            for tool in candidates:
                if (
                    invocation.agent_instance is not None
                    and tool.agent_instance is not None
                    and invocation.agent_instance != tool.agent_instance
                ):
                    gap(tool, "tool_agent_instance_mismatch")
                else:
                    matches.append(tool)
            if len(matches) == 1 and len(values) == 1 and matches[0].output is None:
                matches[0].output = values[0]
            elif len(matches) > 1:
                for tool in matches:
                    if tool.output is None:
                        gap(tool, "tool_output_ambiguous")
    # Exact model ownership may resolve after import/merge. Bind imported and generated
    # diagnostics to the retained attempt only when that target is unique; never fan out.
    scoped_gaps = []
    for item in gaps:
        if item.code == "tool_completion_unavailable" and item.span_id is not None:
            matches = [tool for tool in attempts if item.matches_tool(tool)]
            if (
                len(matches) == 1
                and matches[0].status not in {"unknown", "incomplete"}
                and (
                    matches[0].completed_at is not None or any(t.completed_at is not None for t in matches[0].timings)
                )
            ):
                continue
        if item.span_id is None:
            scoped_gaps.append(item)
            continue
        matches = [tool for tool in attempts if item.matches_tool(tool)]
        if len(matches) == 1:
            target = matches[0]
            scoped_gaps.append(
                item.model_copy(
                    update={
                        "agent_instance": target.agent_instance,
                        "invocation_id": target.invocation_id,
                    }
                )
            )
        else:
            scoped_gaps.append(_unresolved_gap(item, "ambiguous" if matches else "unmatched"))
    return attempts, list(
        {
            (gap.code, gap.agent_instance, gap.invocation_id, gap.span_id, gap.detail): gap for gap in scoped_gaps
        }.values()
    )
