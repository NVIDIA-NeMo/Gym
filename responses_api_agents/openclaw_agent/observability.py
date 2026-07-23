# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseInputItem
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
    model_visible_tool_calls,
)


OpenClawSessionTree = list[tuple[str, str | None, list[dict[str, Any]]]]


def _read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            event = {"raw": line}
        events.append(event if isinstance(event, dict) else {"raw": line})
    return events


def discover_openclaw_session_tree(
    agents_root: Path,
    root_session_id: str,
) -> tuple[OpenClawSessionTree, list[ObservationGap]]:
    """Read exact retained-session lineage from OpenClaw's session stores."""
    agents_root = agents_root.resolve()
    stores = sorted(
        store
        for store in agents_root.glob("*/sessions/sessions.json")
        if not store.is_symlink() and store.resolve().is_relative_to(agents_root)
    )
    if not stores:
        return [], [ObservationGap(code="subagent_hierarchy_unavailable")]

    entries: dict[str, tuple[dict[str, Any], Path]] = {}
    duplicate_keys: set[str] = set()
    gaps: list[ObservationGap] = []
    incomplete = False
    for store in stores:
        try:
            data = json.loads(store.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
            incomplete = True
            gaps.append(ObservationGap(code="agent_session_store_unreadable"))
            continue
        if not isinstance(data, dict):
            incomplete = True
            gaps.append(ObservationGap(code="agent_session_store_unreadable"))
            continue
        for session_key, entry in data.items():
            if not isinstance(session_key, str) or not isinstance(entry, dict):
                incomplete = True
                gaps.append(ObservationGap(code="agent_session_entry_unparseable"))
                continue
            if session_key in entries or session_key in duplicate_keys:
                incomplete = True
                gaps.append(ObservationGap(code="agent_session_identity_ambiguous", detail=session_key))
                entries.pop(session_key, None)
                duplicate_keys.add(session_key)
                continue
            entries[session_key] = (entry, store.parent)

    roots = [key for key, (entry, _) in entries.items() if entry.get("sessionId") == root_session_id]
    if len(roots) != 1:
        gaps.append(
            ObservationGap(
                code="subagent_hierarchy_unavailable",
                detail="root_session_not_found" if not roots else "root_session_ambiguous",
            )
        )
        return [], gaps

    root = roots[0]
    selected = {root}
    parents: dict[str, str | None] = {root: None}
    changed = True
    while changed:
        changed = False
        for key, (entry, _) in entries.items():
            if key in selected:
                continue
            spawned_by = entry.get("spawnedBy")
            parent_key = entry.get("parentSessionKey")
            if spawned_by and parent_key and spawned_by != parent_key:
                if spawned_by in selected or parent_key in selected:
                    incomplete = True
                    gaps.append(
                        ObservationGap(
                            code="subagent_parent_ambiguous",
                            invocation_id=key,
                        )
                    )
                continue
            parent = spawned_by or parent_key
            if isinstance(parent, str) and parent in selected:
                selected.add(key)
                parents[key] = parent
                changed = True

    ordered = [root]
    while len(ordered) < len(selected):
        children = sorted(key for key in selected - set(ordered) if parents.get(key) in ordered)
        if not children:
            incomplete = True
            break
        ordered.extend(children)

    sessions: OpenClawSessionTree = []
    for key in ordered:
        entry, directory = entries[key]
        session_id = entry.get("sessionId")
        candidates: list[Path] = []
        session_file = entry.get("sessionFile")
        if isinstance(session_file, str) and session_file:
            candidates.append(directory / Path(session_file).name)
        if isinstance(session_id, str) and session_id:
            candidates.append(directory / f"{session_id}.jsonl")
        path = next(
            (
                candidate
                for candidate in candidates
                if candidate.is_file()
                and not candidate.is_symlink()
                and candidate.resolve().is_relative_to(agents_root)
            ),
            None,
        )
        events = _read_events(path) if path is not None else []
        if path is None:
            incomplete = True
            gaps.append(
                ObservationGap(
                    code="agent_transcript_unavailable",
                    invocation_id=key,
                    detail=session_id if isinstance(session_id, str) else None,
                )
            )
        sessions.append((key, parents[key], events))

    if incomplete:
        gaps.append(ObservationGap(code="subagent_hierarchy_incomplete"))
    return sessions, gaps


def build_openclaw_observation_tree(
    sessions: Iterable[
        tuple[
            str,
            str | None,
            Iterable[NeMoGymResponseInputItem],
            Iterable[dict[str, Any]],
        ]
    ],
    *,
    source: str = "openclaw",
    model_ref: ModelServerRef | None = None,
) -> AgentObservationBundle:
    """Combine per-session observations after exact store-based lineage discovery."""
    combined = AgentObservationBundle(source=source)
    for invocation_id, parent_id, conversation, events in sessions:
        events = list(events)
        bundle = build_openclaw_observations(
            invocation_id,
            conversation,
            events,
            transcript_available=any(event.get("type") == "message" for event in events),
            source=source,
            prefer_native_session_id=False,
            model_ref=model_ref,
        )
        invocation = next(record for record in bundle.records if isinstance(record, AgentInvocation))
        invocation.parent_invocation_id = parent_id
        combined.records.extend(bundle.records)
        combined.gaps.extend(gap for gap in bundle.gaps if gap.code != "subagent_hierarchy_unavailable")
        if parent_id is not None:
            combined.gaps.append(
                ObservationGap(
                    code="subagent_spawn_tool_unavailable",
                    invocation_id=invocation_id,
                )
            )
    return combined


def _timestamp(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and isfinite(value) and value >= 0:
        return float(value) / 1000 if value >= 100_000_000_000 else float(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()
        except ValueError:
            return None
    return None


def _has_conversation_branches(events: Iterable[dict[str, Any]]) -> bool:
    children: dict[str, int] = {}
    for event in events:
        parent_id = event.get("parentId")
        if isinstance(parent_id, str) and parent_id:
            children[parent_id] = children.get(parent_id, 0) + 1
    return any(count > 1 for count in children.values())


def build_openclaw_observations(
    invocation_id: str,
    conversation: Iterable[NeMoGymResponseInputItem],
    events: Iterable[dict[str, Any]],
    *,
    transcript_available: bool,
    source: str = "openclaw",
    prefer_native_session_id: bool = True,
    model_ref: ModelServerRef | None = None,
) -> AgentObservationBundle:
    events = list(events)
    native_session_id = next(
        (
            event.get("id")
            for event in events
            if isinstance(event, dict)
            and event.get("type") == "session"
            and isinstance(event.get("id"), str)
            and event["id"]
        ),
        None,
    )
    if prefer_native_session_id:
        invocation_id = native_session_id or invocation_id
    conversation = list(conversation)
    invocation = AgentInvocation(invocation_id=invocation_id, conversation=conversation)
    bundle = AgentObservationBundle(
        source=source,
        records=[invocation],
        gaps=[
            ObservationGap(code="subagent_hierarchy_unavailable"),
            ObservationGap(code="model_call_ownership_unavailable"),
            ObservationGap(code="context_compaction_unavailable"),
        ],
    )
    if not transcript_available:
        bundle.gaps.append(ObservationGap(code="agent_transcript_unavailable"))
        return bundle

    response_ids = list(
        dict.fromkeys(
            response_id
            for event in events
            if event.get("type") == "message"
            and isinstance((message := event.get("message")), dict)
            and message.get("role") == "assistant"
            and isinstance((response_id := message.get("responseId")), str)
            and response_id
        )
    )
    if model_ref is not None and response_ids:
        invocation.model_calls = [
            ModelCallRef(model_ref=model_ref, response_id=response_id) for response_id in response_ids
        ]
        bundle.gaps = [gap for gap in bundle.gaps if gap.code != "model_call_ownership_unavailable"]

    if _has_conversation_branches(events):
        bundle.gaps.append(
            ObservationGap(
                code="agent_conversation_branching_unavailable",
                invocation_id=invocation_id,
            )
        )

    bundle.gaps = [gap for gap in bundle.gaps if gap.code != "context_compaction_unavailable"]
    visible_tools = model_visible_tool_calls(conversation)
    tool_counts = Counter(call_id for call_id, _, _ in visible_tools)
    tool_metadata = {
        call_id: (tool_name, status) for call_id, tool_name, status in visible_tools if tool_counts[call_id] == 1
    }
    for call_id, count in tool_counts.items():
        if count > 1:
            bundle.gaps.append(
                ObservationGap(
                    code="tool_call_identity_ambiguous",
                    invocation_id=invocation_id,
                    detail=call_id,
                )
            )
    tools: dict[str, ToolCallObservation] = {}

    for event in events:
        if not isinstance(event, dict) or "raw" in event:
            bundle.gaps.append(ObservationGap(code="agent_artifact_record_unparseable"))
            continue

        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        if event.get("type") == "message" and message.get("role") == "toolResult":
            call_id = message.get("toolCallId") or message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in tool_metadata or call_id in tools:
                bundle.gaps.append(
                    ObservationGap(
                        code="tool_result_unowned",
                        invocation_id=invocation_id,
                        detail=call_id if isinstance(call_id, str) else None,
                    )
                )
                continue

            tool_name, _ = tool_metadata[call_id]
            tool = ToolCallObservation(
                invocation_id=invocation_id,
                tool_call_id=call_id,
                tool_name=tool_name,
            )
            details = message.get("details") if isinstance(message.get("details"), dict) else {}
            duration = details.get("durationMs")
            completed_at = _timestamp(message.get("timestamp"))
            if completed_at is None:
                completed_at = _timestamp(event.get("timestamp"))
            if (
                isinstance(duration, (int, float))
                and not isinstance(duration, bool)
                and isfinite(duration)
                and duration >= 0
                and completed_at is not None
            ):
                tool.duration_ms = float(duration)
                tool.completed_at = completed_at
                tool.started_at = completed_at - float(duration) / 1000
                tool.timing_source = "artifact"

            native_status = details.get("status")
            if message.get("isError") is True or native_status in {"error", "failed"}:
                tool.status = "failed"
            elif native_status in {"completed", "success", "ok"}:
                tool.status = "completed"
            elif native_status == "timeout":
                tool.status = "timeout"
            elif message.get("isError") is False:
                tool.status = "completed"
            else:
                tool.status = "unknown"
            tools[call_id] = tool
            bundle.records.append(tool)
            continue

        if event.get("type") != "compaction":
            continue
        tokens_before = event.get("tokensBefore")
        tokens_after = event.get("tokensAfter")
        summary = event.get("summary")
        compaction = ContextCompactionObservation(
            invocation_id=invocation_id,
            observed_at=_timestamp(event.get("timestamp")),
            trigger=event.get("reason") if isinstance(event.get("reason"), str) else None,
            tokens_before=tokens_before if type(tokens_before) is int and tokens_before >= 0 else None,
            tokens_after=tokens_after if type(tokens_after) is int and tokens_after >= 0 else None,
            outcome="completed",
            summary=summary if isinstance(summary, str) else None,
            first_kept_item_id=(
                event.get("firstKeptEntryId") if isinstance(event.get("firstKeptEntryId"), str) else None
            ),
        )
        bundle.records.append(compaction)
        bundle.gaps.append(
            ObservationGap(
                code="compaction_model_call_boundary_unavailable",
                invocation_id=invocation_id,
            )
        )
        if compaction.summary is None:
            bundle.gaps.append(ObservationGap(code="compaction_summary_unavailable", invocation_id=invocation_id))
        if compaction.tokens_after is None:
            bundle.gaps.append(
                ObservationGap(
                    code="compaction_tokens_after_unavailable",
                    invocation_id=invocation_id,
                )
            )

    for call_id in tool_metadata.keys() - tools.keys():
        bundle.gaps.append(
            ObservationGap(
                code="tool_timing_unavailable",
                invocation_id=invocation_id,
                detail=call_id,
            )
        )
    for tool in tools.values():
        if tool.timing_source is None:
            bundle.gaps.append(
                ObservationGap(
                    code="tool_timing_unavailable",
                    invocation_id=invocation_id,
                    detail=tool.tool_call_id,
                )
            )
        if tool.status == "unknown":
            bundle.gaps.append(
                ObservationGap(
                    code="tool_outcome_unavailable",
                    invocation_id=invocation_id,
                    detail=tool.tool_call_id,
                )
            )
    return bundle
