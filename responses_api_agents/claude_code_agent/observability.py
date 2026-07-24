# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read Claude Code's per-run transcripts into Gym observability records."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
)


SOURCE = "claude_code"


def _gap(code: str, *, invocation_id: str | None = None, detail: str | None = None) -> ObservationGap:
    return ObservationGap(code=code, invocation_id=invocation_id, detail=detail)


def _timestamp(value: Any) -> float | None:
    try:
        result = (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        )
    except (AttributeError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if all(isinstance(item, dict) and item.get("type") == "text" for item in value):
            return "".join(str(item.get("text") or "") for item in value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _status(block: dict[str, Any], result: Any) -> str:
    if block.get("is_error") is True:
        return "failed"
    if isinstance(result, dict):
        if result.get("interrupted") is True:
            return "incomplete"
        value = result.get("status")
        if value in {"completed", "failed", "timeout", "incomplete"}:
            return value
    # A tool_result block is an explicit terminal observation even when Claude Code
    # does not attach a separate status object.
    return "completed"


def _metadata(event: dict[str, Any]) -> dict[str, Any]:
    message = event.get("message")
    for owner in (event, message if isinstance(message, dict) else {}):
        for key in ("compactMetadata", "compact_metadata"):
            if isinstance(metadata := owner.get(key), dict):
                return metadata
    return {}


def _integer(metadata: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _compaction(event: dict[str, Any], invocation_id: str) -> ContextCompactionObservation | None:
    message = event.get("message")
    is_summary = isinstance(message, dict) and message.get("isCompactSummary") is True
    is_boundary = event.get("type") == "system" and event.get("subtype") == "compact_boundary"
    metadata = _metadata(event)
    if not is_summary and not is_boundary and not metadata:
        return None

    trigger = metadata.get("trigger")
    summary = _text(message.get("content")) if is_summary else None
    return ContextCompactionObservation(
        invocation_id=invocation_id,
        observed_at=_timestamp(event.get("timestamp")),
        trigger=trigger if isinstance(trigger, str) else None,
        tokens_before=_integer(metadata, "tokensBefore", "preTokens"),
        tokens_after=_integer(metadata, "tokensAfter", "postTokens"),
        outcome="completed",
        summary=summary or None,
    )


def _message_id(event: dict[str, Any], block_index: int, kind: str) -> str | None:
    event_id = event.get("uuid")
    if isinstance(event_id, str) and event_id:
        return f"{event_id}:{kind}:{block_index}"
    message = event.get("message")
    response_id = message.get("id") if isinstance(message, dict) else None
    if isinstance(response_id, str) and response_id:
        return f"{response_id}:{kind}:{block_index}"
    return None


def _message(item_id: str, text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(id=item_id, content=[NeMoGymResponseOutputText(text=text, annotations=[])])


def _reasoning(item_id: str, block: dict[str, Any]) -> NeMoGymResponseReasoningItem:
    signature = block.get("signature")
    return NeMoGymResponseReasoningItem(
        id=item_id,
        summary=[NeMoGymSummary(text=block["thinking"], type="summary_text")],
        encrypted_content=signature if isinstance(signature, str) else None,
    )


def _tool_call(tool_call_id: str, block: dict[str, Any]) -> NeMoGymResponseFunctionToolCall:
    return NeMoGymResponseFunctionToolCall(
        arguments=json.dumps(block.get("input", {}), ensure_ascii=False, sort_keys=True),
        call_id=tool_call_id,
        name=block.get("name") if isinstance(block.get("name"), str) else "",
        id=tool_call_id,
        status="completed",
    )


def _tool_result(event: dict[str, Any], block: dict[str, Any]) -> NeMoGymFunctionCallOutput:
    event_id = event.get("uuid")
    return NeMoGymFunctionCallOutput(
        call_id=block["tool_use_id"],
        output=_text(block.get("content")),
        id=event_id if isinstance(event_id, str) else None,
        status="completed",
    )


def _read_events(config_dir: Path, gaps: list[ObservationGap]) -> list[tuple[int, dict[str, Any]]]:
    if not config_dir.is_dir():
        gaps.append(_gap("transcript_dir_missing"))
        return []

    transcript_dir = config_dir / "projects"
    if not transcript_dir.is_dir():
        gaps.append(_gap("transcript_dir_missing", detail="projects"))
        return []

    try:
        # Claude Code stores session and subagent transcripts below ``projects``.
        # Other JSONL files in CLAUDE_CONFIG_DIR may belong to staged skills or
        # unrelated CLI state and must not be interpreted as rollout evidence.
        paths = sorted(transcript_dir.rglob("*.jsonl"))
    except OSError:
        gaps.append(_gap("transcript_dir_unreadable"))
        return []

    events: list[tuple[int, dict[str, Any]]] = []
    for path in paths:
        try:
            lines = path.open(encoding="utf-8", errors="replace")
        except OSError:
            gaps.append(_gap("transcript_unreadable", detail=path.name))
            continue
        with lines:
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, UnicodeError):
                    gaps.append(_gap("malformed_transcript_line", detail=f"{path.name}:{line_number}"))
                    continue
                if not isinstance(event, dict):
                    gaps.append(_gap("invalid_transcript_record", detail=f"{path.name}:{line_number}"))
                    continue
                if not isinstance(event.get("sessionId"), str):
                    continue
                events.append((len(events), event))
    return events


def extract_claude_code_observations(
    config_dir: Path,
    *,
    model_ref: ModelServerRef | None = None,
) -> AgentObservationBundle:
    """Extract exact relationships available in one ``CLAUDE_CONFIG_DIR``.

    Transcript IDs and timestamps are used directly. Missing or ambiguous evidence
    is reported as a gap; the extractor never joins calls by text or proximity.
    """

    gaps: list[ObservationGap] = []
    raw_events = _read_events(Path(config_dir), gaps)
    if not raw_events:
        gaps.append(_gap("agent_transcript_unavailable"))
        return AgentObservationBundle(source=SOURCE, gaps=gaps)

    events_by_invocation: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    first_seen: dict[str, int] = {}
    agent_invocations: set[str] = set()

    for ordinal, event in raw_events:
        agent_id = event.get("agentId")
        invocation_id = agent_id if isinstance(agent_id, str) and agent_id else event["sessionId"]
        events_by_invocation[invocation_id].append((ordinal, event))
        first_seen.setdefault(invocation_id, ordinal)
        if isinstance(agent_id, str) and agent_id:
            agent_invocations.add(invocation_id)

    starts: dict[tuple[str, str], list[tuple[float | None, str]]] = defaultdict(list)
    finishes: dict[tuple[str, str], list[tuple[float | None, str]]] = defaultdict(list)
    parents: dict[str, tuple[str, str, str, int]] = {}
    ambiguous_parents: set[str] = set()
    conversations: dict[str, list[Any]] = defaultdict(list)
    model_calls: dict[str, list[ModelCallRef]] = defaultdict(list)
    compactions: list[ContextCompactionObservation] = []

    for invocation_id, entries in events_by_invocation.items():
        entries.sort(
            key=lambda pair: (
                _timestamp(pair[1].get("timestamp")) is None,
                _timestamp(pair[1].get("timestamp")) or 0,
                pair[0],
            )
        )
        items = conversations[invocation_id]
        refs = model_calls[invocation_id]

        def add_gap(code: str, detail: str | None = None) -> None:
            gaps.append(_gap(code, invocation_id=invocation_id, detail=detail))

        seen_response_ids: set[str] = set()
        last_model_call: ModelCallRef | None = None
        pending_compactions: list[ContextCompactionObservation] = []
        previous_compaction: tuple[int, bool, ContextCompactionObservation] | None = None
        for entry_index, (ordinal, event) in enumerate(entries):
            compaction = _compaction(event, invocation_id)
            if compaction is not None:
                message = event.get("message")
                is_summary = isinstance(message, dict) and message.get("isCompactSummary") is True
                if (
                    previous_compaction is not None
                    and previous_compaction[0] + 1 == entry_index
                    and previous_compaction[1] != is_summary
                ):
                    prior = previous_compaction[2]
                    for field in ("trigger", "tokens_before", "tokens_after", "summary"):
                        if getattr(prior, field) is None:
                            setattr(prior, field, getattr(compaction, field))
                    compaction = prior
                else:
                    compaction.before_model_call = last_model_call
                    compactions.append(compaction)
                    pending_compactions.append(compaction)
                    if last_model_call is None:
                        add_gap("compaction_before_model_call_unavailable")
                previous_compaction = (entry_index, is_summary, compaction)
                if compaction.observed_at is None:
                    add_gap("compaction_timestamp_missing")
            else:
                previous_compaction = None

            message = event.get("message")
            if not isinstance(message, dict):
                continue
            role = message.get("role") or event.get("type")
            content = message.get("content")

            if role == "assistant":
                response_id = message.get("id")
                model_call = None
                if not isinstance(response_id, str) or not response_id:
                    add_gap("model_response_id_missing")
                elif model_ref is not None and response_id not in seen_response_ids:
                    model_call = ModelCallRef(model_ref=model_ref, response_id=response_id)
                    refs.append(model_call)
                    seen_response_ids.add(response_id)
                if pending_compactions:
                    for pending in pending_compactions:
                        pending.after_model_call = model_call
                        if model_call is None:
                            add_gap("compaction_after_model_call_unavailable")
                    pending_compactions.clear()
                if model_call is not None:
                    last_model_call = model_call

                if isinstance(content, list):
                    blocks = content
                elif isinstance(content, str):
                    blocks = [{"type": "text", "text": content}]
                else:
                    add_gap("unsupported_assistant_content_block", type(content).__name__)
                    blocks = []
                for block_index, block in enumerate(blocks):
                    if not isinstance(block, dict):
                        add_gap("invalid_assistant_content")
                        continue
                    block_type = block.get("type")
                    item_id = _message_id(event, block_index, str(block_type or "content"))
                    if block_type == "text":
                        text = block.get("text")
                        if not isinstance(text, str) or not text:
                            continue
                        if item_id is None:
                            add_gap("assistant_item_id_missing")
                            continue
                        items.append(_message(item_id, text))
                    elif block_type == "thinking":
                        thinking = block.get("thinking")
                        if not isinstance(thinking, str) or not thinking:
                            continue
                        if item_id is None:
                            add_gap("reasoning_item_id_missing")
                            continue
                        items.append(_reasoning(item_id, block))
                    elif block_type == "tool_use":
                        tool_call_id = block.get("id")
                        if not isinstance(tool_call_id, str) or not tool_call_id:
                            add_gap("tool_call_id_missing")
                            continue
                        tool_name = block.get("name") if isinstance(block.get("name"), str) else ""
                        items.append(_tool_call(tool_call_id, block))
                        starts[(invocation_id, tool_call_id)].append((_timestamp(event.get("timestamp")), tool_name))
                    else:
                        add_gap(
                            "unsupported_assistant_content_block",
                            block_type if isinstance(block_type, str) else None,
                        )

            elif role in {"user", "system", "developer"}:
                if isinstance(content, str):
                    if content:
                        items.append(NeMoGymEasyInputMessage(role=role, content=content))
                    continue
                if not isinstance(content, list):
                    if content is not None:
                        add_gap("unsupported_user_content_block", type(content).__name__)
                    continue

                tool_results: list[dict[str, Any]] = []
                result_metadata = event.get("toolUseResult")
                for block in content:
                    if not isinstance(block, dict):
                        add_gap("invalid_user_content")
                        continue
                    if block.get("type") == "tool_result":
                        tool_results.append(block)
                        tool_call_id = block.get("tool_use_id")
                        if not isinstance(tool_call_id, str) or not tool_call_id:
                            add_gap("tool_result_id_missing")
                            continue
                        tool_status = _status(block, result_metadata)
                        items.append(_tool_result(event, block))
                        finishes[(invocation_id, tool_call_id)].append(
                            (_timestamp(event.get("timestamp")), tool_status)
                        )
                    elif block.get("type") == "text":
                        if isinstance(block.get("text"), str):
                            items.append(NeMoGymEasyInputMessage(role=role, content=block["text"]))
                        else:
                            add_gap("unsupported_user_content_block", "text")
                    else:
                        block_type = block.get("type")
                        add_gap(
                            "unsupported_user_content_block",
                            block_type if isinstance(block_type, str) else None,
                        )

                child_id = result_metadata.get("agentId") if isinstance(result_metadata, dict) else None
                if isinstance(child_id, str) and child_id:
                    if len(tool_results) == 1 and isinstance(tool_results[0].get("tool_use_id"), str):
                        parent = (
                            invocation_id,
                            tool_results[0]["tool_use_id"],
                            _status(tool_results[0], result_metadata),
                            ordinal,
                        )
                        if child_id in parents and parents[child_id][:2] != parent[:2]:
                            parents.pop(child_id)
                            ambiguous_parents.add(child_id)
                            gaps.append(_gap("conflicting_subagent_parent", invocation_id=child_id))
                        elif child_id not in ambiguous_parents:
                            parents.setdefault(child_id, parent)
                    else:
                        add_gap("ambiguous_subagent_relation")

        for _ in pending_compactions:
            add_gap("compaction_after_model_call_unavailable")

    tool_calls: list[ToolCallObservation] = []
    for invocation_id, tool_call_id in sorted(
        set(starts) | set(finishes), key=lambda key: (first_seen.get(key[0], math.inf), key[1])
    ):
        call_starts = starts.get((invocation_id, tool_call_id), [])
        call_finishes = finishes.get((invocation_id, tool_call_id), [])

        def add_tool_gap(code: str) -> None:
            gaps.append(_gap(code, invocation_id=invocation_id, detail=tool_call_id))

        if len(call_starts) > 1 or len(call_finishes) > 1:
            add_tool_gap("ambiguous_tool_artifact")
            continue
        started_at, tool_name = call_starts[0] if call_starts else (None, "")
        completed_at, tool_status = call_finishes[0] if call_finishes else (None, "incomplete")
        if not call_starts:
            add_tool_gap("tool_start_missing")
        if not call_finishes:
            add_tool_gap("tool_result_missing")
        if call_starts and started_at is None:
            add_tool_gap("tool_start_timestamp_missing")
        if call_finishes and completed_at is None:
            add_tool_gap("tool_result_timestamp_missing")
        duration_ms = None
        if started_at is not None and completed_at is not None and completed_at >= started_at:
            duration_ms = (completed_at - started_at) * 1000
        tool_calls.append(
            ToolCallObservation(
                invocation_id=invocation_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name or None,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                timing_source="artifact" if started_at is not None or completed_at is not None else None,
                status=tool_status,
            )
        )

    parent_by_invocation = {child_id: parent[:3] for child_id, parent in parents.items()}
    for child_id, parent in parents.items():
        if child_id not in events_by_invocation:
            first_seen[child_id] = parent[3]
            gaps.append(_gap("subagent_transcript_missing", invocation_id=child_id))

    for invocation_id in agent_invocations - set(parent_by_invocation):
        gaps.append(_gap("subagent_parent_unavailable", invocation_id=invocation_id))

    all_invocation_ids = set(events_by_invocation) | set(parent_by_invocation)
    invocations_by_id: dict[str, AgentInvocation] = {}
    for invocation_id in all_invocation_ids:
        parent = parent_by_invocation.get(invocation_id)
        if parent is None:
            gaps.append(_gap("invocation_outcome_unavailable", invocation_id=invocation_id))
        invocations_by_id[invocation_id] = AgentInvocation(
            invocation_id=invocation_id,
            parent_invocation_id=parent[0] if parent else None,
            spawned_by_tool_call_id=parent[1] if parent else None,
            status=parent[2] if parent else "unknown",
            model_calls=model_calls[invocation_id],
            conversation=conversations[invocation_id],
        )

    def order_key(invocation_id: str) -> tuple[tuple[float, str], ...]:
        path: list[tuple[float, str]] = []
        seen: set[str] = set()
        while invocation_id not in seen:
            seen.add(invocation_id)
            path.append((first_seen.get(invocation_id, math.inf), invocation_id))
            parent = parent_by_invocation.get(invocation_id)
            if parent is None:
                break
            invocation_id = parent[0]
        return tuple(reversed(path))

    ordered_ids = sorted(all_invocation_ids, key=order_key)

    return AgentObservationBundle(
        source=SOURCE,
        records=[
            *(invocations_by_id[invocation_id] for invocation_id in ordered_ids),
            *tool_calls,
            *compactions,
        ],
        gaps=gaps,
    )
