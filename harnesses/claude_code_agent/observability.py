# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read Claude Code's per-run transcripts into Gym observability records."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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


if TYPE_CHECKING:
    from nemo_gym.base_responses_api_model import ModelCallRecord


SOURCE = "claude_code"
_COMPACTION_PROMPT_MARKERS = (
    "Your task is to create a detailed summary of this conversation.",
    "Your task is to create a detailed summary of the conversation so far",
    "Your task is to create a detailed summary of the RECENT portion of the conversation",
)
_COMPACTION_SUMMARY_PREFIX = (
    "This session is being continued from a previous conversation that ran out of context. "
    "The summary below covers the earlier portion of the conversation.\n\n"
)
_COMPACTION_SUFFIX_RE = re.compile(
    r"(?:\n\nIf you need specific details from before compaction "
    r"\(like exact code snippets, error messages, or content you generated\), "
    r"read the full transcript at: [^\n]+)?"
    r"(?:\n\nRecent messages are preserved verbatim\.)?"
    r"(?:\n\nYour REPL VM state has been cleared as part of this compaction\. "
    r"Variables defined in REPL calls before this point are no longer accessible "
    r"— redefine any you still need\.)?"
    r"(?:\n\nContinue the conversation from where it left off without asking the user "
    r"any further questions\. Resume directly — do not acknowledge the summary, do not recap "
    r'what was happening, do not preface with "I\'ll continue" or similar\. '
    r"Pick up the last task as if the break never happened\.)?"
    r"$"
)


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


def _is_compaction_request(request: Any) -> bool:
    if not isinstance(request, dict):
        return False
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    final_message = messages[-1]
    return (
        isinstance(final_message, dict)
        and final_message.get("role") == "user"
        and any(marker in _text(final_message.get("content")) for marker in _COMPACTION_PROMPT_MARKERS)
    )


def _messages_text(response: Any) -> str | None:
    if not isinstance(response, dict) or not isinstance(response.get("content"), list):
        return None
    parts = [
        block.get("text")
        for block in response["content"]
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
    ]
    return "".join(parts) or None


def _normalize_compaction_output(text: str) -> str:
    normalized = re.sub(r"<analysis>[\s\S]*?</analysis>", "", text, count=1)
    summary = re.search(r"<summary>([\s\S]*?)</summary>", normalized)
    if summary is not None:
        normalized = (
            normalized[: summary.start()] + f"Summary:\n{summary.group(1).strip()}" + normalized[summary.end() :]
        )
    return re.sub(r"\n\n+", "\n\n", normalized).strip()


def _matches_compaction_summary(summary: str, model_output: str) -> bool:
    expected = _COMPACTION_SUMMARY_PREFIX + _normalize_compaction_output(model_output)
    return summary.startswith(expected) and _COMPACTION_SUFFIX_RE.fullmatch(summary[len(expected) :]) is not None


def _is_compaction_summary(event: dict[str, Any]) -> bool:
    message = event.get("message")
    return event.get("isCompactSummary") is True or (
        isinstance(message, dict) and message.get("isCompactSummary") is True
    )


def _status(block: dict[str, Any], result: Any) -> str:
    if block.get("is_error") is True:
        return "failed"
    if isinstance(result, dict):
        if result.get("interrupted") is True:
            return "incomplete"
        value = result.get("status")
        if value in {"completed", "failed", "timeout", "cancelled", "incomplete"}:
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
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _compaction_outcome(
    event: dict[str, Any],
    metadata: dict[str, Any],
    *,
    has_completion_marker: bool,
) -> Literal["completed", "failed", "aborted", "unknown"]:
    value = metadata.get("outcome")
    if not isinstance(value, str):
        value = metadata.get("status")
    normalized = value.lower() if isinstance(value, str) else None
    if normalized in {"failed", "failure", "error"}:
        return "failed"
    if normalized in {"aborted", "cancelled", "canceled", "interrupted"}:
        return "aborted"
    message = event.get("message")
    if (
        metadata.get("is_error") is True
        or event.get("is_error") is True
        or (isinstance(message, dict) and message.get("is_error") is True)
    ):
        return "failed"
    if normalized in {"completed", "complete", "success", "succeeded"}:
        return "completed"
    return "completed" if has_completion_marker else "unknown"


def _compaction(event: dict[str, Any], invocation_id: str) -> ContextCompactionObservation | None:
    message = event.get("message")
    is_summary = _is_compaction_summary(event)
    is_boundary = event.get("type") == "system" and event.get("subtype") == "compact_boundary"
    metadata = _metadata(event)
    if not is_summary and not is_boundary and not metadata:
        return None

    trigger = metadata.get("trigger")
    summary = _text(message.get("content")) if is_summary and isinstance(message, dict) else None
    return ContextCompactionObservation(
        invocation_id=invocation_id,
        observed_at=_timestamp(event.get("timestamp")),
        trigger=trigger if isinstance(trigger, str) else None,
        tokens_before=_integer(metadata, "tokensBefore", "preTokens"),
        tokens_after=_integer(metadata, "tokensAfter", "postTokens"),
        outcome=_compaction_outcome(
            event,
            metadata,
            has_completion_marker=is_summary or is_boundary,
        ),
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


def _tool_result(block: dict[str, Any], status: str) -> NeMoGymFunctionCallOutput:
    return NeMoGymFunctionCallOutput(
        call_id=block["tool_use_id"],
        output=_text(block.get("content")),
        status="completed" if status == "completed" else "incomplete",
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


def _would_create_parent_cycle(
    child_id: str,
    parent_id: str,
    parents: dict[str, tuple[str, str, str, int]],
) -> bool:
    seen = {child_id}
    current = parent_id
    while current in parents:
        if current in seen:
            return True
        seen.add(current)
        current = parents[current][0]
    return current in seen


def extract_claude_code_observations(
    config_dir: Path,
    *,
    model_ref: ModelServerRef | None = None,
    root_status: Literal["completed", "failed", "incomplete", "unknown"] = "unknown",
    root_duration_ms: float | None = None,
    root_error_type: str | None = None,
    compaction_attempts: list[dict[str, str]] | None = None,
) -> AgentObservationBundle:
    """Extract exact relationships available in one ``CLAUDE_CONFIG_DIR``.

    Transcript IDs and timestamps are used directly. Missing or ambiguous evidence
    is reported as a gap; the extractor never joins calls by text or proximity.
    """

    gaps: list[ObservationGap] = []
    raw_events = _read_events(Path(config_dir), gaps)
    attempts = [
        ContextCompactionObservation(
            invocation_id=attempt["invocation_id"],
            outcome=attempt["outcome"],
        )
        for attempt in compaction_attempts or []
        if isinstance(attempt, dict)
        and isinstance(attempt.get("invocation_id"), str)
        and attempt.get("invocation_id")
        and attempt.get("outcome") in {"failed", "aborted", "unknown"}
    ]
    if not raw_events and not attempts:
        gaps.append(_gap("agent_transcript_unavailable"))
        return AgentObservationBundle(source=SOURCE, gaps=gaps)
    if not raw_events:
        gaps.append(_gap("agent_transcript_unavailable"))

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
    for index, attempt in enumerate(attempts, start=len(raw_events)):
        events_by_invocation.setdefault(attempt.invocation_id, [])
        first_seen.setdefault(attempt.invocation_id, index)
        gaps.extend(
            (
                _gap("compaction_before_model_call_unavailable", invocation_id=attempt.invocation_id),
                _gap("compaction_after_model_call_unavailable", invocation_id=attempt.invocation_id),
            )
        )

    starts: dict[tuple[str, str], list[tuple[float | None, str]]] = defaultdict(list)
    finishes: dict[tuple[str, str], list[tuple[float | None, str]]] = defaultdict(list)
    parents: dict[str, tuple[str, str, str, int]] = {}
    ambiguous_parents: set[str] = set()
    conversations: dict[str, list[Any]] = defaultdict(list)
    model_calls: dict[str, list[ModelCallRef]] = defaultdict(list)
    compactions = attempts

    for invocation_id, entries in events_by_invocation.items():
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
                is_summary = _is_compaction_summary(event)
                if (
                    previous_compaction is not None
                    and previous_compaction[0] + 1 == entry_index
                    and previous_compaction[1] != is_summary
                ):
                    prior = previous_compaction[2]
                    for field in ("observed_at", "trigger", "tokens_before", "tokens_after", "summary"):
                        if getattr(prior, field) is None:
                            setattr(prior, field, getattr(compaction, field))
                    if prior.outcome == "unknown" or compaction.outcome in {"failed", "aborted"}:
                        prior.outcome = compaction.outcome
                    compaction = prior
                else:
                    compaction.before_model_call = last_model_call
                    compactions.append(compaction)
                    pending_compactions.append(compaction)
                    if last_model_call is None:
                        add_gap("compaction_before_model_call_unavailable")
                previous_compaction = (entry_index, is_summary, compaction)
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
                        items.append(_tool_result(block, tool_status))
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
                            if _would_create_parent_cycle(child_id, invocation_id, parents):
                                gaps.append(
                                    _gap(
                                        "cyclic_subagent_parent",
                                        invocation_id=child_id,
                                        detail=invocation_id,
                                    )
                                )
                            else:
                                parents.setdefault(child_id, parent)
                    else:
                        add_gap("ambiguous_subagent_relation")

        for _ in pending_compactions:
            add_gap("compaction_after_model_call_unavailable")

    for compaction in compactions:
        gaps.append(
            _gap(
                "compaction_model_call_reference_unavailable",
                invocation_id=compaction.invocation_id,
            )
        )
        if compaction.outcome == "unknown":
            gaps.append(_gap("compaction_outcome_unavailable", invocation_id=compaction.invocation_id))
        if compaction.observed_at is None:
            gaps.append(_gap("compaction_timestamp_missing", invocation_id=compaction.invocation_id))

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
        if started_at is not None and completed_at is not None:
            if completed_at >= started_at:
                duration_ms = (completed_at - started_at) * 1000
            else:
                add_tool_gap("tool_timing_invalid")
                completed_at = None
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
        is_root = parent is None and invocation_id not in agent_invocations
        if parent is None and (not is_root or root_status == "unknown"):
            gaps.append(_gap("invocation_outcome_unavailable", invocation_id=invocation_id))
        status = "unknown"
        duration_ms = None
        error_type = None
        if is_root:
            status = root_status
            duration_ms = root_duration_ms
            error_type = root_error_type
        elif parent is not None:
            status = "incomplete" if parent[2] in {"timeout", "cancelled"} else parent[2]
            if parent[2] in {"failed", "timeout", "cancelled"}:
                error_type = parent[2]
        invocations_by_id[invocation_id] = AgentInvocation(
            invocation_id=invocation_id,
            parent_invocation_id=parent[0] if parent else None,
            spawned_by_tool_call_id=parent[1] if parent else None,
            status=status,
            duration_ms=duration_ms,
            error_type=error_type,
            model_calls=model_calls[invocation_id],
            conversation=conversations[invocation_id],
        )

    ordered_ids = sorted(all_invocation_ids, key=lambda invocation_id: (first_seen[invocation_id], invocation_id))

    return AgentObservationBundle(
        source=SOURCE,
        records=[
            *(invocations_by_id[invocation_id] for invocation_id in ordered_ids),
            *tool_calls,
            *compactions,
        ],
        gaps=gaps,
    )


def associate_claude_code_compaction_calls(
    bundle: AgentObservationBundle,
    calls: list[ModelCallRecord],
) -> AgentObservationBundle:
    """Associate hidden summary calls only when Claude's persisted summary matches exactly."""
    if bundle.source != SOURCE:
        return bundle

    result = bundle.model_copy()
    result.records = [
        record.model_copy(update={"model_calls": list(record.model_calls)})
        if isinstance(record, (AgentInvocation, ContextCompactionObservation))
        else record
        for record in bundle.records
    ]
    result.gaps = list(bundle.gaps)
    invocations = {record.invocation_id: record for record in result.records if isinstance(record, AgentInvocation)}
    compactions = [record for record in result.records if isinstance(record, ContextCompactionObservation)]

    def ref_matches_call(reference: ModelCallRef, call: ModelCallRecord) -> bool:
        if reference.model_call_id:
            return reference.model_call_id == call.model_call_id
        return reference.model_ref == call.model_ref and reference.response_id == call.response_id

    def resolve_call_index(reference: ModelCallRef | None) -> int | None:
        if reference is None:
            return None
        matches = [call.call_index for call in calls if ref_matches_call(reference, call)]
        return matches[0] if len(matches) == 1 else None

    owned = {
        index
        for invocation in invocations.values()
        for reference in invocation.model_calls
        for index, call in enumerate(calls)
        if ref_matches_call(reference, call)
    }
    candidates: list[list[int]] = []
    for compaction in compactions:
        before_index = resolve_call_index(compaction.before_model_call)
        after_index = resolve_call_index(compaction.after_model_call)
        model_refs = [
            reference.model_ref
            for reference in (compaction.before_model_call, compaction.after_model_call)
            if reference is not None and reference.model_ref is not None
        ]
        candidates.append(
            [
                index
                for index, call in enumerate(calls)
                if index not in owned
                and compaction.outcome == "completed"
                and not compaction.model_calls
                and compaction.summary is not None
                and call.dialect == "messages"
                and call.status_code is not None
                and 200 <= call.status_code < 300
                and call.error_category is None
                and (call.model_call_id is not None or (call.model_ref is not None and call.response_id is not None))
                and (not model_refs or call.model_ref in model_refs)
                and _is_compaction_request(call.request)
                and (response_text := _messages_text(call.response)) is not None
                and _matches_compaction_summary(compaction.summary, response_text)
                and (
                    compaction.before_model_call is None
                    or (before_index is not None and call.call_index > before_index)
                )
                and (
                    compaction.after_model_call is None or (after_index is not None and call.call_index < after_index)
                )
            ]
        )

    candidate_counts = Counter(index for compaction_candidates in candidates for index in compaction_candidates)
    for compaction, compaction_candidates in zip(compactions, candidates):
        ambiguous = len(compaction_candidates) > 1 or (
            len(compaction_candidates) == 1 and candidate_counts[compaction_candidates[0]] > 1
        )
        if ambiguous:
            result.gaps.append(
                _gap(
                    "compaction_model_call_match_ambiguous",
                    invocation_id=compaction.invocation_id,
                    detail=f"candidate_count={len(compaction_candidates)}",
                )
            )
        if (
            len(compaction_candidates) != 1
            or candidate_counts[compaction_candidates[0]] != 1
            or compaction.invocation_id not in invocations
        ):
            continue
        call = calls[compaction_candidates[0]]
        reference = ModelCallRef(
            model_call_id=call.model_call_id,
            model_ref=call.model_ref,
            response_id=call.response_id,
        )
        compaction.model_calls = [reference]

        invocation = invocations[compaction.invocation_id]
        insertion_index = len(invocation.model_calls)
        for index, existing in enumerate(invocation.model_calls):
            if compaction.after_model_call is not None and existing == compaction.after_model_call:
                insertion_index = index
                break
            if compaction.before_model_call is not None and existing == compaction.before_model_call:
                insertion_index = index + 1
        invocation.model_calls.insert(insertion_index, reference)

    unresolved = {compaction.invocation_id for compaction in compactions if not compaction.model_calls}
    result.gaps = [
        gap
        for gap in result.gaps
        if gap.code != "compaction_model_call_reference_unavailable" or gap.invocation_id in unresolved
    ]
    return result
