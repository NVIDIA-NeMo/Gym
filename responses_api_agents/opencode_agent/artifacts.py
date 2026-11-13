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

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from openai.types.responses import ResponseInputTextParam

from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ObservationGap,
    ToolCallObservation,
    TrajectoryRecord,
    normalize_operation_timing,
    tool_attempt_id,
)
from responses_api_agents.opencode_agent.observability import (
    append_opencode_turns,
    opencode_tool_outcome,
    split_gym_reasoning,
)


def _load_json(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _milliseconds(value: Any) -> Optional[float]:
    """Convert OpenCode's Date.now()-based epoch milliseconds to seconds."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        return None
    return float(value) / 1000


def _empty_answer_part(parts: list[Any]) -> Optional[int]:
    """Return the index of the text part where an assistant message keeps an empty answer, if any.

    Splitting off a Gym Model Server reply's reasoning block can leave the reply with no answer text. When that reply
    ends its message, with no tool call after it and no answer text in any other text part, the message keeps an empty
    answer there. Otherwise a verifier that reads the last assistant message would grade an earlier turn's text. A
    reply followed by a tool call is a tool-call turn and keeps only its reasoning.
    """
    index = None
    for position, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if part.get("type") == "tool":
            index = None
        elif part.get("type") == "text" and isinstance(text, str) and text.strip():
            reasoning, answer = split_gym_reasoning(text)
            if reasoning is None or answer:
                return None
            index = position
    return index


def _empty_answer_part_ids(part_rows: list[sqlite3.Row], assistant_message_ids: set[str]) -> set[str]:
    """Return the IDs of the persisted text parts where an assistant message keeps an empty answer."""
    rows_by_message: dict[str, list[sqlite3.Row]] = {}
    for row in part_rows:
        if row["message_id"] in assistant_message_ids:
            rows_by_message.setdefault(row["message_id"], []).append(row)
    part_ids = set()
    for rows in rows_by_message.values():
        index = _empty_answer_part([_load_json(row["data"]) for row in rows])
        if index is not None:
            part_ids.add(rows[index]["id"])
    return part_ids


def _parse_opencode_session(
    db_path: Path,
    fallback_invocation_id: str,
    trajectory: Optional[TrajectoryRecord] = None,
    *,
    model_ref: ModelServerRef | None = None,
    require_terminal_finish: bool = False,
    message_ids: set[str] | None = None,
    agent_instance: str | None = None,
    clock_domain: str | None = None,
    observation_schema_version: Literal["1.0", "2.0"] = "1.0",
    interrupted: bool = False,
) -> AgentObservationBundle:
    """Read persisted sessions; opt into v2 for qualified timings and stable attempt IDs.

    ``interrupted`` says that a timeout or a rollout stop ended the run, so the step it cut is not missing evidence.
    """
    if observation_schema_version not in {"1.0", "2.0"}:
        raise ValueError("observation_schema_version must be 1.0 or 2.0")
    qualified = observation_schema_version == "2.0"
    if not db_path.is_file():
        return AgentObservationBundle(
            schema_version=observation_schema_version,
            source="opencode",
            records=[AgentInvocation(invocation_id=fallback_invocation_id)],
            gaps=[
                ObservationGap(code="agent_artifact_unavailable"),
                ObservationGap(code="agent_transcript_unavailable"),
                ObservationGap(code="model_call_ownership_unavailable"),
            ],
        )

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        session_rows = con.execute(
            "select id, parent_id, time_created from session order by time_created, id"
        ).fetchall()
        message_rows = con.execute(
            "select id, session_id, data, time_created from message order by time_created, id"
        ).fetchall()
        part_rows = con.execute(
            "select id, message_id, session_id, data, time_created from part order by time_created, id"
        ).fetchall()
    finally:
        con.close()

    # Persisted part identity is stable even when later attempts reuse a native call ID.
    # These are observation IDs only; model execution retains OpenCode's native callID.
    decoded_parts = {row["id"]: _load_json(row["data"]) for row in part_rows}
    normalized_ids = (
        {row["id"]: f"call:{row['id']}" for row in part_rows if decoded_parts[row["id"]].get("type") == "tool"}
        if qualified
        else {}
    )
    all_part_rows = part_rows
    if message_ids is not None:
        all_messages = {row["id"]: _load_json(row["data"]) for row in message_rows}
        all_message_sessions = {row["id"]: row["session_id"] for row in message_rows}
        message_rows = [row for row in message_rows if row["id"] in message_ids]
        part_rows = [row for row in part_rows if row["message_id"] in message_ids]
        active_sessions = {row["session_id"] for row in message_rows}
        session_rows = [row for row in session_rows if row["id"] in active_sessions or row["parent_id"] is None]
    messages = {row["id"]: _load_json(row["data"]) for row in message_rows}
    message_sessions = {row["id"]: row["session_id"] for row in message_rows}
    conversations: dict[str, list[Any]] = {row["id"]: [] for row in session_rows}
    invocation_status: dict[str, str] = {row["id"]: "unknown" for row in session_rows}
    tools: list[ToolCallObservation] = []
    child_tools: dict[str, set[str]] = {}
    child_status: dict[str, str] = {}
    compaction_parts: list[tuple[str, str, float | None, dict[str, Any]]] = []
    gaps: list[ObservationGap] = []
    summary_text: dict[str, list[str]] = {}
    summaries_by_parent: dict[tuple[str, str], list[str]] = {}
    first_item_id_by_message: dict[tuple[str, str], str] = {}
    if message_ids is not None:
        # A compaction can retain an item from an earlier activation. Its identity
        # remains valid context even though that old conversation is not emitted again.
        for row in all_part_rows:
            if row["message_id"] in message_ids or all_messages.get(row["message_id"], {}).get("role") != "assistant":
                continue
            part = _load_json(row["data"])
            text = part.get("text")
            if part.get("type") in {"text", "reasoning"} and isinstance(text, str) and text.strip():
                item_id = row["id"]
                if part["type"] == "text" and model_ref is not None and split_gym_reasoning(text)[0] is not None:
                    # The conversation below emits a split reply's reasoning item first.
                    item_id = f"{row['id']}-reasoning"
            elif part.get("type") == "tool":
                item_id = normalized_ids[row["id"]] if qualified else part.get("callID")
            else:
                continue
            session_id = row["session_id"] or all_message_sessions.get(row["message_id"])
            if isinstance(session_id, str) and isinstance(item_id, str) and item_id:
                first_item_id_by_message.setdefault((session_id, row["message_id"]), item_id)

    for row in message_rows:
        message = messages[row["id"]]
        session_id = row["session_id"]
        if not isinstance(session_id, str) or session_id not in invocation_status:
            gaps.append(ObservationGap(code="agent_artifact_record_unowned", detail=row["id"]))
        elif message.get("role") == "assistant":
            message_time = message.get("time") if isinstance(message.get("time"), dict) else {}
            completed = _milliseconds(message_time.get("completed")) is not None
            if require_terminal_finish:
                # Each model turn can complete while its invocation still has tools/subagents to run.
                finish = message.get("finish")
                if message.get("error") or finish not in {None, "stop", "length", "content-filter", "tool-calls"}:
                    invocation_status[session_id] = "failed"
                else:
                    invocation_status[session_id] = "completed" if finish == "stop" and completed else "incomplete"
            else:
                if isinstance(message.get("error"), dict):
                    invocation_status[session_id] = "failed"
                if invocation_status[session_id] != "failed" and completed:
                    invocation_status[session_id] = "completed"
        if message.get("role") == "assistant" and message.get("summary") is True:
            summary_text[row["id"]] = []
            parent_id = message.get("parentID")
            if isinstance(parent_id, str):
                summaries_by_parent.setdefault((session_id, parent_id), []).append(row["id"])

    empty_answer_part_ids = (
        _empty_answer_part_ids(
            part_rows, {message_id for message_id, message in messages.items() if message.get("role") == "assistant"}
        )
        if model_ref is not None
        else set()
    )
    for row in part_rows:
        part = _load_json(row["data"])
        if not part:
            gaps.append(ObservationGap(code="agent_artifact_record_unparseable"))
            # The part may have been a tool call, so the tool history read is not complete.
            gaps.append(ObservationGap(code="tool_timing_unavailable", detail=row["id"]))
            continue
        ptype = part.get("type")
        message_id = row["message_id"]
        message = messages.get(message_id, {})
        session_id = row["session_id"] or message_sessions.get(message_id)
        if not isinstance(session_id, str):
            gaps.append(ObservationGap(code="agent_artifact_record_unowned"))
            if ptype == "tool":
                # A tool part that belongs to no session is left out, so the tool history read is not complete.
                gaps.append(ObservationGap(code="tool_timing_unavailable", detail=row["id"]))
            continue
        conversation = conversations.setdefault(session_id, [])
        role = message.get("role")
        if ptype == "tool" and role not in ("assistant", "user"):
            # A tool part whose message cannot say who sent it is left out, so the tool history read is not complete.
            gaps.append(ObservationGap(code="tool_timing_unavailable", invocation_id=session_id, detail=row["id"]))

        if ptype == "step-finish":
            continue

        text = part.get("text")
        if ptype == "text" and isinstance(text, str) and text.strip():
            gym_reasoning = None
            if role == "assistant" and model_ref is not None:
                gym_reasoning, text = split_gym_reasoning(text)
            if message_id in summary_text and text:
                summary_text[message_id].append(text)
            if role == "user" and part.get("ignored") is not True:
                conversation.append(NeMoGymEasyInputMessage(role="user", content=text))
            elif role == "assistant":
                if gym_reasoning is not None:
                    reasoning_id = f"{row['id']}-reasoning"
                    conversation.append(
                        NeMoGymResponseReasoningItem(
                            id=reasoning_id, summary=[NeMoGymSummary(type="summary_text", text=gym_reasoning)]
                        )
                    )
                    first_item_id_by_message.setdefault((session_id, message_id), reasoning_id)
                if text or row["id"] in empty_answer_part_ids:
                    item = NeMoGymResponseOutputMessage(
                        id=row["id"],
                        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                    conversation.append(item)
                    first_item_id_by_message.setdefault((session_id, message_id), row["id"])
            continue
        if ptype == "reasoning" and role == "assistant" and isinstance(text, str) and text.strip():
            conversation.append(
                NeMoGymResponseReasoningItem(
                    id=row["id"],
                    summary=[NeMoGymSummary(type="summary_text", text=text)],
                )
            )
            first_item_id_by_message.setdefault((session_id, message_id), row["id"])
            continue
        if ptype == "tool" and role == "assistant":
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            opencode_call_id = part.get("callID")
            observed_call_id = opencode_call_id if isinstance(opencode_call_id, str) and opencode_call_id else None
            call_id = normalized_ids[row["id"]] if qualified else observed_call_id or f"call-{uuid4().hex[:8]}"
            tool_input = state.get("input") or {}
            arguments = json.dumps(tool_input) if isinstance(tool_input, (dict, list)) else str(tool_input)
            opencode_status = state.get("status")
            response_status = "completed" if opencode_status == "completed" else "incomplete"
            call = NeMoGymResponseFunctionToolCall(
                arguments=arguments,
                call_id=call_id,
                name=part.get("tool", ""),
                type="function_call",
                id=call_id,
                status=response_status,
            )
            conversation.append(call)
            first_item_id_by_message.setdefault((session_id, message_id), call_id)
            opencode_time = state.get("time") if isinstance(state.get("time"), dict) else {}
            outcome = opencode_tool_outcome(part.get("tool"), state)
            if outcome.output is not None:
                result = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=outcome.output,
                    status=response_status,
                )
                conversation.append(result)
            if outcome.gap is not None:
                gaps.append(ObservationGap(code=outcome.gap, invocation_id=session_id, detail=observed_call_id))

            if qualified:
                started_at = _milliseconds(opencode_time.get("start"))
                completed_at = _milliseconds(opencode_time.get("end"))
                duration_ms = (
                    float(opencode_time["end"]) - float(opencode_time["start"])
                    if started_at is not None and completed_at is not None
                    else None
                )
                timing, timing_gaps = normalize_operation_timing(
                    boundary="lifecycle",
                    timing_source="opencode.tool_part",
                    clock_domain=clock_domain,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=duration_ms,
                    duration_source="wall_clock" if duration_ms is not None else None,
                )
                producer_id = f"opencode:{fallback_invocation_id}"
                span_id = tool_attempt_id(producer_id=producer_id, invocation_id=session_id, operation_key=row["id"])
                tools.append(
                    ToolCallObservation(
                        invocation_id=session_id,
                        agent_instance=agent_instance,
                        tool_call_id=call_id,
                        native_tool_call_id=observed_call_id,
                        span_id=span_id,
                        producer_id=producer_id,
                        source_message_id=message_id,
                        source_model_ref=model_ref,
                        tool_name=part.get("tool") if isinstance(part.get("tool"), str) else None,
                        timings=[timing] if timing is not None else [],
                        status=outcome.status,
                        error_type=outcome.error_type,
                    )
                )
                if observed_call_id is None:
                    timing_gaps.append("tool_call_identity_unavailable")
                if started_at is None:
                    timing_gaps.append("tool_start_unavailable")
                if completed_at is None:
                    timing_gaps.append("tool_completion_unavailable")
                gaps.extend(
                    ObservationGap(code=code, invocation_id=session_id, agent_instance=agent_instance, span_id=span_id)
                    for code in timing_gaps
                )
            else:
                opencode_start = opencode_time.get("start")
                opencode_end = opencode_time.get("end")
                valid_interval = (
                    isinstance(opencode_start, (int, float))
                    and not isinstance(opencode_start, bool)
                    and isinstance(opencode_end, (int, float))
                    and not isinstance(opencode_end, bool)
                    and opencode_end >= opencode_start
                )
                started_at = _milliseconds(opencode_start) if valid_interval else None
                completed_at = _milliseconds(opencode_end) if valid_interval else None
                duration_ms = float(opencode_end - opencode_start) if valid_interval else None
                if observed_call_id is not None:
                    tools.append(
                        ToolCallObservation(
                            invocation_id=session_id,
                            tool_call_id=observed_call_id,
                            tool_name=part.get("tool") if isinstance(part.get("tool"), str) else None,
                            started_at=started_at,
                            completed_at=completed_at,
                            duration_ms=duration_ms,
                            timing_source="artifact" if started_at is not None else None,
                            status=outcome.status,
                            error_type=outcome.error_type,
                        )
                    )
                else:
                    gaps.append(
                        ObservationGap(
                            code="tool_call_identity_unavailable",
                            invocation_id=session_id,
                            detail=row["id"],
                        )
                    )
                    # The tool ran but has no observation to time, so the tool history read is not complete.
                    gaps.append(
                        ObservationGap(
                            code="tool_timing_unavailable",
                            invocation_id=session_id,
                            detail=row["id"],
                        )
                    )
                if observed_call_id is not None and (
                    started_at is None or (opencode_status in {"completed", "error"} and completed_at is None)
                ):
                    gaps.append(
                        ObservationGap(
                            code="tool_timing_unavailable",
                            invocation_id=session_id,
                            detail=observed_call_id,
                        )
                    )
            metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
            if not metadata and isinstance(part.get("metadata"), dict):
                metadata = part["metadata"]
            child_id = metadata.get("sessionId")
            if isinstance(child_id, str) and observed_call_id is not None:
                child_tools.setdefault(child_id, set()).add(call_id)
                child_status[child_id] = {"timeout": "incomplete", "cancelled": "incomplete"}.get(
                    outcome.status, outcome.status
                )
            continue
        if ptype == "compaction":
            compaction_parts.append((session_id, message_id, _milliseconds(row["time_created"]), part))
            continue

    compactions: list[ContextCompactionObservation] = []
    for session_id, message_id, observed_at, part in compaction_parts:
        summary_ids = summaries_by_parent.get((session_id, message_id), [])
        summary = "\n".join(summary_text.get(summary_ids[0], [])) if len(summary_ids) == 1 else None
        if len(summary_ids) > 1:
            gaps.append(
                ObservationGap(
                    code="compaction_summary_ambiguous",
                    invocation_id=session_id,
                    detail=",".join(summary_ids),
                )
            )
        trigger = "overflow" if part.get("overflow") is True else "automatic" if part.get("auto") is True else "manual"
        tail_start_id = part.get("tail_start_id") if isinstance(part.get("tail_start_id"), str) else None
        first_kept_item_id = (
            first_item_id_by_message.get((session_id, tail_start_id)) if tail_start_id is not None else None
        )
        compactions.append(
            ContextCompactionObservation(
                invocation_id=session_id,
                source_message_ids=summary_ids,
                source_model_ref=model_ref,
                observed_at=observed_at,
                trigger=trigger,
                outcome="completed" if summary else "unknown",
                summary=summary,
                first_kept_item_id=first_kept_item_id,
            )
        )
        if tail_start_id is not None and first_kept_item_id is None:
            gaps.append(
                ObservationGap(
                    code="compaction_first_kept_item_unavailable",
                    invocation_id=session_id,
                    detail=tail_start_id,
                )
            )
        if not summary:
            gaps.append(ObservationGap(code="compaction_summary_unavailable", invocation_id=session_id))
        gaps.append(ObservationGap(code="compaction_token_counts_unavailable", invocation_id=session_id))
        gaps.append(
            ObservationGap(
                code="compaction_model_call_boundary_unavailable",
                invocation_id=session_id,
            )
        )

    session_ids = {row["id"] for row in session_rows}
    invocations = []
    for row in session_rows:
        invocation_id = row["id"]
        parent_id = row["parent_id"]
        spawn_candidates = child_tools.get(invocation_id, set())
        if parent_id is not None and parent_id not in session_ids:
            gaps.append(
                ObservationGap(
                    code="subagent_parent_unavailable",
                    invocation_id=invocation_id,
                    detail=parent_id,
                )
            )
        if len(spawn_candidates) > 1:
            gaps.append(
                ObservationGap(
                    code="subagent_spawn_ambiguous",
                    invocation_id=invocation_id,
                )
            )
        elif parent_id is not None and not spawn_candidates:
            gaps.append(
                ObservationGap(
                    code="subagent_spawn_tool_unavailable",
                    invocation_id=invocation_id,
                )
            )
        invocations.append(
            AgentInvocation(
                invocation_id=invocation_id,
                agent_instance=agent_instance if qualified else None,
                parent_invocation_id=parent_id,
                spawned_by_tool_call_id=next(iter(spawn_candidates)) if len(spawn_candidates) == 1 else None,
                status=(
                    invocation_status.get(invocation_id, "unknown")
                    if invocation_status.get(invocation_id, "unknown") != "unknown"
                    else child_status.get(invocation_id, "unknown")
                ),
                conversation=conversations.get(invocation_id, []),
            )
        )
    if not invocations:
        invocations = [AgentInvocation(invocation_id=fallback_invocation_id)]
        gaps.append(ObservationGap(code="agent_transcript_unavailable"))

    if trajectory is not None:
        append_opencode_turns(
            trajectory,
            session_ids,
            message_rows,
            part_rows,
            model_ref=model_ref,
            gym_reasoning=model_ref is not None,
            normalized_call_ids=normalized_ids,
            interrupted=interrupted,
        )

    return AgentObservationBundle(
        schema_version=observation_schema_version,
        source="opencode",
        records=[*invocations, *tools, *compactions],
        gaps=gaps,
    )


def parse_opencode_session(
    db_path: Path, *, root_session_only: bool = False, gym_reasoning: bool = False
) -> tuple[list[Any], dict[str, int]]:
    """Convert an OpenCode session database into the existing Gym response shape.

    ``root_session_only`` restricts the transcript to sessions without a
    parent: a sub-agent spawned by the ``task`` tool runs in its own session
    (stored with ``parent_id``), and its parts would otherwise interleave with
    the root conversation. Parts are ordered by creation time and then id,
    OpenCode's own order, because parallel tool parts can share a creation
    millisecond. With ``gym_reasoning``, set when OpenCode calls a Gym Model
    Server, the reasoning block that starts a reply's text is left out like
    OpenCode's own reasoning parts, and a message whose reply was only that
    block keeps an empty answer.
    """
    output_items: list[Any] = []
    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    if not db_path.is_file():
        return output_items, {"input_tokens": 0, "output_tokens": 0}

    scope = " where session_id in (select id from session where parent_id is null)" if root_session_only else ""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        roles = {
            row["id"]: _load_json(row["data"]).get("role")
            for row in con.execute(f"select id, data from message{scope}")
        }
        rows = con.execute(f"select id, message_id, data from part{scope} order by time_created, id").fetchall()
    finally:
        con.close()

    empty_answer_part_ids = (
        _empty_answer_part_ids(rows, {message_id for message_id, role in roles.items() if role == "assistant"})
        if gym_reasoning
        else set()
    )
    for row in rows:
        part = _load_json(row["data"])
        ptype = part.get("type")
        if ptype == "step-finish":
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            input_tokens += int(tokens.get("input") or 0) + int(cache.get("read") or 0) + int(cache.get("write") or 0)
            output_tokens += int(tokens.get("output") or 0) + int(tokens.get("reasoning") or 0)
            reasoning_tokens += int(tokens.get("reasoning") or 0)
        elif roles.get(row["message_id"]) == "assistant" and ptype == "text" and (part.get("text") or "").strip():
            text = split_gym_reasoning(part["text"])[1] if gym_reasoning else part["text"]
            if text or row["id"] in empty_answer_part_ids:
                output_items.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg-{len(output_items)}",
                        content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                        role="assistant",
                        status="completed",
                        type="message",
                    )
                )
        elif roles.get(row["message_id"]) == "assistant" and ptype == "tool":
            state = part.get("state") or {}
            call_id = part.get("callID") or f"call-{uuid4().hex[:8]}"
            tool_input = state.get("input") or {}
            arguments = json.dumps(tool_input) if isinstance(tool_input, (dict, list)) else str(tool_input)
            # A call OpenCode recorded as errored or aborted carries its error
            # text in place of an output; the transcript keeps that outcome.
            status = "completed" if state.get("status") in (None, "completed") else "incomplete"
            result = state.get("output") if state.get("output") is not None else state.get("error")
            output_items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=arguments,
                    call_id=call_id,
                    name=part.get("tool", ""),
                    type="function_call",
                    id=call_id,
                    status=status,
                )
            )
            if result is not None:
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call_id,
                        output=str(result),
                        status=status,
                    )
                )

    return output_items, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
    }


parse_opencode_observations = _parse_opencode_session


def _assistant_part_items(
    part: Dict[str, Any], *, gym_reasoning: bool = False, keep_empty_answer: bool = False
) -> List[NeMoGymResponseOutputItem]:
    """Convert one part of an exported OpenCode assistant message into Gym conversation items.

    ``keep_empty_answer`` marks the part where the message keeps an empty answer (see ``_empty_answer_part``).
    """
    # OpenCode types its own parts, so think tags in text are the model's literal text, except the
    # reasoning block a Gym Model Server reply starts with.
    if part["type"] == "text":
        items: List[NeMoGymResponseOutputItem] = []
        reasoning, text = split_gym_reasoning(part["text"]) if gym_reasoning else (None, part["text"])
        if reasoning is not None:
            items.append(
                NeMoGymResponseReasoningItem(
                    id=f"rs_{uuid4().hex}",
                    type="reasoning",
                    summary=[NeMoGymSummary(type="summary_text", text=reasoning)],
                )
            )
        if text or reasoning is None or keep_empty_answer:
            items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{uuid4().hex}",
                    role="assistant",
                    status="completed",
                    type="message",
                    content=[NeMoGymResponseOutputText(type="output_text", text=text, annotations=[])],
                )
            )
        return items
    if part["type"] == "reasoning":
        return [
            NeMoGymResponseReasoningItem(
                id=f"rs_{uuid4().hex}",
                type="reasoning",
                summary=[NeMoGymSummary(type="summary_text", text=part["text"])],
            )
        ]
    if part["type"] == "tool":
        return [
            NeMoGymResponseFunctionToolCall(
                arguments=json.dumps(part["state"]["input"]),
                call_id=part["callID"],
                name=part["tool"],
            ),
            NeMoGymFunctionCallOutput(
                call_id=part["callID"],
                # @bxyu-nvidia: Somehow the output here may be missing...
                output=part["state"].get("output", ""),
            ),
        ]
    if part["type"] in ("step-finish", "step-start", "patch"):
        return []
    # @bxyu-nvidia: Defensive raise in case we're missing something.
    raise NotImplementedError(part)


def parse_opencode_export(
    opencode_export: Dict[str, Any], *, gym_reasoning: bool = False
) -> List[NeMoGymResponseOutputItem]:
    """Convert persisted OpenCode export messages into Gym conversation items.

    With ``gym_reasoning``, set when OpenCode calls a Gym Model Server, the reasoning block that starts a reply's text
    becomes a reasoning item, and a message whose reply was only that block keeps an empty answer.
    """
    messages = []
    for message in opencode_export["messages"]:
        if message["info"]["role"] == "user":
            message_parts = []
            for part in message["parts"]:
                if part["type"] != "text":
                    continue

                message_parts.append(ResponseInputTextParam(text=part["text"], type="input_text"))

            messages.append(NeMoGymEasyInputMessage(content=message_parts, role="user"))
        elif message["info"]["role"] == "assistant":
            empty_answer = _empty_answer_part(message["parts"]) if gym_reasoning else None
            for index, part in enumerate(message["parts"]):
                messages.extend(
                    _assistant_part_items(part, gym_reasoning=gym_reasoning, keep_empty_answer=index == empty_answer)
                )
        else:
            # @bxyu-nvidia: Defensive raise in case we're missing something.
            raise NotImplementedError(message)

    return messages
