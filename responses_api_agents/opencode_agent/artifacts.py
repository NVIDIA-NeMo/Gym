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
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openai.types.responses import ResponseInputTextParam

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ObservationGap,
    ToolCallObservation,
    TrajectoryRecord,
)
from responses_api_agents.opencode_agent.observability import append_opencode_turns


def _load_json(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _milliseconds(value: Any) -> Optional[float]:
    """Convert OpenCode's Date.now()-based epoch milliseconds to seconds."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        return None
    return float(value) / 1000


def _parse_opencode_session(
    db_path: Path,
    fallback_invocation_id: str,
    trajectory: Optional[TrajectoryRecord] = None,
    *,
    require_terminal_finish: bool = False,
) -> AgentObservationBundle:
    """Read OpenCode's persisted session tree before its workspace is removed."""
    if not db_path.is_file():
        return AgentObservationBundle(
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
    summaries_by_parent: dict[str, list[str]] = {}
    first_item_id_by_message: dict[tuple[str, str], str] = {}

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
        if message.get("summary") is True:
            summary_text[row["id"]] = []
            parent_id = message.get("parentID")
            if isinstance(parent_id, str):
                summaries_by_parent.setdefault(parent_id, []).append(row["id"])

    for row in part_rows:
        part = _load_json(row["data"])
        if not part:
            gaps.append(ObservationGap(code="agent_artifact_record_unparseable"))
            continue
        ptype = part.get("type")
        message_id = row["message_id"]
        message = messages.get(message_id, {})
        session_id = row["session_id"] or message_sessions.get(message_id)
        if not isinstance(session_id, str):
            gaps.append(ObservationGap(code="agent_artifact_record_unowned"))
            continue
        conversation = conversations.setdefault(session_id, [])
        role = message.get("role")

        if ptype == "step-finish":
            continue

        text = part.get("text")
        if ptype == "text" and isinstance(text, str) and text.strip():
            if message_id in summary_text:
                summary_text[message_id].append(text)
            if role == "user" and part.get("ignored") is not True:
                conversation.append(NeMoGymEasyInputMessage(role="user", content=text))
            elif role == "assistant":
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
            native_call_id = part.get("callID")
            observed_call_id = native_call_id if isinstance(native_call_id, str) and native_call_id else None
            call_id = observed_call_id or f"call-{uuid4().hex[:8]}"
            tool_input = state.get("input") or {}
            arguments = json.dumps(tool_input) if isinstance(tool_input, (dict, list)) else str(tool_input)
            native_status = state.get("status")
            response_status = "completed" if native_status == "completed" else "incomplete"
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
            native_time = state.get("time") if isinstance(state.get("time"), dict) else {}
            # OpenCode retains raw output in SQLite but substitutes this literal in later model inputs after pruning.
            if native_status == "completed" and native_time.get("compacted") is not None:
                observed_tool_output = "[Old tool result content cleared]"
            else:
                observed_tool_output = state.get("output") if state.get("output") is not None else state.get("error")
            if observed_tool_output is not None:
                result = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=call_id,
                    output=str(observed_tool_output),
                    status=response_status,
                )
                conversation.append(result)

            native_start = native_time.get("start")
            native_end = native_time.get("end")
            valid_interval = (
                isinstance(native_start, (int, float))
                and not isinstance(native_start, bool)
                and isinstance(native_end, (int, float))
                and not isinstance(native_end, bool)
                and native_end >= native_start
            )
            started_at = _milliseconds(native_start) if valid_interval else None
            completed_at = _milliseconds(native_end) if valid_interval else None
            duration_ms = float(native_end - native_start) if valid_interval else None
            status = {
                "completed": "completed",
                "error": "failed",
                "running": "incomplete",
                "pending": "incomplete",
            }.get(native_status, "unknown")
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
                        status=status,
                        error_type="tool_error" if native_status == "error" else None,
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
            if observed_call_id is not None and (
                started_at is None or (native_status in {"completed", "error"} and completed_at is None)
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
                child_tools.setdefault(child_id, set()).add(observed_call_id)
                child_status[child_id] = status
            continue
        if ptype == "compaction":
            compaction_parts.append((session_id, message_id, _milliseconds(row["time_created"]), part))
            continue

    compactions: list[ContextCompactionObservation] = []
    for session_id, message_id, observed_at, part in compaction_parts:
        summary_ids = summaries_by_parent.get(message_id, [])
        summary = "\n".join(summary_text.get(summary_ids[0], [])) if len(summary_ids) == 1 else None
        if len(summary_ids) > 1:
            gaps.append(
                ObservationGap(
                    code="compaction_summary_ambiguous",
                    invocation_id=session_id,
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
        append_opencode_turns(trajectory, session_ids, message_rows, part_rows)

    return AgentObservationBundle(
        source="opencode",
        records=[*invocations, *tools, *compactions],
        gaps=gaps,
    )


def parse_opencode_session(db_path: Path) -> tuple[list[Any], dict[str, int]]:
    """Convert an OpenCode session database into the existing Gym response shape."""
    output_items: list[Any] = []
    input_tokens = 0
    output_tokens = 0
    if not db_path.is_file():
        return output_items, {"input_tokens": 0, "output_tokens": 0}

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        roles = {row["id"]: _load_json(row["data"]).get("role") for row in con.execute("select id, data from message")}
        rows = con.execute("select message_id, data from part order by time_created").fetchall()
    finally:
        con.close()

    for row in rows:
        part = _load_json(row["data"])
        ptype = part.get("type")
        if ptype == "step-finish":
            tokens = part.get("tokens") or {}
            cache = tokens.get("cache") or {}
            input_tokens += int(tokens.get("input") or 0) + int(cache.get("read") or 0)
            output_tokens += int(tokens.get("output") or 0)
        elif roles.get(row["message_id"]) == "assistant" and ptype == "text" and (part.get("text") or "").strip():
            output_items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg-{len(output_items)}",
                    content=[NeMoGymResponseOutputText(type="output_text", text=part["text"], annotations=[])],
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
            output_items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=arguments,
                    call_id=call_id,
                    name=part.get("tool", ""),
                    type="function_call",
                    id=call_id,
                    status="completed",
                )
            )
            if state.get("output") is not None:
                output_items.append(
                    NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=call_id,
                        output=str(state["output"]),
                        status="completed",
                    )
                )

    return output_items, {"input_tokens": input_tokens, "output_tokens": output_tokens}


parse_opencode_observations = _parse_opencode_session


def parse_opencode_export(opencode_export: Dict[str, Any]) -> List[NeMoGymResponseOutputItem]:
    """Convert persisted OpenCode export messages into Gym conversation items."""
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
            converter = ResponsesConverter(return_token_id_information=True)
            for part in message["parts"]:
                if part["type"] == "text":
                    output_items = converter.postprocess_assistant_message_dict(
                        message_dict={
                            "content": part["text"],
                            "role": "assistant",
                        }
                    )
                    messages.extend(output_items)
                elif part["type"] == "reasoning":
                    output_items = converter.postprocess_assistant_message_dict(
                        message_dict={
                            "content": converter._wrap_reasoning_in_think_tags([part["text"]]),
                            "role": "assistant",
                        }
                    )
                    messages.extend(output_items)
                elif part["type"] == "tool":
                    messages.append(
                        NeMoGymResponseFunctionToolCall(
                            arguments=json.dumps(part["state"]["input"]),
                            call_id=part["callID"],
                            name=part["tool"],
                        )
                    )
                    messages.append(
                        NeMoGymFunctionCallOutput(
                            call_id=part["callID"],
                            # @bxyu-nvidia: Somehow the output here may be missing...
                            output=part["state"].get("output", ""),
                        )
                    )
                elif part["type"] in ("step-finish", "step-start", "patch"):
                    pass
                else:
                    # @bxyu-nvidia: Defensive raise in case we're missing something.
                    raise NotImplementedError(part)
        else:
            # @bxyu-nvidia: Defensive raise in case we're missing something.
            raise NotImplementedError(message)

    return messages


def opencode_export_usages(opencode_export: Dict[str, Any]) -> List[NeMoGymResponseUsage]:
    """Retain the token accounting used by the legacy flat-row adapter."""
    usages: List[NeMoGymResponseUsage] = []
    for message in opencode_export["messages"]:
        if message["info"]["role"] != "assistant":
            continue

        token_info = message["info"].get("tokens")
        if not token_info:
            continue

        usage = NeMoGymResponseUsage(
            input_tokens=token_info["input"],
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=token_info["cache"]["read"]),
            output_tokens=token_info["output"],
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=token_info["reasoning"]),
            total_tokens=token_info.get("total", 0),  # Somehow total may be missing
        )
        usages.append(usage)

    return usages
