# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy turns and tool outcomes from OpenCode's persisted provider steps, messages and tool parts."""

import json
import math
import re
from collections import defaultdict
from sqlite3 import Row
from typing import Any, NamedTuple

from nemo_gym.config import ModelServerRef
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryTurn,
)


def _object(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _timestamp(value: Any) -> float | None:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    return value / 1000


def _native_gap_diagnostics(message_ids: set[str], message_rows: list[Row], part_rows: list[Row]) -> str:
    """Bounded structural evidence only: never persist prompts, arguments or tool output here."""
    selected = set(sorted(message_ids)[:32])

    def metadata(row: Row) -> dict[str, Any]:
        value = _object(row["data"])
        result = {"id": str(row["id"])[:128], "time_created_ms": _timestamp(row["time_created"])}
        # Native times are milliseconds; normalize these separately from row timestamps.
        result["time_created_ms"] = result["time_created_ms"] * 1000 if result["time_created_ms"] is not None else None
        for key in ("role", "type"):
            if isinstance(value.get(key), str):
                result[key] = value[key][:128]
        for key, source in (("time", value), ("state_time", value.get("state"))):
            if not isinstance(source, dict):
                continue
            times = source.get("time")
            if isinstance(times, dict):
                result[key] = {
                    name: timestamp * 1000
                    for name in ("created", "completed", "start", "end")
                    if (timestamp := _timestamp(times.get(name))) is not None
                }
        state = value.get("state")
        if isinstance(state, dict) and isinstance(state.get("status"), str):
            result["state_status"] = state["status"][:128]
        return result

    messages = [row for row in message_rows if row["id"] in selected]
    parts = [row for row in part_rows if row["message_id"] in selected]
    return json.dumps(
        {
            "message_count": len(message_ids),
            "selected_message_ids": sorted(selected),
            "missing_message_ids": sorted(selected - {row["id"] for row in messages}),
            "messages": [metadata(row) for row in messages],
            "part_count": len(parts),
            "parts": [metadata(row) | {"message_id": str(row["message_id"])[:128]} for row in parts[:128]],
            "truncated": len(message_ids) > 32 or len(parts) > 128,
        },
        separators=(",", ":"),
    )


# A Gym Model Server's chat completions reply carries the model's reasoning ahead of its text as one
# <think>...</think> block. OpenCode reads reasoning only from a separate field, so it stores that block as the start
# of a text part. Only a block that starts the part is that reasoning; any other think tag is the model's own text.
_GYM_REASONING_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_gym_reasoning(text: str) -> tuple[str | None, str]:
    """Split the reasoning block a Gym Model Server reply starts with from the rest of an OpenCode text part."""
    match = _GYM_REASONING_BLOCK.match(text)
    if match is None:
        return None, text
    return match.group(1), text[match.end() :].lstrip()


def append_opencode_turns(
    trajectory: TrajectoryRecord,
    session_ids: set[str],
    message_rows: list[Row],
    part_rows: list[Row],
    *,
    model_ref: ModelServerRef | None = None,
    gym_reasoning: bool = False,
    normalized_call_ids: dict[str, str] | None = None,
    interrupted: bool = False,
) -> None:
    """Membership comes only from persisted IDs; row order numbers turns within a session.

    OpenCode also persists assistant messages for user shell commands and subtask
    bookkeeping. Only the model processor writes step-start/step-finish parts.
    A saved user message is not the full provider request, so question stays unset.
    step_count counts persisted step-finish events cumulatively within the session.
    With ``gym_reasoning``, the reasoning block a Gym Model Server reply's text
    starts with becomes the turn's reasoning rather than answer text.
    With ``interrupted`` (a timeout or a rollout stop ended the run), the last assistant
    message of a session may lack its step finish because the stop cut that step: it is
    reported as ``turn_interrupted``, with whatever the step had produced, rather than as
    unavailable evidence.
    """
    messages = {row["id"]: row for row in message_rows}
    last_assistant: dict[str, str] = {}
    if interrupted:
        for row in message_rows:
            message = _object(row["data"])
            if message.get("role") == "assistant" and message.get("summary") is not True:
                last_assistant[row["session_id"]] = row["id"]
    cut: dict[str, str] = {}  # session ID -> the one step a stop or timeout cut in it
    parts = defaultdict(list)
    unavailable = set()

    def missing(session_id: str, message_id: str) -> None:
        if last_assistant.get(session_id) == message_id:
            cut[session_id] = message_id
        else:
            unavailable.add(message_id)

    for row in part_rows:
        message = messages.get(row["message_id"])
        if message is None or row["session_id"] not in (None, message["session_id"]):
            unavailable.add(row["message_id"])
            continue
        part = _object(row["data"])
        if not part:
            unavailable.add(row["message_id"])
        parts[row["message_id"]].append((row["id"], part))

    turn_counts = defaultdict(int)
    step_counts = defaultdict(int)
    for row in message_rows:
        message_id, session_id = row["id"], row["session_id"]
        message = _object(row["data"])
        if not message or session_id not in session_ids:
            unavailable.add(message_id)
            continue
        if message.get("role") == "user":
            continue
        if message.get("role") != "assistant":
            unavailable.add(message_id)
            continue
        if message.get("summary") is True:
            continue
        message_parts = parts[message_id]
        types = {part.get("type") for _, part in message_parts if isinstance(part.get("type"), str)}
        if not types.intersection({"step-start", "step-finish"}):
            parent_id = message.get("parentID")
            parent = messages.get(parent_id) if isinstance(parent_id, str) else None
            parent_parts = (
                parts.get(parent_id, []) if parent is not None and parent["session_id"] == session_id else []
            )
            synthetic_parent = any(part.get("type") == "subtask" for _, part in parent_parts) or (
                bool(parent_parts) and all(part.get("synthetic") is True for _, part in parent_parts)
            )
            # A pre-stream error or missing provider steps does not establish an empty response.
            if not synthetic_parent:
                missing(session_id, message_id)
            continue
        turn_counts[session_id] += 1
        step_counts[session_id] += sum(part.get("type") == "step-finish" for _, part in message_parts)
        opencode_time = message.get("time") if isinstance(message.get("time"), dict) else {}
        timestamp = _timestamp(opencode_time.get("created"))
        if timestamp is None:
            timestamp = _timestamp(row["time_created"])
        if timestamp is None:
            unavailable.add(message_id)
            continue
        answer, reasoning, text_parts = [], [], []
        for part_id, part in message_parts:
            kind = part.get("type")
            if not isinstance(kind, str):
                unavailable.add(message_id)
                continue
            if kind in ("text", "reasoning") and part.get("synthetic") is not True and part.get("ignored") is not True:
                if not isinstance(part.get("text"), str):
                    unavailable.add(message_id)
                    continue
                if kind == "text":
                    reply_reasoning, text = (
                        split_gym_reasoning(part["text"]) if gym_reasoning else (None, part["text"])
                    )
                    if reply_reasoning is not None:
                        reasoning.append({"id": f"{part_id}-reasoning", "type": "reasoning", "text": reply_reasoning})
                    if text or reply_reasoning is None:
                        text_parts.append({"id": part_id, "type": "output_text", "text": text})
                else:
                    reasoning.append({"id": part_id, "type": "reasoning", "text": part["text"]})
            elif kind == "tool":
                state = part.get("state")
                if not isinstance(state, dict) or not isinstance(part.get("tool"), str):
                    unavailable.add(message_id)
                    continue
                answer.append(
                    {
                        "id": part_id,
                        "type": "function_call",
                        "call_id": (normalized_call_ids or {}).get(part_id, part.get("callID")),
                        "name": part["tool"],
                        "arguments": state.get("input"),
                    }
                )
            elif kind not in {"step-start", "step-finish", "snapshot", "patch", "retry", "text", "reasoning"}:
                unavailable.add(message_id)
        if "step-finish" not in types:
            missing(session_id, message_id)
            if not answer and not reasoning and not text_parts:
                continue
        # Match the invocation's Responses shape: native text-part identity belongs to
        # the output message, not to output_text (which has no id field).
        answer[:0] = [
            {
                "id": part["id"],
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": part["text"], "annotations": []}],
            }
            for part in text_parts
        ]
        trajectory.turns.append(
            TrajectoryTurn(
                invocation_id=session_id,
                source_message_id=message_id,
                source_model_ref=model_ref,
                task_id=trajectory.task_id,
                rollout_id=trajectory.rollout_id,
                turn_no=turn_counts[session_id],
                timestamp=timestamp,
                answer=answer,
                reasoning_content=reasoning or None,
                step_count=step_counts[session_id],
            )
        )
    # A cut step's gap names its session's invocation: when the stop came after the model's reply returned but before
    # OpenCode saved any of it, that invocation's last call has no turn. Both kinds keep the structural diagnostics,
    # which can still tell a cut step from one whose persistence was lost.
    gaps = [("turn_interrupted", session_id, {message_id}) for session_id, message_id in sorted(cut.items())]
    if unavailable:
        gaps.append(("turns_unavailable", None, unavailable))
    for code, invocation_id, message_ids in gaps:
        trajectory.gaps.append(
            ObservationGap(
                code=code,
                invocation_id=invocation_id,
                detail=",".join(sorted(message_ids))
                + "; native_capture_diagnostics="
                + _native_gap_diagnostics(message_ids, message_rows, part_rows),
            )
        )


def scope_opencode_trajectory(trajectory: TrajectoryRecord, body: Any, rollout_id: str) -> TrajectoryRecord:
    extra = body.model_extra or {}
    task_id = next(
        (
            str(extra[key])
            for key in ("task_id", "problem_id", "instance_id", "_ng_task_index")
            if extra.get(key) is not None
        ),
        "unknown",
    )
    return trajectory.model_copy(
        update={
            "task_id": task_id,
            "rollout_id": rollout_id,
            "turns": [
                turn.model_copy(update={"task_id": task_id, "rollout_id": rollout_id}) for turn in trajectory.turns
            ],
        }
    )


def scope_opencode_activation(
    observations: AgentObservationBundle,
    trajectory: TrajectoryRecord,
    *,
    invocation_prefix: str,
    activation_id: int,
) -> None:
    """Match the launch plugin's activation/harness-session IDs at every evidence boundary."""

    def scoped(harness_session_id: str) -> str:
        return f"{invocation_prefix}:{harness_session_id}"

    for record in observations.records:
        if isinstance(record, AgentInvocation):
            record.harness_session_id = record.invocation_id
            record.activation_id = activation_id
            if record.parent_invocation_id is not None:
                record.parent_invocation_id = scoped(record.parent_invocation_id)
        if hasattr(record, "invocation_id"):
            record.invocation_id = scoped(record.invocation_id)
    for gap in [*observations.gaps, *trajectory.gaps]:
        if gap.invocation_id is not None:
            gap.invocation_id = scoped(gap.invocation_id)
    for turn in trajectory.turns:
        turn.invocation_id = scoped(turn.invocation_id)
        turn.activation_id = activation_id


# OpenCode's shell tool reports a timeout or an abort only as this trailer on a completion with a null exit code.
_SHELL_TERMINATION_TRAILER = re.compile(
    r"\n\n<shell_metadata>\n(?:(?P<timeout>shell tool terminated command after exceeding timeout \d+ ms\.[^\n]*)"
    r"|(?P<aborted>User aborted the command))\n</shell_metadata>\Z"
)
_COMPACTED_TOOL_OUTPUT = "[Old tool result content cleared]"


class OpenCodeToolOutcome(NamedTuple):
    """What one persisted OpenCode tool part says about its execution."""

    status: str
    error_type: str | None
    # The model-visible output; for a tool still running when OpenCode stopped, its persisted partial output.
    output: str | None
    gap: str | None


def opencode_tool_outcome(tool: Any, state: dict[str, Any]) -> OpenCodeToolOutcome:
    """Map a tool part's state to its Gym outcome, reading only what OpenCode recorded."""
    opencode_status = state.get("status")
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
    opencode_time = state.get("time") if isinstance(state.get("time"), dict) else {}
    raw_output = state.get("output")
    interrupted = opencode_status == "error" and metadata.get("interrupted") is True
    if opencode_status == "completed" and opencode_time.get("compacted") is not None:
        # OpenCode retains raw output in SQLite but substitutes this literal in later model inputs after pruning.
        output = _COMPACTED_TOOL_OUTPUT
    elif opencode_status in {"running", "pending"} or (interrupted and isinstance(metadata.get("output"), str)):
        # OpenCode streams a running tool's output into its metadata and sends that to the model on an abort.
        output = metadata.get("output") if isinstance(metadata.get("output"), str) else None
    else:
        output = raw_output if raw_output is not None else state.get("error")
    output = None if output is None else str(output)

    if opencode_status in {"running", "pending"}:
        return OpenCodeToolOutcome(
            "incomplete", None, output, "tool_output_partial" if output is not None else "tool_output_unavailable"
        )
    if interrupted:
        return OpenCodeToolOutcome("cancelled", "tool_aborted", output, None)
    if opencode_status == "error":
        return OpenCodeToolOutcome("failed", "tool_error", output, None)
    if opencode_status != "completed":
        return OpenCodeToolOutcome("unknown", None, output, None)
    if tool != "bash":
        return OpenCodeToolOutcome("completed", None, output, None)
    # Completion describes the tool lifecycle, not shell success: a timeout or an abort also completes it.
    exit_code = metadata.get("exit")
    if type(exit_code) is int:
        return OpenCodeToolOutcome(
            "completed" if exit_code == 0 else "failed", None if exit_code == 0 else "tool_error", output, None
        )
    trailer = _SHELL_TERMINATION_TRAILER.search(raw_output) if isinstance(raw_output, str) else None
    if trailer is not None and trailer.group("timeout") is not None:
        return OpenCodeToolOutcome("timeout", "tool_timeout", output, None)
    if trailer is not None:
        return OpenCodeToolOutcome("cancelled", "tool_aborted", output, None)
    return OpenCodeToolOutcome("unknown", None, output, "tool_outcome_unavailable")
