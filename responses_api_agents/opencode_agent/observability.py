# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy turns from OpenCode's persisted provider-step and message boundaries."""

import json
import math
from collections import Counter, defaultdict
from sqlite3 import Row
from typing import TYPE_CHECKING, Any

from nemo_gym.config_types import ModelServerRef
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    TrajectoryRecord,
    TrajectoryTurn,
)


if TYPE_CHECKING:
    from nemo_gym.base_responses_api_model import ModelCallRecord


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


def append_opencode_turns(
    trajectory: TrajectoryRecord,
    session_ids: set[str],
    message_rows: list[Row],
    part_rows: list[Row],
    *,
    model_ref: ModelServerRef | None = None,
) -> None:
    """Membership comes only from persisted IDs; row order numbers turns within a session.

    OpenCode also persists assistant messages for user shell commands and subtask
    bookkeeping. Only the model processor writes step-start/step-finish parts.
    A saved user message is not the full provider request, so question stays unset.
    step_count counts persisted step-finish events cumulatively within the session.
    """
    messages = {row["id"]: row for row in message_rows}
    parts = defaultdict(list)
    unavailable = set()
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
                unavailable.add(message_id)
            continue
        turn_counts[session_id] += 1
        step_counts[session_id] += sum(part.get("type") == "step-finish" for _, part in message_parts)
        native_time = message.get("time") if isinstance(message.get("time"), dict) else {}
        timestamp = _timestamp(native_time.get("created"))
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
                    text_parts.append({"id": part_id, "type": "output_text", "text": part["text"]})
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
                        "call_id": part.get("callID"),
                        "name": part["tool"],
                        "arguments": state.get("input"),
                    }
                )
            elif kind not in {"step-start", "step-finish", "snapshot", "patch", "retry", "text", "reasoning"}:
                unavailable.add(message_id)
        if "step-finish" not in types:
            unavailable.add(message_id)
            if not answer and not reasoning and not text_parts:
                continue
        if text_parts:
            answer.insert(0, {"id": message_id, "type": "message", "role": "assistant", "content": text_parts})
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
    if unavailable:
        trajectory.gaps.append(ObservationGap(code="turns_unavailable", detail=",".join(sorted(unavailable))))


def associate_opencode_message_calls(
    bundle: AgentObservationBundle,
    trajectory: TrajectoryRecord | None,
    calls: list["ModelCallRecord"],
) -> tuple[AgentObservationBundle, TrajectoryRecord | None]:
    """Join persisted messages to HTTP attempts using session, message and model-server IDs.

    Run after invocation ownership resolution. Summary attempts belong to compactions;
    calls without a saved message (for example title generation) stay invocation-owned.
    No content or timing fallback is used for old binaries or missing artifacts.
    """
    if bundle.source != "opencode":
        return bundle, trajectory
    result = bundle.model_copy(
        update={
            "records": [
                record.model_copy(update={"model_calls": list(record.model_calls)})
                if isinstance(record, ContextCompactionObservation)
                else record
                for record in bundle.records
            ],
            "gaps": [gap for gap in bundle.gaps if not gap.code.startswith("opencode_message_call_")],
        }
    )
    projected = (
        trajectory.model_copy(
            update={
                "turns": [turn.model_copy(update={"model_calls": list(turn.model_calls)}) for turn in trajectory.turns]
            }
        )
        if trajectory is not None
        else None
    )
    targets: dict[tuple[str, str], list[TrajectoryTurn | ContextCompactionObservation]] = defaultdict(list)
    for turn in projected.turns if projected is not None else []:
        if turn.source_message_id:
            targets[(turn.invocation_id, turn.source_message_id)].append(turn)
    for compaction in result.records:
        if isinstance(compaction, ContextCompactionObservation):
            for message_id in set(compaction.source_message_ids):
                targets[(compaction.invocation_id, message_id)].append(compaction)

    call_counts = Counter(call.model_call_id for call in calls if call.model_call_id)
    owners: dict[str, set[str]] = defaultdict(set)
    for invocation in result.records:
        if isinstance(invocation, AgentInvocation):
            for ref in invocation.model_calls:
                if ref.model_call_id:
                    owners[ref.model_call_id].add(invocation.invocation_id)

    # Recompute these links on repeated finalization, including newly ambiguous evidence.
    attributed_ids = {call.model_call_id for call in calls if call.client_assistant_message_id and call.model_call_id}
    for candidates in targets.values():
        for target in candidates:
            target.model_calls = [ref for ref in target.model_calls if ref.model_call_id not in attributed_ids]

    for call in calls:
        if not call.client_assistant_message_id:
            continue
        candidates = targets.get((call.client_session_id or "", call.client_assistant_message_id), [])
        candidates = [target for target in candidates if target.source_model_ref in (None, call.model_ref)]
        code = None
        if len(candidates) != 1:
            code = "ambiguous" if candidates else "unmatched"
        elif candidates[0].source_model_ref is None or call.model_ref is None:
            code = "scope_unavailable"
        elif not call.model_call_id:
            code = "identity_unavailable"
        elif call_counts[call.model_call_id] != 1:
            code = "ambiguous"
        elif owners[call.model_call_id] != {call.client_session_id}:
            code = "ownership_conflict"
        if code is not None:
            result.gaps.append(
                ObservationGap(
                    code=f"opencode_message_call_{code}",
                    invocation_id=call.client_session_id,
                    detail=f"{call.client_assistant_message_id}:{call.model_call_id}",
                )
            )
            continue
        candidates[0].model_calls.append(
            ModelCallRef(model_call_id=call.model_call_id, model_ref=call.model_ref, response_id=call.response_id)
        )
    result.gaps = list({(gap.code, gap.invocation_id, gap.detail): gap for gap in result.gaps}.values())
    return result, projected


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
