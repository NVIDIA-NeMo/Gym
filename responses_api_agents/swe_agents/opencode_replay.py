#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for replaying OpenCode main and subagent session trajectories."""

import json
import re
from copy import deepcopy
from typing import Any, Dict, Optional


_TASK_RESULT_SESSION_RE = re.compile(r"(?:^|\n)task_id:\s*(\S+)")


def parse_replay_subagent_payload(problem_info: Dict[str, Any]) -> Optional[dict]:
    """Decode recorded subagent metadata into a versioned manifest envelope.

    Older rollout rows store ``subagent_trajectories`` as a bare list while
    newer callers may already provide ``{"root_session_id": ..., "sessions":
    [...]}``. Responses metadata values are strings on the wire, so both forms
    also accept JSON-encoded input.
    """
    raw = problem_info.get("subagent_trajectories")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(raw, list):
        return {"version": 1, "sessions": raw}
    if isinstance(raw, dict) and isinstance(raw.get("sessions"), list):
        return raw
    return None


def parse_replay_subagent_manifest(problem_info: Dict[str, Any]) -> Optional[dict]:
    """Decode the internal causal subagent replay manifest."""
    raw = problem_info.get("replay_subagent_manifest")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(raw, dict) and isinstance(raw.get("sessions"), list):
        return raw
    return None


def _chat_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _parse_task_arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _task_result_session_id(content: Any) -> Optional[str]:
    match = _TASK_RESULT_SESSION_RE.search(_chat_message_text(content))
    return match.group(1) if match else None


def extract_task_spawn_records(messages: list) -> list[dict]:
    """Return task calls in parent-message order with returned child IDs."""
    calls: list[dict] = []
    task_outputs: dict[str, str] = {}
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for tool_index, tool_call in enumerate(message.get("tool_calls") or []):
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                if not isinstance(function, dict) or function.get("name") != "task":
                    continue
                call_id = tool_call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    continue
                calls.append(
                    {
                        "spawn_call_id": call_id,
                        "spawn_index": len(calls),
                        "spawn_message_index": message_index,
                        "spawn_tool_index": tool_index,
                        "arguments": _parse_task_arguments(function.get("arguments")),
                    }
                )
            continue
        if message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        if not isinstance(call_id, str):
            continue
        child_session_id = _task_result_session_id(message.get("content"))
        if child_session_id:
            task_outputs[call_id] = child_session_id

    for call in calls:
        call["child_session_id"] = task_outputs.get(call["spawn_call_id"])
    return calls


def extract_responses_task_records(items: list) -> list[dict]:
    """Return completed and incomplete ``task`` calls from Responses items.

    The records have the same core fields as :func:`extract_task_spawn_records`.
    A completed call has ``child_session_id`` populated from its matching
    ``function_call_output``. This intentionally keys by call ID so parallel
    task results may arrive in any order.
    """
    task_outputs: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        call_id = item.get("call_id")
        if not isinstance(call_id, str):
            continue
        child_session_id = _task_result_session_id(item.get("output"))
        if child_session_id:
            task_outputs[call_id] = child_session_id

    calls: list[dict] = []
    for item_index, item in enumerate(items):
        if not isinstance(item, dict) or item.get("type") != "function_call" or item.get("name") != "task":
            continue
        call_id = item.get("call_id") or item.get("id")
        if not isinstance(call_id, str) or not call_id:
            continue
        calls.append(
            {
                "spawn_call_id": call_id,
                "spawn_index": len(calls),
                "spawn_item_index": item_index,
                "arguments": _parse_task_arguments(item.get("arguments")),
                "child_session_id": task_outputs.get(call_id),
            }
        )
    return calls


def completed_tool_turn_cut_indices(items: list, *, require_task: bool = False) -> list[int]:
    """Return output indices after complete Responses-format tool turns.

    One assistant turn may contain multiple parallel function calls. It is
    complete only after every call ID in that group has a corresponding
    ``function_call_output``. When ``require_task`` is true, only groups that
    contain at least one ``task`` call are returned.
    """
    pending_call_ids: set[str] = set()
    group_has_task = False
    cuts: list[int] = []
    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id")
            if not isinstance(call_id, str) or not call_id:
                continue
            pending_call_ids.add(call_id)
            group_has_task = group_has_task or item.get("name") == "task"
            continue
        if item_type != "function_call_output":
            continue
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or call_id not in pending_call_ids:
            continue
        pending_call_ids.remove(call_id)
        if not pending_call_ids:
            if group_has_task or not require_task:
                cuts.append(item_index)
            group_has_task = False
    return cuts


def build_replay_prefix_row(
    rollout_row: dict,
    *,
    source_line: Optional[int] = None,
    strategy: str = "first-task-batch",
    cut_output_index: Optional[int] = None,
    strict: bool = True,
) -> dict:
    """Convert a completed rollout row into a replay-ready prefix row.

    The result intentionally contains only the request plus top-level replay
    provenance; completed response, reward, and verifier artifacts from the
    source rollout are not copied into the new rollout input.
    """
    create_params = rollout_row.get("responses_create_params")
    response = rollout_row.get("response")
    output = response.get("output") if isinstance(response, dict) else None
    if not isinstance(create_params, dict) or not isinstance(create_params.get("input"), list):
        raise ValueError("source row needs responses_create_params.input as a list")
    if not isinstance(output, list):
        raise ValueError("source row needs response.output as a list")

    if cut_output_index is None:
        if strategy == "first-task-batch":
            cuts = completed_tool_turn_cut_indices(output, require_task=True)
            if not cuts:
                raise ValueError("source rollout has no completed task tool turn")
            cut_output_index = cuts[0]
        elif strategy == "last-tool-turn":
            cuts = completed_tool_turn_cut_indices(output)
            if not cuts:
                raise ValueError("source rollout has no completed tool turn")
            cut_output_index = cuts[-1]
        else:
            raise ValueError(f"unknown replay prefix strategy: {strategy}")
    if cut_output_index < 0 or cut_output_index >= len(output):
        raise ValueError(f"cut output index {cut_output_index} is outside response.output with {len(output)} items")
    if cut_output_index not in completed_tool_turn_cut_indices(output):
        raise ValueError(f"cut output index {cut_output_index} is not a complete tool-turn boundary")

    prefix_output = deepcopy(output[: cut_output_index + 1])
    prefix_create_params = deepcopy(create_params)
    prefix_create_params["input"] = deepcopy(create_params["input"]) + prefix_output
    metadata = dict(prefix_create_params.get("metadata") or {})

    payload = parse_replay_subagent_payload(
        {"subagent_trajectories": rollout_row.get("subagent_trajectories") or metadata.get("subagent_trajectories")}
    )
    task_calls = extract_responses_task_records(prefix_output)
    truncated_payload = truncate_replay_subagent_payload(task_calls, payload, strict=strict) if payload else None
    if any(call.get("child_session_id") for call in task_calls) and truncated_payload is None and strict:
        raise ValueError("prefix contains a completed task call but no matching subagent trajectory")
    if truncated_payload is not None:
        metadata["subagent_trajectories"] = json.dumps(truncated_payload, separators=(",", ":"))
    else:
        metadata.pop("subagent_trajectories", None)
    prefix_create_params["metadata"] = metadata

    result = {
        "responses_create_params": prefix_create_params,
        "replay_provenance": {
            "cut_output_index": cut_output_index,
            "strategy": strategy,
        },
    }
    if source_line is not None:
        result["replay_provenance"]["source_line"] = source_line
    if isinstance(rollout_row.get("agent_ref"), dict):
        result["agent_ref"] = deepcopy(rollout_row["agent_ref"])
    return result


def _truncate_session_messages(messages: list, invocation_count: int, session_id: str) -> list:
    """Keep the first ``invocation_count`` user-delimited child invocations."""
    user_indices = [
        index for index, message in enumerate(messages) if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not user_indices:
        raise ValueError(f"recorded subagent {session_id} has no user message")
    if invocation_count <= 0:
        return []
    if invocation_count >= len(user_indices):
        return list(messages)
    return list(messages[: user_indices[invocation_count]])


def truncate_replay_subagent_payload(
    root_task_calls: list[dict],
    payload: dict,
    *,
    strict: bool = True,
) -> Optional[dict]:
    """Return only subagent history causally visible after ``root_task_calls``.

    A child trajectory contains one user-delimited message segment for its
    initial spawn and one more for every later ``task(task_id=...)`` resume.
    The selected main prefix therefore determines how many such segments are
    safe to retain. The same rule is applied recursively to task calls inside
    retained child messages, which excludes nested children that had not yet
    branched when the parent prefix stopped.

    Only task calls with a recorded result are considered complete. This makes
    the function safe for prefixes ending at a complete tool-turn boundary.
    """
    raw_sessions = payload.get("sessions")
    if not isinstance(raw_sessions, list) or not raw_sessions:
        return None

    sessions_by_id: dict[str, dict] = {}
    for raw_session in raw_sessions:
        if not isinstance(raw_session, dict):
            if strict:
                raise ValueError("subagent replay entries must be objects")
            continue
        session_id = raw_session.get("session_id")
        parent_session_id = raw_session.get("parent_session_id")
        messages = raw_session.get("messages")
        if not (isinstance(session_id, str) and isinstance(parent_session_id, str) and isinstance(messages, list)):
            if strict:
                raise ValueError("each subagent replay entry needs session_id, parent_session_id, and messages")
            continue
        if session_id in sessions_by_id:
            if strict:
                raise ValueError(f"duplicate subagent replay session_id: {session_id}")
            continue
        sessions_by_id[session_id] = dict(raw_session)
    if not sessions_by_id:
        return None

    root_session_id = payload.get("root_session_id")
    if not isinstance(root_session_id, str) or not root_session_id:
        root_candidates = {
            session["parent_session_id"]
            for session in sessions_by_id.values()
            if session["parent_session_id"] not in sessions_by_id
        }
        if len(root_candidates) != 1:
            if strict:
                raise ValueError(
                    "cannot infer one replay root session from subagent parent_session_id values; "
                    "provide root_session_id explicitly"
                )
            root_session_id = sorted(root_candidates)[0] if root_candidates else "main"
        else:
            root_session_id = next(iter(root_candidates))

    ordered: list[dict] = []
    visited: set[str] = set()

    def visit(parent_session_id: str, task_calls: list[dict]) -> None:
        invocation_counts: dict[str, int] = {}
        first_calls: dict[str, dict] = {}
        child_order: list[str] = []
        for task_call in task_calls:
            if not isinstance(task_call, dict):
                continue
            child_session_id = task_call.get("child_session_id")
            if not isinstance(child_session_id, str) or not child_session_id:
                continue
            arguments = task_call.get("arguments") or {}
            requested_session_id = arguments.get("task_id") if isinstance(arguments, dict) else None
            if isinstance(requested_session_id, str) and requested_session_id != child_session_id and strict:
                raise ValueError(
                    f"task call {task_call.get('spawn_call_id')} resumes {requested_session_id} "
                    f"but its result identifies {child_session_id}"
                )
            if child_session_id not in invocation_counts:
                invocation_counts[child_session_id] = 0
                first_calls[child_session_id] = task_call
                child_order.append(child_session_id)
            invocation_counts[child_session_id] += 1

        for child_session_id in child_order:
            session = sessions_by_id.get(child_session_id)
            if session is None:
                if strict:
                    raise ValueError(
                        f"completed task call in parent {parent_session_id} returned unknown child {child_session_id}"
                    )
                continue
            if session["parent_session_id"] != parent_session_id:
                if strict:
                    raise ValueError(
                        f"recorded subagent {child_session_id} belongs to {session['parent_session_id']}, "
                        f"not task-call parent {parent_session_id}"
                    )
                continue
            if child_session_id in visited:
                if strict:
                    raise ValueError(f"subagent replay graph revisits {child_session_id}")
                continue
            visited.add(child_session_id)

            truncated = dict(session)
            truncated["messages"] = _truncate_session_messages(
                session["messages"], invocation_counts[child_session_id], child_session_id
            )
            first_call = first_calls[child_session_id]
            truncated.setdefault("spawn_call_id", first_call.get("spawn_call_id"))
            truncated.setdefault("spawn_index", first_call.get("spawn_index"))
            ordered.append(truncated)
            visit(child_session_id, extract_task_spawn_records(truncated["messages"]))

    visit(root_session_id, root_task_calls)
    if not ordered:
        return None
    return {
        "version": payload.get("version", 1),
        "root_session_id": root_session_id,
        "sessions": ordered,
    }


def _first_user_message_text(messages: list) -> str:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return _chat_message_text(message.get("content"))
    return ""


def _order_sessions(root_session_id: str, sessions: list[dict], *, strict: bool) -> list[dict]:
    """Return parent-before-child order, with siblings in parent task-call order."""
    original_index = {session["session_id"]: index for index, session in enumerate(sessions)}
    children: dict[str, list[dict]] = {}
    for session in sessions:
        children.setdefault(session["parent_session_id"], []).append(session)

    def sort_key(session: dict) -> tuple[int, int, int, int]:
        fallback = len(sessions)
        return (
            session.get("spawn_index") if isinstance(session.get("spawn_index"), int) else fallback,
            session.get("spawn_message_index") if isinstance(session.get("spawn_message_index"), int) else fallback,
            session.get("spawn_tool_index") if isinstance(session.get("spawn_tool_index"), int) else fallback,
            original_index[session["session_id"]],
        )

    ordered: list[dict] = []
    visited: set[str] = set()

    def visit(parent_session_id: str) -> None:
        for session in sorted(children.get(parent_session_id, []), key=sort_key):
            session_id = session["session_id"]
            if session_id in visited:
                continue
            visited.add(session_id)
            ordered.append(session)
            visit(session_id)

    visit(root_session_id)
    if len(visited) != len(sessions):
        unreachable = [session["session_id"] for session in sessions if session["session_id"] not in visited]
        if strict:
            raise ValueError(f"subagent sessions are not reachable from replay root {root_session_id}: {unreachable}")
        ordered.extend(session for session in sessions if session["session_id"] not in visited)
    return ordered


def build_replay_subagent_manifest(
    replay_messages_list: list,
    payload: dict,
    *,
    strict: bool = True,
) -> Optional[dict]:
    """Build a causal parent-task-call -> child-session replay manifest.

    ``parent_session_id`` identifies only the parent session. The task call ID
    identifies the exact branch point, including when a parent launches
    multiple children in one parallel tool-call group. For legacy trajectories
    it is recovered from the task result's ``task_id: <session>`` prefix, with
    a unique prompt match as a fallback for an interrupted task whose result
    was never recorded.
    """
    raw_sessions = payload.get("sessions")
    if not isinstance(raw_sessions, list) or not raw_sessions:
        return None

    sessions: list[dict] = []
    seen_session_ids: set[str] = set()
    for raw_session in raw_sessions:
        if not isinstance(raw_session, dict):
            if strict:
                raise ValueError("subagent replay entries must be objects")
            continue
        session_id = raw_session.get("session_id")
        parent_session_id = raw_session.get("parent_session_id")
        messages = raw_session.get("messages")
        if not (isinstance(session_id, str) and isinstance(parent_session_id, str) and isinstance(messages, list)):
            if strict:
                raise ValueError("each subagent replay entry needs session_id, parent_session_id, and messages")
            continue
        if session_id in seen_session_ids:
            if strict:
                raise ValueError(f"duplicate subagent replay session_id: {session_id}")
            continue
        seen_session_ids.add(session_id)
        sessions.append(dict(raw_session))
    if not sessions:
        return None

    session_ids = {session["session_id"] for session in sessions}
    root_session_id = payload.get("root_session_id")
    if not isinstance(root_session_id, str) or not root_session_id:
        root_candidates = {
            session["parent_session_id"] for session in sessions if session["parent_session_id"] not in session_ids
        }
        if len(root_candidates) != 1:
            if strict:
                raise ValueError(
                    "cannot infer one replay root session from subagent parent_session_id values; "
                    "provide root_session_id explicitly"
                )
            root_session_id = sorted(root_candidates)[0] if root_candidates else "main"
        else:
            root_session_id = next(iter(root_candidates))

    parent_messages = {root_session_id: replay_messages_list}
    parent_messages.update({session["session_id"]: session["messages"] for session in sessions})
    calls_by_parent = {
        parent_session_id: extract_task_spawn_records(messages)
        for parent_session_id, messages in parent_messages.items()
    }

    for session in sessions:
        parent_id = session["parent_session_id"]
        calls = calls_by_parent.get(parent_id, [])
        spawn = next(
            (
                call
                for call in calls
                if session.get("spawn_call_id") and call["spawn_call_id"] == session["spawn_call_id"]
            ),
            None,
        )
        if spawn is None:
            direct = [call for call in calls if call.get("child_session_id") == session["session_id"]]
            spawn = next((call for call in direct if not call["arguments"].get("task_id")), None)
        if spawn is None:
            first_user = _first_user_message_text(session["messages"])
            prompt_matches = [
                call
                for call in calls
                if first_user
                and call["arguments"].get("prompt") == first_user
                and not call["arguments"].get("task_id")
            ]
            if len(prompt_matches) == 1:
                spawn = prompt_matches[0]

        if spawn is None:
            if strict:
                raise ValueError(
                    f"cannot link recorded subagent {session['session_id']} to a task call in parent {parent_id}"
                )
            continue

        session["spawn_call_id"] = spawn["spawn_call_id"]
        session["spawn_index"] = spawn["spawn_index"]
        session["spawn_message_index"] = spawn["spawn_message_index"]
        session["spawn_tool_index"] = spawn["spawn_tool_index"]
        subagent_type = spawn["arguments"].get("subagent_type")
        if isinstance(subagent_type, str):
            session["subagent_type"] = subagent_type

    linked_spawns: dict[tuple[str, str], str] = {}
    for session in sessions:
        spawn_call_id = session.get("spawn_call_id")
        if not isinstance(spawn_call_id, str):
            continue
        key = (session["parent_session_id"], spawn_call_id)
        other_session_id = linked_spawns.get(key)
        if other_session_id and other_session_id != session["session_id"]:
            if strict:
                raise ValueError(
                    f"task call {spawn_call_id} in parent {session['parent_session_id']} links to both "
                    f"{other_session_id} and {session['session_id']}"
                )
            continue
        linked_spawns[key] = session["session_id"]

    return {
        "version": 1,
        "root_session_id": root_session_id,
        "sessions": _order_sessions(root_session_id, sessions, strict=strict),
    }


def merge_replay_subagent_trajectories(manifest: Optional[dict], captured: list[dict]) -> list[dict]:
    """Carry recorded prefixes forward and append only each child's live continuation."""
    if manifest is None:
        return captured

    originals = [dict(session) for session in manifest.get("sessions", []) if isinstance(session, dict)]
    captured_by_recorded = {
        session["recorded_session_id"]: session
        for session in captured
        if isinstance(session.get("recorded_session_id"), str)
    }
    live_to_output_id: dict[str, str] = {}
    merged: list[dict] = []

    for original in originals:
        recorded_id = original.get("session_id")
        fresh = captured_by_recorded.get(recorded_id)
        if fresh is None:
            merged.append(original)
            continue

        prefix_count = fresh.get("replay_prefix_message_count")
        fresh_messages = fresh.get("messages") or []
        if not isinstance(prefix_count, int) or prefix_count < 0 or prefix_count > len(fresh_messages):
            prefix_count = min(len(original.get("messages") or []), len(fresh_messages))
        result = {
            **original,
            **{
                key: value
                for key, value in fresh.items()
                if key not in ("session_id", "parent_session_id", "messages")
            },
            "messages": list(original.get("messages") or []) + list(fresh_messages[prefix_count:]),
            "tools": fresh.get("tools") or original.get("tools") or [],
        }
        live_id = fresh.get("session_id")
        if isinstance(live_id, str) and live_id != recorded_id:
            result["live_session_id"] = live_id
            live_to_output_id[live_id] = recorded_id
        merged.append(result)

    matched_live_ids = {
        session.get("session_id")
        for session in captured_by_recorded.values()
        if isinstance(session.get("session_id"), str)
    }
    for fresh in captured:
        if fresh.get("session_id") in matched_live_ids:
            continue
        result = dict(fresh)
        parent_id = result.get("parent_session_id")
        recorded_parent_id = result.get("recorded_parent_session_id")
        if isinstance(recorded_parent_id, str):
            result["parent_session_id"] = recorded_parent_id
        elif parent_id in live_to_output_id:
            result["parent_session_id"] = live_to_output_id[parent_id]
        merged.append(result)
    root_session_id = manifest.get("root_session_id")
    if isinstance(root_session_id, str):
        return _order_sessions(root_session_id, merged, strict=False)
    return merged
