# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical evidence from OpenCode exports, never from CLI summaries."""

import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import cast

from pydantic import JsonValue

from resources_servers.agentic_if.episode.opencode_context import compaction_context
from resources_servers.agentic_if.episode.opencode_profile import model_visible_tool_input, model_visible_tool_output
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence, SessionInfo, SessionStep, ToolCall


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_ARGUMENTS = {
    "read": {"filePath": "path"},
    "write": {"filePath": "path"},
    "edit": {"filePath": "path", "oldString": "old_string", "newString": "content", "replaceAll": "replace_all"},
}


def visible_assistant_text(text: str) -> str:
    """Exclude private reasoning that a model embeds in its public text output.

    Structured reasoning parts are excluded separately. Apply this only to assistant text: the same tags in user
    instructions are literal data. An unfinished reasoning block is not visible assistant prose either.
    """
    text = _THINK.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.split("<think>", 1)[0]


def refused(tool: str, status: object, error: object) -> bool:
    """The harness refused a call before running it, because its arguments do not fit the tool's schema. Read from
    OpenCode's own error text: the profile plugin may show the model other words. The evidence marks such a call
    `rejected`."""
    return status == "error" and str(error).startswith("The %s tool was called with invalid arguments" % tool)


def _object(value: object) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], value) if isinstance(value, dict) else {}


def _time(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _visible_text(parts: Sequence[dict[str, JsonValue]], *, assistant: bool) -> str:
    text = "".join(
        part["text"]
        for part in parts
        if part.get("type") == "text"
        and isinstance(part.get("text"), str)
        and not part.get("synthetic")
        and not part.get("ignored")
    )
    if not assistant:
        return text
    return visible_assistant_text(text)


def _arguments(
    tool: str, state: Mapping[str, JsonValue], model_args: JsonValue, interface: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    """Keep actor arguments, adding canonical execution facts used by checkers."""
    raw = _object(state.get("input"))
    builtin = {key: value for key, value in raw.items() if not key.startswith("__nemo_gym_")}
    stash = _object(raw.get("__nemo_gym_profile"))
    mapping = _object(interface.get("args_map") or stash.get("names"))
    for name, path in mapping.items():
        node = model_args
        if not isinstance(path, str):
            continue
        for segment in path.split("."):
            if not isinstance(node, dict) or segment not in node:
                break
            node = node[segment]
        else:
            builtin[name] = node
    for name, value in _object(interface.get("fixed")).items():
        builtin.setdefault(name, value)
    result = deepcopy(model_args) if isinstance(model_args, dict) else {"raw": deepcopy(model_args)}
    for name, value in builtin.items():
        result[_ARGUMENTS.get(tool, {}).get(name, name)] = deepcopy(value)
    return result


def opencode_evidence(
    export: Mapping[str, JsonValue] | None,
    *,
    child_exports: Sequence[Mapping[str, JsonValue]] = (),
    expected_sessions: Sequence[SessionInfo] = (),
    interfaces: Mapping[str, Mapping[str, JsonValue]] | None = None,
    artifacts: Mapping[str, str] | None = None,
    controlled_messages: frozenset[tuple[str, str]] = frozenset(),
) -> SessionEvidence:
    """Capture grouped messages, correlated tool results and OpenCode's child sessions.

    ``interfaces`` contains trusted, already-selected installed interface definitions.
    ``expected_sessions`` comes from the agent's session inventory so an absent
    delegated export cannot masquerade as an empty child. A cut prefix may be
    complete evidence without being a completed task.
    """
    issues: list[str] = []
    sessions: list[SessionInfo] = []
    timed_steps: list[tuple[float | None, SessionStep]] = []
    messages_seen: set[tuple[str, str]] = set()
    calls_seen: set[tuple[str, str]] = set()
    compactions = []
    multi_session = bool(child_exports)

    def add(step: SessionStep, timestamp: float | None) -> None:
        if multi_session and timestamp is None:
            issues.append(f"missing_cross_session_time:{step.session_id}:{step.id}")
        timed_steps.append((timestamp, step))

    if export is None:
        issues.append("missing_export")
    for document in ([export] if export is not None else []) + list(child_exports):
        info = _object(document.get("info"))
        session_id = info.get("id")
        if not isinstance(session_id, str) or not session_id:
            issues.append("missing_session_id")
            continue
        if any(session.id == session_id for session in sessions):
            issues.append(f"duplicate_session:{session_id}")
            continue
        parent = info.get("parentID")
        if parent is not None and (not isinstance(parent, str) or not parent):
            issues.append(f"invalid_parent:{session_id}")
            parent = None
        sessions.append(SessionInfo(id=session_id, parent_id=parent))
        messages = document.get("messages")
        if not isinstance(messages, list) or not messages:
            issues.append(f"missing_messages:{session_id}")
            continue
        internal, context_records, context_issues = compaction_context(
            document,
            controlled_ids=frozenset(identity for session, identity in controlled_messages if session == session_id),
        )
        compactions.extend(context_records)
        issues.extend(context_issues)
        for message in messages:
            message = _object(message)
            info = _object(message.get("info"))
            identity, role = info.get("id"), info.get("role")
            if not isinstance(identity, str) or not identity:
                issues.append(f"missing_message_id:{session_id}")
                continue
            if (session_id, identity) in messages_seen:
                issues.append(f"duplicate_message:{session_id}:{identity}")
                continue
            messages_seen.add((session_id, identity))
            if info.get("sessionID", session_id) != session_id:
                issues.append(f"wrong_message_session:{session_id}:{identity}")
                continue
            if role not in {"user", "assistant"}:
                issues.append(f"unsupported_message_role:{session_id}:{identity}")
                continue
            if identity in internal or info.get("summary") is True:
                continue
            raw_parts = message.get("parts")
            if not isinstance(raw_parts, list):
                issues.append(f"missing_message_parts:{session_id}:{identity}")
                continue
            parts = [_object(part) for part in raw_parts]
            step = SessionStep(
                id=identity,
                session_id=session_id,
                role=role,
                text=_visible_text(parts, assistant=role == "assistant"),
            )
            timestamp = _time(_object(info.get("time")).get("created"))
            add(step, timestamp)
            for part in parts:
                if part.get("type") != "tool" or role != "assistant":
                    continue
                call_id, tool = part.get("callID"), part.get("tool")
                if not isinstance(call_id, str) or not call_id:
                    issues.append(f"missing_call_id:{session_id}:{identity}")
                    continue
                if not isinstance(tool, str) or not tool:
                    issues.append(f"missing_tool_name:{session_id}:{identity}:{call_id}")
                    continue
                if (session_id, call_id) in calls_seen:
                    issues.append(f"duplicate_call:{session_id}:{call_id}")
                    continue
                calls_seen.add((session_id, call_id))
                state = _object(part.get("state"))
                model_args = deepcopy(model_visible_tool_input(state))
                output = model_visible_tool_output(state)
                step.calls.append(
                    ToolCall(
                        call_id=call_id,
                        invocation_id=session_id,
                        tool=tool,
                        model_tool=tool,
                        model_args=model_args,
                        args=_arguments(tool, state, model_args, (interfaces or {}).get(tool, {})),
                        rejected=tool == "invalid" or refused(tool, state.get("status"), state.get("error")),
                    )
                )
                if state.get("status") not in {"completed", "error"}:
                    issues.append(f"unfinished_tool:{session_id}:{identity}:{call_id}")
                    continue
                if output is None:
                    issues.append(f"missing_tool_result:{session_id}:{identity}:{call_id}")
                    continue
                metadata = _object(state.get("metadata"))
                diff = _object(metadata.get("nemo_gym_profile")).get("workspace_diff")
                add(
                    SessionStep(
                        id=f"{identity}:tool:{call_id}",
                        session_id=session_id,
                        role="tool",
                        call_id=call_id,
                        invocation_id=session_id,
                        tool=tool,
                        output=output if isinstance(output, str) else json.dumps(output, ensure_ascii=False),
                        error=state.get("status") == "error",
                        workspace_diff=diff if isinstance(diff, str) else None,
                    ),
                    _time(_object(state.get("time")).get("end")),
                )

    observed = {session.id: session.parent_id for session in sessions}
    for session in sessions:
        if session.parent_id is not None and session.parent_id not in observed:
            issues.append(f"missing_parent:{session.id}:{session.parent_id}")
    for expected in expected_sessions:
        if expected.id not in observed:
            issues.append(f"missing_session:{expected.id}")
        elif observed[expected.id] != expected.parent_id:
            issues.append(f"wrong_session_parent:{expected.id}")
    # A single OpenCode export already has authoritative message/part order. Across
    # sessions, OpenCode's timestamps order observations; missing times are explicit gaps.
    if multi_session and all(timestamp is not None for timestamp, _ in timed_steps):
        timed_steps.sort(key=lambda item: item[0])
    resolved_facts: dict[str, JsonValue] = {}
    if compactions:
        resolved_facts["opencode_compactions"] = compactions
    directory = _object(export.get("info")).get("directory") if export is not None else None
    if isinstance(directory, str) and directory:
        resolved_facts["workdir"] = directory
    return SessionEvidence(
        status="incomplete" if issues else "complete",
        steps=[step for _, step in timed_steps],
        sessions=sessions,
        issues=list(dict.fromkeys(issues)),
        facts=resolved_facts,
        artifacts=dict(artifacts or {}),
    )
