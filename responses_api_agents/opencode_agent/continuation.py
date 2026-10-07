# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project the native event stream without applying benchmark visibility policy."""

import json
import re

from nemo_gym.interactive_agent_types import AgentActivationEvent


def parse_activation_events(stdout: str) -> list[AgentActivationEvent]:
    """Keep native event ordering, tool evidence, and separate private reasoning."""
    events = []
    kinds = {"step_start", "step_finish", "text", "tool_use", "reasoning", "compaction", "error"}
    for line in stdout.splitlines():
        try:
            native = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(native, dict) or native.get("type") not in kinds:
            continue
        part = native.get("part") or {}
        if not isinstance(part, dict):
            continue
        state = part.get("state") or {}
        if not isinstance(state, dict):
            state = {}
        metadata = {key: value for key, value in native.items() if key not in {"type", "part"}}
        metadata["part"] = part
        if native.get("type") == "error":
            metadata["error"] = native.get("error")
        events.append(
            AgentActivationEvent(
                sequence=len(events),
                kind=native["type"],
                text=part.get("text") if isinstance(part.get("text"), str) else None,
                name=part.get("tool") if isinstance(part.get("tool"), str) else None,
                tool_call_id=part.get("callID") if isinstance(part.get("callID"), str) else None,
                arguments=state.get("input"),
                result=state.get("output", state.get("error")),
                metadata=metadata,
            )
        )
    return events


def visible_activation_log(stdout: str) -> str:
    """Keep unparsed fallback diagnostics without leaking dedicated reasoning events.

    Private reasoning remains available in typed activation events and native
    observation artifacts; the simulator-visible fallback must not reinterpret it
    as narration just because no text/tool event was emitted.
    """
    lines = []
    for line in stdout.splitlines():
        try:
            native = json.loads(line)
        except (ValueError, TypeError):
            native = None
        if isinstance(native, dict) and native.get("type") == "reasoning":
            continue
        lines.append(line)
    return "\n".join(lines)


def native_failure_detail(*, message, classification, session_id, finish_reason, cleanup, stdout):
    """Keep bounded native evidence in an HTTP error without transport metadata."""
    sensitive = re.compile(r"authorization|api[_-]?key|(?:request|response)?headers", re.IGNORECASE)

    def safe(value, depth=0):
        if depth > 6:
            return "[truncated]"
        if isinstance(value, dict):
            return {key: safe(item, depth + 1) for key, item in list(value.items())[:16] if not sensitive.search(key)}
        if isinstance(value, list):
            return [safe(item, depth + 1) for item in value[:8]]
        if isinstance(value, str):
            return "\n".join(
                "[redacted transport metadata]" if sensitive.search(line) else line for line in value.splitlines()
            )[:1024]
        return value

    try:
        parsed_message = json.loads(message)
    except (ValueError, TypeError):
        parsed_message = message
    message = safe(parsed_message)
    if not isinstance(message, str):
        message = json.dumps(message)[:1024]
    events = []
    for line in visible_activation_log(stdout).splitlines()[-3:]:
        try:
            native = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(native, dict):
            events.append(
                {"type": str(native.get("type", "unknown"))[:64], "preview": json.dumps(safe(native))[:1536]}
            )
    return {
        "harness": "opencode",
        "classification": classification,
        "message": message,
        "native_session_id": session_id[:256] if session_id else None,
        "finish_reason": str(finish_reason)[:128] if finish_reason is not None else None,
        "return_code": cleanup.get("return_code") if cleanup else None,
        "timed_out": cleanup.get("timed_out") if cleanup else None,
        "native_event_tail": events,
    }
