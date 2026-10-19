# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project the native event stream without applying benchmark visibility policy."""

import json
from dataclasses import dataclass
from typing import Any, Literal

from nemo_gym.agent_activation_types import AgentActivationEvent
from nemo_gym.agent_utils.process_supervisor import CleanupReceipt


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


@dataclass(frozen=True, kw_only=True)
class NativeOutcome:
    """One stopped OpenCode activation classified from its native checkpoint."""

    status: Literal["completed", "incomplete", "failed"]
    error_type: Literal["server_error", "timeout", "permission_denied"] | None = None
    error: str | None = None


def classify_native_outcome(
    export: dict[str, Any],
    *,
    cleanup: CleanupReceipt | None,
    previous_session_id: str | None,
    session_id: str | None,
    stdout: str,
    continuation: bool,
) -> NativeOutcome:
    """Require a native terminal result or a confirmed interrupted checkpoint."""
    error = cleanup["error"] if cleanup is not None else None
    timed_out = bool(cleanup is not None and cleanup["timed_out"])
    if cleanup is not None and cleanup["return_code"] not in (0, None) and not timed_out:
        error = error or f"OpenCode exited with code {cleanup['return_code']}"
    assistants = [message for message in export.get("messages", []) if message["info"].get("role") == "assistant"]
    for message in assistants:
        if message["info"].get("error"):
            error = error or json.dumps(message["info"]["error"])
    resumed_checkpoint = bool(
        previous_session_id
        and session_id == previous_session_id
        and timed_out
        and cleanup is not None
        and cleanup["cleanup_confirmed"]
        and not any(event.kind == "error" for event in parse_activation_events(stdout))
    )
    if not assistants and not resumed_checkpoint:
        error = error or "OpenCode produced no assistant result"
    permission_denied = False
    incomplete = timed_out
    if assistants and not incomplete:
        last = assistants[-1]
        info = last["info"]
        finish = info.get("finish")
        if finish in {"length", "content-filter"} or (finish == "tool-calls" and not continuation):
            incomplete = True
        elif finish != "stop" or not info.get("time", {}).get("completed"):
            # Pinned OpenCode records a permission rejection as a tool error string.
            # Other tool errors must not be mistaken for a resumable permission yield.
            permission_denied = bool(
                error is None
                and finish == "tool-calls"
                and info.get("time", {}).get("completed")
                and cleanup is not None
                and cleanup["return_code"] == 0
                and cleanup["cleanup_confirmed"]
                and session_id
                and any(
                    part.get("type") == "tool"
                    and part.get("state", {}).get("status") == "error"
                    and part.get("state", {}).get("error")
                    == "The user rejected permission to use this specific tool call."
                    for part in last.get("parts", [])
                )
            )
            if permission_denied:
                incomplete = True
            else:
                error = error or f"OpenCode ended without a successful terminal assistant result (finish={finish!r})"
    if error:
        return NativeOutcome(status="failed", error_type="server_error", error=error)
    if timed_out:
        return NativeOutcome(status="incomplete", error_type="timeout")
    if permission_denied:
        return NativeOutcome(status="incomplete", error_type="permission_denied")
    if incomplete:
        return NativeOutcome(status="incomplete")
    return NativeOutcome(status="completed")


def native_input_acceptance(export: dict[str, Any], *, prompt: str) -> Literal["true", "unknown"]:
    """Confirm acceptance only from this activation's new root user message.

    Snapshot export excludes earlier messages and child conversations. Absence
    remains unknown: a launched process or staged prompt does not prove delivery.
    Model-request observation belongs to captured provider calls, not this export.
    """
    for message in export.get("messages", []):
        if message["info"].get("role") != "user":
            continue
        text = "".join(
            part.get("text", "")
            for part in message.get("parts", [])
            if part.get("type") == "text" and isinstance(part.get("text"), str)
        )
        if text == prompt:
            return "true"
    return "unknown"
