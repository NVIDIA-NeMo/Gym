# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classify stopped OpenCode runs from their saved checkpoints and events."""

import json
from dataclasses import dataclass
from typing import Any, Literal

from nemo_gym.agent_utils.supervisor_client import CleanupReceipt


ERROR_EVENT_CHARS = 4000


@dataclass(frozen=True, kw_only=True)
class OpenCodeRunOutcome:
    """One stopped OpenCode run classified from its saved checkpoint and events."""

    status: Literal["completed", "incomplete", "failed"]
    error_type: Literal["server_error", "timeout", "permission_denied"] | None = None
    error: str | None = None
    incomplete_reason: Literal["max_output_tokens", "content_filter"] | None = None


def classify_opencode_run(
    export: dict[str, Any],
    *,
    cleanup: CleanupReceipt | None,
    previous_session_id: str | None,
    session_id: str | None,
    stdout: str,
    continuation: bool,
) -> OpenCodeRunOutcome:
    """Require an OpenCode terminal result or a confirmed interrupted checkpoint."""
    error = cleanup["error"] if cleanup is not None else None
    timed_out = bool(cleanup is not None and cleanup["timed_out"])
    if cleanup is not None and cleanup["return_code"] not in (0, None) and not timed_out:
        error = error or f"OpenCode exited with code {cleanup['return_code']}"
    assistants = [message for message in export.get("messages", []) if message["info"].get("role") == "assistant"]
    for message in assistants:
        if message["info"].get("error"):
            error = error or json.dumps(message["info"]["error"])
    if continuation and timed_out and (event := opencode_error_event(stdout)) is not None:
        error = error or f"OpenCode emitted an error event: {event}"
    resumable_checkpoint = bool(
        continuation
        and session_id
        and (previous_session_id is None or session_id == previous_session_id)
        and timed_out
        and cleanup is not None
        and cleanup["cleanup_confirmed"]
        and error is None
    )
    if not assistants and not resumable_checkpoint:
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
        return OpenCodeRunOutcome(status="failed", error_type="server_error", error=error)
    if timed_out:
        return OpenCodeRunOutcome(status="incomplete", error_type="timeout")
    if permission_denied:
        return OpenCodeRunOutcome(status="incomplete", error_type="permission_denied")
    if incomplete:
        return OpenCodeRunOutcome(
            status="incomplete",
            incomplete_reason="max_output_tokens"
            if finish == "length"
            else "content_filter"
            if finish == "content-filter"
            else None,
        )
    return OpenCodeRunOutcome(status="completed")


def opencode_error_event(stdout: str) -> str | None:
    """Return the first explicit OpenCode error event, which an interrupted checkpoint must not hide."""
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "error":
            # The event names the failing boundary, for example an APIError with the provider's status.
            return json.dumps(event.get("error") or event)[:ERROR_EVENT_CHARS]
    return None
