# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classify stopped OpenCode runs from their saved checkpoints and events."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from nemo_gym.agents import CleanupReceipt, RunEvidence


ERROR_EVENT_CHARS = 4000


@dataclass(frozen=True, kw_only=True)
class OpenCodeRunOutcome:
    """One stopped OpenCode run classified from its saved checkpoint and events."""

    evidence: RunEvidence
    """The run's evidence for the outcome ladder. The adapter adds the classified error of a provider failure that
    OpenCode reported, and the fault behind a deadline."""
    error_type: Literal["server_error", "timeout", "permission_denied"] | None = None
    """How the run ended, as its observations name it."""
    error: str | None = None
    """Why the run left no usable result."""
    incomplete_reason: Literal["max_output_tokens", "content_filter"] | None = None
    recorded_error: Mapping[str, Any] | None = None
    """The model error OpenCode recorded last, in the shape ``classify_outcome`` reads (``{"error": ...}``): the last
    assistant message's error or, for a timed-out continuation, its error event."""


def classify_opencode_run(
    export: dict[str, Any],
    *,
    cleanup: CleanupReceipt | None,
    previous_session_id: str | None,
    session_id: str | None,
    stdout: str,
    continuation: bool,
) -> OpenCodeRunOutcome:
    """Require an OpenCode terminal result or a confirmed interrupted checkpoint.

    A deadline before any assistant message, unless a continuation saved a resumable checkpoint, keeps the retry it
    had as a 502 (``retry_deadline_before_reply``). ``model_replied`` is whether a saved assistant turn finished
    without an error, so a model error on the first request counts as one before any reply.
    """
    error = cleanup["error"] if cleanup is not None else None
    timed_out = bool(cleanup is not None and cleanup["timed_out"])
    if cleanup is not None and cleanup["return_code"] not in (0, None) and not timed_out:
        error = error or f"OpenCode exited with code {cleanup['return_code']}"
    assistants = [message for message in export.get("messages", []) if message["info"].get("role") == "assistant"]
    recorded_error = None
    for message in assistants:
        if message["info"].get("error"):
            error = error or json.dumps(message["info"]["error"])
            recorded_error = {"error": message["info"]["error"]}
    if continuation and timed_out and (event := _first_error_event(stdout)) is not None:
        # The event names the failing boundary, for example an APIError with the provider's status.
        error = (
            error or f"OpenCode emitted an error event: {json.dumps(event.get('error') or event)[:ERROR_EVENT_CHARS]}"
        )
        recorded_error = {"error": event["error"]} if isinstance(event.get("error"), dict) else recorded_error
    resumable_checkpoint = bool(
        continuation
        and session_id
        and (previous_session_id is None or session_id == previous_session_id)
        and timed_out
        and cleanup is not None
        and cleanup["cleanup_confirmed"]
        and error is None
    )
    no_reply = not assistants and not resumable_checkpoint
    if no_reply:
        error = error or "OpenCode produced no assistant result"
    permission_denied = False
    limit: Literal["output", "turns"] | None = None
    incomplete_reason: Literal["max_output_tokens", "content_filter"] | None = None
    if assistants and not timed_out:
        last = assistants[-1]
        info = last["info"]
        finish = info.get("finish")
        if finish == "length":
            limit, incomplete_reason = "output", "max_output_tokens"
        elif finish == "content-filter":
            # The provider cut the answer short, as at the output limit.
            limit, incomplete_reason = "output", "content_filter"
        elif finish == "tool-calls" and not continuation:
            # OpenCode stopped at its step limit with tool calls still pending.
            limit = "turns"
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
                # A continuation yields to ask for the permission, as at a step limit.
                limit = "turns"
            else:
                error = error or f"OpenCode ended without a successful terminal assistant result (finish={finish!r})"
    if error:
        limit, incomplete_reason, error_type = None, None, "server_error"
    elif timed_out:
        error_type = "timeout"
    else:
        error_type = "permission_denied" if permission_denied else None
    evidence = RunEvidence(
        has_result=error is None and not timed_out and limit is None,
        timed_out=timed_out,
        model_replied=any(
            not message["info"].get("error") and message["info"].get("finish") for message in assistants
        ),
        model_limit=limit,
        retry_deadline_before_reply=timed_out and no_reply,
    )
    return OpenCodeRunOutcome(
        evidence=evidence,
        error_type=error_type,
        error=error,
        incomplete_reason=incomplete_reason,
        recorded_error=recorded_error,
    )


def _first_error_event(stdout: str) -> dict[str, Any] | None:
    """The first explicit OpenCode error event, which an interrupted checkpoint must not hide."""
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "error":
            return event
    return None
