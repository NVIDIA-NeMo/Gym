# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goose session store and stream-json output, converted to Responses items and stop reasons."""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from nemo_gym.core import (
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)


# Goose's own text when the model returned empty completions until it gave up (persisted, exit 0).
GOOSE_EMPTY_RESPONSE_NOTICE = "The model returned an empty response. Please resend your message to continue."
# Goose's MAX_TURNS_MESSAGE, streamed when --max-turns is reached.
GOOSE_MAX_TURNS_NOTICE = "I've reached the maximum number of actions I can do without user input"
# Goose's notices when compaction cannot bring the conversation under the context window.
GOOSE_CONTEXT_NOTICES = (
    "Unable to continue: Context limit still exceeded",
    "Ran into this error trying to compact: Failed to compact:",
)
# Goose 1.54.0's prefixes for a provider error, during a turn or a compaction.
GOOSE_ERROR_NOTICES = ("Ran into this error: ", "Ran into this error trying to compact: ")
# The HTTP status in Goose's provider error text (http_status.rs): "(500 ...)", "Status: 401 ..." or "status 408 ...".
GOOSE_HTTP_STATUS = re.compile(
    r"(?:Server error|Authentication error|Request failed): .*?(?:\(|Status: |status )([45]\d\d)\b"
)


def read_goose_sessions(db_path: Path) -> list[dict[str, Any]]:
    """Read every session (root and delegate children) with its messages, in creation order."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        sessions = [dict(row) for row in con.execute("select * from sessions order by created_at, id")]
        for session in sessions:
            session["messages"] = [
                {
                    "role": row["role"],
                    "content": json.loads(row["content_json"]),
                    "created": row["created_timestamp"],
                    "message_id": row["message_id"],
                    "user_visible": (json.loads(row["metadata_json"] or "{}") or {}).get("userVisible", True),
                }
                for row in con.execute(
                    "select message_id, role, content_json, created_timestamp, metadata_json from messages "
                    "where session_id = ? order by id",
                    (session["id"],),
                )
            ]
            session["usage"] = [
                dict(row)
                for row in con.execute(
                    "select input_tokens, output_tokens, cache_read_tokens from usage_ledger where session_id = ?",
                    (session["id"],),
                )
            ]
    finally:
        con.close()
    return sessions


def goose_messages_to_output(messages: list[dict[str, Any]]) -> list[Any]:
    """Convert the root session's Goose messages into Responses output items.

    Goose stores one assistant message per tool call of a turn and repeats the turn's thinking in each, so
    a thinking block equal to the previous one of the same turn is dropped. A user message that is not a
    tool response starts a new turn, so identical thinking there is kept. User text (the task and injected
    turn context) is input, not output. Messages Goose writes for itself (userVisible false, such as the
    compaction notice) are not policy output either.
    """
    output, last_thinking = [], None
    for message in messages:
        if not message.get("user_visible", True):
            continue
        if message["role"] == "user" and not any(b.get("type") == "toolResponse" for b in message["content"]):
            last_thinking = None
        for block in message["content"]:
            kind = block.get("type")
            if message["role"] == "assistant" and kind == "thinking" and block.get("thinking"):
                if block["thinking"] != last_thinking:
                    output.append(
                        NeMoGymResponseReasoningItem(
                            id=f"rs_{uuid4().hex}",
                            summary=[NeMoGymSummary(text=block["thinking"], type="summary_text")],
                        )
                    )
                last_thinking = block["thinking"]
            elif message["role"] == "assistant" and kind == "text" and block.get("text"):
                output.append(
                    NeMoGymResponseOutputMessage(
                        id=f"msg_{uuid4().hex}", content=[NeMoGymResponseOutputText(text=block["text"])]
                    )
                )
            elif kind == "toolRequest":
                call = block.get("toolCall") or {}
                value = call.get("value") or {}
                output.append(
                    NeMoGymResponseFunctionToolCall(
                        call_id=block["id"],
                        name=value.get("name", ""),
                        # A malformed call keeps Goose's error, which quotes the raw arguments.
                        arguments=str(call["error"])
                        if call.get("status") == "error"
                        else json.dumps(value.get("arguments") or {}),
                    )
                )
            elif kind == "toolResponse":
                result = block.get("toolResult") or {}
                content = (result.get("value") or {}).get("content") or []
                text = "\n".join(
                    part.get("text", "") if part.get("type") == "text" else f"[{part.get('type')} content omitted]"
                    for part in content
                )
                output.append(
                    NeMoGymFunctionCallOutput(call_id=block["id"], output=text or str(result.get("error", "")))
                )
    return output


def goose_stop_reason(stdout: str, sessions: list[dict[str, Any]]) -> tuple[str | None, str]:
    """Why Goose stopped early, or None, and the text of Goose's notice.

    When a model call fails past its retries, compaction cannot fit the context, or --max-turns is reached, Goose ends
    the run with a notice of its own as the last assistant message: text or a system notification that it does not
    persist and exits 0 on, or an error block that it persists and exits 1 on (rejected credentials). Every message
    from the model is persisted under the id it was streamed with, so an unpersisted last message is a notice too.
    """
    last = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # Goose prints a startup banner before the JSON events.
        if isinstance(event, dict) and event.get("type") == "message":
            if (event.get("message") or {}).get("role") == "assistant":
                last = event["message"]
    if not last:
        return None, ""
    if (last.get("metadata") or {}).get("outputTokenLimitReached"):
        # The output or context limit cut the last completion (the vLLM model server returns an empty one).
        return "length", ""
    blocks = [
        block for block in last.get("content") or [] if block.get("type") in ("text", "error", "systemNotification")
    ]
    text = "".join(block.get("text") or block.get("message") or block.get("msg") or "" for block in blocks)
    if text.strip() == GOOSE_EMPTY_RESPONSE_NOTICE:
        return "empty_response", text
    persisted = {m["message_id"] for s in sessions for m in s["messages"]}
    # Goose's own notice is never persisted, or is persisted as an error block (rejected credentials). Persisted text is
    # the policy's own answer, whatever its wording.
    if not (last.get("id") and last["id"] not in persisted) and not any(b.get("type") == "error" for b in blocks):
        return None, ""
    if text.startswith(GOOSE_CONTEXT_NOTICES):
        return "context_limit", text
    if model_server_failure(text):
        return "model_server", text
    return ("max_turns" if text.startswith(GOOSE_MAX_TURNS_NOTICE) else "model_error"), text


def model_server_failure(text: str) -> tuple[str, int | None] | None:
    """The failure kind and HTTP status of Goose's notice for a model call that failed at the model server."""
    error = text.removeprefix(next((h for h in GOOSE_ERROR_NOTICES if text.startswith(h)), ""))
    if error.startswith("Network error:"):
        return MODEL_TRANSPORT_ERROR, None
    if error.startswith("Rate limit exceeded:"):
        return MODEL_REQUEST_ERROR, 429
    match = GOOSE_HTTP_STATUS.match(error)
    return (MODEL_REQUEST_ERROR, int(match[1])) if match else None


def max_concurrent(intervals: list[tuple[int, int]]) -> int:
    """Largest number of closed intervals sharing a point."""
    events = sorted(
        [(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals], key=lambda e: (e[0], -e[1])
    )
    best = current = 0
    for _, step in events:
        current += step
        best = max(best, current)
    return best
