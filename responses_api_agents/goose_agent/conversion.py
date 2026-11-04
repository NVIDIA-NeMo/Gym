# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Goose session store and stream-json output, converted to Responses items and stop reasons."""

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from nemo_gym.core import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)


# Goose's own text when the model returned empty completions until it gave up (persisted, exit 0).
GOOSE_EMPTY_RESPONSE_NOTICE = "The model returned an empty response. Please resend your message to continue."
# summon's wrapper around an async delegate's result in load's tool response.
GOOSE_TASK_RESULT_HEADER = "# Background Task Result"
GOOSE_TASK_OUTPUT = "\n## Output\n\n"
# Goose's MAX_TURNS_MESSAGE, streamed when --max-turns is reached.
GOOSE_MAX_TURNS_NOTICE = "I've reached the maximum number of actions I can do without user input"
# Goose 1.54.0's provider errors for a model call that failed at the model server: unreachable, 5xx, rate limit, rejected
# credentials or a missing model, during a turn or a compaction. The model server or the configuration failed, not the
# policy, so the request fails and is retried.
GOOSE_ERROR_NOTICES = ("Ran into this error: ", "Ran into this error trying to compact: ")
GOOSE_MODEL_SERVER_ERRORS = (
    "Network error",
    "Server error",
    "Rate limit exceeded",
    "Authentication error",
    "Request failed: Resource not found",
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
                        arguments=json.dumps(value.get("arguments") or {}),
                    )
                )
            elif kind == "toolResponse":
                result = block.get("toolResult") or {}
                content = (result.get("value") or {}).get("content") or []
                text = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
                output.append(
                    NeMoGymFunctionCallOutput(call_id=block["id"], output=text or str(result.get("error", "")))
                )
    return output


def goose_stop_reason(stdout: str, sessions: list[dict[str, Any]]) -> str | None:
    """Why Goose stopped early, or None.

    When a model call fails past its retries or --max-turns is reached, Goose ends the run with a notice of its own as
    the last assistant message: plain text that it does not persist and exits 0 on, or an error block that it persists
    and exits 1 on (rejected credentials). Every message from the model is persisted under the id it was streamed with,
    so an unpersisted last message is a notice too. "model_server" marks a notice for a model server that failed, not
    the policy.
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
        return None
    if (last.get("metadata") or {}).get("outputTokenLimitReached"):
        # The output or context limit cut the last completion (the vLLM model server returns an empty one).
        return "length"
    blocks = [block for block in last.get("content") or [] if block.get("type") in ("text", "error")]
    if "".join(block.get("text") or "" for block in blocks).strip() == GOOSE_EMPTY_RESPONSE_NOTICE:
        return "empty_response"
    persisted = {m["message_id"] for s in sessions for m in s["messages"]}
    # Goose's own notice is never persisted, or is persisted as an error block (rejected credentials). Persisted text is
    # the policy's own answer, whatever its wording.
    if not (last.get("id") and last["id"] not in persisted) and not any(b.get("type") == "error" for b in blocks):
        return None
    text = "".join(block.get("text") or block.get("message") or "" for block in blocks)
    if _model_server_error(text):
        return "model_server"
    return "max_turns" if text.startswith(GOOSE_MAX_TURNS_NOTICE) else "model_error"


def _model_server_error(text: str) -> bool:
    if text.startswith("Network error:"):
        return True
    head = next((h for h in GOOSE_ERROR_NOTICES if text.startswith(h)), None)
    return head is not None and text[len(head) :].startswith(GOOSE_MODEL_SERVER_ERRORS)


def delegate_model_server_failure(sessions: list[dict[str, Any]]) -> bool:
    """Whether a delegate's model calls failed at the model server.

    A delegate that hits a provider error ends with Goose's notice as its last message, which it does not persist, and
    summon returns that text to the parent: as the delegate tool's result, or for an async delegate inside load's
    "# Background Task Result" under "## Output". The parent carries on and Goose exits 0. Text a delegate persisted is
    its own answer, whatever its wording.
    """
    persisted = {
        "\n".join(block.get("text", "") for block in message["content"] if block.get("type") == "text")
        for session in sessions
        if session.get("parent_session_id")
        for message in session["messages"]
        if message["role"] == "assistant"
    }
    for session in sessions:
        calls = {
            block["id"]: name
            for message in session["messages"]
            for block in message["content"]
            if block.get("type") == "toolRequest"
            and (name := ((block.get("toolCall") or {}).get("value") or {}).get("name")) in ("delegate", "load")
        }
        for message in session["messages"]:
            for block in message["content"]:
                if block.get("type") != "toolResponse" or block.get("id") not in calls:
                    continue
                content = ((block.get("toolResult") or {}).get("value") or {}).get("content") or []
                text = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
                if calls[block["id"]] == "load":
                    if not text.startswith(GOOSE_TASK_RESULT_HEADER) or GOOSE_TASK_OUTPUT not in text:
                        continue
                    text = text.split(GOOSE_TASK_OUTPUT, 1)[1]
                if text not in persisted and _model_server_error(text):
                    return True
    return False


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
