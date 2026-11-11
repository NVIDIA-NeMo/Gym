# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read an OpenCode 2 session store as the OpenCode 1 store the adapter's readers parse.

OpenCode 2 keeps sessions in ``session_v2`` and typed messages in ``session_message``: one ``assistant`` message per
model step, with its text, reasoning and tool calls inline in ``data.content`` (or in ``session_part`` rows when the
``experimental.part_rows`` option is on), and ``compaction``, ``synthetic``, ``idle`` and other control messages
beside them. ``translate_store`` writes the same conversation into the OpenCode 1.17.11 tables ``session``,
``message`` and ``part``, with the message and part shapes that ``snapshot``, ``artifacts`` and the response
conversion read, so every reader is shared between the two versions:

- ``user`` and ``synthetic`` messages become user messages with a text part (``synthetic: true`` for the latter);
  prompt images become file parts without their bytes.
- An ``assistant`` message becomes an assistant message with a ``step-start`` part, its content, and a
  ``step-finish`` part once the step ended. Its ID is the label the Gym plugin sent with the model call, when the
  plugin recorded one, so captured calls join to it.
- A ``compaction`` message becomes a user message with a ``compaction`` part and an assistant ``summary`` message
  holding the summary, as OpenCode 1 recorded a compaction.
- Tool parts keep OpenCode 2's tool names (``shell``, ``subagent``, ...) and record the text the model read as
  ``output`` (or ``error``); images a tool returned reach the model in a separate message and are not recorded.
- Errors take OpenCode 1's names: a provider failure is an ``APIError`` with its HTTP status, a provider rejection
  of an over-long prompt a ``ContextOverflowError``.
- ``system``, ``idle`` and the ``*-switched`` messages are left out.
"""

import json
import re
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any


# OpenCode 2 error types whose provider call may succeed when repeated; OpenCode 1 marked these ``isRetryable``.
_RETRYABLE_PROVIDER_ERRORS = frozenset(
    {"provider.rate-limit", "provider.internal", "provider.transport", "provider.timeout"}
)
# OpenCode 2 reports a prompt longer than the model's context as ``provider.invalid-request`` (its own overflow
# classification is not saved). These phrases are how OpenAI-compatible servers, vLLM included, say so.
_CONTEXT_OVERFLOW = re.compile(
    r"context_length_exceeded|maximum context length|context window|prompt is too long|too many input tokens",
    re.IGNORECASE,
)
_TOOL_STATUS = {"streaming": "pending", "running": "running", "completed": "completed", "error": "error"}


def _json(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(value) if isinstance(value, (str, bytes)) else value
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def translate_error(error: Any) -> dict[str, Any] | None:
    """OpenCode 1's ``{name, data}`` for an OpenCode 2 error ``{type, message, status?, response?}``."""
    if not isinstance(error, Mapping):
        return None
    kind = error.get("type")
    message = error.get("message")
    message = message if isinstance(message, str) else ""
    status = error.get("status")
    response = error.get("response") if isinstance(error.get("response"), Mapping) else {}
    body = response.get("body") if isinstance(response.get("body"), str) else ""
    data: dict[str, Any] = {"message": message, "type": kind}
    if isinstance(kind, str) and kind.startswith("provider."):
        if (
            kind == "provider.invalid-request"
            and status in (None, 400, 413)
            and _CONTEXT_OVERFLOW.search(f"{message}\n{body}")
        ):
            return {"name": "ContextOverflowError", "data": data}
        if type(status) is int:
            data["statusCode"] = status
        data["isRetryable"] = kind in _RETRYABLE_PROVIDER_ERRORS
        if body:
            data["responseBody"] = body[:4000]
        return {"name": "APIError", "data": data}
    if kind == "aborted":
        return {"name": "MessageAbortedError", "data": data}
    if kind == "output-limit":
        return {"name": "MessageOutputLengthError", "data": data}
    return {"name": "UnknownError", "data": data}


def tool_output_text(content: Any) -> str:
    """The tool message text OpenCode 2 sends for a completed tool's content, as its Chat Completions lowering does.

    One text item is sent as it is; otherwise the text items are joined by newlines and each file item (an image)
    follows in a user message of its own.
    """
    items = [item for item in content if isinstance(item, Mapping)] if isinstance(content, list) else []
    texts = [item["text"] for item in items if item.get("type") == "text" and isinstance(item.get("text"), str)]
    return "\n".join(texts)


def _tool_part(item: Mapping[str, Any]) -> dict[str, Any]:
    state = item.get("state") if isinstance(item.get("state"), Mapping) else {}
    status = _TOOL_STATUS.get(state.get("status"), state.get("status"))
    tool_input = state.get("input")
    if isinstance(tool_input, str):
        # A tool still streaming its arguments holds them as raw text.
        parsed = _json(tool_input)
        tool_input = parsed or ({"raw": tool_input} if tool_input else {})
    metadata = dict(state.get("metadata")) if isinstance(state.get("metadata"), Mapping) else {}
    if isinstance(metadata.get("sessionID"), str):
        # A subagent call names its child session; OpenCode 1's task tool used ``sessionId``.
        metadata["sessionId"] = metadata["sessionID"]
    timing = item.get("time") if isinstance(item.get("time"), Mapping) else {}
    translated: dict[str, Any] = {
        "status": status,
        "input": tool_input if tool_input is not None else {},
        "metadata": metadata,
        "time": {
            key: value
            for key, value in (("start", timing.get("ran", timing.get("created"))), ("end", timing.get("completed")))
            if value is not None
        },
    }
    if status == "completed":
        translated["output"] = tool_output_text(state.get("content"))
    elif status == "error":
        error = state.get("error") if isinstance(state.get("error"), Mapping) else {}
        # The tool message of a failed call is this JSON object.
        translated["error"] = _compact({"error": state.get("error"), "content": state.get("content") or []})
        if error.get("type") == "aborted":
            metadata["interrupted"] = True
    return {"type": "tool", "callID": item.get("id"), "tool": item.get("name"), "state": translated}


def _content_part(item: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = item.get("type")
    if kind in ("text", "reasoning"):
        return {"type": kind, "text": item.get("text") if isinstance(item.get("text"), str) else ""}
    if kind == "tool":
        return _tool_part(item)
    return None


def _usage(data: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if isinstance(data.get("tokens"), Mapping):
        fields["tokens"] = dict(data["tokens"])
    if isinstance(data.get("cost"), (int, float)):
        fields["cost"] = data["cost"]
    return fields


class _Writer:
    """Write OpenCode 1 rows with row times that keep each session's message order."""

    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con
        self.last_time: dict[str, int] = {}

    def message(self, session_id: str, message_id: str, created: Any, info: dict[str, Any], parts: list[dict]) -> None:
        created = created if type(created) is int else 0
        # OpenCode 1 readers order rows by (time_created, id); a session's messages keep their sequence.
        row_time = max(created, self.last_time.get(session_id, -1) + 1)
        self.last_time[session_id] = row_time
        info = {"id": message_id, "sessionID": session_id, **info}
        self.con.execute(
            "insert into message (id, session_id, data, time_created) values (?, ?, ?, ?)",
            (message_id, session_id, json.dumps(info), row_time),
        )
        self.con.executemany(
            "insert into part (id, message_id, session_id, data, time_created) values (?, ?, ?, ?, ?)",
            [
                (f"{message_id}-{index:04d}", message_id, session_id, json.dumps(part), row_time)
                for index, part in enumerate(parts)
            ],
        )


def message_labels(text: str) -> dict[tuple[str, str], str]:
    """The Gym plugin's label file as ``{(session ID, message ID): label}``; unreadable lines are skipped."""
    labels: dict[tuple[str, str], str] = {}
    for line in text.splitlines():
        entry = _json(line)
        if all(isinstance(entry.get(key), str) for key in ("label", "session", "message")):
            labels.setdefault((entry["session"], entry["message"]), entry["label"])
    return labels


def translate_store(source: Path, target: Path, *, labels: Mapping[tuple[str, str], str] | None = None) -> None:
    """Write ``target``, a new SQLite file with OpenCode 1 tables, from the OpenCode 2 store ``source``."""
    labels = labels or {}
    with closing(sqlite3.connect(f"file:{source}?mode=ro", uri=True)) as con:
        con.row_factory = sqlite3.Row
        sessions = con.execute(
            "select id, parent_id, time_created from session_v2 order by time_created, id"
        ).fetchall()
        messages = con.execute(
            "select id, session_id, type, seq, time_created, data from session_message order by session_id, seq"
        ).fetchall()
        part_rows: dict[str, list[dict[str, Any]]] = {}
        for row in con.execute("select message_id, data from session_part order by message_id, ordinal"):
            part_rows.setdefault(row["message_id"], []).append(_json(row["data"]))
    target.unlink(missing_ok=True)
    with closing(sqlite3.connect(target)) as out:
        out.executescript(
            "create table session (id text primary key, parent_id text, time_created integer);"
            "create table message (id text primary key, session_id text, data text, time_created integer);"
            "create table part (id text primary key, message_id text, session_id text, data text, time_created integer);"
        )
        out.executemany(
            "insert into session (id, parent_id, time_created) values (?, ?, ?)",
            [(row["id"], row["parent_id"], row["time_created"]) for row in sessions],
        )
        writer = _Writer(out)
        parent: dict[str, str] = {}
        for row in messages:
            session_id, message_id, kind = row["session_id"], row["id"], row["type"]
            data = _json(row["data"])
            created = (data.get("time") or {}).get("created", row["time_created"])
            if kind in ("user", "synthetic"):
                parts = [{"type": "text", "text": data.get("text") if isinstance(data.get("text"), str) else ""}]
                if kind == "synthetic":
                    parts[0]["synthetic"] = True
                for file in data.get("files") or []:
                    if isinstance(file, Mapping):
                        parts.append({"type": "file", "mime": file.get("mime"), "filename": file.get("name")})
                info = {"role": "user", "time": {"created": created}}
                writer.message(session_id, message_id, row["time_created"], info, parts)
                parent[session_id] = message_id
            elif kind == "assistant":
                model = data.get("model") if isinstance(data.get("model"), Mapping) else {}
                error = translate_error(data.get("error"))
                finish = data.get("finish")
                if error is not None and error["name"] == "APIError":
                    # OpenCode 1 saved a provider failure without a finish reason.
                    finish = None
                elif error is not None:
                    finish = "error"
                info = {
                    "role": "assistant",
                    "parentID": parent.get(session_id),
                    "agent": data.get("agent"),
                    "modelID": model.get("id"),
                    "providerID": model.get("providerID"),
                    "time": {key: value for key, value in (data.get("time") or {}).items() if key != "streamed"},
                    **_usage(data),
                    **({"finish": finish} if finish is not None else {}),
                    **({"error": error} if error is not None else {}),
                }
                content = data.get("content") or part_rows.get(message_id) or []
                parts = [{"type": "step-start"}]
                parts.extend(
                    part for item in content if isinstance(item, Mapping) and (part := _content_part(item)) is not None
                )
                if finish is not None and finish != "error":
                    parts.append({"type": "step-finish", "reason": finish, **_usage(data)})
                writer.message(
                    session_id, labels.get((session_id, message_id), message_id), row["time_created"], info, parts
                )
            elif kind == "compaction":
                request = f"{message_id}-request"
                info = {"role": "user", "time": {"created": created}}
                part = {"type": "compaction", "auto": data.get("reason") == "auto"}
                writer.message(session_id, request, row["time_created"], info, [part])
                completed = data.get("status") == "completed"
                model = data.get("model") if isinstance(data.get("model"), Mapping) else {}
                error = translate_error(data.get("error"))
                summary = {
                    "role": "assistant",
                    "summary": True,
                    "parentID": request,
                    "agent": "compaction",
                    "modelID": model.get("id"),
                    "providerID": model.get("providerID"),
                    "time": {
                        "created": created,
                        **(
                            {"completed": data["time"]["completed"]}
                            if completed and isinstance((data.get("time") or {}).get("completed"), int)
                            else {}
                        ),
                    },
                    **_usage(data),
                    **({"finish": "stop"} if completed else {}),
                    **({"error": error} if error is not None else {}),
                }
                text = data.get("summary") if isinstance(data.get("summary"), str) else ""
                writer.message(session_id, message_id, row["time_created"], summary, [{"type": "text", "text": text}])
        out.commit()


def reached_step_limit(export: Mapping[str, Any], steps: int | None) -> bool:
    """Whether the last prompt of a translated OpenCode 2 export reached the ``agent.build.steps`` limit.

    OpenCode 2 numbers each model step of a prompt from 1 and, from the step at the limit on, adds its maximum-steps
    note to the request and sets ``toolChoice: "none"`` (packages/core/src/session/runner/llm.ts). Each step saves
    one assistant message; a compaction runs inside a step and does not count, and the count restarts with the next
    prompt. Synthetic user messages (a continuation after an interrupted stream) and compaction records do not
    start a prompt.
    """
    if steps is None:
        return False
    messages = export.get("messages") or []
    prompt = max(
        (
            index
            for index, message in enumerate(messages)
            if message["info"].get("role") == "user"
            and not any(
                part.get("type") == "compaction" or part.get("synthetic") for part in message.get("parts") or []
            )
        ),
        default=-1,
    )
    replies = sum(
        1
        for message in messages[prompt + 1 :]
        if message["info"].get("role") == "assistant" and message["info"].get("summary") is not True
    )
    return replies >= steps
