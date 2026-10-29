# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decide how a relaunched OpenCode harness continues its saved conversation, after a stop or a crash.

A launch a checkpoint stopped, or one a crash interrupted, left its conversation in OpenCode's database under the
session's persistent directory. The next launch resumes it when the conversation is the task's own and the model
answered at least once: ``resume_opencode.sh`` re-posts the last user message with empty ``parts``, so the session
loop runs again from it without a new user turn. Otherwise the saved conversation is unusable and the harness
starts the task afresh, from its original prompt in a fresh data directory.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ResumeDecision:
    """``resume`` with the saved ``session_id`` and the request body to re-post, or a fresh start and its reason."""

    resume: bool
    reason: str
    session_id: Optional[str] = None
    payload: Optional[dict[str, Any]] = None


def resume_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-enter the last user message, keeping its model and agent settings, with no new parts."""
    users = [message["info"] for message in messages if message["info"].get("role") == "user"]
    if not users:
        raise ValueError("cannot resume an OpenCode session without a saved user turn")
    # Message IDs order the conversation; a resume updates the user message's timestamps.
    user = max(users, key=lambda info: info["id"])
    payload: dict[str, Any] = {"messageID": user["id"], "parts": []}
    for name in ("agent", "system", "format", "tools"):
        if user.get(name) is not None:
            payload[name] = user[name]
    model = user.get("model")
    if model:
        payload["model"] = {name: model[name] for name in ("providerID", "modelID")}
        if model.get("variant") is not None:
            payload["variant"] = model["variant"]
    return payload


def _text(message: dict[str, Any]) -> str:
    return "".join(part.get("text", "") for part in message.get("parts", []) if part.get("type") == "text")


def resume_decision(export: Optional[dict[str, Any]], prompt: str) -> ResumeDecision:
    """Resume when the saved root session starts with the task's prompt and holds a completed assistant message.

    ``export`` is the conversation ``snapshot.snapshot`` exported from the sandbox's database, ``None`` when there is
    none. A conversation of another task, one the model never answered, or one that cannot be read starts afresh.
    """
    if export is None:
        return ResumeDecision(resume=False, reason="no saved OpenCode conversation")
    messages = export.get("messages") or []
    users = [message for message in messages if message["info"].get("role") == "user"]
    if not users:
        return ResumeDecision(resume=False, reason="the saved conversation has no user turn")
    if _text(users[0]).strip() != prompt.strip():
        return ResumeDecision(resume=False, reason="the saved conversation does not start with this task's prompt")
    answered = [
        message
        for message in messages
        if message["info"].get("role") == "assistant" and (message["info"].get("time") or {}).get("completed")
    ]
    if not answered:
        return ResumeDecision(resume=False, reason="the model had not answered the saved conversation yet")
    try:
        payload = resume_payload(messages)
    except (KeyError, TypeError, ValueError) as error:
        return ResumeDecision(resume=False, reason=f"the saved conversation cannot be resumed: {error}")
    return ResumeDecision(
        resume=True, reason="resuming the saved conversation", session_id=export["session_id"], payload=payload
    )
