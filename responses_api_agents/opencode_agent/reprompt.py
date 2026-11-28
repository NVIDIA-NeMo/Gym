# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decide how a relaunched OpenCode harness continues its saved conversation, after a stop or a crash.

A launch a checkpoint stopped, or one a crash interrupted, left its conversation in OpenCode's database under the
session's persistent directory. The next launch reprompts it when the conversation is the task's own and the model
answered at least once: ``reprompt_opencode.sh`` re-posts the last user message with empty ``parts``, so the session
loop runs again from it without a new user turn.

Otherwise the harness starts the task afresh, from its original prompt in a fresh data directory, and the saved
conversation is set aside. That is the case when there is no saved conversation, when it is another task's, when the
model never answered it, and when it was downloaded but cannot be parsed. A conversation that cannot be downloaded
from the sandbox is different: the read is retried and then fails the launch, so that a conversation that may be
intact is not set aside (``OpenCodeSandboxSession.inspect_conversation``).

A tool call that was running when the launch ended is unfinished in the saved conversation. OpenCode gives the model
``[Tool execution was interrupted]`` as its result, while whatever the command had already done stays in the sandbox.
Such a rollout is kept; ``RepromptDecision.interrupted_tool_calls`` counts those calls so the adapter can mark it.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RepromptDecision:
    """``reprompt`` with the saved ``session_id`` and the request body to re-post, or a fresh start and its reason."""

    reprompt: bool
    reason: str
    session_id: str | None = None
    payload: dict[str, Any] | None = None
    # The tool calls the saved conversation holds unfinished, by ID; OpenCode reports them to the model as
    # interrupted.
    interrupted_tool_calls: tuple[str, ...] = ()


def reprompt_decision(export: dict[str, Any] | None, prompt: str) -> RepromptDecision:
    """Reprompt when the saved root session starts with the task's prompt and holds a completed assistant message.

    ``export`` is the conversation ``snapshot.snapshot`` exported from the sandbox's database, ``None`` when there is
    none. A conversation of another task, or one the model never answered, starts afresh.
    """
    if export is None:
        return RepromptDecision(reprompt=False, reason="no saved OpenCode conversation")
    messages = export.get("messages") or []
    users = [message for message in messages if message["info"].get("role") == "user"]
    if not users:
        return RepromptDecision(reprompt=False, reason="the saved conversation has no user turn")
    if _text(users[0]).strip() != prompt.strip():
        return RepromptDecision(reprompt=False, reason="the saved conversation does not start with this task's prompt")
    answered = [
        message
        for message in messages
        if message["info"].get("role") == "assistant" and (message["info"].get("time") or {}).get("completed")
    ]
    if not answered:
        return RepromptDecision(reprompt=False, reason="the model had not answered the saved conversation yet")
    try:
        payload = reprompt_payload(messages)
    except (KeyError, TypeError, ValueError) as error:
        return RepromptDecision(reprompt=False, reason=f"the saved conversation cannot be reprompted: {error}")
    return RepromptDecision(
        reprompt=True,
        reason="reprompting the saved conversation",
        session_id=export["session_id"],
        payload=payload,
        interrupted_tool_calls=interrupted_tool_calls(messages),
    )


def reprompt_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Re-enter the last user message, keeping its model and agent settings, with no new parts."""
    users = [message for message in messages if message["info"].get("role") == "user"]
    if not users:
        raise ValueError("cannot reprompt an OpenCode session without a saved user turn")
    # Message IDs order the conversation; re-posting the user message updates its timestamps.
    latest = max(users, key=_message_id)
    user = latest["info"]
    payload: dict[str, Any] = {"messageID": _message_id(latest), "parts": []}
    for name in ("agent", "system", "format", "tools"):
        if user.get(name) is not None:
            payload[name] = user[name]
    model = user.get("model")
    if model:
        payload["model"] = {name: model[name] for name in ("providerID", "modelID")}
        if model.get("variant") is not None:
            payload["variant"] = model["variant"]
    return payload


def interrupted_tool_calls(messages: list[dict[str, Any]]) -> tuple[str, ...]:
    """The tool calls the saved conversation holds unfinished (``pending`` or ``running``), by ID.

    The ID is the part's ``callID``, else its ``id``, else its place in the conversation, so a caller that sees the
    same conversation again can tell the calls it already counted.
    """
    return tuple(
        str(part.get("callID") or part.get("id") or f"{_message_id(message)}:{index}")
        for message in messages
        for index, part in enumerate(message.get("parts", []))
        if part.get("type") == "tool" and (part.get("state") or {}).get("status") in ("pending", "running")
    )


def _message_id(message: dict[str, Any]) -> str:
    """A message's id: the export keeps it beside ``info``, because OpenCode stores it on the row, not in the data."""
    return message.get("id") or message["info"]["id"]


def _text(message: dict[str, Any]) -> str:
    # Text OpenCode added itself, for example for an attached file, is not part of the prompt.
    return "".join(
        part.get("text", "")
        for part in message.get("parts", [])
        if part.get("type") == "text" and not part.get("synthetic")
    )
