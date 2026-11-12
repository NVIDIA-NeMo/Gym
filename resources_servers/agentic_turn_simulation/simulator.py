# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tool simulator LLM's side of a call: what it is shown, how its reply becomes the tool's output, and the
output that marks a failed simulation."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from resources_servers.agentic_turn_simulation.items import Items
from resources_servers.agentic_turn_simulation.render import SIMULATOR_RENDER, render_conversation
from resources_servers.agentic_turn_simulation.workspace import Workspace


# `simulator_failure` closes the bracket after the reason.
SIMULATOR_FAILED = "[tool simulator failed"


def simulation_history(turn: Items, call_id: str | None) -> list[dict[str, Any]]:
    """The items of the turn so far that the simulator sees before the call to simulate.

    `turn` is the agent's running output: per step, the model's items, then that step's tool outputs as they
    are produced. The simulator sees every earlier step, this step's message and the outputs of this step's
    earlier calls, but not this step's calls: the model wrote them all at once, so a later sibling call says
    nothing about how an earlier one turned out. A turn that came without the call is all history. The call is the
    last one with its id: parsers that take ids from the model's text can repeat one across steps.
    """
    found = [i for i, x in enumerate(turn) if x.get("type") == "function_call" and x.get("call_id") == call_id]
    if not found:
        return list(turn)
    start = end = found[-1]
    while start > 0 and turn[start - 1].get("type") != "function_call_output":
        start -= 1
    while end < len(turn) and turn[end].get("type") != "function_call_output":
        end += 1
    step_model_items = [x for x in turn[start:end] if x.get("type") != "function_call"]
    sibling_outputs = [x for x in turn[end:] if x.get("type") == "function_call_output"]
    return list(turn[:start]) + step_model_items + sibling_outputs


def simulator_user_message(call: dict[str, Any], history: Items, workspace: Workspace | None) -> str:
    """The conversation with the call to simulate, then the known contents of the files it names."""
    conversation = render_conversation(history, [call], SIMULATOR_RENDER)
    conversation = conversation.replace("<step_under_evaluation>", "<tool_call_to_simulate>").replace(
        "</step_under_evaluation>", "</tool_call_to_simulate>"
    )
    files = workspace.shown_files(call) if workspace is not None else ""
    return conversation + (f"\n\n<workspace_files>\n{files}\n</workspace_files>" if files else "")


def parse_text_reply(message: dict[str, Any], finish_reason: str | None = None) -> str | None:
    """The reply's text verbatim (any format); None when it is empty or was cut off by the output budget.
    Reasoning, inline or in its own channel, is never the output."""
    if finish_reason == "length":
        return None
    content = message.get("content") or ""
    if "</think>" in content:
        content = content.rsplit("</think>", 1)[1]
    content = content.strip("\r\n")  # keep leading spaces: they are part of e.g. line-numbered file contents
    return content if content.strip() else None


def parse_json_text_reply(message: dict[str, Any], finish_reason: str | None = None) -> str | None:
    """For tools that answer in JSON (with the odd plain-text message): a reply that opens a JSON object must hold
    a complete one, kept verbatim with any bracketed note the harness appends (`\\n\\n[...]`) but none of the
    simulator's own commentary; any other reply is text. None when empty, cut off, or an unfinished object."""
    text = parse_text_reply(message, finish_reason)
    if text is None:
        return None
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n?```", stripped, re.S)
    if fenced:
        stripped = fenced.group(1).strip()
    if not stripped.startswith("{"):
        return text
    try:
        _, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    note = re.match(r"\n\n\[.*", stripped[end:], re.S)
    return stripped[:end] + (note.group(0).rstrip() if note else "")


# Harness.reply_format -> its parser. A parser returns the tool's output, or None when the reply is unusable (the
# call is retried).
REPLY_PARSERS: dict[str, Callable[[dict[str, Any], str | None], str | None]] = {
    "text": parse_text_reply,
    "json_text": parse_json_text_reply,
}


def simulator_failure(reason: str) -> str:
    """The tool output when the simulator itself failed. /verify masks a turn that received one."""
    return json.dumps({"status": "error", "message": f"{SIMULATOR_FAILED}: {reason[:120]}]"})


# Where a `simulator_failure` output starts: json.dumps escapes nothing in this prefix.
_FAILURE_START = '{"status": "error", "message": "' + SIMULATOR_FAILED
_JSON = json.JSONDecoder()


def holds_simulator_failure(text: str) -> bool:
    """Whether a tool result holds a complete `simulator_failure` output: that exact JSON object (these two keys, an
    error status, a message `[tool simulator failed: <reason>]`), not merely its phrase, which a file or a command's
    output may contain. The object may sit inside the text: Cline frames a result under its header and joins a
    turn's results into one user message."""
    start = text.find(_FAILURE_START)
    while start >= 0:
        try:
            value, _ = _JSON.raw_decode(text, start)
        except json.JSONDecodeError:
            value = None
        if (
            isinstance(value, dict)
            and value.keys() == {"status", "message"}
            and value["status"] == "error"
            and isinstance(value["message"], str)
            and value["message"].startswith(f"{SIMULATOR_FAILED}: ")
            and value["message"].endswith("]")
        ):
            return True
        start = text.find(_FAILURE_START, start + 1)
    return False
