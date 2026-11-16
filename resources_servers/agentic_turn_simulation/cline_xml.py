# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cline's XML tool protocol as the agent meets it: how a call is found in the agent's text, and Cline's own messages
around it. Standard library only, so the agent can import it without the rest of the server.

Cline gives the model no function tools: the model writes calls as XML in its reply
(`<read_file><path>a.py</path></read_file>`), and Cline runs them in order. The parsing rules are the ones the rows
were converted with: only known tool names count, an unclosed call runs to the end of the text, tool tags inside a
call's body belong to that body, and the parameters are the body's top-level `<param>value</param>` pairs (leading
and trailing newlines stripped).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


KNOWN_TOOLS = tuple(
    "execute_command read_file write_to_file replace_in_file search_files list_files list_code_definition_names "
    "browser_action use_mcp_tool access_mcp_resource ask_followup_question attempt_completion plan_mode_respond "
    "load_mcp_documentation new_task condense summarize_task report_bug new_rule web_fetch web_search "
    "generate_explanation use_skill use_subagents apply_patch focus_chain".split()
)
KNOWN_TOOL_PATTERN = "|".join(KNOWN_TOOLS)
XML_CALL = re.compile(r"<(" + KNOWN_TOOL_PATTERN + r")>(.*?)(?:</\1>|\Z)", re.S)


@dataclass(frozen=True)
class XmlCall:
    name: str
    params: dict[str, str]
    # Where the call sits in the text: [start, end).
    start: int
    end: int


def xml_params(body: str) -> dict[str, str]:
    """Top-level <param>value</param> pairs of one XML tool call body."""
    out: dict[str, str] = {}
    pos = 0
    while True:
        m = re.match(r"\s*<([a-z_]+)>", body[pos:])
        if not m:
            break
        name = m.group(1)
        start = pos + m.end()
        end = body.find(f"</{name}>", start)
        if end < 0:
            out[name] = body[start:].strip("\n")
            break
        out[name] = body[start:end].strip("\n")
        pos = end + len(name) + 3
    return out


def calls(text: str | None) -> list[XmlCall]:
    """The tool calls Cline would run for this reply text, in order ([] when it has no tool tag)."""
    return [
        XmlCall(name=m.group(1), params=xml_params(m.group(2)), start=m.start(), end=m.end())
        for m in XML_CALL.finditer(text or "")
    ]


# Cline's reply to a model response without a tool call (formatResponse.noToolsUsed, as Cline
# sends it): sent as the next user message, before the environment details, and the model retries.
NO_TOOLS_USED = (
    "[ERROR] You did not use a tool in your previous response! Please retry with a tool use.\n"
    "\n"
    "# Reminder: Instructions for Tool Use\n"
    "Tool uses are formatted using XML-style tags. The tool name is enclosed in opening and closing tags, and each "
    "parameter is similarly enclosed within its own set of tags. Here's the structure:\n"
    "<tool_name>\n"
    "<parameter1_name>value1</parameter1_name>\n"
    "<parameter2_name>value2</parameter2_name>\n"
    "...\n"
    "</tool_name>\n"
    "For example:\n"
    "<attempt_completion>\n"
    "<result>\n"
    "I have completed the task...\n"
    "</result>\n"
    "</attempt_completion>\n"
    "Always adhere to this format for all tool uses to ensure proper parsing and execution.\n"
    "\n"
    "# Next Steps\n"
    "\n"
    "If you have completed the user's task, use the attempt_completion tool. \n"
    "If you require additional information from the user, use the ask_followup_question tool. \n"
    "Otherwise, if you have not completed the task and do not need additional information, then proceed with the "
    "next step of the task. \n"
    "(This is an automated message, so do not respond to it conversationally.)"
)
# Cline's default maxConsecutiveMistakes: at this many consecutive replies without a tool call it stops and asks the
# user.
MAX_CONSECUTIVE_MISTAKES = 3
# Cline closes every user message with its editor state in this block.
ENVIRONMENT_DETAILS = "<environment_details>"
_ENVIRONMENT_DETAILS_BLOCK = re.compile(r"<environment_details>.*?</environment_details>", re.S)


def no_tools_used(items: list[dict[str, Any]]) -> list[str]:
    """Cline's next user message after a reply without a tool call, as text parts: NO_TOOLS_USED, then the most
    recent <environment_details> block in `items`' user messages (the turn's input and output so far), if any."""
    for message in reversed(items):
        if message.get("type", "message") != "message" or message.get("role") != "user":
            continue
        content = message.get("content")
        texts = (
            [content]
            if isinstance(content, str)
            else [p.get("text") or "" for p in content or [] if isinstance(p, dict)]
        )
        found = [m.group(0) for text in texts for m in _ENVIRONMENT_DETAILS_BLOCK.finditer(text)]
        if found:
            return [NO_TOOLS_USED, found[-1]]
    return [NO_TOOLS_USED]
