# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cline's XML tool protocol end to end: the transcript as function calls, results framed the way Cline sends them
back, the tools its system prompt documents, and what its results show of the files. Finding the calls in the
agent's text is `cline_xml`'s."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Collection, Mapping

from resources_servers.agentic_turn_simulation.cline_xml import (
    ENVIRONMENT_DETAILS,
    KNOWN_TOOL_PATTERN,
    KNOWN_TOOLS,
    XmlCall,
    calls,
)
from resources_servers.agentic_turn_simulation.harnesses.base import Harness
from resources_servers.agentic_turn_simulation.items import Items, parse_args
from resources_servers.agentic_turn_simulation.workspace import FileFormat, FileTool, FileToolSpec, FileToolWorkspace


# Cline runs a reply's XML calls in order and sends the results back in the next user message, each headed
# `[read_file for 'a.py'] Result:`, with the environment details last. The server reads such a transcript as
# function calls and their outputs, so the simulator and the judge see tool use as tool use and the code gates can
# check it.

# Tools whose call is the agent's reply to the user, by the parameter holding the reply: the judge sees what the
# agent said.
REPLY_TOOLS = {"attempt_completion": "result", "plan_mode_respond": "response"}


def message_text(content: Any) -> str:
    """A Cline message's text. Unlike `items.content_text`, which runs the parts together, its parts are separate
    blocks: the non-empty ones are joined by line breaks."""
    if isinstance(content, list):
        return "\n".join(c.get("text", "") or "" for c in content if isinstance(c, dict) and c.get("text"))
    return content or ""


def _assistant_message(text: str, item: dict[str, Any]) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": text, **({"id": item["id"]} if "id" in item else {})}


def _reply_items(text: str, found: list[XmlCall], item: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    """An assistant message as what the agent said and the calls it made (`found` in its text), in order.

    The calls get made-up ids, `xml{idx}_{k}_{digest}`: the message's index in the transcript, the call's in the
    message, and a hash of the message's text. Converting the same transcript again (at seed, on every
    /simulate_tool_call, at /verify) gives the same ids; the prefix and the turn are converted separately, with idx
    restarting at 0, and the hash keeps their ids apart.
    """
    out: list[dict[str, Any]] = []
    said: list[str] = []
    pos = 0
    digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:8]
    for k, call in enumerate(found):
        said.append(text[pos : call.start].strip())
        pos = call.end
        if call.name in REPLY_TOOLS:
            said.append(call.params.get(REPLY_TOOLS[call.name], "").strip())
            continue
        if any(said):
            out.append(_assistant_message("\n\n".join(p for p in said if p), item))
        said = []
        arguments = json.dumps(call.params, ensure_ascii=False)
        out.append(
            {"type": "function_call", "call_id": f"xml{idx}_{k}_{digest}", "name": call.name, "arguments": arguments}
        )
    said.append(text[pos:].strip())
    if any(said):
        out.append(_assistant_message("\n\n".join(p for p in said if p), item))
    return out


def _result_start(name: str) -> re.Pattern[str]:
    """Where a result for this tool starts in a user message: `[name] Result:` or `[name for ...] Result:`."""
    return re.compile(r"(?:^|(?<=\n))\[" + re.escape(name) + r"(?: for .*?)?\] Result:", re.S)


def open_calls(items: Items) -> list[str]:
    """Ids of the latest calls still waiting for their outputs, in order (a user message closes the wait)."""
    waiting: list[str] = []
    for item in items:
        kind = item.get("type", "message")
        if kind == "function_call":
            waiting.append(item.get("call_id"))
        elif kind == "function_call_output" and item.get("call_id") in waiting:
            waiting.remove(item.get("call_id"))
        elif kind == "message" and item.get("role") == "user":
            waiting = []
    return waiting


# --- results ------------------------------------------------------------------------------------------------

# The parameter a result header names, per tool (`[read_file for 'src/app.ts'] Result:`).
HEADER_TARGET = {
    "execute_command": "command",
    "read_file": "path",
    "write_to_file": "path",
    "replace_in_file": "path",
    "list_files": "path",
    "list_code_definition_names": "path",
    "new_rule": "path",
    "use_mcp_tool": "server_name",
    "access_mcp_resource": "server_name",
    "ask_followup_question": "question",
    "browser_action": "action",
    "web_fetch": "url",
}
_HEADER_LINE = re.compile(r"\A\s*\[[a-z_]+(?: for .*?)?\] Result:\n?", re.S)


def result_header(name: str, params: Mapping[str, Any]) -> str:
    """The line Cline puts before a tool's result."""
    if name == "search_files":
        pattern = params.get("file_pattern")
        where = f" in '{pattern}'" if pattern else ""
        return f"[search_files for '{params.get('regex', '')}'{where}] Result:"
    if name == "use_skill":
        return f'[use_skill for "{params.get("skill_name", "")}"] Result:'
    if name == "new_task":
        return "[new_task for creating a new task] Result:"
    if name in HEADER_TARGET:
        return f"[{name} for '{params.get(HEADER_TARGET[name], '')}'] Result:"
    return f"[{name}] Result:"


def _result_body(output: str) -> str:
    """A result without its header line and the environment details."""
    body = _HEADER_LINE.sub("", output, count=1)
    cut = body.find(ENVIRONMENT_DETAILS)
    return (body[:cut] if cut >= 0 else body).rstrip("\n")


def _split_results(text: str, pending: list[dict[str, Any]], item: dict[str, Any]) -> list[dict[str, Any]]:
    """A user message as the results of the latest reply's calls: one output per pending call whose header is
    found, in order (text before the first header, such as the user's feedback on an approval, goes with the first;
    the environment details with the last). A message without one is passed on unchanged (its text joined)."""
    starts: list[int] = []
    pos = 0
    for call in pending:
        match = _result_start(call["name"]).search(text, pos)
        if match is None:
            break
        starts.append(match.start())
        pos = match.end()
    if not starts:
        return [{**item, "content": text}]
    bounds = [0] + starts[1:] + [len(text)]
    return [
        {"type": "function_call_output", "call_id": call["call_id"], "output": text[lo:hi].rstrip("\n")}
        for call, lo, hi in zip(pending, bounds, bounds[1:])
    ]


# --- the tools its system prompt documents --------------------------------------------------------------------

# A tool's documentation in Cline's system prompt starts at "## read_file" (the full prompt) or at
# "**read_file** — ..." (the compact one) and runs to the next heading or tool.
_DOC_START = re.compile(r"(?m)^(?:## (" + KNOWN_TOOL_PATTERN + r")[ \t]*$|\*\*(" + KNOWN_TOOL_PATTERN + r")\*\*)")
_DOC_END = re.compile(r"(?m)^(?:#{1,2} |====|\*\*(?:" + KNOWN_TOOL_PATTERN + r")\*\*)")
DOC_PARAMETERS = {"type": "object", "description": "XML parameters, as the description documents them"}


def tool_docs(items: Items) -> list[dict[str, Any]]:
    """The tools Cline's system prompt documents, as tool definitions: each tool's own section of the most
    recent system prompt (Cline re-sends it on a mode switch) is its description."""
    prompts = [
        message_text(x.get("content"))
        for x in items
        if x.get("type", "message") == "message" and x.get("role") in ("system", "developer")
    ]
    if not prompts:
        return []
    text, docs, seen = prompts[-1], [], set()
    for m in _DOC_START.finditer(text):
        name = m.group(1) or m.group(2)
        if name in seen:
            continue
        seen.add(name)
        end = _DOC_END.search(text, m.end())
        body = text[m.end() : end.start() if end else len(text)].strip().lstrip("—").strip()
        docs.append({"type": "function", "name": name, "description": body, "parameters": DOC_PARAMETERS})
    return docs


# --- files and shell ----------------------------------------------------------------------------------------

_FINAL_CONTENT = re.compile(r'<(final_file_content|file_content) path="[^"]*">\n(.*?)\n?</\1>', re.S)
_LABELLED = re.compile(r"(\d+) \| ?(.*)\Z", re.S)
_WHOLE_FILE_TAIL = re.compile(r"\n\n\(File has (\d+) lines total\.\)\Z")


class ClineFormat(FileFormat):
    """Learning only: every result ends with editor state the workspace cannot know (open tabs, the time), so no
    call is answered in code. read_file labels lines `N | line` with a `(File has N lines total.)` tail, and a
    write or edit shows the whole file after it (after a failed edit, reverted)."""

    def read_back(self, ws: FileToolWorkspace, args: Mapping[str, Any], output: str) -> str | None:
        if args.get("start_line") not in (None, "", "1") or args.get("end_line") not in (None, ""):
            return None
        body = _result_body(output)
        if body.startswith("[File already read]"):
            body = body.split("\n", 1)[1] if "\n" in body else ""
        tail = _WHOLE_FILE_TAIL.search(body)
        if tail is None:
            return None
        texts = []
        for k, line in enumerate(body[: tail.start()].split("\n"), 1):
            m = _LABELLED.match(line)
            if m is None or int(m.group(1)) != k:
                return None
            texts.append(m.group(2))
        return "\n".join(texts) if len(texts) == int(tail.group(1)) else None

    def content_after(self, ws: FileToolWorkspace, args: Mapping[str, Any], output: str) -> str | None:
        m = _FINAL_CONTENT.search(output)
        return m.group(2) if m else None


CLINE_FILES = FileToolSpec(
    format=ClineFormat(),
    file_tools={
        "read_file": FileTool("read", "path", offset="start_line", limit="end_line"),
        "write_to_file": FileTool("write", "path"),
        "replace_in_file": FileTool("edit", "path", old="diff", new="diff"),
    },
    shell_tools={"execute_command": "command"},
    opaque_tools=frozenset({"use_subagents", "new_task", "apply_patch"}),
    readonly_tools=frozenset(
        "search_files list_files list_code_definition_names ask_followup_question attempt_completion plan_mode_respond "
        "load_mcp_documentation web_fetch web_search browser_action access_mcp_resource use_skill focus_chain condense "
        "summarize_task report_bug generate_explanation".split()
    ),
)
EXIT_FAILED = re.compile(r"Command failed with exit code (-?\d+)")


def cline_shell_failures(items: Items, tool_names: Collection[str] = ()) -> int:
    """execute_command results that report a non-zero exit code or a timeout."""
    command_ids = {
        x.get("call_id") for x in items if x.get("type") == "function_call" and x.get("name") == "execute_command"
    }
    n = 0
    for item in items:
        if item.get("type") != "function_call_output" or item.get("call_id") not in command_ids:
            continue
        body = _result_body(str(item.get("output") or ""))
        if EXIT_FAILED.search(body) or body.startswith(("Command timed out after", "Command execution timed out")):
            n += 1
    return n


class ClineHarness(Harness):
    """Cline: its transcript is read as tool items, and each simulated result is framed as Cline sends it."""

    def to_tool_items(self, items: Items) -> list[dict[str, Any]]:
        """An assistant message becomes what the agent said and the calls it made, and the user message after it
        the outputs of those calls (`_split_results`). Anything else passes through unchanged."""
        out: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []  # the calls of the latest reply, in order
        for idx, item in enumerate(items):
            kind, role = item.get("type", "message"), item.get("role")
            if kind != "message" or role not in ("assistant", "user"):
                out.append(item)
                continue
            text = message_text(item.get("content"))
            if role == "assistant":
                found = calls(text)
                converted = _reply_items(text, found, item, idx) if found else [_assistant_message(text, item)]
                out.extend(converted)
                pending = [x for x in converted if x.get("type") == "function_call"]
            else:
                out.extend(_split_results(text, pending, item))
                pending = []
        return out

    def described_tools(self, tools: Items, items: Items) -> list[dict[str, Any]]:
        return list(tools) if tools else tool_docs(items)

    def locate_call(self, turn: Items, call_id: str) -> tuple[str | None, bool]:
        # XML calls carry no ids of their own: the call to answer is the reply's first one without a result yet.
        waiting = open_calls(turn)
        return (waiting[0] if waiting else None), len(waiting) > 1

    def frame(self, call: dict[str, Any], output: str, *, more_follow: bool = False) -> str:
        """The result as it appears in Cline's next user message: its header line, then the result. While more of
        the reply's results follow, the environment details (which close the message) are left for the last."""
        if more_follow and ENVIRONMENT_DETAILS in output:
            output = output[: output.index(ENVIRONMENT_DETAILS)].rstrip("\n")
        return result_header(call.get("name") or "", parse_args(call) or {}) + "\n" + output

    def scored_turn(self, items: Items) -> list[dict[str, Any]]:
        """A reference turn that ends by asking the user keeps the user's answer as the question's result (as
        Cline does); the scored turn ends at the question."""
        turn = super().scored_turn(items)
        if not turn or turn[-1].get("type") != "function_call_output":
            return turn
        call_id = turn[-1].get("call_id")
        asked = any(
            x.get("type") == "function_call" and x.get("call_id") == call_id and x.get("name") in self.stop_tools
            for x in turn
        )
        return turn[:-1] if asked else turn


# Selected by the row's protocol (`select_harness`): rows name the harness "cline". The name is the
# `harness` value /verify reports.
CLINE = ClineHarness(
    name="cline_xml",
    simulator_prompt="tool-simulator-cline.md",
    judge_prompt="judge-system-harness.md",
    judge_environment="judge-environment-cline.md",
    # attempt_completion and plan_mode_respond are its replies to the user; ask_followup_question its question.
    stop_tools=("ask_followup_question", "attempt_completion", "plan_mode_respond"),
    files=CLINE_FILES,
    shell_failures=cline_shell_failures,
    simulator_sees_tools=True,
    protocol="cline_xml",
    default_tools=frozenset(KNOWN_TOOLS),
)
