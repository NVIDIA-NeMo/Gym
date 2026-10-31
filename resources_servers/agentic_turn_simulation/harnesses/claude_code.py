# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code's harness: its file-tool results in its own words, answered in code where the workspace knows the
result, its shell's output for a read-back of a known file, and how its shell reports a failure."""

from __future__ import annotations

import re
from typing import Any, Collection, Mapping

from resources_servers.agentic_turn_simulation.harnesses.base import Harness, ObstacleTexts
from resources_servers.agentic_turn_simulation.items import Items, calls_and_outputs
from resources_servers.agentic_turn_simulation.workspace import (
    FileFormat,
    FileTool,
    FileToolSpec,
    FileToolWorkspace,
    Prior,
    edit_args,
    int_arg,
    plain_shell_args,
)


CC_TOOL_ERROR = "<tool_use_error>{}</tool_use_error>"
# Since 2.1.128 Write and Edit confirmations end with this.
CC_SUFFIX = " (file state is current in your context — no need to Read it back)"
CC_EMPTY_FILE = "<system-reminder>Warning: the file exists but the contents are empty.</system-reminder>"
CC_REMINDER_TAIL = re.compile(r"(?:\n\n<system-reminder>\n.*?\n</system-reminder>\n?)+\Z", re.S)
CC_NUMBERED = re.compile(r"(\d+)(?:\t(.*))?\Z", re.S)
# Reads Claude Code answers its own way: memory files carry an age note, files already in the context a stub.
CC_SPECIAL_READ = re.compile(r"(?:^|[\\/])(?:CLAUDE(?:\.local)?\.md|AGENTS\.md)$|[\\/]\.claude[\\/]")
CC_MAX_READ_CHARS = 40_000  # well under its 25,000-token and 256 KB read guards
CC_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
# Bash output longer than this is cut in the middle; a read-back answered in code stays well under it.
CC_MAX_SHELL_CHARS = 25_000
CC_SHELL_EMPTY = "(Bash completed with no output)"
CC_SEARCH_REMINDER = (
    "REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks."
)


def _search_failed(query: str, shown: str | None) -> str:
    return (
        f'Web search results for query: "{query}"\n\nWeb search error: unavailable\n\nWeb tool request failed.'
        f"\n\n\n{CC_SEARCH_REMINDER}"
    )


def _no_links(query: str, shown: str | None) -> str:
    return (
        f'Web search results for query: "{query}"\n\nNo links found.\n\nNo web search results found for: {query}'
        f"\n\n\n{CC_SEARCH_REMINDER}"
    )


# How its web tools' failure texts begin (WebFetch) and what follows a failed search's query line (WebSearch).
CC_FETCH_FAILED = ("read ECONNRESET", "The server returned HTTP ")
CC_SEARCH_FAILED = ("\n\nWeb search error: ", "\n\nNo links found.")
# Its web tools' failure texts, for injected obstacles.
CLAUDE_CODE_OBSTACLE_TEXTS = {
    "WebFetch": ObstacleTexts(
        target="url",
        failed=lambda output: output.startswith(CC_FETCH_FAILED),
        transient="read ECONNRESET",
        blocked_route=(
            "The server returned HTTP 403 Forbidden.\n\nThe response body was not retrieved. If this URL requires "
            "authentication, use an authenticated tool (e.g. `gh` for GitHub, or an MCP-provided fetch tool) instead "
            "of WebFetch."
        ),
    ),
    "WebSearch": ObstacleTexts(
        target="query",
        search=True,
        failed=lambda output: any(marker in output for marker in CC_SEARCH_FAILED),
        transient=_search_failed,
        empty_search=_no_links,
    ),
}


class ClaudeCodeFormat(FileFormat):
    """Read: `<n>\\t<line>` per line (unpadded, from 1, or from `offset`; offset 0 labels the first line 0);
    Write/Edit: one-line confirmations; errors in <tool_use_error>. A Write needs the file read first, so only
    an overwrite of a file whose content is known is answered here."""

    dedupes_reads = True

    def observe(self, ws: FileToolWorkspace, name: str, args: Mapping[str, Any], output: str) -> None:
        if name in ("Write", "Edit") and output.startswith(("File created successfully at: ", "The file ")):
            if "(note:" not in output:
                ws.conventions["suffix"] = output.endswith(CC_SUFFIX)

    def _suffix(self, ws: FileToolWorkspace) -> str | None:
        """The confirmation suffix, as this session's results show it or else as the version has it."""
        shown = ws.conventions.get("suffix")
        if shown is None:
            shown = ws.at_least(2, 1, 128)
        return None if shown is None else (CC_SUFFIX if shown else "")

    def read(self, ws: FileToolWorkspace, path: str, content: str, args: Mapping[str, Any]) -> str | None:
        ok_offset, offset = int_arg(args, "offset")
        ok_limit, limit = int_arg(args, "limit")
        if not (ok_offset and ok_limit) or args.get("pages") is not None or CC_SPECIAL_READ.search(path):
            return None
        if (offset is not None and offset < 0) or (limit is not None and limit <= 0):
            return None
        text = content.replace("\r\n", "\n")
        if text == "":
            return CC_EMPTY_FILE if offset in (None, 0, 1) else None
        lines = text.split("\n")
        skip = max((offset or 1) - 1, 0)
        if skip >= len(lines):
            return None  # its "shorter than the provided offset" warning counts lines in a way the data leaves open
        start = 1 if offset is None else offset
        shown = lines[skip : skip + limit] if limit else lines[skip:]
        out = "\n".join(f"{start + i}\t{line}" for i, line in enumerate(shown))
        return out if len(out) <= CC_MAX_READ_CHARS else None

    def read_back(self, ws: FileToolWorkspace, args: Mapping[str, Any], output: str) -> str | None:
        ok_offset, offset = int_arg(args, "offset")
        if not ok_offset or offset not in (None, 0, 1) or args.get("limit") is not None:
            return None
        output = CC_REMINDER_TAIL.sub("", output)
        if output == CC_EMPTY_FILE:
            return ""
        texts = []
        for k, line in enumerate(output.split("\n")):
            m = CC_NUMBERED.match(line)
            if m is None or int(m.group(1)) != k + (0 if offset == 0 else 1):
                return None
            texts.append(m.group(2) or "")
        return "\n".join(texts)

    def write(
        self, ws: FileToolWorkspace, path: str, content: str, prior: Prior, args: Mapping[str, Any]
    ) -> str | None:
        suffix = self._suffix(ws)
        if prior != "known" or suffix is None:
            return None  # a new file, or one never read: created, or refused as unread
        return f"The file {path} has been updated successfully.{suffix}"

    def write_succeeded(self, ws: FileToolWorkspace, output: str) -> bool:
        return output.startswith("File created successfully at: ") or (
            output.startswith("The file ") and " has been updated successfully." in output
        )

    def edit(
        self, ws: FileToolWorkspace, path: str, before: str, args: Mapping[str, Any], tool: FileTool
    ) -> tuple[str | None, str | None]:
        parsed = edit_args(args, tool)
        if parsed is None or "\r" in before:
            return None, None
        old, new, replace_all = parsed
        if old == new:
            return CC_TOOL_ERROR.format("No changes to make: old_string and new_string are exactly the same."), None
        if old == "":
            return CC_TOOL_ERROR.format("Cannot create new file - file already exists."), None
        count = before.count(old)
        if count == 0:
            if not old.isascii() or old.translate(CC_QUOTES) != old or before.translate(CC_QUOTES) != before:
                return None, None  # it retries with escapes or quotes normalised, with a version-dependent note
            return CC_TOOL_ERROR.format(f"String to replace not found in file.\nString: {old}"), None
        if count > 1 and not replace_all:
            return (
                CC_TOOL_ERROR.format(
                    f"Found {count} matches of the string to replace, but replace_all is false. To replace all "
                    "occurrences, set replace_all to true. To replace only one occurrence, please provide more "
                    f"context to uniquely identify the instance.\nString: {old}"
                ),
                None,
            )
        target = old
        if new == "" and not old.endswith("\n") and old + "\n" in before:
            target = old + "\n"  # deleting text takes its line break with it
        after = before.replace(target, new) if replace_all else before.replace(target, new, 1)
        suffix = self._suffix(ws)
        if suffix is None:
            return None, after
        if replace_all:
            return f"The file {path} has been updated. All occurrences were successfully replaced.{suffix}", after
        return f"The file {path} has been updated successfully.{suffix}", after

    def edit_succeeded(self, ws: FileToolWorkspace, output: str) -> bool:
        # The variant noting that the file had changed on disk means the content we know is not the file's.
        return output.startswith("The file ") and " has been updated" in output and "(note:" not in output

    def shell(self, ws: FileToolWorkspace, name: str, args: Mapping[str, Any], stdout: str) -> str | None:
        """Bash: the output without its trailing line breaks, or its no-output note. It trims trailing whitespace
        and drops leading blank lines, so output whose first line is blank or that ends in other whitespace goes to
        the LLM (the exact trimming is not verified), and so does output with a carriage return, and PowerShell."""
        if name != "Bash" or not plain_shell_args(args, "command", ("description", "timeout")):
            return None
        out = stdout.rstrip("\n")
        if out != out.rstrip() or (out and not out.split("\n", 1)[0].strip()) or "\r" in out:
            return None
        return (out or CC_SHELL_EMPTY) if len(out) <= CC_MAX_SHELL_CHARS else None


def claude_code_shell_failures(items: Items, tool_names: Collection[str] = ()) -> int:
    """Bash / PowerShell calls that exited non-zero (`Exit code N` first line) or hit their timeout."""
    n = 0
    for _, output in calls_and_outputs(items, ("Bash", "PowerShell")):
        first = output.split("\n", 1)[0]
        exited_nonzero = first.startswith("Exit code ") and first.removeprefix("Exit code ").strip() not in ("0", "")
        timed_out = output.startswith(("Command did not complete within its", "Command timed out"))
        if exited_nonzero or timed_out:
            n += 1
    return n


CLAUDE_CODE_FILES = FileToolSpec(
    format=ClaudeCodeFormat(),
    file_tools={
        "Read": FileTool("read", "file_path", offset="offset", limit="limit"),
        "Write": FileTool("write", "file_path"),
        "Edit": FileTool("edit", "file_path", replace_all="replace_all"),
    },
    shell_tools={"Bash": "command", "PowerShell": "command"},
    opaque_tools=frozenset({"Agent", "Task", "Workflow", "SendMessage", "EnterWorktree", "ExitWorktree"}),
    readonly_tools=frozenset(
        "Glob Grep LS WebFetch WebSearch TodoWrite TaskCreate TaskGet TaskList TaskUpdate TaskOutput TaskStop "
        "AskUserQuestion EnterPlanMode ExitPlanMode CronCreate CronDelete CronList ScheduleWakeup ReportFindings Skill "
        "ToolSearch".split()
    ),
)


CLAUDE_CODE = Harness(
    name="claude_code",
    simulator_prompt="tool-simulator-claude-code.md",
    judge_prompt="judge-system-harness.md",
    judge_environment="judge-environment-claude-code.md",
    # ExitPlanMode hands the plan to the user for approval.
    stop_tools=("AskUserQuestion", "ExitPlanMode"),
    files=CLAUDE_CODE_FILES,
    shell_failures=claude_code_shell_failures,
    simulator_sees_tools=True,
    gates_missing_arguments=True,
    obstacle_texts=CLAUDE_CODE_OBSTACLE_TEXTS,
)
