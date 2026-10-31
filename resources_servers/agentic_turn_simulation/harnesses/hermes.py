# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermes' harness: its file-tool results in its own JSON, answered in code where the workspace knows the result, its
shell's result for a read-back of a known file, and how its shell reports a failure."""

from __future__ import annotations

import difflib
import json
import ntpath
import os
import re
from typing import Any, Collection, Mapping

from resources_servers.agentic_turn_simulation.harnesses.base import Harness
from resources_servers.agentic_turn_simulation.items import Items, calls_and_outputs, json_object
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


HERMES_LINTED = frozenset({".py", ".js", ".ts", ".go", ".rs"})  # its linters up to 0.11; their verdict is unknown
HERMES_BINARY = re.compile(
    r"\.(?:png|jpe?g|gif|webp|bmp|ico|tiff?|pdf|zip|gz|tgz|tar|7z|rar|exe|dll|so|dylib|bin|class|jar|pyc|"
    r"mp[34]|mov|avi|wav|flac|ogg|woff2?|ttf|otf|sqlite|db)$",
    re.I,
)
HERMES_PROTECTED = re.compile(
    r"(?:^|[\\/])(?:\.ssh|\.aws|\.gnupg|\.kube|\.docker|\.azure|\.hermes)(?:[\\/]|$)|[\\/]\.config[\\/]gh(?:[\\/]|$)|"
    r"(?:^|[\\/])\.(?:bashrc|zshrc|profile|bash_profile|zprofile|netrc|pgpass|npmrc|pypirc)$|"
    r"^/(?:etc|boot|usr/lib/systemd)(?:/|$)|docker\.sock"
)
HERMES_NUMBERED = re.compile(r" *(\d+)\|(.*)\Z", re.S)
HERMES_MAX_LINE = 2000  # characters: a longer line is cut and marked "... [truncated]"
HERMES_MAX_READ = 100_000
# read_file's page in lines: `limit`, by default 2000 from 0.20 and 500 before, at most HERMES_MAX_LIMIT.
HERMES_DEFAULT_LIMIT = 2000
HERMES_DEFAULT_LIMIT_BEFORE_0_20 = 500
HERMES_MAX_LIMIT = 2000
# A truncated page of a file larger than this many bytes, read with a limit above HERMES_LARGE_FILE_LIMIT, comes
# with a hint to read only the section needed.
# terminal output longer than this is cut; a read-back answered in code stays well under it.
HERMES_MAX_SHELL_CHARS = 25_000
HERMES_LARGE_FILE_BYTES = 512_000
HERMES_LARGE_FILE_LIMIT = 200


def _numbered(page: str, offset: int, *, padded: bool) -> str:
    """A page's lines labelled from `offset`, `{n:6d}|` (padded) or `{n}|`, each cut to HERMES_MAX_LINE characters."""
    numbered = []
    for i, line in enumerate(page.split("\n"), offset):
        if len(line) > HERMES_MAX_LINE:
            line = line[:HERMES_MAX_LINE] + "... [truncated]"
        numbered.append(f"{i:6d}|{line}" if padded else f"{i}|{line}")
    return "\n".join(numbered)


def _overlapping_count(text: str, sub: str) -> int:
    n, start = 0, 0
    while (pos := text.find(sub, start)) != -1:
        n, start = n + 1, pos + 1
    return n


class HermesFormat(FileFormat):
    """JSON results (`json.dumps(..., ensure_ascii=False)`, keys in Hermes' order). read_file numbers lines
    `{n:6d}|` up to 0.15.1 and `{n}|` from 0.17, pages 500 lines by default before 0.20 and 2000 from then.
    write_file and patch gained version-specific keys (lint, resolved_path, ...) after 0.11, so only rows up to
    0.11 get them answered here; a patch of a file its linters check, or that fuzzy matching would decide, goes
    to the LLM too."""

    dedupes_reads = True

    def observe(self, ws: FileToolWorkspace, name: str, args: Mapping[str, Any], output: str) -> None:
        if name == "terminal" and "approval" in (json_object(output) or {}):
            # Its command screening is on: later commands may come back flagged, with an approval note.
            ws.conventions["approval"] = True
        if name == "read_file":
            content = (json_object(output) or {}).get("content")
            if isinstance(content, str) and HERMES_NUMBERED.match(content.split("\n", 1)[0]):
                ws.conventions["padded"] = content.startswith(" ")

    def _padded(self, ws: FileToolWorkspace) -> bool | None:
        shown = ws.conventions.get("padded")
        if shown is None and ws.version is not None:
            shown = not ws.at_least(0, 17)
        return shown

    def read(self, ws: FileToolWorkspace, path: str, content: str, args: Mapping[str, Any]) -> str | None:
        ok_offset, offset = int_arg(args, "offset")
        ok_limit, limit = int_arg(args, "limit")
        padded = self._padded(ws)
        if not (ok_offset and ok_limit) or padded is None or HERMES_BINARY.search(path):
            return None
        if limit is None:
            if ws.version is None:
                return None
            limit = HERMES_DEFAULT_LIMIT if ws.at_least(0, 20) else HERMES_DEFAULT_LIMIT_BEFORE_0_20
        offset = 1 if offset is None else offset
        if offset < 1 or limit < 1:
            return None
        limit = min(limit, HERMES_MAX_LIMIT)
        lines = content.split("\n")
        if content.endswith("\n"):
            lines = lines[:-1]
        total = content.count("\n")
        shown = lines[offset - 1 : offset - 1 + limit]
        hints_empty_reads = ws.at_least(0, 20, 5) is not False  # from 0.20.5, or a version the row does not give
        if (not shown or content == "") and hints_empty_reads:
            return None  # an empty file or an offset past the end: those versions say so in a hint
        last = offset - 1 + len(shown)
        page = "\n".join(shown) + ("\n" if shown and (last < len(lines) or content.endswith("\n")) else "")
        body = _numbered(page, offset, padded=padded)
        if len(body) > HERMES_MAX_READ:
            return None  # its read budget: an error or a trimmed page, depending on the version
        size = len(content.encode("utf-8"))
        truncated = total > offset + limit - 1
        out: dict[str, Any] = {"content": body, "total_lines": total, "file_size": size, "truncated": truncated}
        if truncated:
            out["hint"] = (
                f"Use offset={offset + limit} to continue reading "
                f"(showing {offset}-{offset + limit - 1} of {total} lines)"
            )
        out.update({"is_binary": False, "is_image": False})
        if size > HERMES_LARGE_FILE_BYTES and limit > HERMES_LARGE_FILE_LIMIT and truncated:
            out["_hint"] = (
                f"This file is large ({size:,} bytes). Consider reading only the section you need with offset "
                "and limit to keep context usage efficient."
            )
        return json.dumps(out, ensure_ascii=False)

    def read_back(self, ws: FileToolWorkspace, args: Mapping[str, Any], output: str) -> str | None:
        ok_offset, offset = int_arg(args, "offset")
        result = json_object(output)
        if not ok_offset or offset not in (None, 1) or result is None or result.get("truncated") is not False:
            return None
        body, total = result.get("content"), result.get("total_lines")
        if not isinstance(body, str) or not isinstance(total, int) or "error" in result or result.get("dedup"):
            return None
        texts = []
        for k, line in enumerate(body.split("\n"), 1):
            m = HERMES_NUMBERED.match(line)
            if m is None or int(m.group(1)) != k or m.group(2).endswith("... [truncated]"):
                return None
            texts.append(m.group(2))
        content = "\n".join(texts)
        return content if content.count("\n") == total else None

    def write(
        self, ws: FileToolWorkspace, path: str, content: str, prior: Prior, args: Mapping[str, Any]
    ) -> str | None:
        if ws.at_least(0, 12) is not False or prior == "stale" or HERMES_PROTECTED.search(path):
            return None
        return json.dumps({"bytes_written": len(content.encode("utf-8")), "dirs_created": True}, ensure_ascii=False)

    def write_succeeded(self, ws: FileToolWorkspace, output: str) -> bool:
        result = json_object(output)
        return result is not None and "bytes_written" in result and "error" not in result

    def edit(
        self, ws: FileToolWorkspace, path: str, before: str, args: Mapping[str, Any], tool: FileTool
    ) -> tuple[str | None, str | None]:
        parsed = edit_args(args, tool)
        if args.get("mode", "replace") != "replace" or parsed is None:
            return None, None
        old, new, replace_all = parsed
        answer = ws.at_least(0, 12) is False and not HERMES_PROTECTED.search(path) and not path.startswith("~")
        if old == "" or old == new:
            error = "old_string cannot be empty" if old == "" else "old_string and new_string are identical"
            return (json.dumps({"success": False, "error": error}, ensure_ascii=False) if answer else None), None
        count = _overlapping_count(before, old)
        if count == 0:
            return None, None  # its fuzzy strategies may still find a match
        if count > 1 and not replace_all:
            error = (
                f"Found {count} matches for old_string. Provide more context to make it unique, or use "
                "replace_all=True."
            )
            return (json.dumps({"success": False, "error": error}, ensure_ascii=False) if answer else None), None
        if count > 1 and before.count(old) != count:
            return None, None  # overlapping matches
        after = before.replace(old, new)
        ext = os.path.splitext(ntpath.basename(path) if "\\" in path else path)[1]
        # A diff hunk that reaches the last line depends on whether its shell kept the file's final newline.
        last_match_line = before[: before.rfind(old) + len(old)].count("\n")
        near_end = last_match_line + 3 >= len(before.splitlines()) - 1
        if not answer or ext in HERMES_LINTED or near_end:
            return None, after
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True), f"a/{path}", f"b/{path}"
            )
        )
        lint = {"status": "skipped", "message": f"No linter for {ext} files"}
        return json.dumps(
            {"success": True, "diff": diff, "files_modified": [path], "lint": lint}, ensure_ascii=False
        ), after

    def edit_succeeded(self, ws: FileToolWorkspace, output: str) -> bool:
        result = json_object(output)
        return result is not None and result.get("success") is True and not result.get("no_change")

    def shell(self, ws: FileToolWorkspace, name: str, args: Mapping[str, Any], stdout: str) -> str | None:
        """terminal: `{"output", "exit_code": 0, "error": null}` with the output stripped. Output that starts or ends
        with whitespace other than line breaks (the exact stripping is not verified), escape sequences (which it
        removes), a background or PTY run, and sessions whose results showed command screening (a flagged command
        gets an `approval` note) go to the LLM."""
        if name != "terminal" or not plain_shell_args(args, "command", ("timeout", "workdir")):
            return None
        out = stdout.strip("\n")
        if ws.conventions.get("approval") or out != out.strip() or "\x1b" in out or "\r" in out:
            return None
        if len(out) > HERMES_MAX_SHELL_CHARS:
            return None
        return json.dumps({"output": out, "exit_code": 0, "error": None}, ensure_ascii=False)


def hermes_shell_failures(items: Items, tool_names: Collection[str] = ()) -> int:
    """terminal calls whose JSON result has a non-zero exit_code (124 is a timeout, -1 a refused command)."""
    n = 0
    for _, output in calls_and_outputs(items, ("terminal",)):
        result = json_object(output)
        code = result.get("exit_code") if result else None
        # JSON true and false load as bool, a subclass of int: not an exit code.
        if isinstance(code, int) and not isinstance(code, bool) and code != 0:
            n += 1
    return n


HERMES_FILES = FileToolSpec(
    format=HermesFormat(),
    file_tools={
        "read_file": FileTool("read", "path", offset="offset", limit="limit"),
        "write_file": FileTool("write", "path"),
        "patch": FileTool("edit", "path", replace_all="replace_all"),
    },
    shell_tools={"terminal": "command"},
    opaque_tools=frozenset({"delegate_task", "execute_code", "process"}),
    readonly_tools=frozenset(
        "search_files clarify todo memory session_search skill_view skills_list skill_manage vision_analyze web_search "
        "web_extract cronjob send_message".split()
    ),
)


HERMES = Harness(
    name="hermes",
    simulator_prompt="tool-simulator-hermes.md",
    judge_prompt="judge-system-harness.md",
    judge_environment="judge-environment-hermes.md",
    stop_tools=("clarify",),
    # Its tools answer in JSON, with the odd plain-text message.
    reply_format="json_text",
    files=HERMES_FILES,
    shell_failures=hermes_shell_failures,
    simulator_sees_tools=True,
)
