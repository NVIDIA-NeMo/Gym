# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The opencode and Kilo Code harnesses (Kilo Code's CLI is an opencode fork): their shared file-tool results and
shell output for a read-back of a known file, answered in code where the workspace knows the result, and how Kilo
Code's shell reports a failure."""

from __future__ import annotations

import posixpath
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


# Extensions no language server checks: a write or edit of one cannot come back with diagnostics.
OPENCODE_PLAIN = re.compile(
    r"\.(?:md|markdown|txt|text|csv|tsv|log|rst|ini|cfg|conf|env|gitignore|dockerignore|xml|svg)$", re.I
)
OPENCODE_DEFAULT_LIMIT = 2000
OPENCODE_MAX_LINE = 2000
OPENCODE_MAX_BYTES = 45 * 1024  # under its 50 KB read cap
OPENCODE_REFUSED = ("The user has specified a rule", "PermissionDeniedError", "The user rejected permission")
# bash output past its line or byte cap is cut with a note; a read-back answered in code stays well under both.
OPENCODE_MAX_SHELL_LINES = 1_000


def _posix_absolute(path: str) -> bool:
    return path.startswith("/") and posixpath.normpath(path) == path


class OpencodeFormat(FileFormat):
    """read: `<path>` (the absolute path), `<type>file</type>` and `<content>` holding `N: line` lines, then
    `(End of file - total N lines)` or a `(Showing lines A-B of N. Use offset=B+1 to continue.)` page notice.
    write: `Wrote file successfully.`; edit: `Edit applied successfully.`, to either of which a language
    server may add diagnostics, so only files no language server checks are answered here. Its edit falls
    back to fuzzy matching, so only a unique exact match (or replaceAll) is. Windows paths are echoed in a
    normalised form the data leaves open: those reads go to the LLM."""

    # Its read result never shows the file's final line break.
    read_shows_final_line_break = False

    def observe(self, ws: FileToolWorkspace, name: str, args: Mapping[str, Any], output: str) -> None:
        if output.startswith(OPENCODE_REFUSED):
            ws.conventions["permission_rules"] = True

    def read(self, ws: FileToolWorkspace, path: str, content: str, args: Mapping[str, Any]) -> str | None:
        ok_offset, offset = int_arg(args, "offset")
        ok_limit, limit = int_arg(args, "limit")
        if not (ok_offset and ok_limit) or not _posix_absolute(path) or content == "" or "\r" in content:
            return None
        offset = 1 if offset is None else offset
        limit = OPENCODE_DEFAULT_LIMIT if limit is None else limit
        lines = content.split("\n")
        if content.endswith("\n"):
            lines = lines[:-1]
        if offset < 1 or limit < 1 or offset > len(lines):
            return None
        shown = lines[offset - 1 : offset - 1 + limit]
        if any(len(line) > OPENCODE_MAX_LINE for line in shown):
            return None  # its long-line marker
        body = "\n".join(f"{offset + i}: {line}" for i, line in enumerate(shown))
        if len(body.encode("utf-8")) > OPENCODE_MAX_BYTES:
            return None
        end = offset - 1 + len(shown)
        if end < len(lines):
            notice = f"(Showing lines {offset}-{end} of {len(lines)}. Use offset={end + 1} to continue.)"
        else:
            notice = f"(End of file - total {len(lines)} lines)"
        return f"<path>{path}</path>\n<type>file</type>\n<content>\n{body}\n\n{notice}\n</content>"

    def read_back(self, ws: FileToolWorkspace, args: Mapping[str, Any], output: str) -> str | None:
        ok_offset, offset = int_arg(args, "offset")
        if not ok_offset or offset not in (None, 1):
            return None
        m = re.match(
            r"<path>[^\n]*</path>\n<type>file</type>\n<content>\n(.*)\n\n\(End of file - total (\d+) lines\)\n</content>\Z",
            output,
            re.S,
        )
        if m is None:
            return None
        texts = []
        for k, line in enumerate(m.group(1).split("\n"), 1):
            prefix = f"{k}: "
            if not line.startswith(prefix):
                return None
            texts.append(line[len(prefix) :])
        return "\n".join(texts) if len(texts) == int(m.group(2)) else None  # the final line break is not shown

    def write(
        self, ws: FileToolWorkspace, path: str, content: str, prior: Prior, args: Mapping[str, Any]
    ) -> str | None:
        if not OPENCODE_PLAIN.search(path) or ws.conventions.get("permission_rules"):
            return None
        return "Wrote file successfully."

    def write_succeeded(self, ws: FileToolWorkspace, output: str) -> bool:
        return output.startswith("Wrote file successfully.")

    def edit(
        self, ws: FileToolWorkspace, path: str, before: str, args: Mapping[str, Any], tool: FileTool
    ) -> tuple[str | None, str | None]:
        parsed = edit_args(args, tool)
        if parsed is None or "\r" in before:
            return None, None
        old, new, replace_all = parsed
        answer = OPENCODE_PLAIN.search(path) is not None and not ws.conventions.get("permission_rules")
        if old == new:
            return ("No changes to apply: oldString and newString are identical." if answer else None), None
        count = before.count(old) if old else 0
        if count == 0 or (count > 1 and not replace_all):
            return None, None  # its fuzzy matchers decide
        after = before.replace(old, new) if replace_all else before.replace(old, new, 1)
        return ("Edit applied successfully." if answer else None), after

    def edit_succeeded(self, ws: FileToolWorkspace, output: str) -> bool:
        return output.startswith("Edit applied successfully.")

    def shell(self, ws: FileToolWorkspace, name: str, args: Mapping[str, Any], stdout: str) -> str | None:
        """bash: the output as printed. No output (its note differs between versions), long output (cut, with a
        note) and Windows line breaks go to the LLM, and so do the legacy extension's shell tools."""
        if name != "bash" or not plain_shell_args(args, "command", ("description", "workdir", "timeout")):
            return None
        if not stdout or "\r" in stdout or stdout.count("\n") > OPENCODE_MAX_SHELL_LINES:
            return None
        return stdout if len(stdout.encode("utf-8")) <= OPENCODE_MAX_BYTES else None


OPENCODE_FILES = FileToolSpec(
    format=OpencodeFormat(),
    file_tools={
        "read": FileTool("read", "filePath", offset="offset", limit="limit"),
        "write": FileTool("write", "filePath"),
        "edit": FileTool("edit", "filePath", old="oldString", new="newString", replace_all="replaceAll"),
    },
    shell_tools={"bash": "command", "execute_command": "command", "interactive_terminal": "command"},
    opaque_tools=frozenset({"task", "apply_patch", "multiedit", "patch", "new_task", "background_process"}),
    readonly_tools=frozenset(
        "glob grep list ls webfetch websearch codesearch todowrite todoread question plan_exit plan_enter skill "
        "suggest kilo_local_recall read_file list_files search_files ask_followup_question attempt_completion "
        "update_todo_list fetch_instructions switch_mode".split()
    ),
)
# opencode's webfetch failure texts, for injected obstacles. Its websearch shows only a misconfigured backend's
# 403, which no retry or reworded query fixes. Kilo Code's failure texts are not verified, so it has none.
OPENCODE_OBSTACLE_TEXTS = {
    "webfetch": ObstacleTexts(
        target="url",
        failed=lambda output: output.startswith(("Transport error (", "StatusCode: non 2xx status code (")),
        transient=lambda url, shown: f"Transport error (GET {url})",
        blocked_route=lambda url, shown: f"StatusCode: non 2xx status code (403 GET {url})",
    ),
}
KILO_EXIT = re.compile(r"Exit code: (-?\d+)")


def kilo_code_shell_failures(items: Items, tool_names: Collection[str] = ()) -> int | None:
    """Kilo Code's legacy extension reports `Exit code: N` on execute_command results; its CLI's bash (like
    opencode's) shows no exit code, so rows without execute_command are not measured (None)."""
    if "execute_command" not in tool_names:
        return None
    n = 0
    for _, output in calls_and_outputs(items, ("execute_command",)):
        m = KILO_EXIT.search(output.split("\n", 1)[0])
        if m and m.group(1) != "0":
            n += 1
    return n


# Its bash results show no exit code, so its shell failures are not measured.
OPENCODE = Harness(
    name="opencode",
    simulator_prompt="tool-simulator-opencode.md",
    judge_prompt="judge-system-harness.md",
    judge_environment="judge-environment-opencode.md",
    stop_tools=("question", "plan_exit"),
    files=OPENCODE_FILES,
    simulator_sees_tools=True,
    obstacle_texts=OPENCODE_OBSTACLE_TEXTS,
)
# The CLI generation shares opencode's tools and formats; the legacy extension's tools (read_file, apply_diff,
# execute_command, ...) all go to the LLM, and its attempt_completion ends the task.
KILO_CODE = Harness(
    name="kilo_code",
    simulator_prompt="tool-simulator-kilo-code.md",
    judge_prompt="judge-system-harness.md",
    judge_environment="judge-environment-kilo-code.md",
    stop_tools=("question", "plan_exit", "ask_followup_question", "attempt_completion"),
    files=OPENCODE_FILES,
    shell_failures=kilo_code_shell_failures,
    simulator_sees_tools=True,
)
