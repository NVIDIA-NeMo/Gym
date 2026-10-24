# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The simulator's model of the agent's files, and file-tool calls answered in code from it.

`Workspace` is the interface the server uses. `FileToolWorkspace` maps a harness's file tools onto canonical reads,
writes and edits (`FileTool`, listed in a `FileToolSpec`) whose results the harness's `FileFormat` writes and reads.
`ReadBack` is a shell command that only prints a known file (or part of it), which a workspace can answer too.
"""

from __future__ import annotations

import json
import ntpath
import posixpath
import re
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

from resources_servers.agentic_turn_simulation.items import Items, parse_args


# Commands that plausibly change a file they name: its known content is dropped. In-place editors (`sed` or `perl`
# with `i` in an option cluster or `--in-place`, gawk's `-i inplace`), downloads and output options (`curl -o`/`-O`,
# `wget`, `sort -o`), line-ending converters, redirects, file operations, archives and inline scripts.
MUTATING = re.compile(
    r"\b(?:sed|perl)\b[^|;&\n]*\s(?:-\w*i|--in-place\b)|\bg?awk\b[^|;&\n]*\binplace\b|"
    r"\b(?:curl|sort)\b[^|;&\n]*\s(?:-[A-Za-z]*[oO]|--output\b|--remote-name)|\bwget\b|\b(?:dos2unix|unix2dos)\b|"
    r">\s*\S|\btee\b|\bmv\b|\brm\b|\bcp\b|\bpatch\b|\bgit\s+"
    r"(checkout|reset|restore|apply|stash|pull|merge|clone)|\btruncate\b|\bunzip\b|"
    r"\btar\b|python3?\s+-c|\bnode\s+-e"
)
# Windows shells' own mutating commands, beside MUTATING's POSIX ones.
WINDOWS_MUTATING = re.compile(
    r"(?i)\b(?:set-content|add-content|out-file|remove-item|move-item|copy-item|rename-item|new-item|del|erase|"
    r"ren|move|copy|xcopy|robocopy)\b"
)
# Commands that plausibly change files they do not name: after one, no known content is trusted. Formatters and
# fixers, also as a toolchain subcommand or a package script (`go fmt`, `cargo fix`, `npm run format`); package
# managers that rewrite the manifest or lockfile (`npm install x` with a package to add, not a bare `npm install`;
# `yarn add x`, `npm version patch`, `go get`); VCS operations.
BROAD_MUTATION = re.compile(
    r"\b(?:prettier|black|isort|autopep8|yapf|gofmt|rustfmt|clang-format|dotnet\s+format|ruff\s+format)\b|"
    r"--fix\b|--write\b|\b(?:go|cargo|deno|dart|terraform|zig|mix)\s+(?:fmt|format|fix)\b|"
    r"\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?[\w:-]*(?:fmt|format|fix)\b|"
    r"\b(?:npm|pnpm|yarn|bun)[ \t]+(?:install|i|add)(?:[ \t]+-\S*)*[ \t]+(?!\d*[<>])[\w@./~]|"
    r"\b(?:npm|pnpm|yarn|bun)\s+(?:remove|rm|uninstall|un|update|up|upgrade|version|pkg\s+(?:set|delete|fix))\b|"
    r"\b(?:cargo|poetry|uv)\s+(?:add|remove|update|lock)\b|\bgo\s+(?:get|mod\s+tidy)\b|"
    r"\bgit\s+(?:checkout|reset|restore|stash|pull|merge|rebase|switch|apply|am|cherry-pick|revert|clone)\b"
)
# The known files a call names that the simulator is shown: at most this many, each cut to this many characters.
SHOWN_FILES = 3
SHOWN_FILE_CHARS = 30_000

# Shell syntax a read-back never has: pipes, lists, redirection, substitution, globs, escapes, comments.
_SHELL_SYNTAX = re.compile(r"[|&;<>()$`*?\[\]{}!#\\\n\r]")
# A plain path: no whitespace, quotes or shell syntax, and not an option.
_PLAIN_PATH = re.compile(r"(?!-)[\w./+@,=:%~-]+")
_SED_LINES = re.compile(r"([1-9]\d*)(?:,([1-9]\d*))?p")
# A file's lines, each with its line break (the last one may have none).
_LINES = re.compile(r"[^\n]*\n|[^\n]+\Z")


@dataclass(frozen=True)
class ReadBack:
    """A shell command that only prints one file, or part of it: `cat P`, `head [-n N | -N] P`,
    `tail [-n N | -n +N | -N] P`, `wc -l P`, `wc -c P` or `sed -n 'A[,B]p' P`, with one plain path and no other
    shell syntax (`parse`). `stdout` is what GNU coreutils prints for a file's content."""

    program: Literal["cat", "head", "tail", "sed", "wc"]
    # The path as the command names it (relative, absolute or under ~). wc prints it as it stands, so a caller that
    # knows the home directory expands a leading ~ first, as the shell does.
    path: str
    # The lines printed, as a slice of the file's lines (cat, head, tail, sed).
    lines: slice = slice(None)
    # wc: "l" (lines) or "c" (bytes).
    count: Literal["l", "c"] | None = None

    @classmethod
    def parse(cls, command: Any) -> ReadBack | None:
        if not isinstance(command, str) or _SHELL_SYNTAX.search(command):
            return None
        try:
            words = shlex.split(command)
        except ValueError:
            return None
        if len(words) < 2 or not _PLAIN_PATH.fullmatch(words[-1]):
            return None  # no file: it would read its standard input
        program, options, path = words[0], words[1:-1], words[-1]
        if program == "cat" and not options:
            return cls("cat", path)
        if program in ("head", "tail"):
            n = _line_count(options)
            if n is None:
                return None
            start, n = (True, n[1:]) if program == "tail" and n.startswith("+") else (False, n)
            if not re.fullmatch(r"[0-9]+", n):
                return None
            k = int(n)
            if program == "head":
                return cls("head", path, slice(0, k))
            return cls("tail", path, slice(max(k, 1) - 1, None) if start else slice(-k, None) if k else slice(0, 0))
        if program == "sed" and len(options) == 2 and options[0] == "-n":
            m = _SED_LINES.fullmatch(options[1])
            if m is None:
                return None
            first = int(m.group(1))
            last = max(first, int(m.group(2) or first))  # GNU sed prints only the first line when B < A
            return cls("sed", path, slice(first - 1, last))
        if program == "wc" and options in (["-l"], ["-c"]):
            return cls("wc", path, count=options[0][1])
        return None

    def reaches_end(self, content: str) -> bool:
        """Whether the output depends on how the file ends: it counts, or prints the last line (or the file is
        empty)."""
        lines = _LINES.findall(content)
        return self.count is not None or not lines or len(lines) - 1 in range(len(lines))[self.lines]

    def stdout(self, content: str) -> str:
        """What the command prints for a file with this content."""
        if self.count == "l":
            return f"{content.count(chr(10))} {self.path}\n"
        if self.count == "c":
            return f"{len(content.encode('utf-8'))} {self.path}\n"
        return "".join(_LINES.findall(content)[self.lines])


def _line_count(options: list[str]) -> str | None:
    """head / tail's line count from its options (`-n N`, `-nN` or `-N`; 10 without any), as written (`+N` too)."""
    if not options:
        return "10"
    if len(options) == 2 and options[0] == "-n":
        return options[1]
    if len(options) == 1 and re.fullmatch(r"-n\+?\d+|-\d+", options[0]):
        return options[0].removeprefix("-").removeprefix("n")
    return None


def render_files(files: Iterable[tuple[str, str]], limit_chars: int) -> str:
    """Known file contents for the simulator: a <file> block per (path, content)."""
    blocks = []
    for path, content in files:
        cut = "" if len(content) <= limit_chars else f"\n[... {len(content) - limit_chars} more characters ...]"
        blocks.append(f'<file path="{path}">\n{content[:limit_chars]}{cut}\n</file>')
    return "\n\n".join(blocks)


class Workspace(ABC):
    """What the simulator knows about the agent's files: path -> content, or None when the file exists (or did)
    but its content is unknown. It learns from the recorded history, then from every answer."""

    def __init__(self) -> None:
        self.files: dict[str, str | None] = {}

    def replay(self, items: Items) -> None:
        """Learn from the history: each call with the result it actually returned."""
        calls: dict[Any, dict[str, Any]] = {}
        for item in items:
            kind = item.get("type", "message")
            if kind == "message" and item.get("role") == "user":
                self.new_turn()
            elif kind == "function_call":
                calls[item.get("call_id")] = item
            elif kind == "function_call_output" and item.get("call_id") in calls:
                self.observe(calls[item["call_id"]], item.get("output"))

    def new_turn(self) -> None:
        """A user message starts a new turn."""

    @abstractmethod
    def observe(self, call: dict[str, Any], output: Any) -> None:
        """Learn from one call and the result it returned (recorded, or simulated by the LLM)."""

    @abstractmethod
    def answer(self, call: dict[str, Any], tool_names: frozenset[str]) -> str | None:
        """The call's result when it follows from what is known (the workspace is updated as the tool would change
        the files), else None for the simulator LLM."""

    @abstractmethod
    def shown_files(self, call: dict[str, Any]) -> str:
        """Known contents of the files this call names, for the simulator's context ("" when none): at most
        SHOWN_FILES files, each cut to SHOWN_FILE_CHARS characters."""


# Paths are taken as the harness gives them: POSIX paths as they are, Windows paths (drive letter or UNC)
# case-insensitively with either slash. Nothing is resolved against a home or working directory the trace does
# not state, so one file named two ways is simply not matched (the LLM answers).
WINDOWS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\[^\\]+\\)")


def path_key(path: Any) -> str | None:
    """How the workspace keys a path: Windows paths case-folded with backslashes, others as given (normalised)."""
    if not isinstance(path, str) or not path.strip():
        return None
    p = path.strip()
    if WINDOWS_PATH.match(p):
        return ntpath.normcase(ntpath.normpath(p))
    return posixpath.normpath(p)


def mentions(key: str, text: str, *, short_names: bool = False) -> bool:
    """Does the text name this file (its path, or its basename as a word)? A basename of three characters or fewer
    (`a.c`) counts only with `short_names`: too common a word to pick the files the simulator is shown, but what
    forgetting needs, since a command that may have changed a file had better forget too much than too little."""
    windows = WINDOWS_PATH.match(key) is not None
    if windows:
        text = text.lower().replace("/", "\\")
    if key in text:
        return True
    base = ntpath.basename(key) if windows else posixpath.basename(key)
    if not base or (len(base) <= 3 and not short_names):
        return False
    return re.search(r"(?<![\w.-])" + re.escape(base) + r"(?![\w-])", text) is not None


def plain_shell_args(args: Mapping[str, Any], command: str, allowed: Iterable[str]) -> bool:
    """Whether a shell call sets nothing but its command and `allowed` arguments (any other one, such as a background
    flag, is unset or false)."""
    keep = {command, *allowed}
    return not any(value for key, value in args.items() if key not in keep)


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _string_values(v)]
    if isinstance(value, list):
        return [s for v in value for s in _string_values(v)]
    return []


@dataclass(frozen=True)
class FileTool:
    """One harness tool as a canonical file operation, by its argument names."""

    op: Literal["read", "write", "edit"]
    path: str
    content: str = "content"  # write
    old: str = "old_string"  # edit
    new: str = "new_string"  # edit
    replace_all: str | None = None  # edit: the flag argument, if the tool has one
    offset: str | None = None  # read
    limit: str | None = None  # read


# What the workspace knows about a file before a write: its content ("known"), nothing ("unseen"), or that it was
# there but may have changed since ("stale").
Prior = Literal["known", "unseen", "stale"]


class FileFormat:
    """A harness's file-tool results. Each method gets the session's workspace (the row's harness version and the
    conventions earlier results showed) and returns the tool's output, or None when the result is not determined
    by the arguments and the known content (the LLM answers). The defaults decline everything."""

    # A repeated read of an unchanged range may come back as a "you already read this" stub (the LLM decides).
    dedupes_reads: bool = False
    # A whole-file read result shows whether the file ends with a line break. When not, content learned from a read
    # may lack the one the file has, so a shell read-back that prints the file's end goes to the LLM.
    read_shows_final_line_break: bool = True

    def observe(self, ws: FileToolWorkspace, name: str, args: Mapping[str, Any], output: str) -> None:
        """Learn this harness's conventions from a result (e.g. which version's wording it uses)."""

    def read(self, ws: FileToolWorkspace, path: str, content: str, args: Mapping[str, Any]) -> str | None:
        return None

    def read_back(self, ws: FileToolWorkspace, args: Mapping[str, Any], output: str) -> str | None:
        """The file's whole content from a read result, or None (partial, failed or not parseable)."""
        return None

    def write(
        self, ws: FileToolWorkspace, path: str, content: str, prior: Prior, args: Mapping[str, Any]
    ) -> str | None:
        return None

    def write_succeeded(self, ws: FileToolWorkspace, output: str) -> bool:
        return False

    def edit(
        self, ws: FileToolWorkspace, path: str, before: str, args: Mapping[str, Any], tool: FileTool
    ) -> tuple[str | None, str | None]:
        """(the output or None, the content after the edit or None when it fails or is not determined) for an
        edit of a file whose content is known.

        Also called to replay a recorded edit that succeeded; then only the content after it is used, so return it
        whenever it is determined, even when the output is not."""
        return None, None

    def edit_succeeded(self, ws: FileToolWorkspace, output: str) -> bool:
        return False

    def content_after(self, ws: FileToolWorkspace, args: Mapping[str, Any], output: str) -> str | None:
        """The whole file as a write or edit result shows it, if it does."""
        return None

    def shell(self, ws: FileToolWorkspace, name: str, args: Mapping[str, Any], stdout: str) -> str | None:
        """The result of shell tool `name` for a command that exits 0 and prints `stdout` (a read-back of a known
        file), or None when this tool, these arguments or this output are not answered in code."""
        return None


@dataclass(frozen=True, eq=False)
class FileToolSpec:
    """Which of a harness's tools touch files, and how."""

    format: FileFormat
    file_tools: Mapping[str, FileTool]
    # Shell tools by the argument that holds the command: a mutating command forgets the files it names.
    shell_tools: Mapping[str, str] = field(default_factory=dict)
    # Tools that hand work to another agent or run code that may change any file: everything is forgotten.
    opaque_tools: frozenset[str] = frozenset()
    # Tools that never change files. Any other tool forgets the files its arguments name.
    readonly_tools: frozenset[str] = frozenset()


def int_arg(args: Mapping[str, Any], key: str | None) -> tuple[bool, int | None]:
    """(valid, value): a missing or null argument is (True, None); a non-integer one is invalid."""
    value = args.get(key) if key else None
    if value is None:
        return True, None
    if isinstance(value, bool) or not isinstance(value, int):
        return False, None
    return True, value


def edit_args(args: Mapping[str, Any], tool: FileTool) -> tuple[str, str, bool] | None:
    """An edit call's (old, new, replace_all), or None when one of them has the wrong type."""
    old, new = args.get(tool.old), args.get(tool.new)
    replace_all = args.get(tool.replace_all, False) if tool.replace_all else False
    if not isinstance(old, str) or not isinstance(new, str) or not isinstance(replace_all, bool):
        return None
    return old, new, replace_all


def parse_version(value: Any) -> tuple[int, ...] | None:
    """The version's numbers: "2.1.220" -> (2, 1, 220), "2026.7.2-beta.7" -> (2026, 7, 2, 7); None without any."""
    numbers = re.findall(r"\d+", value) if isinstance(value, str) else []
    return tuple(int(n) for n in numbers) or None


class FileToolWorkspace(Workspace):
    """The workspace of a harness with file tools. It learns contents from writes and edits that succeeded and
    from complete reads, forgets what a shell command or another tool may have changed, and answers a call in code
    only when the result follows from what it knows: a read or edit of a file whose whole content is known, or a
    write whose outcome does not depend on the unknown."""

    def __init__(self, spec: FileToolSpec, version: str | None = None) -> None:
        super().__init__()
        self.spec = spec
        self.version = parse_version(version)
        # What earlier results showed of the harness's wording, by the format's own keys.
        self.conventions: dict[str, Any] = {}
        # The path as the agent last wrote it, per key (what the simulator is shown).
        self.spelled: dict[str, str] = {}
        # Per key, the (offset, limit) ranges read since the file last changed.
        self.read_ranges: dict[str, set[str]] = {}
        # Keys whose known content may lack the file's final line break (read_shows_final_line_break).
        self.end_unknown: set[str] = set()

    def at_least(self, *version: int) -> bool | None:
        """Is the row's harness version at least this one? None when the row does not say."""
        return None if self.version is None else self.version >= version

    def remember(self, path: str, content: str | None, *, end_known: bool = True) -> None:
        key = path_key(path)
        if key is not None:
            self.files[key] = content
            self.spelled[key] = path.strip()
            self.read_ranges.pop(key, None)
            if end_known:
                self.end_unknown.discard(key)
            else:
                self.end_unknown.add(key)

    def lookup(self, path: Any) -> tuple[str | None, str | None]:
        """(the path's key, the file's content when known). A known content also means the file exists; an
        unknown one says nothing (a command that touched it may have deleted it)."""
        key = path_key(path)
        content = self.files.get(key) if key is not None else None
        return key, content if isinstance(content, str) else None

    def prior(self, key: str) -> Prior:
        if isinstance(self.files.get(key), str):
            return "known"
        return "stale" if key in self.files else "unseen"

    def forget(self, keys: Sequence[str] | None = None) -> None:
        for key in list(self.files) if keys is None else keys:
            if isinstance(self.files.get(key), str):
                self.files[key] = None
            self.read_ranges.pop(key, None)

    def observe(self, call: dict[str, Any], output: Any) -> None:
        name, args = call.get("name"), parse_args(call)
        if args is None or not isinstance(name, str):
            return
        text = output if isinstance(output, str) else ""
        self.spec.format.observe(self, name, args, text)
        tool = self.spec.file_tools.get(name)
        if tool is not None:
            self._observe_file_tool(tool, args, text)
        elif name in self.spec.shell_tools:
            self.forget_mutated(args.get(self.spec.shell_tools[name]))
        elif name in self.spec.opaque_tools:
            self.forget()
        elif name not in self.spec.readonly_tools:
            mentioned = " ".join(_string_values(args))
            self.forget([k for k in self.files if mentions(k, mentioned, short_names=True)])

    def _observe_file_tool(self, tool: FileTool, args: dict[str, Any], output: str) -> None:
        fmt, path = self.spec.format, args.get(tool.path)
        key = path_key(path)
        if key is None:
            return
        if tool.op == "read":
            content = fmt.read_back(self, args, output)
            if content is not None:
                self.remember(path, content, end_known=fmt.read_shows_final_line_break)
            self.read_ranges.setdefault(key, set()).add(self._read_span(tool, args))
            return
        shown = fmt.content_after(self, args, output)
        if shown is not None:
            self.remember(path, shown)
        elif tool.op == "write":
            if fmt.write_succeeded(self, output) and isinstance(args.get(tool.content), str):
                self.remember(path, args[tool.content])
        elif fmt.edit_succeeded(self, output):
            before = self.files.get(key)
            after = fmt.edit(self, path, before, args, tool)[1] if isinstance(before, str) else None
            self.remember(path, after, end_known=key not in self.end_unknown)

    def forget_mutated(self, cmd: Any) -> None:
        """Forget what a shell command may have changed: everything after a broad mutation, else the files a mutating
        command names."""
        cmd = cmd if isinstance(cmd, str) else ""
        if BROAD_MUTATION.search(cmd):
            self.forget()
        elif MUTATING.search(cmd) or WINDOWS_MUTATING.search(cmd):
            self.forget([k for k in self.files if mentions(k, cmd, short_names=True)])

    def shown_files(self, call: dict[str, Any]) -> str:
        text = " ".join(_string_values(parse_args(call) or {}))
        hits = [k for k, c in self.files.items() if isinstance(c, str) and mentions(k, text)]
        return render_files(((self.spelled.get(k, k), self.files[k]) for k in hits[:SHOWN_FILES]), SHOWN_FILE_CHARS)

    def answer(self, call: dict[str, Any], tool_names: frozenset[str]) -> str | None:
        name = call.get("name")
        tool = self.spec.file_tools.get(name) if isinstance(name, str) else None
        args = parse_args(call)
        if name not in tool_names or args is None:
            return None
        if name in self.spec.shell_tools:
            return self._read_back(name, args)
        if tool is None:
            return None
        path = args.get(tool.path)
        key, content = self.lookup(path)
        if key is None:
            return None
        fmt, path = self.spec.format, path.strip()
        if tool.op == "read":
            span = self._read_span(tool, args)
            if content is None or (fmt.dedupes_reads and span in self.read_ranges.get(key, ())):
                return None
            out = fmt.read(self, path, content, args)
            if out is not None:
                self.read_ranges.setdefault(key, set()).add(span)
            return out
        if tool.op == "write":
            new = args.get(tool.content)
            if not isinstance(new, str):
                return None
            out = fmt.write(self, path, new, self.prior(key), args)
            if out is not None:
                self.remember(path, new)
            return out
        if content is None:
            return None
        out, after = fmt.edit(self, path, content, args, tool)
        if out is not None and after is not None:
            self.remember(path, after, end_known=key not in self.end_unknown)
        return out

    def _read_back(self, name: str, args: Mapping[str, Any]) -> str | None:
        """A shell read-back of a known file, by absolute POSIX path only: the shell's working directory and home are
        unknown. wc (GNU and BSD pad differently), head of 0 lines (BSD head rejects the count) and a sed range
        ending on a last line without a line break (BSD sed may add one) go to the LLM: the machine's OS is unknown
        too."""
        read_back = ReadBack.parse(args.get(self.spec.shell_tools[name]))
        if read_back is None or read_back.program == "wc" or not read_back.path.startswith("/"):
            return None
        if read_back.program == "head" and read_back.lines.stop == 0:
            return None
        key, content = self.lookup(read_back.path)
        if content is None or (key in self.end_unknown and read_back.reaches_end(content)):
            return None
        stdout = read_back.stdout(content)
        if read_back.program == "sed" and stdout and not stdout.endswith("\n"):
            return None
        return self.spec.format.shell(self, name, args, stdout)

    def _read_span(self, tool: FileTool, args: Mapping[str, Any]) -> str:
        """The (offset, limit) range a read asks for, as a set member."""
        span = (args.get(tool.offset) if tool.offset else None, args.get(tool.limit) if tool.limit else None)
        return json.dumps(span, sort_keys=True, default=str)
