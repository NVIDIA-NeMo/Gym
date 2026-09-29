# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Python port of Codex's ``apply_patch`` (codex-rs/apply-patch, default ``NormalizeToLf`` mode).

Derived from openai/codex (Apache-2.0, Copyright OpenAI); modified by NVIDIA as described below.

Parsing, matching, error texts, and the success summary follow upstream. As in the Codex tool
handler, every hunk is verified (files read, context located) before anything is written, so
content mismatches never leave a partial edit. Deviation: with a workspace root, paths that
resolve outside it are rejected, because this runs without Codex's sandbox.

Standard library only: the file doubles as the ``apply_patch`` executable in exec sessions.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional, Union


BEGIN_PATCH_MARKER = "*** Begin Patch"
END_PATCH_MARKER = "*** End Patch"
ADD_FILE_MARKER = "*** Add File: "
DELETE_FILE_MARKER = "*** Delete File: "
UPDATE_FILE_MARKER = "*** Update File: "
MOVE_TO_MARKER = "*** Move to: "
EOF_MARKER = "*** End of File"
CHANGE_CONTEXT_MARKER = "@@ "
EMPTY_CHANGE_CONTEXT_MARKER = "@@"
ENVIRONMENT_ID_MARKER = "*** Environment ID:"

# Rust's char::is_whitespace (Unicode White_Space); str.strip() also strips \x1c-\x1f.
_WHITESPACE = (
    "\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)
_NOT_HUNK_HEADER = (
    "'{}' is not a valid hunk header. Valid hunk headers: "
    "'*** Add File: {{path}}', '*** Delete File: {{path}}', '*** Update File: {{path}}'"
)
_UNEXPECTED_UPDATE_LINE = (
    "Unexpected line found in update hunk: '{}'. Every line should start with "
    "' ' (context line), '+' (added line), or '-' (removed line)"
)
_EXPECTED_CONTEXT_MARKER = "Expected update hunk to start with a @@ context marker, got: '{}'"


def _trim(text: str) -> str:
    return text.strip(_WHITESPACE)


def _trim_end(text: str) -> str:
    return text.rstrip(_WHITESPACE)


class ApplyPatchError(Exception):
    """An apply_patch failure; ``str()`` is the upstream error text."""


class InvalidPatchError(ApplyPatchError):
    def __init__(self, message: str) -> None:
        super().__init__(f"invalid patch: {message}")
        self.message = message


class InvalidHunkError(ApplyPatchError):
    def __init__(self, message: str, line_number: int) -> None:
        super().__init__(f"invalid hunk at line {line_number}, {message}")
        self.message = message
        self.line_number = line_number


@dataclass
class UpdateFileChunk:
    change_context: Optional[str] = None
    old_lines: list[str] = field(default_factory=list)
    new_lines: list[str] = field(default_factory=list)
    is_end_of_file: bool = False

    def is_empty(self) -> bool:
        return not self.old_lines and not self.new_lines


@dataclass
class AddFile:
    path: str
    contents: str = ""


@dataclass
class DeleteFile:
    path: str


@dataclass
class UpdateFile:
    path: str
    move_path: Optional[str] = None
    chunks: list[UpdateFileChunk] = field(default_factory=list)


Hunk = Union[AddFile, DeleteFile, UpdateFile]


@dataclass
class ParsedPatch:
    hunks: list[Hunk]
    environment_id: Optional[str] = None


class _Parser:
    """Line-at-a-time port of upstream ``StreamingPatchParser``."""

    def __init__(self) -> None:
        self.mode = "not_started"
        self.hunk_line_number = 0
        self.line_number = 0
        self.hunks: list[Hunk] = []
        self.environment_id: Optional[str] = None

    def _ensure_update_hunk_is_not_empty(self, line: str) -> None:
        hunk = self.hunks[-1] if self.hunks else None
        if not isinstance(hunk, UpdateFile):
            return
        if not hunk.chunks and self.mode == "update_file":
            raise InvalidHunkError(f"Update file hunk for path '{hunk.path}' is empty", self.hunk_line_number)
        if hunk.chunks and hunk.chunks[-1].is_empty():
            if line == END_PATCH_MARKER:
                raise InvalidHunkError("Update hunk does not contain any lines", self.line_number)
            raise InvalidHunkError(_UNEXPECTED_UPDATE_LINE.format(line), self.line_number)

    def _handle_headers_and_end_patch(self, trimmed: str) -> bool:
        if self.mode == "started_patch" and trimmed.startswith(ENVIRONMENT_ID_MARKER):
            if self.environment_id is not None:
                raise InvalidPatchError("apply_patch environment_id cannot be specified more than once")
            environment_id = _trim(trimmed[len(ENVIRONMENT_ID_MARKER) :])
            if not environment_id:
                raise InvalidPatchError("apply_patch environment_id cannot be empty")
            self.environment_id = environment_id
            return True
        if trimmed == END_PATCH_MARKER:
            self._ensure_update_hunk_is_not_empty(trimmed)
            self.mode = "ended_patch"
            return True
        for marker, mode in (
            (ADD_FILE_MARKER, "add_file"),
            (DELETE_FILE_MARKER, "delete_file"),
            (UPDATE_FILE_MARKER, "update_file"),
        ):
            if trimmed.startswith(marker):
                self._ensure_update_hunk_is_not_empty(trimmed)
                path = trimmed[len(marker) :]
                self.hunks.append(
                    {"add_file": AddFile, "delete_file": DeleteFile, "update_file": UpdateFile}[mode](path)
                )
                self.mode = mode
                if mode == "update_file":
                    self.hunk_line_number = self.line_number
                return True
        return False

    def process_line(self, line: str) -> None:
        trimmed = _trim(line)
        if self.mode == "not_started":
            if trimmed == BEGIN_PATCH_MARKER:
                self.mode = "started_patch"
                return
            raise InvalidPatchError("The first line of the patch must be '*** Begin Patch'")
        if self.mode in ("started_patch", "delete_file"):
            if self._handle_headers_and_end_patch(trimmed):
                return
            raise InvalidHunkError(_NOT_HUNK_HEADER.format(trimmed), self.line_number)
        if self.mode == "add_file":
            if self._handle_headers_and_end_patch(trimmed):
                return
            if line.startswith("+"):
                self.hunks[-1].contents += line[1:] + "\n"
                return
            raise InvalidHunkError(_NOT_HUNK_HEADER.format(trimmed), self.line_number)
        if self.mode == "update_file":
            self._process_update_line(line)
            return
        # ended_patch
        if trimmed:
            raise InvalidPatchError("The last line of the patch must be '*** End Patch'")

    def _process_update_line(self, line: str) -> None:
        update_line = _trim_end(line)
        if self._handle_headers_and_end_patch(update_line):
            return
        hunk = self.hunks[-1]
        chunks = hunk.chunks
        last = chunks[-1] if chunks else None
        is_context_marker = update_line == EMPTY_CHANGE_CONTEXT_MARKER or update_line.startswith(CHANGE_CONTEXT_MARKER)
        if last is not None and last.is_end_of_file:
            if not update_line:
                return
            if not is_context_marker:
                raise InvalidHunkError(_EXPECTED_CONTEXT_MARKER.format(line), self.line_number)
        if not chunks and hunk.move_path is None and update_line.startswith(MOVE_TO_MARKER):
            hunk.move_path = update_line[len(MOVE_TO_MARKER) :]
            return
        if is_context_marker and last is not None and last.is_empty():
            raise InvalidHunkError(_UNEXPECTED_UPDATE_LINE.format(line), self.line_number)
        if update_line == EMPTY_CHANGE_CONTEXT_MARKER:
            chunks.append(UpdateFileChunk())
            return
        if update_line.startswith(CHANGE_CONTEXT_MARKER):
            chunks.append(UpdateFileChunk(change_context=update_line[len(CHANGE_CONTEXT_MARKER) :]))
            return
        if update_line == EOF_MARKER:
            if last is not None and last.is_empty():
                raise InvalidHunkError("Update hunk does not contain any lines", self.line_number)
            if last is not None:
                last.is_end_of_file = True
            return
        if line == "" or line[0] in " +-":
            if not chunks:
                chunks.append(UpdateFileChunk())
            chunk = chunks[-1]
            if line == "":
                chunk.old_lines.append("")
                chunk.new_lines.append("")
            elif line[0] == " ":
                chunk.old_lines.append(line[1:])
                chunk.new_lines.append(line[1:])
            elif line[0] == "+":
                chunk.new_lines.append(line[1:])
            else:
                chunk.old_lines.append(line[1:])
            return
        if last is not None and not last.is_empty():
            raise InvalidHunkError(_EXPECTED_CONTEXT_MARKER.format(line), self.line_number)
        raise InvalidHunkError(_UNEXPECTED_UPDATE_LINE.format(line), self.line_number)

    def finish(self, last_line: str) -> None:
        if last_line:
            self.line_number += 1
            if _trim(last_line) == END_PATCH_MARKER:
                self._ensure_update_hunk_is_not_empty(_trim(last_line))
                self.mode = "ended_patch"
            else:
                self.process_line(last_line)
        if self.mode != "ended_patch":
            raise InvalidPatchError("The last line of the patch must be '*** End Patch'")


def _check_boundaries_strict(lines: list[str]) -> list[str]:
    first = _trim(lines[0]) if lines else None
    last = _trim(lines[-1]) if lines else None
    if first == BEGIN_PATCH_MARKER and last == END_PATCH_MARKER:
        return lines
    if first is not None and first != BEGIN_PATCH_MARKER:
        raise InvalidPatchError("The first line of the patch must be '*** Begin Patch'")
    raise InvalidPatchError("The last line of the patch must be '*** End Patch'")


def _check_boundaries_lenient(lines: list[str]) -> list[str]:
    try:
        return _check_boundaries_strict(lines)
    except InvalidPatchError:
        # Models sometimes send a shell heredoc body verbatim: <<'EOF' ... EOF.
        if len(lines) >= 4 and lines[0] in ("<<EOF", "<<'EOF'", '<<"EOF"') and lines[-1].endswith("EOF"):
            return _check_boundaries_strict(lines[1:-1])
        raise


def parse_patch(patch: str) -> ParsedPatch:
    """Parse patch text in upstream's lenient mode; raises InvalidPatchError / InvalidHunkError."""
    trimmed = _trim(patch)
    # Rust str::lines(): split on \n and drop one trailing \r per line.
    lines = [line[:-1] if line.endswith("\r") else line for line in trimmed.split("\n")] if trimmed else []
    patch_lines = _check_boundaries_lenient(lines)
    parser = _Parser()
    for line in patch_lines[:-1]:
        parser.line_number += 1
        parser.process_line(line)
    parser.finish(patch_lines[-1] if patch_lines else "")
    return ParsedPatch(hunks=parser.hunks, environment_id=parser.environment_id)


_PUNCTUATION = str.maketrans(
    {
        **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-"),
        **dict.fromkeys("\u2018\u2019\u201a\u201b", "'"),
        **dict.fromkeys("\u201c\u201d\u201e\u201f", '"'),
        **dict.fromkeys("\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000", " "),
    }
)


def _normalise(text: str) -> str:
    return _trim(text).translate(_PUNCTUATION)


def seek_sequence(lines: list[str], pattern: list[str], start: int, eof: bool) -> Optional[int]:
    """Find ``pattern`` in ``lines`` at or after ``start`` (at end of file when ``eof``), trying exact,
    then trailing-whitespace-insensitive, then whitespace-insensitive, then punctuation-normalised matches."""
    if not pattern:
        return start
    if len(pattern) > len(lines):
        return None
    search_start = len(lines) - len(pattern) if eof else start
    candidates = range(search_start, len(lines) - len(pattern) + 1)
    for transform in (None, _trim_end, _trim, _normalise):
        wanted = pattern if transform is None else [transform(line) for line in pattern]
        for i in candidates:
            window = lines[i : i + len(pattern)]
            if (window if transform is None else [transform(line) for line in window]) == wanted:
                return i
    return None


def _compute_replacements(
    original_lines: list[str], path: str, chunks: list[UpdateFileChunk]
) -> list[tuple[int, int, list[str]]]:
    replacements: list[tuple[int, int, list[str]]] = []
    line_index = 0
    for chunk in chunks:
        if chunk.change_context is not None:
            idx = seek_sequence(original_lines, [chunk.change_context], line_index, False)
            if idx is None:
                raise ApplyPatchError(f"Failed to find context '{chunk.change_context}' in {path}")
            line_index = idx + 1
        if not chunk.old_lines:
            insertion_idx = (
                len(original_lines) - 1 if original_lines and original_lines[-1] == "" else len(original_lines)
            )
            replacements.append((insertion_idx, 0, list(chunk.new_lines)))
            continue
        pattern, new_slice = chunk.old_lines, chunk.new_lines
        found = seek_sequence(original_lines, pattern, line_index, chunk.is_end_of_file)
        if found is None and pattern[-1] == "":
            pattern = pattern[:-1]
            if new_slice and new_slice[-1] == "":
                new_slice = new_slice[:-1]
            found = seek_sequence(original_lines, pattern, line_index, chunk.is_end_of_file)
        if found is None:
            raise ApplyPatchError(f"Failed to find expected lines in {path}:\n" + "\n".join(chunk.old_lines))
        replacements.append((found, len(pattern), list(new_slice)))
        line_index = found + len(pattern)
    replacements.sort(key=lambda replacement: replacement[0])
    return replacements


def _io_error_text(error: OSError) -> str:
    """Render like Rust's io::Error Display, e.g. 'No such file or directory (os error 2)'."""
    if error.errno is not None:
        return f"{os.strerror(error.errno)} (os error {error.errno})"
    return str(error)


def _read_text(path: str, context: str) -> str:
    try:
        with open(path, "rb") as file:
            data = file.read()
    except OSError as error:
        raise ApplyPatchError(f"{context}: {_io_error_text(error)}") from None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        # Rust's Utf8Error Display.
        if error.reason == "unexpected end of data":
            detail = f"incomplete utf-8 byte sequence from index {error.start}"
        else:
            detail = f"invalid utf-8 sequence of {error.end - error.start} bytes from index {error.start}"
        raise ApplyPatchError(f"{context}: {detail}") from None


def derive_new_contents(path: str, chunks: list[UpdateFileChunk]) -> str:
    original = _read_text(path, f"Failed to read file to update {path}")
    lines = original.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for start, old_len, new_segment in reversed(_compute_replacements(lines, path, chunks)):
        lines[start : start + old_len] = new_segment
    if not lines or lines[-1] != "":
        lines.append("")
    return "\n".join(lines)


def _resolve(cwd: str, path: str) -> str:
    return os.path.normpath(os.path.join(cwd, path))


def _check_in_workspace(path: str, workspace_root: Optional[str]) -> None:
    if workspace_root is None:
        return
    root = os.path.realpath(workspace_root)
    if os.path.commonpath([root, os.path.realpath(path)]) != root:
        raise ApplyPatchError(f"path {path} is outside the workspace {workspace_root}")


def verify_patch(patch: str, cwd: str, workspace_root: Optional[str] = None) -> ParsedPatch:
    """Upstream's pre-application verification: parse, then read every target and locate every
    chunk. Raises ApplyPatchError whose text follows 'apply_patch verification failed: '."""
    parsed = parse_patch(patch)
    if parsed.environment_id is not None:
        raise ApplyPatchError("apply_patch environment selection is unavailable for this turn")
    seen: set[str] = set()
    for hunk in parsed.hunks:
        path = _resolve(cwd, hunk.path)
        if path in seen:
            raise InvalidPatchError(f"multiple operations target {path}")
        seen.add(path)
        _check_in_workspace(path, workspace_root)
        if isinstance(hunk, DeleteFile):
            _read_text(path, f"Failed to read {path}")
        elif isinstance(hunk, UpdateFile):
            derive_new_contents(path, hunk.chunks)
            if hunk.move_path is not None:
                _check_in_workspace(_resolve(cwd, hunk.move_path), workspace_root)
    return parsed


def _write_file(path: str, contents: str) -> None:
    try:
        with open(path, "wb") as file:
            file.write(contents.encode("utf-8"))
    except FileNotFoundError:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as file:
            file.write(contents.encode("utf-8"))


def _remove_file(path: str, context: str) -> None:
    try:
        if os.path.isdir(path):
            raise IsADirectoryError(context)
        os.remove(path)
    except OSError:
        # anyhow's Display of a `.with_context(...)` error is the context alone.
        raise ApplyPatchError(context) from None


def apply_hunks(hunks: list[Hunk], cwd: str) -> str:
    """Apply hunks in order, like upstream ``apply_hunks_to_files``; returns the success summary."""
    if not hunks:
        raise ApplyPatchError("No files were modified.")
    added: list[str] = []
    modified: list[str] = []
    deleted: list[str] = []
    for hunk in hunks:
        path = _resolve(cwd, hunk.path)
        try:
            if isinstance(hunk, AddFile):
                _write_file(path, hunk.contents)
                added.append(hunk.path)
            elif isinstance(hunk, DeleteFile):
                _remove_file(path, f"Failed to delete file {path}")
                deleted.append(hunk.path)
            else:
                new_contents = derive_new_contents(path, hunk.chunks)
                if hunk.move_path is not None:
                    _write_file(_resolve(cwd, hunk.move_path), new_contents)
                    _remove_file(path, f"Failed to remove original {path}")
                    modified.append(hunk.move_path)
                else:
                    try:
                        with open(path, "wb") as file:
                            file.write(new_contents.encode("utf-8"))
                    except OSError:
                        raise ApplyPatchError(f"Failed to write file {path}") from None
                    modified.append(hunk.path)
        except OSError as error:
            raise ApplyPatchError(_io_error_text(error)) from None
    lines = ["Success. Updated the following files:"]
    lines += [f"A {path}" for path in added] + [f"M {path}" for path in modified] + [f"D {path}" for path in deleted]
    return "\n".join(lines) + "\n"


def _format_seconds(seconds: float) -> str:
    # Rust's f32 Display of a value rounded to one decimal: 0 -> "0", 0.25 -> "0.3".
    return f"{round(seconds, 1):g}"


def run_apply_patch_tool(patch: str, cwd: str, workspace_root: Optional[str] = None) -> tuple[str, bool]:
    """The model-visible result of the Codex apply_patch tool: (output text, success)."""
    try:
        parsed = verify_patch(patch, cwd, workspace_root)
    except ApplyPatchError as error:
        return f"apply_patch verification failed: {error}", False
    started = time.monotonic()
    try:
        stdout, stderr, exit_code = apply_hunks(parsed.hunks, cwd), "", 0
    except ApplyPatchError as error:
        stdout, stderr, exit_code = "", f"{error}\n", 1
    elapsed = _format_seconds(time.monotonic() - started)
    return f"Exit code: {exit_code}\nWall time: {elapsed} seconds\nOutput:\n{stdout}{stderr}", exit_code == 0


def main(argv: Optional[list[str]] = None) -> int:
    """The standalone ``apply_patch`` command: PATCH as the only argument, or on stdin."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) > 1:
        print("Error: apply_patch accepts exactly one argument.", file=sys.stderr)
        return 2
    if args:
        patch = args[0]
    else:
        patch = sys.stdin.read()
        if not patch:
            print("Usage: apply_patch 'PATCH'\n       echo 'PATCH' | apply_patch", file=sys.stderr)
            return 2
    try:
        hunks = parse_patch(patch).hunks
    except InvalidHunkError as error:
        print(f"Invalid patch hunk on line {error.line_number}: {error.message}", file=sys.stderr)
        return 1
    except InvalidPatchError as error:
        print(f"Invalid patch: {error.message}", file=sys.stderr)
        return 1
    workspace_root = os.environ.get("CODEX_TOOLS_WORKSPACE_ROOT") or None
    try:
        for hunk in hunks:
            paths = [hunk.path] + ([hunk.move_path] if isinstance(hunk, UpdateFile) and hunk.move_path else [])
            for path in paths:
                _check_in_workspace(_resolve(os.getcwd(), path), workspace_root)
        sys.stdout.write(apply_hunks(hunks, os.getcwd()))
    except ApplyPatchError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
