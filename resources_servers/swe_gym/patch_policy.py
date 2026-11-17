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
"""Fail-closed intake of a small, regular-file-only Git text diff.

This is a submission boundary, not a Python or pytest execution sandbox. The path
checks run on the resources server, independently of the agent's Git executable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable


MAX_PATCH_BYTES = 1024 * 1024


def trusted_test_paths(test_patch: str) -> set[str]:
    """Paths in the trusted hidden-test patch, including tests outside named test directories."""
    return {line[6:] for line in test_patch.splitlines() if line.startswith(("--- a/", "+++ b/"))}


_PATH = re.compile(r"[A-Za-z0-9_./-]+")
_HEADER = re.compile(r"diff --git a/(\S+) b/(\S+)")
_HUNK = re.compile(r"@@ -(\d{1,9})(?:,(\d{1,9}))? \+(\d{1,9})(?:,(\d{1,9}))? @@(?: .*)?")
_INDEX = re.compile(r"index [0-9a-f]+\.\.[0-9a-f]+(?: 100(?:644|755))?")
_PROTECTED_NAMES = {
    "conftest.py",
    "pytest.ini",
    "tox.ini",
    "setup.cfg",
    "pyproject.toml",
    "sitecustomize.py",
    "usercustomize.py",
}


class PatchPolicyError(ValueError):
    """The declared submission violates the patch contract (a scored policy failure)."""


def validate_patch(patch: str, *, protected_paths: Iterable[str] = ()) -> tuple[str, ...]:
    """Return changed paths, rejecting ambiguous headers, special files and grading inputs.

    Only ordinary Git unified diffs with unquoted ASCII paths are accepted. Renames,
    copies, mode-only edits, binary patches, symlinks and submodules are unsupported.
    Reject the whole patch; never salvage a source edit from a tampering submission.
    """
    try:
        size = len(patch.encode("utf-8", errors="strict"))
    except UnicodeError as exc:
        raise PatchPolicyError("patch must be UTF-8 text") from exc
    if size > MAX_PATCH_BYTES:
        raise PatchPolicyError(f"patch exceeds {MAX_PATCH_BYTES} bytes")
    if "\x00" in patch or "\r" in patch:
        raise PatchPolicyError("patch contains NUL or carriage returns")
    if not patch:
        return ()
    if not patch.endswith("\n"):
        raise PatchPolicyError("patch must end with a newline")
    lines = patch.splitlines(keepends=True)
    protected = frozenset(protected_paths)
    paths: list[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(lines):
        header = _HEADER.fullmatch(lines[i].removesuffix("\n"))
        if not header or header[1] != header[2]:
            raise PatchPolicyError("expected matching canonical diff --git paths")
        path = header[1]
        parts = path.split("/")
        if (
            not _PATH.fullmatch(path)
            or path.startswith("/")
            or len(path) > 4096
            or any(len(part) > 255 for part in parts)
            or any(part in {"", ".", ".."} or part.startswith(".") for part in parts)
        ):
            raise PatchPolicyError(f"unsafe patch path: {path!r}")
        if path in protected or parts[-1] in _PROTECTED_NAMES or parts[-1].endswith((".pth", ".pyc")):
            raise PatchPolicyError(f"protected grading or environment path: {path!r}")
        if path in seen:
            raise PatchPolicyError(f"duplicate patch path: {path!r}")
        paths.append(path)
        seen.add(path)
        i += 1
        change = None
        if i < len(lines) and lines[i].startswith(("new file mode ", "deleted file mode ")):
            mode = lines[i].strip()
            if mode not in {
                "new file mode 100644",
                "new file mode 100755",
                "deleted file mode 100644",
                "deleted file mode 100755",
            }:
                raise PatchPolicyError("only regular file modes are allowed")
            change = "new" if mode.startswith("new") else "deleted"
            i += 1
        index = None
        if i < len(lines) and _INDEX.fullmatch(lines[i].removesuffix("\n")):
            index = lines[i].split()[1].split("..")
            i += 1
        if change and index and (i == len(lines) or lines[i].startswith("diff --git ")):
            before, after = index
            empty_blob = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
            absent, present = (before, after) if change == "new" else (after, before)
            if set(absent) == {"0"} and len(present) >= 7 and empty_blob.startswith(present):
                continue
            raise PatchPolicyError("invalid empty-file object IDs")
        old = "/dev/null" if change == "new" else f"a/{path}"
        new = "/dev/null" if change == "deleted" else f"b/{path}"
        if lines[i : i + 2] != [f"--- {old}\n", f"+++ {new}\n"]:
            raise PatchPolicyError("missing or mismatched unified diff paths")
        i += 2
        hunks = 0
        while i < len(lines) and lines[i].startswith("@@ "):
            hunk = _HUNK.fullmatch(lines[i].removesuffix("\n"))
            if not hunk:
                raise PatchPolicyError("invalid hunk header")
            remaining_old = int(hunk[2]) if hunk[2] is not None else 1
            remaining_new = int(hunk[4]) if hunk[4] is not None else 1
            i += 1
            while remaining_old or remaining_new:
                if i >= len(lines) or lines[i][0] not in " +-":
                    raise PatchPolicyError("truncated hunk")
                prefix = lines[i][0]
                remaining_old -= prefix in " -"
                remaining_new -= prefix in " +"
                if remaining_old < 0 or remaining_new < 0:
                    raise PatchPolicyError("hunk counts do not match content")
                i += 1
                if i < len(lines) and lines[i] == "\\ No newline at end of file\n":
                    i += 1
            hunks += 1
        if not hunks:
            raise PatchPolicyError("each file must contain a text hunk")
    return tuple(paths)
