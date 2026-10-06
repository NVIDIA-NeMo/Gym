# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Togetherbench/SWE-Together, Apache-2.0, revision 891d19eb4b3a64a47c3d49bbd066a311e0133254.

import re


def _trim_diff(diff: str) -> str:
    """Remove leading/trailing *empty* lines only. ``str.strip()`` would also eat a
    final blank context line (a lone space) when the last hunk ends on an empty
    source line, leaving the hunk one line short of its header and making
    ``git apply`` reject the patch as corrupt."""
    lines = diff.split("\n")
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


_JUNK_RE = re.compile(
    r"(^|/)(\.venv|venv|node_modules|__pycache__|\.pytest_cache|\.mypy_cache"
    r"|\.ruff_cache|\.tox|\.desloppify|\.coverage|\.cache|\.eggs|\.gradle"
    r"|\.next|site-packages"
    r"|\.go|pkg/mod|\.cargo|\.npm|\.pnpm-store|\.m2|\.ivy2|\.nuget|\.stack-work|\.pub-cache"
    r")(/|$)"
)

USER_SIM_DIFF_MAX_CHARS = 200_000


def truncate_diff_for_user_sim(diff: str, limit: int = USER_SIM_DIFF_MAX_CHARS) -> str:
    """Cut a diff at ``limit`` chars on a line boundary, appending a marker with
    the total size and file count so the user-sim knows it saw a prefix."""
    if len(diff) <= limit:
        return diff
    nfiles = diff.count("\ndiff --git ") + int(diff.startswith("diff --git "))
    head = diff[:limit]
    cut = head.rfind("\n")
    if cut > 0:
        head = head[:cut]
    return (
        f"{head}\n[... diff truncated for the user simulator: {len(diff):,} chars "
        f"across {nfiles} files; showing the first {len(head):,} chars ...]"
    )


def _strip_junk(diff: str) -> str:
    """Drop diff blocks for run-generated junk dirs (.venv, node_modules, …).

    Preserves the per-repo ``=== <path> (… vs …) ===`` section headers and every
    non-junk file block. A ``diff --git a/<junk>/…`` header switches to skip mode
    until the next file block or section header.
    """
    if not diff:
        return diff
    out, skip = [], False
    for ln in diff.split("\n"):
        if ln.startswith("=== ") and ln.endswith(" ==="):
            skip = False
            out.append(ln)
            continue
        m = re.match(r"diff --git a/(.+?) b/", ln)
        if m:
            skip = bool(_JUNK_RE.search(m.group(1)))
        if not skip:
            out.append(ln)
    return _trim_diff("\n".join(out))
