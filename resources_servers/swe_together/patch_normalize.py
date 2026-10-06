# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from Togetherbench/SWE-Together, Apache-2.0, revision 891d19eb4b3a64a47c3d49bbd066a311e0133254.

"""Make a harness-recorded patch applicable with ``git apply``.

``capture_git_diff`` writes ``agent/final.patch`` with two artefacts that plain
``git apply`` rejects:

* a ``=== <repo> (cumulative vs harbor-base) ===`` banner line per repo section
  ("unrecognized input");
* in patches recorded before ``repo_diff._trim_diff``, a last hunk whose trailing
  blank context line (a lone ``" "``) was eaten by ``str.strip()``, leaving the
  hunk one line short of its ``@@`` header ("corrupt patch at line N").

Both are properties of the recording, not of the agent's work, so the judge
normalises them before applying and the repair tool rewrites stored patches.
Kept free of harness imports so ``eval/`` can use it without pulling in Harbor.
"""

from __future__ import annotations

import re


BANNER_RE = re.compile(r"^=== .* ===$")
_BANNER_PATH_RE = re.compile(r"^=== (\S+) \(", re.M)
_HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
_SUBMODULE_RE = re.compile(r"^index [0-9a-f]+\.\.[0-9a-f]+ 160000$", re.M)
_BINARY_RE = re.compile(r"^Binary files .* differ$", re.M)


def banner_repos(diff: str) -> list[str]:
    """Repo paths named by the recorder's ``=== <path> (...) ===`` sections, in
    file order."""
    return [m.group(1) for m in _BANNER_PATH_RE.finditer(diff)]


def _is_scratch(path: str) -> bool:
    """Clones the agent made itself (under /tmp) rather than the task repo."""
    return path == "/tmp" or path.startswith("/tmp/")


def main_repo_path(diff: str) -> str | None:
    """Absolute path of the task repo the diff was recorded against: the first
    banner that is not an agent scratch clone under /tmp (older recordings list
    repos alphabetically, so ``/tmp/...`` could precede ``/workspace``); falls
    back to the first banner; ``None`` without banners."""
    repos = banner_repos(diff)
    if not repos:
        return None
    return next((r for r in repos if not _is_scratch(r)), repos[0])


def strip_banners(diff: str) -> str:
    return "\n".join(line for line in diff.split("\n") if not BANNER_RE.match(line))


def main_repo_section(diff: str) -> str:
    """The diff restricted to the task repo's section (see
    :func:`main_repo_path`). Hunks against scratch clones the agent created
    elsewhere cannot apply in the judge sandbox and are not what is graded."""
    lines = diff.split("\n")
    starts = [i for i, line in enumerate(lines) if BANNER_RE.match(line)]
    if len(starts) <= 1:
        return diff
    target = main_repo_path(diff)
    for n, i in enumerate(starts):
        if _BANNER_PATH_RE.match(lines[i]).group(1) == target:
            end = starts[n + 1] if n + 1 < len(starts) else len(lines)
            return "\n".join(lines[i:end])
    return diff


def drop_unapplyable_blocks(diff: str) -> tuple[str, dict[str, int]]:
    """Remove file blocks that ``git apply`` can never apply from a recorded
    text diff and that carry no gradable source change:

    * submodule pointer updates (mode ``160000`` — needs the submodule checked
      out at the new commit);
    * ``Binary files ... differ`` placeholders (``git diff`` without
      ``--binary``; the agent added/changed a zip/png).

    Returns the trimmed diff and counts of what was dropped."""
    blocks = re.split(r"^(?=diff --git )", diff, flags=re.M)
    kept: list[str] = []
    dropped = {"submodule": 0, "binary": 0}
    for b in blocks:
        if not b.startswith("diff --git "):
            kept.append(b)
            continue
        if _SUBMODULE_RE.search(b):
            dropped["submodule"] += 1
        elif _BINARY_RE.search(b):
            dropped["binary"] += 1
        else:
            kept.append(b)
    return "".join(kept), dropped


def restore_trailing_context(diff: str) -> str:
    """Primary repair for a last hunk that is short of its header by k trailing
    context lines (the whitespace-only lines ``str.strip()`` removed): drop them
    and shrink the ``@@`` counts by k so the hunk is self-consistent. Context
    carries no change, so the applied result is identical — unless the lost
    lines were the hunk's only trailing context, in which case git anchors the
    hunk at end-of-file and rejects it; see :func:`apply_candidates`."""
    lines = _rstrip_empty(diff)
    hi, counts = _short_last_hunk(lines)
    if hi is None:
        return "\n".join(lines)
    lines[hi] = _rewrite_hunk_counts(lines[hi], *counts)
    return "\n".join(lines)


def _rstrip_empty(diff: str) -> list[str]:
    lines = diff.split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _short_last_hunk(lines: list[str]) -> tuple[int | None, tuple[int, int]]:
    """Index of the last hunk header and the body's (old, new) counts when that
    hunk is short of its header by the same number k ≥ 1 of lines on both sides
    (i.e. k trailing context lines were lost); ``(None, (0, 0))`` otherwise."""
    hunks = [i for i, line in enumerate(lines) if line.startswith("@@")]
    if not hunks:
        return None, (0, 0)
    hi = hunks[-1]
    m = _HUNK_RE.match(lines[hi])
    if not m:
        return None, (0, 0)
    old_n, new_n = int(m.group(1) or 1), int(m.group(2) or 1)
    body = [line for line in lines[hi + 1 :] if not line.startswith("\\ ")]
    old = sum(1 for line in body if line[:1] in (" ", "-"))
    new = sum(1 for line in body if line[:1] in (" ", "+"))
    k = old_n - old
    if k >= 1 and new_n - new == k:
        return hi, (old, new)
    return None, (0, 0)


def _rewrite_hunk_counts(header: str, old: int, new: int) -> str:
    m = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$", header)
    return f"@@ -{m.group(1)},{old} +{m.group(2)},{new} @@{m.group(3)}"


def apply_candidates(diff: str) -> list[str]:
    """Patch texts to try in order with ``git apply``, all derived from the
    task repo's section of the recorded diff with submodule-pointer and binary
    placeholder blocks removed. The first is the primary normalisation; when a
    trailing context line is missing a second candidate re-adds it as an empty
    line (right when the lost line was genuinely empty, e.g. a blank line at
    end of file). Candidates differ only in that unknowable whitespace line."""
    if not diff.strip():
        return []
    base, _ = drop_unapplyable_blocks(strip_banners(main_repo_section(diff)))
    if not base.strip():
        return []
    out = [restore_trailing_context(base) + "\n"]
    lines = _rstrip_empty(base)
    hi, counts = _short_last_hunk(lines)
    if hi is not None:
        m = _HUNK_RE.match(lines[hi])
        k = int(m.group(1) or 1) - counts[0]
        out.append("\n".join(lines + [" "] * k) + "\n")
    return out


def normalize_for_git_apply(diff: str) -> str:
    """Banner-free, trailing-context-complete, newline-terminated patch text.
    Returns ``""`` for an empty/whitespace-only input."""
    if not diff.strip():
        return ""
    return restore_trailing_context(strip_banners(diff)) + "\n"
