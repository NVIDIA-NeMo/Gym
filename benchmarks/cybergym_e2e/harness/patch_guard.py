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
"""Reject agent patches that touch files the benchmark marks immutable (CyberGym `immutable_files`).

When it runs: ONCE per submission, not per agent action. The agent edits files freely during the
episode; this only screens the final `fix.patch` deliverable -- `run.py`'s score() calls
strip_immutable() on it right before the patch is applied in the validation VM (and validate.py runs
it on the ground-truth patch for the known-answer control).

Why: CyberGym's validate.py reads `immutable_files` into the task config but never enforces it, so a
patch could "pass" stage 3 (tests pass) by editing a project's tests/ instead of fixing the bug.
strip_immutable() drops those hunks; unsafe/unparsable patches are rejected outright.

Rejects: any path under an immutable prefix (old or new side, incl. renames/copies/deletes),
absolute paths or '..' components (escape the repo), symlink modes (point outside later), binary
patches (can't be inspected), and diff formats this guard cannot parse -- context or old-style
"normal" diffs -- which GNU `patch` would still apply (so they must not slip past unstripped).
Paths are tried at strip levels 0-3, matching validate.apply_patch.

  python patch_guard.py <patch> [--immutable tests/ ...]   # exit 1 if rejected
"""

import argparse
import re
import sys
from pathlib import PurePosixPath


_HEADER = re.compile(r"^(?:--- |\+\+\+ |rename from |rename to |copy from |copy to )(\S+)")
_GIT = re.compile(r"^diff --git (\S+) (\S+)")


def patch_paths(text):
    """Every file path the patch names (headers, git extended headers), /dev/null excluded."""
    paths = set()
    for line in text.splitlines():
        if m := _GIT.match(line):
            paths.update(m.groups())
        elif m := _HEADER.match(line):
            paths.add(m.group(1))
    paths.discard("/dev/null")
    return paths


def _stripped(path):
    parts = PurePosixPath(path).parts
    return {"/".join(parts[n:]) for n in range(4) if len(parts) > n}


def check_patch(text, immutable):
    """Return a list of violations; empty means the patch is allowed."""
    problems = []
    if re.search(r"^GIT binary patch|^Binary files ", text, re.M):
        problems.append("binary patch")
    # Only git/unified diffs are parsed for immutable-path stripping. A context diff or an old-style
    # "normal" diff would slip past the stripper (patch_paths/strip_immutable don't model them) yet GNU
    # `patch` applies them -- so reject those formats anywhere in the text. Scanning the WHOLE patch also
    # means a valid unified section can't conceal an unsupported one (reject mixed input).
    if re.search(r"^\*{15,}|^\*\*\* \d|^\d+(?:,\d+)?[acd]\d+(?:,\d+)?\s*$", text, re.M):
        problems.append("unsupported patch format (context/normal diff)")
    if re.search(r"^(?:new file mode|new mode|old mode|deleted file mode) 120000", text, re.M):
        problems.append("symlink in patch")
    prefixes = [p.rstrip("/") for p in immutable]
    for path in sorted(patch_paths(text)):
        if path.startswith("/") or ".." in PurePosixPath(path).parts:
            problems.append(f"path escapes repo: {path}")
            continue
        for cand in _stripped(path):
            hit = next((p for p in prefixes if cand == p or cand.startswith(p + "/")), None)
            if hit:
                problems.append(f"touches immutable {hit}/: {path}")
                break
    return problems


def strip_immutable(text, immutable):
    """Drop every per-file section touching an immutable path; keep the rest of the patch.

    Returns (patch, dropped_paths). Raises ValueError for what can't be made safe by dropping
    (escaping paths, symlinks, binary, unparsable formats) -- those reject the whole patch.
    """
    fatal = [p for p in check_patch(text, []) if not p.startswith("touches immutable")]
    if fatal:
        raise ValueError("; ".join(fatal))
    lines = text.splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if ln.startswith("diff --git ")]
    if not starts:  # plain unified diff: sections start at '--- ' followed by '+++ '
        starts = [i for i, ln in enumerate(lines)
                  if ln.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ ")]
    if not starts:
        # Fail closed: a non-empty patch we can't section into recognized git/unified file blocks must
        # NOT pass through unstripped (it could still apply via `patch` and touch immutable files).
        if text.strip():
            raise ValueError("unrecognized patch format (no git/unified file sections)")
        return text, []
    sections = [lines[:starts[0]]] + [lines[a:b] for a, b in zip(starts, starts[1:] + [len(lines)])]
    kept, dropped = [sections[0]], []
    for sec in sections[1:]:
        body = "".join(sec)
        if check_patch(body, immutable):
            dropped += sorted(patch_paths(body))
        else:
            kept.append(sec)
    return "".join("".join(s) for s in kept), dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patch")
    ap.add_argument("--immutable", action="append", default=None, help="default: tests/")
    a = ap.parse_args()
    problems = check_patch(open(a.patch, encoding="utf-8", errors="replace").read(), a.immutable or ["tests/"])
    for p in problems:
        print(f"REJECT {p}")
    print("OK" if not problems else f"{len(problems)} violation(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
