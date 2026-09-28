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
"""Model-patch capture shared by the SWE-style resources servers (swe_rebench, scale_swe,
swemer_v1, swemer_v2, swe_next).

``worktree``
    ``git add -N . && git diff <base>``: everything on disk that differs from the base commit,
    committed or not. The historical default, unchanged.

``committed``
    ``git diff <base> <tip>``, where ``<tip>`` is the most advanced commit the agent left on HEAD
    or any local branch (so a branch the agent switched away from still counts). Uncommitted
    edits and untracked files are ignored, as in DeepSWE grading. Pair it with a prompt that asks
    the model to work on a new branch and commit.

After the anti-cheat scrub the repo's only branch is ``_nel_work`` and git has no committer
identity; :func:`prepare_git_for_commits` fixes the latter at seed time.
"""

import re
import shlex
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from time import time
from typing import Any, Literal


PatchCaptureMode = Literal["worktree", "committed"]
PATCH_CAPTURE_MODES: tuple[str, ...] = ("worktree", "committed")

GIT_COMMIT_IDENTITY_EMAIL = "agent@nemo-gym.local"
GIT_COMMIT_IDENTITY_NAME = "NeMo Gym Agent"


@dataclass
class TipCandidate:
    ref: str
    sha: str
    is_descendant: bool
    commits_since_base: int


@dataclass
class PatchCapture:
    patch: str
    mode: str
    source: str  # "worktree" | "committed" | "none" | "golden"
    base_commit: str = ""
    tip_commit: str | None = None
    branch: str | None = None
    commits: int = 0
    worktree_dirty: bool = False
    untracked_files: int = 0
    worktree_patch_bytes: int = 0
    committed_patch_bytes: int = 0
    collection_time_s: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def patch_bytes(self) -> int:
        return len(self.patch.encode("utf-8", errors="replace"))

    def response_fields(self, include_patch: bool) -> dict[str, Any]:
        """The provenance a rollout row carries."""
        return {
            "patch_source": self.source,
            "patch_branch": self.branch,
            "patch_commits": self.commits,
            "worktree_dirty": self.worktree_dirty,
            "model_patch_bytes": self.patch_bytes,
            "model_patch": self.patch if include_patch else None,
        }

    @classmethod
    def static(cls, patch: str, mode: str, source: str) -> "PatchCapture":
        """A capture that did not touch a sandbox (golden patch, or extraction failed upstream)."""
        return cls(patch=patch, mode=mode, source=source)


_DIFF_HEADER = re.compile(r"^diff --git (?:a/(?P<a>\S+)|\"a/(?P<qa>[^\"]+)\") ", re.M)


def patch_files(patch: str) -> list[str]:
    """Paths named in a unified diff's ``diff --git`` headers, in order."""
    return [m.group("a") or m.group("qa") for m in _DIFF_HEADER.finditer(patch)]


def drop_sections_under(patch: str, dirs: Iterable[str]) -> str:
    """Drop every ``diff --git`` section whose path is one of ``dirs`` or lies inside one."""
    prefixes = tuple(d.rstrip("/") for d in dirs if d.strip("/"))
    if not patch or not prefixes:
        return patch
    kept = []
    for section in re.split(r"(?=^diff --git )", patch, flags=re.M):
        if not section.strip():
            continue
        files = patch_files(section)
        path = files[0] if files else None
        if path is not None and any(path == d or path.startswith(d + "/") for d in prefixes):
            continue
        kept.append(section)
    return "".join(kept)


def select_tip(candidates: Iterable[TipCandidate], head_sha: str | None) -> TipCandidate | None:
    """Descendants of base beat non-descendants, more commits beat fewer, HEAD breaks ties."""
    best: TipCandidate | None = None
    best_key: tuple[int, int, int] | None = None
    for c in candidates:
        key = (int(c.is_descendant), c.commits_since_base, int(c.sha == head_sha))
        if best_key is None or key > best_key:
            best, best_key = c, key
    return best


def parse_tip_candidates(listing: str, base_commit: str) -> tuple[list[TipCandidate], str | None]:
    """Parse ``_TIP_LISTING_SCRIPT`` output; refs pointing at ``base_commit`` are not candidates."""
    candidates: dict[str, TipCandidate] = {}
    head_sha: str | None = None
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "HEAD_SHA":
            head_sha = parts[1]
            continue
        if len(parts) != 4:
            continue
        ref, sha, anc, count = parts
        if sha == base_commit:
            continue
        try:
            c = TipCandidate(ref=ref, sha=sha, is_descendant=anc == "1", commits_since_base=int(count))
        except ValueError:
            continue
        prev = candidates.get(sha)
        if prev is None or (prev.ref == "HEAD" and ref != "HEAD"):  # prefer a branch name over "HEAD"
            candidates[sha] = c
    return list(candidates.values()), head_sha


# One shell round trip: every local branch plus HEAD with sha, is-descendant-of-base, commits since base.
_TIP_LISTING_SCRIPT = """
cd {wd} || exit 96
base={base}
echo "HEAD_SHA $(git rev-parse HEAD 2>/dev/null)"
for ref in $(git for-each-ref --format='%(refname:short)' refs/heads 2>/dev/null) HEAD; do
  sha=$(git rev-parse "$ref" 2>/dev/null) || continue
  [ "$sha" = "$base" ] && continue
  if git merge-base --is-ancestor "$base" "$sha" 2>/dev/null; then anc=1; else anc=0; fi
  n=$(git rev-list --count "$base..$sha" 2>/dev/null || echo 0)
  echo "$ref $sha $anc $n"
done
exit 0
"""


async def prepare_git_for_commits(sandbox: Any, workdir: str, log_prefix: str = "patch_capture") -> None:
    """Give the sandbox a committer identity (globally and in the repo) so the agent can commit.
    Never raises."""
    wd = shlex.quote(workdir)
    email, name = shlex.quote(GIT_COMMIT_IDENTITY_EMAIL), shlex.quote(GIT_COMMIT_IDENTITY_NAME)
    try:
        result = await sandbox.exec(
            f"git config --global user.email {email} && git config --global user.name {name}"
            f" && git config --global --add safe.directory {wd}"
            f" ; git -C {wd} config user.email {email} ; git -C {wd} config user.name {name}",
            timeout_s=60,
        )
        if result.return_code != 0:
            print(f"[{log_prefix}] git identity setup returned {result.return_code}: {result.stderr}", flush=True)
    except Exception as exc:  # pragma: no cover - provider hiccup, not a capture concern
        print(f"[{log_prefix}] git identity setup failed: {exc}", flush=True)


async def _exec(sandbox: Any, command: str, *, timeout_s: int = 300) -> Any:
    result = await sandbox.exec(command, timeout_s=timeout_s)
    if result.return_code != 0:
        raise RuntimeError(result.stderr or result.stdout or f"exit {result.return_code}")
    return result


async def capture_model_patch(
    sandbox: Any,
    workdir: str,
    base_commit: str,
    mode: str = "worktree",
    pristine_untracked: frozenset[str] | set[str] = frozenset(),
    drop_sections: Callable[[str, Iterable[str]], str] | None = None,
) -> PatchCapture:
    """Capture the agent's patch from its sandbox per ``mode``. ``drop_sections(patch, paths)``
    strips files that were already untracked before the agent started. Raises ``RuntimeError``
    when git fails; an agent that left nothing yields an empty patch."""
    if mode not in PATCH_CAPTURE_MODES:
        raise ValueError(f"unknown patch_capture_mode {mode!r}; expected one of {PATCH_CAPTURE_MODES}")
    started = time()
    wd, base = shlex.quote(workdir), shlex.quote(base_commit)
    cap = PatchCapture(patch="", mode=mode, source="none", base_commit=base_commit)

    def _clean(patch: str) -> str:
        return drop_sections(patch, pristine_untracked) if drop_sections and pristine_untracked else patch

    # `git ls-files --others` reports a nested checkout as "dir/", while a diff names it "dir", so
    # the exact-path drop above misses it; treat "dir/" entries as prefixes for the committed diff.
    # The worktree diff is left exactly as it always was.
    pristine_dirs = tuple(p.rstrip("/") for p in pristine_untracked if p.endswith("/"))

    # Working-tree state, read before `add -N` touches the index.
    status = await _exec(sandbox, f"git -C {wd} status --porcelain --untracked-files=all")
    untracked = [ln[3:].strip() for ln in (status.stdout or "").splitlines() if ln.startswith("??")]
    cap.worktree_dirty = any(ln.strip() and not ln.startswith("??") for ln in (status.stdout or "").splitlines())
    cap.untracked_files = len([p for p in untracked if p not in pristine_untracked])

    # Committed work.
    listing = await _exec(sandbox, _TIP_LISTING_SCRIPT.format(wd=wd, base=base))
    candidates, head_sha = parse_tip_candidates(listing.stdout or "", base_commit)
    tip = select_tip(candidates, head_sha)
    committed_patch = ""
    if tip is not None:
        cap.tip_commit, cap.branch, cap.commits = tip.sha, tip.ref, tip.commits_since_base
        if not tip.is_descendant:
            cap.warnings.append(f"tip {tip.sha[:12]} ({tip.ref}) does not descend from base; history was rewritten")
        diff = await _exec(sandbox, f"git -C {wd} --no-pager diff {base} {shlex.quote(tip.sha)}")
        committed_patch = drop_sections_under(_clean(diff.stdout or ""), pristine_dirs)
    cap.committed_patch_bytes = len(committed_patch.encode("utf-8", errors="replace"))

    # Working tree vs base; intent-to-add so new files appear. Diagnostic only in `committed` mode,
    # so a failure there must not throw away a committed patch that was already captured.
    try:
        diff = await _exec(sandbox, f"git -C {wd} add -N . && git -C {wd} --no-pager diff {base}")
        worktree_patch = _clean(diff.stdout or "")
    except RuntimeError as exc:
        if mode == "committed" and tip is not None:
            worktree_patch = ""
            cap.warnings.append(f"worktree diff failed (diagnostic only): {exc}")
        else:
            raise
    cap.worktree_patch_bytes = len(worktree_patch.encode("utf-8", errors="replace"))

    if mode == "worktree":
        cap.patch, cap.source = worktree_patch, "worktree"
    else:
        cap.patch, cap.source = committed_patch, ("committed" if tip is not None else "none")
        if tip is None and cap.worktree_patch_bytes:
            cap.warnings.append("no commit found; uncommitted work in the tree was not captured")

    cap.collection_time_s = time() - started
    return cap
