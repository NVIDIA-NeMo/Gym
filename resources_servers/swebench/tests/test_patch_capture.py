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
"""Tests for the shared worktree/committed patch capture (patch_capture.py): the pure
tip-selection logic, and the async capture against a fake sandbox that answers the git commands
the way a real repo would in each scenario."""

from types import SimpleNamespace

import pytest

from resources_servers.swebench.patch_capture import (
    PATCH_CAPTURE_MODES,
    PatchCapture,
    TipCandidate,
    capture_model_patch,
    drop_sections_under,
    parse_tip_candidates,
    patch_files,
    prepare_git_for_commits,
    select_tip,
)


BASE = "b" * 40
C1 = "1" * 40
C2 = "2" * 40
C3 = "3" * 40

WORKTREE_DIFF = "diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-a\n+b\n"
COMMITTED_DIFF = "diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-a\n+c\n"
GITLINK = (
    "diff --git a/extern/cmake b/extern/cmake\nnew file mode 160000\n--- /dev/null\n+++ b/extern/cmake\n"
    "@@ -0,0 +1 @@\n+Subproject commit abc1234\n"
)


class _FakeSandbox:
    """Answers the git commands capture_model_patch issues, keyed on a substring of each."""

    def __init__(
        self,
        *,
        status: str = "",
        listing: str = "",
        committed_diff: str = COMMITTED_DIFF,
        worktree_diff: str = WORKTREE_DIFF,
        fail_on: str | None = None,
    ) -> None:
        self.status = status
        self.listing = listing
        self.committed_diff = committed_diff
        self.worktree_diff = worktree_diff
        self.fail_on = fail_on
        self.commands: list[str] = []

    async def exec(self, command: str, **kwargs):
        self.commands.append(command)
        if self.fail_on and self.fail_on in command:
            return SimpleNamespace(return_code=128, stdout="", stderr="fatal: git failed")
        if "status --porcelain" in command:
            out = self.status
        elif "for-each-ref" in command:
            out = self.listing
        elif "add -N ." in command:
            out = self.worktree_diff
        elif "--no-pager diff " in command:
            out = self.committed_diff
        elif "config" in command:
            out = ""
        else:  # pragma: no cover
            raise AssertionError(f"unexpected command: {command}")
        return SimpleNamespace(return_code=0, stdout=out, stderr="")


def _listing(*rows: tuple[str, str, int, int], head: str) -> str:
    lines = [f"HEAD_SHA {head}"] + [f"{ref} {sha} {anc} {n}" for ref, sha, anc, n in rows]
    return "\n".join(lines) + "\n"


def test_select_tip_prefers_descendant_with_most_commits_then_head():
    cands = [TipCandidate("a", C1, True, 2), TipCandidate("b", C2, True, 3), TipCandidate("rebased", C3, False, 9)]
    assert select_tip(cands, head_sha=C1).sha == C2
    tie = [TipCandidate("a", C1, True, 2), TipCandidate("b", C2, True, 2)]
    assert select_tip(tie, head_sha=C2).ref == "b"
    assert select_tip(tie, head_sha=None).ref == "a"
    assert select_tip([], head_sha=None) is None


def test_parse_tip_candidates_skips_base_and_prefers_branch_name_over_head():
    listing = _listing(("_nel_work", BASE, 1, 0), ("fix", C1, 1, 2), ("HEAD", C1, 1, 2), head=C1)
    cands, head = parse_tip_candidates(listing, BASE)
    assert head == C1
    assert [(c.ref, c.sha, c.commits_since_base) for c in cands] == [("fix", C1, 2)]


def test_patch_files_and_drop_sections_under():
    quoted = 'diff --git "a/dir with space/f.txt" "b/dir with space/f.txt"\nnew file mode 100644\n'
    assert patch_files(WORKTREE_DIFF + quoted) == ["src/x.py", "dir with space/f.txt"]
    inside = "diff --git a/extern/other/x.txt b/extern/other/x.txt\nnew file mode 100644\n"
    assert drop_sections_under(WORKTREE_DIFF + GITLINK + inside, ["extern/cmake/", "extern/other/"]) == WORKTREE_DIFF
    other = "diff --git a/externals.py b/externals.py\n--- a/externals.py\n+++ b/externals.py\n@@ -1 +1 @@\n-a\n+b\n"
    assert drop_sections_under(other, ["extern/"]) == other  # shared prefix string is not "inside"


@pytest.mark.asyncio
async def test_worktree_mode_is_the_historical_diff():
    sb = _FakeSandbox(status=" M src/x.py\n?? notes.txt\n", listing=_listing(("_nel_work", BASE, 1, 0), head=BASE))
    cap = await capture_model_patch(sb, "/repo", BASE, mode="worktree")
    assert cap.patch == WORKTREE_DIFF and cap.source == "worktree"
    assert cap.worktree_dirty is True and cap.untracked_files == 1 and cap.tip_commit is None
    status_i = next(i for i, c in enumerate(sb.commands) if "status" in c)
    add_i = next(i for i, c in enumerate(sb.commands) if "add -N" in c)
    assert status_i < add_i  # status is read before `add -N` mutates the index


@pytest.mark.asyncio
async def test_committed_mode_diffs_base_to_branch_tip():
    sb = _FakeSandbox(status="", listing=_listing(("_nel_work", BASE, 1, 0), ("fix-123", C2, 1, 3), head=C2))
    cap = await capture_model_patch(sb, "/repo", BASE, mode="committed")
    assert cap.patch == COMMITTED_DIFF and cap.source == "committed"
    assert cap.tip_commit == C2 and cap.branch == "fix-123" and cap.commits == 3 and cap.warnings == []
    assert any(f"--no-pager diff {BASE} {C2}" in c for c in sb.commands)


@pytest.mark.asyncio
async def test_committed_mode_finds_branch_after_checkout_back_to_base():
    sb = _FakeSandbox(status="", listing=_listing(("_nel_work", BASE, 1, 0), ("fix", C1, 1, 1), head=BASE))
    cap = await capture_model_patch(sb, "/repo", BASE, mode="committed")
    assert cap.source == "committed" and cap.branch == "fix" and cap.patch == COMMITTED_DIFF


@pytest.mark.asyncio
async def test_committed_mode_with_no_commit_is_empty_and_warns():
    sb = _FakeSandbox(status=" M src/x.py\n", listing=_listing(("_nel_work", BASE, 1, 0), head=BASE))
    cap = await capture_model_patch(sb, "/repo", BASE, mode="committed")
    assert cap.patch == "" and cap.source == "none" and cap.worktree_dirty is True
    assert any("not captured" in w for w in cap.warnings)


@pytest.mark.asyncio
async def test_committed_mode_flags_rewritten_history():
    sb = _FakeSandbox(status="", listing=_listing(("_nel_work", C3, 0, 4), head=C3))
    cap = await capture_model_patch(sb, "/repo", BASE, mode="committed")
    assert cap.source == "committed" and cap.tip_commit == C3 and any("does not descend" in w for w in cap.warnings)


@pytest.mark.asyncio
async def test_pristine_untracked_files_and_nested_repo_dirs_are_stripped():
    def drop(patch, paths):
        return "" if "src/x.py" in paths else patch

    sb = _FakeSandbox(status="?? src/x.py\n", listing=_listing(("fix", C1, 1, 1), head=C1))
    cap = await capture_model_patch(
        sb, "/repo", BASE, mode="committed", pristine_untracked=frozenset({"src/x.py"}), drop_sections=drop
    )
    assert cap.committed_patch_bytes == 0 and cap.worktree_patch_bytes == 0 and cap.untracked_files == 0

    # ls-files reports a nested checkout as "extern/cmake/"; a diff names it "extern/cmake". The
    # committed diff drops it; the worktree diff is deliberately left as it always was.
    sb2 = _FakeSandbox(
        status="?? extern/\n",
        listing=_listing(("fix", C1, 1, 1), head=C1),
        committed_diff=COMMITTED_DIFF + GITLINK,
        worktree_diff=COMMITTED_DIFF + GITLINK,
    )
    cap2 = await capture_model_patch(
        sb2, "/repo", BASE, mode="committed", pristine_untracked=frozenset({"extern/cmake/"})
    )
    assert cap2.patch == COMMITTED_DIFF
    assert cap2.worktree_patch_bytes == len(COMMITTED_DIFF + GITLINK)


@pytest.mark.asyncio
async def test_worktree_diff_failure_is_non_fatal_only_when_a_commit_was_captured():
    sb = _FakeSandbox(status="", listing=_listing(("fix", C1, 1, 1), head=C1), fail_on="add -N .")
    cap = await capture_model_patch(sb, "/repo", BASE, mode="committed")
    assert cap.patch == COMMITTED_DIFF and any("diagnostic only" in w for w in cap.warnings)
    with pytest.raises(RuntimeError):
        await capture_model_patch(_FakeSandbox(fail_on="add -N ."), "/repo", BASE, mode="worktree")
    with pytest.raises(RuntimeError):
        await capture_model_patch(_FakeSandbox(fail_on="status --porcelain"), "/repo", BASE, mode="committed")


@pytest.mark.asyncio
async def test_unknown_mode_is_rejected_before_touching_the_sandbox():
    sb = _FakeSandbox()
    with pytest.raises(ValueError):
        await capture_model_patch(sb, "/repo", BASE, mode="head_only")
    assert sb.commands == [] and "committed" in PATCH_CAPTURE_MODES


@pytest.mark.asyncio
async def test_prepare_git_for_commits_sets_identity_globally_and_in_repo():
    sb = _FakeSandbox()
    await prepare_git_for_commits(sb, "/repo with space", "t")
    (cmd,) = sb.commands
    assert "config --global user.email" in cmd and "safe.directory '/repo with space'" in cmd
    assert "-C '/repo with space' config user.email" in cmd


def test_response_fields_are_minimal_and_gate_the_patch():
    cap = PatchCapture(patch="x" * 10, mode="committed", source="committed", branch="fix", commits=2)
    assert cap.response_fields(include_patch=False) == {
        "patch_source": "committed",
        "patch_branch": "fix",
        "patch_commits": 2,
        "worktree_dirty": False,
        "model_patch_bytes": 10,
        "model_patch": None,
    }
    assert cap.response_fields(include_patch=True)["model_patch"] == "x" * 10
    assert PatchCapture.static("p", "worktree", "golden").response_fields(True)["patch_source"] == "golden"
