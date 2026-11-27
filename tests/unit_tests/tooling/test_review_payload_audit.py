# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT = REPO_ROOT / ".agents/skills/review-pr-team/scripts/audit_review_payload.py"
SHA = "a" * 40
SOURCE = "def add(a, b):\n    total = a + b\n    return total\n\n\ndef unchanged():\n    return 1\n"
DIFF = """diff --git a/pkg/math.py b/pkg/math.py
--- a/pkg/math.py
+++ b/pkg/math.py
@@ -1,3 +1,3 @@
 def add(a, b):
-    total = a - b
+    total = a + b
     return total
"""


def run_audit(tmp_path: Path, comments: list[dict], ruff_config: str | None = None) -> tuple[int, str, dict]:
    worktree = tmp_path / "worktree"
    (worktree / "pkg").mkdir(parents=True)
    if ruff_config is not None:
        (worktree / "ruff.toml").write_text(ruff_config)
    (worktree / "pkg/math.py").write_text(SOURCE)
    (tmp_path / "pr.diff").write_text(DIFF)
    (tmp_path / "review.json").write_text(json.dumps({"commit_id": SHA, "comments": comments}))
    result = subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            "--payload",
            str(tmp_path / "review.json"),
            "--worktree",
            str(worktree),
            "--diff",
            str(tmp_path / "pr.diff"),
            "--out",
            str(tmp_path / "clean.json"),
        ],
        capture_output=True,
        text=True,
    )
    clean = json.loads((tmp_path / "clean.json").read_text())
    return result.returncode, result.stdout, clean


def test_a_well_formed_comment_passes_and_private_keys_are_removed(tmp_path: Path) -> None:
    body = (
        "The sum overflows for large inputs.\n\n"
        f"See [`pkg/math.py:2`](https://github.com/NVIDIA-NeMo/Gym/blob/{SHA}/pkg/math.py#L2).\n\n"
        "```suggestion\n    total = int(a) + int(b)\n```"
    )
    code, out, clean = run_audit(
        tmp_path, [{"path": "pkg/math.py", "line": 2, "side": "RIGHT", "body": body, "_expect": "total"}]
    )

    assert code == 0, out
    assert "ALL CHECKS PASSED" in out
    assert clean["comments"] == [{"path": "pkg/math.py", "line": 2, "side": "RIGHT", "body": body}]


def test_each_check_fails_on_its_own_defect(tmp_path: Path) -> None:
    comments = [
        # Line 6 is outside the only hunk, which covers lines 1-3.
        {"path": "pkg/math.py", "line": 6, "body": "Outside the diff."},
        # The anchored line does not contain the expected text.
        {"path": "pkg/math.py", "line": 2, "body": "Wrong line.", "_expect": "return"},
        # A link to a branch instead of a commit, and a file reference that is not a link.
        {"path": "pkg/math.py", "line": 2, "body": "See https://github.com/NVIDIA-NeMo/Gym/blob/main/pkg/math.py#L2"},
        {"path": "pkg/math.py", "line": 2, "body": "Compare with `pkg/other.py:10`."},
        # A suggestion that deletes the return statement it replaces.
        {"path": "pkg/math.py", "start_line": 2, "line": 3, "body": "```suggestion\n    total = a + b\n```"},
        # A suggestion that leaves the file unparseable.
        {"path": "pkg/math.py", "line": 2, "body": "```suggestion\n    total = (a +\n```"},
    ]
    code, out, _ = run_audit(tmp_path, comments)
    lines = out.splitlines()

    assert code == 1
    assert "REFUSING TO POST" in out
    assert any("hunk: FAIL" in line for line in lines)
    assert any("anchor: FAIL" in line and "'return'" in line for line in lines)
    assert any("perm: FAIL: link not pinned" in line for line in lines)
    assert any("perm: FAIL: file reference is not a link: pkg/other.py:10" in line for line in lines)
    assert any("sugg: FAIL: the block removes" in line for line in lines)
    assert any("sugg: FAIL: the file no longer parses" in line for line in lines)


def test_anchors_must_be_a_forward_range_on_the_right_side(tmp_path: Path) -> None:
    comments = [
        {"path": "pkg/math.py", "start_line": 3, "line": 2, "body": "Reversed range."},
        {"path": "pkg/math.py", "line": 2, "side": "LEFT", "body": "Old side.", "_expect": "a + b"},
        {"path": "pkg/math.py", "start_line": 2, "line": 3, "start_side": "LEFT", "body": "Mixed sides."},
    ]
    code, out, _ = run_audit(tmp_path, comments)
    lines = out.splitlines()

    assert code == 1
    assert any("hunk: FAIL: lines 3-2" in line for line in lines)
    assert sum("anchor: FAIL: only RIGHT-side comments" in line for line in lines) == 2


def test_a_valid_range_with_context_lines_passes(tmp_path: Path) -> None:
    comments = [
        {"path": "pkg/math.py", "start_line": 1, "line": 3, "side": "RIGHT", "body": "Whole hunk.", "_expect": "def"},
    ]
    code, out, _ = run_audit(tmp_path, comments)

    assert code == 0, out


def test_an_added_class_or_nested_function_does_not_hide_a_removed_return(tmp_path: Path) -> None:
    comments = [
        # Two declarations or returns before and after, but add() no longer returns.
        {
            "path": "pkg/math.py",
            "start_line": 1,
            "line": 3,
            "body": "```suggestion\ndef add(a, b):\n    class Result:\n        pass\n```",
        },
        # The return moves into a nested function, so add() still returns None.
        {
            "path": "pkg/math.py",
            "start_line": 2,
            "line": 3,
            "body": "```suggestion\n    def inner():\n        return a + b\n```",
        },
    ]
    code, out, _ = run_audit(tmp_path, comments)
    failures = [line for line in out.splitlines() if "sugg: FAIL" in line]

    assert code == 1
    assert len(failures) == 2
    assert all("a return in `add`" in line for line in failures)


def test_a_return_turned_into_a_string_counts_as_removed(tmp_path: Path) -> None:
    # The block still has a line starting with "return", but it is inside a string, so add() returns None.
    body = '```suggestion\n    total = a + b\n    """\n    return total\n    """\n```'
    comment = {"path": "pkg/math.py", "start_line": 2, "line": 3, "body": body, "_expect": "total"}
    code, out, _ = run_audit(tmp_path, [comment])

    assert code == 1
    assert "sugg: FAIL: the block removes a return in `add`" in out


@pytest.mark.skipif(shutil.which("ruff") is None, reason="ruff is not installed")
def test_formatting_uses_the_pr_worktree_configuration(tmp_path: Path) -> None:
    # Fine at the repository's line length, but too long for the PR worktree's 40-column setting.
    body = "```suggestion\n    total = int(a) + int(b) + int(a) + int(b)\n```"
    comment = {"path": "pkg/math.py", "line": 2, "body": body, "_expect": "total"}
    code, out, _ = run_audit(tmp_path, [comment], ruff_config="line-length = 40\n")

    assert code == 1
    assert "sugg: FAIL: ruff format would change the file" in out
