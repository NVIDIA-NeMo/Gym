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
import subprocess
import sys
from pathlib import Path


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


def run_audit(tmp_path: Path, comments: list[dict]) -> tuple[int, str, dict]:
    worktree = tmp_path / "worktree"
    (worktree / "pkg").mkdir(parents=True)
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
