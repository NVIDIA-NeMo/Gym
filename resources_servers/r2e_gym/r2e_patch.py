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
"""Row-shaping helpers for R2E-Gym-Subset: golden patch reconstruction, issue text, ids, image names.

The Hub row does not carry a patch. It carries ``parsed_commit_content``: the fixing commit parsed into
per-file hunks (plus the full old/new file contents, which is why that field runs to megabytes).
``golden_patch`` rebuilds the unified diff from the hunks exactly as R2E-Gym's own ``ParsedCommit.get_patch``
does (https://github.com/R2E-Gym/R2E-Gym, MIT: ``r2egym/commit_models/diff_classes.py``) so the
golden-patch sweep grades the same patch the upstream harness calls "gold" -- without importing r2egym.
"""

from __future__ import annotations

import re
from typing import Any, Iterable


DATASET_NAME = "R2E-Gym/R2E-Gym-Subset"
IMAGE_REGISTRY = "docker.io"

# Status lines the test runner prints; the row's expected_output_json uses the same vocabulary.
_LINE_PREFIX = {"context": " ", "added": "+", "deleted": "-", "note": "\\ "}


def is_test_file(path: str) -> bool:
    """R2E-Gym's own rule for which diff sections are test code (``FileDiff.is_test_file``)."""
    parts = path.split("/")
    return (
        path.endswith("_test.py")
        or path.startswith("test_")
        or parts[-1].startswith("test_")
        or any(p in parts for p in ("tests", "Tests", "test", "Test"))
    )


def _range(r: dict[str, Any]) -> str:
    return f"{r['start']}" if r.get("length") is None else f"{r['start']},{r['length']}"


def file_diff_patch(file_diff: dict[str, Any]) -> str:
    """One file's section of the unified diff, rebuilt from R2E-Gym's parsed representation."""
    path = file_diff["header"]["file"]["path"]
    out = [f"diff --git a/{path} b/{path}\n"]
    misc = file_diff["header"].get("misc_line")
    if misc:
        out.append(misc + "\n")
    index = file_diff.get("index_line")
    if index:
        mode = index.get("mode") or ""
        out.append(f"index {index['old_commit_hash']}..{index['new_commit_hash']}{' ' if mode else ''}{mode}\n")
    if file_diff.get("is_binary_file"):
        out.append((file_diff.get("binary_line") or "") + "\n")
    minus, plus = file_diff.get("minus_file"), file_diff.get("plus_file")
    if minus and plus:
        out.append(f"--- {minus['path']}\n+++ {plus['path']}\n")
    for hunk in file_diff.get("hunks") or []:
        d = hunk["descriptor"]
        header = f"@@ -{_range(d['old_range'])} +{_range(d['new_range'])} @@"
        if d.get("section"):
            header += f" {d['section']}"
        out.append(header + "\n")
        for line in hunk["line_group"]["all_lines"]:
            out.append(f"{_LINE_PREFIX[line['type']]}{line['content']}\n")
    return "".join(out)


def golden_patch(
    parsed_commit: dict[str, Any],
    *,
    test_file: bool = True,
    non_test_file: bool = True,
    only_python: bool = True,
    exclude_files: Iterable[str] = (),
) -> str:
    """The patch R2E-Gym's harness applies in ``gold`` mode: every Python file diff of the commit.

    Defaults match ``ParsedCommit.get_patch()``: test and non-test Python files, nothing else.
    """
    exclude = set(exclude_files)
    out = []
    for file_diff in parsed_commit.get("file_diffs") or []:
        path = file_diff["header"]["file"]["path"]
        if path in exclude or (only_python and not path.endswith(".py")):
            continue
        test = is_test_file(path)
        if (test and test_file) or (not test and non_test_file):
            out.append(file_diff_patch(file_diff))
    return "".join(out)


def issue_text(problem_statement: str) -> str:
    """The task instruction: the text inside ``[ISSUE] ... [/ISSUE]`` when present (R2E-Gym's
    ``get_task_instruction``), else the whole statement."""
    match = re.search(r"\[ISSUE\](.*)\[/ISSUE\]", problem_statement, re.DOTALL)
    return (match.group(1) if match else problem_statement).strip()


def instance_id_for(repo_name: str, commit_hash: str) -> str:
    """R2E rows have no id; ``<repo>__<commit>`` is unique (one image per commit) and filesystem-safe."""
    return f"{repo_name}__{commit_hash}"


def image_name(docker_image: str) -> str:
    """Fully qualified image ref for the sandbox provider (``namanjain12/<repo>_final:<commit>`` on Docker Hub)."""
    return docker_image if docker_image.startswith(f"{IMAGE_REGISTRY}/") else f"{IMAGE_REGISTRY}/{docker_image}"
