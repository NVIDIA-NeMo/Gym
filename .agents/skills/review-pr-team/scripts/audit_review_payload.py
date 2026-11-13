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
"""Check a pending-review payload before it is posted to GitHub.

The payload is the JSON body for `POST /repos/{owner}/{repo}/pulls/{pr}/reviews`.
Each comment may carry an `_expect` key: text that must appear on the lines the comment is anchored to.
Keys that start with `_` are removed from the payload written to `--out`.

Exit status is 1 when any check fails, so the payload is never posted with a failing row.
"""

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
SUGGESTION_BLOCK = re.compile(r"```suggestion\n(.*?)```", re.DOTALL)
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
FENCED_BLOCK = re.compile(r"```.*?```", re.DOTALL)
UNPINNED_BLOB_LINK = re.compile(r"github\.com/[^/\s]+/[^/\s]+/blob/(?![0-9a-f]{40}/)[^\s)]+")
BARE_FILE_LINE = re.compile(r"\b[\w./-]+\.(?:py|yaml|yml|md|mdx|toml|json):\d+")
STRUCTURE = re.compile(r"^\s*(?:async\s+def|def|class|return)\b", re.MULTILINE)


def right_side_hunks(diff_text: str) -> dict[str, list[range]]:
    """Map each changed file to the right-side line ranges of its diff hunks."""
    hunks: dict[str, list[range]] = {}
    path = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip()
            path = None if target == "/dev/null" else target.removeprefix("b/")
            continue
        match = HUNK_HEADER.match(line)
        if match and path is not None:
            start = int(match.group(1))
            length = int(match.group(2)) if match.group(2) is not None else 1
            hunks.setdefault(path, []).append(range(start, start + length))
    return hunks


def check_comment(comment: dict, worktree: Path, hunks: dict[str, list[range]]) -> dict[str, str]:
    """Return a status (`ok`, `warn: ...`, or `FAIL: ...`) for each check on one comment."""
    path = comment["path"]
    end = comment["line"]
    start = comment.get("start_line", end)
    body = comment["body"]
    results: dict[str, str] = {}

    in_one_hunk = any(start in hunk and end in hunk for hunk in hunks.get(path, []))
    results["hunk"] = "ok" if in_one_hunk else f"FAIL: lines {start}-{end} are not inside one hunk of {path}"

    file_path = worktree / path
    lines = file_path.read_text().splitlines() if file_path.is_file() else []
    anchored = "\n".join(lines[start - 1 : end])
    expect = comment.get("_expect")
    if not lines:
        results["anchor"] = f"FAIL: {path} does not exist in the worktree"
    elif expect is None:
        results["anchor"] = "warn: no _expect given"
    elif expect in anchored:
        results["anchor"] = "ok"
    else:
        results["anchor"] = f"FAIL: {expect!r} is not on lines {start}-{end}"

    prose = MARKDOWN_LINK.sub("", FENCED_BLOCK.sub("", body))
    unpinned = UNPINNED_BLOB_LINK.findall(body)
    bare = BARE_FILE_LINE.findall(prose)
    if unpinned:
        results["perm"] = f"FAIL: link not pinned to a commit: {unpinned[0]}"
    elif bare:
        results["perm"] = f"FAIL: file reference is not a link: {bare[0]}"
    else:
        results["perm"] = "ok"

    blocks = SUGGESTION_BLOCK.findall(body)
    if not blocks:
        results["sugg"] = "n/a"
    elif len(blocks) > 1:
        results["sugg"] = "FAIL: more than one suggestion block in one comment"
    else:
        results["sugg"] = check_suggestion(path, lines, start, end, blocks[0])

    results["mermaid"] = "warn: render it with mermaid-cli" if "```mermaid" in body else "n/a"
    return results


def check_suggestion(path: str, lines: list[str], start: int, end: int, block: str) -> str:
    """Apply a suggestion block to the file and check the result still parses and keeps its structure."""
    replaced = "\n".join(lines[start - 1 : end])
    if len(STRUCTURE.findall(block)) < len(STRUCTURE.findall(replaced)):
        return "FAIL: the block removes a def, class, or return that the anchored lines contain"
    new_text = "\n".join(lines[: start - 1] + block.rstrip("\n").split("\n") + lines[end:]) + "\n"
    if path.endswith(".py"):
        try:
            ast.parse(new_text)
        except SyntaxError as error:
            return f"FAIL: the file no longer parses: {error.msg} at line {error.lineno}"
        ruff = shutil.which("ruff")
        if ruff is None:
            return "warn: parses; ruff not found, formatting not checked"
        check = subprocess.run(
            [ruff, "format", "--check", "--stdin-filename", path, "-"],
            input=new_text,
            capture_output=True,
            text=True,
            errors="replace",
        )
        if check.returncode != 0:
            return "FAIL: ruff format would change the file"
    elif path.endswith((".yaml", ".yml")):
        try:
            yaml.safe_load(new_text)
        except yaml.YAMLError as error:
            return f"FAIL: the file no longer parses as YAML: {error}"
    return "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--payload", type=Path, required=True, help="review JSON with optional _expect keys")
    parser.add_argument("--worktree", type=Path, required=True, help="checkout of the PR head the payload targets")
    parser.add_argument("--diff", type=Path, required=True, help="output of `gh pr diff <PR>`")
    parser.add_argument("--out", type=Path, help="where to write the payload without `_` keys")
    args = parser.parse_args(argv)

    payload = json.loads(args.payload.read_text())
    hunks = right_side_hunks(args.diff.read_text())
    failed = False
    print("comment  path:line  hunk  anchor  perm  sugg  mermaid")
    for index, comment in enumerate(payload["comments"], start=1):
        results = check_comment(comment, args.worktree, hunks)
        summary = "  ".join("FAIL" if value.startswith("FAIL") else value.split(":")[0] for value in results.values())
        print(f"C{index}  {comment['path']}:{comment['line']}  {summary}")
        for name, value in results.items():
            if value.startswith(("FAIL", "warn")):
                print(f"    {name}: {value}")
        failed = failed or any(value.startswith("FAIL") for value in results.values())

    if args.out is not None:
        clean = dict(payload)
        clean["comments"] = [{k: v for k, v in c.items() if not k.startswith("_")} for c in payload["comments"]]
        args.out.write_text(json.dumps(clean, indent=1))
    print("REFUSING TO POST: fix the failing rows" if failed else "ALL CHECKS PASSED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
