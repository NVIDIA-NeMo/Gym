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
"""Grade a pull request's risk as ``low``, ``mid`` or ``high``.

The CI-Events ``pr-risk-label`` agent runs this file from the pull request's target branch tip on two git worktrees,
the merge base and the head:

    python3 <target>/.github/autobot/pr_risk.py --base <merge-base> --head <head>

It prints the tier on stdout and its evidence on stderr. The agent may raise the tier, never lower it.

- ``low``: only docs changed, or at most 100 changed lines with no public-surface or scoring change.
- ``mid``: 101 to 400 changed lines, or a scoring change.
- ``high``: a protected file or public-surface change, or more than 400 changed lines.

Changed lines are git's added plus removed lines outside docs, lockfiles and JSONL data. Protected files are the
globs listed in ``protected_files.yaml`` next to this file. The public surface is each ``nemo_gym`` package
``__init__.py`` that assigns ``_PUBLIC_API`` or ``_LAZY_EXPORTS`` at the top level, plus both ``<submodule>.py`` and
``<submodule>/__init__.py`` for every submodule any such assignment lists. That rule is deliberately coarse: it does
not model which mapping the loader ends up using, so it can over-count but does not miss a listed file. Scoring is the
resources servers and the verification, verifiers and metrics packages, outside their tests and data.
"""

import argparse
import ast
import os
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path


LOW_MAX_LINES = 100
MID_MAX_LINES = 400
DOC_PATTERNS = ("*.md", "*.mdx", "*.rst", "docs/*", "fern/*", "LICENSE", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg")
UNCOUNTED_PATTERNS = ("*.lock", "*.jsonl")
SCORING_PREFIXES = ("resources_servers/", "nemo_gym/verification/", "nemo_gym/verifiers/", "nemo_gym/metrics/")
NON_SCORING_DIRS = {"tests", "data"}
EXPORT_TABLES = {"_PUBLIC_API", "_LAZY_EXPORTS"}
PROTECTED_FILES = Path(__file__).with_name("protected_files.yaml")


def matches(path: str, patterns: tuple[str, ...]) -> bool:
    """Return whether ``path`` or its file name matches a glob in ``patterns``."""
    return any(fnmatch(path, pattern) or fnmatch(Path(path).name, pattern) for pattern in patterns)


def git(tree: Path, *args: str) -> str:
    """Run git in ``tree`` and return its stdout."""
    return os.fsdecode(subprocess.run(["git", "-C", str(tree), *args], check=True, capture_output=True).stdout)


def changed_lines(base: Path, head: Path) -> dict[str, int]:
    """Return each path changed from ``base``'s commit to ``head``'s and its added plus removed lines.

    The diff runs in ``base`` so the target branch's ``.gitattributes`` apply: a pull request that marks its own
    files ``-diff`` or ``binary`` cannot hide their lines.
    """
    head_commit = git(head, "rev-parse", "HEAD").strip()
    stats = {}
    for record in git(base, "diff", "--numstat", "-z", "--no-renames", "HEAD", head_commit).split("\0"):
        if record:
            added, removed, path = record.split("\t", 2)
            stats[path] = 0 if added == "-" else int(added) + int(removed)
    return stats


def public_surface(tree: Path) -> set[str]:
    """Return each package initializer in ``tree`` that assigns an export table and every submodule it lists.

    Only regular files are read, so an initializer symlinked to a FIFO or device cannot stall grading.
    """
    files = set()
    for init in (tree / "nemo_gym").rglob("__init__.py"):
        if not init.is_file():
            continue
        try:
            body = ast.parse(init.read_bytes()).body
        except (OSError, SyntaxError, ValueError):
            continue
        package = init.parent.relative_to(tree).as_posix()
        for node in body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue
            if not any(isinstance(target, ast.Name) and target.id in EXPORT_TABLES for target in targets):
                continue
            files.add(f"{package}/__init__.py")
            for value in node.value.values if isinstance(node.value, ast.Dict) else []:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    module = f"{package}/{value.value.replace('.', '/')}"
                    files |= {f"{module}.py", f"{module}/__init__.py"}
    return files


def protected_patterns(path: Path = PROTECTED_FILES) -> list[str]:
    """Return the globs listed as ``- <glob>`` lines in ``path``, or none when it is absent."""
    if not path.is_file():
        return []
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line[2:].strip().strip("'\"") for line in lines if line.startswith("- ")]


def is_scoring(path: str) -> bool:
    """Return whether ``path`` is reward-computing code rather than its tests or data."""
    return path.startswith(SCORING_PREFIXES) and not NON_SCORING_DIRS & set(Path(path).parts[:-1])


def grade(base: Path, head: Path, protected: list[str]) -> tuple[str, list[str]]:
    """Return the tier of the change from ``base`` to ``head`` and the evidence behind it."""
    changed = changed_lines(base, head)
    code = [path for path in changed if not matches(path, DOC_PATTERNS)]
    lines = sum(changed[path] for path in code if not matches(path, UNCOUNTED_PATTERNS))
    public = public_surface(base) | public_surface(head)
    reasons = [f"protected: {path}" for path in changed if any(fnmatch(path, glob) for glob in protected)]
    reasons += [f"public surface: {path}" for path in changed if path in public]
    scoring = [f"scoring: {path}" for path in code if is_scoring(path)]
    evidence = [f"{lines} changed lines in {len(code)} non-doc files", *reasons, *scoring]
    if reasons or lines > MID_MAX_LINES:
        return "high", evidence
    if scoring or lines > LOW_MAX_LINES:
        return "mid", evidence
    return "low", evidence


def main() -> None:
    """Print the tier of ``--base`` to ``--head`` on stdout and its evidence on stderr."""
    parser = argparse.ArgumentParser(description="Grade a pull request's risk as low, mid or high.")
    parser.add_argument("--base", type=Path, required=True, help="git worktree of the merge base")
    parser.add_argument("--head", type=Path, required=True, help="git worktree of the pull request head")
    args = parser.parse_args()
    tier, evidence = grade(args.base, args.head, protected_patterns())
    print(tier)
    print("\n".join(evidence), file=sys.stderr)


if __name__ == "__main__":
    main()
