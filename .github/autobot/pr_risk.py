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
"""Grade a pull request's risk as ``low``, ``mid`` or ``high`` from its merge-base and head trees.

The CI-Events ``pr-risk-label`` agent always runs this file from the pull request's target branch tip, never from
the merge base or the head, on the change from the merge base to the head:

    python3 <target>/.github/autobot/pr_risk.py --base <merge-base> --head <head>

It prints exactly one tier on stdout and its evidence on stderr. The agent may raise the tier after reading the code,
never lower it. The tiers:

- ``low``: only docs changed, or at most 100 changed lines with no public-surface or scoring impact.
- ``mid``: 101 to 400 changed lines, or a scoring change, with no public-surface impact.
- ``high``: any public-surface change, or more than 400 changed lines.

Changed lines are added plus removed lines outside docs, lockfiles, JSONL data and binary files. The public surface is
each name a ``nemo_gym`` package exports through ``_PUBLIC_API`` and that name's signature, plus the dependencies,
entry points and Python requirement in the root ``pyproject.toml``. Scoring is the code that computes rewards:
resources servers and the shared verification, verifiers and metrics packages, excluding their tests and data.

Standard library only, so it runs on Python 3.12 or newer without the repository installed.
"""

import argparse
import ast
import difflib
import filecmp
import os
import sys
import tomllib
from collections.abc import Iterator
from fnmatch import fnmatch
from functools import cache
from pathlib import Path


LOW_MAX_LINES = 100
MID_MAX_LINES = 400
SKIPPED_NAMES = frozenset({".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".ruff_cache"})
DOC_PATTERNS = ("*.md", "*.mdx", "*.rst", "docs/*", "fern/*", "LICENSE", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg")
UNCOUNTED_PATTERNS = ("*.lock", "*.jsonl")
SCORING_PREFIXES = ("resources_servers/", "nemo_gym/verification/", "nemo_gym/verifiers/", "nemo_gym/metrics/")
NON_SCORING_DIRS = frozenset({"tests", "data"})
PACKAGING_KEYS = ("dependencies", "optional-dependencies", "scripts", "requires-python")
MAX_LISTED = 10


def walk(root: Path) -> Iterator[str]:
    """Yield each file and symlink under ``root`` as a POSIX path relative to it, skipping VCS and cache entries."""
    for directory, dirnames, filenames in os.walk(root):
        links = [name for name in dirnames if os.path.islink(os.path.join(directory, name))]
        dirnames[:] = [name for name in dirnames if name not in SKIPPED_NAMES and name not in links]
        for name in (*filenames, *links):
            if name not in SKIPPED_NAMES:
                yield (Path(directory) / name).relative_to(root).as_posix()


def same_file(before: Path, after: Path) -> bool:
    """Return whether two files have the same content, or two symlinks the same target."""
    if before.is_symlink() or after.is_symlink():
        return before.is_symlink() and after.is_symlink() and os.readlink(before) == os.readlink(after)
    return filecmp.cmp(before, after, shallow=False)


def changed_files(base: Path, head: Path) -> list[str]:
    """Return the sorted paths that were added, removed or modified between the two trees."""
    base_files, head_files = set(walk(base)), set(walk(head))
    modified = {path for path in base_files & head_files if not same_file(base / path, head / path)}
    return sorted((base_files ^ head_files) | modified)


def matches(path: str, patterns: tuple[str, ...]) -> bool:
    """Return whether ``path`` or its file name matches a glob in ``patterns``; ``*`` also matches ``/``."""
    return any(fnmatch(path, pattern) or fnmatch(Path(path).name, pattern) for pattern in patterns)


def read_lines(path: Path) -> list[str] | None:
    """Return a file's lines, a symlink's target, ``[]`` when absent, or ``None`` when the file is binary."""
    if path.is_symlink():
        return [os.readlink(path)]
    if not path.is_file():
        return []
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace").splitlines()


def changed_line_count(base: Path, head: Path, path: str) -> int:
    """Count one file's added plus removed lines; a binary file counts as zero."""
    before, after = read_lines(base / path), read_lines(head / path)
    if before is None or after is None:
        return 0
    opcodes = difflib.SequenceMatcher(None, before, after).get_opcodes()
    return sum(i2 - i1 + j2 - j1 for tag, i1, i2, j1, j2 in opcodes if tag != "equal")


def is_scoring(path: str) -> bool:
    """Return whether ``path`` is reward-computing code rather than its tests, data or docs."""
    in_support_dir = bool(set(Path(path).parts[:-1]) & NON_SCORING_DIRS)
    return path.startswith(SCORING_PREFIXES) and not in_support_dir and not matches(path, DOC_PATTERNS)


@cache
def parse(path: Path) -> ast.Module | None:
    """Parse a Python file once, or return ``None`` when it is absent or not valid Python."""
    try:
        return ast.parse(path.read_bytes())
    except (OSError, SyntaxError, ValueError):
        return None


def assigns(node: ast.stmt, name: str) -> bool:
    """Return whether a statement assigns a plain ``name``."""
    if isinstance(node, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    return isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name


def public_api(tree: Path) -> dict[tuple[str, str], str]:
    """Map each ``(package, name)`` that a ``nemo_gym`` package lists in ``_PUBLIC_API`` to its submodule."""
    exports = {}
    for init in sorted((tree / "nemo_gym").rglob("__init__.py")):
        package = ".".join(init.parent.relative_to(tree).parts)
        module = parse(init)
        for node in module.body if module else []:
            if assigns(node, "_PUBLIC_API") and isinstance(node.value, ast.Dict):
                for key, value in zip(node.value.keys, node.value.values, strict=True):
                    if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                        exports[(package, key.value)] = value.value
    return exports


def module_path(tree: Path, module: str) -> Path:
    """Return the file that defines a dotted module inside ``tree``."""
    stem = tree.joinpath(*module.split("."))
    return stem / "__init__.py" if stem.is_dir() else stem.with_suffix(".py")


def is_public_member(node: ast.stmt) -> bool:
    """Return whether a class-body statement is part of the class's public surface."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return not node.name.startswith("_") or node.name == "__init__"
    targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
    return isinstance(node, ast.Assign | ast.AnnAssign) and not any(
        isinstance(target, ast.Name) and target.id.startswith("_") for target in targets
    )


def signature(node: ast.stmt) -> str:
    """Fingerprint a definition's contract: a function's signature, a class's surface or a constant's value."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        returns = ast.dump(node.returns) if node.returns else "None"
        return f"{type(node).__name__} {node.name}({ast.dump(node.args)}) -> {returns}"
    if isinstance(node, ast.ClassDef):
        header = [ast.dump(part) for part in (*node.decorator_list, *node.bases, *node.keywords)]
        members = [signature(item) for item in node.body if is_public_member(item)]
        return "\n".join([f"class {node.name}", *header, *members])
    return ast.dump(node)


def binds(node: ast.stmt, name: str) -> bool:
    """Return whether a top-level statement binds ``name``."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.TypeAlias):
        return getattr(node.name, "id", node.name) == name
    if isinstance(node, ast.ImportFrom):
        return any((alias.asname or alias.name) == name for alias in node.names)
    return assigns(node, name)


def export_signature(tree: Path, module: str, name: str, hops: int = 3) -> str:
    """Fingerprint ``name`` as defined in ``module``, following re-exports from other ``nemo_gym`` modules."""
    parsed = parse(module_path(tree, module))
    for node in parsed.body if parsed else []:
        if not binds(node, name):
            continue
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("nemo_gym.") and hops:
            original = next(alias.name for alias in node.names if (alias.asname or alias.name) == name)
            return export_signature(tree, node.module, original, hops - 1)
        return signature(node)
    return f"undefined {module}.{name}"


def public_api_changes(base: Path, head: Path, changed: list[str]) -> list[str]:
    """Describe each exported name that was added, removed or changed its signature."""
    before, after = public_api(base), public_api(head)
    reasons = [f"export added: {package}.{name}" for package, name in sorted(after.keys() - before.keys())]
    reasons += [f"export removed: {package}.{name}" for package, name in sorted(before.keys() - after.keys())]
    if not any(path.startswith("nemo_gym/") and path.endswith(".py") for path in changed):
        return reasons
    for package, name in sorted(before.keys() & after.keys()):
        old = export_signature(base, f"{package}.{before[(package, name)]}", name)
        new = export_signature(head, f"{package}.{after[(package, name)]}", name)
        if old != new:
            reasons.append(f"export changed: {package}.{name}")
    return reasons


def packaging(tree: Path) -> dict[str, object]:
    """Return the user-facing packaging contract from the root ``pyproject.toml``."""
    try:
        project = tomllib.loads((tree / "pyproject.toml").read_text()).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return {key: project.get(key) for key in PACKAGING_KEYS}


def packaging_changes(base: Path, head: Path, changed: list[str]) -> list[str]:
    """Describe each changed packaging field of the root ``pyproject.toml``."""
    if "pyproject.toml" not in changed:
        return []
    before, after = packaging(base), packaging(head)
    return [f"pyproject.toml project.{key} changed" for key in PACKAGING_KEYS if before.get(key) != after.get(key)]


def grade(base: Path, head: Path) -> tuple[str, list[str]]:
    """Return the tier of the change from ``base`` to ``head`` and the evidence behind it."""
    changed = changed_files(base, head)
    code = [path for path in changed if not matches(path, DOC_PATTERNS)]
    if not code:
        return "low", [f"no changes outside docs ({len(changed)} doc files)"]
    lines = sum(changed_line_count(base, head, path) for path in code if not matches(path, UNCOUNTED_PATTERNS))
    surface = public_api_changes(base, head, changed) + packaging_changes(base, head, changed)
    scoring = [path for path in code if is_scoring(path)]
    evidence = [f"{lines} changed lines in {len(code)} non-doc files", *surface[:MAX_LISTED]]
    evidence += [f"scoring: {path}" for path in scoring[:MAX_LISTED]]
    if surface or lines > MID_MAX_LINES:
        return "high", evidence
    if scoring or lines > LOW_MAX_LINES:
        return "mid", evidence
    return "low", evidence


def main(argv: list[str] | None = None) -> int:
    """Print the tier of ``--base`` to ``--head`` on stdout and its evidence on stderr."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True, help="the merge-base tree")
    parser.add_argument("--head", type=Path, required=True, help="the pull request head tree")
    args = parser.parse_args(argv)
    for tree in (args.base, args.head):
        if not tree.is_dir():
            parser.error(f"{tree} is not a directory")
    tier, evidence = grade(args.base.resolve(), args.head.resolve())
    print(tier)
    print("\n".join(evidence), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
