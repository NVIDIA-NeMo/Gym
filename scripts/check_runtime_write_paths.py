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
"""Fail on server code that would write into the Gym source checkout while it runs.

A server starts with its own source folder as the working directory.
These therefore resolve inside the checkout:

- ``cwd``: a call to ``Path.cwd()`` or ``os.getcwd()``.
- ``relative-default``: a setting or parameter named ``*_dir`` or ``*_root`` whose default is a relative path.
  For example ``artifacts_dir: Path = Path("results/x")`` or ``log_dir: str = "cache/x"``.
- ``yaml-relative-default``: a ``*_dir`` or ``*_root`` key in a YAML config whose value is a relative path.
  An ``${oc.env:...}`` or ``${oc.select:...}`` interpolation counts by its fallback, which applies when nothing is set.
  A reference to another key, such as ``${results_dir}/x``, counts as absolute.

Use nemo_gym/core/runtime_files.py instead.
A path that is only read, such as a bundled prompt, can say so with a ``# runtime-path: read`` comment.
The comment goes on the flagged line, in Python or YAML.
Sites that predate this check are listed in scripts/runtime_write_paths_allowlist.txt as ``<file> <rule> <scope>``.
The scope is the enclosing class and function, or the setting's name.
The check also fails on a listed site that no longer exists, so the list only shrinks.

Usage: python scripts/check_runtime_write_paths.py [--root DIR] [--allowlist FILE] [--write-allowlist]
"""

import argparse
import ast
import re
import sys
import warnings
from pathlib import Path

import yaml


SCANNED_DIRS = ("nemo_gym", "resources_servers", "responses_api_agents", "responses_api_models", "environment_servers")
# The CLI runs in the user's own working directory, where relative paths are what the user means.
EXCLUDED_PREFIXES = ("nemo_gym/cli/",)
# Benchmark and environment configs start the servers above, so their folder settings count too.
SCANNED_YAML_DIRS = (*SCANNED_DIRS, "benchmarks", "environments")
# Keys under these run only while preparing a dataset, by hand, not inside a server.
EXCLUDED_YAML_KEYS = ("prepare_script_args",)
# The innermost interpolation of a value: no nested `${` and no closing brace inside it.
_INTERPOLATION = re.compile(r"\$\{([^${}]*)\}")
DEFAULT_ALLOWLIST = Path("scripts/runtime_write_paths_allowlist.txt")
_PATH_SUFFIXES = ("_dir", "_root")
READ_MARKER = "runtime-path: read"


def _is_excluded(relative: str) -> bool:
    parts = relative.split("/")
    name = parts[-1]
    return (
        relative.startswith(EXCLUDED_PREFIXES)
        or "tests" in parts
        or "scripts" in parts
        or name == "conftest.py"
        or name.startswith("test_")
        # Dataset preparation runs once, by hand, not inside a server.
        or name.startswith(("prepare_", "create_", "generate_", "download_", "convert_"))
    )


def _is_relative_path_literal(node: ast.expr | None) -> bool:
    """Whether ``node`` is ``"rel/path"`` or ``Path("rel/path")``, including inside ``Field(default=...)``."""
    if node is None:
        return False
    if isinstance(node, ast.Constant):
        value = node.value
        return isinstance(value, str) and value != "" and not value.startswith(("/", "~", "$"))
    if isinstance(node, ast.Call):
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name in ("Path", "PurePath") and len(node.args) == 1:
            return _is_relative_path_literal(node.args[0])
        if name == "Field":
            default = node.args[0] if node.args else None
            for keyword in node.keywords:
                if keyword.arg == "default":
                    default = keyword.value
            return _is_relative_path_literal(default)
    return False


def _is_cwd_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == "getcwd" or (func.attr == "cwd" and not node.args)
    return isinstance(func, ast.Name) and func.id == "getcwd"


class _Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.findings: list[tuple[int, str, str]] = []

    def _scoped(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        arguments = node.args
        positional = arguments.posonlyargs + arguments.args
        pairs = list(zip(positional[len(positional) - len(arguments.defaults) :], arguments.defaults))
        pairs += [(arg, default) for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults)]
        for arg, default in pairs:
            if arg.arg.endswith(_PATH_SUFFIXES) and _is_relative_path_literal(default):
                self.findings.append((default.lineno, "relative-default", ".".join([*self.scope, node.name, arg.arg])))
        self._scoped(node)

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if (
            isinstance(node.target, ast.Name)
            and node.target.id.endswith(_PATH_SUFFIXES)
            and _is_relative_path_literal(node.value)
        ):
            self.findings.append((node.lineno, "relative-default", ".".join([*self.scope, node.target.id])))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_cwd_call(node):
            self.findings.append((node.lineno, "cwd", ".".join(self.scope) or "<module>"))
        self.generic_visit(node)


def scan(root: Path) -> list[tuple[str, int, str, str]]:
    """Return ``(file, line, rule, scope)`` for every finding under ``root``."""
    findings = []
    for directory in SCANNED_DIRS:
        for path in sorted((root / directory).rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if _is_excluded(relative) or "/.venv/" in f"/{relative}" or "/." in f"/{relative}":
                continue
            source = path.read_text(encoding="utf-8")
            try:
                with warnings.catch_warnings():
                    # Invalid escape sequences in the scanned code are not this check's concern.
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(source, filename=relative)
            except SyntaxError as error:
                findings.append((relative, error.lineno or 0, "syntax-error", str(error.msg)))
                continue
            visitor = _Visitor()
            visitor.visit(tree)
            lines = source.splitlines()
            findings.extend(
                (relative, line, rule, scope)
                for line, rule, scope in visitor.findings
                if READ_MARKER not in lines[line - 1]
            )
    return findings


def _interpolation_fallback(value: str) -> str:
    """The value with each interpolation replaced by its fallback, or by ``/`` when it names another key."""
    while (match := _INTERPOLATION.search(value)) is not None:
        resolver, _, arguments = match.group(1).partition(":")
        name, comma, default = arguments.partition(",")
        # Gym's config parser always sets these to absolute paths, so their fallback never applies.
        always_set = name.strip() in ("results_dir", "cache_dir")
        if resolver in ("oc.env", "oc.select") and comma and not always_set:
            replacement = default.strip().strip("'\"")
        else:
            replacement = "/"
        value = value[: match.start()] + replacement + value[match.end() :]
    return value


def _is_relative_yaml_path(node: yaml.Node) -> bool:
    if not isinstance(node, yaml.ScalarNode) or node.tag != "tag:yaml.org,2002:str":
        return False
    value = _interpolation_fallback(node.value).strip()
    return value not in ("", "null", "None", "???") and not value.startswith(("/", "~", "$"))


def _yaml_findings(node: yaml.Node, path: list[str], lines: list[str], findings: list[tuple[int, str]]) -> None:
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            key = str(key_node.value)
            if key in EXCLUDED_YAML_KEYS:
                continue
            line = value_node.start_mark.line
            if key.endswith(_PATH_SUFFIXES) and _is_relative_yaml_path(value_node) and READ_MARKER not in lines[line]:
                findings.append((line + 1, ".".join([*path, key])))
            _yaml_findings(value_node, [*path, key], lines, findings)
    elif isinstance(node, yaml.SequenceNode):
        for index, item in enumerate(node.value):
            _yaml_findings(item, [*path, str(index)], lines, findings)


def scan_yaml(root: Path) -> list[tuple[str, int, str, str]]:
    """Return ``(file, line, rule, scope)`` for every relative folder setting in the YAML configs under ``root``."""
    findings = []
    for directory in SCANNED_YAML_DIRS:
        for path in sorted((root / directory).rglob("*.yaml")):
            relative = path.relative_to(root).as_posix()
            if _is_excluded(relative) or "/." in f"/{relative}":
                continue
            source = path.read_text(encoding="utf-8")
            try:
                documents = list(yaml.compose_all(source))
            except yaml.YAMLError as error:
                findings.append((relative, 0, "syntax-error", type(error).__name__))
                continue
            lines = source.splitlines()
            for document in documents:
                found: list[tuple[int, str]] = []
                if document is not None:
                    _yaml_findings(document, [], lines, found)
                findings.extend((relative, line, "yaml-relative-default", scope) for line, scope in found)
    return findings


def read_allowlist(path: Path) -> set[tuple[str, str, str]]:
    entries = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            file, rule, scope = line.split(maxsplit=2)
            entries.add((file, rule, scope))
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--allowlist", type=Path, default=None)
    parser.add_argument("--write-allowlist", action="store_true", help="Rewrite the allowlist from the findings.")
    args = parser.parse_args(argv)
    allowlist_path = args.allowlist or args.root / DEFAULT_ALLOWLIST

    findings = scan(args.root) + scan_yaml(args.root)
    keys = {(file, rule, scope) for file, _, rule, scope in findings}
    if args.write_allowlist:
        header = allowlist_path.read_text(encoding="utf-8").split("\n\n", 1)[0] if allowlist_path.exists() else ""
        body = "\n".join(" ".join(key) for key in sorted(keys))
        allowlist_path.write_text(f"{header}\n\n{body}\n" if header else f"{body}\n", encoding="utf-8")
        return 0

    allowed = read_allowlist(allowlist_path) if allowlist_path.exists() else set()
    new = [(file, line, rule, scope) for file, line, rule, scope in findings if (file, rule, scope) not in allowed]
    stale = sorted(allowed - keys)
    for file, line, rule, scope in new:
        print(f"{file}:{line}: {rule} in {scope}: resolves against the server's source folder.")
    for file, rule, scope in stale:
        print(f"{allowlist_path}: '{file} {rule} {scope}' no longer matches any code; remove it from the list.")
    if new:
        print(
            "\nUse nemo_gym/core/runtime_files.py: scratch_dir for files needed only during a request, "
            "debug_artifact_dir for debug output, cache_dir for caches and results_dir for run output.",
            file=sys.stderr,
        )
    return 1 if new or stale else 0


if __name__ == "__main__":
    sys.exit(main())
