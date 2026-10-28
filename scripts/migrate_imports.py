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
"""Rewrite old `nemo_gym` import paths to the paths that replaced them.

The moves come from `nemo_gym/_moved_paths.py` in the checkout that holds this script.
Run it on an open branch, or on a downstream project, after updating to a Gym checkout that includes the moves:

    python scripts/migrate_imports.py <file or directory> ...

In `.py` files it rewrites:

- `from nemo_gym.old import a, b`, splitting the statement when its names moved to different modules.
- `from nemo_gym import old_module`.
- `from nemo_gym.old import a, b` written as text on one line, such as a code sample in docs or a template in a string.
- Every other dotted mention of an old path.
  That covers `import nemo_gym.old`, attribute chains such as `nemo_gym.old.name`, `mock.patch` targets, and names in config strings such as `"nemo_gym.old:Class"`.

In every file it also rewrites filesystem paths of moved modules, such as `nemo_gym/old.py` and `nemo_gym/old/`, as CODEOWNERS, CI path filters, and docs spell them.
Docs, config, and CI files (`.md`, `.mdx`, `.yaml`, `.yml`, `.toml`, `.json`, `.sh`, `.txt`, `.cfg`, `.ini`, `CODEOWNERS`, `Dockerfile`, `Makefile`) get only those two text rewrites.
Relative imports that reach a moved module are reported, not rewritten, and so is a `monkeypatch.setattr` or `patch.object` that patches a split-out name through its old module, because that patch reaches only code that still looks the name up there.
Run `ruff check --fix` and `ruff format` on the changed files afterwards to sort the rewritten imports.
"""

import argparse
import ast
import re
import sys
import warnings
from pathlib import Path
from typing import Iterable, Mapping, Optional


# Files that name old paths on purpose: the move list itself, the tests of old paths, and the docs redirects, whose
# sources are the URLs of moved modules.
KEEP_OLD_PATHS = (
    "fern/docs.yml",
    "nemo_gym/_moved_paths.py",
    "scripts/migrate_imports.py",
    "tests/unit_tests/fixtures/import_surface.json",
    "tests/unit_tests/test_migrate_imports.py",
    "tests/unit_tests/test_moved_paths.py",
)
# Frozen docs for released versions describe the paths those releases shipped with.
RELEASED_DOCS = re.compile(r"(^|/)fern/versions/v\d[^/]*/")
PATCH_THROUGH_MODULE = re.compile(
    r"(?:setattr|patch\.object)\(\s*(?P<module>[A-Za-z_][\w.]*)\s*,\s*[\"'](?P<name>\w+)[\"']"
)
TEXT_SUFFIXES = {".md", ".mdx", ".yaml", ".yml", ".toml", ".json", ".sh", ".txt", ".cfg", ".ini"}
TEXT_NAMES = {"CODEOWNERS", "Dockerfile", "Makefile"}
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".ruff_cache", ".pytest_cache"}


def _load_moves() -> tuple[dict[str, str], dict[str, dict[str, tuple[str, str]]]]:
    # Read the moves from the checkout this script belongs to, not from whichever nemo_gym is installed.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nemo_gym._moved_paths import MOVED_MODULES, MOVED_NAMES

    modules = {old: moved.new for old, moved in MOVED_MODULES.items()}
    names = {}
    for module, moved_names in MOVED_NAMES.items():
        names[module] = {}
        for name, moved in moved_names.items():
            new_module, _, new_name = moved.new.partition(":")
            names[module][name] = (new_module, new_name or name)
    return modules, names


def _chars(line: str, byte_offset: int) -> int:
    """Convert an `ast` column, counted in UTF-8 bytes, to a character offset in `line`."""
    return len(line.encode()[:byte_offset].decode(errors="ignore"))


class Migrator:
    """Rewrites old paths given module moves (`old -> new`) and name moves (`module -> {name -> (module, name)}`)."""

    def __init__(self, modules: Mapping[str, str], names: Mapping[str, Mapping[str, tuple[str, str]]]) -> None:
        self.modules = dict(modules)
        self.names = {module: dict(moved) for module, moved in names.items()}
        dotted = {old: new for old, new in self.modules.items()}
        for module, moved in self.names.items():
            for name, (new_module, new_name) in moved.items():
                dotted[f"{module}.{name}"] = f"{new_module}.{new_name}"
        # Longest first, so a moved submodule or name wins over its moved parent.
        self._dotted = dotted
        alternatives = "|".join(re.escape(old) for old in sorted(dotted, key=len, reverse=True))
        self._dotted_re = re.compile(rf"(?<![\w.])({alternatives})(?!\w)") if dotted else None
        # Filesystem paths of moved modules and packages, as CODEOWNERS, CI path filters, and docs spell them.
        slashed = {old.replace(".", "/"): new.replace(".", "/") for old, new in self.modules.items()}
        alternatives = "|".join(re.escape(old) for old in sorted(slashed, key=len, reverse=True))
        self._slashed = slashed
        # Library reference URLs (`.../nemo-gym/nemo_gym/old`) are left alone: fern/docs.yml redirects them.
        self._slashed_re = (
            re.compile(rf"(?<![\w.])(?<!nemo-gym/)({alternatives})(?=\.py\b|/|(?!\w|\.\w))") if slashed else None
        )
        # One-line `from <module> import <names>` written as text, for modules that names were split out of.
        alternatives = "|".join(re.escape(module) for module in sorted(self.names, key=len, reverse=True))
        self._text_import_re = (
            re.compile(
                rf"^(?P<indent>[ \t]*)from (?P<module>{alternatives}) import (?P<names>[^\n#()\\]+?)(?P<comment>[ \t]*#[^\n]*)?$",
                re.M,
            )
            if self.names
            else None
        )
        # `monkeypatch.setattr(<old module>, "<name>", ...)` and `patch.object(...)` through a module that names left.
        self._moved_name_owner = {name: module for module, moved in self.names.items() for name in moved}

    def module_path(self, module: str) -> str:
        """The current path of `module`, following module moves (longest moved prefix wins)."""
        old = module
        while old:
            if old in self.modules:
                return self.modules[old] + module[len(old) :]
            old = old.rpartition(".")[0]
        return module

    def rewrite_dotted(self, text: str) -> str:
        """Rewrite dotted mentions (`nemo_gym.old.name`) and filesystem paths (`nemo_gym/old.py`) of moved code.

        One-line `from <module> import <names>` text, such as a code sample in docs or a template in a string, is rewritten too.
        """
        if self._text_import_re is not None:
            text = self._text_import_re.sub(self._rewrite_text_import, text)
        if self._dotted_re is not None:
            text = self._dotted_re.sub(lambda match: self._dotted[match.group(1)], text)
        if self._slashed_re is not None:
            text = self._slashed_re.sub(lambda match: self._slashed[match.group(1)], text)
        return text

    def _destinations(self, module: str, names: list[tuple[str, Optional[str]]]) -> dict[str, list[str]]:
        """Group `(name, asname)` pairs imported from `module` by the module each now lives in."""
        groups: dict[str, list[str]] = {}
        for name, asname in names:
            if name in self.names.get(module, {}):
                target_module, target_name = self.names[module][name]
            elif name == "*":
                target_module, target_name = self.module_path(module), name
            else:
                # Covers `from nemo_gym import old_module` and names of a module that moved whole.
                target_module, _, target_name = self.module_path(f"{module}.{name}").rpartition(".")
            spelled = target_name if target_name == name and asname is None else f"{target_name} as {asname or name}"
            groups.setdefault(target_module, []).append(spelled)
        return groups

    def _rewrite_import_from(self, node: ast.ImportFrom, statement: str) -> Optional[str]:
        """Rewrite one `from nemo_gym... import ...` statement, keeping its formatting when nothing splits."""
        module = node.module
        groups = self._destinations(module, [(alias.name, alias.asname) for alias in node.names])
        original = [alias.name if alias.asname is None else f"{alias.name} as {alias.asname}" for alias in node.names]
        if len(groups) == 1:
            [(target, spelled)] = groups.items()
            if spelled == original:
                if target == module:
                    return None
                # Every name goes to one module under its own name: change only the module path.
                return re.sub(rf"^from\s+{re.escape(module)}\b", f"from {target}", statement, count=1)
        indent = " " * node.col_offset
        lines = [f"from {target} import {', '.join(spelled)}" for target, spelled in groups.items()]
        return f"\n{indent}".join(lines)

    def _rewrite_text_import(self, match: re.Match) -> str:
        """Rewrite a one-line `from <module> import <names>` written as text, such as a code sample in docs."""
        module = match.group("module")
        names = []
        for part in match.group("names").split(","):
            name, _, asname = part.strip().partition(" as ")
            if not name.isidentifier() or (asname and not asname.strip().isidentifier()):
                return match.group(0)
            names.append((name, asname.strip() or None))
        groups = self._destinations(module, names)
        if list(groups) == [module]:
            return match.group(0)
        indent, comment = match.group("indent"), match.group("comment") or ""
        lines = [f"{indent}from {target} import {', '.join(spelled)}" for target, spelled in groups.items()]
        return "\n".join(lines) + comment

    def rewrite_python(self, source: str, *, path: str = "<source>") -> tuple[str, list[str]]:
        """Return the rewritten source and notes about relative imports into moved modules."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)  # invalid escapes in files being rewritten
            tree = ast.parse(source)
        lines = source.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        edits, notes = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                if any(self.module_path(f"nemo_gym.{alias.name}") != f"nemo_gym.{alias.name}" for alias in node.names):
                    notes.append(f"{path}:{node.lineno}: relative import may reach a moved module; check by hand")
                continue
            # `import nemo_gym.old` needs no statement rewrite: the dotted-path pass below covers it.
            if not (isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "nemo_gym"):
                continue
            start = offsets[node.lineno - 1] + _chars(lines[node.lineno - 1], node.col_offset)
            end = offsets[node.end_lineno - 1] + _chars(lines[node.end_lineno - 1], node.end_col_offset)
            new = self._rewrite_import_from(node, source[start:end])
            if new is not None:
                edits.append((start, end, new))
        for match in PATCH_THROUGH_MODULE.finditer(source):
            owner = self._moved_name_owner.get(match.group("name"))
            if owner is not None and owner.rpartition(".")[2] == match.group("module").rpartition(".")[2]:
                line = source.count("\n", 0, match.start()) + 1
                new_module, _ = self.names[owner][match.group("name")]
                notes.append(
                    f"{path}:{line}: patches `{match.group('name')}` through `{match.group('module')}`, but it moved to "
                    f"`{new_module}`; the patch reaches only code that still looks it up in the old module, so check by hand"
                )
        for start, end, new in sorted(edits, reverse=True):
            source = source[:start] + new + source[end:]
        # Everything else: attribute chains, patch targets, and dotted strings.
        # Rewritten import statements hold only new paths, which no old path matches.
        return self.rewrite_dotted(source), notes


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file():
            yield path
            continue
        for child in sorted(path.rglob("*")):
            if child.is_file() and not SKIP_DIRS.intersection(child.parts):
                yield child


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--check", action="store_true", help="Report files that would change; change nothing.")
    args = parser.parse_args(argv)
    migrator = Migrator(*_load_moves())
    changed = []
    for path in _iter_files(args.paths):
        if path.resolve().as_posix().endswith(KEEP_OLD_PATHS) or RELEASED_DOCS.search(path.as_posix()):
            continue
        if path.suffix == ".py":
            try:
                source = path.read_text()
                new, notes = migrator.rewrite_python(source, path=str(path))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for note in notes:
                print(note, file=sys.stderr)
        elif path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES:
            try:
                source = path.read_text()
            except UnicodeDecodeError:
                continue
            new = migrator.rewrite_dotted(source)
        else:
            continue
        if new != source:
            changed.append(path)
            if not args.check:
                path.write_text(new)
    for path in changed:
        print(path)
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
