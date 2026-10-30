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
"""The import-linter contract sees every module of nemo_gym and every import between them.

A module outside every layer is not checked at all, so each module must be in a layer, or be listed here with the
reason it stays outside. Two kinds of import reach a module the contract does not check: a name re-exported by a
package that is in no layer itself, and an old path from the compatibility table. Code inside nemo_gym uses
neither.
"""

import ast
import tomllib
from pathlib import Path

from nemo_gym._moved_paths import MOVED_MODULES, MOVED_NAMES, _names_were_listed_under, _resolve


REPO = Path(__file__).resolve().parents[2]

OUTSIDE_THE_LAYERS = {
    "__init__": "the package root",
    "package_info": "version metadata",
    "_moved_paths": "the old-path compatibility table, which nemo_gym/__init__.py installs first",
    "verifiers": "standard-library-only helpers that verifiers running inside sandboxes import by name",
}


def _layer_modules() -> set[str]:
    contract = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["importlinter"]["contracts"][0]
    return {module.strip() for layer in contract["layers"] for module in layer.split(":")}


def _top_level_names() -> set[str]:
    package = REPO / "nemo_gym"
    return {path.stem for path in package.glob("*.py")} | {path.parent.name for path in package.glob("*/__init__.py")}


def _module_files() -> dict[str, Path]:
    """Every module under nemo_gym/ by dotted name; a package is named by its directory."""
    modules = {}
    for path in (REPO / "nemo_gym").rglob("*.py"):
        parts = path.relative_to(REPO).with_suffix("").parts
        modules[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    return modules


def _in_layer(module: str, layered: set[str]) -> bool:
    return any(module == layer or module.startswith(f"{layer}.") for layer in layered)


def _imports(module: str, path: Path) -> list[tuple[str, str | None, int]]:
    """`(module, name, line)` for each import in a file; `name` is None for `import module`."""
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found = []
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            found += [(alias.name, None, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package.split(".")[: len(package.split(".")) - node.level + 1] if node.level else []
            source = ".".join([*base, node.module] if node.module else base)
            found += [(source, alias.name, node.lineno) for alias in node.names]
    return found


def test_every_module_is_in_a_layer() -> None:
    layered = _layer_modules()
    outside = {f"nemo_gym.{name}" for name in OUTSIDE_THE_LAYERS if name != "__init__"}
    unlayered = sorted(
        module
        for module in _module_files()
        if not _in_layer(module, layered | outside)
        # A package listed as some of its submodules (the root, cli, sandbox, ...) is not checked itself.
        and not any(layer.startswith(f"{module}.") for layer in layered)
    )
    assert not unlayered, (
        f"{unlayered} are in no layer of the import-linter contract in pyproject.toml, so nothing checks their "
        "imports. Add each to the layer of the package it belongs to, or to OUTSIDE_THE_LAYERS with the reason."
    )


def test_every_module_kept_outside_the_layers_exists() -> None:
    assert not set(OUTSIDE_THE_LAYERS) - _top_level_names()


def test_nemo_gym_imports_no_name_through_a_package_outside_the_layers() -> None:
    """The contract records `from nemo_gym.cli import RunHelper` as an import of `nemo_gym.cli` alone.

    When that package is in no layer, the import of the module that defines the name is never checked.
    """
    layered = _layer_modules()
    modules = _module_files()
    unchecked = {
        package for package, path in modules.items() if path.name == "__init__.py" and not _in_layer(package, layered)
    }
    outside = {f"nemo_gym.{name}" for name in OUTSIDE_THE_LAYERS if name != "__init__"}
    defined_by = {}
    for package in unchecked:
        tree = ast.parse(modules[package].read_text())
        defined_by[package] = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        # Re-exports of modules outside the layers, such as `__version__`, hide nothing the contract checks.
        defined_by[package] |= {
            alias.asname or alias.name
            for source, node in ((s, n) for n in tree.body if isinstance(n, ast.ImportFrom) for s in [n.module])
            if source in outside
            for alias in node.names
        }
        defined_by[package] |= {
            target.id
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name)
        }
    through_package = sorted(
        f"{path.relative_to(REPO)}:{line} imports {name} from {source}"
        for module, path in modules.items()
        for source, name, line in _imports(module, path)
        if source in unchecked
        and name is not None
        and f"{source}.{name}" not in modules
        and name not in defined_by[source]
    )
    assert not through_package, (
        f"{through_package} import a name that a package outside the layers re-exports, so the contract never "
        "sees the module that defines it. Import it from that module."
    )


def test_nemo_gym_imports_nothing_through_an_old_path() -> None:
    """An old path has no file behind it, so the contract drops the import, or records it against the old module."""
    modules = _module_files()
    through_old_path = sorted(
        f"{path.relative_to(REPO)}:{line} imports {source if name is None else f'{source}.{name}'}"
        for module, path in modules.items()
        if module != "nemo_gym._moved_paths"
        for source, name, line in _imports(module, path)
        if source.startswith("nemo_gym")
        and (
            _resolve(source, MOVED_MODULES) is not None
            or (name is not None and _resolve(f"{source}.{name}", MOVED_MODULES) is not None)
            or any(name in MOVED_NAMES.get(old, {}) for old in _names_were_listed_under(source))
        )
    )
    assert not through_old_path, (
        f"{through_old_path} import through a path in nemo_gym/_moved_paths.py. Import from the new path, which "
        "`python scripts/migrate_imports.py <paths>` rewrites."
    )
