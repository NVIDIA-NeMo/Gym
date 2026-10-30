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
"""Each package's public API: names servers import from ``nemo_gym.<package>``."""

import ast
import importlib
import importlib.util
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
from scripts.migrate_imports import KEEP_LINE_MARKER


REPO = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO / "nemo_gym"
# Every package that declares a public API.
PACKAGES = sorted(path.parent.name for path in PACKAGE_DIR.glob("*/__init__.py") if "_PUBLIC_API" in path.read_text())


def _public_api(package: str) -> dict[str, str]:
    return getattr(importlib.import_module(f"nemo_gym.{package}"), "_PUBLIC_API", {})


PUBLIC_NAMES = [(package, name) for package in PACKAGES for name in _public_api(package)]


@pytest.mark.parametrize("package", PACKAGES)
def test_each_public_name_is_the_object_its_submodule_defines(package: str) -> None:
    module = importlib.import_module(f"nemo_gym.{package}")
    for name, submodule in _public_api(package).items():
        assert getattr(module, name) is getattr(importlib.import_module(f"nemo_gym.{package}.{submodule}"), name)
        assert name in module.__all__
        assert name in dir(module)


@pytest.mark.parametrize("package", PACKAGES)
def test_importing_a_package_loads_none_of_its_lazy_submodules(package: str) -> None:
    # Run in a fresh interpreter: this process has imported most submodules already.
    lazy = sorted(set(_public_api(package).values()))
    code = (
        "import sys, nemo_gym.{p}; print(sorted(m for m in {lazy!r} if f'nemo_gym.{p}.' + m in sys.modules))"
    ).format(p=package, lazy=lazy)
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert loaded.strip() == "[]"


def test_packages_declare_public_apis() -> None:
    assert PACKAGES


@pytest.mark.parametrize("package", PACKAGES)
def test_an_unknown_name_is_an_attribute_error(package: str) -> None:
    with pytest.raises(AttributeError, match="has no attribute 'not_a_public_name'"):
        importlib.import_module(f"nemo_gym.{package}").not_a_public_name  # noqa: B018


@pytest.mark.parametrize(("package", "name"), PUBLIC_NAMES)
def test_a_patch_of_the_submodule_shows_through_the_package(
    package: str, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    replacement = object()
    monkeypatch.setattr(importlib.import_module(f"nemo_gym.{package}.{_public_api(package)[name]}"), name, replacement)
    assert getattr(importlib.import_module(f"nemo_gym.{package}"), name) is replacement


@pytest.mark.parametrize("package", PACKAGES)
def test_type_checking_imports_match_the_public_api(package: str) -> None:
    # Type checkers read the TYPE_CHECKING imports; the package serves _PUBLIC_API. A name in one but not the
    # other is Any to a type checker, or invisible to it.
    tree = ast.parse((PACKAGE_DIR / package / "__init__.py").read_text())
    block = next(node for node in tree.body if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING")
    prefix = f"nemo_gym.{package}."
    declared = {
        alias.asname or alias.name: node.module.removeprefix(prefix)
        for node in block.body
        if isinstance(node, ast.ImportFrom) and node.module.startswith(prefix)
        for alias in node.names
    }
    assert declared == _public_api(package)


def _tracked_python(top: str) -> list[Path]:
    tracked = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO, capture_output=True, text=True, check=True)
    return [Path(line) for line in tracked.stdout.splitlines() if (Path(line).parts[0] == "tests") == (top == "tests")]


def _code_outside_the_library() -> list[Path]:
    """Tracked Python files outside nemo_gym/ and its own tests (tests/): servers, their tests, scripts."""
    return [REPO / path for path in _tracked_python("outside") if path.parts[0] != "nemo_gym"]


def test_the_librarys_tests_import_from_the_module_that_defines_each_name() -> None:
    # tests/ imports like nemo_gym/: from the defining module, which is also what it patches.
    found = []
    for path in _tracked_python("tests"):
        try:
            tree = ast.parse((REPO / path).read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module in {f"nemo_gym.{p}" for p in PACKAGES}
            ):
                continue
            for alias in node.names:
                if importlib.util.find_spec(f"{node.module}.{alias.name}") is None:
                    found.append(f"{path}:{node.lineno} imports {alias.name} from {node.module}")
    assert not found, f"{found} should import from the module that defines the name."


def test_code_outside_the_library_imports_exported_names_from_the_public_api() -> None:
    exported = {
        f"nemo_gym.{package}.{submodule}.{name}": package
        for package, name in PUBLIC_NAMES
        for submodule in [_public_api(package)[name]]
    }
    found = []
    for path in _code_outside_the_library():
        source = path.read_text(errors="replace")
        lines = source.splitlines()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module):
                continue
            if KEEP_LINE_MARKER in "\n".join(lines[node.lineno - 1 : node.end_lineno]):
                continue  # an old-layout fallback for a sandbox that runs an older nemo_gym
            for alias in node.names:
                package = exported.get(f"{node.module}.{alias.name}")
                if package is not None:
                    found.append(f"{path.relative_to(REPO)}:{node.lineno} imports {alias.name} from {node.module}")
    assert not found, f"{found} should import from the package's public API: from nemo_gym.<package> import Name."


def test_a_submodule_failing_with_attribute_error_reports_the_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "facade_probe"
    package.mkdir()
    (package / "__init__.py").write_text(
        "from nemo_gym._public_api import lazy_exports\n"
        "__getattr__, __dir__ = lazy_exports(__name__, {'Thing': 'broken'})\n"
    )
    (package / "broken.py").write_text("import types\n\ntypes.no_such_attribute\nThing = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "facade_probe", raising=False)
    with pytest.raises(ImportError, match="importing facade_probe.broken failed: .*no_such_attribute") as raised:
        exec("from facade_probe import Thing")
    assert isinstance(raised.value.__cause__, AttributeError)
    sys.modules.pop("facade_probe.broken", None)
    sys.modules.pop("facade_probe", None)


def test_each_root_lazy_export_names_the_module_that_defines_it() -> None:
    # A stale target still resolves through the old path's module object, so check the module's own namespace.
    import nemo_gym

    stale = []
    for name, module in nemo_gym._LAZY_EXPORTS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name not in vars(importlib.import_module(f"nemo_gym.{module}")):
                stale.append(f"{name}: nemo_gym.{module}")
    assert not stale, f"These nemo_gym._LAZY_EXPORTS entries name a module that does not define the name: {stale}"
