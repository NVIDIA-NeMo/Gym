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
"""Old import paths for code moved by the `nemo_gym` package reorganization.

Every move is recorded here, with the release it first ships in.
The old path keeps working until `removed_in`, which is `since` plus `DEPRECATION_WINDOW_MINOR_RELEASES` minor releases.
Each use of an old path emits a `DeprecationWarning` that names the new path, both releases, and the running version.
A unit test fails once the running version reaches `removed_in` while the entry still exists, so the removal cannot be forgotten.

There are two kinds of move:

- A whole module or package moves (`MOVED_MODULES`).
  `MovedModuleFinder` serves the old name by importing the new module and registering the same module object under the old name.
  The module never runs twice, so its classes, registries, and context variables stay single.
  `mock.patch("<old path>.<name>")` still patches the code that runs.
  Submodules of a moved package resolve the same way.
- Names split out of a module that stays (`MOVED_NAMES`).
  The old module installs `moved_names_getattr` as its `__getattr__`, which returns the identical object from its new module.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import warnings
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Mapping, Optional


DEPRECATION_WINDOW_MINOR_RELEASES = 2


@dataclass(frozen=True)
class MovedPath:
    """Where a module or name moved to, and the release the move first ships in.

    Attributes:
        new: The new dotted path: a module for a module move; `module` or `module:new_name` for a name move.
        since: The first release (`MAJOR.MINOR.PATCH`) that ships the new path.
    """

    new: str
    since: str

    @property
    def removed_in(self) -> str:
        """The release that drops the old path."""
        major, minor, _ = (int(part) for part in self.since.split("."))
        return f"{major}.{minor + DEPRECATION_WINDOW_MINOR_RELEASES}.0"


# Old module path -> new module path.
# A key also covers every submodule under it.
MOVED_MODULES: dict[str, MovedPath] = {
    "nemo_gym.config_types": MovedPath(new="nemo_gym.config.config_types", since="0.8.0"),
    "nemo_gym.global_config": MovedPath(new="nemo_gym.config.global_config", since="0.8.0"),
    "nemo_gym.secret_utils": MovedPath(new="nemo_gym.config.secret_utils", since="0.8.0"),
    "nemo_gym.server_metadata": MovedPath(new="nemo_gym.config.server_metadata", since="0.8.0"),
}

# Old module path -> {old name -> new location}, for names split out of a module that still exists.
MOVED_NAMES: dict[str, dict[str, MovedPath]] = {}


def _running_version() -> str:
    from nemo_gym.package_info import __version__

    return __version__


def deprecation_message(old: str, moved: MovedPath, new: str) -> str:
    """The warning text for a use of the old path `old`, which now lives at `new`."""
    return (
        f"`{old}` moved to `{new}` in nemo_gym {moved.since} and the old path will be removed in nemo_gym "
        f"{moved.removed_in} (running {_running_version()}). Import from `{new}` instead."
    )


def _file_prefix(path: str) -> str:
    # CPython matches skip_file_prefixes only as a strict prefix; "module." also keeps "module_utils.py" out.
    return os.path.splitext(path)[0] + "."


_STDLIB = os.path.dirname(os.__file__)
# Frames a moved-path warning skips to reach the code that used the old path: this module, the import machinery,
# the package root (`from nemo_gym import <moved module>`), and pkgutil and mock (`mock.patch("<old path>.name")`).
_SKIP_FILE_PREFIXES = (
    _file_prefix(__file__),
    _file_prefix(os.path.join(os.path.dirname(__file__), "__init__.py")),
    os.path.dirname(importlib.__file__) + os.sep,
    "<frozen importlib",
    _file_prefix(os.path.join(_STDLIB, "pkgutil.py")),
    _file_prefix(os.path.join(_STDLIB, "unittest", "mock.py")),
)


def warn_moved(message: str, *, old_module: str) -> None:
    """Warn about a moved path at the frame that used it, past the getter that calls this and `old_module`'s own
    `__getattr__` wrappers."""
    files = [sys._getframe(1).f_code.co_filename, getattr(sys.modules.get(old_module), "__file__", None)]
    skip = _SKIP_FILE_PREFIXES + tuple(_file_prefix(file) for file in files if file)
    warnings.warn(message, DeprecationWarning, skip_file_prefixes=skip)


def _resolve(fullname: str, moves: Mapping[str, MovedPath]) -> Optional[tuple[str, str, MovedPath]]:
    """Return `(old key, new module name, entry)` for the longest moved prefix of `fullname`, if any."""
    old = fullname
    while True:
        moved = moves.get(old)
        if moved is not None:
            return old, moved.new + fullname[len(old) :], moved
        old, sep, _ = old.rpartition(".")
        if not sep:
            return None


class MovedModuleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serve old module paths as aliases of the modules that replaced them.

    It must sit first on `sys.meta_path`.
    Once an old package name is bound to the new package, the normal path finder would find the old package's submodules through the new package's `__path__`.
    It would then run a second copy of each submodule under its old name.
    """

    def __init__(self, moves: Mapping[str, MovedPath]) -> None:
        self._moves = moves
        self._own_specs: dict[str, importlib.machinery.ModuleSpec] = {}

    def find_spec(self, fullname: str, path: Any = None, target: Optional[ModuleType] = None) -> Any:
        resolved = _resolve(fullname, self._moves)
        if resolved is None:
            return None
        _, new_name, _ = resolved
        # Report the new module's file and whether it is a package.
        # `importlib.util.find_spec(<old path>).origin` then names the file that runs.
        new_spec = importlib.util.find_spec(new_name)
        if new_spec is None:
            return None
        return importlib.machinery.ModuleSpec(
            fullname, self, origin=new_spec.origin, is_package=new_spec.submodule_search_locations is not None
        )

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        old, new_name, moved = _resolve(spec.name, self._moves)
        warnings.warn(
            deprecation_message(spec.name, moved, new_name),
            DeprecationWarning,
            skip_file_prefixes=_SKIP_FILE_PREFIXES,
        )
        # The import system keeps the module's own __name__ and __loader__, and sets __spec__ to the alias spec;
        # exec_module puts the module's own spec back.
        # It binds the module under the old name in sys.modules and on the old parent package.
        module = importlib.import_module(new_name)
        self._own_specs[module.__name__] = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        # The module already ran when it was imported under its new name.
        # Its own spec keeps `importlib.util.find_spec(<new path>)` and `importlib.reload` working.
        module.__spec__ = self._own_specs.pop(module.__name__, module.__spec__)

    def get_code(self, fullname: str) -> Any:
        """Give `python -m <old path>` the new module's code, with the same warning."""
        _, new_name, moved = _resolve(fullname, self._moves)
        warnings.warn(deprecation_message(fullname, moved, new_name), DeprecationWarning, stacklevel=2)
        return importlib.util.find_spec(new_name).loader.get_code(new_name)


def install_moved_module_finder() -> None:
    """Put the finder for `MOVED_MODULES` first on `sys.meta_path`; calling it again changes nothing."""
    if not any(isinstance(finder, MovedModuleFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, MovedModuleFinder(MOVED_MODULES))


def _names_were_listed_under(module_name: str) -> list[str]:
    """`module_name`, then each old path that `MOVED_MODULES` maps to it, directly or as part of a moved package."""
    paths = [module_name]
    for old, moved in MOVED_MODULES.items():
        if module_name == moved.new or module_name.startswith(moved.new + "."):
            paths.append(old + module_name[len(moved.new) :])
    return paths


def moved_names_getattr(module_name: str) -> Callable[[str], Any]:
    """Build a module `__getattr__` that serves the names `MOVED_NAMES[module_name]` lists.

    Assign the result to `__getattr__` in the old module.
    In a package, it also serves each moved submodule that `MOVED_MODULES` lists under the package,
    so `package.old_submodule` resolves without an import of it first.
    Names the old module still defines or imports never reach it.
    If the module itself moves later, its entries stay under its old path, and this `__getattr__`
    keeps serving them once the module runs under its new name.
    """

    warned: set[str] = set()

    def __getattr__(name: str) -> Any:
        for listed_as in _names_were_listed_under(module_name):
            moved = MOVED_NAMES.get(listed_as, {}).get(name)
            if moved is not None:
                break
        else:
            for listed_as in _names_were_listed_under(module_name):
                if f"{listed_as}.{name}" in MOVED_MODULES:
                    # The finder warns, and binds the module on the package.
                    return importlib.import_module(f"{listed_as}.{name}")
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        new_module, _, new_name = moved.new.partition(":")
        new_name = new_name or name
        # Look the name up on every access, so a patch of the new module shows through; warn once per old path.
        value = getattr(importlib.import_module(new_module), new_name)
        if name not in warned:
            warn_moved(
                deprecation_message(f"{listed_as}.{name}", moved, f"{new_module}.{new_name}"), old_module=module_name
            )
            warned.add(name)
        return value

    return __getattr__
