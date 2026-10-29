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
    "nemo_gym.decorators": MovedPath(new="nemo_gym.utils.decorators", since="0.8.0"),
    "nemo_gym.path_utils": MovedPath(new="nemo_gym.utils.path_utils", since="0.8.0"),
    "nemo_gym.install_lock": MovedPath(new="nemo_gym.utils.install_lock", since="0.8.0"),
    "nemo_gym.cli._compat": MovedPath(new="nemo_gym.utils.compat", since="0.8.0"),
    "nemo_gym.cli._venv_setup": MovedPath(new="nemo_gym.utils.venv_setup", since="0.8.0"),
}

# Old module path -> {old name -> new location}, for names split out of a module that still exists.
MOVED_NAMES: dict[str, dict[str, MovedPath]] = {
    "nemo_gym.server_utils": {
        # server_utils imported this before the split and no longer uses it.
        "NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME": MovedPath(new="nemo_gym.config.global_config", since="0.8.0"),
        "IS_NEMO_GYM_FASTAPI_ENTRYPOINT_KEY_NAME": MovedPath(new="nemo_gym.utils.server_process", since="0.8.0"),
        "IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME": MovedPath(new="nemo_gym.utils.server_process", since="0.8.0"),
        "NEMO_GYM_FASTAPI_NUM_WORKERS": MovedPath(new="nemo_gym.utils.server_process", since="0.8.0"),
        "_has_injected_global_config_env": MovedPath(new="nemo_gym.utils.server_process", since="0.8.0"),
        "get_nemo_gym_fastapi_num_workers": MovedPath(new="nemo_gym.utils.server_process", since="0.8.0"),
        "is_nemo_gym_fastapi_entrypoint": MovedPath(new="nemo_gym.utils.server_process", since="0.8.0"),
        "is_nemo_gym_fastapi_worker": MovedPath(new="nemo_gym.utils.server_process", since="0.8.0"),
        "set_is_nemo_gym_fastapi_entrypoint": MovedPath(new="nemo_gym.utils.server_process", since="0.8.0"),
        "set_is_nemo_gym_fastapi_worker": MovedPath(new="nemo_gym.utils.server_process", since="0.8.0"),
        "set_nemo_gym_fastapi_num_workers": MovedPath(new="nemo_gym.utils.server_process", since="0.8.0"),
    },
    "nemo_gym.cli.utils": {
        "did_you_mean": MovedPath(new="nemo_gym.utils.text", since="0.8.0"),
    },
}


def _running_version() -> str:
    from nemo_gym.package_info import __version__

    return __version__


def deprecation_message(old: str, moved: MovedPath, new: str) -> str:
    """The warning text for a use of the old path `old`, which now lives at `new`."""
    return (
        f"`{old}` moved to `{new}` in nemo_gym {moved.since} and the old path will be removed in nemo_gym "
        f"{moved.removed_in} (running {_running_version()}). Import from `{new}` instead."
    )


# Frames to skip so the warning points at the code that imported the old path: this file and importlib.
_SKIP_FILE_PREFIXES = (__file__, os.path.dirname(importlib.__file__) + os.sep, "<frozen importlib")


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
        # The import system keeps the module's own __name__, __spec__ and __loader__.
        # It binds the module under the old name in sys.modules and on the old parent package.
        return importlib.import_module(new_name)

    def exec_module(self, module: ModuleType) -> None:
        # The module already ran when it was imported under its new name.
        pass


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
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        new_module, _, new_name = moved.new.partition(":")
        new_name = new_name or name
        # Look the name up on every access, so a patch of the new module shows through; warn once per old path.
        value = getattr(importlib.import_module(new_module), new_name)
        if name not in warned:
            warnings.warn(
                deprecation_message(f"{listed_as}.{name}", moved, f"{new_module}.{new_name}"),
                DeprecationWarning,
                stacklevel=2,
            )
            warned.add(name)
        return value

    return __getattr__
