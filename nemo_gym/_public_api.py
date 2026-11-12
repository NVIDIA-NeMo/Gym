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
"""Lazy public APIs for nemo_gym's packages.

Each package's ``__init__.py`` lists the names it offers to code outside the library, such as servers, and the
submodule that defines each one. A name's submodule is imported on first use, so importing a package imports
none of its submodules and adds no import edges between packages. Code inside nemo_gym imports submodules
directly, which keeps every dependency visible to the import-linter contract.
"""

import importlib
from collections.abc import Callable, Mapping
from typing import Any


def lazy_exports(package: str, exports: Mapping[str, str]) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Build a package's ``__getattr__`` and ``__dir__`` for ``exports``, which maps each name to its submodule."""

    # Built on first use and kept, so its record of warned names lasts and each old name warns once.
    moved_getattr: Callable[[str], Any] | None = None

    def __getattr__(name: str) -> Any:
        nonlocal moved_getattr
        submodule = exports.get(name)
        if submodule is None:
            # A name the package exported before it moved or was renamed resolves through MOVED_NAMES.
            if moved_getattr is None:
                from nemo_gym._moved_paths import moved_names_getattr

                moved_getattr = moved_names_getattr(package)
            return moved_getattr(name)
        try:
            module = importlib.import_module(f"{package}.{submodule}")
        except AttributeError as exc:
            # A from-import probes with hasattr, which would swallow this, import the submodule again, and
            # report only "cannot import name". Raise once, with the cause.
            raise ImportError(f"importing {package}.{submodule} failed: {exc}", name=f"{package}.{submodule}") from exc
        # Not cached on the package: a patch of the submodule's name stays visible here.
        return getattr(module, name)

    def __dir__() -> list[str]:
        return sorted(set(importlib.import_module(package).__dict__) | set(exports))

    return __getattr__, __dir__
