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

    def __getattr__(name: str) -> Any:
        submodule = exports.get(name)
        if submodule is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        # Not cached on the package: a patch of the submodule's name stays visible here.
        return getattr(importlib.import_module(f"{package}.{submodule}"), name)

    def __dir__() -> list[str]:
        return sorted(set(importlib.import_module(package).__dict__) | set(exports))

    return __getattr__, __dir__
