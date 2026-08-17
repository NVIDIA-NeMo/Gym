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
"""Backward-compatibility helpers for symbols relocated during the CLI refactor.

It provides a lazy import mechanism for symbols that were moved to the `nemo_gym.cli`
package and emits a `DeprecationWarning` pointing at the correct import path.

Note that an eager re-export `from nemo_gym.cli.x import foo` at the top of a
core module would create a circular import.
"""

import importlib
import warnings
from typing import Any, Callable, Mapping


def moved_attr_getter(module_name: str, moved: Mapping[str, str]) -> Callable[[str], Any]:
    """Build a module `__getattr__` that re-exports relocated symbols with a deprecation warning.

    Args:
        module_name: The `__name__` of the module installing the shim (the old location).
        moved: Maps each old attribute name to its new location. The value is either
            `"package.module"` (same attribute name) or `"package.module:new_name"`
            when the symbol was also renamed.

    Returns:
        A function suitable for assigning to a module's `__getattr__`.
    """

    def __getattr__(name: str) -> Any:
        target = moved.get(name)
        if target is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        new_module, _, new_name = target.partition(":")
        new_name = new_name or name
        warnings.warn(
            f"`{module_name}.{name}` has moved to `{new_module}.{new_name}` and will be removed in a "
            f"future release. Update your import to `from {new_module} import {new_name}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(new_module), new_name)

    return __getattr__
