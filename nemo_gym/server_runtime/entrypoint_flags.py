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
"""Reading literal class flags, such as `ray_enabled`, from a server entrypoint's source without importing it."""

import ast
import warnings
from pathlib import Path


def _declared_class_flag(class_node: ast.ClassDef, name: str) -> bool | None:
    """The literal boolean assigned to `name` in a class body, annotated or not, or None when there is none."""
    for item in class_node.body:
        if isinstance(item, ast.Assign):
            targets = item.targets
        elif isinstance(item, ast.AnnAssign):
            targets = [item.target]
        else:
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in targets):
            continue
        if isinstance(item.value, ast.Constant) and isinstance(item.value.value, bool):
            return item.value.value
    return None


def entrypoint_class_flag(entrypoint_fpath: Path, name: str) -> bool | None:
    """The boolean class attribute `name` declared by the server an entrypoint starts, read from source.

    Gym settles some things before any server exists, and each server runs in its own venv, so this reads the
    source instead of importing it. It finds the classes the entrypoint calls `run_webserver()` on and checks
    their literal `name` declarations (`name = True` or `name: ClassVar[bool] = True` in the class body).

    Returns True when any of them declares True, False when all of them declare False, and None when that can't
    be settled from the source alone, such as an inherited declaration, a class imported from another module, or
    an unreadable entrypoint.
    """
    try:
        source = entrypoint_fpath.read_text()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None

    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    invoked_class_names = {
        node.func.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_webserver"
        and isinstance(node.func.value, ast.Name)
    }
    if not invoked_class_names:
        return None
    declarations = [
        _declared_class_flag(classes[class_name], name) if class_name in classes else None
        for class_name in invoked_class_names
    ]
    if True in declarations:
        return True
    if all(declaration is False for declaration in declarations):
        return False
    return None
