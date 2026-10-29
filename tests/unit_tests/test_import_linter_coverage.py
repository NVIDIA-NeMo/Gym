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
"""Every top-level module and package of nemo_gym is in a layer of the import-linter contract.

A module outside every layer is not checked at all, so a new top-level module must join a layer, or be listed
here with the reason it stays outside.
"""

import tomllib
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

OUTSIDE_THE_LAYERS = {
    "__init__": "the package root",
    "package_info": "version metadata",
    "_moved_paths": "the old-path compatibility table, which nemo_gym/__init__.py installs first",
    "_public_api": "builds each package's lazy public API from the standard library alone",
    "verifiers": "standard-library-only helpers that verifiers running inside sandboxes import by name",
}


def _layer_modules() -> set[str]:
    contract = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["importlinter"]["contracts"][0]
    return {module.strip() for layer in contract["layers"] for module in layer.split(":")}


def _top_level_names() -> set[str]:
    package = REPO / "nemo_gym"
    return {path.stem for path in package.glob("*.py")} | {path.parent.name for path in package.glob("*/__init__.py")}


def test_every_top_level_module_is_in_a_layer() -> None:
    layered = _layer_modules()
    unlayered = sorted(
        name
        for name in _top_level_names() - set(OUTSIDE_THE_LAYERS)
        if not any(module == f"nemo_gym.{name}" or module.startswith(f"nemo_gym.{name}.") for module in layered)
    )
    assert not unlayered, (
        f"{unlayered} are in no layer of the import-linter contract in pyproject.toml, so nothing checks their "
        "imports. Add each to the layer of the package it belongs to, or to OUTSIDE_THE_LAYERS with the reason."
    )


def test_every_module_kept_outside_the_layers_exists() -> None:
    assert not set(OUTSIDE_THE_LAYERS) - _top_level_names()
