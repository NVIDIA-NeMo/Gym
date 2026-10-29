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

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_DIR = Path(__file__).resolve().parents[2] / "nemo_gym"
# Every package that declares a public API.
PACKAGES = sorted(path.parent.name for path in PACKAGE_DIR.glob("*/__init__.py") if "_PUBLIC_API" in path.read_text())


def _public_api(package: str) -> dict[str, str]:
    return getattr(importlib.import_module(f"nemo_gym.{package}"), "_PUBLIC_API", {})


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
    eager = {"sandbox", "telemetry"}  # their existing public names import their submodules eagerly
    if package not in eager:
        assert loaded.strip() == "[]"


def test_packages_declare_public_apis() -> None:
    assert PACKAGES


def test_an_unknown_name_is_an_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'not_a_public_name'"):
        importlib.import_module(f"nemo_gym.{PACKAGES[0]}").not_a_public_name


def test_a_patch_of_the_submodule_shows_through_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    package = next(p for p in PACKAGES if _public_api(p))
    name, submodule = next(iter(_public_api(package).items()))
    replacement = object()
    monkeypatch.setattr(importlib.import_module(f"nemo_gym.{package}.{submodule}"), name, replacement)
    assert getattr(importlib.import_module(f"nemo_gym.{package}"), name) is replacement
