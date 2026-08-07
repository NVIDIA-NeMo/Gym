# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Guards the ``nemo_gym`` public API surface.

Environments are expected to import base classes and core types from ``nemo_gym`` directly rather
than from internal module paths. These tests pin that contract: every advertised name resolves, the
lazy top-level export is the *same object* as the deep import, and the top-level import stays lazy so
``import nemo_gym`` does not eagerly pull in the whole server stack.
"""

import importlib
import subprocess
import sys

import pytest

import nemo_gym
from nemo_gym import _LAZY_EXPORTS


@pytest.mark.parametrize("name", sorted(_LAZY_EXPORTS))
def test_public_symbol_is_accessible(name: str):
    assert getattr(nemo_gym, name) is not None


@pytest.mark.parametrize("name, module_name", sorted(_LAZY_EXPORTS.items()))
def test_public_symbol_matches_deep_import(name: str, module_name: str):
    """The top-level re-export must be the identical object exposed by the internal module.

    This is what lets downstream code migrate to the public path without behavior changes, and keeps
    ``isinstance`` / subclass checks working across the two import styles.
    """
    module = importlib.import_module(f"nemo_gym.{module_name}")
    assert getattr(nemo_gym, name) is getattr(module, name)


def test_all_is_sorted_and_covers_lazy_exports():
    assert nemo_gym.__all__ == sorted(nemo_gym.__all__)
    assert set(_LAZY_EXPORTS).issubset(nemo_gym.__all__)


def test_dir_includes_lazy_exports():
    listed = dir(nemo_gym)
    assert set(_LAZY_EXPORTS).issubset(listed)


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        _ = nemo_gym.DefinitelyNotARealSymbol


def test_top_level_import_is_lazy():
    """Importing ``nemo_gym`` must not eagerly import the heavy submodules.

    Run in a fresh interpreter so the assertion is not confounded by other tests that already imported
    these modules. Accessing a symbol should then import its backing module on demand.
    """
    script = (
        "import sys\n"
        "import nemo_gym\n"
        "assert 'nemo_gym.base_resources_server' not in sys.modules, 'import was not lazy'\n"
        "assert nemo_gym.SimpleResourcesServer is not None\n"
        "assert 'nemo_gym.base_resources_server' in sys.modules, 'access did not import module'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
