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
"""Importing nemo_gym.sandbox is cheap: its public names load their modules on first use."""

import subprocess
import sys


def test_importing_the_package_loads_only_the_package_and_the_lazy_export_helper() -> None:
    # A fresh interpreter, because this process has imported the providers already. Measured after the root
    # package, whose own imports are not this package's cost.
    code = "import sys, nemo_gym; before = set(sys.modules); import nemo_gym.sandbox; print(sorted(set(sys.modules) - before))"
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert loaded.strip() == repr(["nemo_gym._public_api", "nemo_gym.sandbox"])


def test_a_public_registry_function_sees_the_built_in_providers() -> None:
    # The built-in providers register when the registry module loads, which the first registry name triggers.
    code = "from nemo_gym.sandbox import list_providers; print('docker' in list_providers())"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "True"
