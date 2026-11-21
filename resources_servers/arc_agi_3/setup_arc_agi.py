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
"""Make ``import arc_agi`` work in a Gym server venv.

``arc_agi`` 0.9.8 imports Flask (for its REST server) at import time, but nemo-gym's ``[tool.uv]
exclude-dependencies`` drops ``flask``, ``werkzeug`` and ``blinker`` (they are mlflow server dependencies there). The
ARC-AGI-3 components declare them as exceptions (``scripts/component-lock-policy.json``), so their venvs have them.
``ensure_flask`` covers any other interpreter: it installs the versions of the ARC-AGI-3 competition wheelhouse with
that interpreter's pip when Flask is missing.
"""

import importlib
import importlib.util
import subprocess
import sys


# The ARC Prize 2026 ARC-AGI-3 competition wheelhouse (arc_agi_3_wheels/).
FLASK_PINS = ("flask==3.1.3", "werkzeug==3.1.8", "blinker==1.9.0")


def ensure_flask() -> None:
    if importlib.util.find_spec("flask") is not None:
        return
    if importlib.util.find_spec("pip") is None:
        raise RuntimeError(
            f"arc_agi needs flask, which nemo-gym's uv exclude-dependencies removes, and {sys.executable} has no pip "
            f"to install it; run: {sys.executable} -m ensurepip && {sys.executable} -m pip install {' '.join(FLASK_PINS)}"
        )
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *FLASK_PINS],
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"pip install {' '.join(FLASK_PINS)} failed: {result.stderr.strip()[-2000:]}")
    importlib.invalidate_caches()
    importlib.import_module("flask")
