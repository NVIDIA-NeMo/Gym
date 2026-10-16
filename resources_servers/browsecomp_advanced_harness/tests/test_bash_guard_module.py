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
"""bash_guard.py runs inside a stock python image, so it must import with the stdlib alone."""

import subprocess
import sys
from pathlib import Path

from resources_servers.browsecomp_advanced_harness import app, bash_guard


HARNESS_DIR = Path(__file__).resolve().parents[1]


def test_imports_with_stdlib_only():
    # -I ignores env vars and user site; -S skips site-packages entirely.
    code = f"import sys; sys.path.insert(0, {str(HARNESS_DIR)!r}); import bash_guard"
    result = subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_app_uses_the_shared_guard():
    assert app._bash_denylisted is bash_guard._bash_denylisted
    assert app._bash_allowlisted is bash_guard._bash_allowlisted


def test_format_bash_output_shapes():
    assert bash_guard.format_bash_output("out\n", "", 0) == "out\n\n[exit_code=0]"
    assert bash_guard.format_bash_output("", "err", 2) == "--- stderr ---\nerr\n[exit_code=2]"
    long = "x" * (bash_guard._BASH_HEAD_BYTES + bash_guard._BASH_TAIL_BYTES) * 3
    assert bash_guard.format_bash_output(long, "", 0).endswith("[exit_code=0 truncated]")
