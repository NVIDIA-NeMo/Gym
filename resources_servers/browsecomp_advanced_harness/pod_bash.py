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
"""Run one bash_command inside a sandbox and print the model-visible result.

Usage: python3 pod_bash.py <workspace> <duration_s> <base64 keystrokes>

Stdlib only, plus the pocketshell package staged next to this file. The output
matches TavilySearchResourcesServer._run_bash_readonly byte for byte; the outer
timeout stays with the caller, as it does on the host.
"""

import base64
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bash_guard import _bash_allowlisted, _bash_denylisted, format_bash_output  # noqa: E402
from pocketshell import run as pocketshell_run  # noqa: E402


def run_bash(keystrokes: str, workspace: str, duration: float) -> str:
    blocked = _bash_denylisted(keystrokes) or _bash_allowlisted(keystrokes)
    if blocked is not None:
        return f"[blocked: {blocked}]\n[exit_code=-3]"
    try:
        result = pocketshell_run(keystrokes or "", workspace, duration)
    except Exception as e:
        return f"[bash error: {e}]\n[exit_code=-2]"
    return format_bash_output(result.stdout, result.stderr, result.exit_code)


def main() -> None:
    workspace, duration, encoded = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    keystrokes = base64.b64decode(encoded).decode("utf-8", errors="replace")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout.write(run_bash(keystrokes, workspace, duration))


if __name__ == "__main__":
    main()
