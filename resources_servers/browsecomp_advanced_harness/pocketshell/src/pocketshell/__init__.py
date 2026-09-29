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
"""pocketshell -- an in-process, workspace-confined read-only shell.

Drop-in replacement for ``bash -c`` in LLM agent harnesses that only ever need
to let a model inspect files it has already retrieved. Because nothing is
spawned, there is no sandbox to build or maintain, and because every path is
resolved against a workspace root, filesystem escape is structurally impossible
rather than blocked by an enumerated deny-list.

    >>> from pocketshell import run
    >>> r = run('grep -c "" pages/*.txt | head -3', workspace="/tmp/ws")
    >>> r.stdout, r.exit_code
"""

from .fsview import PathEscape, Workspace
from .regex_xlate import bre_to_python, compile_pattern, ere_to_python
from .shell import DEFAULT_TIMEOUT, PocketShellError, Result, run
from .syntax import ParseError, parse


__version__ = "0.1.0"
__all__ = [
    "run",
    "Result",
    "Workspace",
    "PathEscape",
    "ParseError",
    "PocketShellError",
    "parse",
    "bre_to_python",
    "ere_to_python",
    "compile_pattern",
    "DEFAULT_TIMEOUT",
]
