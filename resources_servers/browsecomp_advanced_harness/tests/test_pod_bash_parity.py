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
"""pod_bash.py must give the model the same bash_command output as the host path.

Runs pod_bash.py the way the sandbox does: a separate interpreter with the stdlib
only, from a staged copy of the runtime files.
"""

import asyncio
import base64
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from resources_servers.browsecomp_advanced_harness import app as harness


HARNESS_DIR = Path(__file__).resolve().parents[1]


@pytest.fixture
def stage(tmp_path):
    stage = tmp_path / "rt"
    stage.mkdir()
    shutil.copy(HARNESS_DIR / "bash_guard.py", stage)
    shutil.copy(HARNESS_DIR / "pod_bash.py", stage)
    shutil.copytree(
        HARNESS_DIR / "pocketshell" / "src" / "pocketshell",
        stage / "pocketshell",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return stage


@pytest.fixture
def ws(tmp_path):
    ws = tmp_path / "ws"
    pages = ws / "pages"
    pages.mkdir(parents=True)
    (pages / "0001_browse_alpha.txt").write_text("Alpha Beta\nDavid Pemantle 1983\nPhD thesis 1994\nlast line\n")
    (pages / "0002_browse_beta.txt").write_text("beta\nno match\n")
    (pages / "0003_search_big.txt").write_text("".join(f"line {i} of a long page\n" for i in range(5000)))
    (ws / "manifest.tsv").write_text("0001\tbrowse\thttps://a.example\tAlpha\t120\n")
    return ws


def host_output(keystrokes, ws, duration=10.0):
    stub = types.SimpleNamespace(_bash_semaphore=asyncio.Semaphore(1))
    bound = types.MethodType(harness.TavilySearchResourcesServer._run_bash_readonly, stub)
    return asyncio.run(bound(keystrokes, duration, str(ws)))


def pod_output(keystrokes, ws, stage, duration=10.0):
    encoded = base64.b64encode(keystrokes.encode()).decode()
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(stage / "pod_bash.py"), str(ws), str(duration), encoded],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.mark.parametrize(
    "keystrokes",
    [
        'grep -i "david" pages/0001_browse_alpha.txt',
        "grep -c line pages/*.txt | sort",
        "cat manifest.tsv",
        "cd pages && ls",
        "cat pages/missing.txt",
        "cat pages/0003_search_big.txt",
        "rm -rf pages",
        "jq . manifest.tsv",
        "cat /etc/passwd",
        "",
    ],
)
def test_pod_matches_host(keystrokes, ws, stage):
    assert pod_output(keystrokes, ws, stage) == host_output(keystrokes, ws)


def test_truncation_case_is_really_truncated(ws, stage):
    assert "[exit_code=0 truncated]" in pod_output("cat pages/0003_search_big.txt", ws, stage)


def test_blocked_case_is_really_blocked(ws, stage):
    assert pod_output("rm -rf pages", ws, stage) == "[blocked: command 'rm' is blocked]\n[exit_code=-3]"
