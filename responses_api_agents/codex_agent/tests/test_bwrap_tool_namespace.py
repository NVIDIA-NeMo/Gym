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

"""Codex tool commands cannot kill the harness: real bwrap, real supervisor and runner.

Runs ``responses_api_agents.codex_agent.tool_namespace_check`` (Linux with user namespaces).
NEMO_GYM_TEST_BWRAP selects the bwrap binary (default: bwrap on PATH); NEMO_GYM_REQUIRE_BWRAP=1
turns a skip into a failure (cluster validation); NEMO_GYM_TEST_CODEX_RUNTIME, a directory
installed by install_codex_runtime.sh, adds the run with the real Codex CLI.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from responses_api_agents.codex_agent.tool_namespace_check import run_check


BWRAP = os.environ.get("NEMO_GYM_TEST_BWRAP") or shutil.which("bwrap")
CODEX_RUNTIME = os.environ.get("NEMO_GYM_TEST_CODEX_RUNTIME")
REQUIRED = os.environ.get("NEMO_GYM_REQUIRE_BWRAP") == "1"


def _user_namespaces_work() -> bool:
    if BWRAP is None or not sys.platform.startswith("linux"):
        return False
    probe = subprocess.run(
        [BWRAP, "--unshare-user", "--unshare-pid", "--ro-bind", "/", "/", "--proc", "/proc", "true"],
        capture_output=True,
        timeout=30,
    )
    return probe.returncode == 0


AVAILABLE = _user_namespaces_work()
if REQUIRED and not AVAILABLE:
    raise RuntimeError(f"NEMO_GYM_REQUIRE_BWRAP=1 but bwrap ({BWRAP}) cannot create user namespaces here")

pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(not AVAILABLE, reason="needs Linux, bwrap and user namespaces"),
]


def _assert_ok(report: dict) -> None:
    assert report["ok"], report


async def test_tool_commands_cannot_kill_the_harness(tmp_path: Path) -> None:
    report = await run_check(BWRAP, base_dir=str(tmp_path))
    _assert_ok(report)
    assert all(report["checks"].values())


async def test_without_the_namespace_the_same_commands_kill_the_harness(tmp_path: Path) -> None:
    # Shows the check is sensitive: kill -9 -1 from the harness's own namespace ends the episode.
    report = await run_check(BWRAP, base_dir=str(tmp_path), control=True)
    _assert_ok(report)
    assert report["turn_completed"] is False


@pytest.mark.skipif(not CODEX_RUNTIME, reason="set NEMO_GYM_TEST_CODEX_RUNTIME to a Codex runtime directory")
async def test_real_codex_tool_commands_cannot_kill_the_harness(tmp_path: Path) -> None:
    report = await run_check(BWRAP, base_dir=str(tmp_path), codex_runtime=CODEX_RUNTIME)
    _assert_ok(report)
    assert report["model_requests"] == 5  # kill, view, exec_command, write_stdin, answer
