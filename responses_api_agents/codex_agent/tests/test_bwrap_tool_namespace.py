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

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from responses_api_agents.codex_agent.tests.tool_namespace_check import run_check


BWRAP = os.environ.get("NEMO_GYM_TEST_BWRAP") or shutil.which("bwrap")
CODEX_RUNTIME = os.environ.get("NEMO_GYM_TEST_CODEX_RUNTIME")
REQUIRED = os.environ.get("NEMO_GYM_REQUIRE_BWRAP") == "1"


def _nested_user_namespaces_unavailable() -> str | None:
    """None when bwrap can create a user and pid namespace inside another one, else why not.

    The tool namespace is a bwrap inside the sandbox's bwrap. A host can allow the outer namespace and refuse the
    nested one, e.g. when an AppArmor user-namespace restriction confines processes inside an unprivileged namespace.
    """
    if BWRAP is None or not sys.platform.startswith("linux"):
        return "needs Linux and bwrap"
    namespace = ["--unshare-user", "--unshare-pid", "--ro-bind", "/", "/", "--proc", "/proc", "--dev", "/dev"]
    probe = subprocess.run([BWRAP, *namespace, BWRAP, *namespace, "true"], capture_output=True, text=True, timeout=30)
    if probe.returncode == 0:
        return None
    return f"bwrap ({BWRAP}) cannot create a nested user namespace here: {probe.stderr.strip()[-300:]}"


UNAVAILABLE = _nested_user_namespaces_unavailable()
if REQUIRED and UNAVAILABLE:
    raise RuntimeError(f"NEMO_GYM_REQUIRE_BWRAP=1, but {UNAVAILABLE}")

pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(UNAVAILABLE is not None, reason=UNAVAILABLE or ""),
]


def _assert_ok(report: dict) -> None:
    # The whole report: pytest's assertion repr truncates it, hiding the setup output and checks that explain a failure.
    if not report["ok"]:
        pytest.fail(json.dumps(report, indent=1, default=str), pytrace=False)


async def test_tool_commands_cannot_kill_the_harness(tmp_path: Path) -> None:
    report = await run_check(BWRAP, base_dir=str(tmp_path))
    _assert_ok(report)
    assert all(report["checks"].values())


async def test_without_the_namespace_the_same_commands_kill_the_harness(tmp_path: Path) -> None:
    # Shows the check is sensitive: kill -9 -1 from the harness's own namespace ends the episode.
    report = await run_check(BWRAP, base_dir=str(tmp_path), control=True)
    _assert_ok(report)
    assert "execute_error" not in report and "kill -9 -1" in report["kill_report"]
    assert report["turn_completed"] is False


@pytest.mark.skipif(not CODEX_RUNTIME, reason="set NEMO_GYM_TEST_CODEX_RUNTIME to a Codex runtime directory")
async def test_real_codex_tool_commands_cannot_kill_the_harness(tmp_path: Path) -> None:
    report = await run_check(BWRAP, base_dir=str(tmp_path), codex_runtime=CODEX_RUNTIME)
    _assert_ok(report)
    assert report["model_requests"] == 5  # kill, view, exec_command, write_stdin, answer
