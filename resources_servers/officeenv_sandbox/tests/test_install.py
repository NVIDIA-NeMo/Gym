# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandboxed server imports the OfficeEnv server's module before OfficeEnv's own, so it checks OfficeEnv's install
the same way (resources_servers/officeenv/README.md, "The OfficeEnv version"). This test runs whether or not the venv
has OfficeEnv."""

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
# Imports the server as Gym starts it, with OfficeEnv hidden whether or not the venv has it
WITHOUT_OFFICEENV = """
import importlib.util

find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args, **kwargs: None if name == "officeenv" else find_spec(name, *args, **kwargs)
import resources_servers.officeenv_sandbox.app
"""


def test_without_officeenv_the_server_stops_naming_the_step_that_installs_it() -> None:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    result = subprocess.run(
        [sys.executable, "-c", WITHOUT_OFFICEENV], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120
    )
    assert result.returncode != 0
    assert "ImportError: OfficeEnv is not installed in this server's venv" in result.stderr, result.stderr[-2000:]
    assert "resources_servers/officeenv/README.md (OFFICEENV_GIT_URL)" in result.stderr
