# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OfficeEnv installs apart from this server's locked dependencies until it is published (README.md, "The OfficeEnv
version"). The other tests need it and skip without it; the first one here runs either way, and the second needs it
to tell an older release from none."""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
# Imports the server as Gym starts it, with a module of OfficeEnv's hidden whether or not the venv has it
HIDING = """
import importlib.util

find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args, **kwargs: None if name == {hidden!r} else find_spec(name, *args, **kwargs)
import resources_servers.officeenv.app
"""


def import_hiding(module: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))}
    script = HIDING.format(hidden=module)
    return subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120
    )


def test_without_officeenv_the_server_stops_naming_the_step_that_installs_it() -> None:
    result = import_hiding("officeenv")
    assert result.returncode != 0
    assert "ImportError: OfficeEnv is not installed in this server's venv" in result.stderr, result.stderr[-2000:]
    assert "resources_servers/officeenv/README.md (OFFICEENV_GIT_URL)" in result.stderr


def test_with_an_officeenv_older_than_1_13_the_server_stops_naming_both_versions() -> None:
    """Workspace mode and the file call (officeenv.files) are OfficeEnv 1.13.0's: a venv with an older release has the
    server stop at its import, naming the release it needs and the one it found."""
    if importlib.util.find_spec("officeenv") is None:
        pytest.skip("OfficeEnv is not installed (README.md, 'The OfficeEnv version')")
    result = import_hiding("officeenv.files")
    assert result.returncode != 0
    needs = r"ImportError: This server needs OfficeEnv 1\.13\.0 or later \(officeenv\.files\), and its venv has OfficeEnv \d"
    assert re.search(needs, result.stderr), result.stderr[-2000:]
    assert "resources_servers/officeenv/README.md names (OFFICEENV_GIT_URL)" in result.stderr
