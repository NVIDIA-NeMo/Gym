# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clean-checkout import coverage for the OpenAir resource server."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


RESOURCE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = RESOURCE_DIR.parents[1]


def test_resource_server_imports_without_external_worktree() -> None:
    """The server must import using only files in its own checkout."""

    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(RESOURCE_DIR), str(REPO_ROOT))),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app import OpenAirCongestionEnv; assert OpenAirCongestionEnv is not None",
        ],
        cwd=RESOURCE_DIR,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_import_guard_does_not_mask_missing_transitive_dependency(tmp_path) -> None:
    """A broken dependency must not be reported as a missing colocated package."""

    fake_package = tmp_path / "openair_congestion"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text("import fable_missing_dependency\n")
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(tmp_path), str(RESOURCE_DIR))),
    }
    completed = subprocess.run(
        [sys.executable, "-c", "import backends"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "fable_missing_dependency" in completed.stderr
    assert "Could not import the colocated" not in completed.stderr
