# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures of the Compose tests.

The main and sidecar image references, their recorded configurations, the overlay document and
the fixture loader that test_compose_config.py, test_compose_fields.py and test_compose_volumes.py
all read. Not a pytest conftest, so each test module imports the names it uses.
"""

from pathlib import Path


"""A stand-in for a provider's CLI binary whose command can be made to hang, for the runners' cancellation tests."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest


requires_sh = pytest.mark.skipif(
    shutil.which("sh") is None or shutil.which("sleep") is None, reason="sh and sleep not available"
)


@dataclass(frozen=True)
class StandInCli:
    """``path`` records its PID in ``pid_file``, then hangs while ``hang`` exists and exits 0 otherwise."""

    path: Path
    pid_file: Path
    hang: Path

    def last_pid(self) -> int:
        return int(self.pid_file.read_text())


def stand_in_cli(tmp_path: Path) -> StandInCli:
    cli = StandInCli(path=tmp_path / "cli", pid_file=tmp_path / "pid", hang=tmp_path / "hang")
    cli.path.write_text(f'#!/bin/sh\necho $$ > "{cli.pid_file}"\n[ -e "{cli.hang}" ] && exec sleep 600\nexit 0\n')
    cli.path.chmod(0o755)
    cli.hang.touch()
    return cli


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
