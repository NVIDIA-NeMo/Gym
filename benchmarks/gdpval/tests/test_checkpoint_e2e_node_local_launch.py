# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path


PACKAGE = Path(__file__).parents[1] / "hsg" / "checkpoint_e2e"
EXISTING_LAUNCHER = PACKAGE / "run_existing_judge.sh"
EXISTING_CONTROLLER = PACKAGE / "existing_judge_controller.sbatch"
MARS_HELPER = PACKAGE / "mars_node_local.sh"


def test_existing_judge_controller_receives_execution_package() -> None:
    launcher = EXISTING_LAUNCHER.read_text(encoding="utf-8")
    controller = EXISTING_CONTROLLER.read_text(encoding="utf-8")
    lines = launcher.splitlines()
    controller_index = next(
        index
        for index, line in enumerate(lines)
        if '"$ACTIVE_PACKAGE/existing_judge_controller.sbatch"' in line
    )

    assert ': "${CHECKPOINT_E2E_EXECUTION_PACKAGE:?set CHECKPOINT_E2E_EXECUTION_PACKAGE}"' in controller
    assert (
        'CHECKPOINT_E2E_EXECUTION_PACKAGE="$ACTIVE_PACKAGE"'
        in lines[controller_index - 1]
    )


def test_node_local_cache_marker_read_is_idempotent(tmp_path: Path) -> None:
    marker = tmp_path / ".mars-ready"
    script = r"""
set -euo pipefail
source "$1"
marker=$2
printf '%s\n' expected > "$marker"
chmod 0400 "$marker"
mars_marker_matches "$marker" expected
! mars_marker_matches "$marker" stale
ln -s "$marker" "$marker.link"
! mars_marker_matches "$marker.link" expected
"""

    subprocess.run(
        ["bash", "-c", script, "marker-test", str(MARS_HELPER), str(marker)],
        check=True,
    )
    helper = MARS_HELPER.read_text(encoding="utf-8")
    assert '$(<"$marker" 2>/dev/null)' not in helper


def test_affected_shell_entrypoints_parse() -> None:
    subprocess.run(
        [
            "bash",
            "-n",
            str(EXISTING_LAUNCHER),
            str(EXISTING_CONTROLLER),
            str(MARS_HELPER),
        ],
        check=True,
    )
