# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


PACKAGE = Path(__file__).parents[1] / "hsg" / "checkpoint_e2e"
PORT_HELPER = PACKAGE / "judge_ports.sh"


def _select_window(tmp_path: Path, value: str) -> subprocess.CompletedProcess[str]:
    range_file = tmp_path / "ip_local_port_range"
    range_file.write_text(value, encoding="utf-8")
    return subprocess.run(
        [
            "bash",
            "-c",
            """
source "$1"
if gdpval_select_judge_port_window "$2"; then
    printf '%s:%s:%s:%s\n' \
        "$GDPVAL_JUDGE_PORT_BASE" \
        "$GDPVAL_JUDGE_PORT_SLOT_WIDTH" \
        "$GDPVAL_JUDGE_PORT_SLOT_COUNT" \
        "$GDPVAL_JUDGE_PORT_WINDOW_HIGH"
else
    exit $?
fi
""",
            "judge-port-test",
            str(PORT_HELPER),
            str(range_file),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


@pytest.mark.parametrize("ephemeral_range", ["9000 65000\n", "32768 60999\n"])
def test_judge_ports_use_the_complete_bounded_window_below_ephemeral_ports(
    tmp_path: Path, ephemeral_range: str
) -> None:
    result = _select_window(tmp_path, ephemeral_range)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "2000:20:200:5999\n"


@pytest.mark.parametrize(
    ("ephemeral_range", "expected"),
    [("3000 4000\n", "4001:20:99:5980\n"), ("4000 5000\n", "2000:20:100:3999\n")],
)
def test_judge_ports_choose_the_larger_safe_side_of_a_split_window(
    tmp_path: Path, ephemeral_range: str, expected: str
) -> None:
    result = _select_window(tmp_path, ephemeral_range)

    assert result.returncode == 0, result.stderr
    assert result.stdout == expected


@pytest.mark.parametrize(
    "ephemeral_range",
    ["", "not-a-port-range\n", "9000\n", "65000 9000\n", "0 65000\n", "9000 70000\n", "9000 65000 extra\n"],
)
def test_judge_ports_reject_malformed_kernel_ranges(tmp_path: Path, ephemeral_range: str) -> None:
    result = _select_window(tmp_path, ephemeral_range)

    assert result.returncode == 64
    assert "GDPVAL_JUDGE_PORT_FAIL:" in result.stderr


def test_judge_ports_fail_when_the_bounded_window_has_no_complete_safe_slot(tmp_path: Path) -> None:
    result = _select_window(tmp_path, "1024 65535\n")

    assert result.returncode == 64
    assert "no complete non-ephemeral 20-port slot exists in 2000-5999" in result.stderr


def test_judge_sources_the_port_helper_before_selecting_and_probing() -> None:
    judge = (PACKAGE / "judge.sbatch").read_text(encoding="utf-8")
    launcher = (PACKAGE / "run_checkpoint_e2e.sh").read_text(encoding="utf-8")

    source = 'source "$E2E_DIR/judge_ports.sh"'
    selection = "gdpval_select_judge_port_window"
    probe = "/dev/tcp/127.0.0.1/$candidate_port"
    assert source in judge
    assert judge.index(source) < judge.index(selection) < judge.index(probe)
    assert "JUDGE_PORT_SLOT_WIDTH=$GDPVAL_JUDGE_PORT_SLOT_WIDTH" in judge
    assert 'JUDGE_PORTS_SH="$SCRIPT_DIR/judge_ports.sh"' in launcher
    assert '"$JUDGE_PORTS_SH"' in launcher
    assert 'JUDGE_PROGRESS_SH="$SCRIPT_DIR/judge_progress.sh"' in launcher
    assert '"$JUDGE_PROGRESS_SH"' in launcher


def test_judge_port_helper_parses_as_bash() -> None:
    subprocess.run(["bash", "-n", str(PORT_HELPER)], check=True)
