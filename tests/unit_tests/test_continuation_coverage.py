# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks/nemotron_3.5_super/check_continuation.py"


def coverage(tmp_path: Path, *, expected: int, successful: int, percent: int = 10) -> subprocess.CompletedProcess[str]:
    (tmp_path / "run_materialized_inputs.jsonl").write_text("{}\n" * expected)
    (tmp_path / "run.jsonl").write_text("{}\n" * successful)
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "run.jsonl"), "--max-missing-percent", str(percent)],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("expected", "successful", "percent", "status"),
    [
        (100, 100, 10, 0),
        (100, 90, 10, 75),
        (100, 89, 10, 76),
        (1, 0, 10, 76),
        (100, 99, 0, 76),
        (0, 0, 10, 78),
        (10, 11, 10, 78),
        (100, 99, 101, 78),
    ],
)
def test_coverage_thresholds(tmp_path: Path, expected: int, successful: int, percent: int, status: int) -> None:
    result = coverage(tmp_path, expected=expected, successful=successful, percent=percent)
    assert result.returncode == status, result.stderr


def test_continuation_requires_progress_and_finishes(tmp_path: Path) -> None:
    assert coverage(tmp_path, expected=100, successful=90).returncode == 75
    assert coverage(tmp_path, expected=100, successful=91).returncode == 75
    assert coverage(tmp_path, expected=100, successful=91).returncode == 77
    assert coverage(tmp_path, expected=100, successful=90).returncode == 77
    assert coverage(tmp_path, expected=100, successful=100).returncode == 0
    assert (tmp_path / "run_continuation_success_count").read_text() == "91\n"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("state", ["corrupt", "-1", "101"])
def test_invalid_saved_count_is_not_reset(tmp_path: Path, state: str) -> None:
    (tmp_path / "run_continuation_success_count").write_text(state)
    result = coverage(tmp_path, expected=100, successful=99)
    assert result.returncode == 78, result.stderr
    assert (tmp_path / "run_continuation_success_count").read_text() == state


@pytest.mark.parametrize("content", [None, "", "{}\n{broken", "[]\n", "\n"])
def test_missing_or_invalid_materialized_rows(tmp_path: Path, content: str | None) -> None:
    if content is not None:
        (tmp_path / "run_materialized_inputs.jsonl").write_text(content)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "run.jsonl"), "--max-missing-percent", "10"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 78, result.stderr
    assert not (tmp_path / "run_continuation_success_count").exists()
