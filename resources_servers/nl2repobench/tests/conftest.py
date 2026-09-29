# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest


def _write_task(task_dir: Path, *, name: str, test_case_count: int = 5) -> None:
    task_dir.mkdir(parents=True)
    (task_dir / "start.md").write_text(f"# {name}\n\nImplement a tiny package called {name}.\n", encoding="utf-8")
    (task_dir / "test_case_count.txt").write_text(str(test_case_count), encoding="utf-8")
    (task_dir / "test_commands.json").write_text(
        json.dumps(["pip install -e .", "pytest --continue-on-collection-errors -n 0 tests -v -s"]),
        encoding="utf-8",
    )
    (task_dir / "test_files.json").write_text(json.dumps(["tests"]), encoding="utf-8")


@pytest.fixture
def task_assets(tmp_path: Path) -> Path:
    """A single synthetic NL2RepoBench task directory (mirrors the upstream flat layout)."""

    tasks_dir = tmp_path / "tasks"
    _write_task(tasks_dir / "example-task", name="example-task", test_case_count=5)
    return tasks_dir


@pytest.fixture
def two_task_assets(tmp_path: Path) -> Path:
    """Two synthetic NL2RepoBench task directories, for expected_task_count mismatch tests."""

    tasks_dir = tmp_path / "tasks"
    _write_task(tasks_dir / "example-task-one", name="example-task-one", test_case_count=5)
    _write_task(tasks_dir / "example-task-two", name="example-task-two", test_case_count=10)
    return tasks_dir
