# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NL2RepoBench-specific validation and lookup around the flat task schema."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from resources_servers.nl2repobench.task_schema import Task


# Verified against a live shallow clone of the upstream repository on 2026-09-07:
# `test_files/` contains exactly 104 task subdirectories.
EXPECTED_TASK_COUNT = 104
# Pinned upstream commit, discovered via `git rev-parse HEAD` against a fresh shallow clone of
# https://github.com/multimodal-art-projection/NL2RepoBench.
NL2REPOBENCH_SOURCE_REVISION = "781a1da1ee41fb8edb0bed22f586d69111610edf"  # pragma: allowlist secret
REQUIRED_TASK_FILES = ("start.md", "test_case_count.txt", "test_commands.json", "test_files.json")


def task_id(task: Task) -> str:
    """Return the NL2RepoBench task ID (upstream ``proName``, i.e. its directory name)."""

    return task.name


def task_image(task: Task) -> str:
    """Return the pinned, immutable per-task base image."""

    return f"ghcr.io/multimodal-art-projection/nl2repobench/{task.name}:1.0"


def _validate_nl2repobench_task(task_dir: Path) -> None:
    missing = [relative_path for relative_path in REQUIRED_TASK_FILES if not (task_dir / relative_path).is_file()]
    if missing:
        raise FileNotFoundError(f"NL2RepoBench task {task_dir.name!r} is missing required files: {', '.join(missing)}")
    symlinks = [relative_path for relative_path in REQUIRED_TASK_FILES if (task_dir / relative_path).is_symlink()]
    if symlinks:
        raise ValueError(
            f"NL2RepoBench task {task_dir.name!r} contains unsupported symlink assets: {', '.join(symlinks)}"
        )


class NL2RepoBenchTaskStore:
    """Immutable ID index over validated NL2RepoBench task objects."""

    def __init__(
        self,
        tasks_dir: str | Path,
        *,
        expected_task_count: int = EXPECTED_TASK_COUNT,
    ) -> None:
        self.tasks_dir = Path(tasks_dir).expanduser().resolve()
        if not self.tasks_dir.is_dir():
            raise FileNotFoundError(f"NL2RepoBench tasks directory does not exist: {self.tasks_dir}")

        task_dirs = sorted(path for path in self.tasks_dir.iterdir() if path.is_dir())
        for task_dir in task_dirs:
            _validate_nl2repobench_task(task_dir)

        if len(task_dirs) != expected_task_count:
            raise ValueError(
                f"Expected {expected_task_count} NL2RepoBench tasks in {self.tasks_dir}, found {len(task_dirs)}"
            )

        tasks = [Task(task_dir) for task_dir in task_dirs]
        for task in tasks:
            if task.test_case_count <= 0:
                raise ValueError(f"NL2RepoBench task {task.name!r} has non-positive test_case_count")
            if not task.test_commands.commands:
                raise ValueError(f"NL2RepoBench task {task.name!r} has an empty test_commands list")
            if not task.test_files.files:
                raise ValueError(f"NL2RepoBench task {task.name!r} has an empty test_files list")
        self._tasks = {task_id(task): task for task in tasks}

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[Task]:
        return iter(self._tasks.values())

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def get(self, current_task_id: str) -> Task:
        try:
            return self._tasks[current_task_id]
        except KeyError as error:
            raise KeyError(f"Unknown NL2RepoBench task id: {current_task_id!r}") from error
