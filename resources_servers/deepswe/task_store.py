# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSWE-specific validation and lookup around its Pier-compatible task model."""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

from resources_servers.deepswe.task_schema import (
    MAIN_SERVICE_NAME,
    NetworkMode,
    Task,
    VerifierCollectConfig,
    VerifierEnvironmentMode,
    resolve_effective_verifier_env_config,
)


EXPECTED_TASK_COUNT = 113
DEEPSWE_SOURCE_REVISION = "435ee89ec2f2e2289f33b0da4f992f0b7b7266b9"  # pragma: allowlist secret
REQUIRED_TASK_FILES = (
    "task.toml",
    "instruction.md",
    "tests/test.sh",
    "tests/test.patch",
    "tests/grader.py",
    "tests/config.json",
    "solution/solution.patch",
)


def task_id(task: Task) -> str:
    """Return the DeepSWE task ID carried by validated metadata."""

    value = task.config.metadata.get("task_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"DeepSWE task {task.task_dir.name!r} is missing metadata.task_id")
    return value


def task_image(task: Task) -> str:
    """Return the immutable agent image declared by the upstream task."""

    value = task.config.environment.docker_image
    if not isinstance(value, str) or not value:
        raise ValueError(f"DeepSWE task {task_id(task)!r} is missing environment.docker_image")
    return value


def task_collect_hook(task: Task) -> VerifierCollectConfig:
    """Return the single upstream commit-only verifier collect hook."""

    hooks = task.config.verifier.collect
    if len(hooks) != 1:
        raise ValueError(f"DeepSWE task {task_id(task)!r} must define exactly one [[verifier.collect]] hook")
    return hooks[0]


def task_sandbox_resources(task: Task, *, phase: str) -> dict[str, float | int]:
    """Resolve Pier's phase-specific task resources into a Gym sandbox request."""

    if phase == "agent":
        environment = task.config.environment
    elif phase in {"verifier", "golden-verifier"}:
        environment = resolve_effective_verifier_env_config(task.config, None)
        if environment is None:
            raise ValueError(f"DeepSWE task {task_id(task)!r} must use a separate verifier environment")
    else:
        raise ValueError(f"Unknown DeepSWE sandbox phase: {phase!r}")

    cpu = environment.cpus
    memory_mib = environment.memory_mb
    storage_mib = environment.storage_mb
    if cpu is None or cpu <= 0 or memory_mib is None or memory_mib <= 0 or storage_mib <= 0:
        raise ValueError(f"DeepSWE task {task_id(task)!r} has invalid {phase} sandbox resource limits")
    return {
        "cpu": cpu,
        "memory_mib": memory_mib,
        "disk_gib": math.ceil(storage_mib / 1024),
    }


def task_verifier_files(task: Task) -> dict[str, Path]:
    """Return the held-out files staged into the fresh verifier sandbox."""

    return {
        "test.sh": task.paths.tests_dir / "test.sh",
        "test.patch": task.paths.tests_dir / "test.patch",
        "grader.py": task.paths.tests_dir / "grader.py",
        "config.json": task.paths.tests_dir / "config.json",
    }


def task_solution_patch_path(task: Task) -> Path:
    """Return the held-out oracle patch path."""

    return task.paths.solution_dir / "solution.patch"


def _validate_deepswe_task(task: Task, *, directory_name: str) -> None:
    """Enforce the fixed DeepSWE v1.1 contract beyond Pier's generic schema."""

    current_task_id = task_id(task)
    if current_task_id != directory_name:
        raise ValueError(
            f"DeepSWE task directory {directory_name!r} disagrees with metadata.task_id {current_task_id!r}"
        )

    missing = [relative_path for relative_path in REQUIRED_TASK_FILES if not (task.task_dir / relative_path).is_file()]
    if missing:
        raise FileNotFoundError(f"DeepSWE task {current_task_id!r} is missing required files: {', '.join(missing)}")
    symlinks = [relative_path for relative_path in REQUIRED_TASK_FILES if (task.task_dir / relative_path).is_symlink()]
    if symlinks:
        raise ValueError(
            f"DeepSWE task {current_task_id!r} contains unsupported symlink assets: {', '.join(symlinks)}"
        )

    config = task.config
    if config.schema_version != "1.3":
        raise ValueError(f"DeepSWE task {current_task_id!r} must use schema_version 1.3")
    if task.has_steps:
        raise ValueError(f"DeepSWE task {current_task_id!r} must be a single-step task")
    if config.verifier.environment_mode != VerifierEnvironmentMode.SEPARATE:
        raise ValueError(f"DeepSWE task {current_task_id!r} must use a separate verifier environment")
    if config.verifier.network_mode != NetworkMode.NO_NETWORK or config.agent.network_mode != NetworkMode.NO_NETWORK:
        raise ValueError(f"DeepSWE task {current_task_id!r} must disable agent and verifier internet access")

    base_commit = config.metadata.get("base_commit_hash")
    repository_url = config.metadata.get("repository_url")
    language = config.metadata.get("language")
    if (
        not isinstance(base_commit, str)
        or not 7 <= len(base_commit) <= 40
        or any(character not in "0123456789abcdef" for character in base_commit)
    ):
        raise ValueError(f"DeepSWE task {current_task_id!r} has an invalid base commit: {base_commit!r}")
    if not isinstance(repository_url, str) or not repository_url or not isinstance(language, str) or not language:
        raise ValueError(f"DeepSWE task {current_task_id!r} is missing repository_url or language")

    collect = task_collect_hook(task)
    expected_collect_command = (
        "cd /app && mkdir -p /logs/artifacts && git config --global --add safe.directory /app "
        f"&& git diff --binary {base_commit} HEAD > /logs/artifacts/model.patch"
    )
    if collect.service != MAIN_SERVICE_NAME or collect.command != expected_collect_command:
        raise ValueError(f"DeepSWE task {current_task_id!r} has an unexpected verifier collect hook")
    if collect.timeout_sec <= 0:
        raise ValueError(f"DeepSWE task {current_task_id!r} has an invalid verifier collect timeout")
    if config.verifier.timeout_sec <= 0:
        raise ValueError(f"DeepSWE task {current_task_id!r} has an invalid verifier timeout")
    if config.agent.timeout_sec is None or config.agent.timeout_sec <= 0:
        raise ValueError(f"DeepSWE task {current_task_id!r} has an invalid agent timeout")

    task_image(task)
    task_sandbox_resources(task, phase="agent")
    task_sandbox_resources(task, phase="verifier")


class DeepSWETaskStore:
    """Immutable ID index over validated DeepSWE task objects."""

    def __init__(
        self,
        tasks_dir: str | Path,
        *,
        expected_task_count: int = EXPECTED_TASK_COUNT,
    ) -> None:
        self.tasks_dir = Path(tasks_dir).expanduser().resolve()
        if not self.tasks_dir.is_dir():
            raise FileNotFoundError(f"DeepSWE tasks directory does not exist: {self.tasks_dir}")

        task_dirs = sorted(path.parent for path in self.tasks_dir.glob("*/task.toml"))
        if len(task_dirs) != expected_task_count:
            raise ValueError(
                f"Expected {expected_task_count} DeepSWE tasks in {self.tasks_dir}, found {len(task_dirs)}"
            )

        tasks = [Task(task_dir) for task_dir in task_dirs]
        for task, task_dir in zip(tasks, task_dirs, strict=True):
            _validate_deepswe_task(task, directory_name=task_dir.name)
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
            raise KeyError(f"Unknown DeepSWE task id: {current_task_id!r}") from error
