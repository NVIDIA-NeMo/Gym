# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated DeepSWE v1.1 task metadata and verifier assets."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_TASK_COUNT = 113
DEEPSWE_SOURCE_REVISION = "435ee89ec2f2e2289f33b0da4f992f0b7b7266b9"
REQUIRED_TASK_FILES = (
    "task.toml",
    "instruction.md",
    "tests/test.sh",
    "tests/test.patch",
    "tests/grader.py",
    "tests/config.json",
    "solution/solution.patch",
)


@dataclass(frozen=True)
class DeepSWETask:
    """One authoritative task definition, kept entirely on the control plane."""

    task_id: str
    task_dir: Path
    image: str
    instruction: str
    repository_url: str
    base_commit: str
    language: str
    verifier_timeout_s: float
    collect_command: str
    collect_timeout_s: float
    agent_timeout_s: float
    cpu: float
    memory_mib: int
    disk_gib: int

    @property
    def verifier_files(self) -> dict[str, Path]:
        return {
            "test.sh": self.task_dir / "tests" / "test.sh",
            "test.patch": self.task_dir / "tests" / "test.patch",
            "grader.py": self.task_dir / "tests" / "grader.py",
            "config.json": self.task_dir / "tests" / "config.json",
        }

    @property
    def solution_patch_path(self) -> Path:
        return self.task_dir / "solution" / "solution.patch"


def _required_mapping(value: Any, *, field: str, task_id: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"DeepSWE task {task_id!r} must define a [{field}] table")
    return value


def _load_task(task_dir: Path) -> DeepSWETask:
    task_toml_path = task_dir / "task.toml"
    with task_toml_path.open("rb") as stream:
        raw = tomllib.load(stream)

    metadata = _required_mapping(raw.get("metadata"), field="metadata", task_id=task_dir.name)
    task_id = str(metadata.get("task_id") or "")
    if task_id != task_dir.name:
        raise ValueError(f"DeepSWE task directory {task_dir.name!r} disagrees with metadata.task_id {task_id!r}")

    missing = [relative_path for relative_path in REQUIRED_TASK_FILES if not (task_dir / relative_path).is_file()]
    if missing:
        raise FileNotFoundError(f"DeepSWE task {task_id!r} is missing required files: {', '.join(missing)}")
    symlinks = [relative_path for relative_path in REQUIRED_TASK_FILES if (task_dir / relative_path).is_symlink()]
    if symlinks:
        raise ValueError(f"DeepSWE task {task_id!r} contains unsupported symlink assets: {', '.join(symlinks)}")

    verifier = _required_mapping(raw.get("verifier"), field="verifier", task_id=task_id)
    agent = _required_mapping(raw.get("agent"), field="agent", task_id=task_id)
    environment = _required_mapping(raw.get("environment"), field="environment", task_id=task_id)
    verifier_environment = _required_mapping(
        verifier.get("environment"), field="verifier.environment", task_id=task_id
    )
    collect_hooks = verifier.get("collect")
    if not isinstance(collect_hooks, list) or len(collect_hooks) != 1:
        raise ValueError(f"DeepSWE task {task_id!r} must define exactly one [[verifier.collect]] hook")
    collect = _required_mapping(collect_hooks[0], field="verifier.collect", task_id=task_id)

    if raw.get("schema_version") != "1.3":
        raise ValueError(f"DeepSWE task {task_id!r} must use schema_version 1.3")
    if verifier.get("environment_mode") != "separate":
        raise ValueError(f"DeepSWE task {task_id!r} must use a separate verifier environment")
    if verifier.get("network_mode") != "no-network" or agent.get("network_mode") != "no-network":
        raise ValueError(f"DeepSWE task {task_id!r} must disable agent and verifier internet access")

    base_commit = str(metadata.get("base_commit_hash") or "")
    repository_url = str(metadata.get("repository_url") or "")
    language = str(metadata.get("language") or "")
    if not 7 <= len(base_commit) <= 40 or any(character not in "0123456789abcdef" for character in base_commit):
        raise ValueError(f"DeepSWE task {task_id!r} has an invalid base commit: {base_commit!r}")
    if not repository_url or not language:
        raise ValueError(f"DeepSWE task {task_id!r} is missing repository_url or language")
    collect_command = str(collect.get("command") or "")
    expected_collect_command = (
        "cd /app && mkdir -p /logs/artifacts && git config --global --add safe.directory /app "
        f"&& git diff --binary {base_commit} HEAD > /logs/artifacts/model.patch"
    )
    if collect_command != expected_collect_command:
        raise ValueError(f"DeepSWE task {task_id!r} has an unexpected verifier collect command")
    collect_timeout_s = float(collect.get("timeout_sec", 0))
    if collect_timeout_s <= 0:
        raise ValueError(f"DeepSWE task {task_id!r} has an invalid verifier collect timeout")
    selected_image = str(environment.get("docker_image") or "")
    if not selected_image:
        raise ValueError(f"DeepSWE task {task_id!r} is missing environment.docker_image")

    cpu = float(verifier_environment.get("cpus", environment.get("cpus", 0)))
    memory_mib = int(verifier_environment.get("memory_mb", environment.get("memory_mb", 0)))
    storage_mb = int(verifier_environment.get("storage_mb", environment.get("storage_mb", 0)))
    if cpu <= 0 or memory_mib <= 0 or storage_mb <= 0:
        raise ValueError(f"DeepSWE task {task_id!r} has invalid sandbox resource limits")

    return DeepSWETask(
        task_id=task_id,
        task_dir=task_dir,
        image=selected_image,
        instruction=(task_dir / "instruction.md").read_text(encoding="utf-8"),
        repository_url=repository_url,
        base_commit=base_commit,
        language=language,
        verifier_timeout_s=float(verifier.get("timeout_sec", 1800)),
        collect_command=collect_command,
        collect_timeout_s=collect_timeout_s,
        agent_timeout_s=float(agent.get("timeout_sec", 5400)),
        cpu=cpu,
        memory_mib=memory_mib,
        disk_gib=math.ceil(storage_mb / 1024),
    )


class DeepSWETaskStore:
    """Immutable, exhaustively validated DeepSWE task lookup."""

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

        self._tasks = {task_dir.name: _load_task(task_dir) for task_dir in task_dirs}

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks.values())

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def get(self, task_id: str) -> DeepSWETask:
        try:
            return self._tasks[task_id]
        except KeyError as error:
            raise KeyError(f"Unknown DeepSWE task id: {task_id!r}") from error
