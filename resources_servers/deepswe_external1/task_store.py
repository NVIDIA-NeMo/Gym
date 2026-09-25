# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated control-plane task packages for the DeepSWE execution contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from resources_servers.deepswe.task_schema import (
    AgentConfig,
    EnvironmentConfig,
    NetworkMode,
    Task,
    TaskConfig,
    TaskPaths,
    VerifierCollectConfig,
    VerifierConfig,
    VerifierEnvironmentMode,
)


ASSET_PATHS = (
    "instruction.md",
    "tests/test.sh",
    "tests/test.patch",
    "tests/grader.py",
    "tests/config.json",
    "solution/solve.sh",
    "solution/solution.patch",
)
MAX_ASSET_BYTES = 64 * 1024 * 1024


class AssetDigest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    mode: int = Field(default=0o644, ge=0, le=0o777)


class PhaseLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cpus: int = Field(gt=0)
    memory_mb: int = Field(gt=0)
    storage_mb: int = Field(gt=0)
    timeout_sec: float = Field(gt=0, allow_inf_nan=False)
    env: dict[str, str] = Field(default_factory=dict)


class TaskDefinition(BaseModel):
    """Prepared metadata; original test/solution files remain outside model input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
    image: str = Field(min_length=1)
    verifier_image: str = Field(min_length=1)
    workdir: Literal["/app"] = "/app"
    base_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    agent: PhaseLimits
    verifier: PhaseLimits
    solution_timeout_sec: float = Field(default=1800, gt=0, allow_inf_nan=False)
    collect_timeout_sec: float = Field(default=300, gt=0, allow_inf_nan=False)
    assets: dict[str, AssetDigest]

    @model_validator(mode="after")
    def exact_assets(self) -> TaskDefinition:
        if set(self.assets) != set(ASSET_PATHS):
            raise ValueError(
                "Prepared tasks must declare exactly the instruction, four verifier and two solution files"
            )
        return self

    def fingerprint(self) -> str:
        """Identify image selection, resources, base commit and every trusted asset."""
        serialized = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()


class PreparedTask(Task):
    """Expose prepared task packages through the existing DeepSWE task interface."""

    def __init__(self, task_dir: Path) -> None:
        if task_dir.is_symlink():
            raise ValueError("Prepared task directories must not be symlinks")
        self.task_dir = task_dir.resolve()
        self.paths = TaskPaths(self.task_dir)
        metadata = self.task_dir / "task.json"
        if metadata.is_symlink() or metadata.stat().st_size > 1024 * 1024:
            raise ValueError("Invalid prepared task metadata file")
        self.definition = TaskDefinition.model_validate_json(metadata.read_bytes())
        if self.definition.task_id != self.task_dir.name:
            raise ValueError("Prepared task directory and task ID disagree")
        self.validate_assets()
        self.name = self.definition.task_id
        # Unlike the benchmark loader, preserve the exact delivered prompt, including canaries.
        self.instruction = self.asset_path("instruction.md").read_text(encoding="utf-8")
        agent = self.definition.agent
        verifier = self.definition.verifier
        self.config = TaskConfig(
            metadata={"task_id": self.definition.task_id, "base_commit_hash": self.definition.base_commit},
            environment=EnvironmentConfig(
                docker_image=self.definition.image,
                cpus=agent.cpus,
                memory_mb=agent.memory_mb,
                storage_mb=agent.storage_mb,
                env=agent.env,
            ),
            agent=AgentConfig(timeout_sec=agent.timeout_sec, network_mode=NetworkMode.NO_NETWORK),
            verifier=VerifierConfig(
                timeout_sec=verifier.timeout_sec,
                network_mode=NetworkMode.NO_NETWORK,
                environment_mode=VerifierEnvironmentMode.SEPARATE,
                environment=EnvironmentConfig(
                    docker_image=self.definition.verifier_image,
                    cpus=verifier.cpus,
                    memory_mb=verifier.memory_mb,
                    storage_mb=verifier.storage_mb,
                    env=verifier.env,
                ),
                collect=[
                    VerifierCollectConfig(
                        command=(
                            "set -eu; cd /app; mkdir -p /logs/artifacts; "
                            "git config --global --add safe.directory /app; "
                            "git diff --binary --no-ext-diff --no-textconv --no-color "
                            f"{self.definition.base_commit} HEAD -- . > /logs/artifacts/model.patch"
                        ),
                        timeout_sec=self.definition.collect_timeout_sec,
                    )
                ],
            ),
        )

    def asset_path(self, relative_path: str) -> Path:
        """Return an allowlisted regular file without following asset symlinks."""
        if relative_path not in self.definition.assets:
            raise ValueError("Undeclared task asset")
        candidate = self.task_dir / relative_path
        relative = candidate.relative_to(self.task_dir)
        current = self.task_dir
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                raise ValueError("Task assets and their parent directories must not be symlinks")
        if not candidate.is_file() or candidate.stat().st_size > MAX_ASSET_BYTES:
            raise ValueError("Task asset is missing, not regular, or oversized")
        return candidate

    def validate_assets(self) -> None:
        """Check all bytes against the preparation manifest before using this task."""
        for name, expected in self.definition.assets.items():
            asset = self.asset_path(name)
            if hashlib.sha256(asset.read_bytes()).hexdigest() != expected.sha256:
                raise ValueError(f"Task asset checksum changed: {name}")
            if asset.stat().st_mode & 0o777 != expected.mode:
                raise ValueError(f"Task asset permissions changed: {name}")
        config = json.loads(self.asset_path("tests/config.json").read_text(encoding="utf-8"))
        if config.get("base_commit") != self.definition.base_commit:
            raise ValueError("Prepared base commit disagrees with the native grader")


class PreparedTaskStore:
    """ID lookup over immutable, privately prepared task packages."""

    def __init__(self, tasks_dir: Path, *, expected_task_count: int) -> None:
        if not tasks_dir.is_dir():
            raise FileNotFoundError(f"Prepared task directory does not exist: {tasks_dir}")
        packages = sorted(path.parent for path in tasks_dir.glob("*/task.json"))
        if len(packages) != expected_task_count:
            raise ValueError(f"Expected {expected_task_count} prepared tasks, found {len(packages)}")
        self._tasks = {path.name: PreparedTask(path) for path in packages}

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[PreparedTask]:
        return iter(self._tasks.values())

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(self._tasks)

    def get(self, task_id: str) -> PreparedTask:
        return self._tasks[task_id]
