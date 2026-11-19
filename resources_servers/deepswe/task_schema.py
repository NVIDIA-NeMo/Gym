# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Narrow Pier-compatible schema for the pinned DeepSWE v1.1 task format.

The field and resolution behavior here is adapted from datacurve-pier 0.3.1's
Apache-2.0-licensed ``pier.models.task`` implementation. DeepSWE needs only
task loading; keeping that small contract local avoids pulling Pier's unrelated
agent and model-provider dependency graph into the Gym resources server.
"""

from __future__ import annotations

import re
import tomllib
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


MAIN_SERVICE_NAME = "main"
_CANARY_LINE_RE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)
_PACKAGE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$")


class NetworkMode(str, Enum):
    NO_NETWORK = "no-network"
    PUBLIC = "public"
    ALLOWLIST = "allowlist"


class VerifierEnvironmentMode(str, Enum):
    SHARED = "shared"
    SEPARATE = "separate"


class PackageInfo(BaseModel):
    name: str
    description: str = ""
    authors: list[dict[str, Any]] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not _PACKAGE_NAME_RE.match(value) or ".." in value:
            raise ValueError(f"Invalid task package name: {value!r}")
        return value


class EnvironmentConfig(BaseModel):
    build_timeout_sec: float = 600.0
    docker_image: str | None = None
    os: str = "linux"
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int = 10240
    gpus: int = 0
    network_mode: NetworkMode | None = None
    env: dict[str, str] = Field(default_factory=dict)
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)


class AgentConfig(BaseModel):
    timeout_sec: float | None = None
    network_mode: NetworkMode | None = None


class VerifierCollectConfig(BaseModel):
    command: str
    service: str = MAIN_SERVICE_NAME
    timeout_sec: float = 60.0
    user: str | int | None = None


class VerifierConfig(BaseModel):
    timeout_sec: float = 600.0
    network_mode: NetworkMode | None = None
    environment_mode: VerifierEnvironmentMode | None = None
    environment: EnvironmentConfig | None = None
    env: dict[str, str] = Field(default_factory=dict)
    collect: list[VerifierCollectConfig] = Field(default_factory=list)


class TaskConfig(BaseModel):
    schema_version: str = "1.2"
    task: PackageInfo | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    solution: dict[str, Any] = Field(default_factory=dict)
    steps: list[dict[str, Any]] | None = None
    artifacts: list[str | dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def model_validate_toml(cls, toml_data: str) -> "TaskConfig":
        return cls.model_validate(tomllib.loads(toml_data))


class TaskPaths:
    def __init__(self, task_dir: Path) -> None:
        self.task_dir = task_dir

    @property
    def config_path(self) -> Path:
        return self.task_dir / "task.toml"

    @property
    def instruction_path(self) -> Path:
        return self.task_dir / "instruction.md"

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / "tests"

    @property
    def solution_dir(self) -> Path:
        return self.task_dir / "solution"


def strip_canary(text: str) -> str:
    lines = text.split("\n")
    index = 0
    while index < len(lines) and _CANARY_LINE_RE.match(lines[index].strip()):
        index += 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    return "\n".join(lines[index:])


class Task:
    def __init__(self, task_dir: Path | str) -> None:
        self.task_dir = Path(task_dir).resolve()
        self.paths = TaskPaths(self.task_dir)
        self.config = TaskConfig.model_validate_toml(self.paths.config_path.read_text(encoding="utf-8"))
        self.name = self.config.task.name if self.config.task is not None else self.task_dir.name
        self.instruction = (
            "" if self.has_steps else strip_canary(self.paths.instruction_path.read_text(encoding="utf-8"))
        )

    @property
    def has_steps(self) -> bool:
        return bool(self.config.steps)


def resolve_effective_verifier_env_config(
    task_config: TaskConfig,
    step_config: Any | None,
) -> EnvironmentConfig | None:
    if step_config is not None:
        raise ValueError("DeepSWE v1.1 does not support stepped tasks")
    verifier = task_config.verifier
    mode = verifier.environment_mode
    if mode is None:
        mode = VerifierEnvironmentMode.SEPARATE if verifier.environment is not None else VerifierEnvironmentMode.SHARED
    if mode != VerifierEnvironmentMode.SEPARATE:
        return None
    return verifier.environment or task_config.environment.model_copy(deep=True)
