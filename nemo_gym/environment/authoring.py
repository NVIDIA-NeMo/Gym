# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load authored ``environment.yaml`` definitions and materialize their tasks."""

from __future__ import annotations

import importlib
import importlib.util
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, model_validator

from nemo_gym.config_types import ConfigError
from nemo_gym.episode_types import MaterializedTask, TaskId
from nemo_gym.single_agent_episode_types import (
    SINGLE_AGENT_TASK_INPUT_CONTRACT,
    SingleAgentTaskInput,
)


class EnvironmentDefinitionError(ConfigError):
    """An authored environment definition is invalid or unsupported."""


class _DefinitionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class VerifierDefinition(_DefinitionModel):
    implementation: str = Field(min_length=1)
    verifier_input: dict[str, JsonValue] = Field(default_factory=dict)


class TaskDefinition(_DefinitionModel):
    id: str | None = Field(default=None, min_length=1)
    instruction: str = Field(min_length=1)
    verifier: VerifierDefinition
    task_data: dict[str, JsonValue] = Field(default_factory=dict)


class TasksetDefinition(_DefinitionModel):
    tasks: dict[str, TaskDefinition] = Field(min_length=1)


class MCPServerDefinition(_DefinitionModel):
    name: str = Field(min_length=1)
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)


class RuntimeDefinition(_DefinitionModel):
    dockerfile: str = Field(min_length=1)
    workdir: str = Field(min_length=1, pattern=r"^/")
    mcp_servers: list[MCPServerDefinition] = Field(default_factory=list)


class EnvironmentDefinition(_DefinitionModel):
    """Authored tasks, runtime requirements, and verification."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    license: str = Field(min_length=1)
    episode_protocol: str = Field(min_length=1)

    task: TaskDefinition | None = None
    instruction: str | None = Field(default=None, min_length=1)
    task_model: str | None = Field(default=None, min_length=1)
    tasksets: dict[str, str | TasksetDefinition] = Field(default_factory=dict)
    verifier: VerifierDefinition | None = None
    runtime: RuntimeDefinition

    @model_validator(mode="after")
    def validate_task_sources(self) -> "EnvironmentDefinition":
        if (self.task is None) == (not self.tasksets):
            raise ValueError("declare exactly one of task or tasksets")
        file_tasksets = [name for name, declaration in self.tasksets.items() if isinstance(declaration, str)]
        if file_tasksets and (self.instruction is None or self.verifier is None):
            raise ValueError("file-backed tasksets require top-level instruction and verifier declarations")
        return self


@dataclass(frozen=True)
class LoadedEnvironment:
    """A validated environment definition plus its trusted root."""

    root: Path
    definition_path: Path
    definition: EnvironmentDefinition

    def resolve_file(self, reference: str, *, description: str) -> Path:
        candidate = Path(reference)
        if candidate.is_absolute():
            raise EnvironmentDefinitionError(f"{description} must be relative to the environment root: {reference}")
        resolved = (self.root / candidate).resolve()
        if not resolved.is_relative_to(self.root):
            raise EnvironmentDefinitionError(f"{description} escapes the environment root: {reference}")
        if not resolved.is_file():
            raise EnvironmentDefinitionError(f"{description} was not found: {resolved}")
        return resolved


def load_environment(path: str | Path) -> LoadedEnvironment:
    """Load an environment root or an explicit ``environment.yaml`` path."""

    requested = Path(path).expanduser()
    definition_path = requested / "environment.yaml" if requested.is_dir() else requested
    definition_path = definition_path.resolve()
    try:
        raw = yaml.safe_load(definition_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise EnvironmentDefinitionError(f"Environment definition was not found: {definition_path}") from error
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise EnvironmentDefinitionError(
            f"Could not read environment definition {definition_path}: {error}"
        ) from error
    if not isinstance(raw, Mapping):
        raise EnvironmentDefinitionError(f"Environment definition must contain a YAML mapping: {definition_path}")
    try:
        environment = EnvironmentDefinition.model_validate(raw)
    except ValidationError as error:
        issues = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'environment'}: {item['msg']}"
            for item in error.errors(include_url=False, include_context=False, include_input=False)
        )
        raise EnvironmentDefinitionError(f"Invalid environment definition {definition_path}: {issues}") from error

    loaded = LoadedEnvironment(
        root=definition_path.parent.resolve(),
        definition_path=definition_path,
        definition=environment,
    )
    loaded.resolve_file(environment.runtime.dockerfile, description="runtime.dockerfile")
    if environment.task is not None:
        loaded.resolve_file(environment.task.instruction, description="task.instruction")
        _validate_local_object_reference(loaded, environment.task.verifier.implementation, "task verifier")
    return loaded


def materialize_single_task(loaded: LoadedEnvironment) -> MaterializedTask[SingleAgentTaskInput]:
    """Compile a singleton task into the built-in single-agent wire contract."""

    environment = loaded.definition
    if environment.episode_protocol != SINGLE_AGENT_TASK_INPUT_CONTRACT:
        raise EnvironmentDefinitionError(
            f"Unsupported episode protocol {environment.episode_protocol!r}; "
            f"expected {SINGLE_AGENT_TASK_INPUT_CONTRACT!r}"
        )
    if environment.task is None:
        raise EnvironmentDefinitionError("materialize_single_task requires a singleton task declaration")
    task = environment.task
    if task.id is None:
        raise EnvironmentDefinitionError("a singleton task requires task.id")
    instruction = loaded.resolve_file(task.instruction, description="task.instruction").read_text(encoding="utf-8")
    return MaterializedTask[SingleAgentTaskInput](
        task_id=TaskId(taskset=environment.name, task_id=task.id, revision=environment.version),
        task_input=SingleAgentTaskInput(
            responses_create_params={
                "input": [{"role": "user", "content": instruction}],
            },
            task_data=task.task_data,
        ),
    )


def materialize_single_task_jsonl(loaded: LoadedEnvironment) -> str:
    """Serialize the singleton materialized task as one JSONL row."""

    task = materialize_single_task(loaded)
    return json.dumps(task.model_dump(mode="json", exclude_none=True), separators=(",", ":")) + "\n"


def load_environment_object(
    loaded: LoadedEnvironment,
    reference: str,
    *,
    description: str,
) -> Any:
    """Resolve a trusted environment-local file or importable module object."""

    module_reference, separator, object_name = reference.partition(":")
    if not separator or not module_reference or not object_name:
        raise EnvironmentDefinitionError(f"{description} must have the form module-or-file:object")
    if module_reference.endswith(".py") or "/" in module_reference:
        path = loaded.resolve_file(module_reference, description=description)
        digest = sha256(str(path).encode()).hexdigest()[:16]
        module_name = f"_nemo_gym_environment_{digest}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise EnvironmentDefinitionError(f"Could not load {description}: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_reference)
    return _resolve_object(module, object_name, description)


def load_environment_callable(
    loaded: LoadedEnvironment,
    reference: str,
    *,
    description: str,
) -> Callable[..., Any]:
    value = load_environment_object(loaded, reference, description=description)
    if not callable(value):
        raise EnvironmentDefinitionError(f"{description} is not callable: {reference}")
    return value


def _resolve_object(module: ModuleType, object_name: str, description: str) -> Any:
    value: Any = module
    for part in object_name.split("."):
        try:
            value = getattr(value, part)
        except AttributeError as error:
            raise EnvironmentDefinitionError(
                f"{description} object {object_name!r} was not found in {module.__name__!r}"
            ) from error
    return value


def _validate_local_object_reference(
    loaded: LoadedEnvironment,
    reference: str,
    description: str,
) -> None:
    module_reference, separator, object_name = reference.partition(":")
    if not separator or not module_reference or not object_name:
        raise EnvironmentDefinitionError(f"{description} must have the form module-or-file:object")
    if module_reference.endswith(".py") or "/" in module_reference:
        loaded.resolve_file(module_reference, description=description)


__all__ = [
    "EnvironmentDefinition",
    "EnvironmentDefinitionError",
    "LoadedEnvironment",
    "load_environment",
    "load_environment_callable",
    "load_environment_object",
    "materialize_single_task",
    "materialize_single_task_jsonl",
]
