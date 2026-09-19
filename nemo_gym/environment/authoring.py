# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load ``environment.yaml`` definitions and materialize their tasks."""

from __future__ import annotations

import importlib
import importlib.util
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError, model_validator

from nemo_gym.config_types import ConfigError
from nemo_gym.episode_types import MaterializedTask, TaskId
from nemo_gym.single_agent_episode_types import (
    SINGLE_AGENT_TASK_INPUT_CONTRACT,
    SingleAgentTaskInput,
)


class EnvironmentDefinitionError(ConfigError):
    """An environment definition is invalid or unsupported."""


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


class _TasksetRow(_DefinitionModel):
    task_id: str = Field(min_length=1)
    task_data: dict[str, JsonValue]


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
    episode_protocol: str = Field(
        min_length=1,
        description="Protocol used to materialize and run each task episode.",
    )

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
        if file_tasksets and (self.instruction is None or self.task_model is None or self.verifier is None):
            raise ValueError(
                "file-backed tasksets require top-level instruction, task_model, and verifier declarations"
            )
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


@dataclass(frozen=True)
class MaterializedEnvironmentTask:
    """A protocol-shaped task plus the verifier selected for that task."""

    materialized: MaterializedTask[BaseModel]
    verifier: VerifierDefinition


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
    else:
        if environment.task_model is not None:
            _validate_local_object_reference(loaded, environment.task_model, "task model")
        if environment.instruction is not None:
            loaded.resolve_file(environment.instruction, description="taskset instruction")
        if environment.verifier is not None:
            _validate_local_object_reference(loaded, environment.verifier.implementation, "taskset verifier")
        for taskset_name, taskset in environment.tasksets.items():
            if isinstance(taskset, str):
                loaded.resolve_file(taskset, description=f"tasksets.{taskset_name}")
                continue
            for task_name, task in taskset.tasks.items():
                loaded.resolve_file(
                    task.instruction,
                    description=f"tasksets.{taskset_name}.tasks.{task_name}.instruction",
                )
                _validate_local_object_reference(
                    loaded,
                    task.verifier.implementation,
                    f"tasksets.{taskset_name}.tasks.{task_name}.verifier",
                )
    return loaded


def materialize_tasks(
    loaded: LoadedEnvironment,
    *,
    taskset: str | None = None,
) -> tuple[MaterializedEnvironmentTask, ...]:
    """Compile selected tasks using the environment's declared episode protocol."""

    environment = loaded.definition
    if environment.episode_protocol == SINGLE_AGENT_TASK_INPUT_CONTRACT:
        return _materialize_single_agent_tasks(loaded, taskset=taskset)
    raise EnvironmentDefinitionError(f"Unsupported episode protocol {environment.episode_protocol!r}")


def _materialize_single_agent_tasks(
    loaded: LoadedEnvironment,
    *,
    taskset: str | None,
) -> tuple[MaterializedEnvironmentTask, ...]:
    environment = loaded.definition
    if environment.task is not None:
        if taskset not in (None, environment.name):
            raise EnvironmentDefinitionError(
                f"Singleton environment {environment.name!r} does not define taskset {taskset!r}"
            )
        return (_materialize_declared_task(loaded, environment.name, environment.task.id, environment.task),)

    selected = environment.tasksets
    if taskset is not None:
        try:
            selected = {taskset: environment.tasksets[taskset]}
        except KeyError as error:
            raise EnvironmentDefinitionError(f"Unknown taskset {taskset!r}") from error

    tasks: list[MaterializedEnvironmentTask] = []
    task_ids: set[TaskId] = set()
    for taskset_name, declaration in selected.items():
        if isinstance(declaration, str):
            taskset_tasks = _materialize_file_taskset(loaded, taskset_name, declaration)
        else:
            taskset_tasks = tuple(
                _materialize_declared_task(loaded, taskset_name, task.id or task_name, task)
                for task_name, task in declaration.tasks.items()
            )
        for environment_task in taskset_tasks:
            task_id = environment_task.materialized.task_id
            if task_id in task_ids:
                raise EnvironmentDefinitionError(f"Duplicate task identity: {task_id}")
            task_ids.add(task_id)
            tasks.append(environment_task)
    return tuple(tasks)


def materialize_single_task(loaded: LoadedEnvironment) -> MaterializedTask[BaseModel]:
    """Compile the task from a singleton environment."""

    if loaded.definition.task is None:
        raise EnvironmentDefinitionError("materialize_single_task requires a singleton task declaration")
    return materialize_tasks(loaded)[0].materialized


def materialize_tasks_jsonl(loaded: LoadedEnvironment, *, taskset: str | None = None) -> str:
    """Serialize selected materialized tasks as JSONL rows."""

    return "".join(
        json.dumps(environment_task.materialized.model_dump(mode="json", exclude_none=True), separators=(",", ":"))
        + "\n"
        for environment_task in materialize_tasks(loaded, taskset=taskset)
    )


def _materialize_declared_task(
    loaded: LoadedEnvironment,
    taskset_name: str,
    task_id: str | None,
    task: TaskDefinition,
) -> MaterializedEnvironmentTask:
    if task_id is None:
        raise EnvironmentDefinitionError("a singleton task requires task.id")
    instruction = loaded.resolve_file(task.instruction, description="task instruction").read_text(encoding="utf-8")
    verifier = VerifierDefinition(
        implementation=task.verifier.implementation,
        verifier_input=_resolve_task_data_references(task.verifier.verifier_input, task.task_data),
    )
    return MaterializedEnvironmentTask(
        materialized=_materialized_task(loaded, taskset_name, task_id, instruction, task.task_data),
        verifier=verifier,
    )


def _materialize_file_taskset(
    loaded: LoadedEnvironment,
    taskset_name: str,
    reference: str,
) -> tuple[MaterializedEnvironmentTask, ...]:
    environment = loaded.definition
    if environment.instruction is None or environment.task_model is None or environment.verifier is None:
        raise EnvironmentDefinitionError(
            "file-backed tasksets require top-level instruction, task_model, and verifier declarations"
        )
    taskset_path = loaded.resolve_file(reference, description=f"tasksets.{taskset_name}")
    instruction_template = loaded.resolve_file(
        environment.instruction,
        description="taskset instruction",
    ).read_text(encoding="utf-8")
    task_model = load_environment_object(loaded, environment.task_model, description="task model")
    task_data_adapter = TypeAdapter(task_model)
    tasks: list[MaterializedEnvironmentTask] = []
    for line_number, line in enumerate(taskset_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        location = f"{taskset_path}:{line_number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise EnvironmentDefinitionError(f"Invalid taskset JSON at {location}: {error.msg}") from error
        try:
            row = _TasksetRow.model_validate(raw)
            validated_task_data = task_data_adapter.validate_python(row.task_data)
        except ValidationError as error:
            issues = "; ".join(
                f"{'.'.join(str(part) for part in item['loc']) or 'row'}: {item['msg']}"
                for item in error.errors(include_url=False, include_context=False, include_input=False)
            )
            raise EnvironmentDefinitionError(f"Invalid taskset row at {location}: {issues}") from error
        task_data = task_data_adapter.dump_python(validated_task_data, mode="json")
        if not isinstance(task_data, dict):
            raise EnvironmentDefinitionError(f"Task model must produce an object at {location}")
        instruction = _render_instruction(instruction_template, task_data, location=location)
        verifier = VerifierDefinition(
            implementation=environment.verifier.implementation,
            verifier_input=_resolve_task_data_references(environment.verifier.verifier_input, task_data),
        )
        tasks.append(
            MaterializedEnvironmentTask(
                materialized=_materialized_task(loaded, taskset_name, row.task_id, instruction, task_data),
                verifier=verifier,
            )
        )
    if not tasks:
        raise EnvironmentDefinitionError(f"Taskset contains no tasks: {taskset_path}")
    return tuple(tasks)


def _materialized_task(
    loaded: LoadedEnvironment,
    taskset_name: str,
    task_id: str,
    instruction: str,
    task_data: dict[str, JsonValue],
) -> MaterializedTask[SingleAgentTaskInput]:
    return MaterializedTask[SingleAgentTaskInput](
        task_id=TaskId(taskset=taskset_name, task_id=task_id, revision=loaded.definition.version),
        task_input=SingleAgentTaskInput(
            responses_create_params={
                "input": [{"role": "user", "content": instruction}],
            },
            task_data=task_data,
        ),
    )


_INSTRUCTION_PLACEHOLDER = re.compile(r"{{\s*([^{}]+?)\s*}}")
_TASK_DATA_REFERENCE = re.compile(r"task_data((?:\.[A-Za-z_][A-Za-z0-9_]*)+)$")


def _render_instruction(template: str, task_data: Mapping[str, JsonValue], *, location: str) -> str:
    def replace(match: re.Match[str]) -> str:
        expression = match.group(1)
        reference = _TASK_DATA_REFERENCE.fullmatch(expression)
        if reference is None:
            raise EnvironmentDefinitionError(
                f"Unsupported instruction placeholder {expression!r} for taskset row at {location}; "
                "expected {{ task_data.field }}"
            )
        value = _resolve_task_data_path(task_data, reference.group(1), location=location)
        if isinstance(value, (dict, list)):
            raise EnvironmentDefinitionError(
                f"Instruction placeholder {expression!r} at {location} must resolve to a scalar"
            )
        return str(value)

    return _INSTRUCTION_PLACEHOLDER.sub(replace, template)


def _resolve_task_data_references(
    value: JsonValue,
    task_data: Mapping[str, JsonValue],
) -> JsonValue:
    if isinstance(value, str):
        reference = _TASK_DATA_REFERENCE.fullmatch(value)
        if reference is not None:
            return _resolve_task_data_path(task_data, reference.group(1), location="verifier_input")
        return value
    if isinstance(value, list):
        return [_resolve_task_data_references(item, task_data) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_task_data_references(item, task_data) for key, item in value.items()}
    return value


def _resolve_task_data_path(
    task_data: Mapping[str, JsonValue],
    dotted_path: str,
    *,
    location: str,
) -> JsonValue:
    value: JsonValue = dict(task_data)
    for part in dotted_path.removeprefix(".").split("."):
        if not isinstance(value, dict) or part not in value:
            raise EnvironmentDefinitionError(f"Unknown task_data reference {dotted_path!r} in {location}")
        value = value[part]
    return value


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
    "MaterializedEnvironmentTask",
    "load_environment",
    "load_environment_callable",
    "load_environment_object",
    "materialize_single_task",
    "materialize_tasks",
    "materialize_tasks_jsonl",
]
