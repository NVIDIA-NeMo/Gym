# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment servers' task input modules.

An environment server with a typed episode protocol ships ``task_input.py`` next to its ``app.py``.
The module exports ``TaskInput``, the type its ``request_model`` (``BaseEpisodeRequest[TaskInput]``) is parameterized by.
It may also export ``ACCEPTS_FLAT_ROWS = True`` when the server's ``/run`` takes a flat row too.
Such a server materializes the row with ``materialize_task(row, taskset=task_source)`` and validates the result as ``TaskInput``.
The server and data tooling share this one definition.

A relay ships the module with ``TaskInput = None``.
Its ``/run`` passes a flat row to the agent's ``/run``, which validates a ``BaseRunRequest``, and it takes no task rows.
An environment server without the module is not checked for task rows, as a server without ``task_data.py`` skips that check.
Its flat rows are checked as run requests, which every server that takes flat rows requires.

The module is loaded without the server's dependencies.
So it may import only the standard library, pydantic, resources servers' ``task_data`` modules,
and ``nemo_gym`` packages at or below ``nemo_gym.data`` in the import-linter layer order.
It describes the episode wire contract.
The resources server's task-owned fields stay in ``task_data.py`` (``nemo_gym/data/task_data.py``),
which feeds normalization, unknown-key reports and ``gym env schema``.
"""

from __future__ import annotations

import functools
import importlib.util
import sys
import typing
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from nemo_gym.core.episode_types import MaterializedTask
from nemo_gym.data.task_data import find_server_dir


TASK_INPUT_MODULE_NAME = "task_input"
TASK_INPUT_EXPORT_NAME = "TaskInput"
ACCEPTS_FLAT_ROWS_EXPORT_NAME = "ACCEPTS_FLAT_ROWS"
# Component packages a task input module may import from.
# Each import must resolve under the same root as the task input module itself.
_COMPONENT_PACKAGES = ("resources_servers", "environment_servers")


class TaskInputModuleError(Exception):
    """An environment server's ``task_input.py`` exists but does not load or does not satisfy the protocol."""


@dataclass(frozen=True)
class TaskInputContract:
    """What an environment server validates: task rows as ``task``, and whether ``/run`` takes flat rows.

    A relay has no ``task_input`` and no ``task``: it takes no task rows.
    """

    task_input: type | None
    task: TypeAdapter | None
    accepts_flat_rows: bool

    @property
    def is_relay(self) -> bool:
        """Whether the server passes flat rows to an agent's ``/run`` and takes no task rows."""
        return self.task_input is None

    @property
    def requires_responses_create_params(self) -> bool:
        """Whether ``TaskInput`` requires ``responses_create_params``, so every task carries its model request."""
        fields = getattr(self.task_input, "model_fields", {})
        return "responses_create_params" in fields and fields["responses_create_params"].is_required()


@dataclass(frozen=True)
class TaskInputLookup:
    """An environment server implementation's task input module, as data tooling finds it."""

    implementation: str
    server_dir: Path | None
    contract: TaskInputContract | None

    @property
    def missing(self) -> str | None:
        """Why there is no contract (no server folder, or no ``task_input.py``), or None when there is one."""
        if self.server_dir is None:
            return f"no environment_servers/{self.implementation} folder is found"
        if self.contract is None:
            return f"{self.implementation} ships no task_input.py"
        return None


def find_task_input_contract(implementation: str) -> TaskInputLookup:
    """Find and load ``environment_servers/<implementation>/task_input.py`` under the component search roots.

    Raises ``TaskInputModuleError`` when the module exists but does not load.
    Each module is loaded once per resolved folder, so a changed working directory or search root finds its own copy.
    """
    server_dir = find_server_dir(implementation, base_folder="environment_servers")
    if server_dir is None:
        return TaskInputLookup(implementation=implementation, server_dir=None, contract=None)
    return TaskInputLookup(
        implementation=implementation, server_dir=server_dir, contract=_load_once(server_dir.resolve())
    )


# Keyed by resolved folder, so it holds one entry per environment server folder a process loads, at most this many.
@functools.lru_cache(maxsize=256)
def _load_once(server_dir: Path) -> TaskInputContract | None:
    return load_task_input_module(server_dir)


def load_task_input_module(server_dir: Path) -> TaskInputContract | None:
    """Load ``<server_dir>/task_input.py``, or return None when the server ships none.

    The module is loaded by file path, as ``task_data.py`` is, so servers under extra component roots load too.
    Raises ``TaskInputModuleError`` when the module does not import or exports no ``TaskInput``.
    It also raises when ``TaskInput`` validates with a ``resources_servers`` or ``environment_servers`` type from another root than its own.
    That happens when a checkout would validate against an installed copy's model, or the reverse.
    """
    module_path = server_dir / f"{TASK_INPUT_MODULE_NAME}.py"
    if not module_path.is_file():
        return None

    module_name = f"nemo_gym_task_input.{server_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib internals
        raise TaskInputModuleError(f"Could not build an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise TaskInputModuleError(f"Failed to import {module_path}: {e}") from e

    if not hasattr(module, TASK_INPUT_EXPORT_NAME):
        raise TaskInputModuleError(
            f"{module_path} does not export `{TASK_INPUT_EXPORT_NAME}`. "
            "Export the task input type, or `TaskInput = None` for a relay."
        )
    task_input = getattr(module, TASK_INPUT_EXPORT_NAME)
    accepts_flat_rows = getattr(module, ACCEPTS_FLAT_ROWS_EXPORT_NAME, False)
    if not isinstance(accepts_flat_rows, bool):
        raise TaskInputModuleError(f"{module_path}: `{ACCEPTS_FLAT_ROWS_EXPORT_NAME}` must be a bool.")
    if task_input is None:
        if accepts_flat_rows:
            raise TaskInputModuleError(
                f"{module_path}: a relay (`TaskInput = None`) passes flat rows on; it does not set "
                f"`{ACCEPTS_FLAT_ROWS_EXPORT_NAME}`."
            )
        return TaskInputContract(task_input=None, task=None, accepts_flat_rows=False)
    _check_component_imports_share_root(module_path, task_input)
    try:
        adapter = TypeAdapter(MaterializedTask[task_input])
    except Exception as e:
        raise TaskInputModuleError(
            f"`{TASK_INPUT_EXPORT_NAME}` in {module_path} is not a valid task input: {e}"
        ) from e
    return TaskInputContract(task_input=task_input, task=adapter, accepts_flat_rows=accepts_flat_rows)


def _check_component_imports_share_root(module_path: Path, task_input: type) -> None:
    """Refuse a ``TaskInput`` built from a component type that lives under another root than the module.

    ``from resources_servers.x.task_data import TaskData`` resolves through ``sys.path``, not the task input module's own root.
    So a checkout in the working directory could validate against the installed wheel's copy of the model.
    The check follows the types ``TaskInput`` validates with, so no spelling of the import gets past it.
    """
    root = module_path.resolve().parents[2]
    for component_type in _types_validated_by(task_input):
        name = getattr(component_type, "__module__", None)
        if not isinstance(name, str) or not name.startswith(_COMPONENT_PACKAGES):
            continue
        source = getattr(sys.modules.get(name), "__file__", None)
        if source is not None and not Path(source).resolve().is_relative_to(root):
            raise TaskInputModuleError(
                f"{module_path} validates with {component_type.__qualname__} from {source}, which is not under {root}. "
                "Run from the root that holds this server, or pass that root with --search-dir or NEMO_GYM_EXTRA_ROOTS."
            )


def _types_validated_by(task_input: type) -> Iterator[type]:
    """``task_input``, its base classes, and every class its fields name, through generics and nested models."""
    seen: set[int] = set()
    pending: list[object] = [task_input]
    while pending:
        annotation = pending.pop()
        if id(annotation) in seen:
            continue
        seen.add(id(annotation))
        pending.extend(typing.get_args(annotation))
        origin = typing.get_origin(annotation)
        if origin is not None:
            pending.append(origin)
        if not isinstance(annotation, type):
            continue
        yield annotation
        pending.extend(annotation.__mro__[1:])
        if issubclass(annotation, BaseModel):
            pending.extend(field.annotation for field in annotation.model_fields.values())
