# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment servers' task input modules.

An environment server with a typed episode protocol ships ``task_input.py`` next to its ``app.py``.
The module exports ``TaskInput``, the type its ``request_model`` (``BaseEpisodeRequest[TaskInput]``) is parameterized by.
It may also export ``ACCEPTS_FLAT_ROWS = True`` when the server's ``/run`` takes a flat row too.
Such a server materializes the row with ``materialize_task(row, taskset=task_source)`` and validates the result as ``TaskInput``.
The server, collation and ``gym env validate`` share this one definition.

An environment server without the module is a relay.
Its ``/run`` passes a flat row to the agent's ``/run``, which validates a ``BaseRunRequest``.

The module is loaded without the server's dependencies.
So it may import only the standard library, pydantic, resources servers' ``task_data`` modules,
and ``nemo_gym`` packages at or below ``nemo_gym.data`` in the import-linter layer order.
It describes the episode wire contract.
The resources server's task-owned fields stay in ``task_data.py`` (``nemo_gym/data/task_data.py``),
which feeds normalization, unknown-key reports and ``gym env schema``.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Optional

from pydantic import TypeAdapter

from nemo_gym.core.episode_types import MaterializedTask


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
    """What an environment server validates: task rows as ``task``, and whether ``/run`` takes flat rows."""

    task_input: type
    task: TypeAdapter
    accepts_flat_rows: bool

    @property
    def requires_responses_create_params(self) -> bool:
        """Whether ``TaskInput`` requires ``responses_create_params``, so every task carries its model request."""
        fields = getattr(self.task_input, "model_fields", {})
        return "responses_create_params" in fields and fields["responses_create_params"].is_required()


def load_task_input_module(server_dir: Path) -> Optional[TaskInputContract]:
    """Load ``<server_dir>/task_input.py``, or return None when the server ships none (a relay).

    The module is loaded by file path, as ``task_data.py`` is, so servers under extra component roots load too.
    Raises ``TaskInputModuleError`` when the module does not import or exports no ``TaskInput``.
    It also raises when the module imported a ``resources_servers`` or ``environment_servers`` module from another root than its own.
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

    task_input = getattr(module, TASK_INPUT_EXPORT_NAME, None)
    if task_input is None:
        raise TaskInputModuleError(f"{module_path} does not export `{TASK_INPUT_EXPORT_NAME}`.")
    _check_component_imports_share_root(module_path, module)
    accepts_flat_rows = getattr(module, ACCEPTS_FLAT_ROWS_EXPORT_NAME, False)
    if not isinstance(accepts_flat_rows, bool):
        raise TaskInputModuleError(f"{module_path}: `{ACCEPTS_FLAT_ROWS_EXPORT_NAME}` must be a bool.")
    try:
        adapter = TypeAdapter(MaterializedTask[task_input])
    except Exception as e:
        raise TaskInputModuleError(
            f"`{TASK_INPUT_EXPORT_NAME}` in {module_path} is not a valid task input: {e}"
        ) from e
    return TaskInputContract(task_input=task_input, task=adapter, accepts_flat_rows=accepts_flat_rows)


def _check_component_imports_share_root(module_path: Path, module: ModuleType) -> None:
    """Refuse a task input module that imported a component module from another root than its own.

    ``from resources_servers.x.task_data import TaskData`` resolves through ``sys.path``, not the task input module's own root.
    So a checkout in the working directory could validate against the installed wheel's copy of the model.
    """
    root = module_path.resolve().parents[2]
    for value in vars(module).values():
        name = value.__name__ if isinstance(value, ModuleType) else getattr(value, "__module__", None)
        if not isinstance(name, str) or not name.startswith(_COMPONENT_PACKAGES):
            continue
        source = getattr(sys.modules.get(name), "__file__", None)
        if source is not None and not Path(source).resolve().is_relative_to(root):
            raise TaskInputModuleError(
                f"{module_path} imports {name} from {source}, which is not under {root}. "
                "Run from the root that holds this server, or put that root first on PYTHONPATH."
            )
