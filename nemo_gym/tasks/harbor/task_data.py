# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dataset's ``task_data.py``: the ``TaskData`` model every row's task fields are validated against.

A Gym-authored dataset may ship ``task_data.py`` beside ``dataset.toml``, a module exporting one
Pydantic model named ``TaskData`` that declares the task-owned fields of its rows. A task's fields
come from ``task_data.json`` in its folder (a parameterized task's rendered row; a hand-written task
may carry one too) and travel in the row's ``task_input.task_data`` beside the reserved ``_ng_*``
provenance keys. The same check runs at the three points a row crosses a boundary:

- at materialize, ``prepare_target`` loads the module once per dataset and refuses the whole
  dataset, naming the offending task, when a task's fields do not satisfy the model (the folders'
  ``task_data.json`` files are checked before any image is recorded or built, the loaded tasks again
  before a row is written);
- at seed, the ``harbor`` environment server validates the row it is handed and refuses the
  episode (422, not retried) when it does not satisfy the model or differs from the task folder's;
- at collate, ``gym dataset collate`` validates every row of the taskset the same way.

Unknown keys fail everywhere: ``TaskData`` is the complete description of the fields, whatever its
``extra`` setting, so a typo in a row cannot pass as an optional field. The reserved keys are never
part of ``TaskData`` and never admitted as task fields. A dataset without ``task_data.py`` is not
checked beyond the reserved keys (every hub dataset).

The module is imported, not parsed: ``TaskData`` is a plain Pydantic class and may use whatever the
standard library and Pydantic offer. It is loaded under a module name derived from its path, so two
datasets with the same folder name keep separate modules.
"""

import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pydantic import AliasChoices, AliasPath, BaseModel, TypeAdapter, ValidationError

from nemo_gym.episode_types import is_materialized_task_row
from nemo_gym.task_data import TaskDataValidationReport, TaskDataValidator
from nemo_gym.tasks.harbor.errors import HarborTaskError


# Beside dataset.toml at the dataset root: the module exporting ``TaskData``.
TASK_DATA_MODULE = "task_data.py"
# Inside a task folder: the task's own field values, a JSON object.
TASK_DATA_FILE = "task_data.json"
TASK_DATA_EXPORT = "TaskData"
# Keys the framework writes into a row's task_data (``_ng_digest``, ``_ng_dataset_digest``); never task fields.
RESERVED_PREFIX = "_ng_"


class TaskDataError(HarborTaskError):
    """``task_data.py`` does not load, or a task's fields do not satisfy its ``TaskData``."""


def is_reserved_key(key: str) -> bool:
    return key.startswith(RESERVED_PREFIX)


def task_fields(task_data: Mapping[str, Any]) -> dict[str, Any]:
    """The task-owned part of a row's ``task_data``: everything but the reserved ``_ng_*`` keys."""
    return {key: value for key, value in task_data.items() if not is_reserved_key(key)}


def _format_validation_error(error: ValidationError) -> str:
    parts = []
    for item in error.errors():
        location = ".".join(str(piece) for piece in item.get("loc", ())) or "(root)"
        parts.append(f"{location}: {item.get('msg', 'invalid')}")
    return "; ".join(parts)


def _alias_keys(alias: Any) -> set[str]:
    """The top-level row keys a field's ``validation_alias`` reads: a string, an ``AliasPath``'s first element, or
    every choice of an ``AliasChoices``."""
    if isinstance(alias, str):
        return {alias}
    if isinstance(alias, AliasPath):
        first = alias.path[0] if alias.path else None
        return {first} if isinstance(first, str) else set()
    if isinstance(alias, AliasChoices):
        return set().union(*(_alias_keys(choice) for choice in alias.choices))
    return set()


@dataclass(frozen=True)
class TaskDataSchema:
    """A loaded ``task_data.py``: where it came from and the model it exports."""

    path: Path
    model: type[BaseModel]

    @property
    def declared_keys(self) -> frozenset[str]:
        """The keys a row may carry: every field's name and every name its aliases admit.

        A ``validation_alias`` may be a string, an :class:`AliasPath` (its first element is the row key) or
        :class:`AliasChoices` over either, so a row pydantic itself accepts is never refused as undeclared.
        """
        keys: set[str] = set()
        for name, info in self.model.model_fields.items():
            keys.add(name)
            if info.alias:
                keys.add(info.alias)
            keys.update(_alias_keys(info.validation_alias))
        return frozenset(keys)

    def validate(self, task_data: Mapping[str, Any], *, task_id: str) -> BaseModel:
        """The row's task fields as an instance of ``TaskData``; raises :class:`TaskDataError` naming the task.

        Whatever the model raises is that error: a pydantic :class:`ValidationError` with its fields spelled out, or
        anything else an author's validator lets escape (a ``KeyError`` from a lookup, a ``TypeError`` on a value of
        the wrong type), named with its type. Every caller (prepare, the seed, collate) handles one exception class,
        so a validator defect refuses the dataset, fails the episode as terminal or is reported against the row
        instead of crashing the command or being retried as a provider failure.
        """
        fields = task_fields(task_data)
        unknown = sorted(set(fields) - self.declared_keys)
        if unknown:
            raise TaskDataError(
                f"task {task_id!r}: task_data has keys {unknown} that {TASK_DATA_EXPORT} in {self.path} "
                "does not declare"
            )
        try:
            return self.model.model_validate(fields)
        except ValidationError as exc:
            raise TaskDataError(
                f"task {task_id!r}: task_data does not satisfy {TASK_DATA_EXPORT} in {self.path}: "
                f"{_format_validation_error(exc)}"
            ) from None
        except Exception as exc:
            raise TaskDataError(
                f"task {task_id!r}: {TASK_DATA_EXPORT} in {self.path} raised {type(exc).__name__}: {exc} while "
                "validating task_data"
            ) from exc


def load_task_data_schema(dataset_root: Path) -> TaskDataSchema | None:
    """Import ``<dataset_root>/task_data.py``; ``None`` when the dataset ships none.

    Raises :class:`TaskDataError` when the module does not import, exports no ``TaskData``, or
    exports one that is not a Pydantic model class.
    """
    path = Path(dataset_root) / TASK_DATA_MODULE
    if not path.is_file():
        return None
    resolved = path.resolve()
    module_name = f"nemo_gym_harbor_task_data_{hashlib.sha256(str(resolved).encode()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib internals
        raise TaskDataError(f"{path}: could not build an import spec")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so dataclass and typing machinery that looks the module up finds it.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:
        sys.modules.pop(module_name, None)
        if not isinstance(exc, Exception):
            raise
        raise TaskDataError(f"{path}: failed to import: {type(exc).__name__}: {exc}") from exc
    model = getattr(module, TASK_DATA_EXPORT, None)
    if model is None:
        raise TaskDataError(f"{path} does not export `{TASK_DATA_EXPORT}`; define `class TaskData(BaseModel)` in it")
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        raise TaskDataError(f"{path}: `{TASK_DATA_EXPORT}` must be a Pydantic BaseModel subclass, got {model!r}")
    return TaskDataSchema(path=path, model=model)


def read_task_fields(task_folder: Path) -> dict[str, Any]:
    """The task's own field values from ``<task_folder>/task_data.json``; ``{}`` when the file is absent.

    The file must hold a JSON object whose keys are not reserved. Raises :class:`TaskDataError` otherwise.
    """
    path = Path(task_folder) / TASK_DATA_FILE
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise TaskDataError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TaskDataError(f"{path}: must hold a JSON object, got {type(data).__name__}")
    reserved = sorted(key for key in data if is_reserved_key(key))
    if reserved:
        raise TaskDataError(f"{path}: keys {reserved} are reserved (`{RESERVED_PREFIX}*` belongs to the framework)")
    return data


def check_task_folders(schema: TaskDataSchema | None, task_folders: Any) -> None:
    """Validate each folder's ``task_data.json`` against ``schema`` before the tasks load; the first violation refuses
    the dataset.

    ``task_folders`` are paths. The fields need no loaded task, so prepare runs this right after the module imports,
    before any base image is resolved or any Dockerfile built and pushed for a dataset that is refused as a whole
    anyway; :func:`check_tasks` over the loaded tasks stays the final guard. A file that does not read (not a JSON
    object, a reserved key) is left to the loader, which skips that task with the file's error. Without a schema
    nothing is checked.
    """
    if schema is None:
        return
    for folder in task_folders:
        try:
            fields = read_task_fields(folder)
        except TaskDataError:
            continue
        schema.validate(fields, task_id=Path(folder).name)


def check_tasks(schema: TaskDataSchema | None, tasks: Any) -> None:
    """Validate every loaded task's fields against ``schema``; the first violation refuses the dataset.

    ``tasks`` are :class:`~nemo_gym.tasks.harbor.task.HarborTask` instances, checked in order. Without a
    schema nothing is checked: a dataset that ships no ``task_data.py`` behaves as before.
    """
    if schema is None:
        return
    for task in tasks:
        schema.validate(task.task_data, task_id=task.task_id)


def check_row_matches_folder(row_task_data: Mapping[str, Any], task: Any) -> None:
    """Refuse a row whose task fields are not the task folder's: the row was edited after materialization.

    The digest the row carries pins the folder; this pins the row to it. Raises :class:`TaskDataError`.
    """
    fields = task_fields(row_task_data)
    if fields != dict(task.task_data):
        extra = sorted(set(fields) - set(task.task_data))
        missing = sorted(set(task.task_data) - set(fields))
        changed = sorted(key for key in set(fields) & set(task.task_data) if fields[key] != task.task_data[key])
        detail = ", ".join(
            part
            for part in (
                f"keys not in the folder: {extra}" if extra else "",
                f"keys missing from the row: {missing}" if missing else "",
                f"values that differ: {changed}" if changed else "",
            )
            if part
        )
        raise TaskDataError(
            f"task {task.task_id!r}: the row's task_data differs from the task folder's {TASK_DATA_FILE} ({detail}); "
            "re-materialize the taskset"
        )


class DatasetRowValidator(TaskDataValidator):
    """The collator's validator for a Harbor taskset whose dataset ships ``task_data.py``.

    Every row passes the environment server's schema first (the reserved keys, as for any taskset), then
    its ``task_input.task_data`` passes the dataset's ``TaskData``, into the same report. The dataset's
    declared keys are not "unknown" to the run: the server's schema admits extras so the dataset's fields
    can travel, and the dataset's model is what decides them.
    """

    def __init__(self, server_name: str, adapter: TypeAdapter, dataset_fpath: str, schema: TaskDataSchema):
        super().__init__(server_name=server_name, adapter=adapter, dataset_fpath=dataset_fpath)
        self._schema = schema

    def validate_row(self, row_index: int, row: dict[str, Any]) -> None:
        super().validate_row(row_index, row)
        for key in self._schema.declared_keys:
            self.report.unknown_keys.pop(key, None)
        task_input = row.get("task_input") if is_materialized_task_row(row) else None
        task_data = task_input.get("task_data") if isinstance(task_input, Mapping) else None
        if not isinstance(task_data, Mapping):
            return  # not a materialized row, or malformed: reported by the server's schema above
        task_id = row["task_id"].get("task_id")
        try:
            self._schema.validate(task_data, task_id=str(task_id if task_id is not None else row_index))
        except TaskDataError as exc:
            self.report.error_rows += 1
            if len(self.report.errors) < TaskDataValidationReport.MAX_RECORDED_ERRORS:
                self.report.errors.append(f"{row_index}: {exc}")
