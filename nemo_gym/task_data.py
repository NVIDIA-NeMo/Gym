# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-server task-data schemas.

A resources server ships a ``task_data.py`` module next to its ``app.py`` exporting a single
symbol ``TaskData``: either a Pydantic ``BaseModel`` subclass or a type (e.g. an
``Annotated[Union[...], Field(discriminator=...)]`` alias) accepted by ``pydantic.TypeAdapter``.
A self-contained agent (one that declares datasets but no ``resources_server`` reference) ships
the same module under ``responses_api_agents/<implementation>/`` and owns its rows' schema; an
agent WITH a reference uses that resources server's schema instead.
An environment server ships the same module under ``environment_servers/<implementation>/`` to
declare the task fields it reads itself. Rows are validated against the environment server they
route to, composed with the bound resources server's schema: each schema validates the fields it
declares plus the fields neither declares. Single-agent environment servers declare
``responses_create_params``, one agent's request to one model, through ``SingleAgentTaskData``.
Rows routed by agent rather than by taskset are single-agent run requests and use it too.
It describes the task-owned fields of that server's dataset rows, written FLAT in the planned
end-state shape: the fields as they will appear inside the unified ``task_data`` row key after
the row-format migration. Framework-owned keys (see ``RESERVED_ROW_KEYS``) are never part of
``TaskData``, and neither is a ``verifier_metadata`` wrapper: rows that still carry one have its
contents spliced up by ``normalize_task_fields`` before validation, so one flat schema validates
both today's rows and post-migration ``task_data`` contents. Fields that today's wire reads
EXCLUSIVELY from inside ``verifier_metadata`` should be annotated
``Field(..., json_schema_extra={"legacy_location": "verifier_metadata"})`` — that reverse map is
what the row-format migration and the dispatch compatibility shim consume, and validation flags
rows that carry such a field only top-level (the server would not see it). Servers whose wire
accepts both placements (e.g. via a before-validator that nests top-level fields itself) must
not carry the marker.

This module is a dependency-light leaf: it may import only the standard library and Pydantic, and
per-server ``task_data.py`` modules may import only the standard library, Pydantic, this module,
and other servers' ``task_data`` modules. That keeps schemas loadable by data tooling (collate,
``gym env schema``, dataset import) without installing any server's requirements.

Conventions for ``TaskData`` models:
- ``model_config = ConfigDict(extra="allow")`` by default. ``extra="forbid"`` is opt-in for
  servers that are already fail-closed. Pydantic's default ``extra="ignore"`` is banned: silently
  dropping row fields is the existing bug class this system exists to catch.
- Required-ness mirrors the server's wire contract (its verify/run request models), not what
  ``verify()`` happens to read. A field the wire requires stays required even if unread.
- Fields may carry ``json_schema_extra={"consumed_by": [...]}`` with values from
  ``{"verify", "metrics", "prompt", "provenance"}``. These tags are purely informational: they
  document what reads a field for humans inspecting ``gym env schema`` output, and no tooling
  consumes them. Fields that are JSON-encoded strings on the wire stay typed ``str``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union, get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from nemo_gym.episode_types import is_materialized_task_row


TASK_DATA_MODULE_NAME = "task_data"
TASK_DATA_EXPORT_NAME = "TaskData"

# Framework-owned top-level row keys. Everything else in a row is task-owned and is the subject of
# the environment server's and resources server's TaskData schemas. ``responses_create_params`` is
# not one of them: it is a task field of single-agent environment servers (``SingleAgentTaskData``).
# (The row-format migration will extend this set with reserved keys such as
# ``task_id``/``subset_for_metrics``/``provenance``; until then some servers legitimately use those
# names as task fields, so they stay task-owned here.)
RESERVED_ROW_KEYS = frozenset(
    {
        "agent_ref",
        "task_source",
        "_ng_task_index",
        "_ng_rollout_index",
    }
)


class SingleAgentResponsesCreateParams(BaseModel):
    """The part of a Responses API request that a dataset row supplies.

    ``input`` may be missing from a source row whose dataset declares a ``prompt_config``, which
    fills it during collation. The environment server validates the full request at runtime; this
    keeps schemas free of server dependencies.
    """

    model_config = ConfigDict(extra="allow")

    input: Optional[Union[str, List[Any]]] = None


class SingleAgentTaskData(BaseModel):
    """Task fields that every single-agent environment server reads: one agent's request to one model.

    The bound resources server's ``TaskData`` declares the remaining fields.
    """

    model_config = ConfigDict(extra="allow")

    responses_create_params: SingleAgentResponsesCreateParams = Field(
        json_schema_extra={"consumed_by": ["prompt"]},
    )


class TaskDataSchemaError(Exception):
    """A server's ``task_data.py`` exists but does not satisfy the protocol."""


def find_server_dir(server_name: str, base_folder: str = "resources_servers") -> Optional[Path]:
    """Locate ``<base_folder>/<server_name>`` via the shared component search roots.

    Resolves against ``_resolve_under_cwd_or_install`` (extra plugin roots first, then cwd, then
    the Gym install root), without the CLI's venv-marker requirement (a schema can exist for a
    server whose venv was never set up). Self-contained agents (which verify in-process) keep
    their schemas under ``responses_api_agents/<name>/``.
    """
    from nemo_gym import _resolve_under_cwd_or_install

    candidate = _resolve_under_cwd_or_install(Path(base_folder) / server_name, validator=Path.is_dir)
    return candidate if candidate.is_dir() else None


def load_task_data_schema(server_dir: Path) -> Optional[TypeAdapter]:
    """Load ``<server_dir>/task_data.py`` and return a ``TypeAdapter`` for its ``TaskData``.

    Returns ``None`` when the module does not exist (the server has not adopted schemas yet).
    Raises ``TaskDataSchemaError`` when the module exists but cannot be imported or does not
    export a usable ``TaskData``.
    """
    module_path = server_dir / f"{TASK_DATA_MODULE_NAME}.py"
    if not module_path.is_file():
        return None

    module_name = f"nemo_gym_task_data.{server_dir.name}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib internals
        raise TaskDataSchemaError(f"Could not build an import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass/typing machinery that looks up sys.modules works.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        sys.modules.pop(module_name, None)
        raise TaskDataSchemaError(f"Failed to import {module_path}: {e}") from e

    task_data = getattr(module, TASK_DATA_EXPORT_NAME, None)
    if task_data is None:
        raise TaskDataSchemaError(
            f"{module_path} does not export `{TASK_DATA_EXPORT_NAME}`. Export a Pydantic model "
            "(or a TypeAdapter-compatible union alias) under that name."
        )
    try:
        return TypeAdapter(task_data)
    except Exception as e:
        raise TaskDataSchemaError(
            f"`{TASK_DATA_EXPORT_NAME}` in {module_path} is not TypeAdapter-compatible: {e}"
        ) from e


LEGACY_METADATA_KEY = "verifier_metadata"
TASK_DATA_ROW_KEY = "task_data"


def normalize_task_fields(row: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    """The task-owned subset of a dataset row, normalized to the flat end-state shape.

    Drops framework keys, then splices the contents of a legacy ``verifier_metadata`` dict and of
    a migrated ``task_data`` dict up to the top level (schemas are written flat, so fields
    validate the same whether a row is flat, legacy-nested, or migrated).
    A key present in two places with the same value is a harmless duplicate; with different
    values it is ambiguous data and gets reported. Returns ``(fields, conflicts)``.
    """
    fields = {k: v for k, v in row.items() if k not in RESERVED_ROW_KEYS}
    conflicts: List[str] = []
    for container_key in (LEGACY_METADATA_KEY, TASK_DATA_ROW_KEY):
        container = fields.pop(container_key, None)
        if isinstance(container, dict):
            for key, value in container.items():
                if key in fields and fields[key] != value:
                    conflicts.append(key)
                    continue
                fields[key] = value
        elif container is not None:
            # A non-dict container is malformed; surface it to the schema as-is.
            fields[container_key] = container
    return fields, conflicts


@dataclass
class TaskDataValidationReport:
    """Accumulated validation outcome for one dataset file against one server's schema."""

    server_name: str
    dataset_fpath: str
    rows: int = 0
    error_rows: int = 0
    errors: List[str] = field(default_factory=list)
    unknown_keys: Dict[str, int] = field(default_factory=dict)
    conflicting_keys: Dict[str, int] = field(default_factory=dict)
    misplaced_keys: Dict[str, int] = field(default_factory=dict)

    MAX_RECORDED_ERRORS = 5

    @property
    def clean(self) -> bool:
        return self.error_rows == 0 and not self.conflicting_keys and not self.misplaced_keys and not self.unknown_keys

    def summary(self) -> str:
        parts = [
            f"{self.dataset_fpath}: {self.error_rows}/{self.rows} rows failed task_data validation "
            f"against the `{self.server_name}` schema."
        ]
        parts.extend(f"  row {msg}" for msg in self.errors)
        if self.error_rows > len(self.errors):
            parts.append(f"  ... and {self.error_rows - len(self.errors)} more rows")
        if self.conflicting_keys:
            keys = ", ".join(f"{k} ({n} rows)" for k, n in sorted(self.conflicting_keys.items()))
            parts.append(f"  keys with DIFFERENT values top-level vs verifier_metadata (ambiguous): {keys}")
        if self.misplaced_keys:
            keys = ", ".join(f"{k} ({n} rows)" for k, n in sorted(self.misplaced_keys.items()))
            parts.append(
                f"  keys this server's wire reads from verifier_metadata but found top-level "
                f"(the server will not see them): {keys}"
            )
        if self.unknown_keys:
            keys = ", ".join(f"{k} ({n} rows)" for k, n in sorted(self.unknown_keys.items()))
            parts.append(f"  keys not declared by the schema (typo, or missing schema field?): {keys}")
        return "\n".join(parts)


def _schema_models(adapter: TypeAdapter) -> List[type[BaseModel]]:
    """The models a schema is built from: the model itself, or each member of a union."""

    def models_of(tp, out):
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            out.append(tp)
            return out
        for arg in get_args(tp):
            models_of(arg, out)
        return out

    return models_of(getattr(adapter, "_type", None), [])


def declared_fields(adapter: TypeAdapter) -> frozenset:
    """The field names, and their aliases, that any model of a schema declares."""
    names = set()
    for model in _schema_models(adapter):
        for field_name, info in model.model_fields.items():
            names.add(field_name)
            if isinstance(info.alias, str):
                names.add(info.alias)
    return frozenset(names)


def legacy_metadata_fields(adapter: TypeAdapter) -> frozenset:
    """Schema fields annotated ``legacy_location: verifier_metadata`` (today's wire reads them there)."""
    names = set()
    for model in _schema_models(adapter):
        for field_name, info in model.model_fields.items():
            extra = info.json_schema_extra
            if isinstance(extra, dict) and extra.get("legacy_location") == LEGACY_METADATA_KEY:
                names.add(field_name)
    return frozenset(names)


SINGLE_AGENT_TASK_DATA = TypeAdapter(SingleAgentTaskData)


class TaskDataValidator:
    """Validates dataset rows against an environment server's and a resources server's ``TaskData``.

    ``environment_adapter`` is the schema of the environment server the rows route to, such as
    ``SINGLE_AGENT_TASK_DATA`` for rows routed by agent. Each schema validates the fields it
    declares plus the fields neither declares, so a resources schema with ``extra="forbid"`` does not
    see the environment server's fields. Undeclared fields are reported against the resources server's
    schema. Either adapter may be ``None`` when its server ships no schema. The environment schema does
    not validate materialized rows (see ``validate_row``).
    """

    def __init__(
        self,
        server_name: str,
        adapter: Optional[TypeAdapter],
        dataset_fpath: str,
        *,
        environment_adapter: Optional[TypeAdapter] = None,
    ):
        self._adapter = adapter
        self._environment_adapter = environment_adapter
        self._fields = declared_fields(adapter) if adapter is not None else frozenset()
        self._environment_fields = (
            declared_fields(environment_adapter) if environment_adapter is not None else frozenset()
        )
        self._legacy_fields = legacy_metadata_fields(adapter) if adapter is not None else frozenset()
        self.report = TaskDataValidationReport(server_name=server_name, dataset_fpath=dataset_fpath)

    def _validate_fields(self, subject: Dict[str, Any], *, environment_applies: bool) -> tuple[List[str], set]:
        """Validate one row's task fields against each schema; return its errors and undeclared fields."""
        errors: List[str] = []
        undeclared: set = set()
        # The environment server's fields stay out of the resources schema's subject either way, so
        # they are never reported as fields the resources server does not declare.
        environment_adapter = self._environment_adapter if environment_applies else None
        for adapter, own, other, reports_unknown in (
            (environment_adapter, self._environment_fields, self._fields, False),
            (self._adapter, self._fields, self._environment_fields, True),
        ):
            if adapter is None:
                continue
            fields = {key: value for key, value in subject.items() if key in own or key not in other}
            try:
                validated = adapter.validate_python(fields)
            except Exception as e:
                errors.append(str(e).strip().replace("\n", "; "))
                continue
            # Pydantic returns the concrete model (the selected union member for union schemas), and
            # with extra="allow" it stores undeclared inputs on __pydantic_extra__ — so unknown-field
            # reporting works uniformly for plain models and discriminated unions.
            if reports_unknown and isinstance(validated, BaseModel):
                undeclared = set(getattr(validated, "__pydantic_extra__", None) or {})
        return errors, undeclared

    def validate_row(self, row_index: int, row: Dict[str, Any]) -> None:
        self.report.rows += 1
        # A materialized row already names its taskset, and the environment server that taskset routes
        # to validates its input at dispatch. The environment schema does not validate it here, so such a
        # row in a dataset routed by agent is not required to be a single-agent run request.
        environment_applies = not is_materialized_task_row(row)
        # New materialized tasks keep the flat source fields; historical single-agent
        # tasks use task_input.task_data. Normalize both through the existing schema path.
        task_input = row.get("task_input")
        materialized = is_materialized_task_row(row)
        if materialized:
            if not isinstance(task_input, Mapping):
                self.report.error_rows += 1
                if len(self.report.errors) < TaskDataValidationReport.MAX_RECORDED_ERRORS:
                    self.report.errors.append(
                        f"{row_index}: task_input must be an object, got {type(task_input).__name__}"
                    )
                return
            task_data = task_input.get("task_data")
            if "task_data" in task_input and not isinstance(task_data, Mapping):
                self.report.error_rows += 1
                if len(self.report.errors) < TaskDataValidationReport.MAX_RECORDED_ERRORS:
                    self.report.errors.append(
                        f"{row_index}: task_input.task_data must be an object, got {type(task_data).__name__}"
                    )
                return
            materialized = "task_data" in task_input
            row = dict(task_input.get("task_data", {}))
            for key, value in task_input.items():
                if key == "task_data":
                    continue
                if key in row and row[key] != value:
                    self.report.conflicting_keys[key] = self.report.conflicting_keys.get(key, 0) + 1
                else:
                    row[key] = value
        # Misplacement: the schema says today's wire reads this field from inside
        # verifier_metadata, but the row carries it only top-level. Validation would accept it
        # (schemas are flat) while the server at runtime would never see it, so it is flagged.
        # Rows already in the migrated format (a task_data key) are exempt: top-level inside
        # task_data is the correct final position.
        if self._legacy_fields and not materialized and TASK_DATA_ROW_KEY not in row:
            nested = row.get(LEGACY_METADATA_KEY)
            nested_keys = set(nested) if isinstance(nested, dict) else set()
            for key in (row.keys() & self._legacy_fields) - nested_keys:
                self.report.misplaced_keys[key] = self.report.misplaced_keys.get(key, 0) + 1
        subject, conflicts = normalize_task_fields(row)
        for key in conflicts:
            self.report.conflicting_keys[key] = self.report.conflicting_keys.get(key, 0) + 1
        errors, undeclared = self._validate_fields(subject, environment_applies=environment_applies)
        if errors:
            self.report.error_rows += 1
            if len(self.report.errors) < TaskDataValidationReport.MAX_RECORDED_ERRORS:
                self.report.errors.append(f"{row_index}: {'; '.join(error[:400] for error in errors)}")
            return
        for key in undeclared:
            self.report.unknown_keys[key] = self.report.unknown_keys.get(key, 0) + 1


def validate_jsonl_rows(
    server_name: str,
    adapter: Optional[TypeAdapter],
    dataset_fpath: str,
    lines: Iterable[str],
    *,
    environment_adapter: Optional[TypeAdapter] = None,
) -> TaskDataValidationReport:
    """Validate an iterable of JSONL lines against one schema; entry point for whole-file validation."""
    validator = TaskDataValidator(
        server_name=server_name,
        adapter=adapter,
        dataset_fpath=dataset_fpath,
        environment_adapter=environment_adapter,
    )
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        validator.validate_row(i, json.loads(line))
    return validator.report
