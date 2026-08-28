# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-server task-data schemas.

A resources server ships a ``task_data.py`` module next to its ``app.py`` exporting a single
symbol ``TaskData``: either a Pydantic ``BaseModel`` subclass or a type (e.g. an
``Annotated[Union[...], Field(discriminator=...)]`` alias) accepted by ``pydantic.TypeAdapter``.
A self-contained agent (one that declares datasets but no ``resources_server`` reference) ships
the same module under ``responses_api_agents/<implementation>/`` and owns its rows' schema; an
agent WITH a reference uses that resources server's schema instead.
It describes the task-owned fields of that server's dataset rows, written FLAT in the planned
end-state shape: the fields as they will appear inside the unified ``task_data`` row key after
the row-format migration. Framework-owned keys (see ``RESERVED_ROW_KEYS``) are never part of
``TaskData``, and neither is a ``verifier_metadata`` wrapper: rows that still carry one have its
contents spliced up by ``normalize_task_fields`` before validation, so one flat schema validates
both today's rows and post-migration ``task_data`` contents. Fields that live inside
``verifier_metadata`` on today's wire should be annotated
``Field(..., json_schema_extra={"legacy_location": "verifier_metadata"})`` — that reverse map is
what the row-format migration and the dispatch compatibility shim consume.

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
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, TypeAdapter


TASK_DATA_MODULE_NAME = "task_data"
TASK_DATA_EXPORT_NAME = "TaskData"

# Framework-owned top-level row keys. Everything else in a row is task-owned and is the subject of
# the server's TaskData schema. (The row-format migration will extend this set with reserved keys such as
# ``task_id``/``subset_for_metrics``/``provenance``; until then some servers legitimately use those
# names as task fields, so they stay task-owned here.)
RESERVED_ROW_KEYS = frozenset(
    {
        "responses_create_params",
        "agent_ref",
        "task_source",
        "_ng_task_index",
        "_ng_rollout_index",
    }
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


def normalize_task_fields(row: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    """The task-owned subset of a dataset row, normalized to the flat end-state shape.

    Drops framework keys, then splices the contents of a legacy ``verifier_metadata`` dict up to
    the top level (the row-format migration renames that bucket into ``task_data``, so schemas are
    written flat).
    A key present in both places with the same value is a harmless duplicate; with different
    values it is ambiguous data and gets reported. Returns ``(fields, conflicts)``.
    """
    fields = {k: v for k, v in row.items() if k not in RESERVED_ROW_KEYS}
    conflicts: List[str] = []
    legacy = fields.pop(LEGACY_METADATA_KEY, None)
    if isinstance(legacy, dict):
        for key, value in legacy.items():
            if key in fields and fields[key] != value:
                conflicts.append(key)
                continue
            fields[key] = value
    elif legacy is not None:
        # A non-dict verifier_metadata is malformed; surface it to the schema as-is.
        fields[LEGACY_METADATA_KEY] = legacy
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

    MAX_RECORDED_ERRORS = 5

    @property
    def clean(self) -> bool:
        return self.error_rows == 0 and not self.conflicting_keys

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
        if self.unknown_keys:
            keys = ", ".join(f"{k} ({n} rows)" for k, n in sorted(self.unknown_keys.items()))
            parts.append(f"  keys not declared by the schema (passed through unvalidated): {keys}")
        return "\n".join(parts)


class TaskDataValidator:
    """Validates dataset rows against a server's ``TaskData`` schema, accumulating a report."""

    def __init__(self, server_name: str, adapter: TypeAdapter, dataset_fpath: str):
        self._adapter = adapter
        self.report = TaskDataValidationReport(server_name=server_name, dataset_fpath=dataset_fpath)

    def validate_row(self, row_index: int, row: Dict[str, Any]) -> None:
        self.report.rows += 1
        subject, conflicts = normalize_task_fields(row)
        for key in conflicts:
            self.report.conflicting_keys[key] = self.report.conflicting_keys.get(key, 0) + 1
        try:
            validated = self._adapter.validate_python(subject)
        except Exception as e:
            self.report.error_rows += 1
            if len(self.report.errors) < TaskDataValidationReport.MAX_RECORDED_ERRORS:
                first_line = str(e).strip().replace("\n", "; ")
                self.report.errors.append(f"{row_index}: {first_line[:400]}")
            return
        # Pydantic returns the concrete model (the selected union member for union schemas), and
        # with extra="allow" it stores undeclared inputs on __pydantic_extra__ — so unknown-field
        # reporting works uniformly for plain models and discriminated unions.
        if isinstance(validated, BaseModel):
            for key in getattr(validated, "__pydantic_extra__", None) or {}:
                self.report.unknown_keys[key] = self.report.unknown_keys.get(key, 0) + 1


def validate_jsonl_rows(
    server_name: str, adapter: TypeAdapter, dataset_fpath: str, lines: Iterable[str]
) -> TaskDataValidationReport:
    """Validate an iterable of JSONL lines against one schema; entry point for whole-file validation."""
    validator = TaskDataValidator(server_name=server_name, adapter=adapter, dataset_fpath=dataset_fpath)
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        validator.validate_row(i, json.loads(line))
    return validator.report
