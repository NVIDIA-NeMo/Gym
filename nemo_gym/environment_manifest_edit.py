# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated bulk edits for manifest-authoritative environment fields."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from nemo_gym.config_types import ConfigError
from nemo_gym.environment_catalog import (
    EnvironmentCatalogEntry,
    discover_environment_catalog,
    validated_manifest_path,
)
from nemo_gym.environment_manifest import EnvironmentManifest, dump_manifest, load_manifest


# Composition stays authored in Hydra config and identity/profile changes require
# moving or deliberately reclassifying a unit.  Bulk edit is for the fields for
# which the manifest is the authoritative authoring surface.
EDITABLE_ROOTS = frozenset(
    {
        "adopted_from",
        "authors",
        "canonical_split",
        "description",
        "determinism",
        "domain",
        "licensing",
        "lifecycle",
        "modality",
        "provides",
        "requires",
        "reward",
        "sandbox",
        "session_model",
        "standard_prompt_config",
        "state",
    }
)
_FIELD_PATH_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")


@dataclass(frozen=True)
class ManifestEditFilters:
    """Selection shared with the local catalog plus explicit profile/name filters."""

    names: frozenset[str] = frozenset()
    domain: str | None = None
    kind: str | None = None
    profile: str | None = None


@dataclass(frozen=True)
class ManifestEdit:
    path: tuple[str, ...]
    value: Any


@dataclass(frozen=True)
class ManifestEditResult:
    selected: tuple[Path, ...]
    changed: tuple[Path, ...]
    dry_run: bool


def parse_manifest_edits(assignments: Sequence[str]) -> tuple[ManifestEdit, ...]:
    """Parse ``field.path=<YAML value>`` assignments with strict field ownership."""

    if not assignments:
        raise ConfigError("Pass at least one manifest edit as --set <field>=<value>.")
    edits: list[ManifestEdit] = []
    seen: set[tuple[str, ...]] = set()
    for assignment in assignments:
        field, separator, raw_value = assignment.partition("=")
        if not separator or not _FIELD_PATH_RE.fullmatch(field):
            raise ConfigError(f"Invalid manifest edit {assignment!r}; expected --set <field>[.<field>]=<YAML value>.")
        path = tuple(field.split("."))
        if path[0] not in EDITABLE_ROOTS:
            editable = ", ".join(sorted(EDITABLE_ROOTS))
            raise ConfigError(
                f"Manifest field '{path[0]}' is not bulk-editable. Composition is authored in config; "
                f"editable manifest fields are: {editable}."
            )
        if path in seen:
            raise ConfigError(f"Manifest field '{field}' was assigned more than once.")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse value for manifest field '{field}': {exc}.") from exc
        edits.append(ManifestEdit(path=path, value=value))
        seen.add(path)
    return tuple(edits)


def _matches(entry: EnvironmentCatalogEntry, filters: ManifestEditFilters) -> bool:
    if filters.names and entry.name not in filters.names:
        return False
    if filters.domain is not None and entry.domain != filters.domain:
        return False
    if filters.kind is not None and entry.kind != filters.kind:
        return False
    return filters.profile is None or entry.integration_profile == filters.profile


def select_manifest_paths(filters: ManifestEditFilters) -> tuple[Path, ...]:
    """Select only valid manifest-backed runnable units from the authoritative catalog."""

    catalog = discover_environment_catalog(include_legacy=False, include_unpublished=True)
    entries = tuple(entry for entry in catalog.entries if entry.manifest_path is not None and _matches(entry, filters))
    if filters.names:
        matched = {entry.name for entry in entries}
        missing = sorted(filters.names - matched)
        if missing:
            raise ConfigError("Unknown manifest-backed environment(s): " + ", ".join(missing) + ".")
    if not entries:
        raise ConfigError("No manifest-backed environments matched the requested filters.")
    return tuple(entry.manifest_path for entry in entries if entry.manifest_path is not None)


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor: dict[str, Any] = payload
    for part in path[:-1]:
        existing = cursor.get(part)
        if existing is None:
            existing = {}
            cursor[part] = existing
        if not isinstance(existing, dict):
            dotted = ".".join(path)
            raise ConfigError(f"Cannot set '{dotted}': '{part}' is not an object in this manifest.")
        cursor = existing
    cursor[path[-1]] = value


def _validated_edit_path(path: str | Path) -> Path:
    try:
        return validated_manifest_path(path)
    except (OSError, ValueError) as error:
        raise ConfigError(f"Refusing unsafe manifest edit target: {error}.") from error


def apply_manifest_edits(
    paths: Iterable[Path],
    edits: Sequence[ManifestEdit],
    *,
    dry_run: bool = False,
) -> ManifestEditResult:
    """Pre-validate every selected manifest, then atomically write changed files."""

    selected = tuple(dict.fromkeys(_validated_edit_path(path) for path in paths))
    if not selected:
        raise ConfigError("No manifest files were selected for editing.")

    planned: list[tuple[Path, EnvironmentManifest]] = []
    for path in selected:
        manifest = load_manifest(path)
        payload = manifest.model_dump(mode="json", exclude_none=True)
        for edit in edits:
            _set_nested(payload, edit.path, edit.value)
        try:
            updated = EnvironmentManifest.model_validate(payload)
        except Exception as exc:
            raise ConfigError(f"Edits would make manifest '{path}' invalid: {exc}") from exc
        if updated != manifest:
            planned.append((path, updated))

    if not dry_run:
        temporary_paths: list[tuple[Path, Path]] = []
        try:
            for path, manifest in planned:
                # A deterministic temporary name can itself be a malicious
                # symlink in an untrusted checkout. mkstemp creates a new file
                # in the destination directory with O_EXCL semantics.
                rendered = dump_manifest(manifest)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    dir=path.parent,
                )
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                        stream.write(rendered)
                    os.chmod(temporary, path.stat().st_mode)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
                temporary_paths.append((temporary, path))
            for temporary, path in temporary_paths:
                # Recheck immediately before replacement so a target swapped
                # for a symlink after pre-validation is rejected.
                _validated_edit_path(path)
                temporary.replace(path)
        finally:
            for temporary, _path in temporary_paths:
                temporary.unlink(missing_ok=True)

    return ManifestEditResult(
        selected=selected,
        changed=tuple(path for path, _manifest in planned),
        dry_run=dry_run,
    )
