# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Manifest-first discovery for runnable NeMo Gym environments and benchmarks.

The catalog is deliberately separate from the CLI renderer. It combines valid ``manifest.yaml`` files with
the existing config-based environment and benchmark discovery during migration, while retaining the same
component-search-root precedence used everywhere else in Gym. Invalid manifests are reported as structured
issues instead of aborting discovery or disappearing silently.
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional

import yaml

from nemo_gym import component_search_roots
from nemo_gym.config_types import ConfigError
from nemo_gym.discovery import resolve_config_paths_static
from nemo_gym.environment_inventory import (
    MIGRATION_INVENTORY_PATH,
    RunnableUnit,
    discover_runnable_units,
    is_generated_migration_draft,
    migration_draft_paths_from_inventory_content,
    tracked_migration_draft_paths,
)
from nemo_gym.environment_manifest import EnvironmentManifest, load_manifest, manifest_required_capabilities
from nemo_gym.environment_validation import compute_composition_hash
from nemo_gym.environment_version_contract import LOCK_RELATIVE_PATH, environment_version_key
from nemo_gym.environment_versioning import load_version_locks, verify_version_lock
from nemo_gym.repository_io import find_repository_root


CatalogKind = Literal["environment", "benchmark"]
CatalogStatus = Literal["experimental", "no-manifest"]
CatalogSource = Literal["manifest", "legacy"]
_CATALOG_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]*$")
_MANIFEST_TREES: tuple[tuple[CatalogKind, str], ...] = (
    ("environment", "environments"),
    ("benchmark", "benchmarks"),
    ("environment", "resources_servers"),
    ("environment", "example_environments"),
)


def _string_value(value: Any) -> Optional[str]:
    """Return the JSON spelling of a string/enum-like value."""
    if value is None:
        return None
    enum_value = getattr(value, "value", value)
    return str(enum_value)


def _path_json(path: Optional[Path]) -> Optional[str]:
    return str(path) if path is not None else None


def _lexical_absolute_path(path: str | Path) -> Path:
    """Return an absolute, normalized path without following symbolic links."""

    return Path(os.path.abspath(os.fspath(path)))


def validated_manifest_path(path: str | Path, *, roots: Iterable[Path] | None = None) -> Path:
    """Validate one manifest as a real file contained in a supported registry tree.

    Catalog paths can later become write targets through ``gym env manifest``.
    Keep their lexical identity and reject symlinks below the explicitly selected
    component root so discovery can never turn an in-tree link into an out-of-tree
    write target.
    """

    candidate = _lexical_absolute_path(path)
    search_roots = tuple(component_search_roots() if roots is None else roots)
    matched_base: Path | None = None
    matched_root: Path | None = None
    relative: Path | None = None
    for root in search_roots:
        absolute_root = _lexical_absolute_path(root)
        for _kind, subdir in _MANIFEST_TREES:
            base = absolute_root / subdir
            try:
                candidate_relative = candidate.relative_to(base)
            except ValueError:
                continue
            matched_root = absolute_root
            matched_base = base
            relative = candidate_relative
            break
        if matched_base is not None:
            break

    if matched_base is None or matched_root is None or relative is None:
        raise ValueError(f"manifest '{candidate}' is outside the supported component registry trees")
    if relative.name != "manifest.yaml" or len(relative.parts) < 2:
        raise ValueError(f"manifest '{candidate}' is not a component manifest.yaml path")

    # The search root itself may intentionally be a symlink supplied through
    # --search-dir. Everything below that trusted boundary must be lexical: a
    # linked registry directory, component directory, or manifest is rejected.
    cursor = matched_base
    if cursor.is_symlink():
        raise ValueError(f"manifest '{candidate}' is under symbolic-link registry directory '{cursor}'")
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"manifest '{candidate}' contains symbolic-link path component '{cursor}'")

    if not candidate.is_file():
        raise ValueError(f"manifest '{candidate}' is not a regular file")
    try:
        candidate.resolve(strict=True).relative_to(matched_root.resolve(strict=True))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise ValueError(f"manifest '{candidate}' resolves outside component root '{matched_root}'") from error
    return candidate


def _fuzzy_matches(query: str, *fields: str) -> bool:
    """Match catalog text without coupling the discovery layer to CLI rendering helpers."""
    needle = query.casefold()
    for field_value in fields:
        haystack = field_value.casefold()
        if needle in haystack:
            return True
        tokens = haystack.replace("_", " ").replace("-", " ").split()
        if difflib.get_close_matches(needle, [haystack, *tokens], n=1, cutoff=0.70):
            return True
    return False


@dataclass(frozen=True)
class EnvironmentCatalogEntry:
    """One immutable, normalized catalog record.

    Fields unavailable to the legacy scraper remain ``None`` rather than being guessed. This distinction is
    what lets filtered catalog views explain which legacy entries they could not evaluate.
    """

    name: str
    kind: CatalogKind
    status: CatalogStatus
    source: CatalogSource
    config_path: Optional[Path]
    manifest_path: Optional[Path]
    version: Optional[str] = None
    integration_profile: Optional[str] = None
    domain: Optional[str] = None
    description: Optional[str] = None
    modality: Optional[str] = None
    licensing: Optional[str] = None
    lifecycle: Optional[str] = None
    authors: tuple[str, ...] = ()
    determinism: Optional[str] = None
    required_capabilities: Optional[frozenset[str]] = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return the stable machine-readable representation used by CLI/catalog generators."""
        return {
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "status": self.status,
            "lifecycle": self.lifecycle,
            "integration_profile": self.integration_profile,
            "domain": self.domain,
            "description": self.description,
            "modality": self.modality,
            "licensing": self.licensing,
            "authors": list(self.authors),
            "determinism": self.determinism,
            "required_capabilities": (
                sorted(self.required_capabilities) if self.required_capabilities is not None else None
            ),
            "source": self.source,
            "config_path": _path_json(self.config_path),
            "manifest_path": _path_json(self.manifest_path),
        }


@dataclass(frozen=True)
class CatalogIssue:
    """A manifest that could not be included in the catalog."""

    path: Path
    message: str
    code: str = "invalid-manifest"

    def to_json_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": str(self.path), "message": self.message}


@dataclass(frozen=True)
class CatalogCoverage:
    """Manifest migration coverage before any catalog filters are applied."""

    total: int
    with_manifest: int
    without_manifest: int
    invalid_manifests: int

    @property
    def percent(self) -> float:
        return 100.0 if self.total == 0 else 100.0 * self.with_manifest / self.total

    def to_json_dict(self) -> dict[str, int | float]:
        return {
            "total": self.total,
            "with_manifest": self.with_manifest,
            "without_manifest": self.without_manifest,
            "invalid_manifests": self.invalid_manifests,
            "percent": self.percent,
        }


@dataclass(frozen=True)
class CatalogFilterLimitation:
    """Entries omitted because a requested field was unavailable, normally on legacy records."""

    field: str
    entry_names: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.entry_names)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "count": self.count,
            "entry_names": list(self.entry_names),
            "message": (
                f"Could not apply '{self.field}' to {self.count} catalog entr{'y' if self.count == 1 else 'ies'}."
            ),
        }


@dataclass(frozen=True)
class CatalogFilters:
    """Exact metadata filters plus a fuzzy free-text query."""

    name: Optional[str] = None
    query: Optional[str] = None
    domain: Optional[str] = None
    kind: Optional[CatalogKind] = None
    modality: Optional[str] = None
    licensing: Optional[str] = None
    lifecycle: Optional[str] = None
    status: Optional[CatalogStatus] = None
    required_capabilities: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class EnvironmentCatalog:
    """Catalog records plus diagnostics and migration/filter reporting."""

    entries: tuple[EnvironmentCatalogEntry, ...]
    coverage: CatalogCoverage
    issues: tuple[CatalogIssue, ...] = ()
    filter_limitations: tuple[CatalogFilterLimitation, ...] = ()

    def filtered(self, filters: Optional[CatalogFilters] = None) -> "EnvironmentCatalog":
        if filters is None:
            return self

        unavailable: dict[str, list[str]] = {}
        matched: list[EnvironmentCatalogEntry] = []

        def exact(entry: EnvironmentCatalogEntry, field_name: str, expected: Optional[str]) -> bool:
            if expected is None:
                return True
            actual = getattr(entry, field_name)
            if actual is None:
                unavailable.setdefault(field_name, []).append(entry.name)
                return False
            return actual.casefold() == expected.casefold()

        for entry in self.entries:
            if filters.name is not None and entry.name != filters.name:
                continue
            if filters.query is not None and not _fuzzy_matches(
                filters.query,
                entry.name,
                entry.description or "",
                entry.domain or "",
            ):
                continue
            if not exact(entry, "domain", filters.domain):
                continue
            if filters.kind is not None and entry.kind != filters.kind:
                continue
            if not exact(entry, "modality", filters.modality):
                continue
            if not exact(entry, "licensing", filters.licensing):
                continue
            if not exact(entry, "lifecycle", filters.lifecycle):
                continue
            if filters.status is not None and entry.status != filters.status:
                continue
            if filters.required_capabilities:
                if entry.required_capabilities is None:
                    unavailable.setdefault("required_capabilities", []).append(entry.name)
                    continue
                if not filters.required_capabilities.issubset(entry.required_capabilities):
                    continue
            matched.append(entry)

        limitations = tuple(
            CatalogFilterLimitation(field=field_name, entry_names=tuple(sorted(set(names))))
            for field_name, names in sorted(unavailable.items())
        )
        return EnvironmentCatalog(
            entries=tuple(matched),
            coverage=self.coverage,
            issues=self.issues,
            filter_limitations=limitations,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.to_json_dict() for entry in self.entries],
            "coverage": self.coverage.to_json_dict(),
            "issues": [issue.to_json_dict() for issue in self.issues],
            "filter_limitations": [limitation.to_json_dict() for limitation in self.filter_limitations],
        }


@dataclass(frozen=True)
class _CatalogCandidate:
    entry: EnvironmentCatalogEntry
    root_index: int
    source_priority: int


@dataclass
class _CatalogDiscovery:
    """Mutable state shared by the manifest, legacy, and finalization phases."""

    candidates: dict[tuple[CatalogKind, str], list[_CatalogCandidate]] = field(default_factory=dict)
    issues: list[CatalogIssue] = field(default_factory=list)
    invalid_rank: dict[tuple[CatalogKind, str], int] = field(default_factory=dict)
    manifested_configs: set[Path] = field(default_factory=set)
    claimed_resource_configs: set[Path] = field(default_factory=set)
    reported_lock_issues: set[tuple[Path, str]] = field(default_factory=set)


def _root_index(path: Path, roots: tuple[Path, ...]) -> int:
    resolved = path.resolve()
    for index, root in enumerate(roots):
        try:
            resolved.relative_to(root.resolve())
            return index
        except ValueError:
            continue
    return len(roots)


def manifest_config_path(manifest_path: Path) -> Optional[Path]:
    """Resolve the sole runnable config owned by a manifest directory."""

    config_path = manifest_path.parent / "config.yaml"
    if config_path.is_file():
        return config_path.resolve()
    configs_dir = manifest_path.parent / "configs"
    preferred = configs_dir / f"{manifest_path.parent.name}.yaml"
    if preferred.is_file():
        return preferred.resolve()
    candidates = sorted((*configs_dir.glob("*.yaml"), *configs_dir.glob("*.yml"))) if configs_dir.is_dir() else []
    return candidates[0].resolve() if len(candidates) == 1 else None


def _manifest_capabilities(data: dict[str, Any]) -> frozenset[str]:
    return frozenset(manifest_required_capabilities(data))


def _entry_from_manifest(manifest_path: Path, manifest: Any | None = None) -> EnvironmentCatalogEntry:
    manifest = manifest or load_manifest(manifest_path)
    data = manifest.model_dump(mode="json")
    return EnvironmentCatalogEntry(
        name=str(data["name"]),
        kind=str(data["kind"]),
        status="experimental",
        source="manifest",
        config_path=manifest_config_path(manifest_path),
        # Preserve the discovered lexical path. Resolving here would erase a
        # manifest symlink before a later bulk-edit safety check could see it.
        manifest_path=_lexical_absolute_path(manifest_path),
        version=str(data["version"]),
        integration_profile=_string_value(data.get("integration_profile")),
        domain=_string_value(data.get("domain")),
        description=_string_value(data.get("description")),
        modality=_string_value(data.get("modality")),
        licensing=_string_value(data.get("licensing")),
        lifecycle=_string_value(data.get("lifecycle")) or "active",
        authors=tuple(str(author) for author in data.get("authors") or []),
        determinism=_string_value(data.get("determinism")),
        required_capabilities=_manifest_capabilities(data),
    )


def _manifest_publication_record(
    root: Path,
    manifest_path: Path,
    config_path: Path,
    manifest: EnvironmentManifest,
) -> tuple[Mapping[str, Any] | None, CatalogIssue | None]:
    """Return the lock record only when the live manifest composition still matches it."""

    repository_root = find_repository_root(manifest_path) or find_repository_root(root) or root.resolve()
    lock_path = repository_root / LOCK_RELATIVE_PATH

    try:
        locks = load_version_locks(lock_path)
    except ConfigError as error:
        return None, CatalogIssue(lock_path, str(error), code="invalid-version-lock")

    version_key = environment_version_key(manifest)
    record = locks["environments"].get(version_key)
    if not isinstance(record, Mapping):
        return None, None

    try:
        resolved_config = resolve_config_paths_static((config_path,))
        composition_hash = compute_composition_hash(resolved_config, manifest)
        verified_record = verify_version_lock(
            repo_root=repository_root,
            manifest_path=manifest_path,
            config_path=config_path,
            manifest=manifest,
            composition_hash=composition_hash,
            require_published=True,
        )
    except Exception as error:
        return None, CatalogIssue(
            lock_path,
            _issue_message(error),
            code="invalid-version-lock",
        )
    return verified_record, None


def _manifest_paths(roots: tuple[Path, ...]):
    for root_index, root in enumerate(roots):
        for kind, subdir in _MANIFEST_TREES:
            base = root / subdir
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("manifest.yaml")):
                expected_name = path.parent.relative_to(base).as_posix()
                yield root_index, kind, expected_name, path


def _issue_message(error: Exception) -> str:
    message = str(error).strip()
    return message or type(error).__name__


def _legacy_resource_name(config_path: Path) -> str:
    """Return a collision-free transitional identity for one runnable config."""

    component = config_path.parent.parent.name
    return f"resources_servers/{component}/{config_path.stem}"


def _read_raw_config_metadata(config_path: Path) -> tuple[str | None, str | None]:
    """Read inline legacy metadata without resolving the runtime composition."""

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None, None
    if not isinstance(payload, Mapping):
        return None, None
    domain: str | None = None
    description: str | None = None
    for instance in payload.values():
        if not isinstance(instance, Mapping):
            continue
        for group in ("resources_servers", "responses_api_agents", "responses_api_models"):
            servers = instance.get(group)
            if not isinstance(servers, Mapping):
                continue
            for server in servers.values():
                if not isinstance(server, Mapping):
                    continue
                if domain is None and server.get("domain"):
                    domain = str(server["domain"])
                if description is None and server.get("description"):
                    description = str(server["description"])
                if domain is not None and description is not None:
                    return domain, description
    return domain, description


def _add_candidate(
    discovery: _CatalogDiscovery,
    entry: EnvironmentCatalogEntry,
    *,
    root_index: int,
    source_priority: int,
) -> None:
    discovery.candidates.setdefault((entry.kind, entry.name), []).append(
        _CatalogCandidate(entry=entry, root_index=root_index, source_priority=source_priority)
    )


def _load_manifest_candidate(
    roots: tuple[Path, ...],
    root_index: int,
    fallback_kind: CatalogKind,
    expected_name: str,
    manifest_path: Path,
) -> tuple[Path, EnvironmentManifest, EnvironmentCatalogEntry]:
    validated_path = validated_manifest_path(manifest_path, roots=(roots[root_index],))
    manifest = load_manifest(validated_path)
    entry = _entry_from_manifest(validated_path, manifest)
    if entry.kind != fallback_kind:
        raise ValueError(f"kind={entry.kind!r} does not match its '{validated_path.parent.parent.name}' registry tree")
    if entry.name != expected_name:
        raise ValueError(f"name={entry.name!r} does not match registry path identity {expected_name!r}")
    if entry.config_path is None:
        raise ValueError("manifest has no unambiguous runnable config.yaml/configs/*.yaml")
    return validated_path, manifest, entry


def _record_manifest_error(
    discovery: _CatalogDiscovery,
    *,
    manifest_path: Path,
    fallback_kind: CatalogKind,
    expected_name: str,
    root_index: int,
    migration_drafts: frozenset[Path],
    error: Exception,
) -> None:
    lexical_manifest_path = _lexical_absolute_path(manifest_path)
    issue_code = "migration-draft" if lexical_manifest_path in migration_drafts else "invalid-manifest"
    discovery.issues.append(
        CatalogIssue(
            path=lexical_manifest_path,
            message=_issue_message(error),
            code=issue_code,
        )
    )
    if issue_code == "invalid-manifest":
        inferred_key = (fallback_kind, expected_name)
        discovery.invalid_rank[inferred_key] = min(
            root_index,
            discovery.invalid_rank.get(inferred_key, root_index),
        )


def _record_publication_issue(discovery: _CatalogDiscovery, issue: CatalogIssue | None) -> None:
    if issue is None:
        return
    issue_key = (issue.path, issue.message)
    if issue_key in discovery.reported_lock_issues:
        return
    discovery.issues.append(issue)
    discovery.reported_lock_issues.add(issue_key)


def _claim_manifest_resource_config(
    discovery: _CatalogDiscovery,
    entry: EnvironmentCatalogEntry,
    manifest: EnvironmentManifest,
    resource_units: tuple[tuple[int, RunnableUnit], ...],
) -> None:
    resource_name = manifest.model_dump(mode="json").get("resources_server")
    if not isinstance(resource_name, str) or not resource_name:
        return
    matching = [(index, unit) for index, unit in resource_units if unit.name == resource_name]
    if not matching:
        return
    first_root = min(index for index, _unit in matching)
    same_root = [unit for index, unit in matching if index == first_root]
    identity_leaf = Path(entry.name).name
    selected = [unit for unit in same_root if unit.config_path.stem == identity_leaf]
    if not selected and len(same_root) == 1:
        selected = same_root
    discovery.claimed_resource_configs.update(unit.config_path.resolve() for unit in selected)


def _discover_manifest_candidates(
    discovery: _CatalogDiscovery,
    *,
    roots: tuple[Path, ...],
    migration_drafts: frozenset[Path],
    resource_units: tuple[tuple[int, RunnableUnit], ...],
    include_unpublished: bool,
    manifest_paths: Iterable[tuple[int, CatalogKind, str, Path]] | None = None,
) -> None:
    candidates = _manifest_paths(roots) if manifest_paths is None else manifest_paths
    for root_index, fallback_kind, expected_name, manifest_path in candidates:
        try:
            validated_path, manifest, entry = _load_manifest_candidate(
                roots,
                root_index,
                fallback_kind,
                expected_name,
                manifest_path,
            )
        except Exception as error:
            _record_manifest_error(
                discovery,
                manifest_path=manifest_path,
                fallback_kind=fallback_kind,
                expected_name=expected_name,
                root_index=root_index,
                migration_drafts=migration_drafts,
                error=error,
            )
            continue

        assert entry.config_path is not None
        publication_record, lock_issue = _manifest_publication_record(
            roots[root_index],
            validated_path,
            entry.config_path,
            manifest,
        )
        _record_publication_issue(discovery, lock_issue)
        if publication_record is None and not include_unpublished:
            continue

        entry = replace(entry, status="experimental")
        _add_candidate(discovery, entry, root_index=root_index, source_priority=0)
        if entry.config_path is not None:
            discovery.manifested_configs.add(entry.config_path.resolve())
        _claim_manifest_resource_config(discovery, entry, manifest, resource_units)


def _discover_legacy_recipe_candidates(
    discovery: _CatalogDiscovery,
    recipe_units: tuple[tuple[int, RunnableUnit], ...],
) -> None:
    for root_index, unit in recipe_units:
        config_path = unit.config_path.resolve()
        domain, description = _read_raw_config_metadata(config_path)
        entry = EnvironmentCatalogEntry(
            name=unit.name,
            kind=unit.kind,
            status="no-manifest",
            source="legacy",
            config_path=config_path,
            manifest_path=None,
            domain=domain,
            description=description,
        )
        _add_candidate(discovery, entry, root_index=root_index, source_priority=1)


def _discover_legacy_resource_candidates(
    discovery: _CatalogDiscovery,
    resource_units: tuple[tuple[int, RunnableUnit], ...],
) -> None:
    ambiguous_components: set[tuple[int, Path]] = set()
    for root_index, unit in resource_units:
        config_path = unit.config_path.resolve()
        # A manifest suppresses only the exact legacy config it selects or claims.
        if config_path in discovery.manifested_configs or config_path in discovery.claimed_resource_configs:
            continue
        if unit.blocker:
            component_path = unit.manifest_path.parent.resolve()
            issue_key = (root_index, component_path)
            if issue_key not in ambiguous_components:
                discovery.issues.append(
                    CatalogIssue(
                        path=component_path,
                        message=unit.blocker,
                        code="ambiguous-legacy-resource",
                    )
                )
                ambiguous_components.add(issue_key)
        domain, description = _read_raw_config_metadata(config_path)
        entry = EnvironmentCatalogEntry(
            name=_legacy_resource_name(config_path),
            kind="environment",
            status="no-manifest",
            source="legacy",
            config_path=config_path,
            manifest_path=None,
            domain=domain,
            description=description,
        )
        _add_candidate(discovery, entry, root_index=root_index, source_priority=1)


def _discover_legacy_candidates(
    discovery: _CatalogDiscovery,
    *,
    recipe_units: tuple[tuple[int, RunnableUnit], ...],
    resource_units: tuple[tuple[int, RunnableUnit], ...],
) -> None:
    _discover_legacy_recipe_candidates(discovery, recipe_units)
    _discover_legacy_resource_candidates(discovery, resource_units)


def _selected_catalog_entries(
    discovery: _CatalogDiscovery,
    roots: tuple[Path, ...],
) -> tuple[EnvironmentCatalogEntry, ...]:
    selected: dict[tuple[CatalogKind, str], EnvironmentCatalogEntry] = {}
    for key, choices in discovery.candidates.items():
        winner = min(choices, key=lambda candidate: (candidate.root_index, candidate.source_priority))
        # Do not silently replace an invalid higher-priority manifest with a lower-root entry.
        if discovery.invalid_rank.get(key, len(roots) + 1) <= winner.root_index:
            continue
        selected[key] = winner.entry
    return tuple(sorted(selected.values(), key=lambda entry: (entry.name.casefold(), entry.kind)))


def _finalize_catalog(
    discovery: _CatalogDiscovery,
    *,
    roots: tuple[Path, ...],
    filters: Optional[CatalogFilters],
) -> EnvironmentCatalog:
    entries = _selected_catalog_entries(discovery, roots)
    selected_keys = {(entry.kind, entry.name) for entry in entries}
    invalid_only = sum(key not in selected_keys for key in discovery.invalid_rank)
    coverage = CatalogCoverage(
        total=len(entries) + invalid_only,
        with_manifest=sum(entry.source == "manifest" for entry in entries),
        without_manifest=sum(entry.status == "no-manifest" for entry in entries) + invalid_only,
        invalid_manifests=sum(issue.code == "invalid-manifest" for issue in discovery.issues),
    )
    return EnvironmentCatalog(entries=entries, coverage=coverage, issues=tuple(discovery.issues)).filtered(filters)


def discover_environment_catalog(
    filters: Optional[CatalogFilters] = None,
    *,
    include_legacy: bool = True,
    include_unpublished: bool = False,
) -> EnvironmentCatalog:
    """Discover the published manifest/legacy union and optionally apply filters.

    Search-root precedence is evaluated before source priority: a user's legacy config shadows a built-in
    manifest of the same kind/name, while a manifest wins over a legacy config in the same root. Authoring
    and CI callers may opt into valid but not-yet-published manifests; runtime discovery does not.
    """
    roots = tuple(component_search_roots())
    migration_drafts = frozenset(draft_path for root in roots for draft_path in tracked_migration_draft_paths(root))
    legacy_units = (
        tuple((root_index, unit) for root_index, root in enumerate(roots) for unit in discover_runnable_units(root))
        if include_legacy
        else ()
    )
    resource_units = tuple(item for item in legacy_units if item[1].registry == "resources_servers")
    recipe_units = tuple(item for item in legacy_units if item[1].registry != "resources_servers")
    discovery = _CatalogDiscovery()
    _discover_manifest_candidates(
        discovery,
        roots=roots,
        migration_drafts=migration_drafts,
        resource_units=resource_units,
        include_unpublished=include_unpublished,
    )
    if include_legacy:
        _discover_legacy_candidates(
            discovery,
            recipe_units=recipe_units,
            resource_units=resource_units,
        )
    return _finalize_catalog(discovery, roots=roots, filters=filters)


def discover_exact_manifest_catalog(
    name: str,
    kind: CatalogKind | None = None,
    *,
    include_unpublished: bool = False,
) -> EnvironmentCatalog:
    """Resolve direct manifest candidates without walking every registry tree.

    Runtime commands usually know the exact environment name. Keeping that path
    separate from catalog listing avoids parsing hundreds of unrelated manifests
    before pre-execution validation. Legacy discovery remains an explicit fallback
    in the CLI because legacy names do not have one uniform on-disk identity.
    """

    if not _CATALOG_NAME_RE.fullmatch(name) or any(segment in {"", ".", ".."} for segment in name.split("/")):
        return EnvironmentCatalog(entries=(), coverage=CatalogCoverage(0, 0, 0, 0))

    return discover_exact_environment_catalog(
        name,
        kind,
        include_legacy=False,
        include_unpublished=include_unpublished,
    )


def _direct_migration_draft_paths(
    roots: tuple[Path, ...],
    manifest_paths: Iterable[tuple[int, CatalogKind, str, Path]],
) -> frozenset[Path]:
    candidates = {_lexical_absolute_path(path) for _root_index, _kind, _name, path in manifest_paths}
    drafts: set[Path] = set()
    for root in roots:
        try:
            content = (root / MIGRATION_INVENTORY_PATH).read_text(encoding="utf-8")
        except OSError:
            continue
        tracked = migration_draft_paths_from_inventory_content(root, content)
        drafts.update(path for path in candidates & tracked if is_generated_migration_draft(path))
    return frozenset(drafts)


def _yaml_declares(value: object, key: str, expected: object | None = None) -> bool:
    if isinstance(value, Mapping):
        if key in value and (expected is None or value[key] == expected):
            return True
        return any(_yaml_declares(item, key, expected) for item in value.values())
    if isinstance(value, list):
        return any(_yaml_declares(item, key, expected) for item in value)
    return False


def _direct_config_payload(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None


def _add_direct_legacy_candidates(
    discovery: _CatalogDiscovery,
    *,
    roots: tuple[Path, ...],
    name: str,
    kind: CatalogKind | None,
) -> None:
    if kind in {None, "environment"}:
        for root_index, root in enumerate(roots):
            for registry in ("environments", "example_environments"):
                config_path = root / registry / name / "config.yaml"
                if not config_path.is_file():
                    continue
                domain, description = _read_raw_config_metadata(config_path)
                _add_candidate(
                    discovery,
                    EnvironmentCatalogEntry(
                        name=name,
                        kind="environment",
                        status="no-manifest",
                        source="legacy",
                        config_path=config_path.resolve(),
                        manifest_path=None,
                        domain=domain,
                        description=description,
                    ),
                    root_index=root_index,
                    source_priority=1,
                )

        resource_parts = Path(name).parts
        if len(resource_parts) == 3 and resource_parts[0] == "resources_servers":
            _prefix, component, flavor = resource_parts
            for root_index, root in enumerate(roots):
                for suffix in (".yaml", ".yml"):
                    config_path = root / "resources_servers" / component / "configs" / f"{flavor}{suffix}"
                    payload = _direct_config_payload(config_path)
                    if not (
                        config_path.is_file()
                        and _yaml_declares(payload, "resources_servers")
                        and _yaml_declares(payload, "responses_api_agents")
                        and _yaml_declares(payload, "datasets")
                    ):
                        continue
                    domain, description = _read_raw_config_metadata(config_path)
                    _add_candidate(
                        discovery,
                        EnvironmentCatalogEntry(
                            name=name,
                            kind="environment",
                            status="no-manifest",
                            source="legacy",
                            config_path=config_path.resolve(),
                            manifest_path=None,
                            domain=domain,
                            description=description,
                        ),
                        root_index=root_index,
                        source_priority=1,
                    )

    if kind in {None, "benchmark"}:
        for root_index, root in enumerate(roots):
            base = root / "benchmarks"
            candidates = (
                base / name / "config.yaml",
                base / name / "config.yml",
                base / f"{name}.yaml",
                base / f"{name}.yml",
            )
            for config_path in dict.fromkeys(candidates):
                payload = _direct_config_payload(config_path)
                if not config_path.is_file() or not _yaml_declares(payload, "type", "benchmark"):
                    continue
                domain, description = _read_raw_config_metadata(config_path)
                _add_candidate(
                    discovery,
                    EnvironmentCatalogEntry(
                        name=name,
                        kind="benchmark",
                        status="no-manifest",
                        source="legacy",
                        config_path=config_path.resolve(),
                        manifest_path=None,
                        domain=domain,
                        description=description,
                    ),
                    root_index=root_index,
                    source_priority=1,
                )


def discover_exact_environment_catalog(
    name: str,
    kind: CatalogKind | None = None,
    *,
    include_legacy: bool = True,
    include_unpublished: bool = False,
) -> EnvironmentCatalog:
    """Discover one exact catalog identity without scanning unrelated configs."""

    if not _CATALOG_NAME_RE.fullmatch(name) or any(segment in {"", ".", ".."} for segment in name.split("/")):
        return EnvironmentCatalog(entries=(), coverage=CatalogCoverage(0, 0, 0, 0))

    roots = tuple(component_search_roots())
    selected_trees = tuple(
        (candidate_kind, subdir)
        for candidate_kind, subdir in _MANIFEST_TREES
        if kind is None or candidate_kind == kind
    )
    direct_paths = tuple(
        (root_index, candidate_kind, name, manifest_path)
        for root_index, root in enumerate(roots)
        for candidate_kind, subdir in selected_trees
        for manifest_path in (root / subdir / name / "manifest.yaml",)
        if manifest_path.is_file() or manifest_path.is_symlink()
    )
    migration_drafts = _direct_migration_draft_paths(roots, direct_paths)
    discovery = _CatalogDiscovery()
    _discover_manifest_candidates(
        discovery,
        roots=roots,
        migration_drafts=migration_drafts,
        resource_units=(),
        include_unpublished=include_unpublished,
        manifest_paths=direct_paths,
    )
    if include_legacy:
        _add_direct_legacy_candidates(discovery, roots=roots, name=name, kind=kind)
    return _finalize_catalog(
        discovery,
        roots=roots,
        filters=CatalogFilters(name=name, kind=kind),
    )
