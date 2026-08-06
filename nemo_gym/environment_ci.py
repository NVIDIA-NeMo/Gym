# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report and enforce the local environment-onboarding checks in CI.

The full manifest/legacy union is inspected on every invocation.  Migration-era
failures are report-only unless the unit itself changed or a score-affecting
shared component fans out to a manifest-backed dependent.  Nothing in this
module imports an environment implementation, starts Ray, or provisions a
service.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from omegaconf import DictConfig, OmegaConf

from nemo_gym.config_types import ConfigError
from nemo_gym.discovery import SERVER_ROLE_BY_GROUP, _parse_no_environment_tolerating_unset_values
from nemo_gym.environment_catalog import (
    EnvironmentCatalog,
    EnvironmentCatalogEntry,
    discover_environment_catalog,
    manifest_config_path,
)
from nemo_gym.environment_files import resolve_runtime_local_references
from nemo_gym.environment_inventory import (
    MIGRATION_INVENTORY_PATH,
    discover_runnable_units,
    is_generated_migration_draft,
    is_runnable_unit_config_content,
)
from nemo_gym.environment_manifest import (
    EnvironmentManifest,
    ManifestError,
    load_manifest,
    manifest_json_schema,
)
from nemo_gym.environment_validation import (
    WorkloadInspection,
    inspect_workload,
    is_credential_key,
    manifest_composition_deltas,
    resolve_component_provenance,
    resolve_component_source_directory,
    resolve_composition_mirror,
    resolve_dataset_preparation_provenance,
    resolve_rollout_driver_provenance,
)
from nemo_gym.environment_versioning import (
    LOCK_RELATIVE_PATH,
    environment_version_key,
    load_version_locks,
    validate_version_locks,
)
from nemo_gym.global_config import (
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    GlobalConfigDictParserConfig,
)
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.verifier_ci_harness import (
    VERIFIER_HARNESS_MODULE,
    build_verifier_harness_invocation,
    select_resources_server_runtime,
    select_sole_resources_server_runtime,
)
from nemo_gym.verifier_fixture import (
    DETERMINISM_ENV_VAR,
    HIGHER_IS_BETTER_ENV_VAR,
    REWARD_RANGE_ENV_VAR,
    VerifierFixtureError,
    load_verifier_fixture,
    validate_verifier_fixture,
    verifier_fixture_environment,
)


_REGISTRY_KINDS = {
    "environments": "environment",
    "benchmarks": "benchmark",
    "resources_servers": "environment",
    "example_environments": "environment",
}
_COMPONENT_ROLES = SERVER_ROLE_BY_GROUP
_DEPENDENCY_TREES = frozenset({*_COMPONENT_ROLES, "benchmarks", "environments"})
_STOCK_ROLLOUT_DEPENDENCY = ("nemo_gym", "rollout_collection.py")
_SANDBOX_PROVIDER_DEPENDENCY_GROUP = "nemo_gym/sandbox/providers"
_SANDBOX_PROVIDER_PATH_PREFIX = ("nemo_gym", "sandbox", "providers")
_INVENTORY_UNIT_STATUSES = frozenset({"planned", "drafted", "already-manifest", "exception"})
_BOOTSTRAP_LEGACY_STATUSES = frozenset({"planned", "exception"})
_METADATA_KEYS = frozenset({"aliases", "description", "license", "licensing", "modality"})
_RUNTIME_KEYS = frozenset({"host", "num_workers", "port"})
_VERIFIER_FIXTURE_RELATIVE_PATH = Path("tests/verifier_cases.jsonl")
_VERIFIER_TEST_RELATIVE_PATH = Path("tests/test_app.py")
_COMPONENT_VERIFIER_TEST_NODE = "test_verifier_fixture"
_VERIFIER_OUTPUT_LIMIT = 12_000
_LEGACY_PYTEST_NODE = "pytest"


@dataclass(frozen=True)
class ChangedFile:
    """One repository-relative Git change and optional base content."""

    path: Path
    status: str = "M"
    old_path: Path | None = None
    before_content: str | None = None

    def paths(self) -> tuple[Path, ...]:
        return (self.path,) if self.old_path is None else (self.path, self.old_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.as_posix(),
            "status": self.status,
            "old_path": self.old_path.as_posix() if self.old_path else None,
        }


@dataclass
class GateUnitResult:
    """Static checks and enforcement state for one catalog unit."""

    name: str
    kind: str
    source: str
    config_path: Path | None
    manifest_path: Path | None
    enforced: bool = False
    reasons: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dependencies: set[tuple[str, str]] = field(default_factory=set)
    composition_hash: str | None = None
    version_key: str | None = None
    composition_change: bool = False
    bootstrap_legacy_exempt: bool = False
    verifier_test_path: Path | None = None
    verifier_component_dir: Path | None = None
    verifier_entrypoint_path: Path | None = None
    verifier_instance_name: str | None = None
    verifier_server_config: dict[str, Any] | None = field(default=None, repr=False)
    resolved_config: DictConfig | None = field(default=None, repr=False)
    verifier_checks: list[VerifierCheckResult] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return self.kind, self.name

    def to_dict(self, repo_root: Path) -> dict[str, Any]:
        def relative(path: Path | None) -> str | None:
            if path is None:
                return None
            try:
                return path.resolve().relative_to(repo_root.resolve()).as_posix()
            except ValueError:
                return path.resolve().as_posix()

        return {
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "config_path": relative(self.config_path),
            "manifest_path": relative(self.manifest_path),
            "mode": "enforce" if self.enforced else "report-only",
            "reasons": sorted(self.reasons),
            "errors": self.errors,
            "warnings": self.warnings,
            "dependencies": [f"{group}/{name}" for group, name in sorted(self.dependencies)],
            "composition_hash": self.composition_hash,
            "version_key": self.version_key,
            "bootstrap_legacy_exempt": self.bootstrap_legacy_exempt,
            "verifier_test_path": relative(self.verifier_test_path),
            "verifier_entrypoint_path": relative(self.verifier_entrypoint_path),
            "verifier_checks": [check.to_dict(repo_root) for check in self.verifier_checks],
        }


@dataclass(frozen=True)
class VerifierCheckResult:
    """One unit-bound execution of an offline scorer or legacy local test."""

    test_path: Path
    passed: bool
    returncode: int | None
    output: str = ""
    node: str = VERIFIER_HARNESS_MODULE

    def to_dict(self, repo_root: Path) -> dict[str, Any]:
        try:
            path = self.test_path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            path = self.test_path.resolve().as_posix()
        return {
            "test_path": path,
            "node": self.node,
            "passed": self.passed,
            "returncode": self.returncode,
            "output": self.output,
        }


@dataclass(frozen=True)
class _VerifierCheckPlan:
    """A resolved, dedupe-safe no-Ray check for one enforced unit."""

    mode: Literal["fixture", "pytest"]
    component_dir: Path
    entrypoint_path: Path
    instance_name: str
    server_config: dict[str, Any]
    check_path: Path
    cache_key: tuple[str, ...]
    fixture_path: Path | None = None
    resolved_config_yaml: str | None = None
    reward_range: tuple[int | float, int | float] | None = None
    higher_is_better: bool = True
    determinism: str | None = None


@dataclass(frozen=True)
class EnvironmentGateReport:
    """Machine-readable CI result with a migration-safe pass/fail policy."""

    repo_root: Path
    enforce_changes: bool
    catalog: EnvironmentCatalog
    changes: tuple[ChangedFile, ...]
    units: tuple[GateUnitResult, ...]
    schema_errors: tuple[str, ...] = ()
    lock_violations: tuple[str, ...] = ()

    @property
    def enforced_errors(self) -> tuple[str, ...]:
        if not self.enforce_changes:
            return ()
        errors = [*self.schema_errors, *self.lock_violations]
        for unit in self.units:
            if unit.enforced:
                errors.extend(f"{unit.kind}:{unit.name}: {message}" for message in unit.errors)
        return tuple(errors)

    @property
    def passed(self) -> bool:
        return not self.enforced_errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "mode": "enforce-changes" if self.enforce_changes else "report-only",
            "coverage": self.catalog.coverage.to_json_dict(),
            "catalog_issues": [issue.to_json_dict() for issue in self.catalog.issues],
            "changed_files": [change.to_dict() for change in self.changes],
            "schema_errors": list(self.schema_errors),
            "lock_violations": list(self.lock_violations),
            "summary": {
                "total": len(self.units),
                "enforced": sum(unit.enforced for unit in self.units),
                "report_only": sum(not unit.enforced for unit in self.units),
                "units_with_errors": sum(bool(unit.errors) for unit in self.units),
                "enforced_errors": len(self.enforced_errors),
            },
            "units": [unit.to_dict(self.repo_root) for unit in self.units],
        }


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _run_git(repo_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )


def changed_files_from_git(repo_root: Path, base_ref: str, head_ref: str = "HEAD") -> tuple[ChangedFile, ...]:
    """Read a rename-aware Git diff without mutating the checkout."""

    result = _run_git(repo_root, "diff", "--name-status", "--find-renames", f"{base_ref}...{head_ref}")
    if result.returncode:
        raise ConfigError(result.stderr.strip() or f"Could not diff {base_ref}...{head_ref}.")
    changes: list[ChangedFile] = []
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        status = fields[0]
        old_path = Path(fields[1]) if status.startswith(("R", "C")) and len(fields) >= 3 else None
        path = Path(fields[2]) if old_path is not None else Path(fields[1])
        before_path = old_path or path
        before = _run_git(repo_root, "show", f"{base_ref}:{before_path.as_posix()}")
        changes.append(
            ChangedFile(
                path=path,
                status=status,
                old_path=old_path,
                before_content=before.stdout if before.returncode == 0 else None,
            )
        )
    return tuple(changes)


def _scrub_mapping(value: object, *, ignored_keys: frozenset[str]) -> object:
    if isinstance(value, Mapping):
        scrubbed: dict[str, object] = {}
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in ignored_keys or is_credential_key(key):
                continue
            scrubbed[str(key)] = _scrub_mapping(item, ignored_keys=ignored_keys)
        return scrubbed
    if isinstance(value, list):
        return [_scrub_mapping(item, ignored_keys=ignored_keys) for item in value]
    return value


def _structured_change_is_only(
    repo_root: Path,
    change: ChangedFile,
    *,
    ignored_keys: frozenset[str],
) -> bool:
    if change.before_content is None or change.status.startswith(("A", "D")):
        return False
    current_path = repo_root / change.path
    if not current_path.is_file() or current_path.suffix.casefold() not in {".json", ".yaml", ".yml"}:
        return False
    try:
        if current_path.suffix.casefold() == ".json":
            before = json.loads(change.before_content)
            after = json.loads(current_path.read_text(encoding="utf-8"))
        else:
            before = yaml.safe_load(change.before_content)
            after = yaml.safe_load(current_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError):
        return False
    return _scrub_mapping(before, ignored_keys=ignored_keys) == _scrub_mapping(after, ignored_keys=ignored_keys)


def is_metadata_only_change(repo_root: Path, change: ChangedFile) -> bool:
    """Whether an authored diff changes only metadata exempt from dependent fan-out."""

    return _structured_change_is_only(repo_root, change, ignored_keys=_METADATA_KEYS)


def _is_documentation_change(change: ChangedFile) -> bool:
    names = {path.name.casefold() for path in change.paths()}
    return all(path.suffix.casefold() == ".md" or path.name.casefold() == "license" for path in change.paths()) or all(
        name.startswith("readme") for name in names
    )


def _affects_composition(repo_root: Path, change: ChangedFile) -> bool:
    if _is_documentation_change(change) or "tests" in change.path.parts:
        return False
    ignored = _METADATA_KEYS | _RUNTIME_KEYS
    return not _structured_change_is_only(repo_root, change, ignored_keys=ignored)


def _dependency_identity_for_path(path: Path) -> tuple[str, str] | None:
    """Map one repository path to the narrow runtime identity that owns it."""

    if path.parts == _STOCK_ROLLOUT_DEPENDENCY:
        return _STOCK_ROLLOUT_DEPENDENCY
    if len(path.parts) >= 5 and path.parts[:3] == _SANDBOX_PROVIDER_PATH_PREFIX:
        return _SANDBOX_PROVIDER_DEPENDENCY_GROUP, path.parts[3]
    if len(path.parts) >= 2 and path.parts[0] in _DEPENDENCY_TREES:
        return path.parts[0], path.parts[1]
    return None


def _component_changes(
    repo_root: Path,
    changes: Sequence[ChangedFile],
) -> dict[tuple[str, str], tuple[bool, tuple[str, ...]]]:
    components: dict[tuple[str, str], tuple[bool, list[str]]] = {}
    for change in changes:
        for path in change.paths():
            identity = _dependency_identity_for_path(path)
            if identity is None:
                continue
            if _is_documentation_change(change) or is_metadata_only_change(repo_root, change):
                continue
            current_affects, paths = components.setdefault(identity, (False, []))
            paths.append(path.as_posix())
            components[identity] = (current_affects or _affects_composition(repo_root, change), paths)
    return {identity: (affects, tuple(sorted(set(paths)))) for identity, (affects, paths) in components.items()}


def _resolve_config(config_path: Path, repo_root: Path) -> DictConfig:
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        OmegaConf.create({"config_paths": [str(config_path.resolve())]}),
    )
    with _working_directory(repo_root):
        return _parse_no_environment_tolerating_unset_values(initial)


def _validate_manifest_location(repo_root: Path, path: Path, manifest: EnvironmentManifest) -> None:
    try:
        relative = path.parent.resolve().relative_to(repo_root.resolve())
    except ValueError as error:
        raise ManifestError(f"Manifest '{path}' is outside repository '{repo_root}'.") from error
    if not relative.parts or relative.parts[0] not in _REGISTRY_KINDS:
        raise ManifestError(f"Manifest '{path}' is not in a supported environment registry tree.")
    expected_kind = _REGISTRY_KINDS[relative.parts[0]]
    expected_name = Path(*relative.parts[1:]).as_posix()
    if manifest.kind.value != expected_kind:
        raise ManifestError(
            f"Manifest '{path}' kind={manifest.kind.value!r} does not match registry kind={expected_kind!r}."
        )
    if manifest.name != expected_name:
        raise ManifestError(
            f"Manifest '{path}' name={manifest.name!r} does not match registry identity={expected_name!r}."
        )


def _declared_component_dependencies(manifest: EnvironmentManifest) -> set[tuple[str, str]]:
    dependencies: set[tuple[str, str]] = set()
    if manifest.resources_server:
        dependencies.add(("resources_servers", manifest.resources_server))
    if manifest.agent_server:
        dependencies.add(("responses_api_agents", manifest.agent_server))
    if manifest.model_server:
        dependencies.add(("responses_api_models", manifest.model_server))
    return dependencies


def _directory_dependency(repo_root: Path | None, directory: Path | None) -> tuple[str, str] | None:
    if repo_root is None or directory is None:
        return None
    try:
        relative = directory.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None
    if len(relative.parts) < 2 or relative.parts[0] not in _DEPENDENCY_TREES:
        return None
    return relative.parts[0], relative.parts[1]


def _inspected_component_dependencies(
    inspection: WorkloadInspection,
    repo_root: Path | None,
    resolved_config: Mapping[str, Any] | DictConfig | None,
) -> set[tuple[str, str]]:
    dependencies: set[tuple[str, str]] = set()
    group_by_role = {role: group for group, role in _COMPONENT_ROLES.items()}
    provenance = {
        (item.role, item.instance, item.implementation): item
        for item in resolve_component_provenance(resolved_config or {})
    }
    for component in inspection.components:
        group = group_by_role.get(component.role)
        if group is None:
            continue
        dependencies.add((group, component.implementation))
        component_provenance = provenance.get((component.role, component.instance, component.implementation))
        source_directory = component_provenance.source_directory if component_provenance is not None else None
        if source_directory is None and repo_root is not None:
            source_directory, _config_path = resolve_component_source_directory(
                group,
                component.implementation,
                inspection.config_paths,
            )
        if identity := _directory_dependency(repo_root, source_directory):
            dependencies.add(identity)
        if component_provenance is not None:
            dependencies.update(
                identity
                for directory in component_provenance.dependency_directories
                if (identity := _directory_dependency(repo_root, directory)) is not None
            )
    return dependencies


def _auxiliary_runtime_dependencies(
    repo_root: Path | None,
    resolved_config: Mapping[str, Any] | DictConfig,
) -> set[tuple[str, str]]:
    directories: list[Path] = []
    for preparation in resolve_dataset_preparation_provenance(resolved_config):
        if preparation.source_directory is not None:
            directories.append(preparation.source_directory)
        directories.extend(preparation.dependency_directories)
    driver = resolve_rollout_driver_provenance(resolved_config)
    if driver is not None:
        if driver.source_directory is not None:
            directories.append(driver.source_directory)
        directories.extend(driver.dependency_directories)
    if repo_root is not None:
        component_bases = {
            component.instance: component.source_directory
            for component in resolve_component_provenance(resolved_config)
            if component.source_directory is not None
        }
        for reference in resolve_runtime_local_references(
            resolved_config,
            repo_root=repo_root,
            allow_external=True,
            base_directories=component_bases,
        ):
            if reference.path is not None:
                directories.append(reference.path if reference.path.is_dir() else reference.path.parent)
    return {
        identity for directory in directories if (identity := _directory_dependency(repo_root, directory)) is not None
    }


def _selected_sandbox_provider_dependencies(
    resolved_config: Mapping[str, Any] | DictConfig,
) -> set[tuple[str, str]]:
    """Return built-in provider-package identities selected by launched servers."""

    dependencies: set[tuple[str, str]] = set()
    group_by_role = {role: group for group, role in _COMPONENT_ROLES.items()}
    for component in resolve_component_provenance(resolved_config):
        group = group_by_role.get(component.role)
        instance_config = resolved_config.get(component.instance)
        if group is None or not isinstance(instance_config, Mapping):
            continue
        implementations = instance_config.get(group)
        if not isinstance(implementations, Mapping):
            continue
        server_config = implementations.get(component.implementation)
        if not isinstance(server_config, Mapping):
            continue
        raw_provider = server_config.get("sandbox_provider")
        if raw_provider is None:
            continue
        try:
            provider_config = resolve_provider_config(raw_provider, resolved_config)
        except (TypeError, ValueError) as error:
            consumer = f"{component.role}:{component.instance}/{component.implementation}"
            raise ConfigError(f"{consumer} has an invalid sandbox_provider: {error}") from error
        dependencies.add((_SANDBOX_PROVIDER_DEPENDENCY_GROUP, next(iter(provider_config))))
    return dependencies


def _uses_stock_rollout_loop(inspection: WorkloadInspection) -> bool:
    return any(
        owner.role == "rollout_driver" and owner.implementation == "nemo_gym.rollout_collection"
        for owner in inspection.responsibilities.rollout_coordination
    )


def _dependencies(
    manifest: EnvironmentManifest,
    inspection: WorkloadInspection | None,
    repo_root: Path | None = None,
    resolved_config: Mapping[str, Any] | DictConfig | None = None,
) -> set[tuple[str, str]]:
    dependencies = _declared_component_dependencies(manifest)
    if inspection is not None:
        dependencies.update(_inspected_component_dependencies(inspection, repo_root, resolved_config))
        if _uses_stock_rollout_loop(inspection):
            dependencies.add(_STOCK_ROLLOUT_DEPENDENCY)
    if resolved_config is not None:
        dependencies.update(_auxiliary_runtime_dependencies(repo_root, resolved_config))
        dependencies.update(_selected_sandbox_provider_dependencies(resolved_config))
    return dependencies


def _selected_resources_server_directory(
    repo_root: Path,
    implementation: str,
    config_paths: Sequence[str],
) -> Path:
    """Resolve an implementation through the exact config inputs of a workload."""

    directory, _config_path = resolve_component_source_directory(
        "resources_servers",
        implementation,
        config_paths,
    )
    if directory is None:
        raise ConfigError(
            f"Resources server implementation {implementation!r} must resolve to exactly one local component "
            "for its verifier CI check; found: none."
        )
    try:
        directory.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ConfigError(
            f"Resources server implementation {implementation!r} resolves outside CI repository "
            f"'{repo_root}': {directory}."
        ) from error
    return directory


def _canonical_verifier_entrypoint(component: Path, server_config: Mapping[str, Any]) -> Path:
    """Resolve the selected Python entrypoint without permitting path redirection."""

    raw_entrypoint = server_config.get("entrypoint")
    if not isinstance(raw_entrypoint, str) or not raw_entrypoint:
        raise ConfigError(f"Selected verifier in '{component}' has no Python entrypoint.")
    relative = Path(raw_entrypoint)
    if relative.is_absolute() or ".." in relative.parts:
        raise ConfigError(f"Selected verifier entrypoint must be component-relative, got {raw_entrypoint!r}.")
    entrypoint = component / relative
    cursor = component
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ConfigError(
                f"Verifier CI entrypoint '{cursor}' is a symbolic link; keep executable source inside the component."
            )
    try:
        entrypoint.resolve(strict=True).relative_to(component.resolve())
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigError(f"Verifier CI entrypoint '{entrypoint}' does not stay inside its component tree.") from error
    if not entrypoint.is_file() or entrypoint.suffix != ".py":
        raise ConfigError(f"Verifier CI entrypoint '{entrypoint}' must be a Python file.")
    return entrypoint


def _canonical_verifier_paths(repo_root: Path, component: Path) -> tuple[Path, Path]:
    """Resolve the fixed offline fixture and pytest node without path discovery.

    CI rejects symlinked registry/test paths and requires the component to be one
    direct child of the repository registry, so a config cannot redirect pytest
    outside the selected component tree.
    """

    registry = repo_root / "resources_servers"
    fixture_path = component / _VERIFIER_FIXTURE_RELATIVE_PATH
    test_path = component / _VERIFIER_TEST_RELATIVE_PATH
    for candidate in (registry, component, component / "tests", fixture_path, test_path):
        if candidate.is_symlink():
            raise ConfigError(
                f"Verifier CI path '{candidate}' is a symbolic link; keep the fixture and scorer test "
                "inside the selected resources-server tree."
            )
    try:
        relative_component = component.resolve().relative_to(registry.resolve())
        if len(relative_component.parts) != 1:
            raise ValueError("component is not a direct registry child")
        fixture_path.resolve().relative_to(component.resolve())
        test_path.resolve().relative_to(component.resolve())
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigError(
            f"Verifier CI paths for component '{component}' do not stay inside its registry tree."
        ) from error
    return fixture_path, test_path


def _validate_manifest_entry(
    entry: EnvironmentCatalogEntry,
    repo_root: Path,
) -> GateUnitResult:
    result = GateUnitResult(
        name=entry.name,
        kind=entry.kind,
        source=entry.source,
        config_path=entry.config_path,
        manifest_path=entry.manifest_path,
    )
    if entry.manifest_path is None:
        result.warnings.append("No manifest; this legacy runnable unit remains grandfathered in report-only mode.")
        if entry.config_path is not None:
            try:
                resolved = _resolve_config(entry.config_path, repo_root)
                result.resolved_config = resolved
                with _working_directory(repo_root):
                    inspection = inspect_workload(resolved, strict_missing_datasets=False)
                result.composition_hash = inspection.composition_hash
                result.dependencies = {
                    *_inspected_component_dependencies(inspection, repo_root, resolved),
                    *_auxiliary_runtime_dependencies(repo_root, resolved),
                    *_selected_sandbox_provider_dependencies(resolved),
                }
            except Exception as error:
                result.errors.append(f"Legacy static config inspection failed: {type(error).__name__}: {error}")
        return result

    try:
        manifest = load_manifest(entry.manifest_path)
        _validate_manifest_location(repo_root, entry.manifest_path, manifest)
    except Exception as error:
        result.errors.append(str(error))
        return result
    result.version_key = environment_version_key(manifest)
    result.dependencies = _dependencies(manifest, None)

    config_path = entry.config_path or manifest_config_path(entry.manifest_path)
    result.config_path = config_path
    if config_path is None or not config_path.is_file():
        result.errors.append("Manifest does not resolve to one unambiguous runnable config.")
        return result
    try:
        resolved = _resolve_config(config_path, repo_root)
        result.resolved_config = resolved
        with _working_directory(repo_root):
            inspection = inspect_workload(
                resolved,
                strict_missing_datasets=True,
                standard_prompt_config=manifest.standard_prompt_config,
                manifest=manifest,
            )
        result.composition_hash = inspection.composition_hash
        result.dependencies = _dependencies(manifest, inspection, repo_root, resolved)
        if inspection.profile != manifest.integration_profile.value:
            result.errors.append(
                f"integration_profile={manifest.integration_profile.value!r}, but config classifies as "
                f"{inspection.profile!r}."
            )
        deltas = manifest_composition_deltas(manifest, resolve_composition_mirror(resolved))
        if deltas:
            result.errors.append("Manifest/config composition mirror differs: " + "; ".join(deltas))
        result.warnings.extend(inspection.warnings)
    except Exception as error:
        result.errors.append(f"Static validation failed: {type(error).__name__}: {error}")
        return result

    try:
        if manifest.resources_server is None:  # guarded by the manifest profile contract
            raise ConfigError(f"Manifest '{manifest.name}' does not select a resources server.")
        component_dir = _selected_resources_server_directory(
            repo_root,
            manifest.resources_server,
            inspection.config_paths,
        )
        verifier_instance_name, verifier_server_config = select_resources_server_runtime(
            resolved,
            manifest.resources_server,
        )
        entrypoint_path = _canonical_verifier_entrypoint(component_dir, verifier_server_config)
        fixture_path, test_path = _canonical_verifier_paths(repo_root, component_dir)
    except ConfigError as error:
        result.errors.append(str(error))
        return result

    result.verifier_component_dir = component_dir
    result.verifier_entrypoint_path = entrypoint_path
    result.verifier_instance_name = verifier_instance_name
    result.verifier_server_config = verifier_server_config

    if not fixture_path.is_file():
        result.errors.append(
            "No standard resources-server verifier fixture was found. Add the three scoring sentinels "
            f"(and a re-seed sentinel when determinism is seeded) at {fixture_path}."
        )
    else:
        try:
            cases = load_verifier_fixture(fixture_path)
            validate_verifier_fixture(
                cases,
                reward_range=manifest.reward.range,
                higher_is_better=manifest.reward.higher_is_better,
                determinism=manifest.determinism.value,
            )
        except VerifierFixtureError as error:
            result.errors.append(f"Verifier fixture '{fixture_path}' failed its scoring contract: {error}")
    if not test_path.is_file():
        result.errors.append(
            f"No canonical offline verifier scorer test was found at '{test_path}'. Add "
            f"{_VERIFIER_TEST_RELATIVE_PATH.as_posix()}::{_COMPONENT_VERIFIER_TEST_NODE} "
            "for the local author workflow. CI independently runs Gym's repository-owned harness."
        )
    else:
        result.verifier_test_path = test_path
    return result


def _entry_root(entry: EnvironmentCatalogEntry, repo_root: Path) -> Path:
    if entry.manifest_path is not None:
        return entry.manifest_path.parent.resolve()
    if entry.kind == "environment" and entry.config_path is not None:
        try:
            relative = entry.config_path.resolve().relative_to(repo_root.resolve())
        except ValueError:
            relative = None
        if relative is not None and len(relative.parts) >= 2 and relative.parts[0] == "resources_servers":
            return (repo_root / "resources_servers" / relative.parts[1]).resolve()
        return entry.config_path.parent.resolve()
    return (repo_root / "benchmarks" / entry.name.split("/", 1)[0]).resolve()


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _schema_sync_errors(repo_root: Path) -> tuple[str, ...]:
    contracts = (
        (
            "manifest",
            repo_root / "schemas" / "environment-manifest.schema.json",
            manifest_json_schema(),
            "scripts/generate_environment_manifest_schema.py",
        ),
    )
    errors: list[str] = []
    for name, path, schema, generator in contracts:
        expected = json.dumps(schema, indent=2, sort_keys=True) + "\n"
        try:
            actual = path.read_text(encoding="utf-8")
        except OSError as error:
            errors.append(f"Could not read generated {name} schema '{path}': {error}.")
            continue
        if actual != expected:
            errors.append(f"Generated {name} schema is stale; run `python {generator}` and commit the result.")
    return tuple(errors)


def _lock_edit_violations(
    changes: Sequence[ChangedFile],
    current_locks: Mapping[str, Any],
) -> tuple[str, ...]:
    """Reject edits/removals of lock records that existed at the PR base."""

    change = next((item for item in changes if item.path == LOCK_RELATIVE_PATH), None)
    if change is None or change.before_content is None:
        return ()
    try:
        before = json.loads(change.before_content)
        before_records = before["environments"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return (f"Could not parse the base version-lock document for '{LOCK_RELATIVE_PATH}'.",)
    if not isinstance(before_records, Mapping):
        return (f"Base version-lock document at '{LOCK_RELATIVE_PATH}' has no environments object.",)

    current_records = current_locks["environments"]
    violations: list[str] = []
    for key, record in sorted(before_records.items()):
        if key not in current_records:
            violations.append(f"{key}: published composition lock was removed")
        elif current_records[key] != record:
            violations.append(f"{key}: published composition lock is immutable and was edited")
    return tuple(violations)


def _lock_violations(
    repo_root: Path,
    units: Sequence[GateUnitResult],
    changes: Sequence[ChangedFile],
) -> tuple[str, ...]:
    try:
        locks = load_version_locks(repo_root / LOCK_RELATIVE_PATH)
    except ConfigError as error:
        return (str(error),)
    hashes = {
        unit.version_key: unit.composition_hash
        for unit in units
        if unit.version_key is not None and unit.composition_hash is not None
    }
    units_by_version = {
        unit.version_key: unit
        for unit in units
        if unit.version_key is not None and unit.composition_hash is not None and unit.manifest_path is not None
    }
    lock_change = next((item for item in changes if item.path == LOCK_RELATIVE_PATH), None)
    base_records: Mapping[str, Any] | None = None
    if lock_change is not None:
        if lock_change.before_content is None:
            base_records = {}
        else:
            try:
                parsed_base = json.loads(lock_change.before_content)
                candidate_records = parsed_base["environments"]
                if isinstance(candidate_records, Mapping):
                    base_records = candidate_records
            except (json.JSONDecodeError, KeyError, TypeError):
                # _lock_edit_violations reports the malformed base below.
                base_records = {}

    violations: list[str] = []
    # A path whose manifest now declares a different version represents an
    # intentional bump. Keep locks that already existed at the PR base as
    # historical while validating the current version independently. A newly
    # added unmatched lock is never historical: allowing it would let a PR squat
    # an arbitrary future version forever.
    for key, record in locks["environments"].items():
        current_unit = units_by_version.get(key)
        if current_unit is not None:
            assert current_unit.manifest_path is not None
            expected_manifest = current_unit.manifest_path.resolve().relative_to(repo_root.resolve()).as_posix()
            if record.get("manifest") != expected_manifest:
                violations.append(
                    f"{key}: composition lock manifest must be '{expected_manifest}', got {record.get('manifest')!r}"
                )
            continue
        if base_records is None or key in base_records:
            hashes[key] = str(record["composition_hash"])
        else:
            hashes[key] = str(record["composition_hash"])
            violations.append(
                f"{key}: newly added composition lock does not match any currently resolved exact "
                "environment version; publish the declared version instead"
            )
    try:
        violations.extend(
            [
                *validate_version_locks(repo_root=repo_root, current_hashes=hashes),
                *_lock_edit_violations(changes, locks),
            ]
        )
    except ConfigError as error:
        return (str(error),)
    locked_keys = locks["environments"]
    for unit in units:
        if (
            unit.enforced
            and unit.manifest_path is not None
            and unit.version_key is not None
            and unit.composition_hash is not None
            and unit.version_key not in locked_keys
        ):
            violations.append(
                f"{unit.version_key}: no published composition lock; run `gym env publish {unit.name} "
                f"--owner <github-handle>` and commit {LOCK_RELATIVE_PATH}"
            )
    return tuple(violations)


def _add_catalog_issues(
    catalog: EnvironmentCatalog,
    units: list[GateUnitResult],
    repo_root: Path,
) -> None:
    known_paths = {unit.manifest_path.resolve() for unit in units if unit.manifest_path is not None}
    for issue in catalog.issues:
        if issue.code != "invalid-manifest":
            continue
        if issue.path.resolve() in known_paths:
            continue
        dependencies: set[tuple[str, str]] = set()
        try:
            raw_manifest = yaml.safe_load(issue.path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            raw_manifest = None
        if isinstance(raw_manifest, Mapping):
            for group, field_name in _COMPONENT_ROLES.items():
                component = raw_manifest.get(field_name)
                if isinstance(component, str) and component:
                    dependencies.add((group, component))
        try:
            relative = issue.path.parent.resolve().relative_to(repo_root.resolve())
            tree = relative.parts[0]
            name = Path(*relative.parts[1:]).as_posix()
            kind = _REGISTRY_KINDS.get(tree, "environment")
        except (ValueError, IndexError):
            name, kind = issue.path.parent.name, "environment"
        units.append(
            GateUnitResult(
                name=name,
                kind=kind,
                source="manifest",
                config_path=manifest_config_path(issue.path),
                manifest_path=issue.path,
                errors=[issue.message],
                dependencies=dependencies,
            )
        )


def _load_gate_units(
    root: Path,
    catalog: EnvironmentCatalog | None,
) -> tuple[EnvironmentCatalog, list[GateUnitResult]]:
    with _working_directory(root):
        discovered = catalog or discover_environment_catalog(include_unpublished=True)
        units = [_validate_manifest_entry(entry, root) for entry in discovered.entries]
    _add_catalog_issues(discovered, units, root)
    return discovered, units


def _gate_unit_root(root: Path, unit: GateUnitResult) -> Path:
    return _entry_root(
        EnvironmentCatalogEntry(
            name=unit.name,
            kind=unit.kind,
            status="experimental" if unit.manifest_path else "no-manifest",
            source="manifest" if unit.manifest_path else "legacy",
            config_path=unit.config_path,
            manifest_path=unit.manifest_path,
        ),
        root,
    )


def _mark_changed_units(
    root: Path,
    changes: Sequence[ChangedFile],
    units: Sequence[GateUnitResult],
    *,
    enforce_changes: bool,
) -> set[Path]:
    matched_paths: set[Path] = set()
    unit_roots = tuple((unit, _gate_unit_root(root, unit)) for unit in units)
    for change in changes:
        if _is_documentation_change(change):
            continue
        for unit, unit_root in unit_roots:
            if not any(_under(root / path, unit_root) for path in change.paths()):
                continue
            unit.enforced = enforce_changes
            unit.reasons.add(f"changed-unit:{change.path.as_posix()}")
            unit.composition_change = unit.composition_change or _affects_composition(root, change)
            matched_paths.update(change.paths())
    return matched_paths


def _mark_component_dependents(
    root: Path,
    changes: Sequence[ChangedFile],
    units: Sequence[GateUnitResult],
    *,
    enforce_changes: bool,
) -> None:
    for identity, (affects_composition, paths) in _component_changes(root, changes).items():
        identity_root = (root / identity[0] / identity[1]).resolve()
        for unit in units:
            if identity not in unit.dependencies:
                continue
            # A runnable unit often records its own recipe directory as a
            # dependency (dataset preparation, prompts, or a custom driver).
            # That is a direct unit change, not downstream fan-out. Keeping the
            # two reasons distinct prevents a legacy unit from exempting itself
            # from the migration requirement.
            if _gate_unit_root(root, unit).resolve() == identity_root:
                continue
            unit.enforced = enforce_changes
            unit.reasons.add(f"dependent:{identity[0]}/{identity[1]} ({', '.join(paths)})")
            unit.composition_change = unit.composition_change or affects_composition


def _unresolved_registry_target(
    root: Path,
    path: Path,
    units: Sequence[GateUnitResult],
) -> tuple[str, str, Path, Path] | None:
    if len(path.parts) < 2 or path.parts[0] not in _REGISTRY_KINDS:
        return None
    registry = path.parts[0]
    kind = _REGISTRY_KINDS[registry]
    unit_root = root / registry / path.parts[1]
    manifest_path = unit_root / "manifest.yaml"
    if registry != "resources_servers":
        return kind, path.parts[1], unit_root / "config.yaml", manifest_path

    is_config = len(path.parts) >= 4 and path.parts[2] == "configs" and path.suffix.casefold() in {".yaml", ".yml"}
    is_manifest = len(path.parts) == 3 and path.name == "manifest.yaml"
    if not (is_config or is_manifest):
        return None
    component_identity = ("resources_servers", path.parts[1])
    if any(component_identity in unit.dependencies for unit in units):
        return None
    name = f"resources_servers/{path.parts[1]}/{path.stem}" if is_config else path.parts[1]
    config_path = root / path if is_config else manifest_config_path(manifest_path)
    return kind, name, config_path, manifest_path


def _add_unresolved_changed_units(
    root: Path,
    changes: Sequence[ChangedFile],
    units: list[GateUnitResult],
    matched_paths: set[Path],
    *,
    enforce_changes: bool,
) -> None:
    by_key = {unit.key: unit for unit in units}
    for change in changes:
        if _is_documentation_change(change) or is_metadata_only_change(root, change) or change.path in matched_paths:
            continue
        target = _unresolved_registry_target(root, change.path, units)
        if target is None:
            continue
        kind, name, config_path, manifest_path = target
        unit = by_key.get((kind, name))
        if unit is None:
            unit = GateUnitResult(
                name=name,
                kind=kind,
                source="unknown",
                config_path=config_path,
                manifest_path=manifest_path if manifest_path.exists() else None,
                errors=["Changed runnable unit does not resolve to a valid manifest and config."],
            )
            units.append(unit)
            by_key[unit.key] = unit
        unit.enforced = enforce_changes
        unit.reasons.add(f"changed-unit:{change.path.as_posix()}")
        unit.composition_change = unit.composition_change or _affects_composition(root, change)


def _bootstrap_inventory_legacy_units(
    root: Path,
    changes: Sequence[ChangedFile],
) -> frozenset[tuple[str, str, Path]]:
    """Validate the one-PR inventory bootstrap and return base-existing legacy units.

    The exemption can only exist while the canonical inventory itself is a new
    Git path. Once it is present in the base, edits or delete/re-add attempts do
    not satisfy this predicate.
    """

    inventory_references = [change for change in changes if MIGRATION_INVENTORY_PATH in change.paths()]
    if len(inventory_references) != 1:
        return frozenset()
    inventory_change = inventory_references[0]
    if (
        inventory_change.path != MIGRATION_INVENTORY_PATH
        or inventory_change.status != "A"
        or inventory_change.old_path is not None
        or inventory_change.before_content is not None
    ):
        return frozenset()

    inventory_path = root / MIGRATION_INVENTORY_PATH
    if inventory_path.is_symlink() or not inventory_path.is_file():
        return frozenset()
    try:
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return frozenset()
    if not isinstance(payload, Mapping) or set(payload) != {"schema_version", "summary", "units", "exceptions"}:
        return frozenset()
    if payload.get("schema_version") != 1:
        return frozenset()
    raw_records = payload.get("units")
    raw_exceptions = payload.get("exceptions")
    if not isinstance(raw_records, list) or not isinstance(raw_exceptions, list):
        return frozenset()

    expected: dict[tuple[str, str, str, str, str], Any] = {}
    try:
        for unit in discover_runnable_units(root):
            config_path = unit.config_path.resolve().relative_to(root).as_posix()
            manifest_path = unit.manifest_path.resolve().relative_to(root).as_posix()
            key = (unit.kind, unit.name, unit.registry, config_path, manifest_path)
            expected[key] = unit
    except (OSError, ValueError):
        return frozenset()

    def record_key(record: object) -> tuple[str, str, str, str, str] | None:
        if not isinstance(record, Mapping):
            return None
        fields = tuple(record.get(field) for field in ("kind", "name", "registry", "config_path", "manifest_path"))
        if not all(isinstance(value, str) and value for value in fields):
            return None
        kind, name, registry, config_path, manifest_path = fields
        if Path(config_path).is_absolute() or Path(manifest_path).is_absolute():
            return None
        return kind, name, registry, config_path, manifest_path

    records: dict[tuple[str, str, str, str, str], Mapping[str, Any]] = {}
    for raw_record in raw_records:
        key = record_key(raw_record)
        if key is None or key not in expected or key in records:
            return frozenset()
        assert isinstance(raw_record, Mapping)
        status = raw_record.get("status")
        if status not in _INVENTORY_UNIT_STATUSES:
            return frozenset()
        unit = expected[key]
        manifest_exists = unit.manifest_path.is_file()
        if manifest_exists:
            expected_status = "drafted" if is_generated_migration_draft(unit.manifest_path) else "already-manifest"
            if status != expected_status:
                return frozenset()
        elif status not in _BOOTSTRAP_LEGACY_STATUSES:
            return frozenset()
        if unit.blocker is not None and (status != "exception" or raw_record.get("reason") != unit.blocker):
            return frozenset()
        if status == "exception" and not isinstance(raw_record.get("reason"), str):
            return frozenset()
        records[key] = raw_record
    if records.keys() != expected.keys():
        return frozenset()

    exception_records = {
        key: record["reason"] for key, record in records.items() if record.get("status") == "exception"
    }
    listed_exceptions: dict[tuple[str, str, str, str, str], str] = {}
    for raw_exception in raw_exceptions:
        key = record_key(raw_exception)
        if key is None or key in listed_exceptions or not isinstance(raw_exception, Mapping):
            return frozenset()
        reason = raw_exception.get("reason")
        if not isinstance(reason, str):
            return frozenset()
        listed_exceptions[key] = reason
    if listed_exceptions != exception_records:
        return frozenset()

    statuses = [record["status"] for record in records.values()]
    expected_summary = {
        "total": len(records),
        "drafted": statuses.count("drafted"),
        "planned": statuses.count("planned"),
        "already_manifest": statuses.count("already-manifest"),
        "exceptions": statuses.count("exception"),
    }
    if payload.get("summary") != expected_summary:
        return frozenset()

    bootstrap_units: set[tuple[str, str, Path]] = set()
    for key, record in records.items():
        if record["status"] not in _BOOTSTRAP_LEGACY_STATUSES:
            continue
        unit = expected[key]
        lexical_config_path = Path(os.path.abspath(unit.lexical_config_path or unit.config_path))
        try:
            config_relative = lexical_config_path.relative_to(root)
        except ValueError:
            return frozenset()
        cursor = root
        has_symlink = False
        for part in config_relative.parts:
            cursor /= part
            if cursor.is_symlink():
                has_symlink = True
                break
        if has_symlink:
            continue
        config_changes = [change for change in changes if change.path == config_relative]
        if len(config_changes) > 1:
            return frozenset()
        if config_changes:
            config_change = config_changes[0]
            if (
                not config_change.status.startswith("M")
                or config_change.old_path is not None
                or config_change.before_content is None
            ):
                continue
            if not is_runnable_unit_config_content(unit.registry, config_change.before_content):
                continue
        manifest_relative = unit.manifest_path.resolve().relative_to(root)
        if any(manifest_relative in change.paths() for change in changes):
            continue
        bootstrap_units.add((unit.kind, unit.name, unit.config_path.resolve()))
    return frozenset(bootstrap_units)


def _require_manifests_for_changed_units(
    units: Sequence[GateUnitResult],
    *,
    bootstrap_legacy_units: frozenset[tuple[str, str, Path]] = frozenset(),
) -> None:
    for unit in units:
        is_direct_change = any(reason.startswith("changed-unit:") for reason in unit.reasons)
        if unit.enforced and unit.manifest_path is None and is_direct_change:
            bootstrap_key = (
                unit.kind,
                unit.name,
                unit.config_path.resolve() if unit.config_path is not None else Path(),
            )
            if bootstrap_key in bootstrap_legacy_units:
                unit.bootstrap_legacy_exempt = True
                unit.warnings.append(
                    "One-time migration-inventory bootstrap: this base-existing legacy unit remains grandfathered."
                )
                continue
            unit.errors.append(
                "A new or changed runnable environment/benchmark requires a complete manifest.yaml; "
                "unchanged legacy units remain grandfathered."
            )


def run_environment_ci_gate(
    repo_root: Path,
    *,
    changes: Sequence[ChangedFile] = (),
    enforce_changes: bool = False,
    catalog: EnvironmentCatalog | None = None,
) -> EnvironmentGateReport:
    """Inspect the full catalog and enforce only changed/affected units."""

    root = repo_root.resolve()
    normalized_changes = tuple(changes)
    effective_changes = normalized_changes
    discovered, units = _load_gate_units(root, catalog)
    matched_paths = _mark_changed_units(root, effective_changes, units, enforce_changes=enforce_changes)
    _mark_component_dependents(root, effective_changes, units, enforce_changes=enforce_changes)
    _add_unresolved_changed_units(
        root,
        effective_changes,
        units,
        matched_paths,
        enforce_changes=enforce_changes,
    )
    bootstrap_legacy_units = _bootstrap_inventory_legacy_units(root, effective_changes)
    _require_manifests_for_changed_units(units, bootstrap_legacy_units=bootstrap_legacy_units)

    lock_violations = list(_lock_violations(root, units, normalized_changes))

    units.sort(key=lambda unit: (unit.name.casefold(), unit.kind, str(unit.manifest_path or unit.config_path)))
    return EnvironmentGateReport(
        repo_root=root,
        enforce_changes=enforce_changes,
        catalog=discovered,
        changes=normalized_changes,
        units=tuple(units),
        schema_errors=_schema_sync_errors(root),
        lock_violations=tuple(sorted(set(lock_violations))),
    )


def _bounded_process_output(stdout: object, stderr: object) -> str:
    def render(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return str(value)

    output = "\n".join(part.strip() for part in (render(stdout), render(stderr)) if part).strip()
    if len(output) > _VERIFIER_OUTPUT_LIMIT:
        return "... output truncated ...\n" + output[-_VERIFIER_OUTPUT_LIMIT:]
    return output


def _run_component_setup_command(
    command: Sequence[str],
    *,
    component_dir: Path,
    timeout_seconds: float,
) -> None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=component_dir,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        output = _bounded_process_output(error.stdout, error.stderr)
        detail = f"\n{output}" if output else ""
        raise ConfigError(
            f"Timed out after {timeout_seconds:g} seconds while preparing verifier dependencies.{detail}"
        ) from error
    except OSError as error:
        raise ConfigError(f"Could not prepare verifier dependency environment: {error}.") from error
    if completed.returncode != 0:
        output = _bounded_process_output(completed.stdout, completed.stderr)
        detail = f"\n{output}" if output else ""
        raise ConfigError(
            f"Verifier dependency setup command exited with {completed.returncode}: {' '.join(command)}{detail}"
        )


def _prepare_component_verifier_environment(
    component_dir: Path,
    *,
    timeout_seconds: float,
) -> Path:
    """Create the standard component venv and install its declared dependencies."""

    requirements_path = component_dir / "requirements.txt"
    pyproject_path = component_dir / "pyproject.toml"
    if requirements_path.is_file() == pyproject_path.is_file():
        expected = "exactly one of requirements.txt or pyproject.toml"
        raise ConfigError(f"Verifier component '{component_dir}' must contain {expected}.")

    venv_dir = component_dir / ".venv"
    python_path = venv_dir / "bin" / "python"
    _run_component_setup_command(
        (
            "uv",
            "venv",
            "--seed",
            "--allow-existing",
            "--python",
            sys.executable,
            str(venv_dir),
        ),
        component_dir=component_dir,
        timeout_seconds=timeout_seconds,
    )

    install_command = ["uv", "pip", "install", "--python", str(python_path)]
    if requirements_path.is_file():
        overrides_path = component_dir / "overrides.txt"
        if overrides_path.is_file():
            install_command.extend(("--override", str(overrides_path)))
        install_command.extend(("-r", str(requirements_path)))
    else:
        install_command.extend(("-e", "."))
    _run_component_setup_command(
        install_command,
        component_dir=component_dir,
        timeout_seconds=timeout_seconds,
    )
    if not python_path.is_file():
        raise ConfigError(f"Verifier component environment did not create Python at '{python_path}'.")
    return python_path


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _checked_file_digest(path: Path, *, label: str) -> str:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ConfigError(f"Could not read {label} '{path}': {error}.") from error
    return hashlib.sha256(payload).hexdigest()


def _resolved_config_yaml(unit: GateUnitResult) -> str:
    if unit.resolved_config is None:
        raise ConfigError(
            f"Legacy unit {unit.kind}:{unit.name} has no exact resolved config; its local check cannot be selected."
        )
    try:
        return OmegaConf.to_yaml(unit.resolved_config, resolve=True)
    except Exception as error:
        raise ConfigError(
            f"Could not serialize the exact resolved config for legacy unit {unit.kind}:{unit.name}: {error}."
        ) from error


def _resolved_config_paths(repo_root: Path, unit: GateUnitResult) -> tuple[str, ...]:
    assert unit.resolved_config is not None
    raw_paths = unit.resolved_config.get("config_paths") or ()
    if isinstance(raw_paths, (str, bytes, bytearray, Mapping)):
        raise ConfigError(f"Legacy unit {unit.kind}:{unit.name} has invalid resolved config_paths={raw_paths!r}.")
    try:
        paths = tuple(
            str((Path(path) if Path(path).is_absolute() else repo_root / str(path)).resolve()) for path in raw_paths
        )
    except TypeError as error:
        raise ConfigError(
            f"Legacy unit {unit.kind}:{unit.name} has invalid resolved config_paths={raw_paths!r}."
        ) from error
    if not paths and unit.config_path is not None:
        paths = (str(unit.config_path.resolve()),)
    if not paths:
        raise ConfigError(f"Legacy unit {unit.kind}:{unit.name} has no resolved config path.")
    return paths


def _verifier_plan_key(
    *,
    mode: Literal["fixture", "pytest"],
    component_dir: Path,
    entrypoint_path: Path,
    instance_name: str,
    server_config: Mapping[str, Any],
    resolved_config_yaml: str,
    check_path: Path,
    reward_range: tuple[int | float, int | float] | None,
    higher_is_better: bool,
    determinism: str | None,
) -> tuple[str, ...]:
    """Identify checks that are equivalent for every runtime-visible input."""

    return (
        mode,
        str(component_dir.resolve()),
        str(entrypoint_path.resolve()),
        _checked_file_digest(entrypoint_path, label="verifier entrypoint"),
        instance_name,
        _stable_json(server_config),
        resolved_config_yaml,
        str(check_path.resolve()),
        _checked_file_digest(check_path, label="verifier check input"),
        _stable_json(reward_range),
        str(higher_is_better),
        determinism or "",
    )


def _manifest_verifier_plan(repo_root: Path, unit: GateUnitResult) -> _VerifierCheckPlan:
    assert unit.manifest_path is not None
    manifest = load_manifest(unit.manifest_path)
    if manifest.resources_server is None:  # guarded by static validation
        raise ConfigError(f"Manifest '{manifest.name}' does not select a resources server.")
    if (
        unit.verifier_component_dir is None
        or unit.verifier_entrypoint_path is None
        or unit.verifier_instance_name is None
        or unit.verifier_server_config is None
    ):
        raise ConfigError(f"Manifest '{manifest.name}' has no prepared canonical verifier runtime.")
    component_dir = unit.verifier_component_dir
    fixture_path, _test_path = _canonical_verifier_paths(repo_root, component_dir)
    entrypoint_path = _canonical_verifier_entrypoint(component_dir, unit.verifier_server_config)
    if entrypoint_path.resolve() != unit.verifier_entrypoint_path.resolve():
        raise ConfigError(
            f"Verifier entrypoint for {manifest.name!r} changed after static validation; refusing to run."
        )
    resolved_yaml = OmegaConf.to_yaml(unit.resolved_config, resolve=True) if unit.resolved_config is not None else ""
    cache_key = _verifier_plan_key(
        mode="fixture",
        component_dir=component_dir,
        entrypoint_path=entrypoint_path,
        instance_name=unit.verifier_instance_name,
        server_config=unit.verifier_server_config,
        resolved_config_yaml=resolved_yaml,
        check_path=fixture_path,
        reward_range=manifest.reward.range,
        higher_is_better=manifest.reward.higher_is_better,
        determinism=manifest.determinism.value,
    )
    return _VerifierCheckPlan(
        mode="fixture",
        component_dir=component_dir,
        entrypoint_path=entrypoint_path,
        instance_name=unit.verifier_instance_name,
        server_config=unit.verifier_server_config,
        check_path=entrypoint_path,
        fixture_path=fixture_path,
        reward_range=manifest.reward.range,
        higher_is_better=manifest.reward.higher_is_better,
        determinism=manifest.determinism.value,
        cache_key=cache_key,
    )


def _legacy_verifier_plan(repo_root: Path, unit: GateUnitResult) -> _VerifierCheckPlan:
    """Prepare the strongest available local check for one dependency-only legacy unit."""

    if unit.resolved_config is None:
        raise ConfigError(
            f"Legacy unit {unit.kind}:{unit.name} did not resolve statically, so no local check can use its config."
        )
    implementation, instance_name, server_config = select_sole_resources_server_runtime(unit.resolved_config)
    with _working_directory(repo_root):
        component_dir = _selected_resources_server_directory(
            repo_root,
            implementation,
            _resolved_config_paths(repo_root, unit),
        )
    entrypoint_path = _canonical_verifier_entrypoint(component_dir, server_config)
    fixture_path, test_path = _canonical_verifier_paths(repo_root, component_dir)
    resolved_yaml = _resolved_config_yaml(unit)

    unit.verifier_component_dir = component_dir
    unit.verifier_entrypoint_path = entrypoint_path
    unit.verifier_instance_name = instance_name
    unit.verifier_server_config = server_config
    if test_path.is_file():
        unit.verifier_test_path = test_path

    if fixture_path.is_file():
        try:
            validate_verifier_fixture(load_verifier_fixture(fixture_path))
        except VerifierFixtureError as error:
            raise ConfigError(f"Legacy verifier fixture '{fixture_path}' is invalid: {error}") from error
        return _VerifierCheckPlan(
            mode="fixture",
            component_dir=component_dir,
            entrypoint_path=entrypoint_path,
            instance_name=instance_name,
            server_config=server_config,
            check_path=entrypoint_path,
            fixture_path=fixture_path,
            resolved_config_yaml=resolved_yaml,
            cache_key=_verifier_plan_key(
                mode="fixture",
                component_dir=component_dir,
                entrypoint_path=entrypoint_path,
                instance_name=instance_name,
                server_config=server_config,
                resolved_config_yaml=resolved_yaml,
                check_path=fixture_path,
                reward_range=None,
                higher_is_better=True,
                determinism=None,
            ),
        )

    if not test_path.is_file():
        raise ConfigError(
            f"Resources Server '{implementation}' has neither the canonical fixture '{fixture_path}' nor "
            f"the legacy local pytest check '{test_path}'."
        )
    return _VerifierCheckPlan(
        mode="pytest",
        component_dir=component_dir,
        entrypoint_path=entrypoint_path,
        instance_name=instance_name,
        server_config=server_config,
        check_path=test_path,
        resolved_config_yaml=resolved_yaml,
        cache_key=_verifier_plan_key(
            mode="pytest",
            component_dir=component_dir,
            entrypoint_path=entrypoint_path,
            instance_name=instance_name,
            server_config=server_config,
            resolved_config_yaml=resolved_yaml,
            check_path=test_path,
            reward_range=None,
            higher_is_better=True,
            determinism=None,
        ),
    )


def _legacy_pytest_environment(plan: _VerifierCheckPlan, repo_root: Path) -> dict[str, str]:
    if plan.resolved_config_yaml is None:  # pragma: no cover - plan construction guarantees this
        raise ConfigError("Legacy pytest check has no exact resolved config.")
    environment = dict(os.environ)
    for name in (
        REWARD_RANGE_ENV_VAR,
        HIGHER_IS_BETTER_ENV_VAR,
        DETERMINISM_ENV_VAR,
        NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    ):
        environment.pop(name, None)
    environment.update(verifier_fixture_environment(update_expected=False))
    environment[NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME] = plan.resolved_config_yaml
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    python_path_entries = [str(plan.component_dir), str(repo_root)]
    if existing_python_path := environment.get("PYTHONPATH"):
        python_path_entries.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path_entries)
    return environment


def _execute_verifier_plan(
    plan: _VerifierCheckPlan,
    *,
    executable: Path,
    repo_root: Path,
    timeout_seconds: float,
) -> VerifierCheckResult:
    immutable_fixture: bytes | None = None
    if plan.mode == "fixture":
        assert plan.fixture_path is not None
        immutable_fixture = plan.fixture_path.read_bytes()
        invocation = build_verifier_harness_invocation(
            python_executable=executable,
            project_root=repo_root,
            component_dir=plan.component_dir,
            entrypoint=plan.entrypoint_path,
            instance_name=plan.instance_name,
            fixture_path=plan.fixture_path,
            server_config=plan.server_config,
            reward_range=plan.reward_range,
            higher_is_better=plan.higher_is_better,
            determinism=plan.determinism,
        )
        command = invocation.command
        cwd = repo_root
        environment = invocation.environment
        stdin: str | None = invocation.stdin
        node = VERIFIER_HARNESS_MODULE
    else:
        command = (str(executable), "-m", "pytest", plan.check_path.relative_to(plan.component_dir).as_posix())
        cwd = plan.component_dir
        environment = _legacy_pytest_environment(plan, repo_root)
        stdin = None
        node = _LEGACY_PYTEST_NODE

    try:
        run_kwargs: dict[str, Any] = {
            "cwd": cwd,
            "env": environment,
            "check": False,
            "capture_output": True,
            "text": True,
            "errors": "replace",
            "timeout": timeout_seconds,
        }
        if stdin is not None:
            run_kwargs["input"] = stdin
        completed = subprocess.run(list(command), **run_kwargs)
        output = _bounded_process_output(completed.stdout, completed.stderr)
        returncode: int | None = completed.returncode
        passed = completed.returncode == 0
    except subprocess.TimeoutExpired as error:
        output = _bounded_process_output(error.stdout, error.stderr)
        returncode = None
        passed = False
        output = f"Timed out after {timeout_seconds:g} seconds." + (f"\n{output}" if output else "")
    except OSError as error:
        output = str(error)
        returncode = None
        passed = False

    if immutable_fixture is not None and plan.fixture_path is not None:
        try:
            fixture_unchanged = plan.fixture_path.read_bytes() == immutable_fixture
        except OSError as error:
            fixture_unchanged = False
            output = f"{output}\nCould not verify fixture remained unchanged: {error}".strip()
        if not fixture_unchanged:
            passed = False
            output = (f"{output}\nVerifier scorer check modified '{plan.fixture_path}' in read-only CI mode.").strip()

    return VerifierCheckResult(
        test_path=plan.check_path,
        passed=passed,
        returncode=returncode,
        output=output if not passed else "",
        node=node,
    )


def run_enforced_verifier_checks(
    report: EnvironmentGateReport,
    *,
    python_executable: str | Path | None = None,
    timeout_seconds: float = 300,
    dependency_timeout_seconds: float = 900,
) -> EnvironmentGateReport:
    """Execute a no-Ray local scorer check for every eligible enforced unit.

    The static gate decides which units are new, changed, or downstream of a
    shared component change. Manifest-backed units use Gym's repository-owned
    fixture harness. During migration, dependency-only legacy units select the
    sole Resources Server from their exact resolved config and prefer that same
    harness, falling back to the component's local ``tests/test_app.py`` when no
    canonical fixture exists. Directly changed legacy units remain blocked on a
    manifest and are never allowed through this fallback, except for the
    validated one-time inventory bootstrap; those units run the same
    exact-config local plan. No path starts Ray, a Gym head server, a resources
    service, an agent, or a model.
    """

    if not report.enforce_changes:
        return report
    if timeout_seconds <= 0 or dependency_timeout_seconds <= 0:
        raise ConfigError("Verifier scorer and dependency setup timeouts must be greater than zero seconds.")

    component_environments: dict[Path, Path] = {}
    execution_cache: dict[tuple[str, ...], VerifierCheckResult] = {}
    for unit in report.units:
        if not unit.enforced:
            continue

        is_legacy = unit.manifest_path is None
        is_direct_change = any(reason.startswith("changed-unit:") for reason in unit.reasons)
        is_dependency = any(reason.startswith("dependent:") for reason in unit.reasons)
        if is_legacy and is_direct_change and not unit.bootstrap_legacy_exempt:
            # Direct legacy changes already carry the mandatory-manifest error;
            # only the validated inventory bootstrap may use the local plan.
            continue
        if is_legacy and not is_dependency and not unit.bootstrap_legacy_exempt:
            # Unrelated report-only legacy inventory must never execute code.
            continue

        legacy_subject = (
            "Bootstrap-exempt legacy local check"
            if unit.bootstrap_legacy_exempt
            else "Dependency-only legacy local check"
        )

        try:
            plan = (
                _legacy_verifier_plan(report.repo_root, unit)
                if is_legacy
                else _manifest_verifier_plan(report.repo_root, unit)
            )
        except (ConfigError, ManifestError, OSError, VerifierFixtureError) as error:
            subject = legacy_subject if is_legacy else "Offline verifier scorer check"
            fallback_path = unit.verifier_entrypoint_path or unit.manifest_path or unit.config_path or report.repo_root
            unit.verifier_checks.append(
                VerifierCheckResult(
                    test_path=fallback_path,
                    passed=False,
                    returncode=None,
                    output=str(error),
                    node="legacy-local-check" if is_legacy else VERIFIER_HARNESS_MODULE,
                )
            )
            unit.errors.append(f"{subject} could not be prepared; the enforced unit was not checked: {error}")
            continue

        result = execution_cache.get(plan.cache_key)
        if result is None:
            try:
                if python_executable is None:
                    executable = component_environments.get(plan.component_dir)
                    if executable is None:
                        executable = _prepare_component_verifier_environment(
                            plan.component_dir,
                            timeout_seconds=dependency_timeout_seconds,
                        )
                        component_environments[plan.component_dir] = executable
                else:
                    executable = Path(python_executable)
                result = _execute_verifier_plan(
                    plan,
                    executable=executable,
                    repo_root=report.repo_root,
                    timeout_seconds=timeout_seconds,
                )
            except (ConfigError, OSError, VerifierFixtureError) as error:
                result = VerifierCheckResult(
                    test_path=plan.check_path,
                    passed=False,
                    returncode=None,
                    output=str(error),
                    node=VERIFIER_HARNESS_MODULE if plan.mode == "fixture" else _LEGACY_PYTEST_NODE,
                )
            execution_cache[plan.cache_key] = result

        # Each dependent receives its own immutable result record even when an
        # equivalent exact-config check was safely deduplicated above.
        unit.verifier_checks.append(result)
        if not result.passed:
            status = f"exit {result.returncode}" if result.returncode is not None else "no exit status"
            subject = legacy_subject if is_legacy else "Offline verifier scorer check"
            unit.errors.append(f"{subject} failed for '{plan.check_path}' via {result.node} ({status}).")

    return report


__all__ = [
    "ChangedFile",
    "EnvironmentGateReport",
    "GateUnitResult",
    "VerifierCheckResult",
    "changed_files_from_git",
    "is_metadata_only_change",
    "run_enforced_verifier_checks",
    "run_environment_ci_gate",
]
