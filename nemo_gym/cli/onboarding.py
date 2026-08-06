# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI adapters for the environment-onboarding contract and catalog."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import rich
from omegaconf import DictConfig
from pydantic import Field
from rich.table import Table

import nemo_gym.environment_execution as environment_execution
from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.agent_registry import discover_agents
from nemo_gym.cli.utils import exit_cleanly_on_config_error, print_rich_table
from nemo_gym.config_types import BaseNeMoGymCLIConfig, ConfigError, Domain
from nemo_gym.discovery import ConfigFlavorCapabilities
from nemo_gym.environment_catalog import (
    EnvironmentCatalogEntry,
    discover_exact_environment_catalog,
    manifest_config_path,
)
from nemo_gym.environment_manifest import (
    EnvironmentKind,
    EnvironmentManifest,
    IntegrationProfile,
    SpecialLicense,
    dump_manifest,
    validate_adopted_from_reference,
)
from nemo_gym.environment_manifest_edit import (
    ManifestEditFilters,
    apply_manifest_edits,
    parse_manifest_edits,
    select_manifest_paths,
)
from nemo_gym.environment_scaffold import ScaffoldError, default_scaffold_description, scaffold_environment
from nemo_gym.environment_validation import (
    WorkloadInspection,
    has_score_affecting_cli_overrides,
    inspect_workload,
    resolve_composition_mirror,
)
from nemo_gym.environment_versioning import VersionLockResult, check_or_record_version_lock
from nemo_gym.global_config import (
    GlobalConfigDictParserConfig,
    StaticValidationConfigParser,
    get_global_config_dict,
)
from nemo_gym.model_registry import discover_models
from nemo_gym.repository_io import atomic_write_text, find_repository_root
from nemo_gym.resources_server_registry import discover_resources_servers
from nemo_gym.sandbox.providers.registry import list_providers


class InitEnvironmentConfig(BaseNeMoGymCLIConfig):
    """Inputs for the profile-aware environment scaffold."""

    init_name: str
    init_kind: EnvironmentKind
    init_profile: IntegrationProfile = IntegrationProfile.STOCK_LOOP
    init_reuse_verifier: str | None = None
    init_version: str = "0.1.0"
    init_domain: Domain = Domain.OTHER
    init_description: str | None = None
    init_modality: str = "text"
    init_licensing: str = SpecialLicense.UNKNOWN.value
    init_authors: list[str] = Field(default_factory=list)
    init_canonical_split: str | None = None


class ValidateEnvironmentConfig(BaseNeMoGymCLIConfig):
    manifest_path: Path | None = None
    sync_manifest: bool = False
    json_format: bool = Field(default=False, alias="json")


class PublishEnvironmentConfig(ValidateEnvironmentConfig):
    environment_ref: str
    publish_owner: list[str] = Field(default_factory=list)
    publish_dry_run: bool = False


class ListComponentsConfig(BaseNeMoGymCLIConfig):
    component_provides: str | None = None
    json_format: bool = Field(default=False, alias="json")


class EditManifestsConfig(BaseNeMoGymCLIConfig):
    """Selection and assignments for ``gym env manifest``."""

    manifest_names: list[str] = Field(default_factory=list)
    manifest_set: list[str] = Field(default_factory=list)
    catalog_domain: str | None = None
    catalog_kind: str | None = None
    catalog_profile: str | None = None
    dry_run: bool = False
    json_format: bool = Field(default=False, alias="json")


@exit_cleanly_on_config_error
def init_environment() -> None:
    """Scaffold a runnable recipe selected by kind and integration profile."""

    config = InitEnvironmentConfig.model_validate(get_global_config_dict())
    if config.init_kind == EnvironmentKind.BENCHMARK and not config.init_canonical_split:
        raise ConfigError("Benchmark scaffolding requires --canonical-split so the evaluation protocol is explicit.")
    if config.init_kind != EnvironmentKind.BENCHMARK and config.init_canonical_split:
        raise ConfigError("--canonical-split applies only to --benchmark scaffolds.")
    if not config.init_authors:
        raise ConfigError("Environment and benchmark scaffolds require at least one explicit --author.")
    authors = config.init_authors
    metadata = {
        "version": config.init_version,
        "domain": config.init_domain.value,
        "description": config.init_description
        or default_scaffold_description(config.init_kind.value, config.init_name),
        "modality": config.init_modality,
        "licensing": config.init_licensing,
        "authors": authors,
    }
    if config.init_canonical_split is not None:
        metadata["canonical_split"] = config.init_canonical_split
    try:
        result = scaffold_environment(
            kind=config.init_kind.value,
            name=config.init_name,
            profile=config.init_profile.value,
            reuse_verifier=config.init_reuse_verifier,
            metadata=metadata,
            root=Path.cwd(),
        )
    except ScaffoldError as error:
        raise ConfigError(str(error)) from error

    if result.created:
        rich.print(f"[green]✓[/green] Created {len(result.created)} file(s) for '{config.init_name}':")
        for path in result.created:
            rich.print(f"  {path.relative_to(result.root)}")
    if result.existing:
        rich.print(
            f"[cyan]i[/cyan] {len(result.existing)} generated file(s) already matched; nothing was overwritten."
        )
    rich.print(
        "\nNext: edit the generated dataset and verifier, run "
        f"[bold]gym env validate {config.init_name}[/bold], generate scorer expectations with "
        f"[bold]gym env test {config.init_name} --update-expected[/bold], and review the resulting diff."
    )


def resolve_catalog_reference(
    reference: str,
    kind: str | None = None,
    *,
    include_unpublished: bool = False,
    allow_version: bool = False,
) -> EnvironmentCatalogEntry:
    """Resolve a live catalog name, optionally checking its current manifest version."""

    name, separator, version = reference.rpartition("@")
    if not separator:
        name, version = reference, ""
    selected_kind = kind if kind in {"environment", "benchmark"} else None
    if version and not allow_version:
        raise ValueError(
            f"Exact environment reference '{name}@{version}' is not executable. Use the live name '{name}'; "
            "NAME@VERSION is accepted only by `gym env validate` and `gym env publish`."
        )

    catalog = discover_exact_environment_catalog(
        name,
        selected_kind,
        include_unpublished=include_unpublished,
    )
    matches = [entry for entry in catalog.entries if not version or entry.version == version]
    if not matches:
        suffix = f"@{version}" if version else ""
        raise ValueError(f"Unknown environment '{name}{suffix}'. Run `gym list catalog` to see available names.")
    if len(matches) > 1:
        kinds = ", ".join(sorted(entry.kind for entry in matches))
        raise ValueError(f"Environment '{name}' exists as multiple kinds ({kinds}); pass --kind.")
    entry = matches[0]
    if entry.config_path is None or not entry.config_path.is_file():
        raise ValueError(f"Catalog entry '{reference}' has no runnable config.yaml next to its manifest.")
    return entry


def _manifest_for_validation(
    raw: Mapping[str, Any], config: ValidateEnvironmentConfig
) -> tuple[Path, EnvironmentManifest] | None:
    return environment_execution.resolve_manifest_for_validation(raw, config.manifest_path)


@dataclass(frozen=True)
class _ValidationResult:
    inspection: WorkloadInspection
    manifest_path: Path | None
    manifest: EnvironmentManifest | None
    profile_warning: str | None
    synced: bool = False


def _perform_validation(
    raw: DictConfig,
    config: ValidateEnvironmentConfig,
) -> _ValidationResult:
    if not raw.get("config_paths"):
        raise ConfigError(
            "Environment validation requires a runnable target or config. Pass an environment name, "
            "--environment/--benchmark/--resources-server, or --config."
        )
    selected = _manifest_for_validation(raw, config)
    manifest_path, manifest = selected if selected is not None else (None, None)
    mirror = resolve_composition_mirror(raw) if manifest is not None else None
    deltas = (
        environment_execution.manifest_contract_deltas(manifest, mirror, raw)
        if manifest is not None and mirror is not None
        else ()
    )
    synced = False
    if manifest is not None and mirror is not None and config.sync_manifest and raw.get("environment_component_swaps"):
        raise ConfigError(
            "Temporary component swaps cannot be written to the manifest with --sync. "
            "Validate the recipe without swaps before synchronizing it."
        )
    if manifest is not None and mirror is not None and deltas:
        if not config.sync_manifest:
            raise ConfigError(
                "Manifest composition is out of sync with its authoritative config. "
                + "\n  - "
                + "\n  - ".join(deltas)
                + "\nRun again with --sync to update the mirror."
            )
        manifest = EnvironmentManifest.model_validate(
            {**manifest.model_dump(mode="json"), **mirror.to_manifest_update()}
        )
        synced = True
    runtime_manifest = (
        environment_execution.manifest_with_authorized_swaps(raw, manifest, mirror)
        if manifest is not None and mirror is not None
        else None
    )
    inspection = inspect_workload(
        raw,
        strict_missing_datasets=manifest is not None,
        standard_prompt_config=runtime_manifest.standard_prompt_config if runtime_manifest else None,
        manifest=runtime_manifest,
    )
    profile_warning = None
    if manifest is not None:
        assert mirror is not None
        profile_warning = environment_execution.profile_mismatch_warning(manifest, inspection.profile)
        if synced:
            dump_manifest(manifest, manifest_path)
    return _ValidationResult(inspection, manifest_path, manifest, profile_warning, synced)


def _validation_payload(result: _ValidationResult) -> dict[str, Any]:
    return {
        "valid": True,
        "manifest_path": str(result.manifest_path) if result.manifest_path else None,
        "manifest": result.manifest.model_dump(mode="json", exclude_none=True) if result.manifest else None,
        "workload": result.inspection.to_dict(),
        "profile_warning": result.profile_warning,
        "manifest_synced": result.synced,
    }


def _render_validation(result: _ValidationResult) -> None:
    if result.manifest is not None:
        rich.print(
            f"[green]✓[/green] Manifest: {result.manifest.name} v{result.manifest.version}  "
            f"kind={result.manifest.kind.value}  profile={result.manifest.integration_profile.value}"
        )
    else:
        rich.print("[yellow]i[/yellow] No manifest found; validating this legacy workload in compatibility mode.")
    if result.synced:
        rich.print(f"[green]✓[/green] Synced composition mirror in {result.manifest_path}.")
    rich.print(f"[green]✓[/green] Config resolves (sha256:{result.inspection.composition_hash[:12]}).")
    if result.inspection.components:
        table = Table(title="Resolved components")
        for column in ("Role", "Instance", "Implementation", "Version", "Boundary", "Constraint"):
            table.add_column(column)
        for component in result.inspection.components:
            table.add_row(
                component.role,
                component.instance,
                component.implementation,
                component.version or "unversioned",
                component.boundary or "local / assigned at launch",
                "pinned" if component.pinned else "swappable",
            )
        print_rich_table(table)
    for dataset in result.inspection.datasets:
        icon = "[green]✓[/green]" if dataset.status == "valid" else "[yellow]i[/yellow]"
        count = f"{dataset.rows} rows" if dataset.rows is not None else dataset.status
        rich.print(f"{icon} Dataset {dataset.name}: {count} ({dataset.detail or dataset.path})")
        if dataset.materialized_sample is not None:
            rich.print(
                "  Materialized sample: " + json.dumps(dataset.materialized_sample, ensure_ascii=False, sort_keys=True)
            )
    for decision in result.inspection.compatibility_decisions:
        rich.print(f"[green]✓[/green] Capability: {decision}")
    for constraint in result.inspection.fixed_constraints:
        rich.print(f"[cyan]i[/cyan] Constraint: {constraint}")
    if result.inspection.overrides:
        rich.print(
            "[cyan]i[/cyan] Score-affecting overrides: " + json.dumps(result.inspection.overrides, sort_keys=True)
        )
    if result.profile_warning:
        rich.print(f"[yellow]Warning:[/yellow] {result.profile_warning}")
    for warning in result.inspection.warnings:
        rich.print(f"[yellow]Warning:[/yellow] {warning}")
    rich.print("[green]✓[/green] Workload is valid; no services or compute were started.")


@exit_cleanly_on_config_error
def validate_environment() -> None:
    """Validate and inspect the fully resolved workload without starting Ray."""

    raw = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        ),
        global_config_dict_parser_cls=StaticValidationConfigParser,
    )
    config = ValidateEnvironmentConfig.model_validate(raw)
    result = _perform_validation(raw, config)
    if config.json_format:
        print(json.dumps(_validation_payload(result), sort_keys=True))
    else:
        _render_validation(result)


@exit_cleanly_on_config_error
def edit_environment_manifests() -> None:
    """Apply one validated manifest-authoritative change over a catalog selection."""

    config = EditManifestsConfig.model_validate(
        get_global_config_dict(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
            ),
            global_config_dict_parser_cls=StaticValidationConfigParser,
        )
    )
    if config.catalog_kind not in {None, "environment", "benchmark"}:
        raise ConfigError("--kind must be 'environment' or 'benchmark'.")
    if config.catalog_profile not in {
        None,
        *(profile.value for profile in IntegrationProfile),
    }:
        raise ConfigError(
            "--profile must be one of: " + ", ".join(profile.value for profile in IntegrationProfile) + "."
        )
    edits = parse_manifest_edits(config.manifest_set)
    paths = select_manifest_paths(
        ManifestEditFilters(
            names=frozenset(config.manifest_names),
            domain=config.catalog_domain,
            kind=config.catalog_kind,
            profile=config.catalog_profile,
        )
    )
    result = apply_manifest_edits(paths, edits, dry_run=config.dry_run)
    payload = {
        "dry_run": result.dry_run,
        "selected": [str(path) for path in result.selected],
        "changed": [str(path) for path in result.changed],
        "assignments": config.manifest_set,
    }
    if config.json_format:
        print(json.dumps(payload, sort_keys=True))
        return
    verb = "Would update" if result.dry_run else "Updated"
    rich.print(f"[green]✓[/green] {verb} {len(result.changed)} of {len(result.selected)} selected manifest(s).")
    for path in result.changed:
        rich.print(f"  {path}")


def _repo_root_for(path: Path) -> Path:
    if root := find_repository_root(path):
        return root
    raise ConfigError(f"Could not find a Gym repository root above '{path}'.")


def _validated_generated_path(repo_root: Path, path: Path) -> Path:
    """Return a lexical in-repository path with no symbolic-link components."""

    root = Path(os.path.abspath(repo_root))
    target = Path(os.path.abspath(path))
    if root.is_symlink():
        raise ConfigError(f"Refusing generated write through symbolic-link repository root '{root}'.")
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise ConfigError(f"Refusing generated write outside repository '{root}': '{target}'.") from error
    cursor = root
    for index, part in enumerate(relative.parts):
        cursor /= part
        if cursor.is_symlink():
            raise ConfigError(f"Refusing generated write through symbolic-link path component '{cursor}'.")
        if index < len(relative.parts) - 1 and cursor.exists() and not cursor.is_dir():
            raise ConfigError(f"Refusing generated write through non-directory path component '{cursor}'.")
    return target


def _atomic_generated_text_write(repo_root: Path, path: Path, content: str) -> None:
    """Atomically write a generated repository file without following symlinks."""

    target = _validated_generated_path(repo_root, path)
    parent = target.parent
    if not parent.exists():
        parent.mkdir()
    _validated_generated_path(repo_root, target)
    mode = target.lstat().st_mode & 0o777 if target.exists() else 0o644
    try:
        atomic_write_text(target, content, mode=mode)
    except OSError as error:
        raise ConfigError(f"Could not write generated repository file '{target}': {error}.") from error


_OWNER_PART = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,38}[A-Za-z0-9])?"
_OWNER_RE = re.compile(rf"^@?{_OWNER_PART}(?:/{_OWNER_PART})?$")


@dataclass(frozen=True)
class _CodeownersPlan:
    repo_root: Path
    path: Path
    rules: tuple[str, ...]
    original_content: str | None
    updated_content: str
    parent_existed: bool

    @property
    def changed(self) -> bool:
        return self.updated_content != self.original_content


def _normalize_owners(values: Sequence[str]) -> tuple[str, ...]:
    owners: list[str] = []
    for value in values:
        if not _OWNER_RE.fullmatch(value):
            raise ConfigError(
                f"'{value}' is not a GitHub user/team handle; pass --owner with the handle that should review this path."
            )
        owner = value if value.startswith("@") else f"@{value}"
        if owner not in owners:
            owners.append(owner)
    if not owners:
        raise ConfigError("Publishing requires at least one explicit CODEOWNER via --owner.")
    return tuple(owners)


def _read_generated_text(repo_root: Path, path: Path) -> str | None:
    target = _validated_generated_path(repo_root, path)
    if not target.exists():
        return None
    if not target.is_file():
        raise ConfigError(f"Generated repository path '{target}' is not a regular file.")
    try:
        return target.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"Could not read generated repository file '{target}': {error}.") from error


def _plan_codeowners_updates(
    manifest_path: Path,
    owned_directories: Sequence[Path],
    owners: tuple[str, ...],
) -> _CodeownersPlan:
    repo_root = _repo_root_for(manifest_path)
    codeowners = _validated_generated_path(repo_root, repo_root / ".github" / "CODEOWNERS")
    desired_rules = tuple(
        f"/{directory.resolve().relative_to(repo_root.resolve()).as_posix()}/ {' '.join(owners)}"
        for directory in dict.fromkeys(owned_directories)
    )
    original_content = _read_generated_text(repo_root, codeowners)
    lines = original_content.splitlines() if original_content is not None else []
    # Publishing may add ownership for a newly introduced path, but changing an
    # existing exact rule is an ownership transfer and must remain an explicit
    # maintainer operation. Parse by tokens so tabs or repeated spaces cannot be
    # used to hide an existing rule from this guard.
    existing_by_pattern: dict[str, list[tuple[str, ...]]] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = tuple(stripped.split())
        existing_by_pattern.setdefault(tokens[0], []).append(tokens)
    missing_rules: list[str] = []
    for rule in desired_rules:
        desired_tokens = tuple(rule.split())
        existing_rules = existing_by_pattern.get(desired_tokens[0], [])
        conflicts = [tokens for tokens in existing_rules if tokens != desired_tokens]
        if conflicts:
            rendered = " | ".join(" ".join(tokens) for tokens in conflicts)
            raise ConfigError(
                f"Refusing to replace existing CODEOWNERS rule for {desired_tokens[0]}: {rendered}. "
                "Ask a maintainer to review ownership transfers explicitly."
            )
        if not existing_rules:
            missing_rules.append(rule)
    if missing_rules:
        lines.extend(missing_rules)
    updated_content = "\n".join(lines) + "\n" if missing_rules else (original_content or "")
    return _CodeownersPlan(
        repo_root=repo_root,
        path=codeowners,
        rules=desired_rules,
        original_content=original_content,
        updated_content=updated_content,
        parent_existed=codeowners.parent.exists(),
    )


def _commit_codeowners(plan: _CodeownersPlan) -> None:
    if not plan.changed:
        return
    if _read_generated_text(plan.repo_root, plan.path) != plan.original_content:
        raise ConfigError(f"CODEOWNERS changed after publication preflight; refusing to overwrite '{plan.path}'.")
    _atomic_generated_text_write(plan.repo_root, plan.path, plan.updated_content)


def _rollback_codeowners(plan: _CodeownersPlan) -> None:
    if _read_generated_text(plan.repo_root, plan.path) != plan.updated_content:
        raise ConfigError(f"CODEOWNERS changed during publication; refusing unsafe rollback of '{plan.path}'.")
    if plan.original_content is None:
        _validated_generated_path(plan.repo_root, plan.path).unlink()
        if not plan.parent_existed:
            try:
                plan.path.parent.rmdir()
            except OSError:
                pass
        return
    _atomic_generated_text_write(plan.repo_root, plan.path, plan.original_content)


def _owned_component_directories(result: _ValidationResult) -> tuple[Path, ...]:
    assert result.manifest is not None and result.manifest_path is not None
    repo_root = _repo_root_for(result.manifest_path)
    normalized_name = re.sub(r"[^a-z0-9_]", "_", result.manifest.name.rsplit("/", 1)[-1])
    generated_names = {normalized_name, f"{normalized_name}_agent"}
    subdirs = {
        "resources_server": "resources_servers",
        "agent_server": "responses_api_agents",
        "model_server": "responses_api_models",
    }
    directories = [result.manifest_path.parent]
    for component in result.inspection.components:
        if component.implementation not in generated_names:
            continue
        directory = repo_root / subdirs[component.role] / component.implementation
        if directory.is_dir() and not _directory_existed_in_head(repo_root, directory):
            directories.append(directory)
    return tuple(dict.fromkeys(directories))


def _directory_existed_in_head(repo_root: Path, directory: Path) -> bool:
    """Return whether ``directory`` has committed content in the repository HEAD.

    The scaffold is normally published before its PR is committed, and may already
    be staged, so the index is deliberately not consulted. An unborn repository has
    no shared history and therefore treats every component as new. If the path is
    not a usable Git work tree, fail closed and do not auto-claim it.
    """

    relative = directory.resolve().relative_to(repo_root.resolve()).as_posix()
    in_work_tree = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"],
        check=False,
        capture_output=True,
        text=True,
    )
    if in_work_tree.returncode != 0 or in_work_tree.stdout.strip() != "true":
        return True
    has_head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if has_head.returncode != 0:
        return False
    tracked = subprocess.run(
        ["git", "-C", str(repo_root), "ls-tree", "-r", "--name-only", "HEAD", "--", relative],
        check=False,
        capture_output=True,
        text=True,
    )
    return tracked.returncode != 0 or bool(tracked.stdout.strip())


def _validate_publish_catalog_identity(
    raw: Mapping[str, Any],
    config: PublishEnvironmentConfig,
    result: _ValidationResult,
) -> None:
    """Bind publication to the exact catalog entry selected by ``environment_ref``."""

    assert result.manifest is not None and result.manifest_path is not None
    try:
        entry = resolve_catalog_reference(
            config.environment_ref,
            result.manifest.kind.value,
            include_unpublished=True,
            allow_version=True,
        )
    except ValueError as error:
        raise ConfigError(str(error)) from error
    if entry.manifest_path is None or entry.manifest_path.resolve() != result.manifest_path.resolve():
        raise ConfigError(
            f"Publish reference {config.environment_ref!r} resolves to manifest {entry.manifest_path}, "
            f"not the requested override {result.manifest_path}; refusing to publish a different catalog entry."
        )
    if entry.name != result.manifest.name or entry.kind != result.manifest.kind.value:
        raise ConfigError(
            f"Publish reference {config.environment_ref!r} does not match the validated manifest identity "
            f"{result.manifest.kind.value}:{result.manifest.name}."
        )
    selected_config_paths = {
        _resolve_under_cwd_or_install(str(path)).resolve() for path in (raw.get("config_paths") or [])
    }
    if entry.config_path is None or entry.config_path.resolve() not in selected_config_paths:
        raise ConfigError(
            f"Publish reference {config.environment_ref!r} resolves to config {entry.config_path}, but that "
            "catalog config was not selected for validation."
        )


def _validate_publish_metadata(manifest: EnvironmentManifest) -> None:
    placeholder = default_scaffold_description(manifest.kind.value, manifest.name)
    if manifest.description == placeholder:
        raise ConfigError(
            "Publishing requires a searchable task description; replace the generated scaffold description."
        )
    if any(author.casefold() == "todo" for author in manifest.authors):
        raise ConfigError("Publishing requires named authors; replace the generated TODO author.")
    if manifest.licensing == SpecialLicense.UNKNOWN.value:
        raise ConfigError("Publishing requires a settled SPDX license or access classification; replace 'unknown'.")
    if manifest.adopted_from is not None:
        source = manifest.adopted_from.source
        if urlsplit(source).scheme.casefold() != "https":
            raise ConfigError(
                "Publishing adopted_from provenance requires a public HTTPS source; "
                "local, SCP-like, SSH, and credential-dependent sources remain valid for local use only."
            )


def _validate_publication_request(
    raw: Mapping[str, Any],
    config: PublishEnvironmentConfig,
    result: _ValidationResult,
) -> None:
    if result.manifest is None or result.manifest_path is None:
        raise ConfigError(
            "Publishing requires a valid manifest.yaml; legacy no-manifest entries remain runnable only."
        )
    if result.profile_warning:
        raise ConfigError(
            "Publishing requires the declared integration profile to match the resolved workload. "
            + result.profile_warning
        )
    if has_score_affecting_cli_overrides(raw):
        raise ConfigError(
            "Publishing accepts only the canonical checked-in composition; remove temporary component swaps or "
            "score-affecting Hydra overrides and persist the intended config first."
        )
    _validate_publish_metadata(result.manifest)
    if result.manifest.adopted_from is not None:
        validate_adopted_from_reference(result.manifest.adopted_from)
    reference_name, separator, reference_version = config.environment_ref.rpartition("@")
    if not separator:
        reference_name, reference_version = config.environment_ref, ""
    if result.manifest.name != reference_name:
        raise ConfigError(
            f"Publish reference {config.environment_ref!r} resolved manifest name={result.manifest.name!r}; "
            "refusing to publish a different unit."
        )
    if reference_version and result.manifest.version != reference_version:
        raise ConfigError(
            f"Publish reference requests version {reference_version!r}, but the manifest declares "
            f"{result.manifest.version!r}."
        )
    _validate_publish_catalog_identity(raw, config, result)
    if result.manifest.licensing in {SpecialLicense.INTERNAL.value, SpecialLicense.PROPRIETARY.value}:
        raise ConfigError(
            f"Cannot publish licensing={result.manifest.licensing!r} into a registry without access control. "
            "Keep using the environment locally until access binding is available."
        )


@dataclass(frozen=True)
class _PublicationPlan:
    repo_root: Path
    lock: VersionLockResult
    codeowners: _CodeownersPlan
    config_path: Path
    dry_run: bool


def _prepare_publication(
    config: PublishEnvironmentConfig,
    result: _ValidationResult,
) -> _PublicationPlan:
    assert result.manifest is not None and result.manifest_path is not None
    owners = _normalize_owners(config.publish_owner)
    repo_root = _repo_root_for(result.manifest_path)
    runnable_config_path = manifest_config_path(result.manifest_path)
    if runnable_config_path is None:
        raise ConfigError("Publishing requires one unambiguous runnable config next to the manifest.")
    lock = check_or_record_version_lock(
        repo_root=repo_root,
        manifest_path=result.manifest_path,
        manifest=result.manifest,
        composition_hash=result.inspection.composition_hash,
        config_path=runnable_config_path,
        dry_run=True,
    )
    return _PublicationPlan(
        repo_root=repo_root,
        lock=lock,
        codeowners=_plan_codeowners_updates(
            result.manifest_path,
            _owned_component_directories(result),
            owners,
        ),
        config_path=runnable_config_path,
        dry_run=config.publish_dry_run,
    )


def _rollback_publication(
    plan: _PublicationPlan,
    error: Exception,
    *,
    codeowners_written: bool,
) -> None:
    rollback_errors: list[str] = []
    if codeowners_written:
        try:
            _rollback_codeowners(plan.codeowners)
        except Exception as rollback_error:
            rollback_errors.append(str(rollback_error))
    if rollback_errors:
        raise ConfigError(
            f"Publication failed ({error}); rollback was incomplete: {'; '.join(rollback_errors)}"
        ) from error


def _commit_publication(result: _ValidationResult, plan: _PublicationPlan) -> VersionLockResult:
    assert result.manifest is not None and result.manifest_path is not None
    if plan.dry_run:
        return plan.lock

    codeowners_written = False
    try:
        if plan.codeowners.changed:
            _commit_codeowners(plan.codeowners)
            codeowners_written = True
        return check_or_record_version_lock(
            repo_root=plan.repo_root,
            manifest_path=result.manifest_path,
            manifest=result.manifest,
            composition_hash=result.inspection.composition_hash,
            config_path=plan.config_path,
            dry_run=False,
        )
    except Exception as error:
        _rollback_publication(plan, error, codeowners_written=codeowners_written)
        raise


def _publication_payload(
    result: _ValidationResult,
    plan: _PublicationPlan,
    lock: VersionLockResult,
) -> dict[str, Any]:
    assert result.manifest is not None and result.manifest_path is not None
    return {
        "published": not plan.dry_run,
        "dry_run": plan.dry_run,
        "name": result.manifest.name,
        "version": result.manifest.version,
        "environment_version_key": lock.key,
        "manifest_path": str(result.manifest_path),
        "codeowners_path": str(plan.codeowners.path),
        "codeowners_rule": plan.codeowners.rules[0],
        "codeowners_rules": list(plan.codeowners.rules),
        "codeowners_changed": plan.codeowners.changed,
        "composition_hash": result.inspection.composition_hash,
        "version_lock_path": str(lock.path),
        "version_lock_changed": lock.changed,
    }


def _render_publication(result: _ValidationResult, plan: _PublicationPlan, lock: VersionLockResult) -> None:
    assert result.manifest is not None and result.manifest_path is not None
    verb = "Would publish" if plan.dry_run else "Published"
    rich.print(
        f"[green]✓[/green] {verb} {result.manifest.name}@{result.manifest.version}.\n"
        f"  manifest: {result.manifest_path}\n"
        f"  composition: sha256:{result.inspection.composition_hash}\n"
        f"  environment version: {lock.key}\n"
        f"  CODEOWNERS: {', '.join(plan.codeowners.rules)}"
    )


@exit_cleanly_on_config_error
def publish_environment() -> None:
    """Validate a manifest-backed unit and register contributor ownership."""

    raw = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        ),
        global_config_dict_parser_cls=StaticValidationConfigParser,
    )
    config = PublishEnvironmentConfig.model_validate(raw)
    result = _perform_validation(raw, config)
    _validate_publication_request(raw, config, result)
    plan = _prepare_publication(config, result)
    lock = _commit_publication(result, plan)
    if config.json_format:
        print(json.dumps(_publication_payload(result, plan, lock), sort_keys=True))
        return
    _render_publication(result, plan, lock)


@dataclass(frozen=True)
class _ComponentRecord:
    name: str
    kind: str
    implementation: str
    instance: str
    config_path: Path | None
    requires: tuple[str, ...]
    provides: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "implementation": self.implementation,
            "instance": self.instance,
            "config_path": str(self.config_path) if self.config_path is not None else None,
            "requires": list(self.requires),
            "provides": list(self.provides),
        }


def _discover_declared_components() -> tuple[_ComponentRecord, ...]:
    records: list[_ComponentRecord] = []

    def add(selector: str, kind: str, capabilities: ConfigFlavorCapabilities) -> None:
        for declaration in capabilities.declarations:
            records.append(
                _ComponentRecord(
                    name=selector,
                    kind=kind,
                    implementation=declaration.implementation,
                    instance=declaration.instance,
                    config_path=capabilities.config_path.resolve(),
                    requires=declaration.requires,
                    provides=declaration.provides,
                )
            )

    for entry in discover_resources_servers().values():
        add(entry.name, "resources-server", entry.capabilities)
    for entry in discover_models().values():
        add(entry.name, "model-server", entry.capabilities)
    for entry in discover_agents().values():
        for flavor, capabilities in entry.capability_flavors.items():
            selector = entry.name if flavor == entry.name else f"{entry.name}/{flavor}"
            add(selector, "agent-server", capabilities)
    provider_root = Path(__file__).resolve().parents[1] / "sandbox" / "providers"
    for provider_name in list_providers():
        config_path = provider_root / provider_name / "configs" / f"{provider_name}.yaml"
        records.append(
            _ComponentRecord(
                name=provider_name,
                kind="sandbox-provider",
                implementation=provider_name,
                instance="sandbox",
                config_path=config_path.resolve() if config_path.is_file() else None,
                requires=(),
                provides=(f"sandbox:{provider_name}",),
            )
        )

    return tuple(
        sorted(
            records,
            key=lambda record: (record.kind, record.name, record.implementation, record.instance),
        )
    )


@exit_cleanly_on_config_error
def list_components() -> None:
    """List reusable deployables, optionally filtered by a provided capability."""

    config = ListComponentsConfig.model_validate(get_global_config_dict())
    components = _discover_declared_components()
    if config.component_provides:
        components = tuple(
            component
            for component in components
            if any(fnmatchcase(capability, config.component_provides) for capability in component.provides)
        )
    if config.json_format:
        print(json.dumps([component.to_dict() for component in components], sort_keys=True))
        return
    if not components:
        qualifier = f" providing '{config.component_provides}'" if config.component_provides else ""
        rich.print(f"[yellow]No components{qualifier} found.[/yellow]")
        return
    table = Table(title=f"Reusable components ({len(components)})")
    for column in ("Name", "Kind", "Implementation", "Provides", "Requires", "Config"):
        table.add_column(column)
    for component in components:
        table.add_row(
            component.name,
            component.kind,
            component.implementation,
            ", ".join(component.provides),
            ", ".join(component.requires),
            str(component.config_path) if component.config_path is not None else "",
        )
    print_rich_table(table)
