# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manifest preflight shared by every environment execution entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig

from nemo_gym import _resolve_under_cwd_or_install, component_search_roots
from nemo_gym.config_types import ConfigError
from nemo_gym.discovery import resolve_config_paths_static
from nemo_gym.environment_inventory import MIGRATION_DRAFT_HEADER, is_tracked_migration_draft
from nemo_gym.environment_manifest import (
    EnvironmentKind,
    EnvironmentManifest,
    ManifestError,
    load_manifest,
)
from nemo_gym.environment_validation import (
    CompositionMirror,
    compute_composition_hash,
    infer_integration_profile,
    manifest_composition_deltas,
    pinned_component_roles,
    resolve_composition_mirror,
    validate_execution_contracts,
)
from nemo_gym.environment_version_contract import LOCK_RELATIVE_PATH, environment_version_key
from nemo_gym.environment_versioning import load_version_locks, verify_version_lock
from nemo_gym.repository_io import find_repository_root


@dataclass(frozen=True)
class ExecutionPreflightResult:
    """The manifest contract checked at an execution boundary."""

    manifest_path: Path
    manifest: EnvironmentManifest
    profile: str


def _candidate_manifest_paths(config_paths: Sequence[object]) -> tuple[Path, ...]:
    search_roots = tuple(root.resolve() for root in component_search_roots())
    roots = set(search_roots)
    draft_status_by_path: dict[Path, bool] = {}
    candidates: list[Path] = []
    for raw_path in config_paths:
        config_path = _resolve_under_cwd_or_install(str(raw_path)).resolve()
        for parent in (config_path.parent, *config_path.parents):
            candidate = parent / "manifest.yaml"
            if candidate.is_file():
                if candidate in candidates:
                    break
                is_draft = draft_status_by_path.get(candidate)
                if is_draft is None:
                    try:
                        with candidate.open(encoding="utf-8") as stream:
                            has_draft_header = stream.read(len(MIGRATION_DRAFT_HEADER)) == MIGRATION_DRAFT_HEADER
                    except (OSError, UnicodeError):
                        has_draft_header = False
                    is_draft = has_draft_header and is_tracked_migration_draft(candidate, search_roots)
                    draft_status_by_path[candidate] = is_draft
                if not is_draft:
                    candidates.append(candidate)
                    break
            if parent in roots:
                break
    return tuple(candidates)


def _validate_manifest_location(path: Path, manifest: EnvironmentManifest) -> None:
    kind_by_tree = {
        "environments": EnvironmentKind.ENVIRONMENT,
        "benchmarks": EnvironmentKind.BENCHMARK,
        "resources_servers": EnvironmentKind.ENVIRONMENT,
        "example_environments": EnvironmentKind.ENVIRONMENT,
    }
    resolved_path = path.resolve()
    if path.name != "manifest.yaml":
        raise ManifestError(f"Environment manifests must be named 'manifest.yaml', got '{path.name}'.")

    for root in component_search_roots():
        try:
            relative_dir = resolved_path.parent.relative_to(root.resolve())
        except ValueError:
            continue
        if not relative_dir.parts or relative_dir.parts[0] not in kind_by_tree:
            continue
        if len(relative_dir.parts) < 2:
            raise ManifestError(
                f"Manifest '{path}' must identify a named unit below one of: " + ", ".join(sorted(kind_by_tree)) + "."
            )
        tree = relative_dir.parts[0]
        expected_name = Path(*relative_dir.parts[1:]).as_posix()
        if manifest.kind != kind_by_tree[tree]:
            raise ManifestError(
                f"Manifest '{path}' declares kind={manifest.kind.value!r}, but its registry tree '{tree}' "
                f"requires kind={kind_by_tree[tree].value!r}."
            )
        if manifest.name != expected_name:
            raise ManifestError(
                f"Manifest '{path}' declares name={manifest.name!r}, but its registry path identifies "
                f"{expected_name!r}."
            )
        return
    raise ManifestError(
        f"Manifest '{path}' is outside the environment registry. Place it below one of: "
        + ", ".join(sorted(kind_by_tree))
        + "."
    )


def resolve_manifest_for_validation(
    raw: Mapping[str, Any] | DictConfig,
    manifest_path: str | Path | None = None,
) -> tuple[Path, EnvironmentManifest] | None:
    """Resolve one explicit or config-adjacent manifest and validate its identity."""

    if manifest_path is not None:
        path = _resolve_under_cwd_or_install(manifest_path).resolve()
        manifest = load_manifest(path)
        _validate_manifest_location(path, manifest)
        return path, manifest
    candidates = _candidate_manifest_paths(raw.get("config_paths") or [])
    if len(candidates) > 1:
        raise ManifestError("Selected configs resolve to more than one manifest: " + ", ".join(map(str, candidates)))
    if not candidates:
        return None
    manifest = load_manifest(candidates[0])
    _validate_manifest_location(candidates[0], manifest)
    return candidates[0], manifest


def _manifest_repository_root(manifest_path: Path) -> Path:
    repository_root = find_repository_root(manifest_path)
    if repository_root is not None:
        return repository_root.resolve()

    resolved_manifest = manifest_path.resolve()
    for root in component_search_roots():
        resolved_root = root.resolve()
        try:
            resolved_manifest.relative_to(resolved_root)
        except ValueError:
            continue
        return resolved_root
    raise ConfigError(f"Manifest '{manifest_path}' is outside every repository or component search root.")


def _selected_config_paths(raw: Mapping[str, Any] | DictConfig) -> frozenset[Path]:
    config_paths = raw.get("config_paths") or []
    if not isinstance(config_paths, Sequence) or isinstance(config_paths, (str, bytes, bytearray)):
        raise ConfigError("Manifest-bound execution requires config_paths to be a list.")
    return frozenset(_resolve_under_cwd_or_install(str(path)).resolve() for path in config_paths)


def _locked_config_path(
    repository_root: Path,
    record: Mapping[str, Any],
    *,
    version_key: str,
) -> Path:
    """Resolve a validated lock path without following symlinks out of the repository."""

    relative_path = Path(str(record["config"]))
    candidate = repository_root
    for part in relative_path.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ConfigError(
                f"Published environment version '{version_key}' uses symbolic-link locked config '{candidate}'."
            )
    if not candidate.is_file():
        raise ConfigError(f"Published environment version '{version_key}' locked config '{candidate}' is not a file.")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(repository_root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as error:
        raise ConfigError(
            f"Published environment version '{version_key}' locked config '{candidate}' is outside its repository."
        ) from error
    return resolved


def _verify_manifest_version_lock(
    raw: Mapping[str, Any] | DictConfig,
    manifest_path: Path,
    manifest: EnvironmentManifest,
) -> None:
    """Verify an existing publication lock without requiring one for local drafts."""

    repository_root = _manifest_repository_root(manifest_path)
    version_key = environment_version_key(manifest)
    records = load_version_locks(repository_root / LOCK_RELATIVE_PATH)["environments"]
    record = records.get(version_key)
    if record is None:
        return

    config_path = _locked_config_path(repository_root, record, version_key=version_key)
    if config_path not in _selected_config_paths(raw):
        raise ConfigError(
            f"Published environment version '{version_key}' is not running its locked config '{config_path}'."
        )

    try:
        hash_config = resolve_config_paths_static((config_path,))
        composition_hash = compute_composition_hash(hash_config, manifest)
        verify_version_lock(
            repo_root=repository_root,
            manifest_path=manifest_path,
            config_path=config_path,
            manifest=manifest,
            composition_hash=composition_hash,
        )
    except ConfigError:
        raise
    except Exception as error:
        message = str(error).strip() or type(error).__name__
        raise ConfigError(f"Could not verify published environment version '{version_key}': {message}") from error


def manifest_contract_deltas(
    manifest: EnvironmentManifest,
    mirror: CompositionMirror,
    raw: Mapping[str, Any] | DictConfig | None = None,
) -> tuple[str, ...]:
    """Compare the manifest mirror and enforce profile-pinned ownership."""

    deltas = manifest_composition_deltas(manifest, mirror)
    authorized_fields: frozenset[str] = frozenset()
    requested_agent_swap = False
    if raw is not None and raw.get("environment_component_swaps"):
        from nemo_gym.environment_component_swaps import (
            authorized_manifest_delta_fields,
            authorized_swap_roles,
        )

        requested_agent_swap = "agent_server" in authorized_swap_roles(raw)
        authorized_fields = authorized_manifest_delta_fields(raw, manifest, mirror)
    if "agent_server" in pinned_component_roles(manifest.integration_profile.value) and (
        requested_agent_swap or any(delta.startswith("agent_server:") for delta in deltas)
    ):
        raise ConfigError(
            f"Profile '{manifest.integration_profile.value}' pins the agent server; "
            "the requested swap is invalid. Restore the declared agent server or publish a new environment shape."
        )
    return tuple(delta for delta in deltas if delta.partition(":")[0] not in authorized_fields)


def manifest_with_authorized_swaps(
    raw: Mapping[str, Any] | DictConfig,
    manifest: EnvironmentManifest,
    mirror: CompositionMirror,
) -> EnvironmentManifest:
    """Project authorized runtime component selections without changing the manifest file."""

    if not raw.get("environment_component_swaps"):
        return manifest
    from nemo_gym.environment_component_swaps import authorized_manifest_delta_fields

    fields = authorized_manifest_delta_fields(raw, manifest, mirror)
    if not fields:
        return manifest
    payload = manifest.model_dump(mode="json")
    for field_name in fields:
        payload[field_name] = getattr(mirror, field_name)
    return EnvironmentManifest.model_validate(payload)


def profile_mismatch_warning(manifest: EnvironmentManifest, profile: str) -> str | None:
    """Describe a mismatch between declared and statically inferred profiles."""

    if manifest.integration_profile.value == profile:
        return None
    return (
        f"Declared integration_profile={manifest.integration_profile.value!r} differs from "
        f"the config classifier result {profile!r}."
    )


def resolve_manifest_execution_binding(
    raw: Mapping[str, Any] | DictConfig,
) -> ExecutionPreflightResult | None:
    """Resolve and synchronize a manifest without validating the full runtime graph."""

    manifest_path = raw.get("manifest_path")
    if not manifest_path:
        return None
    selected = resolve_manifest_for_validation(raw, str(manifest_path))
    assert selected is not None
    resolved_manifest_path, manifest = selected
    mirror = resolve_composition_mirror(raw)
    deltas = manifest_contract_deltas(manifest, mirror, raw)
    profile = infer_integration_profile(raw)
    profile_warning = profile_mismatch_warning(manifest, profile)
    if profile_warning:
        raise ConfigError(profile_warning)
    if deltas:
        raise ConfigError(
            "Manifest composition is out of sync with its authoritative config. " + "\n  - " + "\n  - ".join(deltas)
        )
    _verify_manifest_version_lock(raw, resolved_manifest_path, manifest)
    runtime_manifest = manifest_with_authorized_swaps(raw, manifest, mirror)
    return ExecutionPreflightResult(resolved_manifest_path, runtime_manifest, profile)


def preflight_manifest_execution(
    raw: Mapping[str, Any] | DictConfig,
    *,
    check_launch_sources: bool = True,
) -> ExecutionPreflightResult | None:
    """Validate a manifest-bound launch before runtime initialization."""

    binding = resolve_manifest_execution_binding(raw)
    if binding is None:
        return None
    validate_execution_contracts(
        raw,
        binding.manifest,
        profile=binding.profile,
        check_launch_sources=check_launch_sources,
    )
    return binding


__all__ = [
    "ExecutionPreflightResult",
    "manifest_contract_deltas",
    "manifest_with_authorized_swaps",
    "preflight_manifest_execution",
    "profile_mismatch_warning",
    "resolve_manifest_execution_binding",
    "resolve_manifest_for_validation",
]
