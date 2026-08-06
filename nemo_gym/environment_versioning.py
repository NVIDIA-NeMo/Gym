# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable composition locks for published manifest versions."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from nemo_gym.config_types import ConfigError
from nemo_gym.environment_manifest import NAME_PATTERN, SEMVER_PATTERN, EnvironmentManifest
from nemo_gym.environment_version_contract import (
    LOCK_RELATIVE_PATH,
    LOCK_SCHEMA_VERSION,
    environment_version_key,
)
from nemo_gym.repository_io import atomic_write_text, exclusive_directory_lock


@dataclass(frozen=True)
class VersionLockResult:
    """Outcome of checking or recording one immutable composition lock."""

    path: Path
    key: str
    composition_hash: str
    changed: bool


def _empty_lock_document() -> dict[str, Any]:
    return {"schema_version": LOCK_SCHEMA_VERSION, "environments": {}}


def _validated_relative_path(value: object, *, field_name: str, key: str) -> Path:
    path = Path(value) if isinstance(value, str) else None
    if path is None or path.is_absolute() or ".." in path.parts or not path.parts:
        raise ConfigError(f"Environment composition lock '{key}' has an invalid {field_name!r} path.")
    return path


def _validate_hash(value: object, *, key: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ConfigError(f"Environment composition lock '{key}' has an invalid SHA-256 composition hash.")
    return value


def load_version_locks(path: Path) -> dict[str, Any]:
    """Load and structurally validate the generated composition-lock document."""

    if path.parent.is_symlink() or path.is_symlink():
        raise ConfigError(f"Refusing environment composition locks through symbolic-link path '{path}'.")
    if not path.is_file():
        return _empty_lock_document()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"Could not read environment composition locks at '{path}': {error}.") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise ConfigError(f"Environment composition locks at '{path}' must use schema_version={LOCK_SCHEMA_VERSION}.")
    records = payload.get("environments")
    if not isinstance(records, dict):
        raise ConfigError(f"Environment composition locks at '{path}' must contain an 'environments' object.")
    for key, record in records.items():
        if not isinstance(key, str) or not isinstance(record, dict):
            raise ConfigError(f"Environment composition lock '{key}' at '{path}' is malformed.")
        kind, separator, identity_version = key.partition(":")
        name, at, version = identity_version.rpartition("@")
        if (
            not separator
            or not at
            or kind not in {"environment", "benchmark"}
            or re.fullmatch(NAME_PATTERN, name) is None
            or re.fullmatch(SEMVER_PATTERN, version) is None
        ):
            raise ConfigError(f"Environment composition lock key '{key}' at '{path}' is invalid.")
        _validate_hash(record.get("composition_hash"), key=key)
        manifest_path = _validated_relative_path(record.get("manifest"), field_name="manifest", key=key)
        config_path = _validated_relative_path(record.get("config"), field_name="config", key=key)
        if manifest_path.name != "manifest.yaml":
            raise ConfigError(f"Environment composition lock '{key}' has an invalid manifest filename.")
        if config_path.suffix.casefold() not in {".yaml", ".yml"}:
            raise ConfigError(f"Environment composition lock '{key}' has an invalid config filename.")
        if set(record) != {"composition_hash", "manifest", "config"}:
            raise ConfigError(
                f"Environment composition lock '{key}' must contain only composition_hash, manifest, and config."
            )
    return payload


def _relative_to_root(repo_root: Path, path: Path, *, label: str) -> str:
    try:
        relative = Path(os.path.abspath(path)).relative_to(Path(os.path.abspath(repo_root)))
    except ValueError as error:
        raise ConfigError(f"{label} '{path}' is not inside repository '{repo_root}'.") from error
    if ".." in relative.parts or not relative.parts:
        raise ConfigError(f"{label} '{path}' is not a safe repository path.")
    return relative.as_posix()


def _record(
    *,
    repo_root: Path,
    manifest_path: Path,
    config_path: Path,
    composition_hash: str,
) -> dict[str, str]:
    return {
        "composition_hash": composition_hash,
        "manifest": _relative_to_root(repo_root, manifest_path, label="Manifest"),
        "config": _relative_to_root(repo_root, config_path, label="Config"),
    }


def check_or_record_version_lock(
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest: EnvironmentManifest,
    composition_hash: str,
    config_path: Path | None = None,
    dry_run: bool = False,
) -> VersionLockResult:
    """Reject composition drift or atomically record a new manifest-version lock."""

    _validate_hash(composition_hash, key=environment_version_key(manifest))
    root = Path(os.path.abspath(repo_root))
    if root.is_symlink() or not root.is_dir():
        raise ConfigError(f"Repository root '{root}' must be a regular directory.")
    path = root / LOCK_RELATIVE_PATH
    resolved_config_path = config_path or manifest_path.with_name("config.yaml")
    record = _record(
        repo_root=root,
        manifest_path=manifest_path,
        config_path=resolved_config_path,
        composition_hash=composition_hash,
    )
    key = environment_version_key(manifest)

    try:
        with exclusive_directory_lock(root):
            payload = load_version_locks(path)
            existing = payload["environments"].get(key)
            if existing is not None:
                if existing != record:
                    raise ConfigError(
                        f"Published environment version '{key}' is immutable: its locked composition or path changed. "
                        "Restore the original composition or bump manifest.version."
                    )
                return VersionLockResult(path, key, composition_hash, False)
            if dry_run:
                return VersionLockResult(path, key, composition_hash, True)
            payload["environments"][key] = record
            content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            atomic_write_text(path, content, create_parent=True)
    except OSError as error:
        raise ConfigError(f"Could not update environment composition locks at '{path}': {error}.") from error
    return VersionLockResult(path, key, composition_hash, True)


def verify_version_lock(
    *,
    repo_root: Path,
    manifest_path: Path,
    config_path: Path,
    manifest: EnvironmentManifest,
    composition_hash: str,
    require_published: bool = False,
) -> Mapping[str, str] | None:
    """Verify one live manifest/config/hash tuple against its published lock."""

    root = Path(os.path.abspath(repo_root))
    key = environment_version_key(manifest)
    expected = _record(
        repo_root=root,
        manifest_path=manifest_path,
        config_path=config_path,
        composition_hash=composition_hash,
    )
    record = load_version_locks(root / LOCK_RELATIVE_PATH)["environments"].get(key)
    if record is None:
        if require_published:
            raise ConfigError(f"Environment version '{key}' has no published composition lock.")
        return None
    if record != expected:
        mismatches = ", ".join(
            field_name
            for field_name in ("manifest", "config", "composition_hash")
            if record.get(field_name) != expected[field_name]
        )
        raise ConfigError(
            f"Published environment version '{key}' no longer matches its locked {mismatches}; "
            "restore the published composition or bump manifest.version."
        )
    return record


def validate_version_locks(
    *,
    repo_root: Path,
    current_hashes: Mapping[str, str],
    lock_path: Path | None = None,
) -> tuple[str, ...]:
    """Return composition drift for live versions represented by ``current_hashes``."""

    path = lock_path or repo_root / LOCK_RELATIVE_PATH
    payload = load_version_locks(path)
    violations: list[str] = []
    for key, record in sorted(payload["environments"].items()):
        current_hash = current_hashes.get(key)
        if current_hash is None:
            continue
        locked_hash = str(record["composition_hash"])
        if current_hash != locked_hash:
            violations.append(
                f"{key}: resolved composition sha256:{current_hash} differs from locked sha256:{locked_hash}; "
                "restore the published composition or bump manifest.version"
            )
    return tuple(violations)


__all__ = [
    "LOCK_RELATIVE_PATH",
    "VersionLockResult",
    "check_or_record_version_lock",
    "environment_version_key",
    "load_version_locks",
    "validate_version_locks",
    "verify_version_lock",
]
