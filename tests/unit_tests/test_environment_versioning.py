# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.environment_manifest import EnvironmentManifest
from nemo_gym.environment_versioning import (
    LOCK_RELATIVE_PATH,
    check_or_record_version_lock,
    load_version_locks,
    validate_version_locks,
    verify_version_lock,
)


def _manifest(version: str = "1.2.3") -> EnvironmentManifest:
    return EnvironmentManifest.model_validate(
        {
            "name": "fixture",
            "kind": "environment",
            "version": version,
            "integration_profile": "stock-loop",
            "domain": "other",
            "description": "A complete fixture environment.",
            "modality": "text",
            "licensing": "Apache-2.0",
            "authors": ["Fixture Owner"],
            "resources_server": "fixture",
            "agent_server": "fixture_agent",
            "model_server": "fixture_model",
            "datasets": [
                {
                    "name": "fixture",
                    "type": "example",
                    "jsonl_fpath": "environments/fixture/data.jsonl",
                }
            ],
            "reward": {"range": [0, 1], "higher_is_better": True},
        }
    )


def _paths(root: Path) -> tuple[Path, Path]:
    component = root / "environments" / "fixture"
    component.mkdir(parents=True)
    manifest_path = component / "manifest.yaml"
    config_path = component / "config.yaml"
    manifest_path.write_text("name: fixture\n", encoding="utf-8")
    config_path.write_text("fixture: {}\n", encoding="utf-8")
    return manifest_path, config_path


def test_record_is_minimal_and_idempotent(tmp_path: Path) -> None:
    manifest_path, config_path = _paths(tmp_path)
    manifest = _manifest()
    first = check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        manifest=manifest,
        composition_hash="a" * 64,
        config_path=config_path,
    )
    second = check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        manifest=manifest,
        composition_hash="a" * 64,
        config_path=config_path,
    )
    assert first.changed is True
    assert second.changed is False
    record = load_version_locks(tmp_path / LOCK_RELATIVE_PATH)["environments"][first.key]
    assert record == {
        "composition_hash": "a" * 64,
        "manifest": "environments/fixture/manifest.yaml",
        "config": "environments/fixture/config.yaml",
    }
    assert (tmp_path / LOCK_RELATIVE_PATH).is_file()


def test_dry_run_does_not_create_lock_file(tmp_path: Path) -> None:
    manifest_path, config_path = _paths(tmp_path)
    result = check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        manifest=_manifest(),
        composition_hash="a" * 64,
        config_path=config_path,
        dry_run=True,
    )
    assert result.changed is True
    assert not (tmp_path / LOCK_RELATIVE_PATH).exists()


def test_published_version_is_immutable(tmp_path: Path) -> None:
    manifest_path, config_path = _paths(tmp_path)
    manifest = _manifest()
    check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        manifest=manifest,
        composition_hash="a" * 64,
        config_path=config_path,
    )
    with pytest.raises(ConfigError, match="bump manifest.version"):
        check_or_record_version_lock(
            repo_root=tmp_path,
            manifest_path=manifest_path,
            manifest=manifest,
            composition_hash="b" * 64,
            config_path=config_path,
        )


def test_version_bump_records_new_composition(tmp_path: Path) -> None:
    manifest_path, config_path = _paths(tmp_path)
    for version, digest in (("1.2.3", "a" * 64), ("1.2.4", "b" * 64)):
        check_or_record_version_lock(
            repo_root=tmp_path,
            manifest_path=manifest_path,
            manifest=_manifest(version),
            composition_hash=digest,
            config_path=config_path,
        )
    assert len(load_version_locks(tmp_path / LOCK_RELATIVE_PATH)["environments"]) == 2


def test_validate_version_locks_reports_live_drift(tmp_path: Path) -> None:
    manifest_path, config_path = _paths(tmp_path)
    result = check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        manifest=_manifest(),
        composition_hash="a" * 64,
        config_path=config_path,
    )
    violations = validate_version_locks(repo_root=tmp_path, current_hashes={result.key: "b" * 64})
    assert len(violations) == 1
    assert "bump manifest.version" in violations[0]


def test_verify_version_lock_accepts_exact_published_composition(tmp_path: Path) -> None:
    manifest_path, config_path = _paths(tmp_path)
    manifest = _manifest()
    check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        manifest=manifest,
        composition_hash="a" * 64,
        config_path=config_path,
    )

    record = verify_version_lock(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        config_path=config_path,
        manifest=manifest,
        composition_hash="a" * 64,
        require_published=True,
    )

    assert record is not None
    assert record["composition_hash"] == "a" * 64


def test_verify_version_lock_rejects_published_drift(tmp_path: Path) -> None:
    manifest_path, config_path = _paths(tmp_path)
    manifest = _manifest()
    check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        manifest=manifest,
        composition_hash="a" * 64,
        config_path=config_path,
    )

    with pytest.raises(ConfigError, match="composition_hash"):
        verify_version_lock(
            repo_root=tmp_path,
            manifest_path=manifest_path,
            config_path=config_path,
            manifest=manifest,
            composition_hash="b" * 64,
        )


def test_verify_version_lock_can_require_publication(tmp_path: Path) -> None:
    manifest_path, config_path = _paths(tmp_path)
    manifest = _manifest()

    assert (
        verify_version_lock(
            repo_root=tmp_path,
            manifest_path=manifest_path,
            config_path=config_path,
            manifest=manifest,
            composition_hash="a" * 64,
        )
        is None
    )
    with pytest.raises(ConfigError, match="no published composition lock"):
        verify_version_lock(
            repo_root=tmp_path,
            manifest_path=manifest_path,
            config_path=config_path,
            manifest=manifest,
            composition_hash="a" * 64,
            require_published=True,
        )


def test_load_rejects_extra_record_fields(tmp_path: Path) -> None:
    path = tmp_path / LOCK_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "environments": {
                    "environment:fixture@1.2.3": {
                        "composition_hash": "a" * 64,
                        "manifest": "environments/fixture/manifest.yaml",
                        "config": "environments/fixture/config.yaml",
                        "unexpected": True,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="must contain only"):
        load_version_locks(path)
