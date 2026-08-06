# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

import nemo_gym.environment_manifest_edit as manifest_edit
from nemo_gym.config_types import ConfigError
from nemo_gym.environment_catalog import CatalogCoverage, EnvironmentCatalog, EnvironmentCatalogEntry
from nemo_gym.environment_manifest import EnvironmentManifest, dump_manifest, load_manifest
from nemo_gym.environment_manifest_edit import (
    ManifestEditFilters,
    apply_manifest_edits,
    parse_manifest_edits,
    select_manifest_paths,
)


def _write_manifest(path: Path, *, name: str = "fixture", domain: str = "other") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_manifest(
        EnvironmentManifest.model_validate(
            {
                "name": name,
                "version": "1.0.0",
                "kind": "environment",
                "integration_profile": "stock-loop",
                "domain": domain,
                "description": "An environment bulk-edit fixture.",
                "modality": "text",
                "licensing": "Apache-2.0",
                "authors": ["contributor"],
                "reward": {"range": [0, 1], "higher_is_better": True},
                "resources_server": "fixture",
                "agent_server": "simple_agent",
                "model_server": "policy_model",
                "datasets": [{"name": "fixture", "type": "example", "jsonl_fpath": "data.jsonl"}],
            }
        ),
        path,
    )


def test_parse_manifest_edits_uses_yaml_values_and_rejects_config_mirrors() -> None:
    edits = parse_manifest_edits(
        ["lifecycle=deprecated", "requires=[text-model, observability]", "reward.higher_is_better=false"]
    )
    assert edits[1].value == ["text-model", "observability"]
    assert edits[2].value is False
    with pytest.raises(ConfigError, match="Composition is authored in config"):
        parse_manifest_edits(["agent_server=other_agent"])
    with pytest.raises(ConfigError, match="assigned more than once"):
        parse_manifest_edits(["domain=math", "domain=code"])


def test_apply_manifest_edits_prevalidates_and_supports_dry_run(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    first = tmp_path / "environments" / "first" / "manifest.yaml"
    second = tmp_path / "environments" / "second" / "manifest.yaml"
    _write_manifest(first, name="first")
    _write_manifest(second, name="second")
    original = first.read_text()
    edits = parse_manifest_edits(["licensing=MIT", "requires=[text-model, observability]"])

    preview = apply_manifest_edits([first, second], edits, dry_run=True)
    assert preview.changed == (first.resolve(), second.resolve())
    assert first.read_text() == original

    result = apply_manifest_edits([first, second], edits)
    assert result.changed == (first.resolve(), second.resolve())
    assert load_manifest(first).licensing == "MIT"
    assert load_manifest(second).requires == ["text-model", "observability"]

    invalid = parse_manifest_edits(["reward.range=[2, 1]"])
    before = (first.read_text(), second.read_text())
    with pytest.raises(ConfigError, match="would make manifest.*invalid"):
        apply_manifest_edits([first, second], invalid)
    assert (first.read_text(), second.read_text()) == before


def test_apply_manifest_edits_rejects_manifest_symlink_outside_registry(tmp_path, monkeypatch) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    outside = tmp_path / "outside.yaml"
    _write_manifest(outside)
    original = outside.read_text()
    linked = checkout / "environments" / "fixture" / "manifest.yaml"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(outside)

    with pytest.raises(ConfigError, match="symbolic-link"):
        apply_manifest_edits([linked], parse_manifest_edits(["licensing=MIT"]))

    assert outside.read_text() == original
    assert linked.is_symlink()


def test_select_manifest_paths_filters_catalog_and_reports_unknown_names(tmp_path, monkeypatch) -> None:
    first = tmp_path / "environments" / "first" / "manifest.yaml"
    second = tmp_path / "environments" / "second" / "manifest.yaml"
    _write_manifest(first, name="first", domain="math")
    _write_manifest(second, name="second", domain="coding")
    entries = tuple(
        EnvironmentCatalogEntry(
            name=name,
            kind="environment",
            status="experimental",
            source="manifest",
            config_path=path.parent / "config.yaml",
            manifest_path=path,
            integration_profile="stock-loop",
            domain=domain,
        )
        for name, domain, path in (("first", "math", first), ("second", "coding", second))
    )
    catalog = EnvironmentCatalog(entries=entries, coverage=CatalogCoverage(2, 2, 0, 0))
    monkeypatch.setattr(manifest_edit, "discover_environment_catalog", lambda **_kwargs: catalog)

    assert select_manifest_paths(ManifestEditFilters(domain="math")) == (first,)
    with pytest.raises(ConfigError, match="Unknown manifest-backed.*missing"):
        select_manifest_paths(ManifestEditFilters(names=frozenset({"first", "missing"})))
