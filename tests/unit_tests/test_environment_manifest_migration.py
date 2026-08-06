# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import yaml
from scripts.migrate_environment_manifests import (
    RunnableUnit,
    discover_runnable_units,
    main,
    migrate_environment_manifests,
)

from nemo_gym.environment_inventory import MIGRATION_DRAFT_HEADER, tracked_migration_draft_paths
from nemo_gym.environment_manifest import ManifestError, dump_manifest, load_manifest
from nemo_gym.environment_scaffold import scaffold_environment


def _workload_config(*, agent: str = "simple_agent", driver: str | None = None) -> str:
    config = {
        "scorer": {
            "resources_servers": {
                "fixture": {
                    "entrypoint": "app.py",
                    "domain": "math",
                    "description": "Scores arithmetic responses from a text model",
                    "verified": True,
                    "verified_url": "https://example.test/verification",
                    "value": "Arithmetic evaluation",
                }
            }
        },
        "agent": {
            "responses_api_agents": {
                agent: {
                    "entrypoint": "app.py",
                    "resources_server": {"type": "resources_servers", "name": "scorer"},
                    "model_server": {"type": "responses_api_models", "name": "policy_model"},
                    "datasets": [
                        {
                            "name": "example",
                            "type": "example",
                            "jsonl_fpath": "environments/fixture/data/example.jsonl",
                        }
                    ],
                }
            }
        },
    }
    if driver:
        config["rollout_collection_driver"] = driver
    return yaml.safe_dump(config, sort_keys=False)


def test_migration_drafts_mirror_and_tracks_authored_todos_without_touching_config(tmp_path: Path) -> None:
    config_path = tmp_path / "environments" / "fixture" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    original_config = _workload_config()
    config_path.write_text(original_config)
    unit = RunnableUnit(
        name="fixture",
        kind="environment",
        config_path=config_path,
        manifest_path=config_path.parent / "manifest.yaml",
        registry="environments",
    )

    inventory = migrate_environment_manifests(
        tmp_path,
        inventory_path=Path("migration/inventory.json"),
        write=True,
        units=[unit],
    )

    draft = yaml.safe_load(unit.manifest_path.read_text())
    assert draft["name"] == "fixture"
    assert draft["integration_profile"] == "stock-loop"
    assert draft["domain"] == "math"
    assert draft["description"] == "Scores arithmetic responses from a text model"
    assert draft["resources_server"] == "fixture"
    assert draft["agent_server"] == "simple_agent"
    assert draft["model_server"] == "policy_model"
    assert draft["datasets"][0]["name"] == "example"
    assert draft["version"].startswith("TODO_REQUIRED")
    assert draft["authors"][0].startswith("TODO_REQUIRED")
    assert draft["requires"][0].startswith("TODO_REQUIRED")
    assert config_path.read_text() == original_config

    record = inventory["units"][0]
    assert record["status"] == "drafted"
    assert "draft_sha256" not in record
    assert "reward.range" in record["authored_todos"]
    assert record["legacy_metadata"] == {
        "description": "Scores arithmetic responses from a text model",
        "domain": "math",
        "value": "Arithmetic evaluation",
    }
    assert json.loads((tmp_path / "migration" / "inventory.json").read_text()) == inventory
    with pytest.raises(ManifestError, match="Invalid environment manifest"):
        load_manifest(unit.manifest_path)

    # Regenerating the inventory preserves draft status without rewriting the
    # partially authored file.
    before_draft = unit.manifest_path.read_bytes()
    regenerated = migrate_environment_manifests(
        tmp_path,
        inventory_path=Path("migration/inventory.json"),
        inventory_only=True,
        units=[unit],
    )
    assert regenerated == inventory
    assert unit.manifest_path.read_bytes() == before_draft
    assert tracked_migration_draft_paths(
        tmp_path,
        inventory_path=Path("migration/inventory.json"),
    ) == frozenset({unit.manifest_path.absolute()})


def test_migration_dry_run_is_non_mutating_and_exceptions_are_tracked(tmp_path: Path) -> None:
    config_path = tmp_path / "benchmarks" / "suite" / "configs" / "variant.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "agent": {
                    "responses_api_agents": {
                        "simple_agent": {
                            "datasets": [
                                {
                                    "name": "variant",
                                    "type": "benchmark",
                                    "jsonl_fpath": "data.jsonl",
                                }
                            ]
                        }
                    }
                }
            }
        )
    )

    units = discover_runnable_units(tmp_path)
    assert len(units) == 1
    assert units[0].blocker and "noncanonical" in units[0].blocker
    inventory = migrate_environment_manifests(tmp_path, write=False, units=units)

    assert inventory["summary"] == {
        "total": 1,
        "drafted": 0,
        "planned": 0,
        "already_manifest": 0,
        "exceptions": 1,
    }
    assert inventory["exceptions"][0]["name"] == "suite/configs/variant"
    assert not units[0].manifest_path.exists()
    assert not (tmp_path / "migration").exists()


def test_inventory_only_writes_inventory_without_manifest_drafts(tmp_path: Path) -> None:
    config_path = tmp_path / "environments" / "fixture" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(_workload_config())
    unit = RunnableUnit(
        name="fixture",
        kind="environment",
        config_path=config_path,
        manifest_path=config_path.parent / "manifest.yaml",
        registry="environments",
    )

    inventory = migrate_environment_manifests(
        tmp_path,
        inventory_path=Path("migration/inventory.json"),
        inventory_only=True,
        units=[unit],
    )

    assert inventory["summary"]["planned"] == 1
    assert inventory["summary"]["drafted"] == 0
    assert not unit.manifest_path.exists()
    assert json.loads((tmp_path / "migration" / "inventory.json").read_text()) == inventory


def test_write_cli_requires_and_drafts_one_exact_config(tmp_path: Path) -> None:
    for name in ("first", "second"):
        config_path = tmp_path / "environments" / name / "config.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(_workload_config())

    with pytest.raises(SystemExit):
        main(["--repo-root", str(tmp_path), "--write"])

    result = main(
        [
            "--repo-root",
            str(tmp_path),
            "--inventory",
            "migration/inventory.json",
            "--write",
            "--config",
            "environments/first/config.yaml",
        ]
    )

    assert result == 0
    assert (tmp_path / "environments" / "first" / "manifest.yaml").is_file()
    assert not (tmp_path / "environments" / "second" / "manifest.yaml").exists()
    inventory = json.loads((tmp_path / "migration" / "inventory.json").read_text())
    assert inventory["summary"] == {
        "total": 2,
        "drafted": 1,
        "planned": 1,
        "already_manifest": 0,
        "exceptions": 0,
    }


def test_benchmark_inventory_never_treats_adjacent_manifest_as_a_runnable_config(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmarks" / "fixture"
    benchmark.mkdir(parents=True)
    dataset = {"datasets": [{"name": "fixture", "type": "benchmark", "jsonl_fpath": "data.jsonl"}]}
    (benchmark / "config.yaml").write_text(yaml.safe_dump(dataset))
    (benchmark / "manifest.yaml").write_text(yaml.safe_dump(dataset))

    units = discover_runnable_units(tmp_path)

    assert [(unit.name, unit.config_path) for unit in units] == [("fixture", (benchmark / "config.yaml").resolve())]


def test_resources_server_inventory_refuses_ambiguous_runnable_configs(tmp_path: Path) -> None:
    configs = tmp_path / "resources_servers" / "fixture" / "configs"
    configs.mkdir(parents=True)
    (configs / "first.yaml").write_text(_workload_config())
    (configs / "second.yaml").write_text(_workload_config(agent="browsecomp_agent"))

    units = discover_runnable_units(tmp_path)

    assert len(units) == 2
    assert all(unit.name == "fixture" for unit in units)
    assert all(unit.blocker and "2 runnable configs" in unit.blocker for unit in units)
    inventory = migrate_environment_manifests(tmp_path, write=False, units=units)
    assert inventory["summary"]["exceptions"] == 2
    assert not (tmp_path / "resources_servers" / "fixture" / "manifest.yaml").exists()


def test_valid_manifest_is_not_classified_as_a_migration_draft(tmp_path: Path) -> None:
    scaffold_environment(kind="environment", name="complete", root=tmp_path)
    manifest_path = tmp_path / "environments" / "complete" / "manifest.yaml"
    manifest = load_manifest(manifest_path).model_copy(update={"authors": ["TODO_REQUIRED-contributor"]})
    manifest_path.write_text(f"{MIGRATION_DRAFT_HEADER}\n" + dump_manifest(manifest))
    inventory_path = tmp_path / "migration" / "environment-manifest-inventory.json"
    inventory_path.parent.mkdir()
    inventory_path.write_text(
        json.dumps(
            {
                "units": [
                    {
                        "status": "drafted",
                        "manifest_path": "environments/complete/manifest.yaml",
                    }
                ]
            }
        )
    )

    assert load_manifest(manifest_path).authors == ["TODO_REQUIRED-contributor"]
    assert tracked_migration_draft_paths(tmp_path) == frozenset()
