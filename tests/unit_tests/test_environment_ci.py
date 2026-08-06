# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

import yaml

from nemo_gym.environment_catalog import CatalogCoverage, EnvironmentCatalog, EnvironmentCatalogEntry
from nemo_gym.environment_ci import (
    ChangedFile,
    is_metadata_only_change,
    run_enforced_verifier_checks,
    run_environment_ci_gate,
)
from nemo_gym.environment_inventory import MIGRATION_DRAFT_HEADER, MIGRATION_INVENTORY_PATH
from nemo_gym.environment_manifest import load_manifest, write_manifest_json_schema
from nemo_gym.environment_scaffold import scaffold_environment
from nemo_gym.environment_versioning import LOCK_RELATIVE_PATH, check_or_record_version_lock
from nemo_gym.global_config import NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME
from nemo_gym.verifier_fixture import (
    DETERMINISM_ENV_VAR,
    HIGHER_IS_BETTER_ENV_VAR,
    REWARD_RANGE_ENV_VAR,
    UPDATE_EXPECTED_ENV_VAR,
)
from tests.unit_tests.onboarding_test_utils import complete_scaffold_fixture


def _schema(root: Path) -> None:
    path = root / "schemas" / "environment-manifest.schema.json"
    path.parent.mkdir(parents=True)
    write_manifest_json_schema(path)


def _scaffold_ready_environment(root: Path, name: str, *, profile: str = "stock-loop") -> None:
    scaffold_environment(kind="environment", name=name, profile=profile, root=root)
    manifest = load_manifest(root / "environments" / name / "manifest.yaml")
    complete_scaffold_fixture(
        root / "resources_servers" / name / "tests" / "verifier_cases.jsonl",
        manifest,
    )


def _select_inline_sandbox_provider(root: Path, name: str, provider: str) -> None:
    config_path = root / "environments" / name / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    agent_group = next(
        value["responses_api_agents"]
        for value in config.values()
        if isinstance(value, dict) and isinstance(value.get("responses_api_agents"), dict)
    )
    next(iter(agent_group.values()))["sandbox_provider"] = {provider: {}}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    manifest_path = config_path.with_name("manifest.yaml")
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["sandbox"] = provider
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))


def _manifest_entry(root: Path, name: str, kind: str = "environment") -> EnvironmentCatalogEntry:
    tree = "environments" if kind == "environment" else "benchmarks"
    manifest_path = root / tree / name / "manifest.yaml"
    manifest = load_manifest(manifest_path)
    return EnvironmentCatalogEntry(
        name=name,
        kind=kind,
        status="experimental",
        source="manifest",
        config_path=manifest_path.parent / "config.yaml",
        manifest_path=manifest_path,
        version=manifest.version,
    )


def _legacy_entry(
    root: Path,
    name: str,
    *,
    kind: str = "environment",
    config_path: Path | None = None,
) -> EnvironmentCatalogEntry:
    tree = "environments" if kind == "environment" else "benchmarks"
    return EnvironmentCatalogEntry(
        name=name,
        kind=kind,
        status="no-manifest",
        source="legacy",
        config_path=config_path or root / tree / name / "config.yaml",
        manifest_path=None,
    )


def _catalog(*entries: EnvironmentCatalogEntry) -> EnvironmentCatalog:
    manifested = sum(entry.source == "manifest" for entry in entries)
    return EnvironmentCatalog(
        entries=tuple(entries),
        coverage=CatalogCoverage(
            total=len(entries),
            with_manifest=manifested,
            without_manifest=len(entries) - manifested,
            invalid_manifests=0,
        ),
    )


def _record_entry_lock(root: Path, entry: EnvironmentCatalogEntry) -> None:
    unit = run_environment_ci_gate(root, catalog=_catalog(entry)).units[0]
    assert not unit.errors, unit.errors
    assert unit.composition_hash is not None
    check_or_record_version_lock(
        repo_root=root,
        manifest_path=entry.manifest_path,
        manifest=load_manifest(entry.manifest_path),
        composition_hash=unit.composition_hash,
    )


def _track_generated_draft(root: Path, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"{MIGRATION_DRAFT_HEADER}\n"
        "# This draft is intentionally invalid until every TODO_REQUIRED value is replaced.\n"
        "version: 'TODO_REQUIRED: semantic version'\n"
    )
    manifest_path.write_text(content)
    inventory_path = root / MIGRATION_INVENTORY_PATH
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "summary": {
                    "total": 1,
                    "drafted": 1,
                    "planned": 0,
                    "already_manifest": 0,
                    "exceptions": 0,
                },
                "units": [
                    {
                        "name": manifest_path.parent.name,
                        "kind": "environment",
                        "registry": "environments",
                        "config_path": manifest_path.with_name("config.yaml").relative_to(root).as_posix(),
                        "manifest_path": manifest_path.relative_to(root).as_posix(),
                        "status": "drafted",
                    }
                ],
                "exceptions": [],
            }
        )
        + "\n"
    )


def _write_bootstrap_inventory(
    root: Path,
    name: str,
    *,
    kind: str = "environment",
    registry: str = "environments",
    recorded_config_path: Path | None = None,
) -> str:
    config_path = root / registry / name / "config.yaml"
    inventory_config_path = recorded_config_path or config_path
    manifest_path = config_path.with_name("manifest.yaml")
    content = (
        json.dumps(
            {
                "schema_version": 1,
                "summary": {
                    "total": 1,
                    "drafted": 0,
                    "planned": 1,
                    "already_manifest": 0,
                    "exceptions": 0,
                },
                "units": [
                    {
                        "name": name,
                        "kind": kind,
                        "registry": registry,
                        "config_path": inventory_config_path.relative_to(root).as_posix(),
                        "manifest_path": manifest_path.relative_to(root).as_posix(),
                        "status": "planned",
                    }
                ],
                "exceptions": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    inventory_path = root / MIGRATION_INVENTORY_PATH
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(content)
    return content


def test_default_is_full_inventory_report_only_and_grandfathers_legacy(tmp_path: Path) -> None:
    _schema(tmp_path)
    missing_config = tmp_path / "environments" / "legacy" / "config.yaml"
    entry = EnvironmentCatalogEntry(
        name="legacy",
        kind="environment",
        status="no-manifest",
        source="legacy",
        config_path=missing_config,
        manifest_path=None,
    )

    report = run_environment_ci_gate(tmp_path, catalog=_catalog(entry))

    assert report.passed
    assert report.units[0].errors
    assert report.units[0].enforced is False
    assert report.to_dict()["summary"] == {
        "total": 1,
        "enforced": 0,
        "report_only": 1,
        "units_with_errors": 1,
        "enforced_errors": 0,
    }


def test_changed_manifest_runs_static_validation_and_scoring_fixture(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "fixture")
    entry = _manifest_entry(tmp_path, "fixture")
    _record_entry_lock(tmp_path, entry)
    change = ChangedFile(path=Path("environments/fixture/manifest.yaml"), status="A")

    passing = run_environment_ci_gate(
        tmp_path,
        changes=[change],
        enforce_changes=True,
        catalog=_catalog(entry),
    )
    assert passing.passed
    assert passing.units[0].enforced
    assert passing.units[0].composition_hash
    assert passing.units[0].errors == []

    fixture = tmp_path / "resources_servers" / "fixture" / "tests" / "verifier_cases.jsonl"
    lines = fixture.read_text().splitlines()
    fixture.unlink()
    missing = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=fixture.relative_to(tmp_path), status="D", before_content="\n".join(lines) + "\n")],
        enforce_changes=True,
        catalog=_catalog(entry),
    )
    assert not missing.passed
    assert any("No standard resources-server verifier fixture" in error for error in missing.units[0].errors)

    fixture.write_text("\n".join([*lines[:2], *lines[3:]]) + "\n")
    failing = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=fixture.relative_to(tmp_path), before_content="\n".join(lines) + "\n")],
        enforce_changes=True,
        catalog=_catalog(entry),
    )
    assert not failing.passed
    assert any("scoring contract" in error for error in failing.units[0].errors)


def test_changed_legacy_unit_requires_manifest_but_unchanged_legacy_does_not(tmp_path: Path) -> None:
    _schema(tmp_path)
    config = tmp_path / "environments" / "legacy" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n")
    entry = EnvironmentCatalogEntry(
        name="legacy",
        kind="environment",
        status="no-manifest",
        source="legacy",
        config_path=config,
        manifest_path=None,
    )

    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=config.relative_to(tmp_path), before_content="{}\n")],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not report.passed
    assert report.units[0].enforced
    assert any("requires a complete manifest" in error for error in report.units[0].errors)


def test_new_migration_inventory_bootstrap_grandfathers_and_checks_listed_base_legacy_unit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "bootstrap_legacy")
    manifest_path = tmp_path / "environments" / "bootstrap_legacy" / "manifest.yaml"
    manifest_path.unlink()
    config_path = manifest_path.with_name("config.yaml")
    before = config_path.read_text()
    config_path.write_text(before + "\n# inventory bootstrap edit\n")
    _write_bootstrap_inventory(tmp_path, "bootstrap_legacy")

    report = run_environment_ci_gate(
        tmp_path,
        changes=[
            ChangedFile(path=MIGRATION_INVENTORY_PATH, status="A"),
            ChangedFile(path=config_path.relative_to(tmp_path), status="M", before_content=before),
        ],
        enforce_changes=True,
        catalog=_catalog(_legacy_entry(tmp_path, "bootstrap_legacy")),
    )

    unit = report.units[0]
    assert report.passed
    assert unit.enforced
    assert unit.bootstrap_legacy_exempt
    assert not any("requires a complete manifest" in error for error in unit.errors)
    assert any("One-time migration-inventory bootstrap" in warning for warning in unit.warnings)

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="fixture passed", stderr="")

    monkeypatch.setattr("nemo_gym.environment_ci.subprocess.run", fake_run)
    run_enforced_verifier_checks(report, python_executable="/safe/python")

    assert len(calls) == 1
    assert calls[0][0][0:3] == ["/safe/python", "-m", "nemo_gym.verifier_ci_harness"]
    assert len(unit.verifier_checks) == 1 and unit.verifier_checks[0].passed
    assert report.passed


def test_migration_inventory_bootstrap_never_exempts_a_new_unit(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "bootstrap_new")
    manifest_path = tmp_path / "environments" / "bootstrap_new" / "manifest.yaml"
    manifest_path.unlink()
    config_path = manifest_path.with_name("config.yaml")
    _write_bootstrap_inventory(tmp_path, "bootstrap_new")

    report = run_environment_ci_gate(
        tmp_path,
        changes=[
            ChangedFile(path=MIGRATION_INVENTORY_PATH, status="A"),
            ChangedFile(path=config_path.relative_to(tmp_path), status="A"),
        ],
        enforce_changes=True,
        catalog=_catalog(_legacy_entry(tmp_path, "bootstrap_new")),
    )

    assert not report.passed
    assert any("requires a complete manifest" in error for error in report.units[0].errors)


def test_migration_inventory_bootstrap_rejects_new_symlinked_config_alias(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "bootstrap_alias")
    manifest_path = tmp_path / "environments" / "bootstrap_alias" / "manifest.yaml"
    manifest_path.unlink()
    config_path = manifest_path.with_name("config.yaml")
    target_path = tmp_path / "shared" / "bootstrap_alias.yaml"
    target_path.parent.mkdir()
    target_path.write_text(config_path.read_text())
    config_path.unlink()
    config_path.symlink_to(target_path)
    _write_bootstrap_inventory(
        tmp_path,
        "bootstrap_alias",
        recorded_config_path=target_path,
    )

    report = run_environment_ci_gate(
        tmp_path,
        changes=[
            ChangedFile(path=MIGRATION_INVENTORY_PATH, status="A"),
            ChangedFile(path=config_path.relative_to(tmp_path), status="A"),
        ],
        enforce_changes=True,
        catalog=_catalog(_legacy_entry(tmp_path, "bootstrap_alias")),
    )

    unit = report.units[0]
    assert not report.passed
    assert not unit.bootstrap_legacy_exempt
    assert any("requires a complete manifest" in error for error in unit.errors)


def test_migration_inventory_bootstrap_rejects_newly_runnable_existing_config(tmp_path: Path) -> None:
    _schema(tmp_path)
    scaffold_environment(
        kind="benchmark",
        name="bootstrap_transition",
        metadata={"canonical_split": "test"},
        root=tmp_path,
    )
    manifest_path = tmp_path / "benchmarks" / "bootstrap_transition" / "manifest.yaml"
    manifest = load_manifest(manifest_path)
    complete_scaffold_fixture(
        tmp_path / "resources_servers" / "bootstrap_transition" / "tests" / "verifier_cases.jsonl",
        manifest,
    )
    manifest_path.unlink()
    config_path = manifest_path.with_name("config.yaml")
    _write_bootstrap_inventory(
        tmp_path,
        "bootstrap_transition",
        kind="benchmark",
        registry="benchmarks",
    )

    report = run_environment_ci_gate(
        tmp_path,
        changes=[
            ChangedFile(path=MIGRATION_INVENTORY_PATH, status="A"),
            ChangedFile(
                path=config_path.relative_to(tmp_path),
                status="M",
                before_content="description: previously not a runnable benchmark\n",
            ),
        ],
        enforce_changes=True,
        catalog=_catalog(_legacy_entry(tmp_path, "bootstrap_transition", kind="benchmark")),
    )

    assert not report.passed
    assert any("requires a complete manifest" in error for error in report.units[0].errors)


def test_invalid_or_preexisting_migration_inventory_cannot_bootstrap_legacy_changes(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "bootstrap_guard")
    manifest_path = tmp_path / "environments" / "bootstrap_guard" / "manifest.yaml"
    manifest_path.unlink()
    config_path = manifest_path.with_name("config.yaml")
    before_config = config_path.read_text()
    config_path.write_text(before_config + "\n# guarded edit\n")
    valid_inventory = _write_bootstrap_inventory(tmp_path, "bootstrap_guard")
    inventory_path = tmp_path / MIGRATION_INVENTORY_PATH
    malformed = json.loads(valid_inventory)
    malformed["summary"]["total"] = 2
    inventory_path.write_text(json.dumps(malformed) + "\n")

    invalid = run_environment_ci_gate(
        tmp_path,
        changes=[
            ChangedFile(path=MIGRATION_INVENTORY_PATH, status="A"),
            ChangedFile(path=config_path.relative_to(tmp_path), status="M", before_content=before_config),
        ],
        enforce_changes=True,
        catalog=_catalog(_legacy_entry(tmp_path, "bootstrap_guard")),
    )
    assert not invalid.passed
    assert any("requires a complete manifest" in error for error in invalid.units[0].errors)

    inventory_path.write_text(valid_inventory)
    preexisting = run_environment_ci_gate(
        tmp_path,
        changes=[
            ChangedFile(
                path=MIGRATION_INVENTORY_PATH,
                status="M",
                before_content=valid_inventory,
            ),
            ChangedFile(path=config_path.relative_to(tmp_path), status="M", before_content=before_config),
        ],
        enforce_changes=True,
        catalog=_catalog(_legacy_entry(tmp_path, "bootstrap_guard")),
    )
    assert not preexisting.passed
    assert any("requires a complete manifest" in error for error in preexisting.units[0].errors)

    readded = run_environment_ci_gate(
        tmp_path,
        changes=[
            ChangedFile(
                path=MIGRATION_INVENTORY_PATH,
                status="A",
                before_content=valid_inventory,
            ),
            ChangedFile(path=config_path.relative_to(tmp_path), status="M", before_content=before_config),
        ],
        enforce_changes=True,
        catalog=_catalog(_legacy_entry(tmp_path, "bootstrap_guard")),
    )
    assert not readded.passed
    assert any("requires a complete manifest" in error for error in readded.units[0].errors)


def test_direct_legacy_change_requires_manifest_even_with_real_dependency_fanout(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "legacy_direct")
    manifest_path = tmp_path / "environments" / "legacy_direct" / "manifest.yaml"
    manifest_path.unlink()
    config_path = manifest_path.with_name("config.yaml")
    before_config = config_path.read_text()
    config_path.write_text(before_config + "\n# direct recipe change\n")
    app_path = tmp_path / "resources_servers" / "legacy_direct" / "app.py"
    before_app = app_path.read_text()
    app_path.write_text(before_app + "\n# shared scorer change\n")

    report = run_environment_ci_gate(
        tmp_path,
        changes=[
            ChangedFile(path=config_path.relative_to(tmp_path), before_content=before_config),
            ChangedFile(path=app_path.relative_to(tmp_path), before_content=before_app),
        ],
        enforce_changes=True,
        catalog=_catalog(_legacy_entry(tmp_path, "legacy_direct")),
    )

    unit = report.units[0]
    assert any(reason.startswith("changed-unit:") for reason in unit.reasons)
    assert any(reason.startswith("dependent:resources_servers/legacy_direct") for reason in unit.reasons)
    assert any("requires a complete manifest" in error for error in unit.errors)
    assert not report.passed


def test_direct_legacy_recipe_dependency_does_not_self_fan_out(tmp_path: Path) -> None:
    _schema(tmp_path)
    scaffold_environment(
        kind="benchmark",
        name="legacy_self",
        metadata={"canonical_split": "test"},
        root=tmp_path,
    )
    manifest_path = tmp_path / "benchmarks" / "legacy_self" / "manifest.yaml"
    manifest_path.unlink()
    config_path = manifest_path.with_name("config.yaml")
    before = config_path.read_text()
    config_path.write_text(before + "\n# direct benchmark change\n")

    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=config_path.relative_to(tmp_path), before_content=before)],
        enforce_changes=True,
        catalog=_catalog(_legacy_entry(tmp_path, "legacy_self", kind="benchmark")),
    )

    unit = report.units[0]
    assert ("benchmarks", "legacy_self") in unit.dependencies
    assert any(reason.startswith("changed-unit:") for reason in unit.reasons)
    assert not any(reason.startswith("dependent:benchmarks/legacy_self") for reason in unit.reasons)
    assert any("requires a complete manifest" in error for error in unit.errors)


def test_tracked_migration_draft_addition_is_enforced(
    tmp_path: Path,
) -> None:
    _schema(tmp_path)
    config = tmp_path / "environments" / "legacy" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n")
    manifest = config.with_name("manifest.yaml")
    _track_generated_draft(tmp_path, manifest)
    entry = EnvironmentCatalogEntry(
        name="legacy",
        kind="environment",
        status="no-manifest",
        source="legacy",
        config_path=config,
        manifest_path=None,
    )
    draft_change = ChangedFile(path=manifest.relative_to(tmp_path), status="A")

    draft_only = run_environment_ci_gate(
        tmp_path,
        changes=[draft_change],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not draft_only.passed
    assert draft_only.units[0].enforced
    assert any("requires a complete manifest" in error for error in draft_only.units[0].errors)
    assert draft_only.changes == (draft_change,)

    manifest.write_text(manifest.read_text() + "description: manually edited\n")
    edited_add = run_environment_ci_gate(
        tmp_path,
        changes=[draft_change],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not edited_add.passed
    assert edited_add.units[0].enforced
    assert any("requires a complete manifest" in error for error in edited_add.units[0].errors)

    modified_draft = run_environment_ci_gate(
        tmp_path,
        changes=[
            ChangedFile(
                path=manifest.relative_to(tmp_path),
                status="M",
                before_content=manifest.read_text(),
            )
        ],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not modified_draft.passed
    assert modified_draft.units[0].enforced
    assert any("requires a complete manifest" in error for error in modified_draft.units[0].errors)

    for behavioral_path in (config, config.with_name("app.py")):
        behavioral_path.touch(exist_ok=True)
        with_behavior_change = run_environment_ci_gate(
            tmp_path,
            changes=[
                draft_change,
                ChangedFile(path=behavioral_path.relative_to(tmp_path), status="M", before_content=""),
            ],
            enforce_changes=True,
            catalog=_catalog(entry),
        )
        assert not with_behavior_change.passed
        assert with_behavior_change.units[0].enforced
        assert any("requires a complete manifest" in error for error in with_behavior_change.units[0].errors)


def test_untracked_generated_todo_manifest_change_is_enforced(tmp_path: Path) -> None:
    _schema(tmp_path)
    config = tmp_path / "environments" / "legacy" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n")
    manifest = config.with_name("manifest.yaml")
    manifest.write_text(f"{MIGRATION_DRAFT_HEADER}\nversion: TODO_REQUIRED\n")
    entry = EnvironmentCatalogEntry(
        name="legacy",
        kind="environment",
        status="no-manifest",
        source="legacy",
        config_path=config,
        manifest_path=None,
    )

    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=manifest.relative_to(tmp_path), status="A")],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not report.passed
    assert report.units[0].enforced
    assert any("requires a complete manifest" in error for error in report.units[0].errors)


def test_completed_manifest_cannot_downgrade_into_migration_exemption(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "completed")
    manifest = tmp_path / "environments" / "completed" / "manifest.yaml"
    config = manifest.with_name("config.yaml")
    before_manifest = manifest.read_text()
    before_inventory = json.dumps(
        {
            "schema_version": 1,
            "summary": {
                "total": 1,
                "drafted": 0,
                "planned": 0,
                "already_manifest": 1,
                "exceptions": 0,
            },
            "units": [
                {
                    "name": "completed",
                    "kind": "environment",
                    "registry": "environments",
                    "config_path": config.relative_to(tmp_path).as_posix(),
                    "manifest_path": manifest.relative_to(tmp_path).as_posix(),
                    "status": "already_manifest",
                }
            ],
            "exceptions": [],
        }
    )

    _track_generated_draft(tmp_path, manifest)
    report = run_environment_ci_gate(
        tmp_path,
        changes=[
            ChangedFile(
                path=manifest.relative_to(tmp_path),
                status="M",
                before_content=before_manifest,
            ),
            ChangedFile(
                path=MIGRATION_INVENTORY_PATH,
                status="M",
                before_content=before_inventory,
            ),
        ],
        enforce_changes=True,
        catalog=_catalog(
            EnvironmentCatalogEntry(
                name="completed",
                kind="environment",
                status="no-manifest",
                source="legacy",
                config_path=config,
                manifest_path=None,
            )
        ),
    )

    assert not report.passed
    assert report.units[0].enforced
    assert any("requires a complete manifest" in error for error in report.units[0].errors)


def test_changed_unit_enforces_profile_capability_and_dataset_contracts(tmp_path: Path) -> None:
    _schema(tmp_path)

    _scaffold_ready_environment(tmp_path, "profile_case")
    profile_manifest = tmp_path / "environments" / "profile_case" / "manifest.yaml"
    profile_data = yaml.safe_load(profile_manifest.read_text())
    profile_data["integration_profile"] = "measured-loop"
    profile_manifest.write_text(yaml.safe_dump(profile_data, sort_keys=False))
    profile_report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=profile_manifest.relative_to(tmp_path))],
        enforce_changes=True,
        catalog=_catalog(_manifest_entry(tmp_path, "profile_case")),
    )
    assert not profile_report.passed
    assert any("config classifies as 'stock-loop'" in error for error in profile_report.units[0].errors)

    _scaffold_ready_environment(tmp_path, "capability_case")
    capability_config = tmp_path / "environments" / "capability_case" / "config.yaml"
    capability_data = yaml.safe_load(capability_config.read_text())
    agent_instance = capability_data["capability_case_agent"]["responses_api_agents"]["simple_agent"]
    agent_instance["requires"] = ["missing-capability"]
    capability_config.write_text(yaml.safe_dump(capability_data, sort_keys=False))
    capability_report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=capability_config.relative_to(tmp_path))],
        enforce_changes=True,
        catalog=_catalog(_manifest_entry(tmp_path, "capability_case")),
    )
    assert not capability_report.passed
    assert any("missing-capability" in error for error in capability_report.units[0].errors)

    _scaffold_ready_environment(tmp_path, "dataset_case")
    dataset = tmp_path / "environments" / "dataset_case" / "data" / "example.jsonl"
    dataset.write_text('{"question":"not materialized"}\n')
    dataset_report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=dataset.relative_to(tmp_path))],
        enforce_changes=True,
        catalog=_catalog(_manifest_entry(tmp_path, "dataset_case")),
    )
    assert not dataset_report.passed
    assert any("responses_create_params.input" in error for error in dataset_report.units[0].errors)


def test_shared_component_fans_out_but_metadata_only_diff_is_exempt(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "consumer")
    entry = _manifest_entry(tmp_path, "consumer")
    _record_entry_lock(tmp_path, entry)
    component_config = tmp_path / "resources_servers" / "consumer" / "configs" / "consumer.yaml"
    before = component_config.read_text()
    parsed = yaml.safe_load(before)
    parsed["consumer_resources_server"]["resources_servers"]["consumer"]["description"] = (
        "A metadata-only description that does not affect scoring"
    )
    component_config.write_text(yaml.safe_dump(parsed, sort_keys=False))
    metadata_change = ChangedFile(
        path=component_config.relative_to(tmp_path),
        before_content=before,
    )
    assert is_metadata_only_change(tmp_path, metadata_change)

    metadata_report = run_environment_ci_gate(
        tmp_path,
        changes=[metadata_change],
        enforce_changes=True,
        catalog=_catalog(entry),
    )
    assert metadata_report.passed
    assert metadata_report.to_dict()["summary"]["enforced"] == 0

    app = tmp_path / "resources_servers" / "consumer" / "app.py"
    code_change = ChangedFile(path=app.relative_to(tmp_path), before_content=app.read_text())
    app.write_text(app.read_text() + "\n# score-affecting implementation edit\n")
    dependency_report = run_environment_ci_gate(
        tmp_path,
        changes=[code_change],
        enforce_changes=True,
        catalog=_catalog(entry),
    )
    assert not dependency_report.passed
    assert dependency_report.units[0].enforced
    assert any(
        reason.startswith("dependent:resources_servers/consumer") for reason in dependency_report.units[0].reasons
    )
    assert any("locked sha256" in violation for violation in dependency_report.lock_violations)


def test_shared_component_change_fans_out_to_legacy_dependents(tmp_path: Path) -> None:
    _schema(tmp_path)
    scaffold_environment(kind="environment", name="legacy_consumer", root=tmp_path)
    manifest_path = tmp_path / "environments" / "legacy_consumer" / "manifest.yaml"
    manifest_path.unlink()
    entry = EnvironmentCatalogEntry(
        name="legacy_consumer",
        kind="environment",
        status="no-manifest",
        source="legacy",
        config_path=manifest_path.with_name("config.yaml"),
        manifest_path=None,
    )
    app = tmp_path / "resources_servers" / "legacy_consumer" / "app.py"
    before = app.read_text()
    app.write_text(before + "\n# scoring change\n")

    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=app.relative_to(tmp_path), before_content=before)],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert report.units[0].enforced
    assert any(reason.startswith("dependent:resources_servers/legacy_consumer") for reason in report.units[0].reasons)


def test_stock_rollout_source_fans_out_only_to_manifested_workloads_that_use_it(tmp_path: Path) -> None:
    _schema(tmp_path)
    profiles = {
        "stock_consumer": "stock-loop",
        "measured_consumer": "measured-loop",
        "custom_consumer": "custom-driver",
        "external_consumer": "external-loop",
    }
    entries = []
    for name, profile in profiles.items():
        _scaffold_ready_environment(tmp_path, name, profile=profile)
        entries.append(_manifest_entry(tmp_path, name))

    change = ChangedFile(path=Path("nemo_gym/rollout_collection.py"), before_content="# old stock loop\n")
    report = run_environment_ci_gate(
        tmp_path,
        changes=[change],
        enforce_changes=True,
        catalog=_catalog(*entries),
    )

    enforced = {unit.name for unit in report.units if unit.enforced}
    assert enforced == {"stock_consumer", "measured_consumer"}
    for unit in report.units:
        if unit.name in enforced:
            assert ("nemo_gym", "rollout_collection.py") in unit.dependencies
            assert unit.reasons == {"dependent:nemo_gym/rollout_collection.py (nemo_gym/rollout_collection.py)"}
        else:
            assert ("nemo_gym", "rollout_collection.py") not in unit.dependencies

    unrelated = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=Path("nemo_gym/environment_catalog.py"), before_content="# old core file\n")],
        enforce_changes=True,
        catalog=_catalog(*entries),
    )
    assert not any(unit.enforced for unit in unrelated.units)


def test_selected_sandbox_provider_package_fans_out_once_per_provider(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "sandbox_consumer")
    _select_inline_sandbox_provider(tmp_path, "sandbox_consumer", "docker")
    entry = _manifest_entry(tmp_path, "sandbox_consumer")
    changes = [
        ChangedFile(path=Path("nemo_gym/sandbox/providers/docker/provider.py"), before_content="# old\n"),
        ChangedFile(path=Path("nemo_gym/sandbox/providers/docker/__init__.py"), before_content="# old\n"),
        ChangedFile(path=Path("nemo_gym/sandbox/providers/enroot/provider.py"), before_content="# old\n"),
        ChangedFile(path=Path("nemo_gym/sandbox/providers/registry.py"), before_content="# old\n"),
    ]

    report = run_environment_ci_gate(
        tmp_path,
        changes=changes,
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    unit = report.units[0]
    assert unit.errors == []
    assert unit.enforced
    assert ("nemo_gym/sandbox/providers", "docker") in unit.dependencies
    dependent_reasons = [reason for reason in unit.reasons if reason.startswith("dependent:")]
    assert dependent_reasons == [
        "dependent:nemo_gym/sandbox/providers/docker "
        "(nemo_gym/sandbox/providers/docker/__init__.py, nemo_gym/sandbox/providers/docker/provider.py)"
    ]


def test_sandbox_provider_fanout_preserves_dependency_only_legacy_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "legacy_sandbox")
    _select_inline_sandbox_provider(tmp_path, "legacy_sandbox", "docker")
    manifest_path = tmp_path / "environments" / "legacy_sandbox" / "manifest.yaml"
    manifest_path.unlink()
    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=Path("nemo_gym/sandbox/providers/docker/provider.py"), before_content="# old\n")],
        enforce_changes=True,
        catalog=_catalog(_legacy_entry(tmp_path, "legacy_sandbox")),
    )

    unit = report.units[0]
    assert unit.enforced
    assert not any(reason.startswith("changed-unit:") for reason in unit.reasons)
    assert any(reason.startswith("dependent:nemo_gym/sandbox/providers/docker") for reason in unit.reasons)

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="fixture passed", stderr="")

    monkeypatch.setattr("nemo_gym.environment_ci.subprocess.run", fake_run)
    run_enforced_verifier_checks(report, python_executable="/safe/python")

    assert len(calls) == 1
    assert calls[0][0][0:3] == ["/safe/python", "-m", "nemo_gym.verifier_ci_harness"]
    assert len(unit.verifier_checks) == 1 and unit.verifier_checks[0].passed


def test_dependency_only_legacy_dependents_run_exact_fixture_once_and_each_record_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "legacy_fixture")
    manifest_path = tmp_path / "environments" / "legacy_fixture" / "manifest.yaml"
    manifest_path.unlink()
    config_path = manifest_path.with_name("config.yaml")
    app_path = tmp_path / "resources_servers" / "legacy_fixture" / "app.py"
    before = app_path.read_text()
    app_path.write_text(before + "\n# shared scorer change\n")
    entries = (
        _legacy_entry(tmp_path, "consumer_a", config_path=config_path),
        _legacy_entry(tmp_path, "consumer_b", config_path=config_path),
    )
    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=app_path.relative_to(tmp_path), before_content=before)],
        enforce_changes=True,
        catalog=_catalog(*entries),
    )
    assert all(unit.enforced for unit in report.units)
    assert all(not any(reason.startswith("changed-unit:") for reason in unit.reasons) for unit in report.units)

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="fixture passed", stderr="")

    monkeypatch.setattr("nemo_gym.environment_ci.subprocess.run", fake_run)
    run_enforced_verifier_checks(report, python_executable="/safe/python")

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[0:3] == ["/safe/python", "-m", "nemo_gym.verifier_ci_harness"]
    assert json.loads(kwargs["input"])["entrypoint"] == "app.py"
    assert REWARD_RANGE_ENV_VAR not in kwargs["env"]
    assert HIGHER_IS_BETTER_ENV_VAR not in kwargs["env"]
    assert DETERMINISM_ENV_VAR not in kwargs["env"]
    assert all(len(unit.verifier_checks) == 1 and unit.verifier_checks[0].passed for unit in report.units)
    assert report.passed


def test_dependency_only_legacy_unit_falls_back_to_component_pytest_with_exact_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "legacy_pytest")
    manifest_path = tmp_path / "environments" / "legacy_pytest" / "manifest.yaml"
    manifest_path.unlink()
    component_dir = tmp_path / "resources_servers" / "legacy_pytest"
    (component_dir / "tests" / "verifier_cases.jsonl").unlink()
    app_path = component_dir / "app.py"
    before = app_path.read_text()
    app_path.write_text(before + "\n# shared scorer change\n")
    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=app_path.relative_to(tmp_path), before_content=before)],
        enforce_changes=True,
        catalog=_catalog(_legacy_entry(tmp_path, "legacy_pytest")),
    )

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="component tests passed", stderr="")

    monkeypatch.setattr("nemo_gym.environment_ci.subprocess.run", fake_run)
    run_enforced_verifier_checks(report, python_executable="/safe/python")

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == ["/safe/python", "-m", "pytest", "tests/test_app.py"]
    assert kwargs["cwd"] == component_dir
    exact_config = yaml.safe_load(kwargs["env"][NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME])
    assert exact_config["legacy_pytest_resources_server"]["resources_servers"]["legacy_pytest"]["entrypoint"] == (
        "app.py"
    )
    assert report.units[0].verifier_checks[0].node == "pytest"
    assert report.units[0].verifier_checks[0].passed
    assert report.passed


def test_dependency_only_legacy_unit_fails_when_resources_server_is_ambiguous(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "legacy_ambiguous")
    manifest_path = tmp_path / "environments" / "legacy_ambiguous" / "manifest.yaml"
    manifest_path.unlink()
    component_config = tmp_path / "resources_servers" / "legacy_ambiguous" / "configs" / "legacy_ambiguous.yaml"
    config = yaml.safe_load(component_config.read_text())
    resources_instance = config["legacy_ambiguous_resources_server"]
    config["second_resources_server"] = json.loads(json.dumps(resources_instance))
    component_config.write_text(yaml.safe_dump(config, sort_keys=False))
    app_path = tmp_path / "resources_servers" / "legacy_ambiguous" / "app.py"
    before = app_path.read_text()
    app_path.write_text(before + "\n# shared scorer change\n")
    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=app_path.relative_to(tmp_path), before_content=before)],
        enforce_changes=True,
        catalog=_catalog(_legacy_entry(tmp_path, "legacy_ambiguous")),
    )
    assert report.units[0].enforced

    def unexpected_run(*args, **kwargs):
        raise AssertionError("ambiguous legacy verifier must fail before execution")

    monkeypatch.setattr("nemo_gym.environment_ci.subprocess.run", unexpected_run)
    run_enforced_verifier_checks(report, python_executable="/safe/python")

    check = report.units[0].verifier_checks[0]
    assert not check.passed
    assert "exactly one Resources Server" in check.output
    assert any("was not checked" in error for error in report.units[0].errors)
    assert not report.passed


def test_dependency_only_legacy_unit_fails_when_no_local_check_exists(tmp_path: Path, monkeypatch) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "legacy_unchecked")
    manifest_path = tmp_path / "environments" / "legacy_unchecked" / "manifest.yaml"
    manifest_path.unlink()
    component_dir = tmp_path / "resources_servers" / "legacy_unchecked"
    (component_dir / "tests" / "verifier_cases.jsonl").unlink()
    (component_dir / "tests" / "test_app.py").unlink()
    app_path = component_dir / "app.py"
    before = app_path.read_text()
    app_path.write_text(before + "\n# shared scorer change\n")
    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=app_path.relative_to(tmp_path), before_content=before)],
        enforce_changes=True,
        catalog=_catalog(_legacy_entry(tmp_path, "legacy_unchecked")),
    )

    def unexpected_run(*args, **kwargs):
        raise AssertionError("unchecked legacy verifier must fail before execution")

    monkeypatch.setattr("nemo_gym.environment_ci.subprocess.run", unexpected_run)
    run_enforced_verifier_checks(report, python_executable="/safe/python")

    check = report.units[0].verifier_checks[0]
    assert not check.passed
    assert "neither the canonical fixture" in check.output
    assert any("was not checked" in error for error in report.units[0].errors)
    assert not report.passed


def test_affected_shared_component_runs_exact_manifest_bound_verifier_check(tmp_path: Path, monkeypatch) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "provider")

    # Implementation names need not equal their component directory. The fan-out
    # and executable target must follow the defining config back to `provider`.
    component_config = tmp_path / "resources_servers" / "provider" / "configs" / "provider.yaml"
    component_data = yaml.safe_load(component_config.read_text())
    resources = component_data["provider_resources_server"]["resources_servers"]
    resources["scorer_impl"] = resources.pop("provider")
    component_config.write_text(yaml.safe_dump(component_data, sort_keys=False))
    manifest_path = tmp_path / "environments" / "provider" / "manifest.yaml"
    manifest_data = yaml.safe_load(manifest_path.read_text())
    manifest_data["resources_server"] = "scorer_impl"
    manifest_path.write_text(yaml.safe_dump(manifest_data, sort_keys=False))

    entry = _manifest_entry(tmp_path, "provider")
    _record_entry_lock(tmp_path, entry)
    test_path = tmp_path / "resources_servers" / "provider" / "tests" / "test_app.py"
    before_test = test_path.read_text()
    test_path.write_text(before_test + "\n# shared scorer test maintenance\n")
    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=test_path.relative_to(tmp_path), before_content=before_test)],
        enforce_changes=True,
        catalog=_catalog(entry),
    )
    assert report.passed
    assert ("resources_servers", "provider") in report.units[0].dependencies
    assert any(reason.startswith("dependent:resources_servers/provider") for reason in report.units[0].reasons)

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="one passed", stderr="")

    monkeypatch.setattr("nemo_gym.environment_ci.subprocess.run", fake_run)
    run_enforced_verifier_checks(report, python_executable="/safe/python")

    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command == [
        "/safe/python",
        "-m",
        "nemo_gym.verifier_ci_harness",
        "--component-dir",
        str(tmp_path / "resources_servers" / "provider"),
        "--entrypoint",
        "app.py",
        "--instance-name",
        "provider_resources_server",
        "--fixture",
        str(tmp_path / "resources_servers" / "provider" / "tests" / "verifier_cases.jsonl"),
    ]
    assert kwargs["cwd"] == tmp_path
    assert "shell" not in kwargs
    assert json.loads(kwargs["input"])["entrypoint"] == "app.py"
    assert kwargs["env"][UPDATE_EXPECTED_ENV_VAR] == "0"
    assert kwargs["env"][REWARD_RANGE_ENV_VAR] == "[0.0,1.0]"
    assert kwargs["env"][HIGHER_IS_BETTER_ENV_VAR] == "true"
    assert kwargs["env"][DETERMINISM_ENV_VAR] == "unknown"
    assert report.units[0].verifier_checks[0].passed
    assert report.passed


def test_verifier_check_uses_component_dependency_environment(tmp_path: Path, monkeypatch) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "isolated_dependencies")
    component_dir = tmp_path / "resources_servers" / "isolated_dependencies"
    requirements_path = component_dir / "requirements.txt"
    requirements_path.write_text(requirements_path.read_text() + "component-only-verifier-package==1\n")

    entry = _manifest_entry(tmp_path, "isolated_dependencies")
    _record_entry_lock(tmp_path, entry)
    test_path = component_dir / "tests" / "test_app.py"
    before_test = test_path.read_text()
    test_path.write_text(before_test + "\n# exercise dependency-isolated CI\n")
    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=test_path.relative_to(tmp_path), before_content=before_test)],
        enforce_changes=True,
        catalog=_catalog(entry),
    )
    assert report.passed

    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["uv", "venv"]:
            python_path = component_dir / ".venv" / "bin" / "python"
            python_path.parent.mkdir(parents=True)
            python_path.write_text("")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("nemo_gym.environment_ci.subprocess.run", fake_run)
    run_enforced_verifier_checks(report)

    assert len(calls) == 3
    assert calls[0][0][:2] == ["uv", "venv"]
    assert calls[0][1]["cwd"] == component_dir
    assert calls[1][0] == [
        "uv",
        "pip",
        "install",
        "--python",
        str(component_dir / ".venv" / "bin" / "python"),
        "-r",
        str(requirements_path),
    ]
    assert calls[1][1]["cwd"] == component_dir
    assert calls[2][0][0] == str(component_dir / ".venv" / "bin" / "python")
    assert calls[2][0][1:3] == ["-m", "nemo_gym.verifier_ci_harness"]
    assert report.units[0].verifier_checks[0].passed
    assert report.passed


def test_offline_verifier_execution_catches_scorer_drift_static_fixture_validation_misses(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "drifted_scorer")
    entry = _manifest_entry(tmp_path, "drifted_scorer")
    app_path = tmp_path / "resources_servers" / "drifted_scorer" / "app.py"
    original = app_path.read_text()
    changed = original.replace(
        "reward = 1.0 if output_text == body.expected_answer else 0.0",
        "reward = 0.0",
    )
    assert changed != original
    app_path.write_text(changed)
    # This was the former trust boundary. A component author could replace the
    # selected pytest node with this no-op and make a broken scorer pass CI.
    # The repository-owned harness must ignore it and exercise app.py directly.
    component_test = tmp_path / "resources_servers" / "drifted_scorer" / "tests" / "test_app.py"
    component_test.write_text("def test_verifier_fixture():\n    pass\n")

    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=app_path.relative_to(tmp_path), before_content=original)],
        enforce_changes=True,
        catalog=_catalog(entry),
    )
    assert report.units[0].errors == []

    fixture_path = tmp_path / "resources_servers" / "drifted_scorer" / "tests" / "verifier_cases.jsonl"
    fixture_before = fixture_path.read_bytes()
    run_enforced_verifier_checks(report, python_executable=sys.executable, timeout_seconds=30)

    check = report.units[0].verifier_checks[0]
    assert not check.passed
    assert "expected reward 1" in check.output
    assert any("Offline verifier scorer check failed" in error for error in report.units[0].errors)
    assert fixture_path.read_bytes() == fixture_before
    assert not report.passed


def test_report_only_gate_never_executes_verifier_code(tmp_path: Path, monkeypatch) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "report_only")
    report = run_environment_ci_gate(tmp_path, catalog=_catalog(_manifest_entry(tmp_path, "report_only")))

    def unexpected_run(*args, **kwargs):
        raise AssertionError("report-only mode must not execute verifier code")

    monkeypatch.setattr("nemo_gym.environment_ci.subprocess.run", unexpected_run)
    assert run_enforced_verifier_checks(report) is report
    assert report.units[0].verifier_checks == []


def test_ci_workflows_execute_manifest_bound_verifier_checks_for_each_merge_path() -> None:
    workflows = Path(__file__).parents[2] / ".github" / "workflows"
    workflow = (workflows / "unit-tests.yml").read_text()

    assert workflow.count("--run-verifier-tests") == 3
    assert "needs.detect.outputs.run_servers == 'true'" in workflow
    assert "merge_group:" in workflow
    assert "github.event.merge_group.base_sha" in workflow
    assert "github.event.merge_group.head_sha" in workflow

    main_workflow = (workflows / "full-test-suite.yml").read_text()
    assert main_workflow.count("--run-verifier-tests") == 2
    assert "fetch-depth: 0" in main_workflow
    assert "github.event.before" in main_workflow
    assert 'github.sha }}"' in main_workflow


def test_token_settings_fan_out_but_credential_only_changes_are_exempt(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "token_settings")
    entry = _manifest_entry(tmp_path, "token_settings")
    component_config = tmp_path / "resources_servers" / "token_settings" / "configs" / "token_settings.yaml"
    parsed = yaml.safe_load(component_config.read_text())
    server = parsed["token_settings_resources_server"]["resources_servers"]["token_settings"]
    server.update(
        {
            "api_key": "first-secret",
            "max_output_tokens": 64,
            "tokenizer": "tokenizer-v1",
        }
    )
    component_config.write_text(yaml.safe_dump(parsed, sort_keys=False))
    _record_entry_lock(tmp_path, entry)
    baseline = component_config.read_text()

    credential_data = yaml.safe_load(baseline)
    credential_data["token_settings_resources_server"]["resources_servers"]["token_settings"]["api_key"] = (
        "second-secret"
    )
    component_config.write_text(yaml.safe_dump(credential_data, sort_keys=False))
    credential_change = ChangedFile(path=component_config.relative_to(tmp_path), before_content=baseline)
    assert is_metadata_only_change(tmp_path, credential_change)
    credential_report = run_environment_ci_gate(
        tmp_path,
        changes=[credential_change],
        enforce_changes=True,
        catalog=_catalog(entry),
    )
    assert credential_report.passed
    assert credential_report.to_dict()["summary"]["enforced"] == 0

    for key, value in (("max_output_tokens", 128), ("tokenizer", "tokenizer-v2")):
        component_config.write_text(baseline)
        changed_data = yaml.safe_load(baseline)
        changed_data["token_settings_resources_server"]["resources_servers"]["token_settings"][key] = value
        component_config.write_text(yaml.safe_dump(changed_data, sort_keys=False))
        score_change = ChangedFile(path=component_config.relative_to(tmp_path), before_content=baseline)
        assert not is_metadata_only_change(tmp_path, score_change), key
        report = run_environment_ci_gate(
            tmp_path,
            changes=[score_change],
            enforce_changes=True,
            catalog=_catalog(entry),
        )
        assert not report.passed, key
        assert report.units[0].enforced, key
        assert any(
            reason.startswith("dependent:resources_servers/token_settings") for reason in report.units[0].reasons
        )
        assert any("locked sha256" in violation for violation in report.lock_violations), key


def test_shared_component_helper_change_drifts_lock_but_non_runtime_files_do_not(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "helper_lock")
    entry = _manifest_entry(tmp_path, "helper_lock")
    component_dir = tmp_path / "resources_servers" / "helper_lock"
    app = component_dir / "app.py"
    app.write_text(app.read_text() + "\nfrom resources_servers.helper_lock.scoring import score\n")
    helper = component_dir / "scoring.py"
    helper.write_text("def score(): return 1\n")
    _record_entry_lock(tmp_path, entry)

    before_helper = helper.read_text()
    helper.write_text("def score(): return 0\n")
    drift = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=helper.relative_to(tmp_path), before_content=before_helper)],
        enforce_changes=True,
        catalog=_catalog(entry),
    )
    assert not drift.passed
    assert any("locked sha256" in violation for violation in drift.lock_violations)

    # Restore the published helper, then prove files outside the deployed
    # component composition can still change without forcing a version bump.
    helper.write_text(before_helper)
    test_source = component_dir / "tests" / "test_app.py"
    before_test = test_source.read_text()
    test_source.write_text(before_test + "\n# test-only edit\n")
    test_only = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=test_source.relative_to(tmp_path), before_content=before_test)],
        enforce_changes=True,
        catalog=_catalog(entry),
    )
    assert test_only.passed
    assert test_only.units[0].enforced

    test_source.write_text(before_test)
    unused_config = component_dir / "configs" / "unused.yaml"
    unused_config.write_text("unused: true\n")
    config_only = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=unused_config.relative_to(tmp_path), status="A")],
        enforce_changes=True,
        catalog=_catalog(entry),
    )
    assert config_only.passed
    assert config_only.units[0].enforced


def test_transitive_component_import_change_fans_out_and_drifts_lock(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "import_consumer")
    entry = _manifest_entry(tmp_path, "import_consumer")
    shared_dir = tmp_path / "resources_servers" / "shared_scorer"
    shared_dir.mkdir(parents=True)
    shared_source = shared_dir / "app.py"
    shared_source.write_text("VALUE = 1\n")
    consumer_source = tmp_path / "resources_servers" / "import_consumer" / "app.py"
    consumer_source.write_text(
        consumer_source.read_text() + "\nfrom resources_servers.shared_scorer.app import VALUE\n"
    )
    _record_entry_lock(tmp_path, entry)

    before = shared_source.read_text()
    shared_source.write_text("VALUE = 2\n")
    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=shared_source.relative_to(tmp_path), before_content=before)],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not report.passed
    assert ("resources_servers", "shared_scorer") in report.units[0].dependencies
    assert any(reason.startswith("dependent:resources_servers/shared_scorer") for reason in report.units[0].reasons)
    assert any("locked sha256" in violation for violation in report.lock_violations)


def test_configured_component_module_change_fans_out_and_drifts_lock(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "configured_consumer")
    entry = _manifest_entry(tmp_path, "configured_consumer")
    delegated_dir = tmp_path / "responses_api_agents" / "delegated_agent"
    delegated_dir.mkdir(parents=True)
    delegated_source = delegated_dir / "app.py"
    delegated_source.write_text("VALUE = 1\n")
    config_path = tmp_path / "environments" / "configured_consumer" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["configured_consumer_agent"]["responses_api_agents"]["simple_agent"]["agent_server_module"] = (
        "responses_api_agents.delegated_agent.app"
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    _record_entry_lock(tmp_path, entry)

    before = delegated_source.read_text()
    delegated_source.write_text("VALUE = 2\n")
    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=delegated_source.relative_to(tmp_path), before_content=before)],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not report.passed
    assert ("responses_api_agents", "delegated_agent") in report.units[0].dependencies
    assert any(
        reason.startswith("dependent:responses_api_agents/delegated_agent") for reason in report.units[0].reasons
    )
    assert any("locked sha256" in violation for violation in report.lock_violations)


def test_custom_driver_helper_change_drifts_the_version_lock(tmp_path: Path) -> None:
    _schema(tmp_path)
    scaffold_environment(kind="environment", name="driver_case", profile="custom-driver", root=tmp_path)
    manifest = load_manifest(tmp_path / "environments" / "driver_case" / "manifest.yaml")
    complete_scaffold_fixture(
        tmp_path / "resources_servers" / "driver_case" / "tests" / "verifier_cases.jsonl",
        manifest,
    )
    entry = _manifest_entry(tmp_path, "driver_case")
    driver = tmp_path / "environments" / "driver_case" / "rollout_driver.py"
    helper = driver.with_name("driver_helper.py")
    helper.write_text("VALUE = 1\n")
    driver.write_text(driver.read_text() + "\nfrom .driver_helper import VALUE\n")
    _record_entry_lock(tmp_path, entry)

    before = helper.read_text()
    helper.write_text("VALUE = 2\n")
    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=helper.relative_to(tmp_path), before_content=before)],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not report.passed
    assert any("locked sha256" in violation for violation in report.lock_violations)


def test_shared_custom_driver_change_fans_out_and_drifts_lock(tmp_path: Path) -> None:
    _schema(tmp_path)
    scaffold_environment(kind="environment", name="driver_consumer", profile="custom-driver", root=tmp_path)
    manifest_path = tmp_path / "environments" / "driver_consumer" / "manifest.yaml"
    manifest = load_manifest(manifest_path)
    complete_scaffold_fixture(
        tmp_path / "resources_servers" / "driver_consumer" / "tests" / "verifier_cases.jsonl",
        manifest,
    )
    shared_dir = tmp_path / "responses_api_agents" / "shared_driver"
    shared_dir.mkdir(parents=True)
    shared_driver = shared_dir / "driver.py"
    shared_driver.write_text("async def run(*_args): pass\n")
    config_path = tmp_path / "environments" / "driver_consumer" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["rollout_collection_driver"] = "responses_api_agents.shared_driver.driver:run"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest_data = yaml.safe_load(manifest_path.read_text())
    manifest_data["rollout_driver"] = "responses_api_agents.shared_driver.driver:run"
    manifest_path.write_text(yaml.safe_dump(manifest_data, sort_keys=False))
    entry = _manifest_entry(tmp_path, "driver_consumer")
    _record_entry_lock(tmp_path, entry)

    before = shared_driver.read_text()
    shared_driver.write_text("async def run(*_args): return None\n")
    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=shared_driver.relative_to(tmp_path), before_content=before)],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not report.passed
    assert ("responses_api_agents", "shared_driver") in report.units[0].dependencies
    assert any(reason.startswith("dependent:responses_api_agents/shared_driver") for reason in report.units[0].reasons)
    assert any("locked sha256" in violation for violation in report.lock_violations)


def test_shared_dataset_prepare_helper_change_fans_out_and_drifts_lock(tmp_path: Path) -> None:
    _schema(tmp_path)
    scaffold_environment(
        kind="benchmark",
        name="prepare_consumer",
        metadata={"canonical_split": "test"},
        root=tmp_path,
    )
    manifest_path = tmp_path / "benchmarks" / "prepare_consumer" / "manifest.yaml"
    manifest = load_manifest(manifest_path)
    complete_scaffold_fixture(
        tmp_path / "resources_servers" / "prepare_consumer" / "tests" / "verifier_cases.jsonl",
        manifest,
    )
    shared_dir = tmp_path / "benchmarks" / "shared_prepare"
    shared_dir.mkdir(parents=True)
    (shared_dir / "prepare.py").write_text(
        "from benchmarks.shared_prepare.helper import VALUE\n\ndef prepare():\n    return VALUE\n"
    )
    helper = shared_dir / "helper.py"
    helper.write_text("VALUE = 1\n")

    config_path = manifest_path.with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text())
    dataset = config["prepare_consumer_agent"]["responses_api_agents"]["simple_agent"]["datasets"][0]
    dataset["prepare_script"] = "benchmarks/shared_prepare/prepare.py"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest_data = yaml.safe_load(manifest_path.read_text())
    manifest_data["datasets"][0]["prepare_script"] = "benchmarks/shared_prepare/prepare.py"
    manifest_path.write_text(yaml.safe_dump(manifest_data, sort_keys=False))

    entry = _manifest_entry(tmp_path, "prepare_consumer", kind="benchmark")
    _record_entry_lock(tmp_path, entry)
    before = helper.read_text()
    helper.write_text("VALUE = 2\n")
    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=helper.relative_to(tmp_path), before_content=before)],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not report.passed
    assert ("benchmarks", "shared_prepare") in report.units[0].dependencies
    assert any(reason.startswith("dependent:benchmarks/shared_prepare") for reason in report.units[0].reasons)
    assert any("locked sha256" in violation for violation in report.lock_violations)


def test_cross_package_runtime_file_change_fans_out_and_drifts_lock(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "runtime_ref_consumer")
    manifest_path = tmp_path / "environments" / "runtime_ref_consumer" / "manifest.yaml"
    prompt = tmp_path / "benchmarks" / "shared_prompt" / "prompt.yaml"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("prompt: first\n")

    config_path = manifest_path.with_name("config.yaml")
    config = yaml.safe_load(config_path.read_text())
    agent = config["runtime_ref_consumer_agent"]["responses_api_agents"]["simple_agent"]
    agent["turn2_prompt_fpath"] = "benchmarks/shared_prompt/prompt.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    entry = _manifest_entry(tmp_path, "runtime_ref_consumer")
    _record_entry_lock(tmp_path, entry)
    before = prompt.read_text()
    prompt.write_text("prompt: second\n")
    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=prompt.relative_to(tmp_path), before_content=before)],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not report.passed
    assert ("benchmarks", "shared_prompt") in report.units[0].dependencies
    assert any(reason.startswith("dependent:benchmarks/shared_prompt") for reason in report.units[0].reasons)
    assert any("locked sha256" in violation for violation in report.lock_violations)


def test_version_lock_rejects_drift_and_allows_manifest_version_bump(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "locked")
    entry = _manifest_entry(tmp_path, "locked")
    baseline = run_environment_ci_gate(tmp_path, catalog=_catalog(entry))
    unit = baseline.units[0]
    manifest = load_manifest(entry.manifest_path)
    check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=entry.manifest_path,
        manifest=manifest,
        composition_hash=unit.composition_hash,
    )

    app = tmp_path / "resources_servers" / "locked" / "app.py"
    before_app = app.read_text()
    app.write_text(before_app + "\n# changed scorer\n")
    app_change = ChangedFile(path=app.relative_to(tmp_path), before_content=before_app)
    drift = run_environment_ci_gate(
        tmp_path,
        changes=[app_change],
        enforce_changes=True,
        catalog=_catalog(entry),
    )
    assert not drift.passed
    assert any("locked sha256" in violation for violation in drift.lock_violations)

    manifest_data = yaml.safe_load(entry.manifest_path.read_text())
    before_manifest = entry.manifest_path.read_text()
    manifest_data["version"] = "0.1.1"
    entry.manifest_path.write_text(yaml.safe_dump(manifest_data, sort_keys=False))
    bumped_entry = _manifest_entry(tmp_path, "locked")
    _record_entry_lock(tmp_path, bumped_entry)
    bumped = run_environment_ci_gate(
        tmp_path,
        changes=[
            app_change,
            ChangedFile(
                path=entry.manifest_path.relative_to(tmp_path),
                before_content=before_manifest,
            ),
        ],
        enforce_changes=True,
        catalog=_catalog(bumped_entry),
    )
    assert bumped.passed
    assert bumped.lock_violations == ()


def test_enforced_manifest_requires_a_published_version_lock(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "unpublished")
    entry = _manifest_entry(tmp_path, "unpublished")

    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=entry.manifest_path.relative_to(tmp_path), status="A")],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not report.passed
    assert report.lock_violations == (
        "environment:unpublished@0.1.0: no published composition lock; run `gym env publish unpublished "
        f"--owner <github-handle>` and commit {LOCK_RELATIVE_PATH}",
    )


def test_existing_version_lock_cannot_be_rewritten_to_bypass_drift(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "tampered")
    entry = _manifest_entry(tmp_path, "tampered")
    _record_entry_lock(tmp_path, entry)
    lock_path = tmp_path / LOCK_RELATIVE_PATH
    before_lock = lock_path.read_text()

    app = tmp_path / "resources_servers" / "tampered" / "app.py"
    app.write_text(app.read_text() + "\n# changed scorer\n")
    current = run_environment_ci_gate(tmp_path, catalog=_catalog(entry)).units[0]
    lock_data = json.loads(before_lock)
    lock_data["environments"][current.version_key]["composition_hash"] = current.composition_hash
    lock_path.write_text(json.dumps(lock_data, indent=2, sort_keys=True) + "\n")

    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=LOCK_RELATIVE_PATH, before_content=before_lock)],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not report.passed
    assert any("published composition lock is immutable and was edited" in error for error in report.lock_violations)


def test_new_version_lock_must_match_a_current_exact_manifest_version(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "future_lock")
    entry = _manifest_entry(tmp_path, "future_lock")
    manifest = load_manifest(entry.manifest_path).model_copy(update={"version": "999.0.0"})
    check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=entry.manifest_path,
        config_path=entry.config_path,
        manifest=manifest,
        composition_hash="f" * 64,
    )

    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=LOCK_RELATIVE_PATH, status="A")],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not report.passed
    assert any(
        "does not match any currently resolved exact environment version" in error for error in report.lock_violations
    )


def test_new_version_lock_must_reference_the_exact_current_manifest_path(tmp_path: Path) -> None:
    _schema(tmp_path)
    _scaffold_ready_environment(tmp_path, "wrong_path_lock")
    entry = _manifest_entry(tmp_path, "wrong_path_lock")
    current = run_environment_ci_gate(tmp_path, catalog=_catalog(entry)).units[0]
    check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=entry.manifest_path,
        config_path=entry.config_path,
        manifest=load_manifest(entry.manifest_path),
        composition_hash=current.composition_hash,
    )
    lock_path = tmp_path / LOCK_RELATIVE_PATH
    lock_payload = json.loads(lock_path.read_text())
    lock_payload["environments"][current.version_key]["manifest"] = "environments/someone_else/manifest.yaml"
    lock_path.write_text(json.dumps(lock_payload, indent=2, sort_keys=True) + "\n")

    report = run_environment_ci_gate(
        tmp_path,
        changes=[ChangedFile(path=LOCK_RELATIVE_PATH, status="A")],
        enforce_changes=True,
        catalog=_catalog(entry),
    )

    assert not report.passed
    assert any("composition lock manifest must be" in error for error in report.lock_violations)


def test_stale_schema_is_reported_and_only_fails_enforcement_mode(tmp_path: Path) -> None:
    schema = tmp_path / "schemas" / "environment-manifest.schema.json"
    schema.parent.mkdir(parents=True)
    schema.write_text("{}\n")
    empty = _catalog()

    report_only = run_environment_ci_gate(tmp_path, catalog=empty)
    enforced = run_environment_ci_gate(tmp_path, enforce_changes=True, catalog=empty)

    assert report_only.passed
    assert report_only.schema_errors
    assert not enforced.passed
