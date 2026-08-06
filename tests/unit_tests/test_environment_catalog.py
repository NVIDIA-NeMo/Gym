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
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import nemo_gym.environment_catalog as catalog_module
from nemo_gym.config_types import ConfigError
from nemo_gym.environment_catalog import (
    CatalogFilters,
    discover_environment_catalog,
    discover_exact_environment_catalog,
    discover_exact_manifest_catalog,
)
from nemo_gym.environment_inventory import MIGRATION_DRAFT_HEADER, MIGRATION_INVENTORY_PATH
from nemo_gym.environment_manifest import load_manifest
from nemo_gym.environment_versioning import LOCK_RELATIVE_PATH, check_or_record_version_lock
from nemo_gym.registry import EnvironmentEntry


class _FakeManifest:
    def __init__(self, data: dict) -> None:
        self.data = data

    @property
    def kind(self):
        return SimpleNamespace(value=self.data["kind"])

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def version(self) -> str:
        return self.data["version"]

    def model_dump(self, **_kwargs) -> dict:
        return self.data


def _write_manifest(root: Path, kind_dir: str, name: str, data: dict) -> Path:
    path = root / kind_dir / name / "manifest.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data))
    (path.parent / "config.yaml").write_text("{}\n")
    return path


def _manifest(name: str, kind: str, **updates) -> dict:
    data = {
        "name": name,
        "version": "1.2.3",
        "kind": kind,
        "integration_profile": "stock-loop",
        "domain": "math",
        "description": f"The {name} workload",
        "modality": "text",
        "licensing": "Apache-2.0",
        "authors": ["owner"],
        "reward": {"range": [0, 1], "higher_is_better": True},
        "determinism": "seeded",
        "lifecycle": "active",
        "resources_server": name,
        "agent_server": "simple_agent",
        "model_server": "policy_model",
        "datasets": [
            {
                "name": name,
                "type": "benchmark" if kind == "benchmark" else "example",
                "jsonl_fpath": f"{kind}s/{name}/data.jsonl",
                **({"prepare_script": f"benchmarks/{name}/prepare.py"} if kind == "benchmark" else {}),
            }
        ],
        "requires": ["text-model"],
    }
    if kind == "benchmark":
        data.update(canonical_split="test", standard_prompt_config=f"benchmarks/{name}/prompt.yaml")
    data.update(updates)
    return data


def _publish_manifest(repo_root: Path, manifest_path: Path, digest: str = "a" * 64) -> None:
    check_or_record_version_lock(
        repo_root=repo_root,
        manifest_path=manifest_path,
        config_path=manifest_path.with_name("config.yaml"),
        manifest=load_manifest(manifest_path),
        composition_hash=digest,
    )


def _configure_discovery(monkeypatch, roots: list[Path], *, environments=None, benchmarks=None) -> None:
    monkeypatch.setattr(catalog_module, "component_search_roots", lambda: roots)
    monkeypatch.setattr(
        catalog_module,
        "load_manifest",
        lambda path: _FakeManifest(json.loads(path.read_text())),
    )
    monkeypatch.setattr(
        catalog_module,
        "resolve_config_paths_static",
        lambda paths: {"config_paths": [str(path) for path in paths]},
    )
    monkeypatch.setattr(catalog_module, "compute_composition_hash", lambda _config, _manifest: "a" * 64)


def test_discovers_manifests_from_both_trees_and_emits_json(tmp_path: Path, monkeypatch) -> None:
    env_manifest = _write_manifest(tmp_path, "environments", "alpha", _manifest("alpha", "environment"))
    _write_manifest(tmp_path, "benchmarks", "beta", _manifest("beta", "benchmark", domain="knowledge"))
    _configure_discovery(monkeypatch, [tmp_path])

    catalog = discover_environment_catalog(include_unpublished=True)

    assert [(entry.name, entry.kind, entry.status) for entry in catalog.entries] == [
        ("alpha", "environment", "experimental"),
        ("beta", "benchmark", "experimental"),
    ]
    assert catalog.coverage.to_json_dict() == {
        "total": 2,
        "with_manifest": 2,
        "without_manifest": 0,
        "invalid_manifests": 0,
        "percent": 100.0,
    }
    payload = catalog.to_json_dict()["entries"][0]
    assert payload["required_capabilities"] == ["text-model"]
    assert payload["manifest_path"] == str(env_manifest.resolve())
    assert payload["config_path"] == str((env_manifest.parent / "config.yaml").resolve())


def test_manifest_becomes_discoverable_only_after_publication(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _write_manifest(tmp_path, "environments", "alpha", _manifest("alpha", "environment"))
    config_path = manifest_path.with_name("config.yaml")
    legacy = EnvironmentEntry(
        name="alpha",
        config_path=config_path,
        path=config_path.parent,
        domain="math",
        description="Legacy runnable",
    )
    _configure_discovery(monkeypatch, [tmp_path], environments={"alpha": legacy})

    assert [(entry.name, entry.source) for entry in discover_environment_catalog().entries] == [("alpha", "legacy")]
    assert [entry.source for entry in discover_environment_catalog(include_unpublished=True).entries] == ["manifest"]

    _publish_manifest(tmp_path, manifest_path)

    published = discover_environment_catalog()
    assert [(entry.name, entry.source, entry.status) for entry in published.entries] == [
        ("alpha", "manifest", "experimental")
    ]


def test_nested_component_root_uses_its_git_repository_publication_lock(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".git").mkdir()
    component_root = tmp_path / "plugins" / "team"
    manifest_path = _write_manifest(component_root, "environments", "alpha", _manifest("alpha", "environment"))
    _publish_manifest(tmp_path, manifest_path)
    _configure_discovery(monkeypatch, [component_root])

    catalog = discover_environment_catalog(include_legacy=False)

    assert [(entry.name, entry.source) for entry in catalog.entries] == [("alpha", "manifest")]


def test_invalid_version_lock_is_reported_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_manifest(tmp_path, "environments", "alpha", _manifest("alpha", "environment"))
    _write_manifest(tmp_path, "environments", "beta", _manifest("beta", "environment"))
    lock_path = tmp_path / LOCK_RELATIVE_PATH
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text('{"schema_version": 99, "environments": {}}')
    _configure_discovery(monkeypatch, [tmp_path])

    catalog = discover_environment_catalog(include_legacy=False, include_unpublished=True)

    assert [issue.code for issue in catalog.issues] == ["invalid-version-lock"]


def test_catalog_rejects_published_composition_drift(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _write_manifest(tmp_path, "environments", "alpha", _manifest("alpha", "environment"))
    _configure_discovery(monkeypatch, [tmp_path])
    _publish_manifest(tmp_path, manifest_path)
    monkeypatch.setattr(catalog_module, "compute_composition_hash", lambda _config, _manifest: "b" * 64)

    catalog = discover_environment_catalog(include_legacy=False)

    assert catalog.entries == ()
    assert [issue.code for issue in catalog.issues] == ["invalid-version-lock"]
    assert "composition_hash" in catalog.issues[0].message


@pytest.mark.parametrize(
    ("field_name", "changed_path"),
    [
        ("manifest", "environments/other/manifest.yaml"),
        ("config", "environments/other/config.yaml"),
    ],
)
def test_catalog_rejects_published_path_drift(
    tmp_path: Path,
    monkeypatch,
    field_name: str,
    changed_path: str,
) -> None:
    manifest_path = _write_manifest(tmp_path, "environments", "alpha", _manifest("alpha", "environment"))
    _configure_discovery(monkeypatch, [tmp_path])
    _publish_manifest(tmp_path, manifest_path)
    lock_path = tmp_path / LOCK_RELATIVE_PATH
    lock = json.loads(lock_path.read_text())
    lock["environments"]["environment:alpha@1.2.3"][field_name] = changed_path
    lock_path.write_text(json.dumps(lock))

    catalog = discover_environment_catalog(include_legacy=False)

    assert catalog.entries == ()
    assert [issue.code for issue in catalog.issues] == ["invalid-version-lock"]
    assert field_name in catalog.issues[0].message


def test_catalog_reports_locked_config_resolution_errors(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _write_manifest(tmp_path, "environments", "alpha", _manifest("alpha", "environment"))
    _configure_discovery(monkeypatch, [tmp_path])
    _publish_manifest(tmp_path, manifest_path)

    def fail_resolution(_paths):
        raise ConfigError("fixture config cannot resolve")

    monkeypatch.setattr(catalog_module, "resolve_config_paths_static", fail_resolution)

    catalog = discover_environment_catalog(include_legacy=False)

    assert catalog.entries == ()
    assert [issue.code for issue in catalog.issues] == ["invalid-version-lock"]
    assert "fixture config cannot resolve" in catalog.issues[0].message


def test_catalog_rejects_manifest_symlink_outside_component_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "checkout"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "manifest.yaml"
    target.write_text(json.dumps(_manifest("escaped", "environment")))
    linked = root / "environments" / "escaped" / "manifest.yaml"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(target)
    _configure_discovery(monkeypatch, [root])

    catalog = discover_environment_catalog(include_legacy=False)

    assert catalog.entries == ()
    assert catalog.coverage.invalid_manifests == 1
    assert len(catalog.issues) == 1
    assert catalog.issues[0].path == linked.absolute()
    assert "symbolic-link" in catalog.issues[0].message


def test_exact_manifest_lookup_does_not_walk_unrelated_registry_entries(tmp_path: Path, monkeypatch) -> None:
    selected = _write_manifest(tmp_path, "environments", "selected", _manifest("selected", "environment"))
    _write_manifest(tmp_path, "environments", "unrelated", _manifest("unrelated", "environment"))
    _configure_discovery(monkeypatch, [tmp_path])

    loaded: list[Path] = []

    def load(path: Path):
        loaded.append(path)
        return _FakeManifest(json.loads(path.read_text()))

    monkeypatch.setattr(catalog_module, "load_manifest", load)
    monkeypatch.setattr(
        catalog_module,
        "_manifest_paths",
        lambda _roots: (_ for _ in ()).throw(AssertionError("exact lookup must not scan registry trees")),
    )

    catalog = discover_exact_manifest_catalog("selected", "environment", include_unpublished=True)

    assert [entry.name for entry in catalog.entries] == ["selected"]
    assert loaded == [selected.absolute()]


def test_exact_manifest_lookup_rejects_path_traversal_without_touching_files(tmp_path: Path, monkeypatch) -> None:
    _configure_discovery(monkeypatch, [tmp_path])
    monkeypatch.setattr(
        catalog_module,
        "load_manifest",
        lambda _path: (_ for _ in ()).throw(AssertionError("unsafe name must not be inspected")),
    )

    catalog = discover_exact_manifest_catalog("../outside", include_unpublished=True)

    assert catalog.entries == ()
    assert catalog.issues == ()


def test_exact_legacy_lookup_reads_only_the_selected_recipe(tmp_path: Path, monkeypatch) -> None:
    selected = tmp_path / "environments" / "selected" / "config.yaml"
    selected.parent.mkdir(parents=True)
    selected.write_text("description: Selected legacy recipe\n")
    unrelated = tmp_path / "environments" / "unrelated" / "config.yaml"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("description: Unrelated legacy recipe\n")
    _configure_discovery(monkeypatch, [tmp_path])
    catalog = discover_exact_environment_catalog("selected", "environment")

    assert [(entry.name, entry.config_path) for entry in catalog.entries] == [("selected", selected.resolve())]


def test_exact_legacy_lookup_supports_nested_benchmark_identity(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "benchmarks" / "suite" / "variant.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("dataset:\n  type: benchmark\n")
    _configure_discovery(monkeypatch, [tmp_path])

    catalog = discover_exact_environment_catalog("suite/variant", "benchmark")

    assert [(entry.name, entry.config_path) for entry in catalog.entries] == [("suite/variant", config_path.resolve())]


def test_discovers_runnable_resources_server_manifest_and_its_config(tmp_path: Path, monkeypatch) -> None:
    manifest_path = tmp_path / "resources_servers" / "standalone" / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(_manifest("standalone", "environment")))
    config_path = manifest_path.parent / "configs" / "standalone.yaml"
    config_path.parent.mkdir()
    config_path.write_text("{}\n")
    _configure_discovery(monkeypatch, [tmp_path])

    entry = discover_environment_catalog(include_unpublished=True).entries[0]

    assert entry.name == "standalone"
    assert entry.config_path == config_path.resolve()


def test_legacy_resources_configs_are_not_collapsed_or_name_collided(tmp_path: Path, monkeypatch) -> None:
    component = tmp_path / "resources_servers" / "shared"
    configs = component / "configs"
    configs.mkdir(parents=True)
    runnable = {
        "resources_servers": {"shared": {}},
        "responses_api_agents": {"shared": {}},
        "datasets": {"shared": {}},
    }
    (configs / "first.yaml").write_text(json.dumps(runnable))
    (configs / "second.yaml").write_text(json.dumps(runnable))
    _configure_discovery(monkeypatch, [tmp_path])

    catalog = discover_environment_catalog(include_unpublished=True)

    assert [entry.name for entry in catalog.entries] == [
        "resources_servers/shared/first",
        "resources_servers/shared/second",
    ]
    assert catalog.coverage.total == 2
    assert catalog.coverage.without_manifest == 2
    assert catalog.coverage.invalid_manifests == 0
    assert [(issue.code, issue.path) for issue in catalog.issues] == [
        ("ambiguous-legacy-resource", component.resolve())
    ]


def test_resource_manifest_replaces_only_its_selected_legacy_config(tmp_path: Path, monkeypatch) -> None:
    component = tmp_path / "resources_servers" / "shared"
    manifest_path = component / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(_manifest("shared", "environment")))
    configs = component / "configs"
    configs.mkdir()
    runnable = {
        "resources_servers": {"shared": {}},
        "responses_api_agents": {"shared": {}},
        "datasets": {"shared": {}},
    }
    (configs / "shared.yaml").write_text(json.dumps(runnable))
    extra = configs / "variant.yaml"
    extra.write_text(json.dumps(runnable))
    _configure_discovery(monkeypatch, [tmp_path])

    catalog = discover_environment_catalog(include_unpublished=True)

    assert [(entry.name, entry.source) for entry in catalog.entries] == [
        ("resources_servers/shared/variant", "legacy"),
        ("shared", "manifest"),
    ]
    assert catalog.coverage.total == 2


def test_recipe_manifest_claims_its_single_reused_resource_composition(tmp_path: Path, monkeypatch) -> None:
    _write_manifest(
        tmp_path,
        "environments",
        "consumer",
        _manifest("consumer", "environment", resources_server="shared"),
    )
    config_path = tmp_path / "resources_servers" / "shared" / "configs" / "shared.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "shared": {"resources_servers": {"shared": {}}},
                "agent": {"responses_api_agents": {"simple_agent": {}}},
                "datasets": {"datasets": {}},
            }
        )
    )
    _configure_discovery(monkeypatch, [tmp_path])

    catalog = discover_environment_catalog(include_unpublished=True)

    assert [(entry.name, entry.source) for entry in catalog.entries] == [("consumer", "manifest")]
    assert catalog.coverage.total == 1


def test_unions_legacy_entries_and_manifest_wins_in_same_root(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _write_manifest(tmp_path, "environments", "alpha", _manifest("alpha", "environment"))
    legacy_path = manifest_path.parent / "config.yaml"
    legacy = EnvironmentEntry(
        name="alpha",
        config_path=legacy_path,
        path=legacy_path.parent,
        domain="other",
        description="legacy alpha",
    )
    other_path = tmp_path / "environments" / "legacy" / "config.yaml"
    other_path.parent.mkdir(parents=True)
    other_path.write_text("{}\n")
    other = EnvironmentEntry(
        name="legacy",
        config_path=other_path,
        path=other_path.parent,
        domain="agent",
        description="legacy only",
    )
    _configure_discovery(monkeypatch, [tmp_path], environments={"alpha": legacy, "legacy": other})

    catalog = discover_environment_catalog(include_unpublished=True)

    by_name = {entry.name: entry for entry in catalog.entries}
    assert by_name["alpha"].source == "manifest"
    assert by_name["legacy"].status == "no-manifest"
    assert by_name["legacy"].lifecycle is None
    assert catalog.coverage.with_manifest == 1
    assert catalog.coverage.without_manifest == 1


def test_root_precedence_beats_manifest_priority(tmp_path: Path, monkeypatch) -> None:
    high = tmp_path / "high"
    low = tmp_path / "low"
    _write_manifest(low, "environments", "same", _manifest("same", "environment"))
    high_config = high / "environments" / "same" / "config.yaml"
    high_config.parent.mkdir(parents=True)
    high_config.write_text("{}\n")
    legacy = EnvironmentEntry(
        name="same",
        config_path=high_config,
        path=high_config.parent,
        domain="agent",
        description="user override",
    )
    _configure_discovery(monkeypatch, [high, low], environments={"same": legacy})

    entry = discover_environment_catalog().entries[0]

    assert entry.name == "same"
    assert entry.source == "legacy"
    assert entry.config_path == high_config.resolve()


def test_invalid_manifest_is_reported_without_hiding_other_entries(tmp_path: Path, monkeypatch) -> None:
    bad = _write_manifest(tmp_path, "environments", "bad", {"invalid": True})
    _write_manifest(tmp_path, "environments", "good", _manifest("good", "environment"))
    _configure_discovery(monkeypatch, [tmp_path])

    def load(path: Path):
        if path == bad:
            raise ValueError("name: field required")
        return _FakeManifest(json.loads(path.read_text()))

    monkeypatch.setattr(catalog_module, "load_manifest", load)

    catalog = discover_environment_catalog(include_unpublished=True)

    assert [entry.name for entry in catalog.entries] == ["good"]
    assert len(catalog.issues) == 1
    assert catalog.issues[0].path == bad.resolve()
    assert "field required" in catalog.issues[0].message
    assert catalog.coverage.total == 2
    assert catalog.coverage.with_manifest == 1
    assert catalog.coverage.without_manifest == 1
    assert catalog.coverage.invalid_manifests == 1


def test_only_inventory_tracked_generated_todo_manifest_is_a_nonblocking_migration_draft(
    tmp_path: Path,
    monkeypatch,
) -> None:
    draft_path = tmp_path / "environments" / "legacy" / "manifest.yaml"
    draft_path.parent.mkdir(parents=True)
    draft_content = (
        f"{MIGRATION_DRAFT_HEADER}\n"
        "# This draft is intentionally invalid until every TODO_REQUIRED value is replaced.\n"
        "version: 'TODO_REQUIRED: semantic version'\n"
    )
    draft_path.write_text(draft_content)
    config_path = draft_path.with_name("config.yaml")
    config_path.write_text("{}\n")
    legacy = EnvironmentEntry(
        name="legacy",
        config_path=config_path,
        path=config_path.parent,
        domain="math",
        description="Legacy runnable remains discoverable",
    )
    inventory_path = tmp_path / MIGRATION_INVENTORY_PATH
    inventory_path.parent.mkdir(parents=True)
    inventory = {
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
                "name": "legacy",
                "kind": "environment",
                "registry": "environments",
                "config_path": "environments/legacy/config.yaml",
                "manifest_path": "environments/legacy/manifest.yaml",
                "status": "drafted",
            }
        ],
        "exceptions": [],
    }
    inventory_path.write_text(json.dumps(inventory) + "\n")
    _configure_discovery(monkeypatch, [tmp_path], environments={"legacy": legacy})
    monkeypatch.setattr(catalog_module, "load_manifest", lambda _path: (_ for _ in ()).throw(ValueError("TODO")))

    catalog = discover_environment_catalog()

    assert [(entry.name, entry.source, entry.status) for entry in catalog.entries] == [
        ("legacy", "legacy", "no-manifest")
    ]
    assert [(issue.code, issue.path) for issue in catalog.issues] == [("migration-draft", draft_path.absolute())]
    assert catalog.coverage.total == 1
    assert catalog.coverage.invalid_manifests == 0
    assert catalog.coverage.without_manifest == 1

    inventory["units"][0]["status"] = "planned"
    inventory_path.write_text(json.dumps(inventory) + "\n")
    untracked = discover_environment_catalog()
    untracked_exact = discover_exact_environment_catalog("legacy", include_unpublished=True)
    assert [issue.code for issue in untracked.issues] == ["invalid-manifest"]
    assert untracked.entries == ()
    assert untracked_exact.entries == ()
    assert [issue.code for issue in untracked_exact.issues] == ["invalid-manifest"]
    assert untracked.coverage.invalid_manifests == 1

    inventory["units"][0]["status"] = "drafted"
    inventory_path.write_text(json.dumps(inventory) + "\n")
    draft_path.write_text(
        f"{MIGRATION_DRAFT_HEADER}\n"
        "# This comment alone mentioning TODO_REQUIRED does not make a draft.\n"
        "version: 1.0.0\n"
    )
    completed_todos = discover_environment_catalog()
    assert [issue.code for issue in completed_todos.issues] == ["invalid-manifest"]


def test_manifest_identity_and_runnable_config_are_enforced(tmp_path: Path, monkeypatch) -> None:
    wrong_name = _write_manifest(
        tmp_path,
        "environments",
        "directory_name",
        _manifest("declared_name", "environment"),
    )
    missing_config = _write_manifest(
        tmp_path,
        "benchmarks",
        "missing_config",
        _manifest("missing_config", "benchmark"),
    )
    (missing_config.parent / "config.yaml").unlink()
    _configure_discovery(monkeypatch, [tmp_path])

    catalog = discover_environment_catalog()

    assert catalog.entries == ()
    assert {issue.path for issue in catalog.issues} == {wrong_name.resolve(), missing_config.resolve()}
    assert any("path identity" in issue.message for issue in catalog.issues)
    assert any("runnable config" in issue.message for issue in catalog.issues)


def test_filters_manifest_metadata_and_reports_unfilterable_legacy_entries(tmp_path: Path, monkeypatch) -> None:
    _write_manifest(
        tmp_path,
        "environments",
        "alpha",
        _manifest("alpha", "environment", requires=["text-model", "tools"], description="Financial documents"),
    )
    legacy_path = tmp_path / "environments" / "legacy" / "config.yaml"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(
        "legacy:\n  resources_servers:\n    scorer:\n      domain: math\n      description: Financial legacy data\n"
    )
    legacy = EnvironmentEntry(
        name="legacy",
        config_path=legacy_path,
        path=legacy_path.parent,
        domain="math",
        description="Financial legacy data",
    )
    _configure_discovery(monkeypatch, [tmp_path], environments={"legacy": legacy})

    catalog = discover_environment_catalog(
        CatalogFilters(query="math", modality="text", required_capabilities=frozenset({"tools"})),
        include_unpublished=True,
    )

    assert [entry.name for entry in catalog.entries] == ["alpha"]
    assert [(limitation.field, limitation.entry_names) for limitation in catalog.filter_limitations] == [
        ("modality", ("legacy",)),
    ]
    # Coverage describes the full union, not just the filtered view.
    assert catalog.coverage.total == 2


def test_legacy_benchmark_metadata_is_preserved(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "benchmarks" / "beta" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "benchmark:\n"
        "  responses_api_agents:\n"
        "    agent:\n"
        "      datasets: [{name: beta, type: benchmark, jsonl_fpath: data.jsonl}]\n"
        "  resources_servers:\n"
        "    scorer:\n"
        "      domain: knowledge\n"
        "      description: A legacy benchmark\n"
    )
    benchmark = SimpleNamespace(path=config_path)
    _configure_discovery(monkeypatch, [tmp_path], benchmarks={"beta": benchmark})

    entry = discover_environment_catalog().entries[0]

    assert entry.kind == "benchmark"
    assert entry.status == "no-manifest"
    assert entry.domain == "knowledge"
    assert entry.description == "A legacy benchmark"
