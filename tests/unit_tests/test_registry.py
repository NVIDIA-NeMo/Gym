# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from pathlib import Path

import pytest

import nemo_gym.registry as registry_module
from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.environment.manifest import dump_manifest
from nemo_gym.registry import (
    RegistryError,
    _discover_environments_in_dir,
    discover_environment_catalog,
    discover_environments,
    resolve_catalog_entry,
)


def _make_env(environments_dir: Path, name: str, config_body: str) -> Path:
    env_dir = environments_dir / name
    env_dir.mkdir(parents=True)
    config_path = env_dir / "config.yaml"
    config_path.write_text(config_body)
    return config_path


_ENV_CONFIG = """{name}:
  resources_servers:
    {name}:
      entrypoint: app.py
      domain: agent
      description: {name} test environment
{name}_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: {name}
      model_server:
        type: responses_api_models
        name: policy_model
"""


def _manifest(name: str, kind: str = "environment", **updates) -> dict:
    dataset = {
        "name": name,
        "type": "benchmark" if kind == "benchmark" else "example",
        "jsonl_fpath": f"{kind}s/{name}/data/example.jsonl",
    }
    data = {
        "name": name,
        "version": "0.1.0",
        "kind": kind,
        "integration_profile": "custom-gym-verifier",
        "domain": "other",
        "description": f"{name} manifest entry",
        "modality": "text",
        "licensing": "Apache-2.0",
        "authors": ["Test Author"],
        "reward": {"range": [0, 1], "higher_is_better": True},
        "resources_server": name.replace("/", "_"),
        "agent_server": "simple_agent",
        "model_server": "policy_model",
        "datasets": [dataset],
    }
    if kind == "benchmark":
        prompt = f"benchmarks/{name}/prompt.yaml"
        dataset.update(prepare_script=f"benchmarks/{name}/prepare.py", prompt_config=prompt)
        data.update(canonical_split="test", standard_prompt_config=prompt)
    data.update(updates)
    return data


def _write_manifest(
    root: Path,
    tree: str,
    path_name: str,
    data: dict,
    *,
    with_config: bool = True,
) -> Path:
    manifest_path = root / tree / path_name / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(dump_manifest(data), encoding="utf-8")
    if with_config:
        manifest_path.with_name("config.yaml").write_text("{}\n", encoding="utf-8")
    return manifest_path


class TestDiscoverEnvironments:
    def test_discovers_by_directory_name_with_metadata(self, tmp_path: Path) -> None:
        envs_dir = tmp_path / "environments"
        _make_env(envs_dir, "alpha", _ENV_CONFIG.format(name="alpha"))
        _make_env(envs_dir, "beta", _ENV_CONFIG.format(name="beta"))

        environments = _discover_environments_in_dir(envs_dir)

        assert set(environments) == {"alpha", "beta"}
        alpha = environments["alpha"]
        assert alpha.name == "alpha"
        assert alpha.config_path == envs_dir / "alpha" / "config.yaml"
        assert alpha.path == envs_dir / "alpha"
        assert alpha.description == "alpha test environment"
        assert alpha.domain == "agent"

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        assert _discover_environments_in_dir(tmp_path / "does_not_exist") == {}

    def test_ignores_dirs_without_config_and_loose_files(self, tmp_path: Path) -> None:
        envs_dir = tmp_path / "environments"
        _make_env(envs_dir, "real", _ENV_CONFIG.format(name="real"))
        (envs_dir / "not_an_env").mkdir()  # dir without a config.yaml
        (envs_dir / "__init__.py").write_text("")  # loose file

        assert set(_discover_environments_in_dir(envs_dir)) == {"real"}

    def test_ignores_tombstone_directories(self, tmp_path: Path) -> None:
        envs_dir = tmp_path / "environments"
        manifest_path = _write_manifest(tmp_path, "environments", "moved", _manifest("moved"))
        manifest_path.parent.joinpath(registry_module.ENVIRONMENT_TOMBSTONE_FILENAME).write_text("replacement\n")

        assert _discover_environments_in_dir(envs_dir) == {}

    def test_unparseable_or_metadataless_configs_still_discovered(self, tmp_path: Path) -> None:
        # Configs without a parseable resources_servers block (or malformed YAML) must still be
        # discovered by name, just with no description/domain — never crash discovery.
        envs_dir = tmp_path / "environments"
        _make_env(envs_dir, "no_rs", "agent_only:\n  responses_api_agents:\n    a: {}\n")  # no resources_servers
        _make_env(envs_dir, "top_list", "- x\n- y\n")  # top-level not a mapping
        _make_env(envs_dir, "scalar_top", "top: just_a_string\n")  # top-level value not a dict
        _make_env(envs_dir, "rs_not_dict", "top:\n  resources_servers: not_a_mapping\n")  # rs not a dict
        _make_env(envs_dir, "broken", "key: [unclosed\n")  # malformed YAML -> load raises

        environments = _discover_environments_in_dir(envs_dir)

        assert set(environments) == {"no_rs", "top_list", "scalar_top", "rs_not_dict", "broken"}
        for entry in environments.values():
            assert entry.description is None
            assert entry.domain is None

    def test_metadata_tolerates_unset_interpolations(self, tmp_path: Path) -> None:
        # A config referencing an unset interpolation must still be discoverable: the shared metadata
        # reader reads the inline `domain` from the raw config and tolerates the unresolved value.
        envs_dir = tmp_path / "environments"
        _make_env(
            envs_dir,
            "needs_key",
            "needs_key:\n"
            "  resources_servers:\n"
            "    needs_key:\n"
            "      entrypoint: app.py\n"
            "      domain: other\n"
            "      api_key: ${some_unset_key}\n",
        )

        environments = _discover_environments_in_dir(envs_dir)
        assert "needs_key" in environments
        assert environments["needs_key"].domain == "other"


class TestRealEnvironments:
    def test_workplace_assistant_is_discoverable(self) -> None:
        # The repo ships environments/workplace_assistant/ — the registry must find it by name.
        environments = discover_environments()
        assert "workplace_assistant" in environments
        assert environments["workplace_assistant"].config_path.name == "config.yaml"


class TestDiscoverEnvironmentsAcrossRoots:
    def test_extra_root_surfaces_user_environments_alongside_builtins(self, tmp_path: Path, monkeypatch) -> None:
        _make_env(tmp_path / "environments", "custom_env", _ENV_CONFIG.format(name="custom_env"))
        monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(tmp_path))

        environments = discover_environments()

        assert "custom_env" in environments  # a user-supplied environment is discovered
        assert "workplace_assistant" in environments  # ...alongside the built-ins

    def test_cwd_is_scanned_by_default(self, tmp_path: Path, monkeypatch) -> None:
        _make_env(tmp_path / "environments", "cwd_env", _ENV_CONFIG.format(name="cwd_env"))
        monkeypatch.chdir(tmp_path)

        assert "cwd_env" in discover_environments()


class TestEnvironmentCatalog:
    def test_discovers_manifest_and_legacy_union(self, tmp_path: Path, monkeypatch) -> None:
        manifest_path = _write_manifest(tmp_path, "environments", "manifest_env", _manifest("manifest_env"))
        _make_env(tmp_path / "environments", "legacy_env", _ENV_CONFIG.format(name="legacy_env"))
        benchmark = tmp_path / "benchmarks" / "legacy_benchmark" / "config.yaml"
        benchmark.parent.mkdir(parents=True)
        benchmark.write_text(
            "bench_agent:\n"
            "  responses_api_agents:\n"
            "    simple_agent:\n"
            "      domain: math\n"
            "      description: Legacy benchmark\n"
            "      datasets:\n"
            "      - {name: test, type: benchmark, jsonl_fpath: data.jsonl}\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(registry_module, "component_search_roots", lambda: [tmp_path])

        entries = {(entry.kind, entry.name): entry for entry in discover_environment_catalog()}

        assert set(entries) == {
            ("environment", "manifest_env"),
            ("environment", "legacy_env"),
            ("benchmark", "legacy_benchmark"),
        }
        manifest_entry = entries[("environment", "manifest_env")]
        assert manifest_entry.status == "experimental"
        assert manifest_entry.manifest_path == manifest_path
        assert manifest_entry.version == "0.1.0"
        assert manifest_entry.integration_profile == "custom-gym-verifier"
        legacy_benchmark = entries[("benchmark", "legacy_benchmark")]
        assert legacy_benchmark.status == "no-manifest"
        assert legacy_benchmark.domain == "math"
        assert legacy_benchmark.description == "Legacy benchmark"

    def test_nested_manifest_identity_uses_its_relative_path(self, tmp_path: Path, monkeypatch) -> None:
        _write_manifest(tmp_path, "environments", "suite/alpha", _manifest("suite/alpha"))
        monkeypatch.setattr(registry_module, "component_search_roots", lambda: [tmp_path])

        entry = resolve_catalog_entry("suite/alpha")

        assert entry.path == tmp_path / "environments" / "suite" / "alpha"

    @pytest.mark.parametrize(
        ("tree", "path_name", "manifest"),
        [
            ("environments", "path-name", _manifest("declared-name")),
            ("benchmarks", "wrong-kind", _manifest("wrong-kind", "environment")),
        ],
    )
    def test_rejects_manifest_identity_that_disagrees_with_path(
        self,
        tmp_path: Path,
        monkeypatch,
        tree: str,
        path_name: str,
        manifest: dict,
    ) -> None:
        _write_manifest(tmp_path, tree, path_name, manifest)
        monkeypatch.setattr(registry_module, "component_search_roots", lambda: [tmp_path])

        with pytest.raises(RegistryError, match="catalog path"):
            discover_environment_catalog()

    def test_manifest_requires_a_sibling_config(self, tmp_path: Path, monkeypatch) -> None:
        _write_manifest(
            tmp_path,
            "environments",
            "missing_config",
            _manifest("missing_config"),
            with_config=False,
        )
        monkeypatch.setattr(registry_module, "component_search_roots", lambda: [tmp_path])

        with pytest.raises(RegistryError, match="sibling config.yaml"):
            discover_environment_catalog()

    def test_malformed_manifest_does_not_break_catalog_discovery(self, tmp_path: Path, monkeypatch) -> None:
        directory = tmp_path / "environments" / "draft"
        directory.mkdir(parents=True)
        (directory / "manifest.yaml").write_text("name: [invalid\n", encoding="utf-8")
        (directory / "config.yaml").write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(registry_module, "component_search_roots", lambda: [tmp_path])

        entries = discover_environment_catalog()

        assert resolve_catalog_entry("draft", "environment", entries=entries).status == "no-manifest"

    def test_root_precedence_wins_before_manifest_precedence(self, tmp_path: Path, monkeypatch) -> None:
        high = tmp_path / "high"
        low = tmp_path / "low"
        _make_env(high / "environments", "shared", _ENV_CONFIG.format(name="shared"))
        _write_manifest(low, "environments", "shared", _manifest("shared"))
        _write_manifest(high, "environments", "manifest_wins", _manifest("manifest_wins"))
        monkeypatch.setattr(registry_module, "component_search_roots", lambda: [high, low])

        entries = {(entry.kind, entry.name): entry for entry in discover_environment_catalog()}

        assert entries[("environment", "shared")].status == "no-manifest"
        assert entries[("environment", "shared")].path == high / "environments" / "shared"
        assert entries[("environment", "manifest_wins")].status == "experimental"

    def test_benchmark_manifest_suppresses_only_its_sibling_config(self, tmp_path: Path, monkeypatch) -> None:
        _write_manifest(tmp_path, "benchmarks", "suite", _manifest("suite", "benchmark"))
        flavored = tmp_path / "benchmarks" / "suite" / "strict.yaml"
        flavored.write_text(
            "agent:\n  responses_api_agents:\n    a:\n      datasets:\n      - {type: benchmark}\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(registry_module, "component_search_roots", lambda: [tmp_path])

        entries = {(entry.kind, entry.name): entry for entry in discover_environment_catalog()}

        assert entries[("benchmark", "suite")].status == "experimental"
        assert entries[("benchmark", "suite/strict")].status == "no-manifest"
        assert ("benchmark", "suite/manifest") not in entries

    def test_resolution_requires_kind_only_for_cross_kind_collision(self, tmp_path: Path, monkeypatch) -> None:
        _write_manifest(tmp_path, "environments", "shared", _manifest("shared"))
        _write_manifest(tmp_path, "benchmarks", "shared", _manifest("shared", "benchmark"))
        monkeypatch.setattr(registry_module, "component_search_roots", lambda: [tmp_path])

        entries = discover_environment_catalog()
        with pytest.raises(RegistryError, match="ambiguous"):
            resolve_catalog_entry("shared", entries=entries)
        assert resolve_catalog_entry("shared", "environment", entries=entries).kind == "environment"
        assert resolve_catalog_entry("shared", "benchmark", entries=entries).kind == "benchmark"

    def test_manifest_metadata_is_available_through_legacy_environment_api(self, tmp_path: Path) -> None:
        manifest_path = _write_manifest(tmp_path, "environments", "alpha", _manifest("alpha"))

        environments = _discover_environments_in_dir(tmp_path / "environments")

        assert environments["alpha"].manifest_path == manifest_path
        assert environments["alpha"].status == "experimental"

    def test_discovers_only_runnable_resource_server_configs(self, tmp_path: Path, monkeypatch) -> None:
        configs = tmp_path / "resources_servers" / "mcqa" / "configs"
        configs.mkdir(parents=True)
        configs.joinpath("mcqa.yaml").write_text(
            "resource:\n"
            "  resources_servers:\n"
            "    mcqa: {entrypoint: app.py, domain: knowledge, description: Multiple choice scoring}\n"
            "agent:\n"
            "  responses_api_agents:\n"
            "    simple_agent:\n"
            "      datasets:\n"
            "      - {name: train, type: train, jsonl_fpath: train.jsonl}\n",
            encoding="utf-8",
        )
        configs.joinpath("science.yaml").write_text(
            "resource:\n"
            "  resources_servers:\n"
            "    mcqa: {entrypoint: app.py, domain: knowledge}\n"
            "agent:\n"
            "  responses_api_agents:\n"
            "    simple_agent:\n"
            "      datasets:\n"
            "      - {name: test, type: benchmark, jsonl_fpath: test.jsonl}\n",
            encoding="utf-8",
        )
        pure = tmp_path / "resources_servers" / "judge" / "configs"
        pure.mkdir(parents=True)
        pure.joinpath("judge.yaml").write_text(
            "judge:\n  resources_servers:\n    judge: {entrypoint: app.py, domain: other}\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(registry_module, "component_search_roots", lambda: [tmp_path])

        entries = {(entry.kind, entry.name): entry for entry in discover_environment_catalog()}

        assert set(entries) == {
            ("environment", "resources_servers/mcqa"),
            ("benchmark", "resources_servers/mcqa/science"),
        }
        assert entries[("environment", "resources_servers/mcqa")].resources_server_selector == "mcqa"
        science = entries[("benchmark", "resources_servers/mcqa/science")]
        assert science.resources_server_selector == "mcqa/science"
        assert science.path == tmp_path / "resources_servers" / "mcqa"

        deduplicated = registry_module._discover_resource_workloads(
            tmp_path / "resources_servers",
            claimed=frozenset({("environment", "mcqa")}),
        )
        assert set(deduplicated) == {("benchmark", "resources_servers/mcqa/science")}


class TestReadEnvironmentDetails:
    def test_extracts_resources_servers_agent_datasets_and_value(self, tmp_path: Path) -> None:
        from nemo_gym.registry import read_environment_details

        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "env:\n"
            "  resources_servers:\n"
            "    my_rs:\n"
            "      domain: agent\n"
            "      description: Desc\n"
            "      value: The value\n"
            "env_agent:\n"
            "  responses_api_agents:\n"
            "    simple_agent:\n"
            "      datasets:\n"
            "      - {name: train, type: train}\n"
            "      - {name: example, type: example}\n"
        )

        details = read_environment_details(cfg)

        assert details["domain"] == "agent" and details["description"] == "Desc"
        assert details["value"] == "The value"
        assert details["resources_servers"] == ["my_rs"]
        assert details["agent"] == "simple_agent"
        assert details["datasets"] == ["train", "example"]
