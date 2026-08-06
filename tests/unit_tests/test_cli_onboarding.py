# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import nemo_gym.cli.catalog as catalog_cli
import nemo_gym.cli.onboarding as onboarding
import nemo_gym.environment_catalog as environment_catalog
import nemo_gym.environment_execution as environment_execution
from nemo_gym.config_types import ConfigError
from nemo_gym.discovery import resolve_config_paths_static
from nemo_gym.environment_inventory import MIGRATION_DRAFT_HEADER
from nemo_gym.environment_manifest import AdoptedFrom, EnvironmentManifest, ManifestError, dump_manifest, load_manifest
from nemo_gym.environment_validation import compute_composition_hash, inspect_workload, resolve_component_provenance
from nemo_gym.environment_versioning import LOCK_RELATIVE_PATH, check_or_record_version_lock


@pytest.fixture(autouse=True)
def _component_root_is_test_checkout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(environment_execution, "component_search_roots", lambda: [tmp_path])
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(tmp_path))


def _raw_workload(
    tmp_path: Path,
    *,
    resources: str = "scorer",
    agent: str = "simple_agent",
    model: str = "dummy_model",
):
    dataset_path = tmp_path / "data.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "What is 6 x 7?"}],
                }
            }
        )
        + "\n"
    )
    config_path = tmp_path / "environments" / "fixture" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}\n")
    source_components = [
        ("resources_servers", resources),
        ("responses_api_agents", agent),
    ]
    if model != "dummy_model":
        source_components.append(("responses_api_models", model))
    for group, implementation in source_components:
        source_dir = tmp_path / group / implementation
        source_dir.mkdir(parents=True, exist_ok=True)
        (source_dir / "app.py").write_text("# fixture entrypoint\n")
        (source_dir / "pylock.nemo-gym.toml").write_text('lock-version = "1.0"\npackages = []\n')
    return OmegaConf.create(
        {
            "config_paths": [str(config_path)],
            "resource": {
                "resources_servers": {
                    resources: {
                        "entrypoint": "app.py",
                        "grading_mode": "strict",
                        "provides": ["verification"],
                    }
                }
            },
            "agent": {
                "responses_api_agents": {
                    agent: {
                        "entrypoint": "app.py",
                        "requires": ["text-model", "verification"],
                        "resources_server": {"type": "resources_servers", "name": "resource"},
                        "model_server": {"type": "responses_api_models", "name": "policy_model"},
                        "datasets": [
                            {
                                "name": "example",
                                "type": "example",
                                "jsonl_fpath": str(dataset_path),
                                "num_repeats": 1,
                            }
                        ],
                    }
                }
            },
            "policy_model": {
                "responses_api_models": {
                    model: {
                        "entrypoint": "app.py",
                        "provides": ["text-model"],
                    }
                }
            },
        }
    )


def _manifest(
    tmp_path: Path,
    *,
    resources: str = "scorer",
    agent: str = "simple_agent",
    profile: str = "stock-loop",
    sandbox: str | None = None,
) -> Path:
    path = tmp_path / "environments" / "fixture" / "manifest.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    dump_manifest(
        EnvironmentManifest.model_validate(
            {
                "name": "fixture",
                "version": "1.0.0",
                "kind": "environment",
                "integration_profile": profile,
                "domain": "other",
                "description": "A local onboarding validation fixture.",
                "modality": "text",
                "licensing": "Apache-2.0",
                "authors": ["fixture-owner"],
                "reward": {"range": [0, 1], "higher_is_better": True},
                "resources_server": resources,
                "agent_server": agent,
                "model_server": "policy_model",
                "grading_mode": "strict",
                "sandbox": sandbox,
                "datasets": [
                    {
                        "name": "example",
                        "type": "example",
                        "jsonl_fpath": str(tmp_path / "data.jsonl"),
                        "num_repeats": 1,
                    }
                ],
            }
        ),
        path,
    )
    return path


def test_validation_is_static_and_returns_resolved_inspection(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path)
    manifest_path = _manifest(tmp_path)

    result = onboarding._perform_validation(
        raw,
        onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path),
    )

    assert result.manifest is not None
    assert result.inspection.profile == "stock-loop"
    assert {component.role for component in result.inspection.components} == {
        "resources_server",
        "agent_server",
        "model_server",
        "rollout_driver",
    }
    assert result.inspection.datasets[0].status == "valid"
    assert result.synced is False


def test_validation_keeps_remote_provenance_check_offline(tmp_path: Path, monkeypatch) -> None:
    raw = _raw_workload(tmp_path)
    manifest_path = _manifest(tmp_path)
    manifest = load_manifest(manifest_path)
    dump_manifest(
        EnvironmentManifest.model_validate(
            {
                **manifest.model_dump(mode="json"),
                "adopted_from": {
                    "source": "https://github.com/org/upstream",
                    "ref": "v1.2.3",
                    "reconciled": "2026-08-03",
                },
            }
        ),
        manifest_path,
    )

    checked = []
    monkeypatch.setattr(onboarding, "validate_adopted_from_reference", checked.append)
    result = onboarding._perform_validation(
        raw,
        onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path),
    )

    assert result.manifest is not None and result.manifest.adopted_from is not None
    assert checked == []


def test_publish_checks_remote_provenance_reference_once(tmp_path: Path, monkeypatch) -> None:
    raw = _raw_workload(tmp_path)
    manifest_path = _manifest(tmp_path)
    manifest = load_manifest(manifest_path)
    dump_manifest(
        EnvironmentManifest.model_validate(
            {
                **manifest.model_dump(mode="json"),
                "adopted_from": {
                    "source": "https://github.com/org/upstream",
                    "ref": "v1.2.3",
                    "reconciled": "2026-08-03",
                },
            }
        ),
        manifest_path,
    )
    checked = []
    monkeypatch.setattr(
        onboarding,
        "validate_adopted_from_reference",
        lambda provenance: checked.append(provenance),
    )
    result = onboarding._perform_validation(
        raw,
        onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path),
    )

    onboarding._validate_publication_request(
        raw,
        onboarding.PublishEnvironmentConfig(environment_ref="fixture@1.0.0", publish_owner=["@alice"]),
        result,
    )

    assert checked == [result.manifest.adopted_from]


def test_execution_preflight_requires_an_explicit_manifest_binding(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path, model="fixture_model")

    assert environment_execution.preflight_manifest_execution(raw) is None

    manifest_path = _manifest(tmp_path)
    assert environment_execution.preflight_manifest_execution(raw) is None

    raw["manifest_path"] = str(manifest_path)
    result = environment_execution.preflight_manifest_execution(raw)

    assert result is not None
    assert result.manifest_path == manifest_path.resolve()
    assert result.manifest is not None
    assert result.manifest.name == "fixture"


def test_execution_preflight_rejects_declared_profile_mismatch(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path)
    manifest_path = _manifest(tmp_path)
    raw["manifest_path"] = str(manifest_path)
    manifest = load_manifest(manifest_path)
    dump_manifest(
        EnvironmentManifest.model_validate(
            {**manifest.model_dump(mode="json"), "integration_profile": "measured-loop"}
        ),
        manifest_path,
    )

    with pytest.raises(ConfigError, match="differs from the config classifier"):
        environment_execution.preflight_manifest_execution(raw)


def test_execution_preflight_never_honors_manifest_sync_override(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path, resources="replacement")
    manifest_path = _manifest(tmp_path)
    raw["manifest_path"] = str(manifest_path)
    before = manifest_path.read_bytes()
    raw["sync_manifest"] = True

    with pytest.raises(ConfigError, match="out of sync"):
        environment_execution.preflight_manifest_execution(raw)

    assert manifest_path.read_bytes() == before


def test_implicit_manifest_resolution_only_checks_adjacent_draft_candidates(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _manifest(tmp_path)
    unrelated = tmp_path / "benchmarks" / "unrelated" / "manifest.yaml"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text(
        f"{MIGRATION_DRAFT_HEADER}\nname: unrelated\nversion: 'TODO_REQUIRED: replace semantic version'\n"
    )
    checked: list[Path] = []
    original = environment_execution.is_tracked_migration_draft

    def record_draft_check(path: Path, roots) -> bool:
        checked.append(path)
        return original(path, roots)

    monkeypatch.setattr(environment_execution, "is_tracked_migration_draft", record_draft_check)
    raw = OmegaConf.create(
        {
            "config_paths": [
                str(manifest_path.with_name("config.yaml")),
                str(manifest_path.with_name("config.yaml")),
            ]
        }
    )

    resolved = environment_execution.resolve_manifest_for_validation(raw)

    assert resolved is not None
    assert resolved[0] == manifest_path.resolve()
    assert checked == []


def test_implicit_manifest_resolution_rejects_local_untracked_migration_draft(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "environments" / "fixture" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n")
    config_path.with_name("manifest.yaml").write_text(
        f"{MIGRATION_DRAFT_HEADER}\nname: fixture\nversion: 'TODO_REQUIRED: replace semantic version'\n"
    )
    checked: list[Path] = []
    original = environment_execution.is_tracked_migration_draft

    def record_draft_check(path: Path, roots) -> bool:
        checked.append(path)
        return original(path, roots)

    monkeypatch.setattr(environment_execution, "is_tracked_migration_draft", record_draft_check)
    raw = OmegaConf.create({"config_paths": [str(config_path), str(config_path)]})

    with pytest.raises(ManifestError):
        environment_execution.resolve_manifest_for_validation(raw)
    assert checked == [config_path.with_name("manifest.yaml").resolve()]


def test_implicit_manifest_resolution_rejects_multiple_complete_manifests(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path)
    first_manifest_path = _manifest(tmp_path)
    second_config_path = tmp_path / "environments" / "second" / "config.yaml"
    second_config_path.parent.mkdir(parents=True)
    second_config_path.write_text("{}\n")
    second_manifest_path = second_config_path.with_name("manifest.yaml")
    first_manifest = load_manifest(first_manifest_path)
    dump_manifest(
        EnvironmentManifest.model_validate({**first_manifest.model_dump(mode="json"), "name": "second"}),
        second_manifest_path,
    )
    raw["config_paths"].append(str(second_config_path))

    with pytest.raises(ManifestError, match="more than one manifest"):
        environment_execution.resolve_manifest_for_validation(raw)


def test_explicit_manifest_resolution_keeps_migration_draft_selection_strict(monkeypatch) -> None:
    repo_root = Path(onboarding.__file__).resolve().parents[2]
    manifest_path = repo_root / "benchmarks" / "aime24" / "manifest.yaml"
    monkeypatch.setattr(environment_execution, "component_search_roots", lambda: [repo_root])
    raw = OmegaConf.create(
        {
            "config_paths": [str(repo_root / "benchmarks" / "aime24" / "config.yaml")],
            "manifest_path": str(manifest_path),
        }
    )

    with pytest.raises(ManifestError):
        environment_execution.preflight_manifest_execution(raw)


def test_implicit_manifest_resolution_keeps_untracked_invalid_manifest_strict(tmp_path: Path) -> None:
    config_path = tmp_path / "environments" / "fixture" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}\n")
    config_path.with_name("manifest.yaml").write_text("name: fixture\n")
    raw = OmegaConf.create({"config_paths": [str(config_path)]})

    with pytest.raises(ManifestError):
        environment_execution.resolve_manifest_for_validation(raw)


def test_execution_preflight_does_not_inspect_datasets_or_hash_components(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.environment_validation as environment_validation

    raw = _raw_workload(tmp_path, model="fixture_model")
    manifest_path = _manifest(tmp_path)
    raw["manifest_path"] = str(manifest_path)
    (tmp_path / "data.jsonl").unlink()

    def fail_full_inspection(*_args, **_kwargs):
        pytest.fail("execution preflight called the full authoring inspector")

    monkeypatch.setattr(onboarding, "inspect_workload", fail_full_inspection)
    monkeypatch.setattr(environment_validation, "_component_content_version", fail_full_inspection)

    result = environment_execution.preflight_manifest_execution(raw)

    assert result is not None
    assert result.profile == "stock-loop"


def test_execution_preflight_allows_runtime_model_options_but_rejects_locked_config_drift(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path, model="fixture_model")
    manifest_path = _manifest(tmp_path)
    manifest = load_manifest(manifest_path)
    config_path = manifest_path.with_name("config.yaml")
    raw["manifest_path"] = str(manifest_path)
    check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        config_path=config_path,
        manifest=manifest,
        composition_hash=compute_composition_hash(resolve_config_paths_static((config_path,)), manifest),
    )

    raw["policy_model"]["responses_api_models"]["fixture_model"]["temperature"] = 0.7
    assert environment_execution.preflight_manifest_execution(raw) is not None

    config_path.write_text("responses_create_params:\n  temperature: 0.2\n")
    with pytest.raises(ConfigError, match="locked composition_hash"):
        environment_execution.preflight_manifest_execution(raw)


def test_execution_preflight_requires_the_locked_config_selection(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path, model="fixture_model")
    manifest_path = _manifest(tmp_path)
    manifest = load_manifest(manifest_path)
    config_path = manifest_path.with_name("config.yaml")
    raw["manifest_path"] = str(manifest_path)
    check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        config_path=config_path,
        manifest=manifest,
        composition_hash=compute_composition_hash(resolve_config_paths_static((config_path,)), manifest),
    )
    unrelated_config = tmp_path / "environments" / "unrelated" / "config.yaml"
    unrelated_config.parent.mkdir(parents=True)
    unrelated_config.write_text("{}\n")
    raw["config_paths"] = [str(unrelated_config)]

    with pytest.raises(ConfigError, match="not running its locked config"):
        environment_execution.preflight_manifest_execution(raw)


def test_execution_preflight_uses_the_lock_to_select_one_of_multiple_config_flavors(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path, model="fixture_model")
    manifest_path = _manifest(tmp_path)
    manifest_path.with_name("config.yaml").unlink()
    configs_dir = manifest_path.parent / "configs"
    configs_dir.mkdir()
    locked_config = configs_dir / "locked.yaml"
    locked_config.write_text("{}\n")
    (configs_dir / "other.yaml").write_text("{}\n")
    manifest = load_manifest(manifest_path)
    raw["config_paths"] = [str(locked_config)]
    raw["manifest_path"] = str(manifest_path)
    check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        config_path=locked_config,
        manifest=manifest,
        composition_hash=compute_composition_hash(resolve_config_paths_static((locked_config,)), manifest),
    )

    assert environment_execution.preflight_manifest_execution(raw) is not None


def test_execution_preflight_rejects_incompatible_component_capabilities(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path, model="fixture_model")
    manifest_path = _manifest(tmp_path)
    raw["manifest_path"] = str(manifest_path)
    raw["agent"]["responses_api_agents"]["simple_agent"]["requires"] = ["verification", "audio-model"]

    with pytest.raises(ConfigError, match="Incompatible component capabilities"):
        environment_execution.preflight_manifest_execution(raw)


def test_execution_preflight_accepts_matching_registered_sandbox_provider(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    raw = _raw_workload(tmp_path, model="fixture_model")
    raw["sandbox"] = {"docker": {}}
    raw["agent"]["responses_api_agents"]["simple_agent"]["sandbox_provider"] = "sandbox"
    raw["manifest_path"] = str(_manifest(tmp_path, sandbox="docker"))

    result = environment_execution.preflight_manifest_execution(raw)

    assert result is not None
    assert result.manifest.sandbox == "docker"


@pytest.mark.parametrize(
    ("manifest_sandbox", "selected_provider", "message"),
    [
        ("docker", None, "does not select a sandbox_provider"),
        ("docker", "opensandbox", "does not match the selected registered sandbox provider"),
        ("not_registered", "not_registered", "is not registered or could not be loaded"),
    ],
)
def test_execution_preflight_rejects_invalid_sandbox_selection_before_launch(
    tmp_path: Path,
    manifest_sandbox: str,
    selected_provider: str | None,
    message: str,
) -> None:
    raw = _raw_workload(tmp_path, model="fixture_model")
    if selected_provider is not None:
        raw["agent"]["responses_api_agents"]["simple_agent"]["sandbox_provider"] = {selected_provider: {}}
    raw["manifest_path"] = str(_manifest(tmp_path, sandbox=manifest_sandbox))

    with pytest.raises(ConfigError, match=message):
        environment_execution.preflight_manifest_execution(raw)


@pytest.mark.parametrize(
    ("group", "implementation"),
    [
        ("resources_servers", "scorer"),
        ("responses_api_agents", "simple_agent"),
        ("responses_api_models", "fixture_model"),
    ],
)
def test_execution_preflight_requires_manifest_bound_entrypoints(
    tmp_path: Path,
    group: str,
    implementation: str,
) -> None:
    raw = _raw_workload(tmp_path, model="fixture_model")
    raw["manifest_path"] = str(_manifest(tmp_path))
    (tmp_path / group / implementation / "app.py").unlink()

    with pytest.raises(ConfigError, match="entrypoint .* does not exist"):
        environment_execution.preflight_manifest_execution(raw)


def test_execution_preflight_requires_every_launched_entrypoint(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path, model="fixture_model")
    raw["manifest_path"] = str(_manifest(tmp_path))
    raw["unrelated_service"] = {
        "responses_api_models": {"unrelated": {"entrypoint": "app.py"}},
    }

    with pytest.raises(ConfigError, match="unrelated.*source directory does not exist"):
        environment_execution.preflight_manifest_execution(raw)


def test_execution_preflight_rejects_entrypoint_outside_component(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path, model="fixture_model")
    raw["manifest_path"] = str(_manifest(tmp_path))
    sibling = tmp_path / "responses_api_models" / "sibling"
    sibling.mkdir()
    (sibling / "app.py").write_text("# sibling entrypoint\n")
    raw["policy_model"]["responses_api_models"]["fixture_model"]["entrypoint"] = "../sibling/app.py"

    with pytest.raises(ConfigError, match="outside component source directory"):
        environment_execution.preflight_manifest_execution(raw)


def test_execution_preflight_accepts_an_explicit_cross_registry_source_owner(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path, model="fixture_model")
    raw["manifest_path"] = str(_manifest(tmp_path))
    bridge = tmp_path / "resources_servers" / "model_bridge"
    bridge.mkdir()
    bridge_source = bridge / "app.py"
    bridge_source.write_text("VALUE = 1\n")
    bridge_config = bridge / "configs" / "model_bridge.yaml"
    bridge_config.parent.mkdir()
    bridge_config.write_text(
        "bridge_model:\n"
        "  responses_api_models:\n"
        "    fixture_model:\n"
        "      entrypoint: ../../resources_servers/model_bridge/app.py\n"
    )
    raw["config_paths"].append(str(bridge_config))
    raw["policy_model"]["responses_api_models"]["fixture_model"]["entrypoint"] = (
        "../../resources_servers/model_bridge/app.py"
    )

    assert environment_execution.preflight_manifest_execution(raw) is not None
    model_provenance = next(item for item in resolve_component_provenance(raw) if item.role == "model_server")
    assert model_provenance.entrypoint_source_directory == bridge
    assert bridge in model_provenance.dependency_directories
    baseline = inspect_workload(raw).composition_hash
    bridge_source.write_text("VALUE = 2\n")

    assert inspect_workload(raw).composition_hash != baseline


def test_full_manifest_validation_requires_manifest_bound_entrypoints(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path)
    manifest_path = _manifest(tmp_path)
    (tmp_path / "responses_api_agents" / "simple_agent" / "app.py").unlink()

    with pytest.raises(ConfigError, match="entrypoint .* does not exist"):
        onboarding._perform_validation(
            raw,
            onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path),
        )


def test_missing_manifest_entrypoint_fails_before_ray(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.cli.env as cli_env

    raw = _raw_workload(tmp_path)
    raw["manifest_path"] = str(_manifest(tmp_path))
    (tmp_path / "resources_servers" / "scorer" / "app.py").unlink()
    events: list[str] = []

    monkeypatch.setattr(cli_env, "initialize_ray", lambda: events.append("ray"))

    with pytest.raises(ConfigError, match="entrypoint .* does not exist"):
        cli_env.RunHelper().start(None, global_config_dict=raw)

    assert events == []


def test_run_helper_rejects_static_dummy_model_before_ray(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.cli.env as cli_env

    raw = _raw_workload(tmp_path)
    raw["manifest_path"] = str(_manifest(tmp_path))
    events: list[str] = []
    monkeypatch.setattr(cli_env, "initialize_ray", lambda: events.append("ray"))

    with pytest.raises(ConfigError, match="dummy_model.*source directory does not exist"):
        cli_env.RunHelper().start(None, global_config_dict=raw)

    assert events == []


def test_run_helper_rejects_empty_entrypoint_before_ray(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.cli.env as cli_env

    raw = _raw_workload(tmp_path)
    raw["manifest_path"] = str(_manifest(tmp_path))
    raw["resource"]["resources_servers"]["scorer"]["entrypoint"] = ""
    events: list[str] = []
    monkeypatch.setattr(cli_env, "initialize_ray", lambda: events.append("ray"))

    with pytest.raises(ConfigError, match="does not declare a Python entrypoint"):
        cli_env.RunHelper().start(None, global_config_dict=raw)

    assert events == []


def test_run_helper_rejects_unknown_server_group_before_ray(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.cli.env as cli_env

    raw = _raw_workload(tmp_path, model="fixture_model")
    raw["manifest_path"] = str(_manifest(tmp_path))
    raw["rogue"] = {"custom_servers": {"bad": {"entrypoint": "missing.py"}}}
    events: list[str] = []
    monkeypatch.setattr(cli_env, "initialize_ray", lambda: events.append("ray"))

    with pytest.raises(ConfigError, match="unsupported server group 'custom_servers'"):
        cli_env.RunHelper().start(None, global_config_dict=raw)

    assert events == []


def test_run_helper_preflights_inherited_config_before_ray(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.cli.env as cli_env
    import nemo_gym.global_config as global_config

    raw = _raw_workload(tmp_path, model="fixture_model")
    raw["manifest_path"] = str(_manifest(tmp_path))
    raw["agent"]["responses_api_agents"]["simple_agent"]["requires"].append("missing-protocol")
    events: list[str] = []
    monkeypatch.setattr(global_config, "_GLOBAL_CONFIG_DICT", None)
    monkeypatch.setenv(global_config.NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, OmegaConf.to_yaml(raw))
    monkeypatch.setattr(cli_env, "initialize_ray", lambda: events.append("ray"))

    with pytest.raises(ConfigError, match="missing-protocol"):
        cli_env.RunHelper().start(None)

    assert events == []


def test_eval_preflight_reaches_data_preparation_with_a_missing_dataset(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.cli.eval as cli_eval
    import nemo_gym.environment_execution as environment_execution
    import nemo_gym.train_data_utils as train_data_utils

    raw = _raw_workload(tmp_path, model="fixture_model")
    manifest_path = _manifest(tmp_path)
    raw["manifest_path"] = str(manifest_path)
    raw["output_jsonl_fpath"] = str(tmp_path / "rollouts.jsonl")
    raw["split"] = "benchmark"
    (tmp_path / "data.jsonl").unlink()
    events: list[str] = []

    def parsed_config():
        events.append("preflight")
        environment_execution.preflight_manifest_execution(raw)
        return raw

    class DataPreparationReached(RuntimeError):
        pass

    class RecordingTrainDataProcessor:
        def run(self, _config):
            events.append("data-preparation")
            raise DataPreparationReached

    monkeypatch.setattr(train_data_utils, "TrainDataProcessor", RecordingTrainDataProcessor)
    monkeypatch.setattr(cli_eval, "get_global_config_dict", parsed_config)

    with pytest.raises(DataPreparationReached):
        cli_eval.e2e_rollout_collection()

    assert events == ["preflight", "data-preparation"]


def test_eval_loads_custom_driver_before_data_preparation_or_services(tmp_path: Path, monkeypatch, capsys) -> None:
    import nemo_gym.cli.eval as cli_eval
    import nemo_gym.train_data_utils as train_data_utils

    raw = _raw_workload(tmp_path)
    raw["output_jsonl_fpath"] = str(tmp_path / "rollouts.jsonl")
    raw["split"] = "benchmark"
    raw["rollout_collection_driver"] = "missing_driver.module:run"
    events: list[str] = []

    def missing_driver(_module_name):
        events.append("driver-import")
        raise ModuleNotFoundError("missing dependency")

    def unexpected_data_preparation(_self, _config):
        events.append("data-preparation")

    def unexpected_service_start(_self, *_args, **_kwargs):
        events.append("service-start")

    monkeypatch.setattr(cli_eval, "get_global_config_dict", lambda: raw)
    monkeypatch.setattr(cli_eval.importlib, "import_module", missing_driver)
    monkeypatch.setattr(train_data_utils.TrainDataProcessor, "run", unexpected_data_preparation)
    monkeypatch.setattr(cli_eval.RunHelper, "start", unexpected_service_start)

    with pytest.raises(SystemExit) as error:
        cli_eval.e2e_rollout_collection()

    assert error.value.code == 1
    assert events == ["driver-import"]
    assert "Cannot load rollout_collection_driver" in capsys.readouterr().out


def test_no_serve_eval_invokes_custom_driver_and_captures_bundle(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.cli.eval as cli_eval
    import nemo_gym.rollout_collection as rollout_collection
    import nemo_gym.trajectory_bundle as trajectory_bundle

    raw = _raw_workload(tmp_path)
    raw["input_jsonl_fpath"] = str(tmp_path / "data.jsonl")
    raw["output_jsonl_fpath"] = str(tmp_path / "rollouts.jsonl")
    raw["rollout_collection_driver"] = "package.driver:collect"
    captured = SimpleNamespace(rollout_driver="package.driver:collect")
    events: list[tuple[str, object]] = []

    async def driver(config, resolved_config):
        events.append(("driver", (config, resolved_config)))

    async def unexpected_stock_run(_self, _config):
        events.append(("stock", _config))

    def write_bundle(**kwargs):
        events.append(("bundle", kwargs))
        return tmp_path / "rollouts.bundle.json"

    monkeypatch.setattr(cli_eval, "get_global_config_dict", lambda: raw)
    monkeypatch.setattr(cli_eval, "validate_rollout_driver_contract", lambda config: ((), ()))
    monkeypatch.setattr(cli_eval, "_load_rollout_collection_driver", lambda _reference: driver)
    monkeypatch.setattr(rollout_collection.RolloutCollectionHelper, "run_from_config", unexpected_stock_run)
    monkeypatch.setattr(trajectory_bundle, "captured_environment_from_config", lambda _config: captured)
    monkeypatch.setattr(trajectory_bundle, "write_trajectory_bundle", write_bundle)

    cli_eval.collect_rollouts()

    assert [event for event, _payload in events] == ["driver", "bundle"]
    typed_config, resolved_config = events[0][1]
    assert isinstance(typed_config, rollout_collection.RolloutCollectionConfig)
    assert resolved_config["rollout_collection_driver"] == "package.driver:collect"
    bundle_kwargs = events[1][1]
    assert bundle_kwargs["environment"].rollout_driver == "package.driver:collect"


def test_no_serve_eval_without_custom_driver_uses_stock_collector(tmp_path: Path, monkeypatch) -> None:
    import nemo_gym.cli.eval as cli_eval
    import nemo_gym.rollout_collection as rollout_collection
    import nemo_gym.trajectory_bundle as trajectory_bundle

    raw = _raw_workload(tmp_path)
    raw["input_jsonl_fpath"] = str(tmp_path / "data.jsonl")
    raw["output_jsonl_fpath"] = str(tmp_path / "rollouts.jsonl")
    events: list[str] = []

    async def stock_run(_self, config):
        assert isinstance(config, rollout_collection.RolloutCollectionConfig)
        events.append("stock")

    monkeypatch.setattr(cli_eval, "get_global_config_dict", lambda: raw)
    monkeypatch.setattr(rollout_collection.RolloutCollectionHelper, "run_from_config", stock_run)
    monkeypatch.setattr(
        trajectory_bundle,
        "write_trajectory_bundle",
        lambda **_kwargs: events.append("bundle"),
    )

    cli_eval.collect_rollouts()

    assert events == ["stock"]


def test_validate_command_rejects_empty_default_workload(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        onboarding,
        "get_global_config_dict",
        lambda **_kwargs: OmegaConf.create(
            {
                "policy_model": {"responses_api_models": {"dummy_model": {"entrypoint": "app.py"}}},
                "json": True,
            }
        ),
    )

    with pytest.raises(SystemExit) as error:
        onboarding.validate_environment()

    assert error.value.code == 1
    assert "requires a runnable target or config" in capsys.readouterr().out


def test_validation_can_sync_unpinned_mirror_but_never_a_pinned_agent(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path, resources="replacement")
    manifest_path = _manifest(tmp_path)

    with pytest.raises(ConfigError, match="out of sync") as error:
        onboarding._perform_validation(
            raw,
            onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path),
        )
    assert "--sync" in str(error.value)

    result = onboarding._perform_validation(
        raw,
        onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path, sync_manifest=True),
    )
    assert result.synced is True
    assert load_manifest(manifest_path).resources_server == "replacement"

    pinned_raw = _raw_workload(tmp_path, resources="replacement", agent="measured_agent")
    with pytest.raises(ConfigError, match="pins the agent server"):
        onboarding._perform_validation(
            pinned_raw,
            onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path, sync_manifest=True),
        )


def test_validation_reinspects_synced_manifest_before_returning(tmp_path: Path) -> None:
    raw = _raw_workload(tmp_path)
    manifest_path = _manifest(tmp_path)
    original = load_manifest(manifest_path)
    stale = EnvironmentManifest.model_validate({**original.model_dump(mode="json"), "grading_mode": "legacy-mode"})
    dump_manifest(stale, manifest_path)
    stale_inspection = onboarding.inspect_workload(
        raw,
        strict_missing_datasets=True,
        standard_prompt_config=stale.standard_prompt_config,
        manifest=stale,
    )

    result = onboarding._perform_validation(
        raw,
        onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path, sync_manifest=True),
    )

    persisted = load_manifest(manifest_path)
    expected = onboarding.inspect_workload(
        raw,
        strict_missing_datasets=True,
        standard_prompt_config=persisted.standard_prompt_config,
        manifest=persisted,
    )
    assert persisted.grading_mode == "strict"
    assert result.inspection.composition_hash == expected.composition_hash
    assert result.inspection.compatibility_decisions == expected.compatibility_decisions
    assert result.inspection.composition_hash != stale_inspection.composition_hash


def test_validation_warns_and_publish_rejects_profile_mismatch(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    raw = _raw_workload(tmp_path)
    manifest_path = _manifest(tmp_path)
    stock_manifest = load_manifest(manifest_path)
    dump_manifest(
        EnvironmentManifest.model_validate(
            {**stock_manifest.model_dump(mode="json"), "integration_profile": "measured-loop"}
        ),
        manifest_path,
    )
    before_manifest = manifest_path.read_bytes()
    result = onboarding._perform_validation(
        raw,
        onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path, sync_manifest=True),
    )
    assert result.profile_warning is not None
    assert "differs from the config classifier" in result.profile_warning
    assert manifest_path.read_bytes() == before_manifest

    monkeypatch.setattr(onboarding, "get_global_config_dict", lambda **_kwargs: raw)
    onboarding.validate_environment()
    validation_output = " ".join(capsys.readouterr().out.split())
    assert "Warning" in validation_output
    assert "differs from the config classifier" in validation_output

    publish_raw = OmegaConf.merge(
        raw,
        {
            "environment_ref": "fixture@1.0.0",
            "publish_owner": ["@nvidia-nemo/automation"],
        },
    )
    monkeypatch.setattr(onboarding, "get_global_config_dict", lambda **_kwargs: publish_raw)

    with pytest.raises(SystemExit):
        onboarding.publish_environment()
    assert "differs from the config classifier" in " ".join(capsys.readouterr().out.split())


def test_init_command_scaffolds_from_reserved_config(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        onboarding,
        "get_global_config_dict",
        lambda **_kwargs: OmegaConf.create(
            {
                "init_name": "guided",
                "init_kind": "benchmark",
                "init_profile": "stock-loop",
                "init_domain": "math",
                "init_description": "A guided arithmetic benchmark.",
                "init_authors": ["contributor"],
                "init_canonical_split": "held_out_test",
            }
        ),
    )

    onboarding.init_environment()

    manifest = load_manifest(tmp_path / "benchmarks" / "guided" / "manifest.yaml")
    assert manifest.name == "guided"
    assert manifest.kind.value == "benchmark"
    assert manifest.canonical_split == "held_out_test"
    assert "gym env validate guided" in " ".join(capsys.readouterr().out.split())


def test_benchmark_init_requires_explicit_canonical_split(monkeypatch) -> None:
    monkeypatch.setattr(
        onboarding,
        "get_global_config_dict",
        lambda **_kwargs: OmegaConf.create({"init_name": "guided", "init_kind": "benchmark"}),
    )

    with pytest.raises(SystemExit):
        onboarding.init_environment()


def test_init_requires_explicit_authorship(monkeypatch) -> None:
    monkeypatch.setattr(
        onboarding,
        "get_global_config_dict",
        lambda **_kwargs: OmegaConf.create({"init_name": "guided", "init_kind": "environment"}),
    )

    with pytest.raises(SystemExit):
        onboarding.init_environment()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"description": "Starter environment for fixture"}, "searchable task description"),
        ({"authors": ["TODO"]}, "named authors"),
        ({"licensing": "unknown"}, "settled SPDX license"),
    ],
)
def test_publish_rejects_unchanged_scaffold_metadata(tmp_path: Path, updates: dict, message: str) -> None:
    manifest = load_manifest(_manifest(tmp_path)).model_copy(update=updates)

    with pytest.raises(ConfigError, match=message):
        onboarding._validate_publish_metadata(manifest)


def test_publish_rejects_nonpublic_upstream_provenance(tmp_path: Path) -> None:
    manifest = load_manifest(_manifest(tmp_path)).model_copy(
        update={
            "adopted_from": AdoptedFrom(
                source="file:///private/upstream",
                ref="0123456789abcdef0123456789abcdef01234567",
                reconciled="2026-08-06",
            )
        }
    )

    with pytest.raises(ConfigError, match="public HTTPS source"):
        onboarding._validate_publish_metadata(manifest)


def test_manifest_bulk_edit_command_validates_and_writes_selection(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    manifest_path = _manifest(tmp_path)
    monkeypatch.setattr(onboarding, "select_manifest_paths", lambda _filters: (manifest_path,))
    monkeypatch.setattr(
        onboarding,
        "get_global_config_dict",
        lambda **_kwargs: OmegaConf.create(
            {
                "manifest_names": ["fixture"],
                "manifest_set": ["lifecycle=deprecated", "requires=[text-model, observability]"],
                "json": True,
            }
        ),
    )

    onboarding.edit_environment_manifests()

    payload = json.loads(capsys.readouterr().out)
    assert payload["changed"] == [str(manifest_path.resolve())]
    manifest = load_manifest(manifest_path)
    assert manifest.lifecycle.value == "deprecated"
    assert manifest.requires == ["text-model", "observability"]


def test_codeowners_update_is_idempotent_and_supports_dry_run(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    manifest_path = tmp_path / "environments" / "fixture" / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("name: fixture\n")

    plan = onboarding._plan_codeowners_updates(
        manifest_path,
        (manifest_path.parent,),
        ("@alice", "@team-review"),
    )
    assert plan.changed is True
    assert not plan.path.exists()

    onboarding._commit_codeowners(plan)
    assert plan.path.read_text() == "/environments/fixture/ @alice @team-review\n"

    repeated = onboarding._plan_codeowners_updates(
        manifest_path,
        (manifest_path.parent,),
        ("@alice", "@team-review"),
    )
    assert repeated.changed is False
    assert repeated.rules == plan.rules


def test_codeowners_update_refuses_to_transfer_an_existing_rule(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    manifest_path = tmp_path / "environments" / "fixture" / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("name: fixture\n")
    codeowners = tmp_path / ".github" / "CODEOWNERS"
    codeowners.parent.mkdir()
    original = "/environments/fixture/\t@maintainers\n"
    codeowners.write_text(original)

    with pytest.raises(ConfigError, match="Refusing to replace existing CODEOWNERS rule"):
        onboarding._plan_codeowners_updates(manifest_path, (manifest_path.parent,), ("@alice",))

    assert codeowners.read_text() == original


def test_codeowners_commit_rejects_a_change_after_preflight(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    manifest_path = tmp_path / "environments" / "fixture" / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("name: fixture\n")
    plan = onboarding._plan_codeowners_updates(
        manifest_path,
        (manifest_path.parent,),
        ("@alice",),
    )
    plan.path.parent.mkdir()
    concurrent_content = "# Concurrent policy update\n"
    plan.path.write_text(concurrent_content)

    with pytest.raises(ConfigError, match="changed after publication preflight"):
        onboarding._commit_codeowners(plan)

    assert plan.path.read_text() == concurrent_content


@pytest.mark.parametrize("linked_level", ["directory", "file"])
def test_codeowners_update_rejects_symlink_write_paths(tmp_path: Path, linked_level: str) -> None:
    (tmp_path / ".git").mkdir()
    manifest_path = tmp_path / "environments" / "fixture" / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("name: fixture\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "protected"
    protected.write_text("do not overwrite\n")

    if linked_level == "directory":
        (tmp_path / ".github").symlink_to(outside, target_is_directory=True)
    else:
        (tmp_path / ".github").mkdir()
        (tmp_path / ".github" / "CODEOWNERS").symlink_to(protected)

    with pytest.raises(ConfigError, match="symbolic-link"):
        onboarding._plan_codeowners_updates(manifest_path, (manifest_path.parent,), ("@alice",))

    assert protected.read_text() == "do not overwrite\n"


def test_publish_checks_identity_and_owns_only_new_recipe_components(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    raw = _raw_workload(tmp_path, resources="fixture")
    manifest_path = _manifest(tmp_path, resources="fixture")
    (tmp_path / "resources_servers" / "fixture").mkdir(parents=True, exist_ok=True)
    result = onboarding._perform_validation(
        raw,
        onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path),
    )
    publish_raw = OmegaConf.merge(
        raw,
        {
            "environment_ref": "fixture@1.0.0",
            "publish_owner": ["@nvidia-nemo/automation"],
            "publish_dry_run": False,
            "json": True,
        },
    )
    monkeypatch.setattr(onboarding, "get_global_config_dict", lambda **_kwargs: publish_raw)
    monkeypatch.setattr(onboarding, "_perform_validation", lambda *_args: result)

    onboarding.publish_environment()

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "fixture"
    assert payload["codeowners_rules"] == [
        "/environments/fixture/ @nvidia-nemo/automation",
        "/resources_servers/fixture/ @nvidia-nemo/automation",
    ]
    assert payload["composition_hash"] == result.inspection.composition_hash
    assert payload["version_lock_changed"] is True
    lock_payload = json.loads((tmp_path / LOCK_RELATIVE_PATH).read_text())
    lock_record = lock_payload["environments"]["environment:fixture@1.0.0"]
    assert lock_record["composition_hash"] == result.inspection.composition_hash
    assert lock_record == {
        "composition_hash": result.inspection.composition_hash,
        "manifest": "environments/fixture/manifest.yaml",
        "config": "environments/fixture/config.yaml",
    }
    codeowners = (tmp_path / ".github" / "CODEOWNERS").read_text()
    assert "/responses_api_agents/simple_agent/" not in codeowners

    publish_raw.environment_ref = "someone_else"
    with pytest.raises(SystemExit):
        onboarding.publish_environment()
    assert "refusing to publish a different unit" in " ".join(capsys.readouterr().out.split())


@pytest.mark.parametrize("preexisting_codeowners", [False, True])
def test_publish_lock_failure_rolls_back_codeowners(
    tmp_path: Path,
    monkeypatch,
    capsys,
    preexisting_codeowners: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    raw = _raw_workload(tmp_path)
    manifest_path = _manifest(tmp_path)
    result = onboarding._perform_validation(
        raw,
        onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path),
    )
    codeowners = tmp_path / ".github" / "CODEOWNERS"
    original_codeowners = "# Existing policy\n" if preexisting_codeowners else None
    if original_codeowners is not None:
        codeowners.parent.mkdir(exist_ok=True)
        codeowners.write_text(original_codeowners)
    publish_raw = OmegaConf.merge(
        raw,
        {
            "environment_ref": "fixture@1.0.0",
            "publish_owner": ["@alice"],
            "json": True,
        },
    )
    real_version_lock = onboarding.check_or_record_version_lock

    def fail_version_lock(**kwargs):
        if kwargs["dry_run"]:
            return real_version_lock(**kwargs)
        raise ConfigError("forced version-lock write failure")

    monkeypatch.setattr(onboarding, "get_global_config_dict", lambda **_kwargs: publish_raw)
    monkeypatch.setattr(onboarding, "_perform_validation", lambda *_args: result)
    monkeypatch.setattr(onboarding, "check_or_record_version_lock", fail_version_lock)

    with pytest.raises(SystemExit):
        onboarding.publish_environment()

    assert "forced version-lock write failure" in capsys.readouterr().out
    if original_codeowners is None:
        assert not codeowners.exists()
    else:
        assert codeowners.read_text() == original_codeowners
    assert not (tmp_path / LOCK_RELATIVE_PATH).exists()

    monkeypatch.setattr(environment_catalog, "component_search_roots", lambda: [tmp_path])
    catalog = environment_catalog.discover_environment_catalog(include_legacy=False)
    assert catalog.entries == ()


def test_publish_payload_has_no_certificate_contract(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    raw = _raw_workload(tmp_path)
    manifest_path = _manifest(tmp_path)
    result = onboarding._perform_validation(
        raw,
        onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path),
    )
    publish_raw = OmegaConf.merge(
        raw,
        {
            "environment_ref": "fixture@1.0.0",
            "publish_owner": ["@alice"],
            "json": True,
        },
    )
    monkeypatch.setattr(onboarding, "get_global_config_dict", lambda **_kwargs: publish_raw)
    monkeypatch.setattr(onboarding, "_perform_validation", lambda *_args: result)

    onboarding.publish_environment()

    payload = json.loads(capsys.readouterr().out)
    assert payload["environment_version_key"] == "environment:fixture@1.0.0"
    assert "status" not in payload
    assert "validation" not in payload
    assert "certificate_leaf" not in payload


def test_publish_rejects_manifest_override_outside_selected_catalog_entry(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    raw = _raw_workload(tmp_path)
    manifest_path = _manifest(tmp_path)
    result = onboarding._perform_validation(raw, onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path))
    canonical_manifest = tmp_path / "canonical" / "manifest.yaml"
    canonical_config = tmp_path / "canonical" / "config.yaml"
    monkeypatch.setattr(
        onboarding,
        "resolve_catalog_reference",
        lambda *_args, **_kwargs: SimpleNamespace(
            name="fixture",
            kind="environment",
            version="1.0.0",
            manifest_path=canonical_manifest,
            config_path=canonical_config,
        ),
    )
    monkeypatch.setattr(
        onboarding,
        "get_global_config_dict",
        lambda **_kwargs: OmegaConf.merge(
            raw,
            {
                "manifest_path": str(manifest_path),
                "environment_ref": "fixture@1.0.0",
                "publish_owner": ["@alice"],
            },
        ),
    )
    monkeypatch.setattr(onboarding, "_perform_validation", lambda *_args: result)

    with pytest.raises(SystemExit):
        onboarding.publish_environment()

    assert "refusing to publish a different catalog entry" in " ".join(capsys.readouterr().out.split())
    assert not (tmp_path / ".github" / "CODEOWNERS").exists()


def test_component_ownership_excludes_head_paths_but_keeps_staged_new_scaffold(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    shared_resource = tmp_path / "resources_servers" / "fixture"
    shared_resource.mkdir(parents=True)
    (shared_resource / "app.py").write_text("# shared verifier\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "resources_servers/fixture/app.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.com",
            "commit",
            "-qm",
            "shared component",
        ],
        check=True,
    )
    staged_agent = tmp_path / "responses_api_agents" / "fixture_agent"
    staged_agent.mkdir(parents=True)
    (staged_agent / "app.py").write_text("# new scaffold agent\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "responses_api_agents/fixture_agent/app.py"], check=True)
    raw = _raw_workload(tmp_path, resources="fixture", agent="fixture_agent")
    manifest_path = _manifest(
        tmp_path,
        resources="fixture",
        agent="fixture_agent",
        profile="measured-loop",
    )
    result = onboarding._perform_validation(raw, onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path))

    assert onboarding._owned_component_directories(result) == (
        manifest_path.parent,
        staged_agent,
    )


def test_owner_validation_requires_github_handles() -> None:
    assert onboarding._normalize_owners(["alice", "@team-name", "@nvidia-nemo/automation", "alice"]) == (
        "@alice",
        "@team-name",
        "@nvidia-nemo/automation",
    )
    with pytest.raises(ConfigError, match="GitHub user/team handle"):
        onboarding._normalize_owners(["Ada Lovelace"])


def test_manifest_registry_path_identity_is_enforced(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _manifest(tmp_path)
    registry_path = tmp_path / "environments" / "different_name" / "manifest.yaml"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(manifest_path.read_text())
    monkeypatch.setattr(environment_execution, "component_search_roots", lambda: [tmp_path])

    with pytest.raises(ConfigError, match="registry path identifies 'different_name'"):
        onboarding._manifest_for_validation(
            {},
            onboarding.ValidateEnvironmentConfig(manifest_path=registry_path),
        )


def test_manifest_location_rejects_unsupported_registry_tree(tmp_path: Path, monkeypatch) -> None:
    source = _manifest(tmp_path)
    manifest = load_manifest(source)
    unsupported = tmp_path / ".github" / "manifest.yaml"
    unsupported.parent.mkdir()
    dump_manifest(manifest, unsupported)
    monkeypatch.setattr(environment_execution, "component_search_roots", lambda: [tmp_path])

    with pytest.raises(ConfigError, match="outside the environment registry"):
        onboarding._manifest_for_validation(
            {"config_paths": [str(tmp_path / "environments" / "fixture" / "config.yaml")]},
            onboarding.ValidateEnvironmentConfig(manifest_path=unsupported),
        )


def test_publish_resolution_keeps_exact_reference_on_live_authoring_source(tmp_path, monkeypatch) -> None:
    (tmp_path / ".git").mkdir()
    manifest_path = _manifest(tmp_path)
    config_path = manifest_path.with_name("config.yaml")
    config_path.write_text("fixture: {}\n")
    monkeypatch.setattr(environment_catalog, "component_search_roots", lambda: [tmp_path])

    entry = onboarding.resolve_catalog_reference(
        "fixture@1.0.0",
        include_unpublished=True,
        allow_version=True,
    )

    assert entry.manifest_path == manifest_path.resolve()
    assert entry.config_path == config_path.resolve()


def test_runtime_exact_reference_is_unsupported() -> None:
    with pytest.raises(ValueError, match="not executable.*accepted only"):
        onboarding.resolve_catalog_reference("fixture@1.0.0", include_unpublished=True)


def test_exact_reference_resolution_does_not_walk_the_full_catalog(tmp_path, monkeypatch) -> None:
    manifest_path = _manifest(tmp_path)
    manifest_path.with_name("config.yaml").write_text("fixture: {}\n")
    monkeypatch.setattr(environment_catalog, "component_search_roots", lambda: [tmp_path])
    monkeypatch.setattr(
        environment_catalog,
        "discover_environment_catalog",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("exact lookup must not scan the catalog")),
    )

    entry = onboarding.resolve_catalog_reference("fixture", include_unpublished=True)

    assert entry.manifest_path == manifest_path.absolute()
    assert entry.config_path == manifest_path.with_name("config.yaml").resolve()


def test_component_discovery_filters_declared_capabilities(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "resources_servers" / "shared" / "configs" / "shared.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        """\
shared:
  resources_servers:
    verifier:
      entrypoint: app.py
      provides: [verification, tools]
      requires: text-model
"""
    )
    monkeypatch.setattr("nemo_gym.discovery.component_search_roots", lambda: [tmp_path])
    monkeypatch.setattr(onboarding, "list_providers", lambda: [])

    records = onboarding._discover_declared_components()

    assert len(records) == 1
    assert records[0].name == "shared"
    assert records[0].implementation == "verifier"
    assert records[0].instance == "shared"
    assert records[0].provides == ("verification", "tools")
    assert records[0].requires == ("text-model",)


def test_component_discovery_keeps_config_flavor_contracts_separate(tmp_path: Path, monkeypatch) -> None:
    configs = tmp_path / "resources_servers" / "shared" / "configs"
    configs.mkdir(parents=True)
    (configs / "a.yaml").write_text(
        "shared_a:\n  resources_servers:\n    verifier:\n      entrypoint: app.py\n      provides: [verification]\n"
    )
    (configs / "b.yaml").write_text(
        "shared_b:\n  resources_servers:\n    verifier:\n      entrypoint: app.py\n      provides: [tools]\n"
        "      requires: [sandbox:docker]\n"
    )
    monkeypatch.setattr("nemo_gym.discovery.component_search_roots", lambda: [tmp_path])
    monkeypatch.setattr(onboarding, "list_providers", lambda: [])

    records = onboarding._discover_declared_components()

    assert [(record.name, record.implementation, record.provides, record.requires) for record in records] == [
        ("shared/a", "verifier", ("verification",), ()),
        ("shared/b", "verifier", ("tools",), ("sandbox:docker",)),
    ]


def test_component_listing_includes_registered_sandbox_providers(monkeypatch, capsys) -> None:
    monkeypatch.setattr(onboarding, "discover_resources_servers", lambda: {})
    monkeypatch.setattr(onboarding, "discover_models", lambda: {})
    monkeypatch.setattr(onboarding, "discover_agents", lambda: {})
    monkeypatch.setattr(onboarding, "list_providers", lambda: ["docker", "opensandbox"])
    monkeypatch.setattr(
        onboarding,
        "get_global_config_dict",
        lambda: {"component_provides": "sandbox:*", "json": True},
    )

    onboarding.list_components()

    provider_root = Path(onboarding.__file__).resolve().parents[1] / "sandbox/providers"
    assert json.loads(capsys.readouterr().out) == [
        {
            "config_path": str(provider_root / provider / "configs" / f"{provider}.yaml"),
            "implementation": provider,
            "instance": "sandbox",
            "kind": "sandbox-provider",
            "name": provider,
            "provides": [f"sandbox:{provider}"],
            "requires": [],
        }
        for provider in ("docker", "opensandbox")
    ]


def test_catalog_migration_diagnostics_are_compact_unless_facts_requested(tmp_path: Path, capsys) -> None:
    catalog = environment_catalog.EnvironmentCatalog(
        entries=(),
        coverage=environment_catalog.CatalogCoverage(
            total=0,
            with_manifest=0,
            without_manifest=0,
            invalid_manifests=0,
        ),
        issues=(
            environment_catalog.CatalogIssue(
                path=tmp_path / "one" / "manifest.yaml",
                message="first verbose draft error",
                code="migration-draft",
            ),
            environment_catalog.CatalogIssue(
                path=tmp_path / "two" / "manifest.yaml",
                message="second verbose draft error",
                code="migration-draft",
            ),
            environment_catalog.CatalogIssue(
                path=tmp_path / "legacy",
                message="verbose ambiguity",
                code="ambiguous-legacy-resource",
            ),
        ),
    )

    catalog_cli._render_catalog_issues(catalog)
    compact = " ".join(capsys.readouterr().out.split())
    assert "2 generated migration drafts" in compact
    assert "1 ambiguous legacy component" in compact
    assert "verbose" not in compact

    catalog_cli._render_catalog_issues(catalog, verbose=True)
    detailed = capsys.readouterr().out
    assert "first verbose draft error" in detailed
    assert "second verbose draft error" in detailed
    assert "verbose ambiguity" in detailed
