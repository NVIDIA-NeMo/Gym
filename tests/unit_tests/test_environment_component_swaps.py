# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import nemo_gym.cli.onboarding as onboarding
import nemo_gym.environment_execution as environment_execution
from nemo_gym.config_types import ConfigError
from nemo_gym.discovery import resolve_config_paths_static
from nemo_gym.environment_manifest import EnvironmentManifest, dump_manifest, load_manifest
from nemo_gym.environment_validation import compute_composition_hash, inspect_workload, resolve_composition_mirror
from nemo_gym.environment_versioning import check_or_record_version_lock
from nemo_gym.global_config import GlobalConfigDictParserConfig, StaticValidationConfigParser
from nemo_gym.trajectory_bundle import captured_environment_from_config


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(payload), path)
    return path


def _write_simple_agent_source(root: Path, implementation: str, profile: str) -> None:
    method = "responses" if profile == "measured-loop" else "run"
    source_path = root / "responses_api_agents" / implementation / "app.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        "from responses_api_agents.simple_agent.app import SimpleAgent\n\n"
        f"class FixtureAgent(SimpleAgent):\n    async def {method}(self):\n        pass\n"
    )


def _manifest(
    tmp_path: Path,
    *,
    profile: str = "external-loop",
    model_instance: str = "policy_model",
    rollout_driver: str = "fixture.driver:collect",
) -> Path:
    path = tmp_path / "environments" / "r4_fixture" / "manifest.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "r4_fixture",
        "version": "1.0.0",
        "kind": "environment",
        "integration_profile": profile,
        "domain": "other",
        "description": "Component replacement fixture.",
        "modality": "text",
        "licensing": "Apache-2.0",
        "authors": ["fixture-owner"],
        "reward": {"range": [0, 1], "higher_is_better": True},
        "resources_server": "base_resource_r4",
        "agent_server": "base_agent_r4",
        "model_server": model_instance,
        "grading_mode": "strict",
        "datasets": [
            {
                "name": "fixture",
                "type": "example",
                "jsonl_fpath": str(tmp_path / "data.jsonl"),
            }
        ],
    }
    if profile == "custom-driver":
        payload["rollout_driver"] = rollout_driver
    dump_manifest(EnvironmentManifest.model_validate(payload), path)
    return path


def _base_config(
    tmp_path: Path,
    *,
    profile: str = "external-loop",
    model_instance: str = "policy_model",
    rollout_driver: str = "fixture.driver:collect",
) -> Path:
    (tmp_path / "data.jsonl").write_text(
        json.dumps({"responses_create_params": {"input": [{"role": "user", "content": "question"}]}}) + "\n"
    )
    payload = {
        "resource_instance": {
            "resources_servers": {
                "base_resource_r4": {
                    "entrypoint": "app.py",
                    "domain": "other",
                    "grading_mode": "strict",
                    "provides": ["verification"],
                }
            }
        },
        "agent_instance": {
            "responses_api_agents": {
                "base_agent_r4": {
                    "entrypoint": "app.py",
                    "requires": ["verification", "text-model"],
                    "resources_server": {"type": "resources_servers", "name": "resource_instance"},
                    "model_server": {"type": "responses_api_models", "name": model_instance},
                    "datasets": [
                        {
                            "name": "fixture",
                            "type": "example",
                            "jsonl_fpath": str(tmp_path / "data.jsonl"),
                        }
                    ],
                }
            }
        },
        model_instance: {
            "responses_api_models": {
                "base_model_r4": {
                    "entrypoint": "app.py",
                    "provides": ["text-model"],
                }
            }
        },
    }
    if profile == "custom-driver":
        payload["rollout_collection_driver"] = rollout_driver
    config_path = _write_yaml(tmp_path / "environments" / "r4_fixture" / "config.yaml", payload)
    _write_simple_agent_source(tmp_path, "base_agent_r4", profile)
    return config_path


def _parse_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    role: str,
    replacement_path: Path,
    profile: str = "external-loop",
    model_instance: str = "policy_model",
    overrides: dict | None = None,
):
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(tmp_path))
    initial = {
        "config_paths": [
            str(_base_config(tmp_path, profile=profile, model_instance=model_instance)),
            str(replacement_path),
        ],
        "manifest_path": str(_manifest(tmp_path, profile=profile, model_instance=model_instance)),
        "environment_component_swaps": {role: str(replacement_path)},
        **(overrides or {}),
    }
    return StaticValidationConfigParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(initial),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
        )
    )


def _parse_driver_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    declared: str,
    selected: str,
):
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(tmp_path))
    config_path = _base_config(tmp_path, profile="custom-driver", rollout_driver=declared)
    manifest_path = _manifest(tmp_path, profile="custom-driver", rollout_driver=declared)
    return StaticValidationConfigParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [str(config_path)],
                    "manifest_path": str(manifest_path),
                    "rollout_collection_driver": selected,
                    "environment_component_swaps": {"rollout_driver": selected},
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
        )
    )


def test_resource_swap_preserves_recipe_and_referenced_helper_service(tmp_path: Path, monkeypatch) -> None:
    replacement = _write_yaml(
        tmp_path / "resources_servers" / "replacement_resource_r4" / "configs" / "replacement_resource_r4.yaml",
        {
            "replacement_resource": {
                "resources_servers": {
                    "replacement_resource_r4": {
                        "entrypoint": "app.py",
                        "domain": "other",
                        "grading_mode": "strict",
                        "requires": ["judge-model"],
                        "provides": ["verification"],
                        "judge_model": {"type": "responses_api_models", "name": "replacement_judge"},
                    }
                }
            },
            "replacement_judge": {
                "responses_api_models": {"judge_model_r4": {"entrypoint": "app.py", "provides": ["judge-model"]}}
            },
            "bundled_agent": {
                "responses_api_agents": {
                    "unwanted_agent_r4": {
                        "entrypoint": "app.py",
                        "datasets": [{"name": "wrong", "jsonl_fpath": "missing.jsonl"}],
                    }
                }
            },
        },
    )

    resolved = _parse_swap(tmp_path, monkeypatch, role="resources_server", replacement_path=replacement)

    assert list(resolved.resource_instance.resources_servers) == ["replacement_resource_r4"]
    assert resolved.agent_instance.responses_api_agents.base_agent_r4.resources_server.name == "resource_instance"
    assert "bundled_agent" not in resolved
    assert resolved.replacement_judge.responses_api_models.judge_model_r4.provides == ["judge-model"]
    preflight = environment_execution.preflight_manifest_execution(resolved, check_launch_sources=False)
    assert preflight is not None
    assert preflight.manifest.resources_server == "replacement_resource_r4"


def test_model_swap_replaces_non_dummy_model_in_place(tmp_path: Path, monkeypatch) -> None:
    replacement = _write_yaml(
        tmp_path / "responses_api_models" / "replacement_model_r4" / "configs" / "replacement_model_r4.yaml",
        {
            "replacement_model": {
                "responses_api_models": {"replacement_model_r4": {"entrypoint": "app.py", "provides": ["text-model"]}}
            }
        },
    )

    resolved = _parse_swap(tmp_path, monkeypatch, role="model_server", replacement_path=replacement)

    assert list(resolved.policy_model.responses_api_models) == ["replacement_model_r4"]
    assert resolved.agent_instance.responses_api_agents.base_agent_r4.model_server.name == "policy_model"
    assert environment_execution.preflight_manifest_execution(resolved, check_launch_sources=False) is not None
    inspection = inspect_workload(resolved)
    assert any(component.implementation == "replacement_model_r4" for component in inspection.components)
    assert len(inspection.composition_hash) == 64


def test_published_component_swap_verifies_the_checked_in_base_composition(tmp_path: Path, monkeypatch) -> None:
    replacement = _write_yaml(
        tmp_path / "responses_api_models" / "replacement_model_r4" / "configs" / "replacement_model_r4.yaml",
        {
            "replacement_model": {
                "responses_api_models": {"replacement_model_r4": {"entrypoint": "app.py", "provides": ["text-model"]}}
            }
        },
    )
    resolved = _parse_swap(tmp_path, monkeypatch, role="model_server", replacement_path=replacement)
    config_path = tmp_path / "environments" / "r4_fixture" / "config.yaml"
    manifest_path = config_path.with_name("manifest.yaml")
    manifest = load_manifest(manifest_path)
    base = resolve_config_paths_static((config_path,))
    base_hash = compute_composition_hash(base, manifest)
    check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=manifest_path,
        config_path=config_path,
        manifest=manifest,
        composition_hash=base_hash,
    )

    assert compute_composition_hash(resolved, manifest) != base_hash
    assert environment_execution.preflight_manifest_execution(resolved, check_launch_sources=False) is not None

    changed_base = OmegaConf.load(config_path)
    changed_base.resource_instance.resources_servers.base_resource_r4.grading_mode = "lenient"
    OmegaConf.save(changed_base, config_path)
    with pytest.raises(ConfigError, match="locked composition_hash"):
        environment_execution.preflight_manifest_execution(resolved, check_launch_sources=False)


def test_capture_records_effective_resource_swap_provenance(tmp_path: Path, monkeypatch) -> None:
    replacement = _write_yaml(
        tmp_path / "resources_servers" / "replacement_resource_r4" / "configs" / "replacement_resource_r4.yaml",
        {
            "replacement_resource": {
                "resources_servers": {
                    "replacement_resource_r4": {
                        "entrypoint": "app.py",
                        "domain": "other",
                        "grading_mode": "lenient",
                        "provides": ["verification"],
                    }
                }
            }
        },
    )
    for group, implementation in (
        ("resources_servers", "replacement_resource_r4"),
        ("responses_api_agents", "base_agent_r4"),
        ("responses_api_models", "base_model_r4"),
    ):
        source = tmp_path / group / implementation / "app.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# fixture\n")
    resolved = _parse_swap(tmp_path, monkeypatch, role="resources_server", replacement_path=replacement)

    preflight = environment_execution.preflight_manifest_execution(resolved)
    assert preflight is not None
    expected_hash = inspect_workload(resolved, manifest=preflight.manifest).composition_hash
    captured = captured_environment_from_config(resolved)

    assert captured is not None
    assert captured.resources_server == "replacement_resource_r4"
    assert captured.grading_mode == "lenient"
    assert captured.composition_hash == expected_hash


def test_swap_metadata_authorizes_only_its_component_delta(tmp_path: Path, monkeypatch) -> None:
    replacement = _write_yaml(
        tmp_path / "resources_servers" / "replacement_resource_r4" / "configs" / "replacement_resource_r4.yaml",
        {
            "replacement_resource": {
                "resources_servers": {
                    "replacement_resource_r4": {
                        "entrypoint": "app.py",
                        "domain": "other",
                        "grading_mode": "strict",
                        "provides": ["verification"],
                    }
                }
            }
        },
    )
    resolved = _parse_swap(tmp_path, monkeypatch, role="resources_server", replacement_path=replacement)
    manifest = load_manifest(tmp_path / "environments" / "r4_fixture" / "manifest.yaml")

    assert (
        environment_execution.manifest_contract_deltas(manifest, resolve_composition_mirror(resolved), resolved) == ()
    )
    resolved.agent_instance.responses_api_agents.base_agent_r4.datasets[0].name = "changed"
    deltas = environment_execution.manifest_contract_deltas(manifest, resolve_composition_mirror(resolved), resolved)
    assert len(deltas) == 1
    assert deltas[0].startswith("datasets:")


def test_rollout_driver_swap_projects_effective_manifest_and_capture(tmp_path: Path, monkeypatch) -> None:
    declared = "environments.r4_fixture.base_driver:collect"
    selected = "environments.r4_fixture.replacement_driver:collect"
    driver_dir = tmp_path / "environments" / "r4_fixture"
    driver_dir.mkdir(parents=True, exist_ok=True)
    (driver_dir / "base_driver.py").write_text("async def collect(*_args):\n    pass\n")
    (driver_dir / "replacement_driver.py").write_text("async def collect(*_args):\n    pass\n")
    for group, implementation in (
        ("resources_servers", "base_resource_r4"),
        ("responses_api_models", "base_model_r4"),
    ):
        source = tmp_path / group / implementation / "app.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# fixture\n")

    resolved = _parse_driver_swap(tmp_path, monkeypatch, declared=declared, selected=selected)
    manifest = load_manifest(driver_dir / "manifest.yaml")
    mirror = resolve_composition_mirror(resolved)
    base = resolve_config_paths_static((driver_dir / "config.yaml",))
    base_hash = compute_composition_hash(base, manifest)
    check_or_record_version_lock(
        repo_root=tmp_path,
        manifest_path=driver_dir / "manifest.yaml",
        config_path=driver_dir / "config.yaml",
        manifest=manifest,
        composition_hash=base_hash,
    )

    assert OmegaConf.to_container(resolved.environment_component_swaps, resolve=True) == {
        "rollout_driver": {
            "declared": declared,
            "selected": selected,
            "integration_profile": "custom-driver",
        }
    }
    assert environment_execution.manifest_contract_deltas(manifest, mirror, resolved) == ()
    assert compute_composition_hash(resolved, manifest) != base_hash
    preflight = environment_execution.preflight_manifest_execution(resolved)
    assert preflight is not None
    assert preflight.manifest.rollout_driver == selected
    captured = captured_environment_from_config(resolved)
    assert captured is not None
    assert captured.rollout_driver == selected
    assert captured.composition_hash == inspect_workload(resolved, manifest=preflight.manifest).composition_hash


def test_rollout_driver_swap_authorizes_no_unrelated_manifest_delta(tmp_path: Path, monkeypatch) -> None:
    resolved = _parse_driver_swap(
        tmp_path,
        monkeypatch,
        declared="fixture.base_driver:collect",
        selected="fixture.replacement_driver:collect",
    )
    manifest = load_manifest(tmp_path / "environments" / "r4_fixture" / "manifest.yaml")
    resolved.agent_instance.responses_api_agents.base_agent_r4.datasets[0].name = "changed"

    deltas = environment_execution.manifest_contract_deltas(
        manifest,
        resolve_composition_mirror(resolved),
        resolved,
    )

    assert len(deltas) == 1
    assert deltas[0].startswith("datasets:")


def test_rollout_driver_swap_rejects_raw_resolved_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(tmp_path))
    config_path = _base_config(tmp_path, profile="custom-driver")
    manifest_path = _manifest(tmp_path, profile="custom-driver")
    raw = {
        "config_paths": [str(config_path)],
        "manifest_path": str(manifest_path),
        "rollout_collection_driver": "fixture.replacement_driver:collect",
        "environment_component_swaps": {
            "rollout_driver": {
                "declared": "fixture.driver:collect",
                "selected": "fixture.replacement_driver:collect",
                "integration_profile": "custom-driver",
            }
        },
    }

    with pytest.raises(ConfigError, match="must be an explicit 'module.path:function' selection"):
        StaticValidationConfigParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=OmegaConf.create(raw),
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
            )
        )


def test_rollout_driver_swap_requires_a_manifest_declaration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(tmp_path))
    config_path = _base_config(tmp_path)
    manifest_path = _manifest(tmp_path)
    raw = {
        "config_paths": [str(config_path)],
        "manifest_path": str(manifest_path),
        "rollout_collection_driver": "fixture.replacement_driver:collect",
        "environment_component_swaps": {"rollout_driver": "fixture.replacement_driver:collect"},
    }

    with pytest.raises(ConfigError, match="does not declare a rollout driver"):
        StaticValidationConfigParser().parse(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=OmegaConf.create(raw),
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
            )
        )


def test_sync_never_writes_a_temporary_component_swap(tmp_path: Path, monkeypatch) -> None:
    replacement = _write_yaml(
        tmp_path / "resources_servers" / "replacement_resource_r4" / "configs" / "replacement_resource_r4.yaml",
        {
            "replacement_resource": {
                "resources_servers": {
                    "replacement_resource_r4": {
                        "entrypoint": "app.py",
                        "domain": "other",
                        "grading_mode": "lenient",
                        "provides": ["verification"],
                    }
                }
            }
        },
    )
    resolved = _parse_swap(tmp_path, monkeypatch, role="resources_server", replacement_path=replacement)
    resolved.agent_instance.responses_api_agents.base_agent_r4.datasets[0].name = "changed"
    manifest_path = tmp_path / "environments" / "r4_fixture" / "manifest.yaml"
    original = manifest_path.read_text()
    monkeypatch.setattr(
        onboarding, "inspect_workload", lambda *_args, **_kwargs: SimpleNamespace(profile="external-loop")
    )

    with pytest.raises(ConfigError, match="Temporary component swaps cannot be written"):
        onboarding._perform_validation(
            resolved,
            onboarding.ValidateEnvironmentConfig(manifest_path=manifest_path, sync_manifest=True),
        )
    assert manifest_path.read_text() == original


@pytest.mark.parametrize("profile", ["external-loop", "custom-driver"])
def test_unpinned_agent_swap_preserves_datasets_and_rewires_refs(tmp_path: Path, monkeypatch, profile: str) -> None:
    replacement = _write_yaml(
        tmp_path / "responses_api_agents" / "replacement_agent_r4" / "configs" / "replacement_agent_r4.yaml",
        {
            "replacement_agent": {
                "responses_api_agents": {
                    "replacement_agent_r4": {
                        "entrypoint": "app.py",
                        "requires": ["verification", "text-model"],
                        "resources_server": {"type": "resources_servers", "name": "wrong_resource"},
                        "model_server": {"type": "responses_api_models", "name": "wrong_model"},
                        "datasets": [{"name": "wrong", "jsonl_fpath": "missing.jsonl"}],
                    }
                }
            },
            "wrong_resource": {
                "resources_servers": {"wrong_resource_r4": {"entrypoint": "app.py", "domain": "other"}}
            },
            "wrong_model": {"responses_api_models": {"wrong_model_r4": {"entrypoint": "app.py"}}},
        },
    )
    _write_simple_agent_source(tmp_path, "replacement_agent_r4", "external-loop")

    resolved = _parse_swap(
        tmp_path,
        monkeypatch,
        role="agent_server",
        replacement_path=replacement,
        profile=profile,
    )
    agent = resolved.agent_instance.responses_api_agents.replacement_agent_r4

    assert agent.resources_server.name == "resource_instance"
    assert agent.model_server.name == "policy_model"
    assert [dataset.name for dataset in agent.datasets] == ["fixture"]
    assert "wrong_resource" not in resolved
    assert "wrong_model" not in resolved
    assert resolve_composition_mirror(resolved).agent_server == "replacement_agent_r4"


def test_agent_swap_rewires_aliases_and_preserves_transitive_sidecars(tmp_path: Path, monkeypatch) -> None:
    replacement = _write_yaml(
        tmp_path / "responses_api_agents" / "replacement_agent_r4" / "configs" / "replacement_agent_r4.yaml",
        {
            "replacement_agent": {
                "responses_api_agents": {
                    "replacement_agent_r4": {
                        "entrypoint": "app.py",
                        "requires": ["verification", "text-model"],
                        "resources_server": {"type": "resources_servers", "name": "agent_resource"},
                        "model_server": {"type": "responses_api_models", "name": "policy_model"},
                        "user_model_server": {"type": "responses_api_models", "name": "policy_model"},
                        "critic_model_server": {"type": "responses_api_models", "name": "critic_model"},
                    }
                }
            },
            "critic_model": {
                "responses_api_models": {
                    "critic_model_r4": {
                        "entrypoint": "app.py",
                        "audit_server": {"type": "resources_servers", "name": "critic_audit"},
                    }
                }
            },
            "critic_audit": {
                "resources_servers": {
                    "critic_audit_r4": {
                        "entrypoint": "app.py",
                        "domain": "other",
                    }
                }
            },
            "agent_resource": {"resources_servers": {"unwanted_resource_r4": {"entrypoint": "app.py"}}},
            "policy_model": {"responses_api_models": {"unwanted_model_r4": {"entrypoint": "app.py"}}},
        },
    )
    _write_simple_agent_source(tmp_path, "replacement_agent_r4", "external-loop")

    resolved = _parse_swap(
        tmp_path,
        monkeypatch,
        role="agent_server",
        replacement_path=replacement,
        model_instance="primary_model",
    )
    agent = resolved.agent_instance.responses_api_agents.replacement_agent_r4

    assert agent.resources_server.name == "resource_instance"
    assert agent.model_server.name == "primary_model"
    assert agent.user_model_server.name == "primary_model"
    assert agent.critic_model_server.name == "critic_model"
    assert resolved.critic_model.responses_api_models.critic_model_r4.audit_server.name == "critic_audit"
    assert "critic_audit_r4" in resolved.critic_audit.resources_servers
    assert "agent_resource" not in resolved
    assert "policy_model" not in resolved
    assert environment_execution.preflight_manifest_execution(resolved, check_launch_sources=False) is not None


def test_component_swap_rejects_an_unresolved_sidecar_reference(tmp_path: Path, monkeypatch) -> None:
    replacement = _write_yaml(
        tmp_path / "responses_api_models" / "replacement_model_r4" / "configs" / "replacement_model_r4.yaml",
        {
            "replacement_model": {
                "responses_api_models": {
                    "replacement_model_r4": {
                        "entrypoint": "app.py",
                        "audit_server": {"type": "resources_servers", "name": "missing_audit"},
                    }
                }
            }
        },
    )

    with pytest.raises(ConfigError, match="missing dependency resources_servers/'missing_audit'"):
        _parse_swap(tmp_path, monkeypatch, role="model_server", replacement_path=replacement)


@pytest.mark.parametrize("profile", ["stock-loop", "measured-loop"])
def test_pinned_agent_swap_fails_loudly_with_profile(tmp_path: Path, monkeypatch, profile: str) -> None:
    replacement = _write_yaml(
        tmp_path / "responses_api_agents" / "replacement_agent_r4" / "configs" / "replacement_agent_r4.yaml",
        {
            "replacement_agent": {
                "responses_api_agents": {
                    "replacement_agent_r4": {
                        "entrypoint": "app.py",
                        "resources_server": {"type": "resources_servers", "name": "resource_instance"},
                        "model_server": {"type": "responses_api_models", "name": "policy_model"},
                    }
                }
            }
        },
    )

    with pytest.raises(ConfigError, match=f"Profile '{profile}' pins the agent server"):
        _parse_swap(
            tmp_path,
            monkeypatch,
            role="agent_server",
            replacement_path=replacement,
            profile=profile,
        )


def test_incompatible_model_swap_fails_manifest_preflight(tmp_path: Path, monkeypatch) -> None:
    replacement = _write_yaml(
        tmp_path / "responses_api_models" / "image_model_r4" / "configs" / "image_model_r4.yaml",
        {
            "image_model": {
                "responses_api_models": {"image_model_r4": {"entrypoint": "app.py", "provides": ["image-model"]}}
            }
        },
    )
    resolved = _parse_swap(
        tmp_path,
        monkeypatch,
        role="model_server",
        replacement_path=replacement,
        overrides={
            "agent_instance": {
                "responses_api_agents": {
                    "base_agent_r4": {"requires": ["verification", "audio-model"]},
                }
            }
        },
    )

    with pytest.raises(ConfigError, match="Incompatible component capabilities"):
        environment_execution.preflight_manifest_execution(resolved, check_launch_sources=False)


def test_missing_replacement_source_fails_manifest_preflight(tmp_path: Path, monkeypatch) -> None:
    replacement = _write_yaml(
        tmp_path / "responses_api_models" / "replacement_model_r4" / "configs" / "replacement_model_r4.yaml",
        {
            "replacement_model": {
                "responses_api_models": {
                    "replacement_model_r4": {"entrypoint": "missing.py", "provides": ["text-model"]}
                }
            }
        },
    )
    for group, implementation in (
        ("resources_servers", "base_resource_r4"),
        ("responses_api_agents", "base_agent_r4"),
    ):
        source_dir = tmp_path / group / implementation
        source_dir.mkdir(parents=True)
        (source_dir / "app.py").write_text("# fixture\n")
    resolved = _parse_swap(tmp_path, monkeypatch, role="model_server", replacement_path=replacement)

    with pytest.raises(ConfigError, match="replacement_model_r4.*missing.py"):
        environment_execution.preflight_manifest_execution(resolved)


def test_explicit_override_wins_over_transplanted_component(tmp_path: Path, monkeypatch) -> None:
    replacement = _write_yaml(
        tmp_path / "responses_api_models" / "replacement_model_r4" / "configs" / "replacement_model_r4.yaml",
        {
            "replacement_model": {
                "responses_api_models": {
                    "replacement_model_r4": {
                        "entrypoint": "app.py",
                        "provides": ["text-model"],
                        "temperature": 0.1,
                    }
                }
            }
        },
    )
    resolved = _parse_swap(
        tmp_path,
        monkeypatch,
        role="model_server",
        replacement_path=replacement,
        overrides={"policy_model": {"responses_api_models": {"replacement_model_r4": {"temperature": 0.7}}}},
    )

    assert resolved.policy_model.responses_api_models.replacement_model_r4.temperature == 0.7
