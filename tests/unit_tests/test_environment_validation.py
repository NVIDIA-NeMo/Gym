# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import nemo_gym.environment_validation as validation_module
import nemo_gym.global_config as global_config
from nemo_gym.config_types import ConfigError
from nemo_gym.environment_files import iter_runtime_path_values, resolve_runtime_local_references
from nemo_gym.environment_manifest import EnvironmentManifest
from nemo_gym.environment_validation import (
    compute_composition_hash,
    infer_integration_profile,
    inspect_components,
    inspect_sandbox_providers,
    inspect_workload,
    manifest_composition_deltas,
    resolve_composition_mirror,
    validate_capabilities,
    validate_datasets,
    validate_grading_mode_constraints,
    validate_rollout_driver_contract,
    validate_runtime_local_references,
)
from nemo_gym.global_config import GlobalConfigDictParserConfig, StaticValidationConfigParser


def _config(agent: str = "simple_agent", *, dataset: dict | None = None, driver: str | None = None) -> dict:
    config = {
        "resources": {
            "resources_servers": {
                "scorer": {
                    "entrypoint": "app.py",
                    "domain": "other",
                    "provides": ["verification", "tools"],
                }
            }
        },
        "agent": {
            "responses_api_agents": {
                agent: {
                    "entrypoint": "app.py",
                    "requires": ["verification"],
                    "provides": ["text-agent"],
                    "resources_server": {"type": "resources_servers", "name": "resources"},
                    "datasets": [dataset] if dataset else [],
                }
            }
        },
    }
    if driver:
        config["rollout_collection_driver"] = driver
    return config


@pytest.mark.parametrize(
    "config, expected",
    [
        (_config(), "stock-loop"),
        (_config("browsecomp_agent"), "measured-loop"),
        (_config("harbor_agent"), "external-loop"),
        (_config(driver="pkg.module:run"), "custom-driver"),
    ],
)
def test_infer_integration_profile(config, expected) -> None:
    assert infer_integration_profile(config) == expected


@pytest.mark.parametrize(
    ("agent", "forged_profile", "expected"),
    [
        ("simple_agent", "external-loop", "stock-loop"),
        ("browsecomp_agent", "external-loop", "measured-loop"),
        ("labbench2_vlm_agent", "external-loop", "measured-loop"),
        ("harbor_agent", "measured-loop", "external-loop"),
    ],
)
def test_infer_integration_profile_ignores_agent_self_label(agent, forged_profile, expected) -> None:
    config = _config(agent)
    config["agent"]["responses_api_agents"][agent]["integration_profile"] = forged_profile

    assert infer_integration_profile(config) == expected


def test_static_parser_never_probes_network_ports(monkeypatch) -> None:
    monkeypatch.setattr(
        global_config,
        "_find_open_port_using_range",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("port probe must not run")),
    )
    parsed = StaticValidationConfigParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(_config()),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
        )
    )

    ports = sorted(
        component["port"]
        for top_level in (parsed["resources"], parsed["agent"])
        for group in top_level.values()
        for component in group.values()
    )
    assert ports == [10001, 10002]


def test_component_inspection_reports_pinning_boundary_and_capabilities() -> None:
    config = _config()
    config["agent"]["responses_api_agents"]["simple_agent"].update({"host": "127.0.0.1", "port": 8000})
    components = inspect_components(config)
    agent = next(component for component in components if component.role == "agent_server")
    assert agent.pinned is True
    assert agent.boundary == "http://127.0.0.1:8000"
    assert agent.requires == ("verification",)
    assert agent.version.startswith("sha256:")
    assert len(agent.version.removeprefix("sha256:")) == 64


def test_missing_capability_fails_loudly() -> None:
    config = _config()
    config["agent"]["responses_api_agents"]["simple_agent"]["requires"] = ["image-model"]
    with pytest.raises(ConfigError, match="image-model.*Available capabilities"):
        validate_capabilities(inspect_components(config))


def test_component_cannot_satisfy_its_own_requirement() -> None:
    config = _config()
    resource = config["resources"]["resources_servers"]["scorer"]
    resource["requires"] = ["tools"]
    with pytest.raises(ConfigError, match="resources requires 'tools'"):
        validate_capabilities(inspect_components(config))


def test_component_requirement_is_scoped_to_its_wired_provider() -> None:
    config = _config()
    config["resources"]["resources_servers"]["scorer"]["provides"] = []
    config["unrelated"] = {
        "resources_servers": {
            "other_scorer": {"entrypoint": "app.py", "provides": ["verification"]},
        }
    }

    with pytest.raises(ConfigError, match=r"agent requires 'verification' through wired instances resources"):
        inspect_workload(config)


def test_inspection_matches_runtime_first_component_dispatch() -> None:
    config = {
        "mixed": {
            "resources_servers": {
                "scorer": {"entrypoint": "app.py", "provides": ["verification"]},
            },
            "responses_api_agents": {
                "phantom_agent": {"entrypoint": "never_started.py", "requires": ["missing"]},
            },
        }
    }

    components = inspect_components(config)

    assert [(component.role, component.implementation) for component in components] == [
        ("resources_server", "scorer"),
        ("rollout_driver", "nemo_gym.rollout_collection"),
    ]


def test_manifest_capabilities_are_checked_against_concrete_components() -> None:
    components = inspect_components(_config())
    manifest = EnvironmentManifest.model_construct(requires=["verification"], provides=["tools"])
    decisions = validate_capabilities(components, manifest)
    assert "environment requires verification: satisfied by resources" in decisions
    assert "environment provides tools: backed by resources" in decisions

    unsupported_claim = EnvironmentManifest.model_construct(requires=[], provides=["sandbox:container"])
    with pytest.raises(ConfigError, match="provided capability 'sandbox:container'.*no component backing"):
        validate_capabilities(components, unsupported_claim)


def test_manifest_runtime_capability_requires_a_concrete_provider() -> None:
    manifest = EnvironmentManifest.model_construct(requires=["sandbox:container"], provides=[])

    with pytest.raises(ConfigError, match="environment requires 'sandbox:container'.*Available capabilities"):
        validate_capabilities(inspect_components(_config()), manifest)


def test_declared_model_capabilities_extend_the_text_model_baseline() -> None:
    config = _config()
    config["model"] = {
        "responses_api_models": {
            "policy": {"entrypoint": "app.py", "provides": ["image-model"]},
        }
    }
    model = next(component for component in inspect_components(config) if component.role == "model_server")

    assert model.provides == ("text-model", "image-model")


def test_inference_provider_gemini_inherits_text_model_interface() -> None:
    config_path = Path(__file__).parents[2] / "responses_api_models" / "inference_provider" / "configs" / "gemini.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
    assert isinstance(config, dict)

    model = next(component for component in inspect_components(config) if component.role == "model_server")

    assert model.implementation == "inference_provider"
    assert model.provides == ("text-model",)


def test_manifest_runtime_facts_become_capability_requirements() -> None:
    config = _config()
    resource = config["resources"]["resources_servers"]["scorer"]
    resource["provides"].extend(["session:episode", "state:per_session"])
    config["agent"]["responses_api_agents"]["simple_agent"]["sandbox_provider"] = {"docker": {}}
    config["model"] = {
        "responses_api_models": {
            "policy": {"entrypoint": "app.py", "provides": ["image-model"]},
        }
    }
    manifest = EnvironmentManifest.model_construct(
        modality="image",
        resources_server="scorer",
        model_server="model",
        session_model="episode",
        state="per_session",
        sandbox="docker",
        requires=[],
        provides=[],
    )

    sandbox_components, sandbox_decisions = inspect_sandbox_providers(
        config,
        manifest,
        include_content_versions=False,
    )
    decisions = validate_capabilities(
        (*inspect_components(config), *sandbox_components),
        manifest,
        config=config,
    )

    assert "environment requires image-model: satisfied by model" in decisions
    assert "environment requires session:episode: satisfied by resources" in decisions
    assert "environment requires state:per_session: satisfied by resources" in decisions
    assert "environment requires sandbox:docker: satisfied by agent.sandbox_provider" in decisions
    assert sandbox_decisions == ("sandbox provider 'docker' selected by agent_server:agent/simple_agent: registered",)

    resource["provides"].remove("state:per_session")
    with pytest.raises(ConfigError, match="environment requires 'state:per_session'"):
        validate_capabilities(inspect_components(config), manifest)


def test_manifest_modality_cannot_be_satisfied_by_an_unrelated_model() -> None:
    config = _config()
    config["policy"] = {
        "responses_api_models": {
            "text_policy": {"entrypoint": "app.py", "provides": ["text-model"]},
        }
    }
    config["judge"] = {
        "responses_api_models": {
            "image_judge": {"entrypoint": "app.py", "provides": ["image-model"]},
        }
    }
    manifest = EnvironmentManifest.model_construct(
        modality="image",
        model_server="policy",
        requires=[],
        provides=[],
    )

    with pytest.raises(
        ConfigError,
        match=r"requires 'image-model' for manifest modality.*manifest-selected model_server 'policy' provides: text-model",
    ):
        validate_capabilities(inspect_components(config), manifest)


def test_manifest_selected_text_model_uses_the_role_interface_baseline() -> None:
    config = _config()
    config["model"] = {
        "responses_api_models": {
            "policy": {"entrypoint": "app.py"},
        }
    }
    manifest = EnvironmentManifest.model_construct(
        modality="text",
        model_server="model",
        requires=[],
        provides=[],
    )

    decisions = validate_capabilities(inspect_components(config), manifest)

    assert "environment requires text-model: satisfied by model" in decisions


def test_manifest_selected_resource_and_agent_must_declare_their_interfaces() -> None:
    config = _config()
    resource = config["resources"]["resources_servers"]["scorer"]
    agent = config["agent"]["responses_api_agents"]["simple_agent"]
    resource.pop("provides")
    agent.pop("requires")
    config["model"] = {
        "responses_api_models": {
            "policy": {"entrypoint": "app.py", "provides": ["text-model"]},
        }
    }
    agent["model_server"] = {"type": "responses_api_models", "name": "model"}
    manifest = EnvironmentManifest.model_construct(
        modality="text",
        resources_server="scorer",
        agent_server="simple_agent",
        model_server="model",
        requires=[],
        provides=[],
    )

    with pytest.raises(
        ConfigError,
        match=r"must declare provides: \[verification\].*must declare required protocol\(s\): text-model, verification",
    ):
        validate_capabilities(inspect_components(config), manifest, config=config)


def test_manifest_selected_model_must_resolve_to_a_component() -> None:
    manifest = EnvironmentManifest.model_construct(
        modality="text",
        model_server="policy_model",
        requires=[],
        provides=[],
    )

    with pytest.raises(
        ConfigError,
        match=r"requires 'text-model' for manifest modality.*model_server 'policy_model' does not resolve",
    ):
        validate_capabilities(inspect_components(_config()), manifest)


def test_manifest_modality_does_not_require_a_gym_model_for_external_topology() -> None:
    manifest = EnvironmentManifest.model_construct(
        integration_profile="external-loop",
        modality="text",
        model_server=None,
        requires=[],
        provides=[],
    )

    assert validate_capabilities(inspect_components(_config()), manifest) == (
        "agent requires verification: satisfied by resources",
    )


def test_legacy_workload_without_manifest_keeps_undeclared_model_compatibility() -> None:
    config = _config()
    config["model"] = {
        "responses_api_models": {
            "policy": {"entrypoint": "app.py"},
        }
    }

    assert validate_capabilities(inspect_components(config)) == (
        "agent requires verification: satisfied by resources",
    )


@pytest.mark.parametrize(
    ("field_name", "field_value", "capability"),
    [
        ("session_model", "episode", "session:episode"),
        ("state", "per_session", "state:per_session"),
    ],
)
def test_manifest_runtime_requirement_cannot_be_satisfied_by_an_unrelated_resource(
    field_name: str,
    field_value: str,
    capability: str,
) -> None:
    config = _config()
    # The unrelated instance deliberately shadows the selected implementation's
    # name; resources_server mirrors identify implementations, not instances.
    config["scorer"] = {
        "resources_servers": {
            "other": {"entrypoint": "app.py", "provides": [capability]},
        }
    }
    manifest = EnvironmentManifest.model_construct(
        resources_server="scorer",
        requires=[],
        provides=[],
        **{field_name: field_value},
    )

    with pytest.raises(
        ConfigError,
        match=rf"requires '{capability}' for manifest {field_name}.*manifest-selected resources_server 'scorer' provides",
    ):
        validate_capabilities(inspect_components(config), manifest)


def test_resources_capability_cannot_stand_in_for_selected_sandbox_provider() -> None:
    config = _config()
    config["resources"]["resources_servers"]["scorer"]["provides"].append("sandbox:docker")
    manifest = EnvironmentManifest.model_construct(sandbox="docker", requires=[], provides=[])

    with pytest.raises(ConfigError, match="does not select a sandbox_provider"):
        inspect_sandbox_providers(config, manifest, include_content_versions=False)


def test_manifest_inspection_reports_and_hashes_selected_sandbox_provider() -> None:
    config = _config()
    config["sandbox"] = {"docker": {"create": {"use_init": True}}}
    config["agent"]["responses_api_agents"]["simple_agent"]["sandbox_provider"] = "sandbox"
    manifest = EnvironmentManifest.model_construct(sandbox="docker", requires=[], provides=[])

    first = inspect_workload(config, manifest=manifest)
    sandbox = next(component for component in first.components if component.role == "sandbox_provider")

    assert sandbox.instance == "sandbox"
    assert sandbox.implementation == "docker"
    assert sandbox.provides == ("sandbox:docker",)
    assert sandbox.entrypoint is not None and sandbox.entrypoint.endswith("/docker/provider.py")
    assert sandbox.version is not None and sandbox.version.startswith("sha256:")

    config["sandbox"]["docker"]["create"]["use_init"] = False
    second = inspect_workload(config, manifest=manifest)

    assert second.composition_hash != first.composition_hash
    assert next(component for component in second.components if component.role == "sandbox_provider").version != (
        sandbox.version
    )


def test_sandbox_provider_helper_package_is_hash_locked(tmp_path: Path, monkeypatch) -> None:
    package = tmp_path / "docker"
    package.mkdir()
    (package / "provider.py").write_text("VALUE = 1\n")
    helper = package / "helper.py"
    helper.write_text("HELPER = 1\n")
    monkeypatch.setattr(validation_module, "_sandbox_provider_package_directory", lambda _selection: package)
    config = _config()
    config["sandbox"] = {"docker": {"create": {"use_init": True}}}
    config["agent"]["responses_api_agents"]["simple_agent"]["sandbox_provider"] = "sandbox"
    manifest = EnvironmentManifest.model_construct(sandbox="docker", requires=[], provides=[])

    baseline = inspect_workload(config, manifest=manifest).composition_hash
    helper.write_text("HELPER = 2\n")

    assert inspect_workload(config, manifest=manifest).composition_hash != baseline


def test_no_manifest_composition_hash_includes_selected_sandbox_provider() -> None:
    config = _config()
    config["sandbox"] = {"docker": {"create": {"use_init": True}}}
    config["agent"]["responses_api_agents"]["simple_agent"]["sandbox_provider"] = "sandbox"

    baseline = compute_composition_hash(config)
    config["sandbox"]["docker"]["create"]["use_init"] = False

    assert compute_composition_hash(config) != baseline


def test_sandbox_provider_shared_first_party_helper_is_hash_locked(tmp_path: Path, monkeypatch) -> None:
    package_root = tmp_path / "nemo_gym"
    provider_package = package_root / "sandbox" / "providers" / "docker"
    provider_package.mkdir(parents=True)
    for package in (package_root, package_root / "sandbox", package_root / "sandbox" / "providers"):
        (package / "__init__.py").write_text("")
    (provider_package / "provider.py").write_text("from nemo_gym.sandbox.providers.utils import coerce_config\n")
    helper = package_root / "sandbox" / "providers" / "utils.py"
    helper.write_text("def coerce_config(value): return value\n")
    monkeypatch.setattr(validation_module, "_sandbox_provider_package_directory", lambda _selection: provider_package)
    monkeypatch.setattr(validation_module, "_first_party_package_root", lambda: package_root)
    config = _config()
    config["sandbox"] = {"docker": {"create": {"use_init": True}}}
    config["agent"]["responses_api_agents"]["simple_agent"]["sandbox_provider"] = "sandbox"
    manifest = EnvironmentManifest.model_construct(sandbox="docker", requires=[], provides=[])

    baseline = compute_composition_hash(config, manifest)
    helper.write_text("def coerce_config(value): return None\n")

    assert compute_composition_hash(config, manifest) != baseline


def test_third_party_sandbox_provider_package_is_hash_locked(tmp_path: Path, monkeypatch) -> None:
    provider_package = tmp_path / "third_party_provider"
    provider_package.mkdir()
    (provider_package / "__init__.py").write_text("")
    provider_source = provider_package / "provider.py"
    provider_source.write_text("from .helper import VALUE\n")
    helper = provider_package / "helper.py"
    helper.write_text("VALUE = 1\n")

    class ExternalProvider:
        pass

    ExternalProvider.__module__ = "third_party_provider.provider"
    monkeypatch.setattr(validation_module, "get_provider_class", lambda _name: ExternalProvider)
    monkeypatch.setattr(validation_module, "_sandbox_provider_source", lambda _provider: provider_source)
    config = _config()
    config["sandbox"] = {"external": {}}
    config["agent"]["responses_api_agents"]["simple_agent"]["sandbox_provider"] = "sandbox"
    manifest = EnvironmentManifest.model_construct(sandbox="external", requires=[], provides=[])

    baseline = compute_composition_hash(config, manifest)
    helper.write_text("VALUE = 2\n")

    assert compute_composition_hash(config, manifest) != baseline


def test_third_party_sandbox_provider_distribution_is_hash_locked(tmp_path: Path, monkeypatch) -> None:
    provider_source = tmp_path / "external_provider.py"
    provider_source.write_text("from external_helper import VALUE\n")
    helper = tmp_path / "external_helper.py"
    helper.write_text("VALUE = 1\n")

    class ExternalProvider:
        pass

    class FakeDistribution:
        files = (Path("external_provider.py"), Path("external_helper.py"))

        @staticmethod
        def locate_file(member):
            return tmp_path / str(member)

    ExternalProvider.__module__ = "external_provider"
    monkeypatch.setattr(validation_module, "get_provider_class", lambda _name: ExternalProvider)
    monkeypatch.setattr(validation_module, "_sandbox_provider_source", lambda _provider: provider_source)
    monkeypatch.setattr(
        validation_module,
        "_installed_packages_distributions",
        lambda: {"external_provider": ["external-provider"]},
    )
    monkeypatch.setattr(
        validation_module.importlib_metadata,
        "distribution",
        lambda _name: FakeDistribution(),
    )
    config = _config()
    config["sandbox"] = {"external": {}}
    config["agent"]["responses_api_agents"]["simple_agent"]["sandbox_provider"] = "sandbox"
    manifest = EnvironmentManifest.model_construct(sandbox="external", requires=[], provides=[])

    baseline = compute_composition_hash(config, manifest)
    helper.write_text("VALUE = 2\n")

    assert compute_composition_hash(config, manifest) != baseline


def test_manifest_sandbox_provider_requires_bindable_source(monkeypatch) -> None:
    class UnboundProvider:
        pass

    UnboundProvider.__module__ = "unbound_provider"
    monkeypatch.setattr(validation_module, "get_provider_class", lambda _name: UnboundProvider)
    monkeypatch.setattr(validation_module, "_sandbox_provider_source", lambda _provider: None)
    config = _config()
    config["sandbox"] = {"unbound": {}}
    config["agent"]["responses_api_agents"]["simple_agent"]["sandbox_provider"] = "sandbox"
    manifest = EnvironmentManifest.model_construct(sandbox="unbound", requires=[], provides=[])

    with pytest.raises(ConfigError, match="no readable package, distribution, or source file"):
        compute_composition_hash(config, manifest)


def test_static_grading_mode_validation_uses_entrypoint_source(tmp_path, monkeypatch) -> None:
    source = tmp_path / "resources_servers" / "scorer" / "app.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from typing import Literal\n"
        "raise RuntimeError('validation must not import this module')\n"
        "class Request:\n"
        "    grading_mode: Literal['lenient', 'strict'] = 'strict'\n"
    )
    monkeypatch.setattr(
        validation_module,
        "_resolve_under_cwd_or_install",
        lambda path: tmp_path / path,
    )
    config = _config()
    config["resources"]["resources_servers"]["scorer"]["grading_mode"] = "strict"

    decisions, warnings = validate_grading_mode_constraints(config)

    assert decisions == ("scorer resolved config grading_mode='strict': supported",)
    assert warnings == ()
    assert decisions[0] in inspect_workload(config).compatibility_decisions

    config["resources"]["resources_servers"]["scorer"]["grading_mode"] = "permissive"
    with pytest.raises(ConfigError, match=r"scorer.*unsupported.*permissive.*lenient.*strict"):
        validate_grading_mode_constraints(config)


def test_unprovable_grading_mode_constraint_is_reported_as_limitation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        validation_module,
        "_resolve_under_cwd_or_install",
        lambda path: tmp_path / path,
    )
    config = _config()
    config["resources"]["resources_servers"]["scorer"]["grading_mode"] = "dynamic"

    decisions, warnings = validate_grading_mode_constraints(config)

    assert decisions == ()
    assert len(warnings) == 1
    assert "no finite constraint" in warnings[0]


def test_environment_dataset_requires_materialized_responses_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"question": "unmaterialized"}) + "\n")
    config = _config(dataset={"name": "example", "type": "example", "jsonl_fpath": str(path)})
    with pytest.raises(ConfigError, match="responses_create_params.input"):
        validate_datasets(config, strict_missing=True)

    path.write_text(json.dumps({"responses_create_params": {"input": "hello", "not_a_real_field": 1}}) + "\n")
    with pytest.raises(ConfigError, match="invalid Responses API parameters.*not_a_real_field"):
        validate_datasets(config, strict_missing=True)


def test_benchmark_dataset_is_materialized_through_prompt_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "benchmark.jsonl"
    path.write_text(json.dumps({"question": "2+2", "expected_answer": "4"}) + "\n")
    prompt = tmp_path / "prompt.yaml"
    prompt.write_text('user: "Solve {question}"\n')
    config = _config(
        dataset={
            "name": "math",
            "type": "benchmark",
            "jsonl_fpath": str(path),
            "prompt_config": str(prompt),
        }
    )
    reports = validate_datasets(config, strict_missing=True)
    assert reports[0].rows == 1
    assert "domain JSONL" in reports[0].detail
    assert reports[0].materialized_sample == {
        "responses_create_params": {"input": [{"content": "Solve 2+2", "role": "user"}]}
    }


def test_manifest_workload_requires_resolvable_benchmark_prepare_script(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    for component in (
        tmp_path / "resources_servers" / "scorer",
        tmp_path / "responses_api_agents" / "simple_agent",
        tmp_path / "responses_api_models" / "openai_model",
    ):
        component.mkdir(parents=True)
        (component / "app.py").write_text("\n")
    path = tmp_path / "benchmark.jsonl"
    path.write_text(json.dumps({"responses_create_params": {"input": "hello"}}) + "\n")
    config = _config(
        dataset={
            "name": "math",
            "type": "benchmark",
            "jsonl_fpath": str(path),
            "prepare_script": "benchmarks/math/missing.py",
        }
    )
    config["policy_model"] = {
        "responses_api_models": {
            "openai_model": {"entrypoint": "app.py", "provides": ["text-model"]},
        }
    }
    config["agent"]["responses_api_agents"]["simple_agent"].update(
        {
            "requires": ["verification", "text-model"],
            "model_server": {"type": "responses_api_models", "name": "policy_model"},
        }
    )
    manifest = EnvironmentManifest.model_validate(
        {
            "name": "example",
            "version": "1.0.0",
            "kind": "environment",
            "integration_profile": "stock-loop",
            "domain": "other",
            "description": "Prepare source validation fixture.",
            "modality": "text",
            "licensing": "Apache-2.0",
            "authors": ["contributor"],
            "reward": {"range": [0, 1], "higher_is_better": True},
            "resources_server": "scorer",
            "agent_server": "simple_agent",
            "model_server": "policy_model",
            "datasets": [
                {
                    "name": "math",
                    "type": "benchmark",
                    "jsonl_fpath": str(path),
                    "prepare_script": "benchmarks/math/missing.py",
                }
            ],
        }
    )
    with pytest.raises(ConfigError, match="prepare_script.*does not resolve"):
        inspect_workload(config, strict_missing_datasets=True, manifest=manifest)


def test_materialized_dataset_preview_is_bounded_and_redacts_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "responses_create_params": {
                    "input": "x" * 600,
                    "metadata": {"api_key": "do-not-print"},
                }
            }
        )
        + "\n"
    )
    report = validate_datasets(
        _config(dataset={"name": "example", "type": "example", "jsonl_fpath": str(path)}),
        strict_missing=True,
    )[0]
    assert report.materialized_sample is not None
    params = report.materialized_sample["responses_create_params"]
    assert params["metadata"]["api_key"] == "<redacted>"
    assert "do-not-print" not in json.dumps(report.to_dict())
    assert params["input"].endswith("… <100 chars omitted>")


def test_dataset_validation_streams_large_jsonl_with_bounded_diagnostics(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "large.jsonl"
    valid = json.dumps({"responses_create_params": {"input": "hello"}})
    path.write_text("\n".join([valid] * 2_000) + "\n")

    report = validate_datasets(
        _config(dataset={"name": "large", "type": "example", "jsonl_fpath": str(path)}),
        strict_missing=True,
    )[0]

    assert report.rows == 2_000
    assert report.materialized_sample == {"responses_create_params": {"input": "hello"}}


def test_missing_legacy_dataset_is_reported_but_manifest_mode_rejects(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = _config(dataset={"name": "example", "type": "example", "jsonl_fpath": "missing.jsonl"})
    assert validate_datasets(config)[0].status == "missing"
    with pytest.raises(ConfigError, match="does not exist"):
        validate_datasets(config, strict_missing=True)


def test_manifest_dataset_must_be_inside_a_versioned_search_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-external-data.jsonl"
    outside.write_text(json.dumps({"responses_create_params": {"input": "hello"}}) + "\n")
    config = _config(dataset={"name": "external", "type": "example", "jsonl_fpath": str(outside)})

    with pytest.raises(ConfigError, match="outside every component search root"):
        validate_datasets(config, strict_missing=True)


def test_inspect_workload_returns_stable_hash_and_json_ready_payload(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"responses_create_params": {"input": [{"role": "user", "content": "hello"}]}}) + "\n")
    config = _config(dataset={"name": "example", "type": "example", "jsonl_fpath": str(path)})
    first = inspect_workload(config, strict_missing_datasets=True)
    config["agent"]["responses_api_agents"]["simple_agent"]["port"] = 19000
    config["agent"]["responses_api_agents"]["simple_agent"]["api_key"] = "different-secret"
    second = inspect_workload(config, strict_missing_datasets=True)
    assert first.composition_hash == second.composition_hash
    assert first.to_dict()["datasets"][0]["status"] == "valid"
    assert first.to_dict()["responsibilities"] == {
        "task_preparation": [
            {
                "role": "dataset",
                "instance": "example",
                "implementation": str(path),
            }
        ],
        "model_interaction": [
            {
                "role": "agent_server",
                "instance": "agent",
                "implementation": "simple_agent",
            }
        ],
        "tools_and_state": [
            {
                "role": "resources_server",
                "instance": "resources",
                "implementation": "scorer",
            }
        ],
        "verification": [
            {
                "role": "resources_server",
                "instance": "resources",
                "implementation": "scorer",
            }
        ],
        "rollout_coordination": [
            {
                "role": "rollout_driver",
                "instance": "stock",
                "implementation": "nemo_gym.rollout_collection",
            }
        ],
    }
    assert first.fixed_constraints == ("agent_server:simple_agent is pinned by stock-loop",)
    assert first.compatibility_decisions == ("agent requires verification: satisfied by resources",)

    config["resources"]["resources_servers"]["scorer"]["grading_mode"] = "lenient"
    assert inspect_workload(config).composition_hash != first.composition_hash

    config["resources"]["resources_servers"]["scorer"].pop("grading_mode")
    path.write_text(json.dumps({"responses_create_params": {"input": "changed task"}}) + "\n")
    assert inspect_workload(config).composition_hash != first.composition_hash


def test_external_loop_owns_its_rollout_coordination() -> None:
    inspection = inspect_workload(_config("harbor_agent"))

    assert inspection.responsibilities.rollout_coordination == (
        validation_module.ResponsibilityOwner(
            role="agent_server",
            instance="agent",
            implementation="harbor_agent",
        ),
    )


def test_stock_rollout_driver_source_is_hash_locked(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "rollout_collection.py"
    source.write_text("VALUE = 1\n")
    monkeypatch.setattr(validation_module, "_stock_rollout_driver_source", lambda: source)

    baseline = inspect_workload(_config()).composition_hash
    source.write_text("VALUE = 2\n")

    assert inspect_workload(_config()).composition_hash != baseline


def test_stock_rollout_driver_transitive_first_party_helper_is_hash_locked(tmp_path: Path, monkeypatch) -> None:
    package_root = tmp_path / "nemo_gym"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("")
    source = package_root / "rollout_collection.py"
    source.write_text("from nemo_gym.helper import VALUE\n")
    (package_root / "helper.py").write_text("from nemo_gym.deep_helper import VALUE\n")
    deep_helper = package_root / "deep_helper.py"
    deep_helper.write_text("VALUE = 1\n")
    monkeypatch.setattr(validation_module, "_stock_rollout_driver_source", lambda: source)
    monkeypatch.setattr(validation_module, "_first_party_package_root", lambda: package_root)

    baseline = compute_composition_hash(_config())
    deep_helper.write_text("VALUE = 2\n")

    assert compute_composition_hash(_config()) != baseline


def test_composition_hash_does_not_parse_dataset_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text("not json\n")
    config = _config(dataset={"name": "fixture", "type": "example", "jsonl_fpath": str(dataset)})

    assert len(compute_composition_hash(config)) == 64


def test_resources_server_without_tools_or_state_is_not_reported_as_owner() -> None:
    config = _config()
    config["resources"]["resources_servers"]["scorer"]["provides"] = ["verification"]

    inspection = inspect_workload(config)

    assert inspection.responsibilities.tools_and_state == ()
    assert inspection.responsibilities.verification == (
        validation_module.ResponsibilityOwner(
            role="resources_server",
            instance="resources",
            implementation="scorer",
        ),
    )


def test_custom_driver_owns_rollout_coordination() -> None:
    inspection = inspect_workload(_config(driver="pkg.module:run"))

    assert inspection.responsibilities.rollout_coordination == (
        validation_module.ResponsibilityOwner(
            role="rollout_driver",
            instance="custom",
            implementation="pkg.module:run",
        ),
    )


def test_custom_driver_config_is_reported_and_hash_locked() -> None:
    config = _config(driver="pkg.module:run")
    config["multistage"] = {"enabled": True, "stages": 5, "judge_api_key": "first-secret"}

    baseline = inspect_workload(config)

    assert baseline.overrides["rollout_driver_config"] == {
        "multistage": {"enabled": True, "stages": 5},
    }
    changed = deepcopy(config)
    changed["multistage"]["stages"] = 6
    assert inspect_workload(changed).composition_hash != baseline.composition_hash
    credential_only = deepcopy(config)
    credential_only["multistage"]["judge_api_key"] = "second-secret"
    assert inspect_workload(credential_only).composition_hash == baseline.composition_hash


def test_custom_driver_package_helpers_are_hash_locked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    driver_dir = tmp_path / "benchmarks" / "driver_case"
    driver_dir.mkdir(parents=True)
    (driver_dir / "rollout_driver.py").write_text("from .helper import VALUE\n\nasync def run(*_args): pass\n")
    helper = driver_dir / "helper.py"
    helper.write_text("VALUE = 1\n")
    (driver_dir / "config.yaml").write_text("comment_only: first\n")
    config = _config(driver="benchmarks.driver_case.rollout_driver:run")

    baseline = inspect_workload(config).composition_hash
    helper.write_text("VALUE = 2\n")
    changed_helper = inspect_workload(config).composition_hash
    (driver_dir / "config.yaml").write_text("comment_only: second\n")
    (driver_dir / "README.md").write_text("Driver documentation\n")

    assert changed_helper != baseline
    assert inspect_workload(config).composition_hash == changed_helper


def test_custom_driver_stock_collector_dependency_is_hash_locked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    driver_dir = tmp_path / "environments" / "driver_case"
    driver_dir.mkdir(parents=True)
    (driver_dir / "rollout_driver.py").write_text(
        "from nemo_gym.rollout_collection import RolloutCollectionHelper\n\n"
        "async def run(*_args): return RolloutCollectionHelper\n"
    )
    package_root = tmp_path / "nemo_gym"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("")
    stock_collector = package_root / "rollout_collection.py"
    stock_collector.write_text("class RolloutCollectionHelper: pass\n")
    monkeypatch.setattr(validation_module, "_first_party_package_root", lambda: package_root)
    config = _config(driver="environments.driver_case.rollout_driver:run")

    baseline = compute_composition_hash(config)
    stock_collector.write_text("class RolloutCollectionHelper:\n    changed = True\n")

    assert compute_composition_hash(config) != baseline


def test_dataset_prepare_package_helpers_are_hash_locked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    prepare_dir = tmp_path / "benchmarks" / "prepare_case"
    prepare_dir.mkdir(parents=True)
    (prepare_dir / "prepare.py").write_text("from benchmarks.prepare_case.helper import VALUE\n")
    helper = prepare_dir / "helper.py"
    helper.write_text("VALUE = 1\n")
    config = _config(
        dataset={
            "name": "prepare_case",
            "type": "benchmark",
            "jsonl_fpath": "benchmarks/prepare_case/data/missing.jsonl",
            "prepare_script": "benchmarks/prepare_case/prepare.py",
        }
    )

    baseline = inspect_workload(config).composition_hash
    helper.write_text("VALUE = 2\n")

    assert inspect_workload(config).composition_hash != baseline


def test_dataset_prepare_dynamic_cross_registry_dependency_is_hash_locked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    prepare_dir = tmp_path / "benchmarks" / "prepare_case"
    prepare_dir.mkdir(parents=True)
    (prepare_dir / "prepare.py").write_text('ROOT / "resources_servers" / "shared_prepare" / "helper.py"\n')
    dependency_dir = tmp_path / "resources_servers" / "shared_prepare"
    dependency_dir.mkdir(parents=True)
    helper = dependency_dir / "helper.py"
    helper.write_text("VALUE = 1\n")
    config = _config(
        dataset={
            "name": "prepare_case",
            "type": "benchmark",
            "jsonl_fpath": "benchmarks/prepare_case/data/missing.jsonl",
            "prepare_script": "benchmarks/prepare_case/prepare.py",
        }
    )

    baseline = inspect_workload(config).composition_hash
    helper.write_text("VALUE = 2\n")

    assert inspect_workload(config).composition_hash != baseline


@pytest.mark.parametrize("driver", ["not-a-driver", "module-name:run", "module.path:not-a-function", "pkg.mod:a:b"])
def test_custom_driver_rejects_invalid_callable_syntax(driver: str) -> None:
    with pytest.raises(ConfigError, match="module.path:function"):
        inspect_workload(_config(driver=driver))


def test_custom_driver_rejects_missing_local_module(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigError, match="does not resolve to local source"):
        inspect_workload(_config(driver="environments.missing.rollout_driver:run"))


def test_manifest_custom_driver_requires_local_version_bound_source() -> None:
    with pytest.raises(ConfigError, match="local, version-bound source"):
        validate_rollout_driver_contract(
            _config(driver="third_party.driver:run"),
            require_local_source=True,
        )


def test_custom_driver_validates_local_callable_symbol(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    driver_dir = tmp_path / "environments" / "driver_case"
    driver_dir.mkdir(parents=True)
    (driver_dir / "rollout_driver.py").write_text("async def implementation(*_args): pass\n\nrun = implementation\n")

    decisions, warnings = validate_rollout_driver_contract(
        _config(driver="environments.driver_case.rollout_driver:run"),
        require_local_source=True,
    )

    assert decisions == ("rollout driver environments.driver_case.rollout_driver: local source is version-bound",)
    assert warnings == ()


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("VALUE = 1\n", "does not define symbol 'run'"),
        ("run = 1\n", "symbol 'run'.*is not statically callable"),
    ],
)
def test_custom_driver_rejects_missing_or_noncallable_local_symbol(
    tmp_path: Path,
    monkeypatch,
    source: str,
    message: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    driver_dir = tmp_path / "environments" / "driver_case"
    driver_dir.mkdir(parents=True)
    (driver_dir / "rollout_driver.py").write_text(source)

    with pytest.raises(ConfigError, match=message):
        validate_rollout_driver_contract(
            _config(driver="environments.driver_case.rollout_driver:run"),
            require_local_source=True,
        )


def test_custom_driver_rejects_imports_from_excluded_tests(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    driver_dir = tmp_path / "environments" / "driver_case"
    tests_dir = driver_dir / "tests"
    tests_dir.mkdir(parents=True)
    (driver_dir / "rollout_driver.py").write_text("from .tests.helper import VALUE\n")
    (tests_dir / "helper.py").write_text("VALUE = 1\n")

    with pytest.raises(ConfigError, match="imports.*excluded tests tree"):
        inspect_workload(_config(driver="environments.driver_case.rollout_driver:run"))


def test_aliased_component_source_directory_is_hash_locked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    component_dir = tmp_path / "responses_api_agents" / "shared_agent"
    config_path = component_dir / "configs" / "alias_agent.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("alias_instance:\n  responses_api_agents:\n    alias_agent:\n      entrypoint: app.py\n")
    source = component_dir / "app.py"
    source.write_text("VALUE = 1\n")
    config = _config(agent="alias_agent")
    config["config_paths"] = [str(config_path)]

    baseline = inspect_workload(config).composition_hash
    source.write_text("VALUE = 2\n")

    assert inspect_workload(config).composition_hash != baseline


def test_cross_component_imports_are_hash_locked_transitively(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    base_dir = tmp_path / "resources_servers" / "base_scorer"
    middle_dir = tmp_path / "resources_servers" / "middle_scorer"
    selected_dir = tmp_path / "resources_servers" / "selected_scorer"
    for directory in (base_dir, middle_dir, selected_dir):
        directory.mkdir(parents=True)
    base_source = base_dir / "app.py"
    base_source.write_text("VALUE = 1\n")
    (middle_dir / "app.py").write_text("from resources_servers.base_scorer.app import VALUE\n")
    (selected_dir / "app.py").write_text("from resources_servers.middle_scorer.app import VALUE\n")
    config = _config()
    config["resources"]["resources_servers"] = {
        "selected_scorer": {"entrypoint": "app.py", "provides": ["verification"]},
    }

    baseline = inspect_workload(config).composition_hash
    base_source.write_text("VALUE = 2\n")

    assert inspect_workload(config).composition_hash != baseline


def test_configured_component_modules_are_hash_locked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    selected_dir = tmp_path / "resources_servers" / "selected_scorer"
    selected_dir.mkdir(parents=True)
    (selected_dir / "app.py").write_text("VALUE = 1\n")
    delegated_dir = tmp_path / "responses_api_agents" / "delegated_agent"
    delegated_dir.mkdir(parents=True)
    delegated_source = delegated_dir / "app.py"
    delegated_source.write_text("VALUE = 1\n")
    config = _config()
    config["resources"]["resources_servers"] = {
        "selected_scorer": {
            "entrypoint": "app.py",
            "provides": ["verification"],
            "agent_server_module": "responses_api_agents.delegated_agent.app",
        },
    }

    baseline = inspect_workload(config).composition_hash
    delegated_source.write_text("VALUE = 2\n")

    assert inspect_workload(config).composition_hash != baseline


def test_composition_hash_binds_resolved_scoring_overrides_and_their_files(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "tasks.jsonl"
    input_path.write_text('{"responses_create_params":{"input":"first task"}}\n')
    prompt_path = tmp_path / "prompt.yaml"
    prompt_path.write_text('user: "Solve {question}"\n')
    config = _config()
    config.update(
        {
            "agent_name": "simple_agent",
            "input_jsonl_fpath": str(input_path),
            "limit": 10,
            "num_repeats": 1,
            "num_repeats_add_seed": False,
            "prompt_config": str(prompt_path),
            "responses_create_params": {"temperature": 0.1, "max_output_tokens": 64},
            "split": "test",
        }
    )
    baseline = inspect_workload(config).composition_hash

    mutations = (
        ("agent_name", "another_agent"),
        ("input_jsonl_fpath", str(tmp_path / "other-tasks.jsonl")),
        ("limit", 5),
        ("num_repeats", 2),
        ("num_repeats_add_seed", True),
        ("prompt_config", str(tmp_path / "other-prompt.yaml")),
        ("responses_create_params", {"temperature": 0.7, "max_output_tokens": 64}),
        ("split", "validation"),
    )
    for key, value in mutations:
        changed = deepcopy(config)
        changed[key] = value
        assert inspect_workload(changed).composition_hash != baseline, key

    input_path.write_text('{"responses_create_params":{"input":"changed task"}}\n')
    assert inspect_workload(config).composition_hash != baseline
    input_path.write_text('{"responses_create_params":{"input":"first task"}}\n')

    prompt_path.write_text('user: "Carefully solve {question}"\n')
    assert inspect_workload(config).composition_hash != baseline


def test_composition_hash_binds_top_level_skills_path_and_contents(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    skill_file = tmp_path / "skills-a" / "reasoning" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("---\nname: reasoning\n---\nThink carefully.\n")
    config = _config()
    config["skills"] = {"path": "skills-a"}

    inspection = inspect_workload(config)
    baseline = inspection.composition_hash
    assert inspection.overrides["skills"] == {"path": "skills-a"}

    skill_file.write_text("---\nname: reasoning\n---\nUse a different policy.\n")
    assert inspect_workload(config).composition_hash != baseline

    second_skill = tmp_path / "skills-b" / "reasoning" / "SKILL.md"
    second_skill.parent.mkdir(parents=True)
    second_skill.write_text("---\nname: reasoning\n---\nThink carefully.\n")
    moved = deepcopy(config)
    moved["skills"] = {"path": "skills-b"}
    assert inspect_workload(moved).composition_hash != baseline


def test_token_settings_affect_composition_but_credential_values_do_not(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = _config()
    resource = config["resources"]["resources_servers"]["scorer"]
    resource.update(
        {
            "api_key": "first-secret",
            "judge_api_key": "first-judge-secret",
            "max_output_tokens": 64,
            "tokenizer": "tokenizer-v1",
            "abstention_token": "[IDK]",
            "input_cost_per_token": 0.1,
        }
    )
    baseline = inspect_workload(config).composition_hash

    credentials_changed = deepcopy(config)
    credentials = credentials_changed["resources"]["resources_servers"]["scorer"]
    credentials["api_key"] = "second-secret"
    credentials["judge_api_key"] = "second-judge-secret"
    assert inspect_workload(credentials_changed).composition_hash == baseline

    for key, value in (
        ("max_output_tokens", 128),
        ("tokenizer", "tokenizer-v2"),
        ("abstention_token", "[UNKNOWN]"),
        ("input_cost_per_token", 0.2),
    ):
        changed = deepcopy(config)
        changed["resources"]["resources_servers"]["scorer"][key] = value
        assert inspect_workload(changed).composition_hash != baseline, key

    # Catalog/deployment metadata is ignored only at the component root.  The
    # same field names may be model-visible values in nested request payloads.
    nested = deepcopy(config)
    nested["resources"]["resources_servers"]["scorer"]["request"] = {
        "metadata": {"description": "first rubric", "modality": "text", "token": "A", "value": "A"}
    }
    nested_baseline = inspect_workload(nested).composition_hash
    nested["resources"]["resources_servers"]["scorer"]["request"]["metadata"]["description"] = "new rubric"
    assert inspect_workload(nested).composition_hash != nested_baseline
    token_baseline = inspect_workload(nested).composition_hash
    nested["resources"]["resources_servers"]["scorer"]["request"]["metadata"]["token"] = "B"
    assert inspect_workload(nested).composition_hash != token_baseline


def test_cross_package_runtime_file_is_hash_bound_and_validated(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "benchmarks" / "shared_prompt" / "prompt.yaml"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("prompt: first\n")
    config = _config()
    config["agent"]["responses_api_agents"]["simple_agent"]["turn2_prompt_fpath"] = (
        "benchmarks/shared_prompt/prompt.yaml"
    )

    baseline = inspect_workload(config).composition_hash
    assert validate_runtime_local_references(config) == (
        f"runtime input agent.responses_api_agents.simple_agent.turn2_prompt_fpath resolves to {prompt}",
    )
    prompt.write_text("prompt: second\n")
    assert inspect_workload(config).composition_hash != baseline

    prompt.unlink()
    with pytest.raises(ConfigError, match="turn2_prompt_fpath.*does not exist"):
        validate_runtime_local_references(config)


def test_runtime_file_reference_rejects_outside_and_symlink_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.yaml"
    outside.write_text("prompt: outside\n")
    config = _config()
    agent = config["agent"]["responses_api_agents"]["simple_agent"]
    agent["prompt_fpath"] = str(outside)
    with pytest.raises(ConfigError, match="outside repository"):
        validate_runtime_local_references(config)

    inside = tmp_path / "prompts" / "linked.yaml"
    inside.parent.mkdir()
    inside.symlink_to(outside)
    agent["prompt_fpath"] = str(inside)
    with pytest.raises(ConfigError, match="symbolic-link"):
        validate_runtime_local_references(config)


def test_runtime_file_reference_supports_component_relative_paths_and_ignores_imports(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    component = tmp_path / "resources_servers" / "scorer"
    prompt = component / "prompt_templates" / "judge.txt"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("Judge this response.\n")
    config = _config()
    config["resources"]["resources_servers"]["scorer"]["judge_prompt_template_fpath"] = "prompt_templates/judge.txt"
    config["agent"]["responses_api_agents"]["simple_agent"]["harbor_agent_import_path"] = (
        "responses_api_agents.harbor_agent.custom_agents.terminus_2:Terminus2NemoGym"
    )

    decisions = validate_runtime_local_references(config)
    assert decisions == (
        f"runtime input resources.resources_servers.scorer.judge_prompt_template_fpath resolves to {prompt}",
    )


def test_ioi_shared_directory_is_runtime_scratch_not_an_input(monkeypatch) -> None:
    monkeypatch.delenv("CCC_TEST_FILE", raising=False)
    monkeypatch.delenv("SHARED_TEMP_DIR", raising=False)
    repo_root = Path(__file__).parents[2]
    config = OmegaConf.load(repo_root / "benchmarks/ioi/config.yaml")

    references = dict(iter_runtime_path_values(config))

    assert (
        references["ioi_resources_server.resources_servers.competitive_coding_challenges.test_file"]
        == "benchmarks/ioi/data/ioi24_metadata.json"
    )
    assert not any(field.endswith(".shared_dir") for field in references)


def test_runtime_file_scanner_ignores_top_level_command_paths() -> None:
    references = tuple(
        iter_runtime_path_values(
            {
                "manifest_path": "/outside/manifest.yaml",
                "replay_rollouts_path": "/outside/rollouts.jsonl",
                "output_jsonl_fpath": "/outside/results.jsonl",
                "component": {
                    "rubric_fpath": "prompts/rubric.txt",
                    "runtime_input_paths": {"schema": "schemas/custom.json"},
                },
            }
        )
    )

    assert references == (
        ("component.rubric_fpath", "prompts/rubric.txt"),
        ("component.runtime_input_paths.schema", "schemas/custom.json"),
    )


def test_runtime_path_roles_cover_harbor_outputs_and_multimodal_inputs(monkeypatch) -> None:
    for variable in (
        "LEGAL_AGENT_BENCH_TASK_CACHE_DIR",
        "LEGAL_AGENT_BENCH_RUNTIME_TASKS_DIR",
        "LEGAL_AGENT_BENCH_SKILLS_DIR",
    ):
        monkeypatch.delenv(variable, raising=False)
    repo_root = Path(__file__).parents[2]
    legal_config = OmegaConf.load(repo_root / "resources_servers/legal_agent_bench/configs/legal_agent_bench.yaml")

    legal_fields = dict(iter_runtime_path_values(legal_config))
    assert not any(field.endswith(".harbor_jobs_dir") for field in legal_fields)
    assert any(field.endswith(".harbor_tasks_dir") for field in legal_fields)
    resolve_runtime_local_references(
        legal_config,
        repo_root=repo_root,
        require_existing=True,
    )

    multimodal_config = OmegaConf.load(repo_root / "resources_servers/labbench2_vlm/configs/labbench2_vlm.yaml")
    multimodal_fields = dict(iter_runtime_path_values(multimodal_config))
    media_fields = {field: value for field, value in multimodal_fields.items() if field.endswith(".media_base_dir")}
    assert tuple(media_fields.values()) == ("resources_servers/labbench2_vlm/data",)


def test_real_stateful_workload_satisfies_manifest_runtime_capabilities() -> None:
    repo_root = Path(__file__).parents[2]
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {
            "config_paths": [
                str(repo_root / "resources_servers/example_session_state_mgmt/configs/example_session_state_mgmt.yaml")
            ]
        },
    )
    config = StaticValidationConfigParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
        )
    )
    manifest = EnvironmentManifest.model_validate(
        {
            "name": "example_session_state_mgmt",
            "version": "1.0.0",
            "kind": "environment",
            "integration_profile": "stock-loop",
            "domain": "agent",
            "description": "Session state capability acceptance fixture.",
            "modality": "text",
            "licensing": "Apache-2.0",
            "authors": ["Gym maintainers"],
            "reward": {"range": [0, 1], "higher_is_better": True},
            "determinism": "seeded",
            "resources_server": "example_session_state_mgmt",
            "agent_server": "simple_agent",
            "model_server": "policy_model",
            "session_model": "episode",
            "state": "per_session",
            "datasets": [
                {
                    "name": "example",
                    "type": "example",
                    "jsonl_fpath": "resources_servers/example_session_state_mgmt/data/example.jsonl",
                    "num_repeats": 1,
                }
            ],
            "requires": ["text-model"],
            "provides": ["verification", "tools"],
        }
    )

    inspection = inspect_workload(config, manifest=manifest)

    assert "environment requires session:episode: satisfied by example_session_state_mgmt_resources_server" in (
        inspection.compatibility_decisions
    )
    assert "environment requires state:per_session: satisfied by example_session_state_mgmt_resources_server" in (
        inspection.compatibility_decisions
    )


def test_real_multimodal_workload_has_an_image_capable_model_path() -> None:
    repo_root = Path(__file__).parents[2]
    initial = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {
            "config_paths": [
                str(repo_root / "benchmarks/labbench2_vlm/config.yaml"),
                str(repo_root / "responses_api_models/openai_model/configs/openai_model.yaml"),
            ]
        },
    )
    config = StaticValidationConfigParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=initial,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
        )
    )
    manifest = EnvironmentManifest.model_validate(
        {
            "name": "labbench2_vlm",
            "version": "1.0.0",
            "kind": "benchmark",
            "integration_profile": "measured-loop",
            "domain": "knowledge",
            "description": "Vision-language capability acceptance fixture.",
            "modality": "image",
            "licensing": "CC-BY-SA-4.0",
            "authors": ["Gym maintainers"],
            "reward": {"range": [0, 1], "higher_is_better": True},
            "determinism": "stochastic",
            "resources_server": "labbench2_vlm",
            "agent_server": "labbench2_vlm_agent",
            "model_server": "policy_model",
            "datasets": [
                {
                    "name": "labbench2_vlm",
                    "type": "benchmark",
                    "jsonl_fpath": "benchmarks/labbench2_vlm/data/labbench2_vlm_benchmark.jsonl",
                    "prepare_script": "benchmarks/labbench2_vlm/prepare.py",
                    "num_repeats": 3,
                }
            ],
            "canonical_split": "test",
            "standard_prompt_config": "benchmarks/prompts/generic/math.yaml",
            "requires": ["image-model"],
            "provides": ["verification"],
        }
    )

    inspection = inspect_workload(config, manifest=manifest)

    assert "labbench2_vlm_benchmark_simple_agent requires image-model: satisfied by policy_model" in (
        inspection.compatibility_decisions
    )
    assert "environment requires image-model: satisfied by policy_model" in inspection.compatibility_decisions


def test_runtime_path_roles_capture_gdpval_reference_inputs(tmp_path: Path) -> None:
    references = tmp_path / "reference_deliverables"
    references.mkdir()
    resolved = resolve_runtime_local_references(
        {"gdpval": {"reference_deliverables_dir": str(references)}},
        repo_root=tmp_path,
        require_existing=True,
    )

    assert len(resolved) == 1
    assert resolved[0].path == references


def test_composition_hash_tracks_component_sources_but_excludes_non_runtime_files(tmp_path, monkeypatch) -> None:
    component_dir = tmp_path / "resources_servers" / "scorer"
    (component_dir / "configs").mkdir(parents=True)
    (component_dir / "tests").mkdir()
    (component_dir / "prompts").mkdir()
    (component_dir / "app.py").write_text("from .scoring import score\n")
    helper = component_dir / "scoring.py"
    helper.write_text("def score(): return 1\n")
    readme = component_dir / "README.md"
    readme.write_text("Initial documentation\n")
    other_config = component_dir / "configs" / "unused.yaml"
    other_config.write_text("unused: true\n")
    config_asset = component_dir / "configs" / "rubric.txt"
    config_asset.write_text("Award one point.\n")
    test_source = component_dir / "tests" / "test_app.py"
    test_source.write_text("def test_placeholder(): pass\n")
    prompt = component_dir / "prompts" / "judge.txt"
    prompt.write_text("Award one point.\n")

    monkeypatch.setattr(validation_module, "_resolve_under_cwd_or_install", lambda path: tmp_path / path)
    config = _config()
    # A declared component version is useful display metadata, but must never
    # suppress source-content locking.
    config["resources"]["resources_servers"]["scorer"]["version"] = "1.2.3"
    baseline = inspect_workload(config).composition_hash

    helper.write_text("def score(): return 0\n")
    assert inspect_workload(config).composition_hash != baseline
    helper.write_text("def score(): return 1\n")
    assert inspect_workload(config).composition_hash == baseline

    (component_dir / "pylock.nemo-gym.toml").write_text('lock-version = "1.0"\npackages = []\n')
    dependency_locked = inspect_workload(config).composition_hash
    assert dependency_locked != baseline

    for path, replacement in (
        (readme, "Reworded documentation\n"),
        (other_config, "unused: false\n"),
        (test_source, "def test_placeholder(): assert True\n"),
    ):
        original = path.read_text()
        path.write_text(replacement)
        assert inspect_workload(config).composition_hash == dependency_locked
        path.write_text(original)

    prompt.write_text("Award zero points.\n")
    assert inspect_workload(config).composition_hash != dependency_locked

    prompt.write_text("Award one point.\n")
    config_asset.write_text("Award zero points.\n")
    assert inspect_workload(config).composition_hash != dependency_locked


def test_component_lock_rejects_entrypoint_under_excluded_tests_tree(tmp_path, monkeypatch) -> None:
    component_dir = tmp_path / "resources_servers" / "scorer"
    entrypoint = component_dir / "tests" / "runtime.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("def score(): return 1\n")
    config = _config()
    config["resources"]["resources_servers"]["scorer"]["entrypoint"] = "tests/runtime.py"

    monkeypatch.setattr(validation_module, "_resolve_under_cwd_or_install", lambda path: tmp_path / path)
    with pytest.raises(ConfigError, match="entrypoint.*excluded tests/cache tree"):
        inspect_workload(config)


def test_component_lock_rejects_runtime_import_from_excluded_tests_tree(tmp_path, monkeypatch) -> None:
    component_dir = tmp_path / "resources_servers" / "scorer"
    helper = component_dir / "tests" / "runtime_helper.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("def score(): return 1\n")
    (component_dir / "app.py").write_text("from .tests.runtime_helper import score\n")

    monkeypatch.setattr(validation_module, "_resolve_under_cwd_or_install", lambda path: tmp_path / path)
    with pytest.raises(ConfigError, match="imports.*excluded tests tree"):
        inspect_workload(_config())


def test_composition_hash_tracks_internal_symlink_target_contents(tmp_path, monkeypatch) -> None:
    component_dir = tmp_path / "resources_servers" / "scorer"
    (component_dir / "prompts").mkdir(parents=True)
    target = component_dir / "prompts" / "judge.txt"
    target.write_text("Award one point.\n")
    (component_dir / "prompt.txt").symlink_to(Path("prompts") / "judge.txt")

    monkeypatch.setattr(validation_module, "_resolve_under_cwd_or_install", lambda path: tmp_path / path)
    baseline = inspect_workload(_config()).composition_hash
    target.write_text("Award zero points.\n")
    changed = inspect_workload(_config()).composition_hash
    assert changed != baseline

    link = component_dir / "prompt.txt"
    deployed_content = link.read_bytes()
    link.unlink()
    link.write_bytes(deployed_content)
    assert inspect_workload(_config()).composition_hash == changed


@pytest.mark.parametrize("target_kind", ["file", "directory"])
def test_component_tree_rejects_symlinks_outside_the_deployed_subtree(tmp_path, monkeypatch, target_kind: str) -> None:
    component_dir = tmp_path / "resources_servers" / "scorer"
    component_dir.mkdir(parents=True)
    (component_dir / "app.py").write_text("def score(): return 1\n")
    if target_kind == "file":
        external_target = tmp_path / "external_helper.py"
        external_target.write_text("def score(): return 1\n")
        deployed_link = component_dir / "external_helper.py"
    else:
        external_target = tmp_path / "external_helpers"
        external_target.mkdir()
        (external_target / "scoring.py").write_text("def score(): return 1\n")
        deployed_link = component_dir / "external_helpers"
    deployed_link.symlink_to(external_target, target_is_directory=target_kind == "directory")

    monkeypatch.setattr(validation_module, "_resolve_under_cwd_or_install", lambda path: tmp_path / path)
    with pytest.raises(ConfigError, match="outside the component tree"):
        inspect_workload(_config())


@pytest.mark.parametrize("linked_level", ["component", "registry"])
def test_component_tree_rejects_symlinked_component_roots(tmp_path, monkeypatch, linked_level: str) -> None:
    external_registry = tmp_path / "external_resources"
    external_component = external_registry / "scorer"
    external_component.mkdir(parents=True)
    (external_component / "app.py").write_text("def score(): return 1\n")

    registry = tmp_path / "resources_servers"
    if linked_level == "registry":
        registry.symlink_to(external_registry, target_is_directory=True)
    else:
        registry.mkdir()
        (registry / "scorer").symlink_to(external_component, target_is_directory=True)

    monkeypatch.setattr(validation_module, "_resolve_under_cwd_or_install", lambda path: tmp_path / path)
    with pytest.raises(ConfigError, match="symbolic-link component or registry"):
        inspect_workload(_config())


def test_composition_hash_includes_manifest_reward_and_standard_prompt_contents(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    resource_dir = tmp_path / "resources_servers" / "scorer"
    resource_dir.mkdir(parents=True)
    (resource_dir / "app.py").write_text("def score(): return 1\n")
    dataset = tmp_path / "benchmark.jsonl"
    dataset.write_text(json.dumps({"question": "2+2", "expected_answer": "4"}) + "\n")
    prompt = tmp_path / "prompt.yaml"
    prompt.write_text('user: "Solve {question}"\n')
    config = _config(
        dataset={
            "name": "fixture",
            "type": "benchmark",
            "jsonl_fpath": str(dataset),
            "prompt_config": str(prompt),
        }
    )
    config["policy_model"] = {
        "responses_api_models": {
            "openai_model": {"entrypoint": "app.py", "provides": ["text-model"]},
        }
    }
    config["agent"]["responses_api_agents"]["simple_agent"].update(
        {
            "requires": ["verification", "text-model"],
            "model_server": {"type": "responses_api_models", "name": "policy_model"},
        }
    )
    manifest = EnvironmentManifest.model_validate(
        {
            "name": "fixture",
            "version": "1.0.0",
            "kind": "benchmark",
            "integration_profile": "stock-loop",
            "domain": "other",
            "description": "A benchmark composition hash fixture.",
            "modality": "text",
            "licensing": "Apache-2.0",
            "authors": ["contributor"],
            "reward": {"range": [0, 1], "higher_is_better": True},
            "resources_server": "scorer",
            "agent_server": "simple_agent",
            "model_server": "policy_model",
            "datasets": [
                {
                    "name": "fixture",
                    "type": "benchmark",
                    "jsonl_fpath": str(dataset),
                    "prepare_script": "prepare.py",
                    "prompt_config": str(prompt),
                }
            ],
            "canonical_split": "test",
            "standard_prompt_config": str(prompt),
        }
    )
    first = inspect_workload(config, manifest=manifest).composition_hash

    changed_reward = manifest.model_copy(update={"reward": manifest.reward.model_copy(update={"range": (0.0, 2.0)})})
    assert inspect_workload(config, manifest=changed_reward).composition_hash != first

    prompt.write_text('user: "Carefully solve {question}"\n')
    assert inspect_workload(config, manifest=manifest).composition_hash != first


def test_resolve_composition_mirror_and_deltas() -> None:
    config = _config(dataset={"name": "example", "type": "example", "jsonl_fpath": "data.jsonl"})
    config["resources"]["resources_servers"]["scorer"]["grading_mode"] = "strict"
    config["agent"]["responses_api_agents"]["simple_agent"]["model_server"] = {
        "type": "responses_api_models",
        "name": "policy_model",
    }
    mirror = resolve_composition_mirror(config)
    assert mirror.resources_server == "scorer"
    assert mirror.agent_server == "simple_agent"
    assert mirror.model_server == "policy_model"
    assert mirror.grading_mode == "strict"

    manifest = EnvironmentManifest.model_validate(
        {
            "name": "example",
            "version": "1.0.0",
            "kind": "environment",
            "integration_profile": "stock-loop",
            "domain": "other",
            "description": "Example environment",
            "modality": "text",
            "licensing": "unknown",
            "authors": ["contributor"],
            "reward": {"range": [0, 1], "higher_is_better": True},
            "resources_server": "wrong",
            "grading_mode": "strict",
            "agent_server": "simple_agent",
            "model_server": "policy_model",
            "datasets": [{"name": "example", "type": "example", "jsonl_fpath": "data.jsonl", "num_repeats": 1}],
        }
    )
    assert manifest_composition_deltas(manifest, mirror) == (
        "resources_server: manifest='wrong', resolved config='scorer'",
    )


def test_external_mirror_ignores_unreferenced_static_dummy_model() -> None:
    config = OmegaConf.merge(GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, OmegaConf.create(_config()))
    agent = config.agent.responses_api_agents.simple_agent
    agent.integration_profile = "external-loop"
    agent.pop("model_server", None)

    assert resolve_composition_mirror(config).model_server is None


def test_composition_mirror_uses_agent_selected_resources_root_not_helper() -> None:
    config = _config(dataset={"name": "example", "type": "example", "jsonl_fpath": "data.jsonl"})
    config["resources"]["resources_servers"]["scorer"]["grading_mode"] = "strict"
    config["resources"]["resources_servers"]["scorer"]["helper"] = {
        "type": "resources_servers",
        "name": "helper_resource",
    }
    config["helper_resource"] = {
        "resources_servers": {
            "helper_impl": {
                "entrypoint": "app.py",
                "grading_mode": "helper-only",
            }
        }
    }

    mirror = resolve_composition_mirror(config)

    assert mirror.resources_server == "scorer"
    assert mirror.grading_mode == "strict"
