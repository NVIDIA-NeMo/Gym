# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from nemo_gym.config_types import ConfigError, Domain
from nemo_gym.environment.manifest import (
    EnvironmentManifest,
    IntegrationProfile,
    ManifestError,
    dump_manifest,
    load_manifest,
    manifest_json_schema,
)


REPO_ROOT = Path(__file__).parents[2]


def test_migration_defaults_and_metadata() -> None:
    manifest = EnvironmentManifest.model_validate(_manifest())
    assert manifest.prompt_source == "template"
    assert manifest.config_path == "config.yaml"
    assert manifest.dataset_owner is None
    raw = _manifest()
    raw.update(config_path="../config_web.yaml", dataset_owner=" web_agent ")
    manifest = EnvironmentManifest.model_validate(raw)
    assert manifest.config_path == "../config_web.yaml"
    assert manifest.dataset_owner == "web_agent"


@pytest.mark.parametrize("backend", ["pydantic", "schema"])
@pytest.mark.parametrize("kind", ["environment", "benchmark"])
@pytest.mark.parametrize("source", ["missing", "template", "prepared", "agent"])
@pytest.mark.parametrize("prompt", ["missing", None, "prompt.yaml"])
@pytest.mark.parametrize("dataset_prompt", [None, "runtime.yaml"])
def test_prompt_contract(backend, kind, source, prompt, dataset_prompt) -> None:
    raw = _manifest(kind=kind)
    raw.pop("standard_prompt_config", None)
    if source != "missing":
        raw["prompt_source"] = source
    if prompt != "missing":
        raw["standard_prompt_config"] = prompt
    raw["datasets"][0]["prompt_config"] = dataset_prompt
    effective_source = "template" if source == "missing" else source
    valid = not (
        (kind == "benchmark" and effective_source == "template" and prompt in ("missing", None))
        or (effective_source == "prepared" and prompt not in ("missing", None))
        or (effective_source != "template" and dataset_prompt is not None)
    )
    if backend == "schema":
        assert Draft202012Validator(manifest_json_schema()).is_valid(raw) == valid
    elif valid:
        assert EnvironmentManifest.model_validate(raw).prompt_source == effective_source
    else:
        with pytest.raises(ValidationError, match="prompt"):
            EnvironmentManifest.model_validate(raw)


@pytest.mark.parametrize("source", ["template", "prepared", "agent"])
@pytest.mark.parametrize("split", [None, "missing"])
def test_all_benchmark_prompt_sources_require_canonical_split(source, split) -> None:
    raw = _manifest(kind="benchmark")
    raw.update(prompt_source=source, standard_prompt_config=None)
    raw["datasets"][0].pop("prompt_config")
    raw["canonical_split"] = split
    if split == "missing":
        raw.pop("canonical_split")
    with pytest.raises(ValidationError, match="canonical_split"):
        EnvironmentManifest.model_validate(raw)
    assert not Draft202012Validator(manifest_json_schema()).is_valid(raw)


@pytest.mark.parametrize("profile", list(IntegrationProfile))
@pytest.mark.parametrize("resource", ["missing", None, "verifier"])
def test_resources_server_required_but_nullable_by_profile(profile, resource) -> None:
    raw = _manifest(profile=profile)
    raw["resources_server"] = resource
    if resource == "missing":
        raw.pop("resources_server")
    valid = resource != "missing" and (resource is not None or profile != "custom-gym-verifier")
    assert Draft202012Validator(manifest_json_schema()).is_valid(raw) == valid
    if valid:
        manifest = EnvironmentManifest.model_validate(raw)
        dumped = yaml.safe_load(dump_manifest(manifest))
        assert "resources_server" in dumped
        assert dumped["resources_server"] == resource
        assert EnvironmentManifest.model_validate(dumped) == manifest
    else:
        with pytest.raises(ValidationError, match="resources_server"):
            EnvironmentManifest.model_validate(raw)


@pytest.mark.parametrize("field,value", [("config_path", ""), ("dataset_owner", ""), ("prompt_source", "unknown")])
def test_migration_fields_reject_invalid_values(field, value) -> None:
    raw = _manifest()
    raw[field] = value
    with pytest.raises(ValidationError, match=field):
        EnvironmentManifest.model_validate(raw)
    assert not Draft202012Validator(manifest_json_schema()).is_valid(raw)


@pytest.mark.parametrize("tree", ["benchmarks", "environments"])
@pytest.mark.parametrize("config", ["config.yaml", "../config_web.yaml", "../../shared.yaml"])
def test_resolve_manifest_config_within_catalog(tmp_path: Path, tree: str, config: str) -> None:
    from nemo_gym.environment import manifest as module

    raw = _manifest()
    raw["config_path"] = config
    manifest = EnvironmentManifest.model_validate(raw)
    path = tmp_path / tree / "foo" / "config_web" / "manifest.yaml"
    assert module.resolve_manifest_config_path(path, manifest) == (path.parent / config).resolve()


@pytest.mark.parametrize("config", ["/tmp/config.yaml", "../../../outside.yaml", "../../../environments/a.yaml"])
def test_resolve_manifest_config_rejects_escape(tmp_path: Path, config: str) -> None:
    from nemo_gym.environment import manifest as module

    raw = _manifest()
    raw["config_path"] = config
    manifest = EnvironmentManifest.model_validate(raw)
    path = tmp_path / "benchmarks" / "foo" / "config_web" / "manifest.yaml"
    with pytest.raises(ManifestError, match="config_path"):
        module.resolve_manifest_config_path(path, manifest)


def test_resolve_manifest_config_rejects_symlink_escape(tmp_path: Path) -> None:
    from nemo_gym.environment import manifest as module

    directory = tmp_path / "benchmarks" / "foo"
    directory.mkdir(parents=True)
    (directory / "config.yaml").symlink_to(tmp_path / "outside.yaml")
    with pytest.raises(ManifestError, match="config_path"):
        module.resolve_manifest_config_path(
            directory / "manifest.yaml", EnvironmentManifest.model_validate(_manifest())
        )


def _manifest(*, profile: str = "custom-gym-verifier", kind: str = "environment") -> dict:
    manifest = {
        "name": "my_eval",
        "version": "1.0.0",
        "kind": kind,
        "integration_profile": profile,
        "domain": "math",
        "description": "Competition mathematics evaluation",
        "modality": "text",
        "licensing": "Apache-2.0",
        "authors": ["alice"],
        "reward": {"range": [0, 1], "higher_is_better": True},
        "determinism": "seeded",
        "resources_server": "mcqa",
        "agent_server": "simple_agent",
        "model_server": "policy_model",
        "datasets": [
            {
                "name": "validation",
                "type": "validation",
                "jsonl_fpath": "data/validation.jsonl",
            }
        ],
    }
    if profile == "external-rollout-driver":
        manifest["rollout_driver"] = "my_eval.driver:collect"
    if kind == "benchmark":
        manifest["canonical_split"] = "validation"
        manifest["standard_prompt_config"] = "prompts/standard.yaml"
        manifest["datasets"][0]["type"] = "benchmark"
        manifest["datasets"][0]["prepare_script"] = "prepare.py"
        manifest["datasets"][0]["prompt_config"] = "prompts/standard.yaml"
    return manifest


def test_environment_manifest_parses_and_uses_declared_defaults() -> None:
    raw = _manifest()
    raw.pop("licensing")
    raw.pop("determinism")
    manifest = EnvironmentManifest.model_validate(raw)

    assert manifest.name == "my_eval"
    assert manifest.experimental is True
    assert manifest.licensing == "unknown"
    assert manifest.determinism.value == "unknown"
    assert manifest.lifecycle.value == "active"
    assert manifest.session_model is None

    raw["experimental"] = False
    assert EnvironmentManifest.model_validate(raw).experimental is False


def test_integration_profiles_are_closed_string_values() -> None:
    assert [profile.value for profile in IntegrationProfile] == [
        "custom-gym-verifier",
        "custom-gym-agent-loop",
        "external-agent-loop",
        "external-rollout-driver",
    ]
    assert str(IntegrationProfile.CUSTOM_GYM_VERIFIER) == "custom-gym-verifier"


@pytest.mark.parametrize("missing_field", ["resources_server", "agent_server", "datasets"])
def test_manifest_requires_base_composition(missing_field: str) -> None:
    raw = _manifest()
    raw.pop(missing_field)

    with pytest.raises(ValidationError, match=missing_field):
        EnvironmentManifest.model_validate(raw)


@pytest.mark.parametrize("profile", ["custom-gym-verifier", "custom-gym-agent-loop"])
def test_in_process_profiles_require_a_model_server(profile: str) -> None:
    raw = _manifest(profile=profile)
    raw.pop("model_server")

    with pytest.raises(ValidationError, match="model_server"):
        EnvironmentManifest.model_validate(raw)


def test_external_loop_may_omit_a_model_server() -> None:
    raw = _manifest(profile="external-agent-loop")
    raw.pop("model_server")

    assert EnvironmentManifest.model_validate(raw).model_server is None


def test_custom_driver_requires_a_rollout_driver() -> None:
    raw = _manifest(profile="external-rollout-driver")
    raw.pop("rollout_driver")

    with pytest.raises(ValidationError, match="rollout_driver"):
        EnvironmentManifest.model_validate(raw)


def test_rollout_driver_is_custom_profile_only() -> None:
    stock = _manifest()
    stock["rollout_driver"] = "package.driver:collect"
    with pytest.raises(ValidationError, match="only valid for the external-rollout-driver"):
        EnvironmentManifest.model_validate(stock)

    custom = _manifest(profile="external-rollout-driver")
    custom["rollout_driver"] = "not a callable"
    with pytest.raises(ValidationError, match="rollout_driver"):
        EnvironmentManifest.model_validate(custom)


@pytest.mark.parametrize("missing_field", ["canonical_split", "standard_prompt_config"])
def test_benchmark_requires_protocol_fields(missing_field: str) -> None:
    raw = _manifest(kind="benchmark")
    raw.pop(missing_field)

    with pytest.raises(ValidationError, match=missing_field):
        EnvironmentManifest.model_validate(raw)


def test_benchmark_requires_a_benchmark_dataset() -> None:
    raw = _manifest(kind="benchmark")
    raw["datasets"][0]["type"] = "validation"

    with pytest.raises(ValidationError, match="benchmark dataset"):
        EnvironmentManifest.model_validate(raw)

    raw = _manifest(kind="benchmark")
    raw["datasets"][0].pop("prepare_script")
    with pytest.raises(ValidationError, match="prepare_script"):
        EnvironmentManifest.model_validate(raw)

    raw = _manifest(kind="benchmark")
    raw["datasets"][0].pop("prompt_config")
    assert EnvironmentManifest.model_validate(raw).datasets[0].prompt_config is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("integration_profile", "invented-loop"),
        ("licensing", "not an SPDX id"),
        ("licensing", "FooBar"),
    ],
)
def test_manifest_rejects_invalid_scalar_contracts(field: str, value: str) -> None:
    raw = _manifest()
    raw[field] = value

    with pytest.raises(ValidationError):
        EnvironmentManifest.model_validate(raw)


@pytest.mark.parametrize(("field", "value"), [("name", "Team evaluation"), ("version", "draft-2026")])
def test_manifest_accepts_nonempty_identity_strings(field: str, value: str) -> None:
    raw = _manifest()
    raw[field] = f"  {value}  "

    assert getattr(EnvironmentManifest.model_validate(raw), field) == value


@pytest.mark.parametrize(("field", "value"), [("name", ""), ("name", "   "), ("version", ""), ("version", "   ")])
def test_manifest_rejects_empty_identity_strings(field: str, value: str) -> None:
    raw = _manifest()
    raw[field] = value

    with pytest.raises(ValidationError, match=field):
        EnvironmentManifest.model_validate(raw)


@pytest.mark.parametrize(
    ("licensing", "expected"),
    [
        ("MIT", "MIT"),
        ("mit or apache-2.0", "MIT OR Apache-2.0"),
        ("LicenseRef-Internal-Evaluation", "LicenseRef-Internal-Evaluation"),
        ("internal", "internal"),
        ("proprietary", "proprietary"),
        ("unknown", "unknown"),
    ],
)
def test_manifest_accepts_spdx_expressions_and_private_license_values(licensing: str, expected: str) -> None:
    raw = _manifest()
    raw["licensing"] = licensing

    assert EnvironmentManifest.model_validate(raw).licensing == expected


def test_manifest_rejects_bad_reward_duplicate_authors_and_unknown_fields() -> None:
    raw = _manifest()
    raw["reward"]["range"] = [1, 0]
    raw["authors"] = ["alice", "alice"]
    raw["surprise"] = True

    with pytest.raises(ValidationError) as error:
        EnvironmentManifest.model_validate(raw)
    message = str(error.value)
    assert "lower <= upper" in message
    assert "authors must be unique" in message
    assert "Extra inputs are not permitted" in message


def test_manifest_represents_constant_placeholder_reward() -> None:
    raw = _manifest()
    raw["reward"]["range"] = [0, 0]
    manifest = EnvironmentManifest.model_validate(raw)
    assert manifest.reward.range == (0, 0)
    Draft202012Validator(manifest_json_schema()).validate(raw)


def test_manifest_rejects_duplicate_dataset_names() -> None:
    raw = _manifest()
    raw["datasets"].append(dict(raw["datasets"][0]))

    with pytest.raises(ValidationError, match="dataset names must be unique"):
        EnvironmentManifest.model_validate(raw)


def test_adopted_from_validates_source_format() -> None:
    raw = _manifest()
    raw["adopted_from"] = {
        "source": "https://github.com/example/project.git",
        "ref": "v1.2.0",
        "reconciled": "2026-08-01",
    }
    assert EnvironmentManifest.model_validate(raw).adopted_from.ref == "v1.2.0"

    raw["adopted_from"]["source"] = "github.com/example/project"
    with pytest.raises(ValidationError):
        EnvironmentManifest.model_validate(raw)


def test_load_dump_round_trip_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(dump_manifest(_manifest()))
    assert load_manifest(path).name == "my_eval"

    with pytest.raises(ManifestError, match="was not found"):
        load_manifest(tmp_path / "missing.yaml")

    path.write_text("name: [broken\n")
    with pytest.raises(ManifestError, match="Malformed YAML"):
        load_manifest(path)

    path.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ManifestError, match="expected a YAML mapping"):
        load_manifest(path)

    path.write_text("version: nope\n")
    with pytest.raises(ConfigError, match=r"name.*Field required"):
        load_manifest(path)

    with pytest.raises(ManifestError, match="<memory>"):
        dump_manifest({"name": "incomplete"})


def test_generated_schema_is_machine_readable() -> None:
    schema = manifest_json_schema()

    assert schema["$defs"]["Domain"]["enum"] == [domain.value for domain in Domain]
    assert schema["properties"]["experimental"]["default"] is True
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    assert not list(validator.iter_errors(_manifest()))

    invalid_manifests = []

    for profile in ("custom-gym-verifier", "custom-gym-agent-loop"):
        invalid = _manifest(profile=profile)
        invalid.pop("model_server")
        invalid_manifests.append(invalid)

    invalid = _manifest(profile="external-rollout-driver")
    invalid.pop("rollout_driver")
    invalid_manifests.append(invalid)

    invalid = _manifest()
    invalid["rollout_driver"] = "package.driver:collect"
    invalid_manifests.append(invalid)

    for field in ("canonical_split", "standard_prompt_config"):
        invalid = _manifest(kind="benchmark")
        invalid.pop(field)
        invalid_manifests.append(invalid)

    invalid = _manifest(kind="benchmark")
    invalid["datasets"][0]["type"] = "validation"
    invalid_manifests.append(invalid)

    assert all(list(validator.iter_errors(invalid)) for invalid in invalid_manifests)


def test_neutral_example_matches_manifest_contract() -> None:
    manifest = load_manifest(REPO_ROOT / "examples/environment_manifest.yaml")

    assert manifest.name == "example_environment"
    assert manifest.integration_profile == IntegrationProfile.CUSTOM_GYM_VERIFIER
