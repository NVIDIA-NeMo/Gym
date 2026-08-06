# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.config_types import ConfigError, Domain
from nemo_gym.environment_manifest import (
    EnvironmentManifest,
    ManifestError,
    dump_manifest,
    load_manifest,
    manifest_json_schema,
)


REPO_ROOT = Path(__file__).parents[2]


def _manifest(*, profile: str = "stock-loop", kind: str = "environment") -> dict:
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
    if profile == "custom-driver":
        manifest["rollout_driver"] = "my_eval.driver:collect"
    if kind == "benchmark":
        manifest["canonical_split"] = "validation"
        manifest["standard_prompt_config"] = "prompts/standard.yaml"
        manifest["datasets"][0]["type"] = "benchmark"
        manifest["datasets"][0]["prepare_script"] = "prepare.py"
    return manifest


def test_environment_manifest_parses_and_uses_honest_defaults() -> None:
    raw = _manifest()
    raw.pop("licensing")
    raw.pop("determinism")
    manifest = EnvironmentManifest.model_validate(raw)

    assert manifest.name == "my_eval"
    assert manifest.licensing == "unknown"
    assert manifest.determinism.value == "unknown"
    assert manifest.lifecycle.value == "active"
    assert manifest.session_model is None


@pytest.mark.parametrize(
    "profile, missing_field",
    [
        ("stock-loop", "model_server"),
        ("measured-loop", "agent_server"),
        ("external-loop", "resources_server"),
        ("custom-driver", "rollout_driver"),
    ],
)
def test_profiles_require_their_composition(profile: str, missing_field: str) -> None:
    raw = _manifest(profile=profile)
    raw.pop(missing_field)

    with pytest.raises(ValidationError, match=missing_field):
        EnvironmentManifest.model_validate(raw)


def test_rollout_driver_is_custom_profile_only() -> None:
    stock = _manifest()
    stock["rollout_driver"] = "package.driver:collect"
    with pytest.raises(ValidationError, match="only valid for the custom-driver"):
        EnvironmentManifest.model_validate(stock)

    custom = _manifest(profile="custom-driver")
    custom["rollout_driver"] = "not a callable"
    with pytest.raises(ValidationError, match="string_pattern_mismatch"):
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

    raw["datasets"][0]["type"] = "benchmark"
    raw["datasets"][0].pop("prepare_script", None)
    with pytest.raises(ValidationError, match="requires prepare_script"):
        EnvironmentManifest.model_validate(raw)


@pytest.mark.parametrize(
    "field, value",
    [
        ("name", "../escape"),
        ("version", "1.0"),
        ("integration_profile", "invented-loop"),
        ("licensing", "not an SPDX id"),
    ],
)
def test_manifest_rejects_invalid_scalar_contracts(field: str, value: str) -> None:
    raw = _manifest()
    raw[field] = value

    with pytest.raises(ValidationError):
        EnvironmentManifest.model_validate(raw)


def test_manifest_rejects_bad_reward_duplicates_and_unknown_fields() -> None:
    raw = _manifest()
    raw["reward"]["range"] = [1, 1]
    raw["authors"] = ["alice", "alice"]
    raw["surprise"] = True

    with pytest.raises(ValidationError) as error:
        EnvironmentManifest.model_validate(raw)
    message = str(error.value)
    assert "lower < upper" in message
    assert "authors must be unique" in message
    assert "Extra inputs are not permitted" in message


def test_adopted_from_is_structural_and_offline() -> None:
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

    path.write_text("name: [broken\n")
    with pytest.raises(ManifestError, match="Malformed YAML"):
        load_manifest(path)

    path.write_text("version: nope\n")
    with pytest.raises(ConfigError, match=r"name.*Field required"):
        load_manifest(path)


def test_checked_in_schema_matches_model_and_gym_domain_enum() -> None:
    schema = manifest_json_schema()
    checked_in = json.loads((REPO_ROOT / "schemas/environment-manifest.schema.json").read_text())

    assert checked_in == schema
    assert schema["$defs"]["Domain"]["enum"] == [domain.value for domain in Domain]
    assert len(schema["allOf"]) == 6
