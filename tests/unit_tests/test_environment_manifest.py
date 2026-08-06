# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import nemo_gym.environment_manifest as manifest_module
from nemo_gym.config_types import ConfigError, Domain
from nemo_gym.environment_manifest import (
    Determinism,
    EnvironmentKind,
    EnvironmentManifest,
    Lifecycle,
    ManifestError,
    dump_manifest,
    grading_modes_from_source,
    load_manifest,
    manifest_json_schema,
    resources_server_grading_modes,
    validate_adopted_from_reference,
)


SCHEMA_PATH = Path(__file__).parents[2] / "schemas" / "environment-manifest.schema.json"


def _environment_data(**overrides):
    data = {
        "name": "my_eval",
        "version": "1.2.3",
        "kind": "environment",
        "integration_profile": "stock-loop",
        "domain": "math",
        "description": "Competition mathematics evaluation tasks",
        "modality": "text",
        "licensing": "Apache-2.0",
        "authors": ["alice", "bob"],
        "reward": {"range": [0, 1], "higher_is_better": True},
        "determinism": "seeded",
        "resources_server": "mcqa",
        "agent_server": "simple_agent",
        "model_server": "policy_model",
        "datasets": [
            {
                "name": "validation",
                "type": "validation",
                "jsonl_fpath": "environments/my_eval/data/validation.jsonl",
            }
        ],
        "requires": ["text-model"],
        "provides": ["scalar-reward"],
    }
    data.update(overrides)
    return data


def _benchmark_data(**overrides):
    data = _environment_data(
        name="sciqa",
        version="2.0.0-rc.1+build.7",
        kind="benchmark",
        canonical_split="test",
        standard_prompt_config="benchmarks/prompts/eval/mcq.yaml",
        datasets=[
            {
                "name": "sciqa",
                "type": "benchmark",
                "jsonl_fpath": "benchmarks/sciqa/data/sciqa.jsonl",
                "prepare_script": "benchmarks/sciqa/prepare.py",
                "prompt_config": "benchmarks/prompts/eval/mcq.yaml",
                "num_repeats": 8,
            }
        ],
    )
    data.update(overrides)
    return data


def test_environment_manifest_validates_all_core_fields() -> None:
    manifest = EnvironmentManifest.model_validate(_environment_data())

    assert manifest.kind is EnvironmentKind.ENVIRONMENT
    assert manifest.domain is Domain.MATH
    assert manifest.determinism is Determinism.SEEDED
    assert manifest.lifecycle is Lifecycle.ACTIVE
    assert manifest.datasets and manifest.datasets[0].num_repeats == 1


def test_defaults_admit_unknown_claims_without_inventing_facts() -> None:
    data = _environment_data()
    data.pop("licensing")
    data.pop("determinism")

    manifest = EnvironmentManifest.model_validate(data)

    assert manifest.licensing == "unknown"
    assert manifest.determinism is Determinism.UNKNOWN


def test_benchmark_requires_protocol_fields_and_benchmark_dataset() -> None:
    manifest = EnvironmentManifest.model_validate(_benchmark_data())
    assert manifest.canonical_split == "test"

    for missing_field in ("canonical_split", "standard_prompt_config"):
        data = _benchmark_data()
        data.pop(missing_field)
        with pytest.raises(ValidationError, match=missing_field):
            EnvironmentManifest.model_validate(data)

    with pytest.raises(ValidationError, match="type='benchmark'"):
        EnvironmentManifest.model_validate(
            _benchmark_data(datasets=_environment_data()["datasets"]),
        )


def test_benchmark_dataset_prompt_config_matches_standard_protocol() -> None:
    with pytest.raises(ValidationError, match="prompt_config must match standard_prompt_config"):
        EnvironmentManifest.model_validate(
            _benchmark_data(
                datasets=[
                    {
                        **_benchmark_data()["datasets"][0],
                        "prompt_config": "benchmarks/prompts/eval/different.yaml",
                    }
                ]
            )
        )


@pytest.mark.parametrize(
    ("profile", "removed_field"),
    [
        ("stock-loop", "model_server"),
        ("measured-loop", "agent_server"),
        ("external-loop", "resources_server"),
        ("custom-driver", "rollout_driver"),
    ],
)
def test_profile_specific_composition_is_required(profile: str, removed_field: str) -> None:
    data = _environment_data(integration_profile=profile)
    if profile == "custom-driver":
        data["rollout_driver"] = "my_package.driver:run"
    data.pop(removed_field, None)

    with pytest.raises(ValidationError, match=removed_field):
        EnvironmentManifest.model_validate(data)


def test_non_custom_profile_rejects_rollout_driver() -> None:
    with pytest.raises(ValidationError, match="only valid"):
        EnvironmentManifest.model_validate(_environment_data(rollout_driver="package.driver:run"))


def test_custom_profile_requires_a_python_callable_driver() -> None:
    with pytest.raises(ValidationError, match="module.path:function"):
        EnvironmentManifest.model_validate(
            _environment_data(integration_profile="custom-driver", rollout_driver="not-a-driver")
        )


@pytest.mark.parametrize("version", ["1", "1.2", "01.2.3", "1.02.3", "1.2.03", "v1.2.3"])
def test_version_is_semver(version: str) -> None:
    with pytest.raises(ValidationError):
        EnvironmentManifest.model_validate(_environment_data(version=version))


def test_manifest_rejects_placeholders_bad_ranges_duplicates_and_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="at least 10 characters"):
        EnvironmentManifest.model_validate(_environment_data(description="TODO"))
    with pytest.raises(ValidationError, match="replace the TODO placeholder"):
        EnvironmentManifest.model_validate(_environment_data(description="TODO: write a real description"))
    with pytest.raises(ValidationError, match="lower < upper"):
        EnvironmentManifest.model_validate(_environment_data(reward={"range": [1, 1], "higher_is_better": True}))
    with pytest.raises(ValidationError, match="entries must be unique"):
        EnvironmentManifest.model_validate(_environment_data(authors=["alice", "alice"]))
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EnvironmentManifest.model_validate(_environment_data(verified=True))


def test_upstream_provenance_accepts_https_and_scp_git_urls() -> None:
    reconciled = "2026-08-03"
    for source in ("https://github.com/org/repo", "git@gitlab.example.com:org/repo.git"):
        manifest = EnvironmentManifest.model_validate(
            _environment_data(adopted_from={"source": source, "ref": "v1.2.3", "reconciled": reconciled})
        )
        assert manifest.adopted_from and manifest.adopted_from.reconciled.isoformat() == reconciled


@pytest.mark.parametrize(
    "source",
    [
        "relative/repository",
        "javascript://example.com/org/repo",
        "https://example.com",
        "https://example.com/org/repo?access_token=secret",
        "https://access-token@example.com/org/repo",
        "https://user:secret@example.com/org/repo",
    ],
)
def test_upstream_provenance_rejects_non_cloneable_or_credentialed_sources(source: str) -> None:
    with pytest.raises(ValidationError, match="source"):
        EnvironmentManifest.model_validate(
            _environment_data(adopted_from={"source": source, "ref": "main", "reconciled": "2026-08-03"})
        )


@pytest.mark.parametrize("ref", ["@", "../main", "refs/heads/.hidden", "release.lock", "a//b", "main~1", "a b"])
def test_upstream_provenance_rejects_invalid_git_refs(ref: str) -> None:
    with pytest.raises(ValidationError, match="ref"):
        EnvironmentManifest.model_validate(
            _environment_data(
                adopted_from={"source": "ssh://git@example.com/org/repo.git", "ref": ref, "reconciled": "2026-08-03"}
            )
        )


def test_local_upstream_provenance_resolves_source_and_ref(tmp_path: Path) -> None:
    repository = tmp_path / "upstream"
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    (repository / "README.md").write_text("upstream\n")
    subprocess.run(["git", "-C", str(repository), "add", "README.md"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Gym Test",
            "-c",
            "user.email=gym@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "initial",
        ],
        check=True,
    )
    manifest = EnvironmentManifest.model_validate(
        _environment_data(adopted_from={"source": repository.as_uri(), "ref": "HEAD", "reconciled": "2026-08-03"})
    )
    assert manifest.adopted_from is not None
    assert validate_adopted_from_reference(manifest.adopted_from) is None

    invalid = manifest.adopted_from.model_copy(update={"ref": "missing-ref"})
    with pytest.raises(ManifestError, match="does not resolve to a commit"):
        validate_adopted_from_reference(invalid)


def test_remote_upstream_provenance_resolves_source_and_ref(monkeypatch) -> None:
    manifest = EnvironmentManifest.model_validate(
        _environment_data(
            adopted_from={
                "source": "https://github.com/org/repo",
                "ref": "v1.2.3",
                "reconciled": "2026-08-03",
            }
        )
    )
    assert manifest.adopted_from is not None
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="abc refs/tags/v1.2.3\n", stderr="")
    run = monkeypatch.setattr(manifest_module.subprocess, "run", lambda *args, **kwargs: completed)
    assert run is None
    assert validate_adopted_from_reference(manifest.adopted_from) is None


def test_remote_upstream_provenance_resolves_advertised_commit_sha(monkeypatch) -> None:
    commit = "a" * 40
    provenance = EnvironmentManifest.model_validate(
        _environment_data(
            adopted_from={
                "source": "https://github.com/org/repo",
                "ref": commit,
                "reconciled": "2026-08-03",
            }
        )
    ).adopted_from
    assert provenance is not None
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=f"{commit}\trefs/heads/main\n",
            stderr="",
        )

    monkeypatch.setattr(manifest_module.subprocess, "run", run)
    assert validate_adopted_from_reference(provenance) is None
    assert commands == [["git", "ls-remote", "--exit-code", provenance.source]]


@pytest.mark.parametrize(
    "returncode, message",
    [
        (2, "does not resolve to a commit"),
        (128, "repository access and authentication"),
    ],
)
def test_remote_upstream_provenance_reports_ref_and_access_failures(monkeypatch, returncode, message) -> None:
    provenance = EnvironmentManifest.model_validate(
        _environment_data(
            adopted_from={
                "source": "ssh://git@example.invalid/org/repo.git",
                "ref": "release",
                "reconciled": "2026-08-03",
            }
        )
    ).adopted_from
    assert provenance is not None
    monkeypatch.setattr(
        manifest_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=""),
    )
    with pytest.raises(ManifestError, match=message):
        validate_adopted_from_reference(provenance)


def test_remote_upstream_provenance_timeout_is_actionable(monkeypatch) -> None:
    provenance = EnvironmentManifest.model_validate(
        _environment_data(
            adopted_from={
                "source": "https://example.invalid/org/repo.git",
                "ref": "main",
                "reconciled": "2026-08-03",
            }
        )
    ).adopted_from
    assert provenance is not None

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=5)

    monkeypatch.setattr(manifest_module.subprocess, "run", timeout)
    with pytest.raises(ManifestError, match="Timed out"):
        validate_adopted_from_reference(provenance)


def test_licensing_uses_installed_spdx_registry_with_syntax_fallback(monkeypatch) -> None:
    if manifest_module._canonicalize_license_expression is not None:
        manifest = EnvironmentManifest.model_validate(_environment_data(licensing="apache-2.0"))
        assert manifest.licensing == "Apache-2.0"
        with pytest.raises(ValidationError, match="unknown SPDX"):
            EnvironmentManifest.model_validate(_environment_data(licensing="Definitely-Not-A-License"))

    monkeypatch.setattr(manifest_module, "_canonicalize_license_expression", None)
    assert EnvironmentManifest.model_validate(_environment_data(licensing="LicenseRef-Company-Eval")).licensing == (
        "LicenseRef-Company-Eval"
    )
    with pytest.raises(ValidationError, match="one SPDX license identifier"):
        EnvironmentManifest.model_validate(_environment_data(licensing="not a license"))


def test_grading_mode_is_an_optional_composition_mirror() -> None:
    manifest = EnvironmentManifest.model_validate(_environment_data(grading_mode="lenient_answer_colon_md"))
    assert manifest.grading_mode == "lenient_answer_colon_md"


def test_grading_mode_constraints_are_derived_from_server_source_without_importing(tmp_path: Path) -> None:
    resources_root = tmp_path / "resources_servers"
    source = resources_root / "side_effect" / "app.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "from typing import Literal\n"
        "raise RuntimeError('must not execute')\n"
        "class Config:\n"
        "    grading_mode: Literal['strict', 'lenient'] = 'strict'\n"
    )

    assert grading_modes_from_source(source, resources_root=resources_root) == ("lenient", "strict")


def test_builtin_and_inherited_grading_mode_constraints_are_discoverable_without_schema_binding() -> None:
    constraints = resources_server_grading_modes()
    assert constraints["mcqa"] == (
        "lenient_answer_colon",
        "lenient_answer_colon_md",
        "lenient_boxed",
        "strict_single_letter_boxed",
    )
    assert constraints["gpqa_diamond"] == constraints["mcqa"]

    assert EnvironmentManifest.model_validate(_environment_data(grading_mode="binary")).grading_mode == "binary"


def test_load_dump_round_trip_and_actionable_errors(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(_benchmark_data(), sort_keys=False), encoding="utf-8")

    loaded = load_manifest(path)
    rendered = dump_manifest(loaded)
    output = tmp_path / "round-trip.yaml"
    assert dump_manifest(loaded, output) == rendered
    assert load_manifest(output) == loaded

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("kind: [\n", encoding="utf-8")
    with pytest.raises(ManifestError, match=r"invalid\.yaml.*line"):
        load_manifest(invalid)
    with pytest.raises(ManifestError, match="was not found"):
        load_manifest(tmp_path / "missing.yaml")


def test_dump_manifest_refuses_symlink_destination(tmp_path: Path) -> None:
    manifest = EnvironmentManifest.model_validate(_environment_data())
    target = tmp_path / "outside.yaml"
    target.write_text("untouched\n")
    destination = tmp_path / "manifest.yaml"
    destination.symlink_to(target)

    with pytest.raises(ManifestError, match="symbolic-link destination"):
        dump_manifest(manifest, destination)

    assert target.read_text() == "untouched\n"


def test_manifest_errors_are_config_errors_and_include_field_locations(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(_environment_data(version="not-semver")), encoding="utf-8")

    with pytest.raises(ConfigError, match=r"(?s)manifest\.yaml.*version"):
        load_manifest(path)


def test_checked_in_json_schema_is_synced_and_uses_domain_enum_from_code() -> None:
    generated = manifest_json_schema()
    checked_in = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert checked_in == generated
    assert generated["$defs"]["Domain"]["enum"] == [domain.value for domain in Domain]
    assert generated["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    constraints = {
        condition["if"]["properties"]["resources_server"]["const"]: tuple(
            value for value in condition["then"]["properties"]["grading_mode"]["enum"] if value is not None
        )
        for condition in generated["allOf"]
        if condition.get("if", {}).get("properties", {}).get("resources_server")
    }
    assert constraints == resources_server_grading_modes()
