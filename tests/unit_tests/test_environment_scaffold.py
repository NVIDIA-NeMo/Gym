# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from nemo_gym.environment.manifest import EnvironmentKind, IntegrationProfile, load_manifest
from nemo_gym.environment.scaffold import (
    ScaffoldConflictError,
    ScaffoldError,
    scaffold_environment,
    scaffold_resources_server,
)
from nemo_gym.environment.validation import validate_environment
from nemo_gym.verifier_fixture import exercise_verifier_fixture


@pytest.mark.parametrize("kind", list(EnvironmentKind))
@pytest.mark.parametrize("profile", list(IntegrationProfile))
def test_scaffolds_and_validates_every_kind_and_profile(
    tmp_path: Path, kind: EnvironmentKind, profile: IntegrationProfile
) -> None:
    result = scaffold_environment(root=tmp_path, kind=kind, name="sample", profile=profile)
    parent = "benchmarks" if kind == EnvironmentKind.BENCHMARK else "environments"
    asset = tmp_path / parent / "sample"
    manifest = load_manifest(asset / "manifest.yaml")

    assert result.asset_dir == asset
    assert result.created and not result.existing
    assert manifest.kind == kind
    assert manifest.integration_profile == profile
    assert manifest.determinism.value == "unknown"
    report = validate_environment(asset / "manifest.yaml")
    if profile in {IntegrationProfile.CUSTOM_GYM_AGENT_LOOP, IntegrationProfile.EXTERNAL_AGENT_LOOP}:
        assert report.inferred_profile == "unknown"
        assert report.warnings
    else:
        assert report.inferred_profile == profile.value

    for path in tmp_path.rglob("*.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

    benchmark_files = [asset / "prepare.py", asset / "prompt.yaml", asset / "data/source.jsonl"]
    assert all(path.is_file() for path in benchmark_files) is (kind == EnvironmentKind.BENCHMARK)

    generated_agent = profile in {
        IntegrationProfile.CUSTOM_GYM_AGENT_LOOP,
        IntegrationProfile.EXTERNAL_AGENT_LOOP,
    }
    assert (tmp_path / "responses_api_agents/sample_agent").is_dir() is generated_agent
    if generated_agent:
        agent_source = tmp_path.joinpath("responses_api_agents/sample_agent/app.py").read_text(encoding="utf-8")
        extensions = ("responses", "run") if profile == IntegrationProfile.EXTERNAL_AGENT_LOOP else ("responses",)
        assert all(f"async def {extension}(" in agent_source for extension in extensions)
        assert all(f"super().{extension}(" in agent_source for extension in extensions)
    assert (asset / "rollout_driver.py").is_file() is (profile == IntegrationProfile.EXTERNAL_ROLLOUT_DRIVER)


def test_completed_external_agent_retains_its_profile(tmp_path: Path) -> None:
    asset = scaffold_environment(
        root=tmp_path,
        kind="environment",
        name="external",
        profile="external-agent-loop",
    ).asset_dir
    app_path = tmp_path / "responses_api_agents/external_agent/app.py"
    source = app_path.read_text(encoding="utf-8")
    source = source.replace(
        "return await super().responses(request, response, body)",
        "raise NotImplementedError",
    ).replace(
        "return await super().run(request, body)",
        "return await self.external_framework.run(body)",
    )
    app_path.write_text(source, encoding="utf-8")

    report = validate_environment(asset / "manifest.yaml")

    assert report.inferred_profile == "external-agent-loop"
    assert report.profile_evidence == "agent responses() raises NotImplementedError"


def test_generated_benchmark_prepare_writes_domain_rows(tmp_path: Path) -> None:
    asset = scaffold_environment(root=tmp_path, kind="benchmark", name="science").asset_dir
    prepare = runpy.run_path(str(asset / "prepare.py"))["prepare"]
    output = tmp_path / "prepared.jsonl"

    assert prepare(asset / "data/source.jsonl", output) == output
    assert output.read_text(encoding="utf-8") == '{"question": "What is 6 x 7?", "expected_answer": "42"}\n'


async def test_generated_scorer_fixture_runs_in_process(tmp_path: Path) -> None:
    scaffold_environment(root=tmp_path, kind="environment", name="scored")
    app = runpy.run_path(str(tmp_path / "resources_servers/scored/app.py"))

    results = await exercise_verifier_fixture(app["VERIFIER_FIXTURE"], reward_range=(0, 1), determinism="unknown")

    assert [result.kind for result in results] == ["full_reward", "zero_reward", "malformed"]


def test_standalone_resources_server_keeps_its_combined_composition(tmp_path: Path) -> None:
    server = tmp_path / "resources_servers/shared"
    result = scaffold_resources_server(directory=server)

    assert result.asset_dir == server
    assert {path.relative_to(server).as_posix() for path in result.created} == {
        "README.md",
        "__init__.py",
        "app.py",
        "configs/shared.yaml",
        "data/.gitignore",
        "example.jsonl",
        "requirements.txt",
        "tests/__init__.py",
        "tests/test_app.py",
        "tests/verifier_cases.jsonl",
    }
    config = yaml.safe_load((server / "configs/shared.yaml").read_text(encoding="utf-8"))
    assert config["shared_resources_server"]["resources_servers"]["shared"]["verified"] is False
    # Datasets are declared on the resources server (dataset-decoupling); the agent block carries none.
    datasets = config["shared_resources_server"]["resources_servers"]["shared"]["datasets"]
    assert [dataset["type"] for dataset in datasets] == ["train", "validation", "example"]
    assert "datasets" not in config["shared_simple_agent"]["responses_api_agents"]["simple_agent"]
    assert not (tmp_path / "environments").exists()
    assert not (tmp_path / "benchmarks").exists()


@pytest.mark.parametrize("name", ["a-b", "1sample", "class", "Uppercase"])
def test_standalone_resources_server_rejects_invalid_python_names(tmp_path: Path, name: str) -> None:
    server = tmp_path / "resources_servers" / name

    with pytest.raises(ScaffoldError, match="Python identifier|lowercase"):
        scaffold_resources_server(directory=server)

    assert not server.exists()


def test_standalone_and_manifest_scaffolds_share_resource_component_files(tmp_path: Path) -> None:
    standalone_root = tmp_path / "standalone"
    manifest_root = tmp_path / "manifest"
    standalone = standalone_root / "resources_servers/shared"
    manifest_resource = manifest_root / "resources_servers/shared"

    scaffold_resources_server(directory=standalone)
    scaffold_environment(root=manifest_root, kind="environment", name="shared")

    for relative in (
        "__init__.py",
        "app.py",
        "tests/__init__.py",
        "tests/test_app.py",
        "tests/verifier_cases.jsonl",
    ):
        assert standalone.joinpath(relative).read_bytes() == manifest_resource.joinpath(relative).read_bytes()


async def test_standalone_scorer_fixture_runs_in_process(tmp_path: Path) -> None:
    server = tmp_path / "resources_servers/standalone"
    scaffold_resources_server(directory=server)
    app = runpy.run_path(str(server / "app.py"))

    results = await exercise_verifier_fixture(app["VERIFIER_FIXTURE"], reward_range=(0, 1), determinism="unknown")

    assert [result.kind for result in results] == ["full_reward", "zero_reward", "malformed"]


def test_generated_python_passes_repo_lint_and_format(tmp_path: Path) -> None:
    for index, profile in enumerate(IntegrationProfile):
        scaffold_environment(root=tmp_path, kind="benchmark", name=f"lint_{index}", profile=profile)
    scaffold_resources_server(directory=tmp_path / "resources_servers/standalone_lint")
    python_files = [str(path) for path in tmp_path.rglob("*.py")]
    config = Path(__file__).parents[2] / "pyproject.toml"

    for command in ("check", "format"):
        arguments = [sys.executable, "-m", "ruff", command, "--no-cache", "--config", str(config)]
        if command == "format":
            arguments.append("--check")
        completed = subprocess.run([*arguments, *python_files], capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_identical_rerun_is_a_noop(tmp_path: Path) -> None:
    first = scaffold_environment(root=tmp_path, kind="environment", name="repeatable")
    second = scaffold_environment(root=tmp_path, kind="environment", name="repeatable")

    assert not second.created
    assert set(second.existing) == set(first.created)


def test_conflict_aborts_the_complete_write_set(tmp_path: Path) -> None:
    manifest = tmp_path / "environments/occupied/manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("user content\n", encoding="utf-8")

    with pytest.raises(ScaffoldConflictError):
        scaffold_environment(root=tmp_path, kind="environment", name="occupied")

    assert manifest.read_text(encoding="utf-8") == "user content\n"
    assert list(manifest.parent.iterdir()) == [manifest]
    assert not (tmp_path / "resources_servers/occupied").exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "unknown", "name": "sample"},
        {"kind": "environment", "name": "sample", "profile": "unknown"},
        {"kind": "environment", "name": "../escape"},
        {"kind": "environment", "name": "a-b"},
        {"kind": "environment", "name": "1sample"},
        {"kind": "environment", "name": "class"},
        {"kind": "environment", "name": "sample", "reuse_verifier": "../shared"},
    ],
)
def test_rejects_unsafe_or_unknown_requests_without_writes(tmp_path: Path, kwargs: dict[str, str]) -> None:
    with pytest.raises(ScaffoldError):
        scaffold_environment(root=tmp_path, **kwargs)

    assert not (tmp_path / "environments").exists()
    assert not (tmp_path / "benchmarks").exists()


def test_rejects_unsafe_root_paths(tmp_path: Path) -> None:
    root_file = tmp_path / "root-file"
    root_file.write_text("occupied\n", encoding="utf-8")
    with pytest.raises(ScaffoldError, match="directory"):
        scaffold_environment(root=root_file, kind="environment", name="sample")

    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ScaffoldError, match="symlink"):
        scaffold_environment(root=linked_root, kind="environment", name="sample")

    root = tmp_path / "root"
    root.mkdir()
    (root / "environments").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ScaffoldError, match="symlink"):
        scaffold_environment(root=root, kind="environment", name="sample")


_FIXTURE_EXPORT = """\
from pathlib import Path

from pydantic import BaseModel

from nemo_gym.verifier_fixture import VerifierFixture


class Request(BaseModel):
    value: str


class Server:
    pass


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=Server,
    request_model=Request,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)
"""


def _synthetic_verifier(root: Path, *, fixture_source: str = _FIXTURE_EXPORT, extra_config: str = "") -> None:
    server = root / "resources_servers/shared"
    (server / "configs").mkdir(parents=True)
    (server / "app.py").write_text(fixture_source, encoding="utf-8")
    (server / "configs/shared.yaml").write_text(
        "shared_resources:\n  resources_servers:\n    shared:\n      entrypoint: app.py\n      domain: other\n"
        + extra_config,
        encoding="utf-8",
    )


def test_reuses_only_a_resources_server_with_a_fixture_contract(tmp_path: Path) -> None:
    _synthetic_verifier(tmp_path)

    result = scaffold_environment(
        root=tmp_path,
        kind="benchmark",
        name="reused",
        profile="custom-gym-verifier",
        reuse_verifier="shared",
        reward_range=(0, 1),
        higher_is_better=True,
    )
    manifest = load_manifest(result.asset_dir / "manifest.yaml")

    assert manifest.resources_server == "shared"
    assert not (tmp_path / "resources_servers/reused").exists()
    validate_environment(result.asset_dir / "manifest.yaml")


@pytest.mark.parametrize("fixture_source", ["OTHER_EXPORT = object()\n", "VERIFIER_FIXTURE = object()\n"])
def test_reuse_rejects_a_resources_server_without_a_fixture_declaration(tmp_path: Path, fixture_source: str) -> None:
    _synthetic_verifier(tmp_path, fixture_source=fixture_source)

    with pytest.raises(ScaffoldError, match="VERIFIER_FIXTURE"):
        scaffold_environment(
            root=tmp_path,
            kind="environment",
            name="reused",
            reuse_verifier="shared",
            reward_range=(0, 1),
            higher_is_better=True,
        )

    assert not (tmp_path / "environments/reused").exists()


def test_reuse_requires_an_explicit_reward_contract(tmp_path: Path) -> None:
    _synthetic_verifier(tmp_path)

    with pytest.raises(ScaffoldError, match="requires reward_range and higher_is_better"):
        scaffold_environment(root=tmp_path, kind="environment", name="reused", reuse_verifier="shared")


def test_reward_overrides_require_reuse(tmp_path: Path) -> None:
    with pytest.raises(ScaffoldError, match="only accepted with reuse_verifier"):
        scaffold_environment(
            root=tmp_path,
            kind="environment",
            name="sample",
            reward_range=(0, 1),
            higher_is_better=True,
        )


def test_reuse_rejects_an_invalid_reward_contract(tmp_path: Path) -> None:
    _synthetic_verifier(tmp_path)

    with pytest.raises(ScaffoldError, match="invalid reward contract"):
        scaffold_environment(
            root=tmp_path,
            kind="environment",
            name="reused",
            reuse_verifier="shared",
            reward_range=(1, 0),
            higher_is_better=True,
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ("not-a-mapping\n", "not a mapping"),
        (
            "shared_resources:\n  resources_servers:\n    shared: []\n",
            "config for 'shared' is not a mapping",
        ),
        (
            "shared_resources:\n  resources_servers:\n    other:\n      entrypoint: app.py\n",
            "exactly one resources-server",
        ),
        (
            "shared_resources:\n  resources_servers:\n    shared:\n      domain: other\n",
            "does not define a resources-server entrypoint",
        ),
        (
            "shared_resources:\n  resources_servers:\n    shared:\n      entrypoint: app.py\n"
            "custom_agent:\n  responses_api_agents:\n    custom:\n      entrypoint: app.py\n",
            "may bundle only one simple_agent",
        ),
        (
            "shared_resources:\n  resources_servers:\n    shared:\n      entrypoint: app.py\n"
            "shared_agent:\n  responses_api_agents:\n    simple_agent:\n"
            "      entrypoint: app.py\n      resources_server: {name: wrong}\n",
            "must reference resources instance",
        ),
    ],
)
def test_reuse_rejects_malformed_component_configs(tmp_path: Path, config: str, message: str) -> None:
    _synthetic_verifier(tmp_path)
    config_path = tmp_path / "resources_servers/shared/configs/shared.yaml"
    config_path.write_text(config, encoding="utf-8")

    with pytest.raises(ScaffoldError, match=message):
        scaffold_environment(
            root=tmp_path,
            kind="environment",
            name="reused",
            reuse_verifier="shared",
            reward_range=(0, 1),
            higher_is_better=True,
        )


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ("missing_config", "was not found"),
        ("invalid_yaml", "could not read reused verifier config"),
        ("missing_entrypoint", "entrypoint was not found"),
        ("invalid_entrypoint", "could not inspect reused verifier entrypoint"),
    ],
)
def test_reuse_reports_unreadable_inputs(tmp_path: Path, setup: str, message: str) -> None:
    if setup != "missing_config":
        _synthetic_verifier(tmp_path)
        config_path = tmp_path / "resources_servers/shared/configs/shared.yaml"
        app_path = tmp_path / "resources_servers/shared/app.py"
        if setup == "invalid_yaml":
            config_path.write_text("[", encoding="utf-8")
        elif setup == "missing_entrypoint":
            app_path.unlink()
        else:
            app_path.write_text("def broken(:\n", encoding="utf-8")

    with pytest.raises(ScaffoldError, match=message):
        scaffold_environment(
            root=tmp_path,
            kind="environment",
            name="reused",
            reuse_verifier="shared",
            reward_range=(0, 1),
            higher_is_better=True,
        )


def test_reuse_supports_a_bundled_default_agent(tmp_path: Path) -> None:
    _synthetic_verifier(
        tmp_path,
        extra_config="""
shared_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server: {type: resources_servers, name: shared_resources}
      model_server: {type: responses_api_models, name: policy_model}
      datasets: []
""",
    )

    result = scaffold_environment(
        root=tmp_path,
        kind="environment",
        name="reused",
        reuse_verifier="shared",
        reward_range=(-1, 1),
        higher_is_better=True,
    )

    assert "_inherit_from: shared_agent" in result.asset_dir.joinpath("config.yaml").read_text(encoding="utf-8")
    assert load_manifest(result.asset_dir / "manifest.yaml").reward.range == (-1, 1)
    validate_environment(result.asset_dir / "manifest.yaml")


def test_reuse_supports_every_profile(tmp_path: Path) -> None:
    _synthetic_verifier(tmp_path)

    for profile in IntegrationProfile:
        name = profile.name.lower()
        result = scaffold_environment(
            root=tmp_path,
            kind="environment",
            name=name,
            profile=profile,
            reuse_verifier="shared",
            reward_range=(0, 1),
            higher_is_better=True,
        )
        manifest = load_manifest(result.asset_dir / "manifest.yaml")
        assert manifest.resources_server == "shared"
        assert manifest.integration_profile == profile


def test_reuse_rejects_a_bundled_model_server(tmp_path: Path) -> None:
    _synthetic_verifier(
        tmp_path,
        extra_config="""
shared_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
""",
    )

    with pytest.raises(ScaffoldError, match="may not bundle a model server"):
        scaffold_environment(
            root=tmp_path,
            kind="environment",
            name="reused",
            reuse_verifier="shared",
            reward_range=(0, 1),
            higher_is_better=True,
        )
