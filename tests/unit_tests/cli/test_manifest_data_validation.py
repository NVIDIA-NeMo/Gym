# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path

import pytest
import yaml
from omegaconf import DictConfig

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.cli import env
from nemo_gym.environment.onboarding import (
    EnvironmentOnboardingError,
    VerifierReport,
    exercise_verifier_run,
    prepare_verifier_run,
)
from nemo_gym.environment.registry import EnvironmentCatalogEntry
from nemo_gym.environment.scaffold import scaffold_environment
from nemo_gym.environment.validation import EnvironmentValidationError, EnvironmentValidationReport


@pytest.fixture
def manifest_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> env.TestConfig:
    monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    scaffold_environment(root=tmp_path, kind="environment", name="sample")

    # Run the real exported fixture in this process; subprocess setup belongs to its own runner tests.
    def run_fixture(
        entry: EnvironmentCatalogEntry,
        *,
        update_expected: bool,
        validation: EnvironmentValidationReport,
        setup_config: DictConfig | None = None,
    ) -> VerifierReport:
        return asyncio.run(
            exercise_verifier_run(prepare_verifier_run(entry, validation), update_expected=update_expected)
        )

    monkeypatch.setattr(env, "_run_manifest_verifier", run_fixture)
    return env.TestConfig(entrypoint="resources_servers/sample", should_validate_data=True)


def test_manifest_inputs_and_fixture_replace_legacy_bundle(manifest_server: env.TestConfig, capsys) -> None:
    assert not (manifest_server.resolved_dir_path / "data").exists()

    env._validate_data_single(manifest_server)

    assert "Manifest inputs and verifier fixtures" in capsys.readouterr().out


@pytest.mark.parametrize("broken_fixture", ["missing_export", "wrong_reward"])
def test_manifest_does_not_bypass_a_broken_fixture(manifest_server: env.TestConfig, broken_fixture: str) -> None:
    server = manifest_server.resolved_dir_path
    if broken_fixture == "missing_export":
        app = server / "app.py"
        app.write_text(app.read_text().replace("VERIFIER_FIXTURE =", "_unused_fixture ="))
        error = "must export VERIFIER_FIXTURE"
    else:
        cases_path = server / "tests/verifier_cases.jsonl"
        cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
        cases[0]["expected_reward"] = 0.25
        cases_path.write_text("".join(f"{json.dumps(case)}\n" for case in cases))
        error = "reward mismatch"
    with pytest.raises(EnvironmentOnboardingError, match=error):
        env._validate_data_single(manifest_server)


def test_manifest_does_not_bypass_missing_inputs(manifest_server: env.TestConfig, tmp_path: Path) -> None:
    (tmp_path / "environments/sample/data/example.jsonl").unlink()
    with pytest.raises(EnvironmentValidationError, match="example.jsonl"):
        env._validate_data_single(manifest_server)


def test_manifest_does_not_bypass_stale_composition(manifest_server: env.TestConfig, tmp_path: Path) -> None:
    manifest_path = tmp_path / "environments/sample/manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["agent_server"] = "different_agent"
    manifest_path.write_text(yaml.safe_dump(manifest))
    with pytest.raises(EnvironmentValidationError, match="composition is stale"):
        env._validate_data_single(manifest_server)


def test_existing_legacy_bundle_keeps_its_five_row_check(manifest_server: env.TestConfig) -> None:
    data = manifest_server.resolved_dir_path / "data"
    data.mkdir()
    (data / "example.jsonl").write_text("{}\n")
    with pytest.raises(AssertionError, match="Expected 5 examples"):
        env._validate_data_single(manifest_server)


def test_standalone_server_still_requires_legacy_bundle(manifest_server: env.TestConfig, tmp_path: Path) -> None:
    (tmp_path / "environments/sample/manifest.yaml").unlink()
    with pytest.raises(AssertionError, match="containing 5 examples is required"):
        env._validate_data_single(manifest_server)
