# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the native recipe's model-free commands, without provisioning services."""

import asyncio
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, PARENT_DIR
from nemo_gym.environment.onboarding import verify_environment
from nemo_gym.environment.publication import finalize_publication
from nemo_gym.environment.validation import validate_environment
from nemo_gym.registry import discover_environment_catalog


RECIPE = PARENT_DIR / "fern/versions/latest/pages/get-started/native-onboarding.mdx"


def _step(name: str) -> str:
    assert RECIPE.is_file(), "The native onboarding recipe must exist."
    blocks = re.findall(r'```bash title="' + re.escape(name) + r'"\n(.*?)\n```', RECIPE.read_text(), re.S)
    assert len(blocks) == 1, f"Expected exactly one executable recipe block named {name}."
    return blocks[0]


def _run_step(
    name: str, cwd: Path, *, expected_code: int = 0, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        PATH=f"{Path(sys.executable).parent}:{env['PATH']}",
        PYTHONPATH=str(PARENT_DIR),
        NEMO_GYM_EXTRA_ROOTS=str(cwd),
    )
    env.update(extra_env or {})
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _step(name)],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    return result


def test_native_recipe_scaffold_validate_and_verifier_contract(tmp_path: Path, monkeypatch) -> None:
    """Use the documented commands and real fixture; child provisioning is separate acceptance."""
    _run_step("native-scaffold", tmp_path)
    _run_step("native-metadata", tmp_path)
    validation_result = _run_step("native-validate", tmp_path)
    assert "native_exact_answer" in validation_result.stdout

    monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(tmp_path))
    entry = next(entry for entry in discover_environment_catalog() if entry.name == "native_exact_answer")
    validation = validate_environment(entry.manifest_path, config_path=entry.config_path)
    fixture = asyncio.run(verify_environment(entry))
    assert [case.kind for case in fixture.cases] == ["full_reward", "zero_reward", "malformed"]
    assert [case.observed_rewards for case in fixture.cases[:2]] == [(1.0,), (0.0,)]
    publication = finalize_publication(entry, validation, fixture)
    assert publication.status == "experimental"
    assert publication.verifier_cases == 3

    manifest = yaml.safe_load(Path(entry.manifest_path).read_text())
    assert manifest["canonical_split"] == "example"
    assert manifest["datasets"][0]["type"] == "benchmark"


def test_native_recipe_strict_preflight_rejects_missing_endpoint(tmp_path: Path) -> None:
    _run_step("native-scaffold", tmp_path)
    result = _run_step("native-missing-model", tmp_path, expected_code=1)
    assert "policy_base_url" in result.stdout + result.stderr
    assert "not set" in result.stdout + result.stderr


def test_native_recipe_strict_preflight_accepts_complete_config_without_model_call(tmp_path: Path) -> None:
    _run_step("native-scaffold", tmp_path)
    config = tmp_path / "fixture-model.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "policy_base_url": "https://model.invalid/v1",
                "policy_model_name": "offline-fixture",
                "policy_api_key": "not-a-real-key",
            }
        )
    )
    result = _run_step("native-model-preflight", tmp_path, extra_env={"GYM_MODEL_CONFIG": str(config)})
    assert "Config is valid" in result.stdout
