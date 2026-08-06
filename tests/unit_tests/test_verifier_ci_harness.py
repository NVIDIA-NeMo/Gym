# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import pytest

from nemo_gym.environment_manifest import load_manifest
from nemo_gym.environment_scaffold import scaffold_environment
from nemo_gym.verifier_ci_harness import (
    OFFLINE_VERIFIER_APP_FACTORY,
    VerifierCIHarnessError,
    build_verifier_harness_invocation,
    exercise_selected_verifier,
    select_sole_resources_server_runtime,
)
from nemo_gym.verifier_fixture import (
    DETERMINISM_ENV_VAR,
    HIGHER_IS_BETTER_ENV_VAR,
    REWARD_RANGE_ENV_VAR,
    UPDATE_EXPECTED_ENV_VAR,
)
from tests.unit_tests.onboarding_test_utils import complete_scaffold_fixture


def test_invocation_builder_is_shared_shell_free_contract(tmp_path: Path) -> None:
    component = tmp_path / "resources_servers" / "fixture"
    fixture = component / "tests" / "verifier_cases.jsonl"
    invocation = build_verifier_harness_invocation(
        python_executable="/component/.venv/bin/python",
        project_root=tmp_path,
        component_dir=component,
        entrypoint="app.py",
        instance_name="fixture_resources_server",
        fixture_path=fixture,
        server_config={"entrypoint": "app.py", "option": 3},
        reward_range=(-1, 2),
        higher_is_better=False,
        determinism="seeded",
        base_environment={"PYTHONPATH": "/existing", "SAFE": "1"},
    )

    assert invocation.command == (
        "/component/.venv/bin/python",
        "-m",
        "nemo_gym.verifier_ci_harness",
        "--component-dir",
        str(component),
        "--entrypoint",
        "app.py",
        "--instance-name",
        "fixture_resources_server",
        "--fixture",
        str(fixture),
    )
    assert json.loads(invocation.stdin) == {"entrypoint": "app.py", "option": 3}
    assert invocation.environment["PYTHONPATH"] == os.pathsep.join((str(tmp_path), "/existing"))
    assert invocation.environment[UPDATE_EXPECTED_ENV_VAR] == "0"
    assert invocation.environment[REWARD_RANGE_ENV_VAR] == "[-1,2]"
    assert invocation.environment[HIGHER_IS_BETTER_ENV_VAR] == "false"
    assert invocation.environment[DETERMINISM_ENV_VAR] == "seeded"


def test_invocation_builder_scrubs_optional_manifest_contract_for_legacy_fixture(tmp_path: Path) -> None:
    component = tmp_path / "resources_servers" / "legacy"
    invocation = build_verifier_harness_invocation(
        python_executable="/component/.venv/bin/python",
        project_root=tmp_path,
        component_dir=component,
        entrypoint="app.py",
        instance_name="legacy_resources_server",
        fixture_path=component / "tests" / "verifier_cases.jsonl",
        server_config={"entrypoint": "app.py"},
        base_environment={
            REWARD_RANGE_ENV_VAR: "[-10,10]",
            HIGHER_IS_BETTER_ENV_VAR: "false",
            DETERMINISM_ENV_VAR: "seeded",
        },
    )

    assert invocation.environment[UPDATE_EXPECTED_ENV_VAR] == "0"
    assert REWARD_RANGE_ENV_VAR not in invocation.environment
    assert HIGHER_IS_BETTER_ENV_VAR not in invocation.environment
    assert DETERMINISM_ENV_VAR not in invocation.environment


def test_legacy_runtime_selector_requires_exactly_one_resources_server() -> None:
    single = {
        "scorer_instance": {
            "resources_servers": {
                "scorer": {"entrypoint": "app.py", "grading_mode": "strict"},
            }
        },
        "not_a_server": {"value": 1},
    }

    assert select_sole_resources_server_runtime(single) == (
        "scorer",
        "scorer_instance",
        {"entrypoint": "app.py", "grading_mode": "strict"},
    )
    with pytest.raises(VerifierCIHarnessError, match="exactly one Resources Server.*none"):
        select_sole_resources_server_runtime({})
    with pytest.raises(VerifierCIHarnessError, match="scorer_instance/scorer, second/other"):
        select_sole_resources_server_runtime(
            {
                **single,
                "second": {"resources_servers": {"other": {"entrypoint": "app.py"}}},
            }
        )


def test_harness_uses_generated_named_app_factory(tmp_path: Path) -> None:
    scaffold_environment(kind="environment", name="explicit_factory", root=tmp_path)
    component = tmp_path / "resources_servers" / "explicit_factory"
    manifest = load_manifest(tmp_path / "environments" / "explicit_factory" / "manifest.yaml")
    complete_scaffold_fixture(component / "tests" / "verifier_cases.jsonl", manifest)

    exercise_selected_verifier(
        component_dir=component,
        entrypoint="app.py",
        instance_name="explicit_factory_resources_server",
        server_config={"entrypoint": "app.py"},
        fixture_path=component / "tests" / "verifier_cases.jsonl",
        reward_range=(0, 1),
        higher_is_better=True,
        determinism="unknown",
    )

    assert f"def {OFFLINE_VERIFIER_APP_FACTORY}(" in (component / "app.py").read_text()


def test_harness_exercises_legacy_fixture_without_manifest_contract(tmp_path: Path) -> None:
    scaffold_environment(kind="environment", name="legacy_fixture", root=tmp_path)
    component = tmp_path / "resources_servers" / "legacy_fixture"
    manifest = load_manifest(tmp_path / "environments" / "legacy_fixture" / "manifest.yaml")
    complete_scaffold_fixture(component / "tests" / "verifier_cases.jsonl", manifest)

    exercise_selected_verifier(
        component_dir=component,
        entrypoint="app.py",
        instance_name="legacy_fixture_resources_server",
        server_config={"entrypoint": "app.py"},
        fixture_path=component / "tests" / "verifier_cases.jsonl",
    )


def test_checked_in_mcqa_verifier_exposes_named_app_factory() -> None:
    component = Path(__file__).parents[2] / "resources_servers" / "mcqa"

    exercise_selected_verifier(
        component_dir=component,
        entrypoint="app.py",
        instance_name="mcqa",
        server_config={"entrypoint": "app.py", "grading_mode": None},
        fixture_path=component / "tests" / "verifier_cases.jsonl",
        reward_range=(0, 1),
        higher_is_better=True,
        determinism="unknown",
    )


def test_harness_rejects_entrypoint_without_named_app_factory(tmp_path: Path) -> None:
    scaffold_environment(kind="environment", name="missing_factory", root=tmp_path)
    component = tmp_path / "resources_servers" / "missing_factory"
    entrypoint = component / "app.py"
    entrypoint.write_text(
        entrypoint.read_text().replace(
            f"def {OFFLINE_VERIFIER_APP_FACTORY}(",
            "def renamed_offline_verifier_app(",
        )
    )

    with pytest.raises(VerifierCIHarnessError, match=OFFLINE_VERIFIER_APP_FACTORY):
        exercise_selected_verifier(
            component_dir=component,
            entrypoint="app.py",
            instance_name="missing_factory_resources_server",
            server_config={"entrypoint": "app.py"},
            fixture_path=component / "tests" / "verifier_cases.jsonl",
            reward_range=(0, 1),
            higher_is_better=True,
            determinism="unknown",
        )
