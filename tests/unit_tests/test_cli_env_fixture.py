# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import nemo_gym.cli.env as cli_env
from nemo_gym.config_types import ConfigError
from nemo_gym.environment_manifest import EnvironmentManifest, dump_manifest, load_manifest
from nemo_gym.environment_scaffold import scaffold_environment
from nemo_gym.verifier_ci_harness import VERIFIER_HARNESS_MODULE, exercise_selected_verifier
from tests.unit_tests.onboarding_test_utils import complete_scaffold_fixture


def _raw_for_scaffold(root: Path, name: str):
    return OmegaConf.create(
        {
            "config_paths": [
                str(root / "environments" / name / "config.yaml"),
                str(root / "resources_servers" / name / "configs" / f"{name}.yaml"),
            ]
        }
    )


def test_named_environment_binds_its_resources_server_and_fixture(tmp_path: Path) -> None:
    scaffold_environment(
        kind="environment",
        name="fixture_env",
        root=tmp_path,
        metadata={"reward_range": [-1, 2], "determinism": "seeded"},
    )
    manifest_path = tmp_path / "environments" / "fixture_env" / "manifest.yaml"
    config = cli_env.TestConfig(manifest_path=manifest_path, environment_ref="fixture_env")

    manifest = cli_env._bind_manifest_resources_server(config, _raw_for_scaffold(tmp_path, "fixture_env"))

    assert manifest is not None
    assert config.entrypoint == "resources_servers/fixture_env"
    assert config.resolved_dir_path == tmp_path / "resources_servers" / "fixture_env"
    complete_scaffold_fixture(config.fixture_path, manifest)
    assert cli_env._validate_scoring_fixture(config, manifest, require_expected_values=True)


def test_manifest_reward_range_is_checked_before_scorer_execution(tmp_path: Path) -> None:
    scaffold_environment(kind="environment", name="fixture_env", root=tmp_path)
    manifest_path = tmp_path / "environments" / "fixture_env" / "manifest.yaml"
    config = cli_env.TestConfig(manifest_path=manifest_path, environment_ref="fixture_env")
    manifest = cli_env._bind_manifest_resources_server(config, _raw_for_scaffold(tmp_path, "fixture_env"))
    cases_path = config.fixture_path
    assert manifest is not None
    complete_scaffold_fixture(cases_path, manifest)
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    cases[0]["expected_reward"] = 4
    cases_path.write_text("".join(json.dumps(case) + "\n" for case in cases))

    with pytest.raises(ConfigError, match="reward.range.*full-reward case"):
        cli_env._validate_scoring_fixture(config, manifest, require_expected_values=True)


def test_pytest_receives_manifest_contract_and_update_signal(tmp_path: Path, monkeypatch) -> None:
    scaffold_environment(
        kind="environment",
        name="fixture_env",
        root=tmp_path,
        metadata={"reward_range": [-1, 2], "determinism": "seeded"},
    )
    manifest = load_manifest(tmp_path / "environments" / "fixture_env" / "manifest.yaml")
    config = cli_env.TestConfig(entrypoint="resources_servers/fixture_env", update_expected=True)
    config._resolved_dir_override = tmp_path / "resources_servers" / "fixture_env"
    captured = {}

    monkeypatch.setattr(cli_env, "setup_env_command", lambda *args: "setup")

    def fake_run(command, working_dir, *, project_root):
        captured.update(command=command, working_dir=working_dir, project_root=project_root)
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(cli_env, "run_command", fake_run)

    cli_env._test_single(config, OmegaConf.create({}), manifest)

    assert "NEMO_GYM_UPDATE_EXPECTED=1" in captured["command"]
    assert "NEMO_GYM_FIXTURE_REWARD_RANGE='[-1.0,2.0]'" in captured["command"]
    assert "NEMO_GYM_FIXTURE_DETERMINISM=seeded" in captured["command"]
    assert captured["command"].endswith("pytest")


def test_manifest_preflight_runs_before_component_test_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_gym.environment_execution as environment_execution

    raw = OmegaConf.create({"entrypoint": "resources_servers/fixture"})
    events = []

    class PreflightReached(RuntimeError):
        pass

    monkeypatch.setattr(cli_env, "get_global_config_dict", lambda **_kwargs: raw)
    monkeypatch.setattr(cli_env, "_bind_manifest_resources_server", lambda *_args: object())

    def stop_at_preflight(config) -> None:
        assert config is raw
        events.append("preflight")
        raise PreflightReached

    monkeypatch.setattr(environment_execution, "preflight_manifest_execution", stop_at_preflight)
    monkeypatch.setattr(cli_env, "_validate_scoring_fixture", lambda *_args, **_kwargs: events.append("fixture"))

    with pytest.raises(PreflightReached):
        cli_env.test.__wrapped__()

    assert events == ["preflight"]


def test_named_environment_uses_canonical_harness_after_component_setup(tmp_path: Path, monkeypatch) -> None:
    scaffold_environment(kind="environment", name="fixture_env", root=tmp_path)
    manifest = load_manifest(tmp_path / "environments" / "fixture_env" / "manifest.yaml")
    resource_dir = tmp_path / "resources_servers" / "fixture_env"
    config = cli_env.TestConfig(entrypoint="resources_servers/fixture_env")
    config._resolved_dir_override = resource_dir
    raw = OmegaConf.load(resource_dir / "configs" / "fixture_env.yaml")
    raw.uv_venv_dir = str(tmp_path)
    captured = {}

    monkeypatch.setattr(cli_env, "server_venv_path", lambda *_args: resource_dir / ".venv")

    def fake_run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli_env.subprocess, "run", fake_run)

    assert cli_env._run_canonical_verifier_harness(config, raw, manifest) == 0
    assert captured["command"][:3] == (
        str(resource_dir / ".venv" / "bin" / "python"),
        "-m",
        VERIFIER_HARNESS_MODULE,
    )
    assert json.loads(captured["input"])["entrypoint"] == "app.py"
    assert captured["cwd"] == resource_dir
    assert captured["check"] is False


def test_fixture_contract_covers_novel_reused_and_session_stateful_shapes(tmp_path: Path) -> None:
    novel = scaffold_environment(
        kind="benchmark",
        name="novel_benchmark",
        metadata={"canonical_split": "test"},
        root=tmp_path,
    )
    scaffold_environment(kind="environment", name="shared_scorer", root=tmp_path)
    reused = scaffold_environment(
        kind="benchmark",
        name="reused_benchmark",
        reuse_verifier="shared_scorer",
        metadata={"canonical_split": "test"},
        root=tmp_path,
    )
    stateful = scaffold_environment(
        kind="environment",
        name="stateful_environment",
        root=tmp_path,
        metadata={"determinism": "seeded"},
    )
    stateful_manifest_path = stateful.asset_dir / "manifest.yaml"
    stateful_manifest = load_manifest(stateful_manifest_path)
    dump_manifest(
        EnvironmentManifest.model_validate(
            {
                **stateful_manifest.model_dump(mode="json"),
                "session_model": "episode",
                "state": "per_session",
            }
        ),
        stateful_manifest_path,
    )

    selections = (
        (novel.asset_dir / "manifest.yaml", novel.asset_dir / "config.yaml", "novel_benchmark"),
        (reused.asset_dir / "manifest.yaml", reused.asset_dir / "config.yaml", "shared_scorer"),
        (stateful_manifest_path, stateful.asset_dir / "config.yaml", "stateful_environment"),
    )
    for manifest_path, recipe_path, resource_directory in selections:
        raw = OmegaConf.create(
            {
                "config_paths": [
                    str(recipe_path),
                    str(
                        tmp_path / "resources_servers" / resource_directory / "configs" / f"{resource_directory}.yaml"
                    ),
                ]
            }
        )
        config = cli_env.TestConfig(manifest_path=manifest_path, environment_ref=manifest_path.parent.name)
        manifest = cli_env._bind_manifest_resources_server(config, raw)
        assert manifest is not None
        complete_scaffold_fixture(config.fixture_path, manifest)
        assert cli_env._validate_scoring_fixture(config, manifest, require_expected_values=True)
        component_dir = tmp_path / "resources_servers" / resource_directory
        exercise_selected_verifier(
            component_dir=component_dir,
            entrypoint="app.py",
            instance_name=f"{resource_directory}_resources_server",
            server_config={"entrypoint": "app.py"},
            fixture_path=component_dir / "tests" / "verifier_cases.jsonl",
            reward_range=manifest.reward.range,
            higher_is_better=manifest.reward.higher_is_better,
            determinism=manifest.determinism.value,
        )
