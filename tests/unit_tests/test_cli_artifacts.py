# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

import nemo_gym.cli.env as cli_env
import nemo_gym.cli.main as cli_main
import nemo_gym.environment.artifacts as artifacts
from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME
from nemo_gym.config_types import ConfigError
from nemo_gym.environment.publication import EnvironmentPublicationReport


@pytest.mark.parametrize("command", [["env", "start"], ["eval", "run"], ["eval", "prepare"]])
@pytest.mark.parametrize(
    "reference_args",
    [
        ["--package", "registry.test/alice/alpha:1"],
        ["alice/alpha:1"],
        ["registry.test/alice/alpha:1"],
        ["alpha.tar.gz"],
    ],
)
def test_package_config_and_components_reach_runtime(monkeypatch, tmp_path, command, reference_args):
    package_root = tmp_path / "package"
    config = package_root / "environments/alpha/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    (package_root / "gym-package.json").write_text(json.dumps({"config_path": "environments/alpha/config.yaml"}))
    pull = Mock(return_value=package_root)
    monkeypatch.setattr(artifacts, "pull_environment_package", pull)
    monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, "/original")
    dispatched = []

    def dispatch(target, overrides):
        dispatched.append((target, overrides))
        assert os.environ[NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME].split(os.pathsep) == [
            str(package_root),
            "/custom",
            "/original",
        ]

    monkeypatch.setattr(cli_main, "dispatch", dispatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gym",
            *command,
            *reference_args,
            "--config",
            "override.yaml",
            "--search-dir",
            "/custom",
        ],
    )
    cli_main.main()

    pull.assert_called_once_with(reference_args[-1])
    assert dispatched[0][1] == [f"+config_paths=[{config},override.yaml]"]
    assert os.environ[NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME] == "/original"


@pytest.mark.parametrize("command", [["env", "start"], ["eval", "run"], ["eval", "prepare"]])
def test_positional_reference_does_not_consume_hydra_overrides(monkeypatch, command):
    dispatch = Mock()
    pull = Mock()
    monkeypatch.setattr(cli_main, "dispatch", dispatch)
    monkeypatch.setattr(artifacts, "pull_environment_package", pull)
    monkeypatch.setattr(sys, "argv", ["gym", *command, "+first=1", "--config", "a.yaml", "+second=2"])
    cli_main.main()
    assert dispatch.call_args.args[1] == ["+config_paths=[a.yaml]", "+first=1", "+second=2"]
    pull.assert_not_called()


@pytest.mark.parametrize("command", [["env", "start"], ["eval", "run"], ["eval", "prepare"]])
def test_positional_reference_conflicts_with_package_flag(monkeypatch, capsys, command):
    pull = Mock()
    monkeypatch.setattr(artifacts, "pull_environment_package", pull)
    monkeypatch.setattr(sys, "argv", ["gym", *command, "alice/alpha:1", "--package", "alice/alpha:2"])
    with pytest.raises(SystemExit, match="2"):
        cli_main.main()
    pull.assert_not_called()
    assert "not both" in capsys.readouterr().err


def test_namespaced_reference_requires_configured_registry(monkeypatch, capsys):
    monkeypatch.delenv("GYM_ENV_REGISTRY", raising=False)
    monkeypatch.setattr(sys, "argv", ["gym", "env", "start", "alice/alpha:1"])
    with pytest.raises(SystemExit, match="2"):
        cli_main.main()
    assert "Set GYM_ENV_REGISTRY" in capsys.readouterr().err


def test_example_eval_loads_package_and_starts_servers(monkeypatch, tmp_path):
    (tmp_path / "gym-package.json").write_text(json.dumps({"config_path": "environments/alpha/config.yaml"}))
    monkeypatch.setattr(artifacts, "pull_environment_package", Mock(return_value=tmp_path))
    dispatch = Mock()
    monkeypatch.setattr(cli_main, "dispatch", dispatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gym",
            "eval",
            "run",
            "alice/alpha:1",
            "--split",
            "example",
            "--model-type",
            "openai_model",
            "--limit",
            "1",
            "--output",
            "./results/alpha.jsonl",
        ],
    )
    cli_main.main()
    target, overrides = dispatch.call_args.args
    assert target == "nemo_gym.cli.eval:e2e_rollout_collection"
    assert str(tmp_path / "environments/alpha/config.yaml") in overrides[0]
    assert "responses_api_models/openai_model/configs/openai_model.yaml" in overrides[0]
    assert overrides[1:] == ["+output_jsonl_fpath=./results/alpha.jsonl", "+limit=1", "+split=example"]


@pytest.mark.parametrize("output", [None, "extracted"])
def test_pull_prints_extracted_root(monkeypatch, tmp_path, capsys, output):
    pull = Mock(return_value=tmp_path)
    monkeypatch.setattr(artifacts, "pull_environment_package", pull)
    argv = ["gym", "env", "pull", "alpha.tar.gz", "--verbose"]
    if output:
        argv += ["--output-dir", output]
    monkeypatch.setattr(sys, "argv", argv)

    cli_main.main()

    pull.assert_called_once_with("alpha.tar.gz", Path(output) if output else None)
    assert capsys.readouterr().out.strip() == str(tmp_path)


@pytest.mark.parametrize("command", [["env", "pull", "missing"], ["env", "start", "--package", "missing"]])
def test_package_errors_are_clean(monkeypatch, capsys, command):
    monkeypatch.setattr(artifacts, "pull_environment_package", Mock(side_effect=ConfigError("Package unavailable")))
    monkeypatch.setattr(sys, "argv", ["gym", *command])

    with pytest.raises(SystemExit, match="2"):
        cli_main.main()

    assert "Package unavailable" in capsys.readouterr().err


def test_pull_rejects_runtime_overrides(monkeypatch, capsys):
    pull = Mock()
    monkeypatch.setattr(artifacts, "pull_environment_package", pull)
    monkeypatch.setattr(sys, "argv", ["gym", "env", "pull", "alpha.tar.gz", "+anything=true"])

    with pytest.raises(SystemExit, match="2"):
        cli_main.main()

    pull.assert_not_called()
    assert "does not accept Hydra overrides" in capsys.readouterr().err


@pytest.mark.parametrize(
    "registry", [None, "registry.test/alice/alpha:1.0.0", artifacts.HUB_REGISTRY + "/alice/alpha:1.0.0"]
)
@pytest.mark.parametrize("json_output", [True, False])
def test_publish_packages_only_after_checks(monkeypatch, tmp_path, capsys, registry, json_output):
    output = tmp_path / "alpha.tar.gz"
    argv = ["gym", "env", "publish", "alpha", "--output", str(output)]
    if registry:
        argv += ["--registry", registry]
    if json_output:
        argv += ["--json"]
    monkeypatch.setattr(sys, "argv", argv)
    entry = Mock(manifest_path=tmp_path / "manifest.yaml", config_path=tmp_path / "config.yaml")
    monkeypatch.setattr(cli_env, "resolve_catalog_entry", Mock(return_value=entry))
    steps = Mock()
    report = EnvironmentPublicationReport("alpha", "1.0.0", "environment", None, "manifest.yaml", 3)
    steps.finalize.return_value = report
    steps.build.return_value = output
    immutable = (registry.rsplit(":", 1)[0] if registry else "registry.test/alice/alpha") + "@sha256:123"
    is_hub = bool(registry and registry.startswith(artifacts.HUB_REGISTRY + "/"))
    steps.submit.return_value = "https://hub.example/development.html#alice%2Falpha"
    steps.push.return_value = immutable
    for owner, method, replacement in [
        (cli_env, "validate_environment", steps.validate),
        (cli_env, "_run_manifest_verifier", steps.verify),
        (cli_env, "finalize_publication", steps.finalize),
        (artifacts, "build_environment_package", steps.build),
        (artifacts, "push_environment_package", steps.push),
        (artifacts, "submit_environment_package", steps.submit),
    ]:
        monkeypatch.setattr(owner, method, replacement)

    cli_main.main()

    assert [call[0] for call in steps.mock_calls] == ["validate", "verify", "finalize", "build"] + (
        ["push"] if registry else []
    ) + (["submit"] if is_hub else [])
    steps.build.assert_called_once_with(entry, output)
    if registry:
        steps.push.assert_called_once_with(output, registry)
    result = capsys.readouterr().out
    if json_output:
        payload = json.loads(result)
        assert payload["package_path"] == str(output)
        assert payload.get("registry_reference") == (immutable if registry else None)
        assert payload.get("hub_submission") == (steps.submit.return_value if is_hub else None)
    else:
        assert f"Package: {output}" in result
        assert (f"Published: {immutable}" in result) is bool(registry)


def test_failed_verifier_prevents_packaging(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli_env,
        "_command_overrides",
        lambda: OmegaConf.create({"onboarding_name": "alpha", "package_registry": "registry.test/alpha:1"}),
    )
    monkeypatch.setattr(
        cli_env, "resolve_catalog_entry", Mock(return_value=Mock(manifest_path=tmp_path / "manifest.yaml"))
    )
    monkeypatch.setattr(cli_env, "validate_environment", Mock())
    monkeypatch.setattr(cli_env, "_run_manifest_verifier", Mock(side_effect=ConfigError("Verifier failed")))
    build = Mock()
    monkeypatch.setattr(artifacts, "build_environment_package", build)

    with pytest.raises(SystemExit, match="1"):
        cli_env.publish_environment_manifest()

    build.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_submit_retries_without_upload(monkeypatch, capsys, fails):
    submit = Mock(
        side_effect=ConfigError("SSH unavailable") if fails else None, return_value="https://hub/development"
    )
    push = Mock()
    monkeypatch.setattr(artifacts, "submit_environment_package", submit)
    monkeypatch.setattr(artifacts, "push_environment_package", push)
    monkeypatch.setattr(sys, "argv", ["gym", "env", "submit", "alice/alpha:1.0.0"])
    if fails:
        with pytest.raises(SystemExit, match="2"):
            cli_main.main()
        assert "SSH unavailable" in capsys.readouterr().err
    else:
        cli_main.main()
        assert "https://hub/development" in capsys.readouterr().out
    submit.assert_called_once_with("alice/alpha:1.0.0")
    push.assert_not_called()


def test_publish_reports_uploaded_package_when_hub_fails(monkeypatch, tmp_path, capsys):
    registry = artifacts.HUB_REGISTRY + "/alice/alpha:1.0.0"
    immutable = registry.rsplit(":", 1)[0] + "@sha256:" + "a" * 64
    monkeypatch.setattr(
        cli_env,
        "_command_overrides",
        lambda: OmegaConf.create(
            {
                "onboarding_name": "alpha",
                "package_registry": registry,
            }
        ),
    )
    monkeypatch.setattr(cli_env, "_manifest_entry", Mock(return_value=Mock(manifest_path=tmp_path / "manifest.yaml")))
    monkeypatch.setattr(cli_env, "validate_environment", Mock())
    monkeypatch.setattr(cli_env, "_run_manifest_verifier", Mock())
    monkeypatch.setattr(
        cli_env,
        "finalize_publication",
        Mock(
            return_value=EnvironmentPublicationReport(
                "alpha",
                "1.0.0",
                "environment",
                None,
                "manifest.yaml",
                3,
            )
        ),
    )
    monkeypatch.setattr(artifacts, "build_environment_package", Mock(return_value=tmp_path / "alpha.tar.gz"))
    push = Mock(return_value=immutable)
    monkeypatch.setattr(artifacts, "push_environment_package", push)
    monkeypatch.setattr(artifacts, "submit_environment_package", Mock(side_effect=ConfigError("SSH unavailable")))
    with pytest.raises(SystemExit, match="1"):
        cli_env.publish_environment_manifest()
    push.assert_called_once()
    output = capsys.readouterr()
    rendered = "".join((output.out + output.err).split())
    assert "".join(f"Package published: {immutable}".split()) in rendered
    assert "".join(f"gym env submit {immutable}".split()) in rendered
