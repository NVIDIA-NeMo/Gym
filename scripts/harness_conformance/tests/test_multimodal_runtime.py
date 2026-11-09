# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI must exercise the same isolated executable and dependencies as the agent."""

import hashlib
import json
import subprocess

import pytest
from scripts.harness_conformance import multimodal_runtime as runtime


@pytest.mark.parametrize("failure", [None, "namespace", "version", "executable_integrity", "plugin_integrity"])
def test_provision_checks_real_sandbox_before_export(monkeypatch, tmp_path, failure):
    directory = tmp_path / "runtime with spaces"
    dependencies = directory / "dependencies"
    native = dependencies / "node_modules/opencode-linux-x64-baseline/bin/opencode"
    pins = runtime.runtime_pins("1.17.11", "x64")
    pins["executable_sha256"] = hashlib.sha256(b"\x7fELF").hexdigest()
    monkeypatch.setattr(runtime, "runtime_pins", lambda version, architecture: pins)
    calls = []
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime.sys, "platform", "linux")

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "npm":
            native.parent.mkdir(parents=True)
            native.write_bytes(b"\x7fELFchanged" if failure == "executable_integrity" else b"\x7fELF")
            package = dependencies / "node_modules/@opencode-ai/plugin/package.json"
            package.parent.mkdir(parents=True)
            package.write_text("{}")
            (dependencies / "package-lock.json").write_text(
                json.dumps(
                    {
                        "packages": {
                            f"node_modules/{pins['package']}": {
                                "version": "1.17.11",
                                "integrity": pins["npm_integrity"],
                            },
                            "node_modules/@opencode-ai/plugin": {
                                "version": "1.17.11",
                                "integrity": "changed"
                                if failure == "plugin_integrity"
                                else pins["plugin"]["integrity"],
                            },
                        }
                    }
                )
            )
        if command[0] == "uv" and command[1] == "venv":
            python = directory / "python/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(runtime.sys.executable)
        if command[0] == "/usr/bin/bwrap":
            if failure == "namespace":
                raise subprocess.CalledProcessError(1, command, stderr="user namespace denied")
            return subprocess.CompletedProcess(command, 0, stdout="wrong\n" if failure == "version" else "1.17.11\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    if failure:
        with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
            runtime.provision(directory, bwrap_command=["/usr/bin/bwrap"])
        assert not (directory / "sandbox.json").exists()
    else:
        config = json.loads(runtime.provision(directory, bwrap_command=["/usr/bin/bwrap"]).read_text())
        assert runtime.shlex.split(config["command"]) == [str(native)]
        assert config["sandbox"]["plugin_dependencies"] == str(dependencies)
        assert config["executable_sha256"] == pins["executable_sha256"]
        assert config["runtime_provenance"] == pins
        assert "--ignore-scripts" in calls[0]
        assert "@opencode-ai/plugin@1.17.11" in calls[0]
        assert "--clearenv" in calls[-1]
        assert str(native) in calls[-1]
        assert "import PIL, numpy, scipy, cv2" in calls[-1][-1]
        assert "opencode debug config" in calls[-1][-1]
        assert ".gitignore" in (dependencies / ".gitignore").read_text()


@pytest.mark.parametrize(
    "architecture,package", [("x64", "opencode-linux-x64-baseline"), ("arm64", "opencode-linux-arm64")]
)
def test_runtime_pins_match_production_variant(architecture, package):
    pins = runtime.runtime_pins("1.17.11", architecture)
    assert pins["package"] == package
    assert pins["release_url"].endswith(package + ".tar.gz")
    assert len(pins["executable_sha256"]) == len(pins["release_sha256"]) == 64


def test_new_runtime_release_requires_reviewed_pins():
    with pytest.raises(RuntimeError, match="Update multimodal runtime pins"):
        runtime.runtime_pins("unreviewed-version", "x64")


def test_cli_exports_config_and_native_path(monkeypatch, tmp_path):
    config = tmp_path / "sandbox.json"
    config.write_text(json.dumps({"command": "'/tmp/runtime with spaces/bin/opencode'"}))
    monkeypatch.setattr(runtime, "provision", lambda directory: config)
    env_file = tmp_path / "github-env"
    monkeypatch.setattr(runtime.sys, "argv", ["setup", "--directory", str(tmp_path), "--github-env", str(env_file)])
    monkeypatch.setenv("PATH", "/usr/bin")
    runtime.main()
    exported = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
    assert exported["GYM_MULTIMODAL_SANDBOX_CONFIG"] == str(config)
    assert exported["PATH"] == "/tmp/runtime with spaces/bin:/usr/bin"


def test_scripted_multimodal_probe_does_not_claim_an_enforcing_gateway(monkeypatch, tmp_path):
    import json

    from scripts.harness_conformance.episode import _config

    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"command": str(tmp_path / "opencode"), "sandbox": {"python_env": str(tmp_path)}}))
    monkeypatch.setenv("GYM_MULTIMODAL_SANDBOX_CONFIG", str(runtime))
    config = _config(
        "opencode_multimodal",
        tmp_path,
        {"policy_model": 1, "probe_resources": 2, "probe_agent": 3, "probe_environment": 4},
        30,
    )
    agent = config["probe_agent"]["responses_api_agents"]["opencode_multimodal_agent"]
    assert agent["max_turns"] == 65
    assert not agent.get("turn_budget")
    assert "probe" in config["policy_model"]["responses_api_models"]
    assert config["head_server"]["port"] != config["policy_model"]["responses_api_models"]["probe"]["port"]


@pytest.mark.parametrize("pin", [None, "wrong", "correct"])
def test_session_probe_uses_expected_integrity_pin(monkeypatch, tmp_path, pin):
    from scripts.harness_conformance.episode import _config

    executable = tmp_path / "runtime with spaces" / "opencode"
    executable.parent.mkdir()
    executable.write_bytes(b"controlled executable")
    expected = hashlib.sha256(executable.read_bytes()).hexdigest()
    settings = {
        "command": runtime.shlex.quote(str(executable)),
        "sandbox": {"python_env": str(tmp_path)},
    }
    if pin is not None:
        settings["executable_sha256"] = expected if pin == "correct" else "0" * 64
    config_path = tmp_path / "sandbox.json"
    config_path.write_text(json.dumps(settings))
    monkeypatch.setenv("GYM_MULTIMODAL_SANDBOX_CONFIG", str(config_path))
    args = (
        "opencode_multimodal",
        tmp_path,
        {"policy_model": 1, "probe_resources": 2, "probe_agent": 3, "probe_environment": 4},
        30,
    )
    if pin != "correct":
        with pytest.raises(RuntimeError, match="integrity pin"):
            _config(*args, use_agent_sessions=True)
        return
    config = _config(*args, use_agent_sessions=True)
    agent = config["probe_agent"]["responses_api_agents"]["opencode_multimodal_agent"]
    assert agent["sandbox_executable_path"] == str(executable)
    assert agent["sandbox_executable_sha256"] == expected
