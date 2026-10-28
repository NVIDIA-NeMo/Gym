# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI must exercise the same isolated executable and dependencies as the agent."""

import json
import subprocess

import pytest
from scripts.harness_conformance import multimodal_runtime as runtime


@pytest.mark.parametrize("failure", [None, "namespace", "version"])
def test_provision_checks_real_sandbox_before_export(monkeypatch, tmp_path, failure):
    directory = tmp_path / "runtime with spaces"
    dependencies = directory / "dependencies"
    native = dependencies / "node_modules/opencode-linux-x64/bin/opencode"
    calls = []
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime.sys, "platform", "linux")

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "npm":
            native.parent.mkdir(parents=True)
            native.write_bytes(b"\x7fELF")
            package = dependencies / "node_modules/@opencode-ai/plugin/package.json"
            package.parent.mkdir(parents=True)
            package.write_text("{}")
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
        assert "--ignore-scripts" in calls[0]
        assert "@opencode-ai/plugin@1.17.11" in calls[0]
        assert "--clearenv" in calls[-1]
        assert str(native) in calls[-1]
        assert "import PIL, numpy, scipy, cv2" in calls[-1][-1]
        assert "opencode debug config" in calls[-1][-1]
        assert ".gitignore" in (dependencies / ".gitignore").read_text()


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
