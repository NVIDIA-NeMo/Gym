# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relay configuration, isolated installation, and shell-boundary contracts."""

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from responses_api_agents.codex_agent.relay import CodexRelayConfig, relay_files


INSTALLER = Path(__file__).parents[1] / "install_relay_runtime.sh"


@pytest.mark.parametrize(
    "settings",
    [
        {"output_dir": "relative"},
        {"output_dir": "/"},
        {"output_dir": "//"},
        {"output_dir": "/bad\x00path"},
        {"output_dir": "/logs/../tmp"},
        {"executable": " "},
        {"executable": "bad\x00path"},
        {"version": "latest"},
        {"unknown": True},
    ],
)
def test_invalid_configuration_fails_before_task_setup(settings):
    with pytest.raises(ValidationError):
        CodexRelayConfig(**settings)


@pytest.mark.parametrize("session_name", ["session with spaces", "session with emoji \U0001f600"])
def test_private_launcher_preserves_arguments_and_artifact_destination(tmp_path, session_name):
    directory = tmp_path / session_name
    directory.mkdir()
    script = directory / "codex.js"
    script.write_text("import json, sys; print(json.dumps(sys.argv[1:]))\n")
    settings = CodexRelayConfig(output_dir=str(tmp_path / 'recordings "quoted"'))
    files = relay_files(settings, session_dir=str(directory), codex_command=[sys.executable, str(script)])
    for name, text in files.items():
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    config = tomllib.loads(files["relay/config.toml"])
    launcher = Path(config["agents"]["codex"]["command"])
    launcher.chmod(0o700)
    arguments = ["exec", "a b", "$(touch escaped)", "'quoted'", ""]
    result = subprocess.run([str(launcher), *arguments], capture_output=True, text=True, check=True, timeout=10)
    assert json.loads(result.stdout) == arguments
    assert not Path("escaped").exists()
    component = tomllib.loads(files["relay/plugins.toml"])["components"][0]["config"]
    assert component["enable_full_payloads"]
    destination = str(Path(settings.output_dir) / directory.name)
    assert component["atof"]["sinks"][0]["output_directory"] == destination
    assert component["atif"]["output_directory"] == destination


@pytest.mark.parametrize("version,code", [("0.11.0", 0), ("0.7.3", 1)])
def test_preinstalled_cli_is_verified_without_modifying_task_packages(tmp_path, version, code):
    binary = tmp_path / "custom relay"
    binary.write_text(f"#!/bin/sh\nprintf 'nemo-relay {version}\\n'\n")
    binary.chmod(0o700)
    runtime = tmp_path / "runtime"
    result = subprocess.run(
        ["bash", str(INSTALLER), str(runtime), "0.11.0", str(binary)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == code, result.stderr
    assert not runtime.exists()
    if code == 0:
        assert result.stdout.strip() == str(binary)
    else:
        assert "version mismatch" in result.stderr


def test_cached_cli_needs_no_network_and_releases_install_lock(tmp_path):
    runtime = tmp_path / "runtime"
    binary = runtime / "relay/0.11.0/nemo-relay"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nprintf 'nemo-relay 0.11.0\\n'\n")
    binary.chmod(0o700)
    # An accidental download is a failure, even on a host with working networking.
    curl = tmp_path / "curl"
    curl.write_text("#!/bin/sh\nexit 99\n")
    curl.chmod(0o700)
    result = subprocess.run(
        ["bash", str(INSTALLER), str(runtime), "0.11.0"],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(binary)
    assert not (binary.parent / "install.lock").exists()


def test_failed_cached_cli_releases_install_lock(tmp_path):
    binary = tmp_path / "relay/0.11.0/nemo-relay"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\nexit 17\n")
    binary.chmod(0o700)
    result = subprocess.run(
        ["bash", str(INSTALLER), str(tmp_path), "0.11.0"],
        capture_output=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert not (binary.parent / "install.lock").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux sandbox installer")
def test_corrupt_download_is_not_published_as_a_runtime(tmp_path):
    curl = tmp_path / "curl"
    curl.write_text('#!/bin/bash\nwhile [[ $1 != -o ]]; do shift; done\nprintf corrupt > "$2"\n')
    curl.chmod(0o700)
    runtime = tmp_path / "runtime"
    result = subprocess.run(
        ["bash", str(INSTALLER), str(runtime), "0.11.0"],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "FAILED" in result.stderr
    assert not (runtime / "relay/0.11.0/nemo-relay").exists()
    assert not (runtime / "relay/0.11.0/install.lock").exists()
