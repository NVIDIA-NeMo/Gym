# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared servers use the public Docker provider and its process environment."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from environments.unified_video.prepare import CLAW, component_config, write_launcher
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers.registry import create_provider


def test_generated_config_constructs_the_public_docker_provider(tmp_path, monkeypatch):
    cli_dir = tmp_path / "docker-cli"
    cli_dir.mkdir()
    binary = cli_dir / "docker"
    binary.write_text("#!/bin/sh\nexit 99\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(cli_dir) + os.pathsep + os.environ.get("PATH", ""))
    config = component_config(
        CLAW,
        prepared=SimpleNamespace(tasks=[], folder=tmp_path),
        rows=tmp_path / "rows.jsonl",
        context_tokens=262144,
        output_tokens=8192,
        max_steps=32,
    )
    provider = create_provider(resolve_provider_config("claw_sandbox", config))
    assert provider._binary == str(binary)
    assert "cli_dir" not in config["claw_sandbox"]["docker"]
    assert "client_env_file" not in config["claw_sandbox"]["docker"]


def test_launcher_passes_selected_docker_environment_and_literal_arguments(tmp_path, monkeypatch):
    cli_dir = tmp_path / "docker cli"
    cli_dir.mkdir()
    docker = cli_dir / "docker"
    docker.write_text("#!/bin/sh\nexit 99\n")
    docker.chmod(0o755)
    backend = tmp_path / "backend with spaces"
    backend.mkdir()
    (backend / "client.env").write_text(
        "DOCKER_HOST=tcp://selected.invalid:2376\nDOCKER_TLS_VERIFY=1\n"
        "DOCKER_CERT_PATH=/producer-only/path\nDOCKER_BUILDKIT=0\n"
    )
    keys = ["DOCKER_HOST", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH", "DOCKER_CONFIG", "DOCKER_CONTEXT"]
    program = f"import json,os,sys;print(json.dumps({{'env':{{k:os.environ.get(k) for k in {keys!r}}},'args':sys.argv[1:]}}))"
    gym = cli_dir / "gym"
    gym.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -c {shlex.quote(program)} "$@"\n')
    gym.chmod(0o755)
    monkeypatch.setenv("DOCKER_CONTEXT", "unrelated-context")
    monkeypatch.chdir(tmp_path)
    launcher = write_launcher(tmp_path, backend=Path("backend with spaces"), docker_binary="docker cli/docker")
    result = subprocess.run(
        [str(launcher), "--config", "extra config.yaml"], capture_output=True, text=True, check=True
    )
    observed = json.loads(result.stdout)
    assert observed["env"] == {
        "DOCKER_HOST": "tcp://selected.invalid:2376",
        "DOCKER_TLS_VERIFY": "1",
        "DOCKER_CERT_PATH": str(backend),
        "DOCKER_CONFIG": str(cli_dir),
        "DOCKER_CONTEXT": None,
    }
    assert observed["args"] == [
        "env",
        "start",
        "--config",
        str((tmp_path / "gym_config.yaml").resolve()),
        "--config",
        "extra config.yaml",
    ]


def test_a_different_docker_binary_name_is_not_silently_replaced(tmp_path):
    with pytest.raises(ValueError, match="must name a docker executable"):
        write_launcher(tmp_path, backend=tmp_path, docker_binary=str(tmp_path / "other-docker-wrapper"))
