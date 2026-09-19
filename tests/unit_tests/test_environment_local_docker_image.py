# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest

import nemo_gym.environment.local_docker_image as local_docker_image
from nemo_gym import WORKING_DIR
from nemo_gym.config_types import ConfigError
from nemo_gym.environment.authoring import load_environment


def test_runtime_digest_tracks_content_modes_and_empty_directories(tmp_path: Path) -> None:
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    empty_directory = tmp_path / "empty"
    empty_directory.mkdir()
    initial = local_docker_image._runtime_digest(tmp_path)

    script.write_text("#!/bin/sh\nexit 1\n")
    content_changed = local_docker_image._runtime_digest(tmp_path)
    assert content_changed != initial

    script.chmod(script.stat().st_mode | os.X_OK)
    mode_changed = local_docker_image._runtime_digest(tmp_path)
    assert mode_changed != content_changed

    empty_directory.rmdir()
    directory_changed = local_docker_image._runtime_digest(tmp_path)
    assert directory_changed != mode_changed


def test_run_docker_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(local_docker_image.subprocess, "run", timeout)

    with pytest.raises(ConfigError, match="timed out after 7 seconds"):
        local_docker_image._run_docker("docker", "image", "inspect", "example", timeout_s=7)


def test_build_local_docker_image_reuses_matching_image(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = load_environment(WORKING_DIR / "environments/hello_world")
    calls: list[tuple[tuple[str, ...], int]] = []

    def run_docker(docker: str, *args: str, timeout_s: int) -> subprocess.CompletedProcess[str]:
        calls.append(((docker, *args), timeout_s))
        return subprocess.CompletedProcess((docker, *args), 0, "", "")

    monkeypatch.setattr(local_docker_image.shutil, "which", lambda command: "/usr/bin/docker")
    monkeypatch.setattr(local_docker_image, "_runtime_digest", lambda context: "a" * 64)
    monkeypatch.setattr(local_docker_image, "_run_docker", run_docker)

    image = local_docker_image.build_local_docker_image(loaded)

    assert image == "nemo-gym-hello-world:aaaaaaaaaaaaaaaa"
    assert calls == [
        (
            ("/usr/bin/docker", "image", "inspect", image),
            local_docker_image.DOCKER_INSPECT_TIMEOUT_SECONDS,
        )
    ]


def test_build_local_docker_image_reports_build_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = load_environment(WORKING_DIR / "environments/hello_world")
    calls: list[tuple[str, ...]] = []

    def run_docker(docker: str, *args: str, timeout_s: int) -> subprocess.CompletedProcess[str]:
        calls.append((docker, *args))
        return_code = 1
        stderr = "missing" if args[:2] == ("image", "inspect") else "build exploded"
        return subprocess.CompletedProcess((docker, *args), return_code, "", stderr)

    monkeypatch.setattr(local_docker_image.shutil, "which", lambda command: "/usr/bin/docker")
    monkeypatch.setattr(local_docker_image, "_runtime_digest", lambda context: "b" * 64)
    monkeypatch.setattr(local_docker_image, "_run_docker", run_docker)

    with pytest.raises(ConfigError, match="build exploded"):
        local_docker_image.build_local_docker_image(loaded)

    assert calls[1][1:4] == ("build", "--tag", "nemo-gym-hello-world:bbbbbbbbbbbbbbbb")
