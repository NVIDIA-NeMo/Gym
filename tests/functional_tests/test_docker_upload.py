# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Docker transfer checks; opt in with NEMO_GYM_DOCKER_E2E=1."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.providers.docker.provider import DockerProvider


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("NEMO_GYM_DOCKER_E2E") != "1",
        reason="starts real Docker containers; set NEMO_GYM_DOCKER_E2E=1 to run",
    ),
    pytest.mark.skipif(shutil.which("docker") is None, reason="Docker CLI is unavailable"),
]


@pytest.fixture(scope="module")
def docker_daemon() -> None:
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("Docker daemon is unavailable")
    if result.returncode != 0:
        pytest.skip("Docker daemon is unavailable")


@pytest.mark.parametrize("source_kind", ["regular", "absolute_symlink", "relative_symlink"])
async def test_upload_preserves_files_and_source_links(docker_daemon: None, tmp_path: Path, source_kind: str) -> None:
    original = tmp_path / "original.sh"
    payload = b'#!/bin/sh\nprintf "docker-upload-payload\\n"\n'
    original.write_bytes(payload)
    source = original
    if source_kind != "regular":
        source = tmp_path / "runner.sh"
        source.symlink_to(original if source_kind == "absolute_symlink" else Path("original.sh"))

    image = os.environ.get("NEMO_GYM_DOCKER_TEST_IMAGE", "alpine:3.21")
    provider = DockerProvider(
        create={"start_timeout_s": 60, "extra_run_args": ["--label", "nemo-gym.docker-upload-regression=1"]},
        exec={"default_timeout_s": 30},
    )
    sandbox = AsyncSandbox(provider)
    try:
        await sandbox.start(SandboxSpec(image=image, ttl_s=120))
        await sandbox.upload(source, "/tmp/runner.sh")

        assert source.is_symlink() is (source_kind != "regular")
        assert original.read_bytes() == payload
        if source_kind == "regular":
            command = "test -f /tmp/runner.sh && test ! -L /tmp/runner.sh && sh /tmp/runner.sh"
            expected_output = "docker-upload-payload\n"
        else:
            command = "test -L /tmp/runner.sh && readlink /tmp/runner.sh"
            expected_output = f"{source.readlink()}\n"
        result = await sandbox.exec(command)
        assert result.error_type is None, result
        assert result.return_code == 0, result
        assert result.stdout == expected_output
        downloaded = tmp_path / "downloaded.sh"
        await sandbox.download("/tmp/runner.sh", downloaded)
        if source_kind == "regular":
            assert not downloaded.is_symlink()
            assert downloaded.read_bytes() == payload
        else:
            assert downloaded.is_symlink()
            assert downloaded.readlink() == source.readlink()
    finally:
        await sandbox.stop()
