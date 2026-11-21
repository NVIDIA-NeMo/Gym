# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real Docker uploads; opt in with NEMO_GYM_DOCKER_E2E=1 (requires a Docker daemon)."""

import os
import shlex
from pathlib import Path

import pytest

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxSpec
from nemo_gym.sandbox.providers.docker.provider import DockerProvider


pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(
        os.environ.get("NEMO_GYM_DOCKER_E2E") != "1", reason="set NEMO_GYM_DOCKER_E2E=1 for real Docker"
    ),
]


@pytest.mark.parametrize("source_kind", ["regular", "relative_symlink", "absolute_symlink", "symlink_chain"])
async def test_upload_installs_executable_file_contents(tmp_path: Path, source_kind: str) -> None:
    script = tmp_path / "installer source.sh"
    script.write_text("#!/bin/sh\nprintf 'installed\\n'\n")
    script.chmod(0o751)
    source = script
    if source_kind != "regular":
        source = tmp_path / "installer link.sh"
        source.symlink_to(script if source_kind == "absolute_symlink" else script.name)
        if source_kind == "symlink_chain":
            outer = tmp_path / "outer link.sh"
            outer.symlink_to(source.name)
            source = outer

    sandbox = await AsyncSandbox(DockerProvider()).start(SandboxSpec(image="alpine:3.23"))
    try:
        target = "/tmp/upload path/installer.sh"
        await sandbox.upload(source, target)
        quoted = shlex.quote(target)
        result = await sandbox.exec(f"test -f {quoted} && test ! -L {quoted} && {quoted}")
        assert result.error_type is None
        assert result.return_code == 0, result.stderr
        assert result.stdout == "installed\n"

        downloaded = tmp_path / "downloaded.sh"
        await sandbox.download(target, downloaded)
        assert downloaded.read_bytes() == script.read_bytes()
        assert not downloaded.is_symlink()
    finally:
        await sandbox.stop()


async def test_upload_broken_symlink_reports_failure(tmp_path: Path) -> None:
    source = tmp_path / "broken.sh"
    source.symlink_to("missing.sh")
    sandbox = await AsyncSandbox(DockerProvider()).start(SandboxSpec(image="alpine:3.23"))
    try:
        with pytest.raises(RuntimeError, match="docker cp upload"):
            await sandbox.upload(source, "/tmp/broken.sh")
    finally:
        await sandbox.stop()
