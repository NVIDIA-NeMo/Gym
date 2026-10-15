# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import shutil
import stat
import tarfile

import pytest

from environment_servers.harbor.sandbox_io import _exec_as_root, upload_dir
from nemo_gym.sandbox.providers.base import SandboxExecResult


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [False, True])
async def test_root_fallback_when_identity_switch_is_unavailable(exception):
    class Sandbox:
        def __init__(self):
            self.users = []

        async def exec(self, command, *, cwd, timeout_s, user=None):
            self.users.append(user)
            if user == "root":
                if exception:
                    raise RuntimeError("identity switch requires CAP_SETUID")
                return SandboxExecResult(stdout="", stderr="identity switch requires CAP_SETUID", return_code=1)
            return SandboxExecResult(stdout="ready", stderr="", return_code=0)

    sandbox = Sandbox()
    result = await _exec_as_root(sandbox, "mkdir -p /tests", timeout_s=1)
    assert result.return_code == 0 and sandbox.users == ["root", None]


@pytest.mark.asyncio
async def test_root_command_failure_does_not_retry_as_default_user():
    calls = []

    class Sandbox:
        async def exec(self, command, **kwargs):
            calls.append(kwargs["user"])
            return SandboxExecResult(stdout="", stderr="disk full", return_code=1)

    result = await _exec_as_root(Sandbox(), "mkdir -p /tests", timeout_s=1)
    assert result.return_code == 1 and calls == ["root"]


@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("tar") is None, reason="GNU tar is required")
async def test_transfer_preserves_target_permissions_and_task_file_modes(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir(mode=0o700)
    target.mkdir()
    target.chmod(0o777)
    script = source / "test.sh"
    script.write_text("#!/bin/sh\nprintf works")
    script.chmod(0o751)
    secret = source / "private.txt"
    secret.write_text("private")
    secret.chmod(0o600)

    class LocalSandbox:
        async def upload(self, local, remote):
            # A top-level '.' would restore source metadata onto the receiving directory.
            with tarfile.open(local) as archive:
                assert "." not in archive.getnames()
            shutil.copyfile(local, remote)

        async def exec(self, command, *, cwd, timeout_s, user):
            assert user == "root"
            process = await asyncio.create_subprocess_exec(
                "bash", "-c", command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
            return SandboxExecResult(
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                return_code=process.returncode,
            )

    await upload_dir(LocalSandbox(), source, str(target))
    assert stat.S_IMODE(target.stat().st_mode) == 0o777
    assert stat.S_IMODE((target / "test.sh").stat().st_mode) == 0o751
    assert stat.S_IMODE((target / "private.txt").stat().st_mode) == 0o600
    process = await asyncio.create_subprocess_exec(str(target / "test.sh"), stdout=asyncio.subprocess.PIPE)
    stdout, _ = await process.communicate()
    assert process.returncode == 0 and stdout == b"works"
