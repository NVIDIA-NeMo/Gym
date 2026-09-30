# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import shutil
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.opencode_sandboxed_agent import bootstrap


async def test_existing_python_needs_no_download(monkeypatch):
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult("/usr/bin/python3\n", "", 0)
    download = AsyncMock()
    monkeypatch.setattr(bootstrap, "python_asset", download)
    assert await bootstrap.ensure_python(sandbox, {"user": 1000}) == "/usr/bin/python3"
    download.assert_not_called()
    sandbox.upload.assert_not_called()


@pytest.mark.parametrize("arch,libc", [("x86_64", "gnu"), ("aarch64", "musl")])
async def test_missing_python_installs_as_task_user(tmp_path, monkeypatch, arch, libc):
    archive = tmp_path / "python.tar.gz"
    executable = b"#!/bin/sh\nexit 0\n"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("python/bin/python3")
        member.mode = 0o755
        member.size = len(executable)
        tar.addfile(member, io.BytesIO(executable))
    download = AsyncMock(return_value=archive)
    monkeypatch.setattr(bootstrap, "python_asset", download)
    # Confine generated paths to this test's temporary directory.
    monkeypatch.setattr(bootstrap, "uuid4", lambda: type("ID", (), {"hex": "test"})())
    original_quote = bootstrap.quote
    monkeypatch.setattr(
        bootstrap, "quote", lambda s: original_quote(s.replace("/tmp/nemo-gym-python-test", str(tmp_path / "runtime")))
    )

    class Sandbox:
        async def exec(self, command, **kwargs):
            assert kwargs["user"] == 1000
            if "uname -m" in command:
                return SandboxExecResult(f"{arch}\n{libc}\n", "", 0)
            process = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            return SandboxExecResult(stdout.decode(), stderr.decode(), process.returncode)

        async def upload(self, local, remote):
            shutil.copyfile(local, remote.replace("/tmp/nemo-gym-python-test", str(tmp_path / "runtime")))
            Path(remote.replace("/tmp/nemo-gym-python-test", str(tmp_path / "runtime"))).chmod(0o444)

    await bootstrap.ensure_python(Sandbox(), {"user": 1000})
    download.assert_awaited_once_with(arch, libc)
    assert (tmp_path / "runtime/python/bin/python3").stat().st_mode & 0o111
