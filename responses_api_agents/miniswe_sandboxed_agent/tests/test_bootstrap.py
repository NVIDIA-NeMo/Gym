# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.miniswe_sandboxed_agent import harness as module
from responses_api_agents.miniswe_sandboxed_agent.tests.conftest import ProcessSandbox


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
async def test_bootstrap_uploads_uv_without_sandbox_https(tmp_path, monkeypatch, arch):
    archive = io.BytesIO()
    executable = b'#!/bin/sh\nprintf "%s\\n" "$*" >> "$(dirname "$0")/uv-calls"\n'
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        member = tarfile.TarInfo(f"uv-{arch}-unknown-linux-musl/uv")
        member.size = len(executable)
        tar.addfile(member, io.BytesIO(executable))
    response = MagicMock()
    response.__aenter__.return_value = response
    response.read = AsyncMock(return_value=archive.getvalue())
    download = AsyncMock(return_value=response)
    monkeypatch.setattr(module, "request", download)

    class BootstrapSandbox(ProcessSandbox):
        async def exec(self, command, **kwargs):
            assert kwargs.get("user") == 1000
            if "platform.machine()" in command:
                Path(harness.remote_directory).mkdir()
                return SandboxExecResult(arch + "\n", "", 0)
            # No HTTPS in the sandbox bootstrap: invalid image CA settings must
            # not affect the download performed by the Gym host.
            assert "https://" not in command
            return await super().exec(command, **kwargs)

        async def upload(self, local, remote):
            await super().upload(local, remote)
            # Some providers upload as root. Extraction must create a separate
            # executable owned by the task user, without chmod'ing the upload.
            Path(remote).chmod(0o444)

    sandbox = BootstrapSandbox(tmp_path)
    harness = module.MiniSWEHarness(
        sandbox=sandbox,
        context=module.HarnessContext(session_id="bootstrap", instruction="test", user=1000),
        config=module.MiniSWEConfig(),
        params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        query=AsyncMock(),
        model_name="test",
        directory=tmp_path / "artifacts",
    )
    harness.directory.mkdir()
    harness.remote_directory = str(tmp_path / "remote")
    await harness._install_runner()

    assert f"uv-{arch}-unknown-linux-musl.tar.gz" in download.await_args.args[1]
    assert "ssl" not in download.await_args.kwargs
    response.raise_for_status.assert_called_once()
    remote = Path(harness.remote_directory)
    assert (remote / "uv").read_bytes() == executable
    assert (remote / "uv").stat().st_mode & 0o111 == 0o111
    calls = (remote / "uv-calls").read_text().splitlines()
    assert calls == [
        f"venv {remote}/venv --python 3.13",
        f"pip install --python {remote}/venv/bin/python mini-swe-agent==2.4.6",
    ]
    assert (remote / "runner.py").read_bytes() == Path(module.__file__).with_name("sandbox_runner.py").read_bytes()
    assert not (harness.directory / "uv.tar.gz").exists()
