# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import io
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.miniswe_sandboxed_agent import bootstrap
from responses_api_agents.miniswe_sandboxed_agent import harness as module
from responses_api_agents.miniswe_sandboxed_agent.tests.conftest import ProcessSandbox


def archive(name, content):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        member = tarfile.TarInfo(name)
        member.mode = 0o755
        member.size = len(content)
        tar.addfile(member, io.BytesIO(content))
    return stream.getvalue()


@pytest.mark.parametrize("arch,libc", [("x86_64", "gnu"), ("aarch64", "musl")])
async def test_bootstrap_without_python_caches_assets_across_concurrent_tasks(tmp_path, monkeypatch, arch, libc):
    executable = b'#!/bin/sh\nprintf "%s\\n" "$*" >> "$(dirname "$0")/uv-calls"\n'
    uv = archive(f"uv-{arch}-unknown-linux-musl/uv", executable)
    python = archive("python/bin/python3", b"#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(bootstrap, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(bootstrap, "_DOWNLOAD_LOCK", asyncio.Lock())
    monkeypatch.setitem(bootstrap.PYTHON_SHA256, (arch, libc), hashlib.sha256(python).hexdigest())

    async def fetch(method, url, **kwargs):
        response = MagicMock()
        response.__aenter__.return_value = response
        response.read = AsyncMock(return_value=python if "cpython-" in url else uv)
        return response

    download = AsyncMock(side_effect=fetch)
    monkeypatch.setattr(bootstrap, "request", download)

    class BootstrapSandbox(ProcessSandbox):
        async def exec(self, command, **kwargs):
            assert kwargs.get("user") == 1000
            if "uname -m" in command:
                Path(self.directory).mkdir()
                assert "python3 -c" not in command
                return SandboxExecResult(f"{arch}\n{libc}\n", "", 0)
            assert "https://" not in command
            return await super().exec(command, **kwargs)

        async def upload(self, local, remote):
            await super().upload(local, remote)
            Path(remote).chmod(0o444)  # provider uploads may be root-owned

    harnesses = []
    for index in range(2):
        remote = tmp_path / f"remote-{index}"
        harness = module.MiniSWEHarness(
            sandbox=BootstrapSandbox(remote),
            context=module.HarnessContext(session_id=f"bootstrap-{index}", instruction="test", user=1000),
            config=module.MiniSWEConfig(),
            params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            model_base_url="http://model/v1",
            model_name="test",
            directory=tmp_path / f"artifacts-{index}",
        )
        harness.remote_directory = str(remote)
        harnesses.append(harness)
    await asyncio.gather(*(h._install_miniswe() for h in harnesses))
    assert download.await_count == 2
    for harness in harnesses:
        remote = Path(harness.remote_directory)
        assert (remote / "python/bin/python3").is_file()
        assert (remote / "uv").stat().st_mode & 0o111 == 0o111
        assert (remote / "uv-calls").read_text().splitlines() == [
            f"--no-config venv {remote}/venv --python {remote}/python/bin/python3",
            f"--no-config pip install --python {remote}/venv/bin/python mini-swe-agent==2.4.6",
        ]
        assert not (remote / "runner.py").exists()


async def test_failed_download_does_not_poison_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "CACHE_DIR", tmp_path)
    response = MagicMock()
    response.__aenter__.return_value = response
    response.read = AsyncMock(return_value=b"incomplete")
    monkeypatch.setattr(bootstrap, "request", AsyncMock(return_value=response))
    with pytest.raises(RuntimeError, match="checksum"):
        await bootstrap.cached_download("https://example.com/python.tar.gz", sha256="bad")
    assert list(tmp_path.iterdir()) == []
