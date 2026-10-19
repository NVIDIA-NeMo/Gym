# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gzip
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from responses_api_agents.claude_code_agent import runtime


class Response:
    def __init__(self, data=b"", manifest=None):
        self.data = data
        self.manifest = manifest
        self.content = self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    async def json(self):
        return self.manifest

    async def iter_chunked(self, size):
        yield self.data


@pytest.mark.asyncio
async def test_host_downloads_and_verifies_once_across_concurrent_seeds(tmp_path, monkeypatch):
    data = b"Linux x64 Claude bytes"
    checksum = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(runtime, "CACHE_DIR", tmp_path)

    async def fetch(method, url, **kwargs):
        if url.endswith("manifest.json"):
            return Response(manifest={"platforms": {"linux-x64": {"checksum": checksum}}})
        assert url.endswith("/2.1.108/linux-x64/claude")
        return Response(data=data)

    request = AsyncMock(side_effect=fetch)
    monkeypatch.setattr(runtime, "request", request)
    artifacts = await asyncio.gather(*(runtime.cached_binary("2.1.108", "linux-x64") for _ in range(4)))
    assert len(set(artifacts)) == 1
    path, expected = artifacts[0]
    assert path.read_bytes() == data and expected == checksum
    assert request.await_count == 2
    path.write_bytes(b"corrupt")
    await runtime.cached_binary("2.1.108", "linux-x64")
    assert path.read_bytes() == data
    assert request.await_count == 3


@pytest.mark.asyncio
async def test_bad_host_download_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(runtime, "request", AsyncMock(return_value=Response(data=b"corrupt")))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        await runtime.cached_binary("2.1.108", "linux-arm64", expected="0" * 64)
    assert not list(tmp_path.rglob("claude"))
    assert not list(tmp_path.rglob(".claude-*"))


def result(stdout="", code=0):
    return SimpleNamespace(stdout=stdout, stderr="", return_code=code, error_type=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("cached", [False, True])
async def test_upload_uses_sandbox_architecture_and_skips_healthy_remote_cache(tmp_path, monkeypatch, cached):
    monkeypatch.setattr(runtime, "CACHE_DIR", tmp_path / "cache")
    artifact = tmp_path / "claude"
    artifact.write_bytes(b"arm-musl-native-binary")
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    resolve = AsyncMock(return_value=(artifact, checksum))
    monkeypatch.setattr(runtime, "cached_binary", resolve)
    transfer = AsyncMock(return_value="/session/claude-upload.gz")
    monkeypatch.setattr(runtime, "_upload_archive", transfer)
    sandbox = SimpleNamespace(
        exec=AsyncMock(side_effect=[result("Linux\naarch64\nmusl\n"), result(code=0 if cached else 3), result()]),
        upload=AsyncMock(),
    )
    await runtime.install_sandbox_runtime(
        sandbox, version="2.1.108", directory="/session", runtime="/runtime", timeout=60
    )
    resolve.assert_awaited_once_with("2.1.108", "linux-arm64-musl", expected=None)
    if cached:
        sandbox.upload.assert_not_awaited()
        transfer.assert_not_awaited()
        assert sandbox.exec.await_count == 2
    else:
        assert gzip.decompress(transfer.await_args.args[1].read_bytes()) == artifact.read_bytes()
        command = sandbox.exec.await_args.args[0]
        assert (
            "--source /session/claude-upload.gz" in command
            and "--gzip" in command
            and "--sha256 " + checksum in command
        )
        assert "http" not in command


@pytest.mark.asyncio
async def test_host_local_binary_requires_no_network(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "claude"
    source.write_bytes(b"supplied-linux-binary")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    download = AsyncMock(side_effect=AssertionError("network forbidden"))
    monkeypatch.setattr(runtime, "request", download)
    transfer = AsyncMock(return_value="/session/claude-upload.gz")
    monkeypatch.setattr(runtime, "_upload_archive", transfer)
    sandbox = SimpleNamespace(exec=AsyncMock(side_effect=[result(code=3), result()]), upload=AsyncMock())
    await runtime.install_sandbox_runtime(
        sandbox,
        version="2.1.108",
        directory="/session",
        runtime="/runtime",
        timeout=60,
        local_source=str(source),
        expected=checksum,
    )
    download.assert_not_awaited()
    assert gzip.decompress(transfer.await_args.args[1].read_bytes()) == source.read_bytes()


def test_worker_installer_has_no_download_dependency():
    from responses_api_agents.claude_code_agent import install_claude_runtime

    text = Path(install_claude_runtime.__file__).read_text()
    assert "urllib" not in text and "https://" not in text


@pytest.mark.asyncio
async def test_corrupted_compressed_host_cache_is_rebuilt(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "CACHE_DIR", tmp_path / "cache")
    source = tmp_path / "claude"
    source.write_bytes(b"native bytes" * 100)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    compressed = await runtime.compressed_binary(source, checksum)
    compressed.write_bytes(b"damaged gzip")
    repaired = await runtime.compressed_binary(source, checksum)
    assert gzip.decompress(repaired.read_bytes()) == source.read_bytes()


@pytest.mark.asyncio
async def test_bounded_upload_reassembles_bytes_and_removes_parts(tmp_path):
    archive = tmp_path / "source.gz"
    data = bytes(range(256)) * (1024 * 24 + 7)
    archive.write_bytes(data)
    directory = tmp_path / "remote session"
    directory.mkdir()
    active = maximum = 0
    sizes = []

    async def upload(source, destination):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        sizes.append(source.stat().st_size)
        await asyncio.sleep(0)
        shutil.copyfile(source, destination)
        active -= 1

    async def execute(command, **kwargs):
        process = await asyncio.create_subprocess_shell(command)
        return result(code=await process.wait())

    sandbox = SimpleNamespace(upload=upload, exec=execute)
    destination = await runtime._upload_archive(sandbox, archive, str(directory), 60)
    assert Path(destination).read_bytes() == data
    assert not list(directory.glob("*.part"))
    assert max(sizes) <= 1024 * 1024 and sum(sizes) == len(data)
    assert maximum == 4
