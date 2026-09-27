# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunked file transfers of the OpenSandbox provider: large files move in pieces instead of whole in memory."""

from dataclasses import dataclass
from pathlib import Path

import pytest

import nemo_gym.sandbox.providers.opensandbox.provider as opensandbox_provider
from nemo_gym.sandbox.providers.base import SandboxHandle


CHUNK = 64


@dataclass
class ExecResult:
    return_code: int
    stdout: str = ""
    stderr: str = ""


class RangeServer:
    """Fake `_read_file`: serves byte ranges of one file, or the whole file when told to ignore Range."""

    def __init__(self, content: bytes, honor_range: bool = True):
        self.content = content
        self.honor_range = honor_range
        self.requests: list[str | None] = []

    async def read(self, handle, source_path, range_header=None):
        self.requests.append(range_header)
        if range_header is None or not self.honor_range:
            return self.content
        start, end = (int(x) for x in range_header.removeprefix("bytes=").split("-"))
        if start >= len(self.content):
            raise RuntimeError("HTTP 416 Range Not Satisfiable")
        return self.content[start : end + 1]


class PartStore:
    """Fake sandbox filesystem for uploads: `_write_file` stores parts, `exec` assembles them like `cat` would."""

    def __init__(self, fail_assembly: bool = False):
        self.files: dict[str, bytes] = {}
        self.commands: list[str] = []
        self.fail_assembly = fail_assembly

    async def write(self, handle, target_path, data):
        self.files[target_path] = data

    async def exec(self, handle, command, **kwargs):
        self.commands.append(command)
        if self.fail_assembly:
            return ExecResult(return_code=1, stderr="cat: write error: No space left on device")
        head, _, _ = command.partition(" > ")
        parts = head.split()[1:]
        target = command.split(" > ")[1].split(";")[0].strip()
        self.files[target] = b"".join(self.files.pop(part) for part in parts)
        return ExecResult(return_code=0)


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(opensandbox_provider.OpenSandboxProvider, "TRANSFER_CHUNK_BYTES", CHUNK)
    return opensandbox_provider.OpenSandboxProvider(create={"retries": 0})


@pytest.fixture
def handle():
    return SandboxHandle(sandbox_id="sb-1", provider_name="opensandbox", raw=None)


@pytest.mark.asyncio
async def test_download_assembles_byte_ranges(provider, handle, tmp_path: Path):
    content = bytes(range(256)) * 3  # 768 bytes = 12 full chunks
    content = content[:150]  # 2 full chunks + a 22-byte tail
    server = RangeServer(content)
    provider._read_file = server.read
    target = tmp_path / "out" / "download.tar.gz"
    await provider.download_file(handle, "/tmp/a.tar.gz", target)
    assert target.read_bytes() == content
    assert server.requests == ["bytes=0-63", "bytes=64-127", "bytes=128-191"]


@pytest.mark.asyncio
async def test_download_stops_on_416_when_size_is_a_chunk_multiple(provider, handle, tmp_path: Path):
    content = b"x" * (2 * CHUNK)
    server = RangeServer(content)
    provider._read_file = server.read
    target = tmp_path / "exact.bin"
    await provider.download_file(handle, "/tmp/exact.bin", target)
    assert target.read_bytes() == content
    assert len(server.requests) == 3  # two full pieces, then the 416 that ends the loop


@pytest.mark.asyncio
async def test_download_empty_file(provider, handle, tmp_path: Path):
    server = RangeServer(b"")
    provider._read_file = server.read
    target = tmp_path / "empty.bin"
    await provider.download_file(handle, "/tmp/empty.bin", target)
    assert target.exists() and target.read_bytes() == b""


@pytest.mark.asyncio
async def test_download_falls_back_when_server_ignores_range(provider, handle, tmp_path: Path):
    content = b"y" * (3 * CHUNK + 5)
    server = RangeServer(content, honor_range=False)
    provider._read_file = server.read
    target = tmp_path / "whole.bin"
    await provider.download_file(handle, "/tmp/whole.bin", target)
    assert target.read_bytes() == content
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_download_rejects_unranged_body_mid_stream(provider, handle, tmp_path: Path):
    content = b"z" * (3 * CHUNK)
    server = RangeServer(content)
    calls = 0

    async def flaky(handle_, path, range_header=None):
        nonlocal calls
        calls += 1
        if calls == 2:
            return content  # a proxy that ignores Range only sometimes
        return await server.read(handle_, path, range_header)

    provider._read_file = flaky
    with pytest.raises(RuntimeError, match="unranged body"):
        await provider.download_file(handle, "/tmp/z.bin", tmp_path / "z.bin")


@pytest.mark.asyncio
async def test_download_other_errors_propagate(provider, handle, tmp_path: Path):
    async def missing(handle_, path, range_header=None):
        raise RuntimeError("HTTP 404 file not found")

    provider._read_file = missing
    with pytest.raises(RuntimeError, match="404"):
        await provider.download_file(handle, "/tmp/missing", tmp_path / "missing")


@pytest.mark.asyncio
async def test_small_upload_is_a_single_write(provider, handle, tmp_path: Path):
    store = PartStore()
    provider._write_file = store.write
    provider.exec = store.exec
    source = tmp_path / "small.tar.gz"
    source.write_bytes(b"s" * CHUNK)  # exactly one chunk still goes whole
    await provider.upload_file(handle, source, "/tmp/small.tar.gz")
    assert store.files == {"/tmp/small.tar.gz": b"s" * CHUNK}
    assert store.commands == []


@pytest.mark.asyncio
async def test_large_upload_moves_in_parts_and_is_reassembled(provider, handle, tmp_path: Path):
    store = PartStore()
    provider._write_file = store.write
    provider.exec = store.exec
    content = bytes(range(256)) + b"tail"  # 260 bytes = 4 full parts + 4 bytes
    source = tmp_path / "large.tar.gz"
    source.write_bytes(content)
    await provider.upload_file(handle, source, "/tmp/upload.tar.gz")
    assert store.files == {"/tmp/upload.tar.gz": content}
    assert len(store.commands) == 1
    assert store.commands[0].startswith("cat /tmp/upload.tar.gz.nemo-gym-part0000 ")
    assert "rm -f" in store.commands[0] and "exit $rc" in store.commands[0]


@pytest.mark.asyncio
async def test_large_upload_assembly_failure_raises(provider, handle, tmp_path: Path):
    store = PartStore(fail_assembly=True)
    provider._write_file = store.write
    provider.exec = store.exec
    source = tmp_path / "large.bin"
    source.write_bytes(b"q" * (CHUNK + 1))
    with pytest.raises(RuntimeError, match="assemble .* 2 uploaded parts"):
        await provider.upload_file(handle, source, "/tmp/large.bin")


def test_range_not_satisfiable_matcher():
    class WithStatus(Exception):
        status_code = 416

    assert opensandbox_provider._is_range_not_satisfiable_error(WithStatus("x"))
    assert opensandbox_provider._is_range_not_satisfiable_error(RuntimeError("Status code: 416 | Range Not Satisfiable"))
    assert not opensandbox_provider._is_range_not_satisfiable_error(RuntimeError("HTTP 404 not found"))
    assert not opensandbox_provider._is_range_not_satisfiable_error(RuntimeError("read 416 bytes"))
