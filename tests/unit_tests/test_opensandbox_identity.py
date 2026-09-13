# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle, SandboxPtyError
from nemo_gym.sandbox.providers.opensandbox.provider import (
    OpenSandboxIdentityConfig,
    OpenSandboxProvider,
    SandboxBackendUnreachableError,
)


pytestmark = pytest.mark.sandbox


@pytest.fixture
def local_provider(monkeypatch):
    provider = OpenSandboxProvider(identity={"hostname_suffix": "", "file_chunk_bytes": 7})
    handle = SandboxHandle(
        sandbox_id=Path("/proc/sys/kernel/hostname").read_text().strip(),
        provider_name="opensandbox",
        raw=SimpleNamespace(files=SimpleNamespace(write_file=AsyncMock(), read_bytes=AsyncMock())),
    )

    async def run(_handle, command, **kwargs):
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            cwd=kwargs.get("cwd"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        return SandboxExecResult(stdout.decode(), stderr.decode(), process.returncode)

    monkeypatch.setattr(provider, "_exec_unchecked", run)
    return provider, handle


@pytest.mark.asyncio
async def test_guard_preserves_stdout_and_normal_exit_codes(local_provider):
    provider, handle = local_provider
    result = await provider.exec(handle, "printf 'answer'; exit 97")
    assert result.stdout == "answer"
    assert result.return_code == 97
    evidence = provider.identity_verification(handle)
    assert evidence["confirmed_responses"] == 1
    assert evidence["identity_errors"] == 0


@pytest.mark.asyncio
async def test_missing_cwd_is_checked_then_returned_as_command_error(local_provider, tmp_path):
    provider, handle = local_provider
    result = await provider.exec(handle, "true", cwd=str(tmp_path / "absent"))
    assert result.return_code != 0
    assert provider.identity_verification(handle)["confirmed_responses"] == 1


@pytest.mark.asyncio
async def test_mismatch_prevents_action_and_file_access(local_provider, tmp_path):
    provider, _ = local_provider
    raw = SimpleNamespace(files=SimpleNamespace(write_file=AsyncMock(), read_bytes=AsyncMock()))
    wrong = SandboxHandle(sandbox_id="another-sandbox", provider_name="opensandbox", raw=raw)
    target = tmp_path / "must-not-exist"
    with pytest.raises(SandboxBackendUnreachableError, match="identity"):
        await provider.exec(wrong, f"touch '{target}'")
    with pytest.raises(SandboxBackendUnreachableError, match="identity"):
        await provider._write_file(wrong, str(target), b"private artifact")
    with pytest.raises(SandboxBackendUnreachableError, match="identity"):
        await provider._read_file(wrong, str(target))
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []
    raw.files.write_file.assert_not_called()
    raw.files.read_bytes.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"", b"\x00\xff\x01" * 19, b"line one\nline two\n"])
async def test_guarded_file_roundtrip_and_literal_paths(local_provider, tmp_path, payload):
    provider, handle = local_provider
    target = tmp_path / "artifact ' $(touch injected).bin"
    await provider._write_file(handle, str(target), payload)
    assert target.read_bytes() == payload
    assert await provider._read_file(handle, str(target)) == payload
    assert [path.name for path in tmp_path.iterdir()] == [target.name]
    evidence = provider.identity_verification(handle)
    assert evidence["guarded_uploads"] == 1
    assert evidence["guarded_downloads"] == 1
    assert evidence["guarded_commands"] == evidence["confirmed_responses"]
    handle.raw.files.write_file.assert_not_called()
    handle.raw.files.read_bytes.assert_not_called()


@pytest.mark.asyncio
async def test_missing_confirmation_is_rejected(local_provider, monkeypatch):
    provider, handle = local_provider
    monkeypatch.setattr(
        provider, "_exec_unchecked", AsyncMock(return_value=SandboxExecResult("other output", None, 0))
    )
    with pytest.raises(SandboxBackendUnreachableError, match="identity"):
        await provider.exec(handle, "true")
    assert provider.identity_verification(handle)["identity_errors"] == 1


@pytest.mark.asyncio
async def test_empty_output_confirmation_without_final_newline(local_provider, monkeypatch):
    provider, handle = local_provider

    async def trimmed(_handle, command, **kwargs):
        marker = re.search(r"NEMO_GYM_IDENTITY_[0-9a-f]+", command).group()
        return SandboxExecResult(marker, None, 0)

    monkeypatch.setattr(provider, "_exec_unchecked", trimmed)
    assert (await provider.exec(handle, "true")).stdout is None
    assert provider.identity_verification(handle)["confirmed_responses"] == 1


@pytest.mark.asyncio
async def test_binary_chunks_fit_shell_argument_limits(local_provider, tmp_path):
    provider, handle = local_provider
    provider._identity = OpenSandboxIdentityConfig(hostname_suffix="", file_chunk_bytes=65536)
    data = bytes(range(256)) * 513
    target = tmp_path / "large-binary"
    await provider._write_file(handle, str(target), data)
    assert await provider._read_file(handle, str(target)) == data


@pytest.mark.asyncio
async def test_transport_timeout_remains_explicit(local_provider, monkeypatch):
    provider, handle = local_provider
    monkeypatch.setattr(provider, "_exec_unchecked", AsyncMock(side_effect=TimeoutError("deadline")))
    with pytest.raises(TimeoutError):
        await provider.exec(handle, "true")
    assert provider.identity_verification(handle)["guarded_commands"] == 1
    assert provider.identity_verification(handle)["confirmed_responses"] == 0


@pytest.mark.asyncio
async def test_unguarded_pty_routes_fail_closed(local_provider):
    provider, handle = local_provider
    with pytest.raises(SandboxPtyError, match="identity"):
        await provider.create_pty(handle, None)
    with pytest.raises(SandboxPtyError, match="identity"):
        await provider.attach_pty(handle, "existing")


@pytest.mark.parametrize("chunk_size", [0, 65537])
def test_file_chunk_size_is_bounded(chunk_size):
    with pytest.raises(ValueError, match="file_chunk_bytes"):
        OpenSandboxIdentityConfig(file_chunk_bytes=chunk_size)
