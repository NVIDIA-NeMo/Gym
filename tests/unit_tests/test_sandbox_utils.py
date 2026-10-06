# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.providers.local import LocalProvider
from nemo_gym.sandbox.utils import read_bytes, upload_bytes


pytestmark = pytest.mark.sandbox


@pytest.mark.parametrize("data", [b"", bytes(range(256)) + b"\r\n\r"], ids=["empty", "binary"])
async def test_bytes_round_trip_preserves_content_and_cleans_temporary_files(tmp_path: Path, data: bytes) -> None:
    path = tmp_path / "payload's $(literal).bin"
    async with AsyncSandbox(LocalProvider(), SandboxSpec(workdir=str(tmp_path))) as sandbox:
        await sandbox.start()
        sandbox.upload = AsyncMock(wraps=sandbox.upload)
        sandbox.download = AsyncMock(wraps=sandbox.download)
        sandbox.exec = AsyncMock(side_effect=AssertionError("file transfer must not invoke a shell"))

        await upload_bytes(sandbox, path=str(path), data=data)
        assert path.read_bytes() == data
        assert await read_bytes(sandbox, path=str(path)) == data

        assert not Path(sandbox.upload.await_args.args[0]).parent.exists()
        assert not Path(sandbox.download.await_args.args[1]).parent.exists()
        sandbox.exec.assert_not_awaited()


@pytest.mark.parametrize("operation", ["upload", "download"])
@pytest.mark.parametrize("error_type", [OSError, asyncio.CancelledError])
async def test_bytes_transfer_cleans_temporary_files_on_error(operation: str, error_type: type[BaseException]) -> None:
    sandbox = AsyncMock(spec=AsyncSandbox)
    error = error_type("transfer interrupted")
    temporary_paths: list[Path] = []

    async def fail_transfer(source: str | Path, destination: str | Path) -> None:
        local = Path(source if operation == "upload" else destination)
        assert local.parent.is_dir()
        if operation == "download":
            local.write_bytes(b"partial")
        temporary_paths.append(local)
        raise error

    getattr(sandbox, operation).side_effect = fail_transfer
    with pytest.raises(error_type) as raised:
        if operation == "upload":
            await upload_bytes(sandbox, path="/remote.bin", data=b"\x00\xff")
        else:
            await read_bytes(sandbox, path="/remote.bin")

    assert raised.value is error
    assert len(temporary_paths) == 1
    assert not temporary_paths[0].parent.exists()
