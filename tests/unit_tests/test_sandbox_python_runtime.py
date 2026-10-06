# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox.python_runtime import ensure_python


pytestmark = pytest.mark.sandbox


def result(code=0, error=None):
    return SimpleNamespace(return_code=code, error_type=error, stderr="bootstrap failed")


@pytest.mark.asyncio
async def test_existing_python_needs_no_archive_or_network():
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=result()))
    assert await ensure_python(sandbox) == "python3"
    sandbox.exec.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_runtime_and_transport_anomaly_fail():
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=result(0, "timeout")))
    with pytest.raises(RuntimeError, match="pinned python_runtime_url"):
        await ensure_python(sandbox)


@pytest.mark.asyncio
async def test_verified_archive_is_reusable_and_outside_task_tree():
    sandbox = SimpleNamespace(exec=AsyncMock(side_effect=[result(127), result()]))
    digest = "a" * 64
    executable = await ensure_python(
        sandbox, runtime_url="https://runtime.example/python.tar.gz", runtime_sha256=digest
    )
    assert executable == f"/tmp/nemo-gym-python-{digest}/python/bin/python3"
    command = sandbox.exec.await_args_list[1].args[0]
    assert command.index("sha256sum -c -") < command.index("tar -xzf") < command.index("mv -T")
    assert ' -o "$tmp/archive.tar.gz" -- https://runtime.example/python.tar.gz' in command
    assert sandbox.exec.await_args_list[1].kwargs["timeout_s"] == 180


@pytest.mark.asyncio
@pytest.mark.parametrize("url,digest", [("file:///tmp/runtime", "a" * 64), ("https://runtime.example", "bad")])
async def test_unpinned_or_non_http_archive_is_rejected(url, digest):
    sandbox = SimpleNamespace(exec=AsyncMock(return_value=result(127)))
    with pytest.raises(ValueError):
        await ensure_python(sandbox, runtime_url=url, runtime_sha256=digest)
    assert sandbox.exec.await_count == 1


@pytest.mark.asyncio
async def test_checksum_or_extraction_failure_does_not_return_executable():
    sandbox = SimpleNamespace(exec=AsyncMock(side_effect=[result(127), result(1)]))
    with pytest.raises(RuntimeError, match="bootstrap failed"):
        await ensure_python(sandbox, runtime_url="https://runtime.example/python.tar.gz", runtime_sha256="a" * 64)
