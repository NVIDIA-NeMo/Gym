# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import DEFAULT, AsyncMock

import pytest

from nemo_gym.sandbox import AsyncSandbox, Sandbox, SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.local.provider import LocalProvider
from tests.unit_tests.test_sandbox import FakeSandboxProvider


pytestmark = pytest.mark.sandbox


@pytest.mark.parametrize("failed_operation", ["close", "aclose"])
def test_sync_stop_can_retry_provider_cleanup(tmp_path: Path, failed_operation: str) -> None:
    provider = LocalProvider(workspace_root=str(tmp_path))
    provider.close = AsyncMock(wraps=provider.close)
    provider.aclose = AsyncMock(wraps=provider.aclose)
    failed_call = getattr(provider, failed_operation)
    failed_call.side_effect = [OSError("temporary cleanup failure"), DEFAULT]
    sandbox = Sandbox(provider).start(SandboxSpec())
    try:
        result = sandbox.exec("pwd")
        assert result.return_code == 0
        workspace = Path(result.stdout.strip())
        assert workspace.is_dir()

        with pytest.raises(OSError, match="temporary cleanup failure"):
            sandbox.stop()

        expected_status = SandboxStatus.RUNNING if failed_operation == "close" else SandboxStatus.STOPPED
        assert sandbox.status() == expected_status
        sandbox.stop()
        assert failed_call.await_count == 2
        assert not workspace.exists()
        assert provider.close.await_count == (2 if failed_operation == "close" else 1)
        assert provider.aclose.await_count == (2 if failed_operation == "aclose" else 1)
        assert sandbox.status() == SandboxStatus.STOPPED
        sandbox.stop()
        assert failed_call.await_count == 2
    finally:
        sandbox.stop()


@pytest.mark.parametrize("error", [RuntimeError("stop failed"), TimeoutError(), asyncio.CancelledError()])
async def test_failed_remote_stop_keeps_owned_client_retryable(error):
    provider = FakeSandboxProvider()
    provider.close = AsyncMock(side_effect=[error, None])
    provider.aclose = AsyncMock()
    sandbox = AsyncSandbox(provider, owns_provider=True)
    await sandbox.start(SandboxSpec(image="task"))
    with pytest.raises(type(error)):
        await sandbox.stop()
    provider.aclose.assert_not_awaited()
    await sandbox.stop()
    assert provider.close.await_count == 2
    provider.aclose.assert_awaited_once()
    await sandbox.stop()
    assert provider.close.await_count == 2


async def test_client_close_retry_does_not_repeat_successful_remote_stop():
    provider = FakeSandboxProvider()
    provider.close = AsyncMock()
    provider.aclose = AsyncMock(side_effect=[OSError("client close"), None])
    sandbox = AsyncSandbox(provider, owns_provider=True)
    await sandbox.start(SandboxSpec(image="task"))
    with pytest.raises(OSError, match="client close"):
        await sandbox.stop()
    await sandbox.stop()
    provider.close.assert_awaited_once()
    assert provider.aclose.await_count == 2


async def test_opensandbox_kill_failure_does_not_close_sdk_before_retry():
    from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider

    raw = SimpleNamespace(kill=AsyncMock(side_effect=[RuntimeError("kill failed"), None]), close=AsyncMock())
    provider = OpenSandboxProvider(operations={"retries": 0}, probe={"command": None})
    handle = SandboxHandle(sandbox_id="test", provider_name="opensandbox", raw=raw)
    with pytest.raises(RuntimeError, match="kill failed"):
        await provider.close(handle)
    raw.close.assert_not_awaited()
    await provider.close(handle)
    assert raw.kill.await_count == 2
    raw.close.assert_awaited_once()
