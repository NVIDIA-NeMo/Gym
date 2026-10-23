# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.sandbox import AsyncSandbox, SandboxHandle, SandboxSpec
from tests.unit_tests.test_sandbox import FakeSandboxProvider


pytestmark = pytest.mark.sandbox


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
