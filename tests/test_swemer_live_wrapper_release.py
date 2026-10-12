# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise release retry through the real facade and provider, with a fake remote SDK."""

import asyncio
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.base_resources_server import ReleaseSessionRequest
from nemo_gym.sandbox import AsyncSandbox, SandboxHandle
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider
from nemo_gym.server_utils import SESSION_ID_KEY


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("failure_type", [RuntimeError, TimeoutError, asyncio.CancelledError])
async def test_release_retries_actual_sandbox_termination(version, failure_type):
    raw = SimpleNamespace(kill=AsyncMock(side_effect=failure_type("injected")), close=AsyncMock())
    provider = OpenSandboxProvider(probe={"command": None}, operations={"retries": 0})
    provider.aclose = AsyncMock(wraps=provider.aclose)
    box = AsyncSandbox(provider)
    box._handle = SandboxHandle(sandbox_id="owned", provider_name="opensandbox", raw=raw)
    box._stopped = False
    server = SimpleNamespace(
        _session_id_to_sandbox={"session": box},
        _session_id_to_pristine_untracked={"session": frozenset({"base"})},
        _session_id_to_base_commit={"session": "commit"},
    )
    module = import_module(f"resources_servers.swemer_{version}.app")
    release = getattr(module, f"Swemer{version.upper()}ResourcesServer").release_session
    request = SimpleNamespace(session={SESSION_ID_KEY: "session"})
    body = ReleaseSessionRequest(sandbox_handle="owned")
    with pytest.raises(failure_type):
        await release(server, request, body)
    assert server._session_id_to_sandbox == {"session": box}
    assert server._session_id_to_base_commit == {"session": "commit"}
    assert server._session_id_to_pristine_untracked == {"session": frozenset({"base"})}
    assert (await box.serialize())["sandbox_id"] == "owned"
    assert not box._closed and not box._stopped
    raw.close.assert_not_awaited()
    provider.aclose.assert_not_awaited()
    raw.kill.side_effect = None
    response = await release(server, request, body)
    assert response.released
    assert raw.kill.await_count == 2
    raw.close.assert_awaited_once()
    provider.aclose.assert_awaited_once()
    assert box._closed and box._stopped
    assert not server._session_id_to_sandbox
    assert not server._session_id_to_base_commit
    assert not server._session_id_to_pristine_untracked
    await box.stop()
    assert raw.kill.await_count == 2
