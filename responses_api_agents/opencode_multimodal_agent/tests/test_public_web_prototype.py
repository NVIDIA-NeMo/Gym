# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import socket
from unittest.mock import AsyncMock

import pytest

from responses_api_agents.opencode_multimodal_agent import network_runner as runner


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "fc00::1",
        "fe80::1",
        "::ffff:127.0.0.1",
        "64:ff9b::7f00:1",
        "2002:7f00:1::",
        "2001::1",
    ],
)
def test_nonpublic_and_transition_addresses_refused(address):
    assert not runner.public_address(address)


@pytest.mark.parametrize("address", ["1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"])
def test_native_public_addresses_admitted(address):
    assert runner.public_address(address)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "addresses",
    [
        [(socket.AF_INET, "127.0.0.1")],
        [(socket.AF_INET, "1.1.1.1"), (socket.AF_INET, "10.0.0.1")],
        [(socket.AF_INET, "8.8.8.8")],
    ],
)
async def test_private_mixed_and_protected_destinations_never_connect(monkeypatch, addresses):
    monkeypatch.setattr(runner, "resolved_addresses", AsyncMock(return_value=addresses))
    connect = AsyncMock()
    monkeypatch.setattr(runner.asyncio, "open_connection", connect)
    policy = runner.PublicWebPolicy(set(), {"8.8.8.8"})
    with pytest.raises(PermissionError):
        await policy.connect(("alias.example", 443))
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_connection_uses_validated_numeric_address(monkeypatch):
    resolve = AsyncMock(return_value=[(socket.AF_INET, "1.1.1.1")])
    monkeypatch.setattr(runner, "resolved_addresses", resolve)
    connect = AsyncMock(return_value=("reader", "writer"))
    monkeypatch.setattr(runner.asyncio, "open_connection", connect)
    assert await runner.PublicWebPolicy(set(), set()).connect(("public.example", 443)) == ("reader", "writer")
    resolve.assert_awaited_once_with(("public.example", 443))
    connect.assert_awaited_once_with("1.1.1.1", 443, family=socket.AF_INET, flags=socket.AI_NUMERICHOST)


@pytest.mark.asyncio
async def test_private_model_exception_does_not_allow_other_ports_or_aliases(monkeypatch):
    monkeypatch.setattr(runner, "resolved_addresses", AsyncMock(return_value=[(socket.AF_INET, "127.0.0.1")]))
    connect = AsyncMock()
    monkeypatch.setattr(runner.asyncio, "open_connection", connect)
    policy = runner.PublicWebPolicy({("model.example", 8000)}, {"127.0.0.1"})
    await policy.connect(("model.example", 8000))
    assert connect.await_count == 1
    for dest in [("model.example", 8001), ("alias.example", 8000)]:
        with pytest.raises(PermissionError):
            await policy.connect(dest)
    assert connect.await_count == 1


def test_local_interface_enumeration():
    addresses = runner.local_addresses()
    assert "127.0.0.1" in addresses
