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


@pytest.mark.asyncio
async def test_grant_alias_cannot_share_control_endpoint(monkeypatch):
    monkeypatch.setattr(runner, "resolved_addresses", AsyncMock(return_value=[(socket.AF_INET, "1.1.1.1")]))
    connect = AsyncMock()
    monkeypatch.setattr(runner.asyncio, "open_connection", connect)
    policy = runner.PublicWebPolicy({("tool.example", 443)}, set(), protected_origins={("judge.example", 443)})
    with pytest.raises(PermissionError, match="protected control endpoint"):
        await policy.connect(("tool.example", 443))
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_control_dns_change_and_old_addresses_remain_blocked(monkeypatch):
    control_address = "1.1.1.1"

    async def resolve(destination):
        return [(socket.AF_INET, control_address if destination[0] == "control.example" else destination[0])]

    monkeypatch.setattr(runner, "resolved_addresses", resolve)
    connect = AsyncMock()
    monkeypatch.setattr(runner.asyncio, "open_connection", connect)
    policy = runner.PublicWebPolicy(set(), set(), protected_origins={("control.example", 443)})
    with pytest.raises(PermissionError):
        await policy.connect(("1.1.1.1", 443))
    control_address = "8.8.8.8"
    for address in ("8.8.8.8", "1.1.1.1"):
        with pytest.raises(PermissionError):
            await policy.connect((address, 8000))
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_control_dns_failure_cannot_enable_public_connections(monkeypatch):
    monkeypatch.setattr(runner, "resolved_addresses", AsyncMock(side_effect=OSError("DNS unavailable")))
    connect = AsyncMock()
    monkeypatch.setattr(runner.asyncio, "open_connection", connect)
    policy = runner.PublicWebPolicy(set(), set(), protected_origins={("control.example", 443)})
    with pytest.raises(OSError, match="DNS unavailable"):
        await policy.connect(("1.1.1.1", 443))
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_restricted_mode_does_not_allow_public_destination(monkeypatch):
    monkeypatch.setattr(runner, "resolved_addresses", AsyncMock(return_value=[(socket.AF_INET, "1.1.1.1")]))
    connect = AsyncMock()
    monkeypatch.setattr(runner.asyncio, "open_connection", connect)
    with pytest.raises(PermissionError):
        await runner.PublicWebPolicy(set(), set(), allow_public=False).connect(("public.example", 443))
    connect.assert_not_awaited()


def test_gym_control_inventory_excludes_only_selected_policy():
    from omegaconf import OmegaConf

    from responses_api_agents.opencode_multimodal_agent.network_config import control_service_urls

    config = {"head_server": {"host": "0.0.0.0", "port": 8000}}
    for name, group, port in [
        ("policy", "responses_api_models", 8001),
        ("judge", "responses_api_models", 8002),
        ("agent", "responses_api_agents", 8003),
        ("verifier", "resources_servers", 8004),
        ("environment", "environment_servers", 8005),
    ]:
        config[name] = {group: {"implementation": {"host": "::", "port": port}}}
    assert control_service_urls(OmegaConf.create(config), model_server="policy") == [
        "http://127.0.0.1:8000",
        *[f"http://[::1]:{port}" for port in range(8002, 8006)],
    ]
