# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SandboxNetwork: the egress rule a resources server gives a sandbox it starts."""

import pytest
from pydantic import ValidationError

from nemo_gym.sandbox.config import SandboxConfig
from nemo_gym.server_runtime import sandbox_network
from nemo_gym.server_runtime.base_resources_server import ResourcesSeedRejectedError
from nemo_gym.server_runtime.sandbox_network import SandboxNetwork


OPENSANDBOX = {"opensandbox": {}}


@pytest.fixture
def server_urls(monkeypatch) -> list[str]:
    urls = {"policy_model": "http://10.1.2.3:8000", "search": "http://10.4.5.6:9000", "local": "http://127.0.0.1:1"}
    asked: list[str] = []

    def server_url(name: str) -> str:
        asked.append(name)
        return urls[name]

    monkeypatch.setattr(sandbox_network, "get_server_url", server_url)
    return asked


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"access": "model_only"}, "needs model_server"),
        ({"access": "model_and_tools", "tool_servers": ("search",)}, "needs model_server"),
        ({"access": "model_and_tools", "model_server": "policy_model"}, "needs tool_servers"),
        ({"access": "no_network", "model_server": "policy_model"}, "model_server applies only"),
        ({"access": "inherit", "model_server": "policy_model"}, "model_server applies only"),
        ({"access": "model_only", "model_server": "policy_model", "tool_servers": ("search",)}, "tool_servers"),
        ({"access": "open"}, "access"),
    ],
)
def test_a_network_setting_without_the_servers_it_needs_is_refused(fields: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        SandboxNetwork(**fields)


def test_inherit_keeps_the_provider_network_on_any_provider(server_urls) -> None:
    assert SandboxNetwork().policy({"local": {}}) is None
    assert server_urls == []


def test_no_network_denies_all_egress(server_urls) -> None:
    assert SandboxNetwork(access="no_network").policy(OPENSANDBOX) == {"defaultAction": "deny", "egress": []}
    assert server_urls == []


def test_model_and_tools_allows_the_model_and_tool_server_hosts(server_urls) -> None:
    network = SandboxNetwork(access="model_and_tools", model_server="policy_model", tool_servers=("search",))

    assert network.policy(OPENSANDBOX) == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "10.1.2.3"}, {"action": "allow", "target": "10.4.5.6"}],
    }
    assert server_urls == ["policy_model", "search"]


@pytest.mark.parametrize(
    ("network", "provider", "message"),
    [
        (SandboxNetwork(access="no_network"), {"local": {}}, "requires the OpenSandbox network-policy provider"),
        (SandboxNetwork(access="model_only", model_server="local"), OPENSANDBOX, "sandbox-reachable host"),
    ],
    ids=["provider without network policies", "loopback model server"],
)
def test_an_unenforceable_restriction_is_refused_as_a_rejected_seed(
    server_urls, network: SandboxNetwork, provider: dict, message: str
) -> None:
    with pytest.raises(ResourcesSeedRejectedError, match=message) as error:
        network.policy(provider)

    assert error.value.status_code == 422


def test_a_restriction_and_a_configured_network_policy_are_refused_together() -> None:
    configured = SandboxConfig(provider_options={"network_policy": {"defaultAction": "deny"}})

    with pytest.raises(ValueError, match="not both"):
        SandboxNetwork(access="no_network").check(configured)
    SandboxNetwork().check(configured)
    SandboxNetwork(access="no_network").check(SandboxConfig())
