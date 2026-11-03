# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from nemo_gym.agents.utils.sandbox_network import restrict_sandbox_network
from nemo_gym.sandbox.providers.base import SandboxSpec


MODEL_URL = "http://model.example:8000/v1"
TOOL_URLS = ["http://tools.example:9000/mcp", "http://search.example/mcp"]


@pytest.mark.parametrize("provider_options", [{}, {"network_policy": {"defaultAction": "allow"}}])
def test_inherit_returns_the_spec_unchanged(provider_options: dict[str, object]) -> None:
    spec = SandboxSpec(image="task", provider_options=provider_options)
    restricted = restrict_sandbox_network(
        spec, network_access="inherit", provider_name="opensandbox", model_base_url=MODEL_URL, tool_urls=TOOL_URLS
    )
    assert restricted is spec


@pytest.mark.parametrize(
    ("network_access", "allowed"),
    [
        ("model_only", ["model.example"]),
        ("model_and_tools", ["model.example", "search.example", "tools.example"]),
    ],
)
def test_restricted_access_denies_all_egress_but_the_allowed_hosts(network_access: str, allowed: list[str]) -> None:
    spec = SandboxSpec(image="task", provider_options={"other": 1})
    restricted = restrict_sandbox_network(
        spec, network_access=network_access, provider_name="opensandbox", model_base_url=MODEL_URL, tool_urls=TOOL_URLS
    )
    assert restricted.provider_options == {
        "other": 1,
        "network_policy": {"defaultAction": "deny", "egress": [{"action": "allow", "target": h} for h in allowed]},
    }
    assert restricted.image == "task"
    assert spec.provider_options == {"other": 1}


def test_restricted_access_rejects_an_explicit_network_policy() -> None:
    spec = SandboxSpec(image="task", provider_options={"network_policy": {"defaultAction": "allow"}})
    with pytest.raises(ValueError, match="not both"):
        restrict_sandbox_network(
            spec, network_access="model_only", provider_name="opensandbox", model_base_url=MODEL_URL
        )


def test_restricted_access_fails_closed_on_a_provider_without_network_policies() -> None:
    with pytest.raises(ValueError, match="requires the OpenSandbox"):
        restrict_sandbox_network(
            SandboxSpec(image="task"), network_access="model_only", provider_name="docker", model_base_url=MODEL_URL
        )
